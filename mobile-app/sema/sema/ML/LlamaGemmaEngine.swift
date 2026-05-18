import Foundation

#if canImport(llama)
import llama

/// Thin Swift actor around llama.cpp's C API. Loads the bundled
/// `gemma-4-e2b-ksl-Q4_K_M.gguf` on first use and exposes a single-shot
/// `generate(prompt:maxTokens:)` that returns the full model reply after
/// running prefill + greedy decode to completion.
///
/// Public surface matches the older `LiteRTLMEngine` (warmup / generate /
/// close) so `GemmaTranslator` and `ConversationOrchestrator` don't need to
/// know which backend they're talking to.
actor LlamaGemmaEngine {

    /// Resource basename in the app bundle, no extension. Matches the GGUF
    /// dropped at `mobile-app/sema/sema/Resources/`.
    static let bundledModelName = "gemma-4-e2b-ksl-Q4_K_M"
    static let bundledModelExt  = "gguf"

    private var model: OpaquePointer?
    private var context: OpaquePointer?
    private var vocab: OpaquePointer?
    private var sampler: UnsafeMutablePointer<llama_sampler>?
    private var loadFailure: Error?

    /// Force-load the model so the first user-visible generation doesn't pay
    /// the warmup cost (mmap + Metal kernel compile, ~2-4 s on A17/A18).
    /// Safe to call multiple times.
    func warmup() async throws {
        _ = try ensureLoaded()
    }

    /// Generate a completion for `prompt`. Wraps the prompt in Gemma's
    /// turn markers, runs prefill, then greedy-decodes until EOG, the
    /// `<end_of_turn>` marker, or `maxTokens`, whichever comes first.
    ///
    /// KSL prompts are short (~16 tokens for gloss lists, ~64 for fluent EN).
    /// We clear the KV cache between calls so each generation starts clean.
    func generate(prompt: String, maxTokens: Int = 64) async throws -> String {
        let (ctx, vocab, sampler) = try ensureLoaded()

        // Gemma chat template — same as the previous LiteRT-LM path.
        let wrapped = "<start_of_turn>user\n\(prompt)<end_of_turn>\n<start_of_turn>model\n"

        // Reset KV cache so prior generations don't leak into this prompt.
        llama_memory_clear(llama_get_memory(ctx), true)

        let tokens = tokenize(wrapped, vocab: vocab, addBOS: true, parseSpecial: true)
        if tokens.isEmpty {
            throw LlamaGemmaEngineError.tokenizationFailed
        }

        // Prefill: feed all prompt tokens, request logits only on the last one.
        var batch = llama_batch_init(Int32(max(512, tokens.count)), 0, 1)
        defer { llama_batch_free(batch) }
        batch.n_tokens = 0
        for (i, tok) in tokens.enumerated() {
            llamaBatchAdd(&batch, token: tok, pos: Int32(i), seqIds: [0], wantLogits: false)
        }
        batch.logits[Int(batch.n_tokens) - 1] = 1
        if llama_decode(ctx, batch) != 0 {
            throw LlamaGemmaEngineError.decodeFailed
        }

        var nCur = batch.n_tokens
        var produced = ""
        var pendingUTF8: [CChar] = []

        for _ in 0..<maxTokens {
            let id = llama_sampler_sample(sampler, ctx, batch.n_tokens - 1)
            if llama_vocab_is_eog(vocab, id) { break }

            let piece = tokenToPiece(id, vocab: vocab)
            pendingUTF8.append(contentsOf: piece)
            if let s = String(validatingUTF8: pendingUTF8 + [0]) {
                produced.append(s)
                pendingUTF8.removeAll(keepingCapacity: true)
            }

            // Gemma emits `<end_of_turn>` to close a model turn even when
            // EOG isn't fired by the tokenizer. Stop manually if we see it.
            if produced.contains("<end_of_turn>") { break }

            batch.n_tokens = 0
            llamaBatchAdd(&batch, token: id, pos: nCur, seqIds: [0], wantLogits: true)
            nCur += 1
            if llama_decode(ctx, batch) != 0 {
                throw LlamaGemmaEngineError.decodeFailed
            }
        }

        return Self.firstLine(of: produced)
    }

    /// Tear down the model + context. Frees the ~1.5-2 GB of mmap'd weights
    /// and the KV cache (model_size × n_ctx × bytes/elem). Re-loaded on the
    /// next `generate(...)`.
    func close() {
        if let sampler { llama_sampler_free(sampler) }
        if let context { llama_free(context) }
        if let model   { llama_model_free(model) }
        sampler = nil
        context = nil
        model = nil
        vocab = nil
        loadFailure = nil
        llama_backend_free()
    }

    // MARK: - private

    private func ensureLoaded() throws -> (OpaquePointer, OpaquePointer, UnsafeMutablePointer<llama_sampler>) {
        if let ctx = context, let vocab, let sampler { return (ctx, vocab, sampler) }
        if let loadFailure { throw loadFailure }

        let path: String
        do {
            path = try Self.locateBundledModel()
        } catch {
            loadFailure = error
            throw error
        }

        llama_backend_init()

        var modelParams = llama_model_default_params()
        #if targetEnvironment(simulator)
        // Metal isn't available on the simulator; force CPU fallback.
        modelParams.n_gpu_layers = 0
        #endif
        guard let m = llama_model_load_from_file(path, modelParams) else {
            let err = LlamaGemmaEngineError.modelLoadFailed(path)
            loadFailure = err
            throw err
        }

        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx = 2048
        let nThreads = Int32(max(1, min(8, ProcessInfo.processInfo.processorCount - 2)))
        ctxParams.n_threads = nThreads
        ctxParams.n_threads_batch = nThreads
        guard let c = llama_init_from_model(m, ctxParams) else {
            llama_model_free(m)
            let err = LlamaGemmaEngineError.contextInitFailed
            loadFailure = err
            throw err
        }

        let v = llama_model_get_vocab(m)

        // Greedy sampler — deterministic, matches the temperature=0, topK=1
        // config the LiteRT-LM path used. For richer sampling (top-p, temp),
        // add additional links to the chain here.
        let chain = llama_sampler_chain_init(llama_sampler_chain_default_params())
        llama_sampler_chain_add(chain, llama_sampler_init_greedy())

        self.model = m
        self.context = c
        self.vocab = v
        self.sampler = chain
        return (c, v!, chain!)
    }

    private func tokenize(_ text: String,
                          vocab: OpaquePointer,
                          addBOS: Bool,
                          parseSpecial: Bool) -> [llama_token] {
        let utf8 = text.utf8
        let utf8Count = Int32(utf8.count)
        // Worst-case 1 token per byte plus BOS room.
        let cap = Int(utf8Count) + (addBOS ? 1 : 0) + 8
        let buf = UnsafeMutablePointer<llama_token>.allocate(capacity: cap)
        defer { buf.deallocate() }
        let n = llama_tokenize(vocab, text, utf8Count, buf, Int32(cap), addBOS, parseSpecial)
        if n <= 0 { return [] }
        return Array(UnsafeBufferPointer(start: buf, count: Int(n)))
    }

    private func tokenToPiece(_ token: llama_token, vocab: OpaquePointer) -> [CChar] {
        var size: Int32 = 16
        var buf = UnsafeMutablePointer<CChar>.allocate(capacity: Int(size))
        var n = llama_token_to_piece(vocab, token, buf, size, 0, false)
        if n < 0 {
            // Buffer too small — re-alloc with the exact size the API reported.
            buf.deallocate()
            size = -n
            buf = UnsafeMutablePointer<CChar>.allocate(capacity: Int(size))
            n = llama_token_to_piece(vocab, token, buf, size, 0, false)
        }
        defer { buf.deallocate() }
        if n <= 0 { return [] }
        return Array(UnsafeBufferPointer(start: buf, count: Int(n)))
    }

    private static func locateBundledModel() throws -> String {
        if let path = Bundle.main.path(forResource: bundledModelName,
                                       ofType: bundledModelExt) {
            return path
        }
        throw LlamaGemmaEngineError.modelMissing(bundledModelName + "." + bundledModelExt)
    }

    private static func firstLine(of text: String) -> String {
        let cleaned = text
            .replacingOccurrences(of: "<end_of_turn>", with: "")
            .replacingOccurrences(of: "<start_of_turn>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        for line in cleaned.split(whereSeparator: { $0.isNewline }) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty { return trimmed }
        }
        return cleaned
    }
}

// MARK: - batch helper

private func llamaBatchAdd(_ batch: inout llama_batch,
                           token: llama_token,
                           pos: llama_pos,
                           seqIds: [llama_seq_id],
                           wantLogits: Bool) {
    let i = Int(batch.n_tokens)
    batch.token[i] = token
    batch.pos[i] = pos
    batch.n_seq_id[i] = Int32(seqIds.count)
    for k in 0..<seqIds.count {
        batch.seq_id[i]![k] = seqIds[k]
    }
    batch.logits[i] = wantLogits ? 1 : 0
    batch.n_tokens += 1
}

enum LlamaGemmaEngineError: Error, CustomStringConvertible {
    case modelMissing(String)
    case modelLoadFailed(String)
    case contextInitFailed
    case tokenizationFailed
    case decodeFailed
    case sdkUnavailable

    var description: String {
        switch self {
        case .modelMissing(let name):
            return "\(name) is not in the app bundle. Drop the GGUF into mobile-app/sema/sema/Resources/ and rebuild."
        case .modelLoadFailed(let path):
            return "llama_model_load_from_file failed for \(path) — check the GGUF is intact and the device has enough memory."
        case .contextInitFailed:
            return "llama_init_from_model failed — usually a memory or Metal-init issue."
        case .tokenizationFailed:
            return "Prompt tokenized to zero tokens — empty input or vocab mismatch."
        case .decodeFailed:
            return "llama_decode returned non-zero — KV cache exhausted or backend error."
        case .sdkUnavailable:
            return "llama.cpp xcframework is not linked. Drag mobile-app/sema/Frameworks/llama.xcframework into the sema target and embed-sign it."
        }
    }
}

#else

/// Stand-in when llama.xcframework isn't yet linked, so the rest of the app
/// still compiles. Every call throws `.sdkUnavailable` with instructions.
actor LlamaGemmaEngine {
    static let bundledModelName = "gemma-4-e2b-ksl-Q4_K_M"
    static let bundledModelExt  = "gguf"

    func warmup() async throws { throw LlamaGemmaEngineError.sdkUnavailable }
    func generate(prompt: String, maxTokens: Int = 64) async throws -> String {
        throw LlamaGemmaEngineError.sdkUnavailable
    }
    func close() {}
}

enum LlamaGemmaEngineError: Error, CustomStringConvertible {
    case sdkUnavailable

    var description: String {
        switch self {
        case .sdkUnavailable:
            return "llama.cpp xcframework is not linked. Drag mobile-app/sema/Frameworks/llama.xcframework into the sema target and embed-sign it."
        }
    }
}

#endif
