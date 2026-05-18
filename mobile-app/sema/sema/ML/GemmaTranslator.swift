import Foundation

/// Bidirectional KSL ↔ EN/SW translator.
///
/// Three execution modes per `mobile-app/docs/path_b_avatar.md`:
///   - `.stub`     : canned phrasebook for demos before Gemma is available.
///   - `.server`   : HTTPS endpoint matching `gemma-glossing/README.md`'s
///                   server-fallback contract.
///   - `.onDevice` : llama.cpp inference using `gemma-4-e2b-ksl-Q4_K_M.gguf`
///                   bundled in `Sema/Resources/`. Drives both Path A
///                   (gloss → fluent EN sentence for TTS) and Path B
///                   (EN sentence → KSL gloss list for the avatar) via
///                   few-shot prompts defined in `KSLPrompts`.
actor GemmaTranslator {

    /// Shared engine for `.onDevice` mode. Created lazily on first use,
    /// kept alive for the process lifetime (~1.5-2 GB resident on iPhone).
    private static let sharedEngine = LlamaGemmaEngine()

    enum Mode: Sendable {
        case stub
        case server(URL, token: String?)
        case onDevice
    }

    enum Task: String, Sendable {
        case englishToKSL = "[EN→KSL]"
        case swahiliToKSL = "[SW→KSL]"
        case kslToEnglish = "[KSL→EN]"
        case kslToSwahili = "[KSL→SW]"
    }

    private let mode: Mode

    /// Demo phrasebook used by `.stub`. Maps a *lowercased input string* (the
    /// English sentence on EN→KSL, or the space-joined gloss list on KSL→EN)
    /// to its translation. The same dictionary serves both directions because
    /// the keyspaces don't overlap — English sentences contain commas / `?` /
    /// lowercase letters, gloss lists are uppercase tokens joined by spaces.
    ///
    /// The four demo entries below come from `Resources/demo_recognition.json`
    /// and `Resources/demo_generation.json` — the Hospital and Bank scenarios.
    /// Keep them in sync if those JSON files change.
    private let phrasebook: [String: [String]] = [
        // Generic seed entries (kept from earlier demos).
        "i am going to the hospital tomorrow.": ["TOMORROW", "HOSPITAL", "I-GO"],
        "hello": ["HELLO"],
        "thank you": ["THANK"],
        "good morning": ["MORNING", "GOOD"],
        "how are you": ["HOW", "YOU"],
        "my name is sema": ["ME", "NAME", "SEMA"],
        "i love you": ["ME", "LOVE", "YOU"],
        "where is the bathroom": ["BATHROOM", "WHERE"],

        // --- Hospital scenario ---
        // Sign → Speech: Winnie's gloss → English sentence the TTS speaks.
        "me pain throat headache night what do":
            ["I have a sore throat and mild headaches at night. What should I do?"],
        // Speech → Sign: doctor's English → glosses the avatar plays.
        "try reducing screen time two hours before bed. if it doesn't clear up in ten days, come back and i'll prescribe something.":
            ["TRY", "REDUCE", "SCREEN", "TIME", "TWO", "HOUR", "BEFORE", "BED",
             "IF", "NOT", "BETTER", "TEN", "DAY", "RETURN", "MEDICINE"],
        // Speech → Sign: the PATIENT-side phrasings (previously missing — only
        // the gloss-list key existed, so speaking these would silently miss).
        "i have a sore throat and mild headaches at night. what should i do?":
            ["ME", "PAIN", "THROAT", "HEADACHE", "NIGHT", "WHAT", "DO"],
        "i have a sore throat":              ["ME", "PAIN", "THROAT"],
        "i have a headache":                  ["ME", "HEADACHE"],
        "i have headaches":                   ["ME", "HEADACHE"],
        "i have headaches at night":          ["ME", "HEADACHE", "NIGHT"],
        "my throat hurts":                    ["ME", "THROAT", "PAIN"],
        "what should i do":                   ["WHAT", "ME", "DO"],

        // --- Bank scenario ---
        // Sign → Speech: Andrew's gloss → English sentence.
        "me statement charge not know explain please":
            ["There are charges on my statement I don't recognise. Can you explain them?"],
        // Speech → Sign: bank staff's English → glosses.
        "those are administrative fees tied to a credit card withdrawal made on the 23rd.":
            ["COST", "BANK", "CARD", "TAKE", "DATE", "TWENTY", "THREE"],
        // Speech → Sign: the CUSTOMER-side phrasings (previously missing).
        "there are charges on my statement i don't recognise. can you explain them?":
            ["ME", "STATEMENT", "CHARGE", "NOT", "KNOW", "EXPLAIN", "PLEASE"],
        "explain this charge":                ["EXPLAIN", "CHARGE"],
        "explain my statement":               ["EXPLAIN", "STATEMENT"],
        "please explain this charge":         ["PLEASE", "EXPLAIN", "CHARGE"],
        "i don't recognise this charge":      ["ME", "NOT", "KNOW", "CHARGE"],
        "i don't know this charge":           ["ME", "NOT", "KNOW", "CHARGE"],
        "what is this charge":                ["WHAT", "CHARGE"],
    ]

    init(mode: Mode = .stub) {
        self.mode = mode
    }

    /// Translate `text` for the given task. v0 returns the entire
    /// sequence in one shot; the streaming variant in
    /// `mobile-app/docs/path_b_avatar.md` is a follow-up.
    ///
    /// Return shape:
    ///   - EN→KSL / SW→KSL → array of uppercase gloss tokens
    ///   - KSL→EN / KSL→SW → single-element array with the fluent sentence
    func translate(_ text: String, task: Task = .englishToKSL) async throws -> [String] {
        switch mode {
        case .stub:
            return phrasebookLookup(text)
        case .server(let url, let token):
            return try await callServer(url: url, token: token, task: task, text: text)
        case .onDevice:
            return try await onDeviceTranslate(text, task: task)
        }
    }

    /// Eagerly load the on-device engine. Call from app launch so the first
    /// user-visible translation doesn't pay the ~6 s warmup cost.
    ///
    /// No-op for `.stub` and `.server` modes — calling `sharedEngine.warmup()`
    /// would still try to load the 3.2 GB GGUF off disk and allocate Metal
    /// buffers, which is exactly what we're trying to avoid when the caller
    /// has chosen a non-on-device translator.
    func warmupOnDevice() async throws {
        guard case .onDevice = mode else { return }
        try await Self.sharedEngine.warmup()
    }

    /// Free the on-device engine's ~3 GB Metal buffer + KV cache. The next
    /// translate call will lazily reload (paying the warmup cost again).
    /// Intended for memory-warning handlers and explicit app-background hooks.
    static func closeSharedOnDeviceEngine() async {
        await sharedEngine.close()
    }

    // MARK: - On-device

    private func onDeviceTranslate(_ text: String, task: Task) async throws -> [String] {
        let prompt = KSLPrompts.render(task: task, input: text)
        let reply = try await Self.sharedEngine.generate(
            prompt: prompt,
            maxTokens: KSLPrompts.maxTokens(for: task))
        return parseReply(reply, task: task)
    }

    /// Split the model's reply into the caller's expected shape.
    private nonisolated func parseReply(_ reply: String, task: Task) -> [String] {
        let cleaned = reply
            .trimmingCharacters(in: .whitespacesAndNewlines)
            // Strip Gemma special tokens that occasionally slip past the
            // session's stop-token list (varies by build).
            .replacingOccurrences(of: "<end_of_turn>", with: "")
            .replacingOccurrences(of: "<turn|>", with: "")
            .trimmingCharacters(in: .whitespaces)
        switch task {
        case .englishToKSL, .swahiliToKSL:
            // Split on whitespace, keep only the uppercase-y tokens — the
            // few-shot prompt asks for uppercase gloss only, so anything
            // lowercase is the model wandering off-script and we drop it.
            return cleaned
                .split(whereSeparator: { $0.isWhitespace })
                .map { String($0).trimmingCharacters(in: CharacterSet(charactersIn: ",.;:'\"")) }
                .filter { !$0.isEmpty }
                .map { $0.uppercased() }
        case .kslToEnglish, .kslToSwahili:
            return [cleaned]
        }
    }

    // MARK: - Stub

    private func phrasebookLookup(_ text: String) -> [String] {
        let needle = Self.normalizePhrase(text)
        // Iterate (the table is tiny) and compare phrasebook keys after the
        // same normalisation. This lets ASR transcripts that drop periods /
        // commas / apostrophes still hit the demo's pre-written entries.
        for (rawKey, glosses) in phrasebook
        where Self.normalizePhrase(rawKey) == needle {
            return glosses
        }
        // Word-by-word fallback: uppercase each whitespace-split word.
        return needle
            .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
            .map { $0.uppercased() }
    }

    /// Lowercase + collapse to letters/digits/spaces, so punctuation in the
    /// input (or in the phrasebook keys) doesn't block matching. Hyphens
    /// inside compound glosses (like `I-GO`) survive on output because this
    /// runs only on inputs — phrasebook *values* are emitted verbatim.
    private static func normalizePhrase(_ s: String) -> String {
        var out = ""
        var lastWasSpace = false
        for ch in s.lowercased() {
            if ch.isLetter || ch.isNumber {
                out.append(ch); lastWasSpace = false
            } else if ch.isWhitespace || ch == "-" || ch == "_" {
                if !lastWasSpace && !out.isEmpty { out.append(" "); lastWasSpace = true }
            }
            // everything else (`.`, `,`, `'`, `?`, `!`, etc.) drops silently
        }
        return out.trimmingCharacters(in: .whitespaces)
    }

    // MARK: - Server fallback

    private struct ServerRequest: Codable {
        let task: String
        let input: String
        let max_tokens: Int
    }

    private struct ServerReply: Codable {
        let output: String
        let model_version: String?
    }

    private func callServer(url: URL, token: String?, task: Task, text: String) async throws -> [String] {
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let token, !token.isEmpty {
            req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        let body = ServerRequest(task: task.rawValue, input: text, max_tokens: 64)
        req.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await URLSession.shared.data(for: req)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw GemmaTranslatorError.serverError((response as? HTTPURLResponse)?.statusCode ?? -1)
        }
        let reply = try JSONDecoder().decode(ServerReply.self, from: data)
        return reply.output
            .split(whereSeparator: { $0.isWhitespace })
            .map(String.init)
    }
}

enum GemmaTranslatorError: Error, CustomStringConvertible {
    case serverError(Int)
    var description: String {
        switch self {
        case .serverError(let code):
            return "Gemma server returned HTTP \(code)."
        }
    }
}
