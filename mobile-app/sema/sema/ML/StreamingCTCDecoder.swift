import Foundation

/// Streaming greedy CTC decoder with stable-suffix emission.
///
/// Per `mobile-app/docs/coreml_path_a.md`:
/// - Each call to `ingest(...)` decodes one inference window into a token
///   sequence (argmax → collapse repeats → strip blanks).
/// - The last `historyDepth` decoded sequences are kept; their longest
///   common prefix is the *stable* output.
/// - Only tokens beyond the previously-emitted prefix get returned.
///
/// Caller is responsible for resetting on no-hand timeouts.
final class StreamingCTCDecoder {
    static let blankID = 0

    private let historyDepth: Int
    private let labels: [String]
    private let vocabSize: Int
    private var history: [[Int]] = []
    private var emittedPrefix: [Int] = []

    /// `vocab` is the `{label: id}` dictionary from
    /// `ksl_model.metadata.json` -> `metadata.glossNameToId`.
    init(vocab: [String: Int], historyDepth: Int = 3) {
        precondition(!vocab.isEmpty, "empty vocab")
        self.historyDepth = historyDepth
        self.vocabSize = vocab.count
        var labelsArray = [String](repeating: "<unk>", count: vocab.count)
        for (label, id) in vocab where id >= 0 && id < labelsArray.count {
            labelsArray[id] = label
        }
        self.labels = labelsArray
    }

    func reset() {
        history.removeAll(keepingCapacity: true)
        emittedPrefix.removeAll(keepingCapacity: true)
    }

    /// Ingest one inference window's logits.
    ///
    /// - Parameters:
    ///   - logits: row-major `(T, V)`; length must equal `timeSteps * vocabSize`.
    ///   - timeSteps: T in the (T, V) layout (== `GlossTagger.seqLen`).
    ///   - timestamp: wall-clock for the most recent frame in the window.
    /// - Returns: newly-stable gloss tokens (possibly empty).
    func ingest(logits: [Float], timeSteps: Int, timestamp: TimeInterval) -> [GlossToken] {
        precondition(logits.count == timeSteps * vocabSize,
                     "logits shape mismatch: got \(logits.count), expected \(timeSteps * vocabSize)")
        let decoded = greedyDecode(logits: logits, timeSteps: timeSteps)
        history.append(decoded.tokens)
        if history.count > historyDepth { history.removeFirst() }
        guard history.count == historyDepth else { return [] }

        let stable = longestCommonPrefix(history)
        guard stable.count > emittedPrefix.count else { return [] }

        let new = Array(stable[emittedPrefix.count..<stable.count])
        emittedPrefix = stable
        return new.map { id in
            GlossToken(
                id: id,
                label: id < labels.count ? labels[id] : "<unk>",
                timestamp: timestamp,
                confidence: decoded.peakProb(for: id)
            )
        }
    }

    // MARK: - Internals

    private struct Decoded {
        let tokens: [Int]
        let peakByID: [Int: Float]
        func peakProb(for id: Int) -> Float { peakByID[id, default: 0] }
    }

    private func greedyDecode(logits: [Float], timeSteps: Int) -> Decoded {
        var tokens: [Int] = []
        var peaks: [Int: Float] = [:]
        var prev = -1
        for t in 0..<timeSteps {
            let base = t * vocabSize
            var bestIdx = 0
            var bestVal = -Float.infinity
            for v in 0..<vocabSize {
                let val = logits[base + v]
                if val > bestVal { bestVal = val; bestIdx = v }
            }
            if bestIdx != prev && bestIdx != Self.blankID {
                tokens.append(bestIdx)
                // Softmax peak at this step, computed in log-sum-exp style.
                var sumExp: Float = 0
                for v in 0..<vocabSize {
                    sumExp += exp(logits[base + v] - bestVal)
                }
                let prob = 1.0 / sumExp
                if prob > (peaks[bestIdx] ?? 0) { peaks[bestIdx] = prob }
            }
            prev = bestIdx
        }
        return Decoded(tokens: tokens, peakByID: peaks)
    }

    private func longestCommonPrefix(_ seqs: [[Int]]) -> [Int] {
        guard let first = seqs.first else { return [] }
        var prefix: [Int] = []
        outer: for (i, v) in first.enumerated() {
            for s in seqs.dropFirst() {
                if i >= s.count || s[i] != v { break outer }
            }
            prefix.append(v)
        }
        return prefix
    }
}
