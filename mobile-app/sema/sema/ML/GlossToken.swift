import Foundation

/// A gloss token emitted by the streaming CTC decoder.
struct GlossToken: Sendable, Identifiable, Equatable {
    /// Recognizer vocabulary id (`<blank>` = 0, `<unk>` = 1, others ≥ 2).
    let id: Int
    /// Human-readable gloss string from the vocab JSON sidecar.
    let label: String
    /// Wall-clock at the moment this token was stabilised and emitted.
    let timestamp: TimeInterval
    /// Softmax peak at the emission point, in [0, 1].
    let confidence: Float

    /// Identifiable conformance: `id` field on its own may repeat across
    /// emissions (the same token re-said). Combine with timestamp.
    var stableID: String { "\(id)@\(timestamp)" }

    var description: String { "\(label) (\(String(format: "%.2f", confidence)))" }
}
