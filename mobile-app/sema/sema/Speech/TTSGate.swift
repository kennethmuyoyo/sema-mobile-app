import Foundation

/// Shared flag the speech recogniser reads to know whether TTS is playing.
/// `Synthesizer` flips it; `ContinuousSpeechRecognizer` reads it and pauses
/// while it's `true`. Prevents echo loops where the avatar's spoken text
/// gets picked back up by the mic.
@MainActor
final class TTSGate {
    static let shared = TTSGate()
    private(set) var isSpeaking = false

    func setSpeaking(_ value: Bool) {
        isSpeaking = value
    }
}
