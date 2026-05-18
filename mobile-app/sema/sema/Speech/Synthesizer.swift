import AVFoundation
import Foundation
import Observation

/// AVSpeechSynthesizer wrapper used by Path A's gloss-to-speech tail and
/// any other text-to-audio surface. Flips `TTSGate.shared` so the ASR
/// recogniser pauses while we speak.
@MainActor
@Observable
final class Synthesizer: NSObject {
    private let synth = AVSpeechSynthesizer()
    private(set) var isSpeaking = false

    override init() {
        super.init()
        synth.delegate = self
    }

    func speak(_ text: String, language: String = "en-US") {
        guard !text.isEmpty else { return }
        // Unified conversation uses `.playAndRecord` from `ConversationOrchestrator.start()`.
        // Ensure the session is active before speaking; do not flip to `.playback`
        // (that broke TTS when camera + ASR shared the live session).
        if !AudioSession.shared.isActive {
            try? AudioSession.shared.configure()
        }
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: language)
        synth.speak(utterance)
    }

    func stop() {
        synth.stopSpeaking(at: .immediate)
    }
}

extension Synthesizer: AVSpeechSynthesizerDelegate {
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        Task { @MainActor in
            self.isSpeaking = true
            TTSGate.shared.setSpeaking(true)
        }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in
            // 200 ms tail before re-opening the ASR mic (per asr/contract.md).
            try? await Task.sleep(nanoseconds: 200_000_000)
            self.isSpeaking = false
            TTSGate.shared.setSpeaking(false)
        }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in
            self.isSpeaking = false
            TTSGate.shared.setSpeaking(false)
        }
    }
}
