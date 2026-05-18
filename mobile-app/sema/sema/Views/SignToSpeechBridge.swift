import Foundation
import Observation

/// Owns the sign-language → English → spoken-audio pipeline for the
/// conversation screen. Watches the recognizer's finalized gloss phrase, debounces,
/// translates via Gemma (KSL → English), then speaks the result.
///
/// Kept as an `@Observable` class rather than living inline in the view so
/// the timing logic can be tested independently and the tab body stays
/// declarative.
@MainActor
@Observable
final class SignToSpeechBridge {
    private(set) var spokenSentence: String = ""

    @ObservationIgnored private let translator: GemmaTranslator
    @ObservationIgnored private let speechOutput: SpeechOutputting
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var lastSpokenGloss: String = ""

    /// Window during which we wait for the gloss to stabilise before
    /// translating. Slightly longer than a typical mid-sign pause so we
    /// don't fire a translation mid-utterance.
    @ObservationIgnored private let debounce: Duration = .milliseconds(1200)

    init(translator: GemmaTranslator, speechOutput: SpeechOutputting? = nil) {
        self.translator = translator
        self.speechOutput = speechOutput ?? Synthesizer()
    }

    /// Schedule a translation+speak pass for the latest finalized gloss
    /// phrase. Cancels any previously-scheduled pass so a rapidly-updating
    /// transcript doesn't queue up stale translations.
    func handle(finalizedGloss: String) {
        let gloss = finalizedGloss.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !gloss.isEmpty else { return }

        task?.cancel()
        task = Task { [weak self, gloss] in
            try? await Task.sleep(for: self?.debounce ?? .milliseconds(1200))
            guard let self, !Task.isCancelled else { return }
            await self.translateAndSpeak(gloss: gloss)
        }
    }

    func cancel() {
        task?.cancel()
        task = nil
        speechOutput.stop()
    }

    private func translateAndSpeak(gloss: String) async {
        guard gloss != lastSpokenGloss else { return }

        do {
            let words = try await translator.translate(gloss, task: .kslToEnglish)
            guard !Task.isCancelled else { return }
            let text = words.joined(separator: " ")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return }
            lastSpokenGloss = gloss
            spokenSentence = text
            speechOutput.speak(text, language: "en-US")
        } catch {
            spokenSentence = "sign→speech error: \(error)"
        }
    }
}
