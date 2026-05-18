import Foundation

@MainActor
protocol SpeechOutputting: AnyObject {
    func speak(_ text: String, language: String)
    func stop()
}

extension Synthesizer: SpeechOutputting {}
