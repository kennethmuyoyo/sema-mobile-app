import Foundation
import Observation

@MainActor
@Observable
final class PipelineCoordinator {
    enum CallState: String {
        case idle = "Ready"
        case listening = "Listening"
        case paused = "Paused"
    }

    private struct DemoPhraseStep {
        let spokenInput: String
        let transcript: String
    }

    var state: CallState = .idle
    var transcript = "Tap Start to begin the demo translation flow."
    var spokenInput = "Waiting for speech input..."
    var poseFrame = DemoPoseFrame.sample(phase: 0)
    var isCameraEnabled = false
    var areCaptionsVisible = true

    @ObservationIgnored private var demoTask: Task<Void, Never>?
    private static let demoPhraseSteps = [
        DemoPhraseStep(
            spokenInput: "I am going to the hospital tomorrow.",
            transcript: "KSL gloss: TOMORROW"
        ),
        DemoPhraseStep(
            spokenInput: "I am going to the hospital tomorrow.",
            transcript: "KSL gloss: HOSPITAL"
        ),
        DemoPhraseStep(
            spokenInput: "I am going to the hospital tomorrow.",
            transcript: "KSL gloss: I-GO"
        ),
        DemoPhraseStep(
            spokenInput: "I am going to the hospital tomorrow.",
            transcript: "KSL gloss: TOMORROW HOSPITAL I-GO"
        )
    ]

    var isRunning: Bool {
        state == .listening
    }

    var liveStatusTitle: String {
        switch state {
        case .idle:
            return "Ready"
        case .listening:
            return "Live"
        case .paused:
            return "Paused"
        }
    }

    func toggleRunning() {
        isRunning ? pause() : start()
    }

    func start() {
        state = .listening
        isCameraEnabled = true
        transcript = "KSL gloss: HOSPITAL TOMORROW I-GO"
        spokenInput = "I am going to the hospital tomorrow."
        startDemoLoop()
    }

    func pause() {
        state = .paused
        stopDemoLoop()
    }

    func toggleCamera() {
        isCameraEnabled.toggle()
    }

    func toggleCaptions() {
        areCaptionsVisible.toggle()
    }

    private func startDemoLoop() {
        stopDemoLoop()
        demoTask = Task { [weak self] in
            await self?.runDemoLoop()
        }
    }

    private func stopDemoLoop() {
        demoTask?.cancel()
        demoTask = nil
    }

    private func runDemoLoop() async {
        var phase: Double = 0

        while !Task.isCancelled {
            poseFrame = DemoPoseFrame.sample(phase: phase)
            let step = demoStep(for: phase)

            if transcript != step.transcript {
                transcript = step.transcript
            }

            if spokenInput != step.spokenInput {
                spokenInput = step.spokenInput
            }

            phase += 0.16

            try? await Task.sleep(nanoseconds: 80_000_000)
        }
    }

    private func demoStep(for phase: Double) -> DemoPhraseStep {
        let cycle = Double.pi * 2
        let normalizedPhase = phase.truncatingRemainder(dividingBy: cycle) / cycle
        let index = min(
            Int(normalizedPhase * Double(Self.demoPhraseSteps.count)),
            Self.demoPhraseSteps.count - 1
        )

        return Self.demoPhraseSteps[index]
    }
}
