import SwiftUI

/// Single-screen live translation: avatar plus sign/speech capture with manual Start/Stop.
struct ConversationScreenView: View {
    @Bindable var orchestrator: ConversationOrchestrator

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var isDebugGlossPlayerPresented = false

    var body: some View {
        ZStack {
            BackdropGradient()

            SimpleAvatar3DView(frame: orchestrator.pathB.player.rawFrame)
                .padding(.horizontal, Design.Spacing.l)
                .padding(.top, 56)
                .padding(.bottom, 48)
                .accessibilityLabel("Signing avatar")

            VStack(spacing: 0) {
                topBarWithDebugAffordance
                Spacer()
                bottomPanel
            }
            .padding(.horizontal, Design.Spacing.l)
            .padding(.top, Design.Spacing.m)
            .padding(.bottom, Design.Spacing.l)

            if orchestrator.isLive {
                DebugCameraLandmarkPiP(
                    camera: orchestrator.camera,
                    landmarkFrame: orchestrator.pathA.latestFrame,
                    onTap: { orchestrator.toggleMode() },
                    isWatching: orchestrator.isWatching
                )
                .padding(.top, 64)
                .padding(.trailing, Design.Spacing.m)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
            }
        }
        .background(.black)
        .onChange(of: orchestrator.pathA.finalizedGlossPhrase) { _, new in
            orchestrator.handleFinalizedGloss(new)
        }
        .sheet(isPresented: $isDebugGlossPlayerPresented) {
            DebugGlossPlayer(pathB: orchestrator.pathB)
        }
    }

    /// Hidden affordance: long-press the title to open the gloss-picker
    /// sheet. Gated by `DebugFeatures.glossPlayer` so it's a no-op in
    /// shipped builds — users will never see it without explicit gesture.
    private var topBarWithDebugAffordance: some View {
        let bar = LiveTopBar(
            title: "Sema",
            subtitle: "Live translation",
            isLive: orchestrator.isLive,
            status: orchestrator.statusLabel
        )
        return Group {
            if DebugFeatures.glossPlayer {
                bar.onLongPressGesture(minimumDuration: 0.6) {
                    isDebugGlossPlayerPresented = true
                }
            } else {
                bar
            }
        }
    }

    private var bottomPanel: some View {
        VStack(spacing: Design.Spacing.m) {
            if orchestrator.shouldShowCaptions {
                ConversationCaptionCard(
                    heardText: orchestrator.pathB.heardCaption,
                    avatarSigning: orchestrator.pathB.glossStream.joined(separator: " "),
                    signTranscript: orchestrator.pathA.transcript,
                    topGlossCandidates: orchestrator.pathA.topPredictions,
                    glossEnglishSentence: orchestrator.bridge.spokenSentence,
                    errorMessage: orchestrator.combinedError,
                    isLive: orchestrator.isLive,
                    isListening: orchestrator.isListening,
                    isWatching: orchestrator.isWatching
                )
                .transition(.move(edge: .bottom).combined(with: .opacity))
            } else if let error = orchestrator.combinedError, orchestrator.isLive {
                ConversationCaptionCard(
                    heardText: "",
                    signTranscript: "",
                    errorMessage: error,
                    isLive: true,
                    isListening: orchestrator.isListening,
                    isWatching: orchestrator.isWatching
                )
            }

            if !orchestrator.isLive {
                readinessHint
            }

            ConversationControlDock(
                isLive: orchestrator.isLive,
                canStart: orchestrator.canStart,
                isStarting: orchestrator.isStartingSession,
                areCaptionsVisible: orchestrator.areCaptionsVisible,
                toggleCaptions: { orchestrator.toggleCaptions() },
                toggleSession: {
                    if orchestrator.isLive {
                        orchestrator.pause()
                    } else {
                        orchestrator.start()
                    }
                }
            )
        }
        .padding(.bottom, Design.Spacing.l)
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.2), value: orchestrator.shouldShowCaptions)
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.2), value: orchestrator.areCaptionsVisible)
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.2), value: orchestrator.canStart)
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.2), value: orchestrator.isLive)
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.2), value: orchestrator.statusLabel)
    }

    @ViewBuilder
    private var readinessHint: some View {
        if orchestrator.isPreparing {
            Text("Preparing models…")
                .font(.footnote.weight(.medium))
                .foregroundStyle(.white.opacity(0.72))
        } else if orchestrator.canStart {
            Text("Tap Start to begin translation")
                .font(.footnote.weight(.medium))
                .foregroundStyle(.white.opacity(0.72))
                .accessibilityLabel("Tap Start to begin translation")
        } else if !orchestrator.mediaPermissions.canListen {
            Text("Allow microphone and speech recognition in Settings to start.")
                .font(.footnote.weight(.medium))
                .foregroundStyle(.white.opacity(0.72))
                .multilineTextAlignment(.center)
        } else {
            Text(orchestrator.statusLabel)
                .font(.footnote.weight(.medium))
                .foregroundStyle(.white.opacity(0.72))
        }
    }
}

#Preview {
    ConversationScreenView(orchestrator: .forPreview())
}
