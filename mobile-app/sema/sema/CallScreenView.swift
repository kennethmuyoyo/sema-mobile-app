import SwiftUI

struct CallScreenView: View {
    let coordinator: PipelineCoordinator
    let camera: CameraSessionController

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        ZStack {
            CallStageView(frame: coordinator.poseFrame, isActive: coordinator.isRunning)

            VStack(spacing: 0) {
                topBar
                Spacer()
                bottomOverlays
            }
            .padding(.horizontal, 20)
            .padding(.top, 12)
            .padding(.bottom, 16)

            FloatingPiPView(
                isActive: coordinator.isCameraEnabled && coordinator.isRunning,
                camera: camera,
                bottomReservedHeight: coordinator.areCaptionsVisible ? 248 : 136
            )
        }
        .background(Color.black)
        .ignoresSafeArea(.container, edges: .bottom)
    }

    private var topBar: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Sema")
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(.white)

                Label(coordinator.liveStatusTitle, systemImage: coordinator.isRunning ? "record.circle.fill" : "circle")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(coordinator.isRunning ? .green : .white.opacity(0.62))
            }

            Spacer()
        }
    }

    private var bottomOverlays: some View {
        VStack(spacing: 16) {
            if coordinator.areCaptionsVisible {
                LiveCaptionOverlay(
                    transcript: coordinator.transcript,
                    spokenInput: coordinator.spokenInput
                )
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }

            CallControlDock(
                isRunning: coordinator.isRunning,
                isCameraEnabled: coordinator.isCameraEnabled,
                areCaptionsVisible: coordinator.areCaptionsVisible,
                toggleCamera: coordinator.toggleCamera,
                toggleCaptions: toggleCaptions,
                toggleRunning: toggleRunning
            )
        }
        .animation(reduceMotion ? nil : .snappy(duration: 0.2), value: coordinator.areCaptionsVisible)
        .padding(.bottom, 18)
    }

    private func toggleRunning() {
        if reduceMotion {
            coordinator.toggleRunning()
        } else {
            withAnimation(.snappy(duration: 0.2)) {
                coordinator.toggleRunning()
            }
        }
    }

    private func toggleCaptions() {
        if reduceMotion {
            coordinator.toggleCaptions()
        } else {
            withAnimation(.snappy(duration: 0.2)) {
                coordinator.toggleCaptions()
            }
        }
    }
}

#Preview {
    CallScreenView(coordinator: PipelineCoordinator(), camera: CameraSessionController())
}
