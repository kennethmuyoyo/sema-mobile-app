import SwiftUI

struct CallControlDock: View {
    let isRunning: Bool
    let isCameraEnabled: Bool
    let areCaptionsVisible: Bool
    let toggleCamera: @MainActor () -> Void
    let toggleCaptions: @MainActor () -> Void
    let toggleRunning: @MainActor () -> Void

    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    var body: some View {
        HStack(spacing: 20) {
            CallControlButton(
                title: isCameraEnabled ? "Turn Camera Off" : "Turn Camera On",
                systemImage: isCameraEnabled ? "video.fill" : "video.slash.fill",
                role: .secondary,
                action: toggleCamera
            )

            CallControlButton(
                title: areCaptionsVisible ? "Captions" : "Captions Off",
                systemImage: areCaptionsVisible ? "captions.bubble.fill" : "captions.bubble",
                role: .secondary,
                action: toggleCaptions
            )

            CallControlButton(
                title: isRunning ? "Pause" : "Start",
                systemImage: isRunning ? "pause.fill" : "play.fill",
                role: .primary,
                action: toggleRunning
            )
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 14)
        .background(reduceTransparency ? Color.black : Color.black.opacity(0.56), in: Capsule())
        .overlay {
            Capsule()
                .stroke(Color.white.opacity(0.12), lineWidth: 1)
        }
    }
}

private struct CallControlButton: View {
    enum Role {
        case primary
        case secondary
    }

    let title: String
    let systemImage: String
    let role: Role
    let action: @MainActor () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.headline)
                .frame(width: 48, height: 48)
                .background(backgroundColor, in: Circle())
                .foregroundStyle(foregroundColor)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(title)
    }

    private var backgroundColor: Color {
        switch role {
        case .primary:
            return .white
        case .secondary:
            return .white.opacity(0.16)
        }
    }

    private var foregroundColor: Color {
        switch role {
        case .primary:
            return .black
        case .secondary:
            return .white
        }
    }
}

#Preview {
    ZStack {
        Color.black.ignoresSafeArea()
        CallControlDock(
            isRunning: true,
            isCameraEnabled: true,
            areCaptionsVisible: true,
            toggleCamera: {},
            toggleCaptions: {},
            toggleRunning: {}
        )
    }
}
