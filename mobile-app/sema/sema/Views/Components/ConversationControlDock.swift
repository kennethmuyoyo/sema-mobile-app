import SwiftUI

typealias ConversationControlAction = @MainActor () -> Void

/// Bottom control bar: captions toggle and manual Start/Stop for the live session.
struct ConversationControlDock: View {
    let isLive: Bool
    let canStart: Bool
    var isStarting: Bool = false
    let areCaptionsVisible: Bool
    let toggleCaptions: ConversationControlAction
    let toggleSession: ConversationControlAction

    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    private var sessionEnabled: Bool {
        isLive || (canStart && !isStarting)
    }

    var body: some View {
        HStack(spacing: 32) {
            ConversationControlButton(
                title: areCaptionsVisible ? "Hide Captions" : "Show Captions",
                systemImage: areCaptionsVisible ? "captions.bubble.fill" : "captions.bubble",
                accessibilityValue: areCaptionsVisible ? "On" : "Off",
                role: .secondary,
                action: { toggleCaptions() }
            )

            ConversationControlButton(
                title: isLive ? "Stop" : "Start",
                systemImage: isLive ? "stop.fill" : "play.fill",
                accessibilityValue: sessionAccessibilityValue,
                role: .primary,
                isEnabled: sessionEnabled,
                action: { toggleSession() }
            )
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(reduceTransparency ? Color.black : Color.black.opacity(0.56), in: Capsule())
        .overlay {
            Capsule().stroke(Color.white.opacity(0.12), lineWidth: 1)
        }
    }

    private var sessionAccessibilityValue: String {
        if isLive { return "Running" }
        if canStart { return "Ready" }
        return "Preparing"
    }
}

private struct ConversationControlButton: View {
    enum Role {
        case primary
        case secondary
    }

    let title: String
    let systemImage: String
    let accessibilityValue: String
    let role: Role
    var isEnabled: Bool = true
    let action: ConversationControlAction

    var body: some View {
        Button {
            action()
        } label: {
            Label(title, systemImage: systemImage)
                .labelStyle(.iconOnly)
                .font(.headline)
                .frame(width: 48, height: 48)
                .background(backgroundColor, in: Circle())
                .foregroundStyle(foregroundColor)
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .opacity(isEnabled ? 1 : 0.45)
        .accessibilityLabel(accessibilityLabel)
        .accessibilityValue(accessibilityValue)
        .accessibilityInputLabels([title])
    }

    private var accessibilityLabel: String {
        switch title {
        case "Start": "Start translation"
        case "Stop": "Stop translation"
        default: title
        }
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
        ConversationControlDock(
            isLive: true,
            canStart: true,
            areCaptionsVisible: true,
            toggleCaptions: {},
            toggleSession: {}
        )
        .padding()
    }
}
