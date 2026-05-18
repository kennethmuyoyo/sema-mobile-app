import SwiftUI

/// Header showing the current tab title, mode subtitle, and a live/ready
/// status indicator. Renders as floating typography — no card, no material —
/// so it stays editorial rather than picking up a "cheap web overlay" feel.
/// A drop-shadow on each text line keeps it legible against the Interpret
/// tab's camera feed.
struct LiveTopBar: View {
    let title: String
    let subtitle: String
    let isLive: Bool
    let status: String

    var body: some View {
        HStack(spacing: Design.Spacing.m) {
            VStack(alignment: .leading, spacing: Design.Spacing.xs) {
                titleRow
                statusRow
            }
            Spacer()
        }
    }

    private var titleRow: some View {
        HStack(spacing: Design.Spacing.s) {
            Text(title)
                .font(.title2)
                .bold()
                .foregroundStyle(.white)
                .shadow(color: .black.opacity(0.6), radius: 4, y: 2)
            if isLive {
                PulsingDot(color: Design.BrandColor.live)
            }
        }
    }

    private var statusRow: some View {
        HStack(spacing: Design.Spacing.xs) {
            Text(subtitle)
                .font(.caption)
                .foregroundStyle(Color.white.opacity(0.70))
            Text("·")
                .foregroundStyle(Color.white.opacity(0.45))
            Text(status)
                .font(.caption)
                .bold()
                .foregroundStyle(isLive ? Design.BrandColor.listening : Color.white.opacity(0.70))
                .contentTransition(.opacity)
        }
        .shadow(color: .black.opacity(0.6), radius: 3, y: 1)
    }
}

#Preview("Ready") {
    LiveTopBar(title: "Listen", subtitle: "Speech → Sign", isLive: false, status: "Ready")
        .padding()
        .background(.black)
}

#Preview("Live") {
    LiveTopBar(title: "Listen", subtitle: "Speech → Sign", isLive: true, status: "Live")
        .padding()
        .background(.black)
}
