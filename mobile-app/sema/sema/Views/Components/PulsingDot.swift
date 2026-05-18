import SwiftUI

/// Small filled circle with an outward-expanding ring. Used to indicate a
/// live capture state without occupying much chrome.
///
/// Respects Reduce Motion: when the user has disabled animations, the ring
/// stays static and the dot just renders flat — no infinite-loop animation.
struct PulsingDot: View {
    let color: Color
    var size: CGFloat = 8

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var animate = false

    var body: some View {
        Circle()
            .fill(color)
            .frame(width: size, height: size)
            .overlay(ring)
            .onAppear {
                guard !reduceMotion else { return }
                withAnimation(.easeOut(duration: 1.4).repeatForever(autoreverses: false)) {
                    animate = true
                }
            }
    }

    private var ring: some View {
        Circle()
            .stroke(color.opacity(0.7), lineWidth: 2)
            .scaleEffect(animate ? 2.6 : 1)
            .opacity(animate ? 0 : 0.9)
    }
}

#Preview {
    PulsingDot(color: .red)
        .padding()
        .background(.black)
}
