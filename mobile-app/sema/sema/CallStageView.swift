import SwiftUI

struct CallStageView: View {
    let frame: DemoPoseFrame
    let isActive: Bool

    var body: some View {
        ZStack {
            RadialGradient(
                colors: [
                    Color.cyan.opacity(isActive ? 0.22 : 0.12),
                    Color.blue.opacity(0.12),
                    Color.black
                ],
                center: .center,
                startRadius: 80,
                endRadius: 520
            )
            .ignoresSafeArea()

            AvatarCanvasView(frame: frame, isActive: isActive)
                .padding(.horizontal, 20)
                .padding(.vertical, 96)

            VStack(spacing: 8) {
                Text(isActive ? "Signing Avatar" : "Ready to Translate")
                    .font(.title2.weight(.semibold))
                    .foregroundStyle(.white)

                Text(isActive ? "KSL animation is playing live" : "Start translation to animate signs")
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.68))
            }
            .padding(.top, 80)
            .frame(maxHeight: .infinity, alignment: .top)
        }
        .accessibilityElement(children: .contain)
    }
}

#Preview {
    CallStageView(frame: .sample(phase: 0.3), isActive: true)
}
