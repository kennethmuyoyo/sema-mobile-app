import SwiftUI

/// Background for the Listen tab. A dark linear gradient plus a soft radial
/// halo so the avatar doesn't sit on a flat black void.
struct BackdropGradient: View {
    var body: some View {
        ZStack {
            LinearGradient(
                colors: [
                    Color(red: 0.04, green: 0.05, blue: 0.08),
                    .black
                ],
                startPoint: .top,
                endPoint: .bottom
            )
            RadialGradient(
                colors: [
                    Design.BrandColor.accent.opacity(0.25),
                    .clear
                ],
                center: .center,
                startRadius: 10,
                endRadius: 360
            )
            .blendMode(.plusLighter)
        }
        .ignoresSafeArea()
    }
}

#Preview {
    BackdropGradient()
}
