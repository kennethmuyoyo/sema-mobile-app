import SwiftUI

/// Affordance shown inside a CaptionCard when no transcription has happened
/// yet. Tells the user exactly which action will produce text in this card.
struct EmptyCaptionView: View {
    let systemImage: String
    let title: String
    let description: String

    var body: some View {
        HStack(alignment: .center, spacing: Design.Spacing.m) {
            Image(systemName: systemImage)
                .font(.title2)
                .foregroundStyle(Design.BrandColor.accent)
                .frame(width: 28)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.subheadline)
                    .bold()
                    .foregroundStyle(.white)
                Text(description)
                    .font(.footnote)
                    .foregroundStyle(Color.white.opacity(0.70))
            }
            Spacer(minLength: 0)
        }
    }
}

#Preview {
    EmptyCaptionView(
        systemImage: "mic.fill",
        title: "Tap Start, then speak",
        description: "The avatar will sign in real time."
    )
    .padding()
    .background(.black)
}
