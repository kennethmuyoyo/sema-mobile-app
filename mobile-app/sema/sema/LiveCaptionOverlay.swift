import SwiftUI

struct LiveCaptionOverlay: View {
    let transcript: String
    let spokenInput: String

    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(spokenInput)
                .font(.headline)
                .foregroundStyle(.white)
                .lineLimit(2)

            Text(transcript)
                .font(.subheadline)
                .foregroundStyle(.white.opacity(0.72))
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
        .background(backgroundStyle, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(Color.white.opacity(0.12), lineWidth: 1)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Live captions. \(spokenInput). Gloss. \(transcript)")
    }

    private var backgroundStyle: Color {
        reduceTransparency ? Color.black : Color.black.opacity(0.54)
    }
}

#Preview {
    ZStack {
        Color.black.ignoresSafeArea()
        LiveCaptionOverlay(
            transcript: "KSL gloss: HOSPITAL TOMORROW I-GO",
            spokenInput: "I am going to the hospital tomorrow."
        )
        .padding()
    }
}
