import SwiftUI

/// Rounded caption surface for readable text over the avatar backdrop.
struct CaptionCard<Content: View>: View {
    let content: Content

    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    var body: some View {
        content
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            .background(backgroundStyle, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .stroke(Color.white.opacity(0.12), lineWidth: 1)
            }
    }

    private var backgroundStyle: Color {
        reduceTransparency ? Color.black : Color.black.opacity(0.54)
    }
}

#Preview {
    CaptionCard {
        Text("Hello, how are you today?")
            .foregroundStyle(.white)
    }
    .padding()
    .background(.black)
}
