import SwiftUI

/// Captions for the conversation screen, routed per **track**. Each path owns
/// its own observable state that persists across mode switches; the card just
/// shows the slice belonging to whichever mode is currently active so the two
/// tracks never bleed into each other on screen.
///
///   • **Listening** (speech → sign, generation track) — shows what the mic
///     heard and what the avatar is signing in response.
///   • **Watching**  (sign → speech, recognition track) — shows the
///     top-3 gloss candidates, the assembled glossed sentence, and the
///     English translation beneath.
struct ConversationCaptionCard: View {
    // --- Generation track (listening) ----------------------------------
    let heardText: String
    /// Glosses currently feeding the stitcher → avatar (PathB output).
    var avatarSigning: String = ""

    // --- Recognition track (watching) ----------------------------------
    let signTranscript: String
    var topGlossCandidates: [GlossToken] = []
    var glossEnglishSentence: String = ""

    // --- Shared --------------------------------------------------------
    let errorMessage: String?
    let isLive: Bool
    /// Which track to render. When neither (idle / one-shot error), the
    /// generation track is the default so the card has something to show
    /// once the user starts speaking.
    var isListening: Bool = true
    var isWatching: Bool = false

    var body: some View {
        CaptionCard {
            if isEmpty {
                EmptyCaptionView(
                    systemImage: trackSymbol,
                    title: isLive ? trackEmptyTitle : "Preparing…",
                    description: isLive ? trackEmptyDescription : "Tap Start when models are ready."
                )
            } else {
                VStack(alignment: .leading, spacing: 8) {
                    if isWatching {
                        recognitionSection
                    } else {
                        generationSection
                    }
                    if let errorMessage {
                        Text(errorMessage)
                            .font(.footnote)
                            .foregroundStyle(.orange)
                    }
                }
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(accessibilitySummary)
    }

    /// Generation track — listening mode. Hearing user speaks; avatar signs.
    @ViewBuilder
    private var generationSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            if !heardText.isEmpty {
                labeledRow(label: "Heard", text: heardText, style: .primary)
            }
            // if !avatarSigning.isEmpty {
            //     labeledRow(label: "Signing", text: avatarSigning, style: .secondary)
            // }
        }
    }

    /// Recognition track — watching mode. Deaf user signs; system speaks.
    @ViewBuilder
    private var recognitionSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            if !topGlossCandidates.isEmpty {
                Text("Top suggestions")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.white.opacity(0.55))
                HStack(spacing: 6) {
                    ForEach(topGlossCandidates.prefix(3), id: \.stableID) { token in
                        glossChip(label: token.label, confidence: token.confidence)
                    }
                }
            }
            if !signTranscript.isEmpty {
                labeledRow(label: "Glossed", text: signTranscript, style: .primary)
            }
            if !glossEnglishSentence.isEmpty {
                labeledRow(label: "English", text: glossEnglishSentence, style: .secondary)
            }
        }
    }

    private func glossChip(label: String, confidence: Float) -> some View {
        HStack(spacing: 4) {
            Text(label)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.white)
            Text(String(format: "%.2f", confidence))
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.white.opacity(0.62))
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color.white.opacity(0.12), in: Capsule())
    }

    /// The card considers the active-track's content when deciding emptiness —
    /// idle data from the other track doesn't keep an old caption alive.
    private var isEmpty: Bool {
        if errorMessage != nil { return false }
        if isWatching {
            return signTranscript.isEmpty
                && topGlossCandidates.isEmpty
                && glossEnglishSentence.isEmpty
        }
        return heardText.isEmpty && avatarSigning.isEmpty
    }

    private var trackSymbol: String { isWatching ? "hand.raised" : "waveform" }
    private var trackEmptyTitle: String {
        isWatching ? "Watching for signs…" : "Listening for speech…"
    }
    private var trackEmptyDescription: String {
        isWatching
            ? "Sign in front of the camera — recognition runs automatically."
            : "Speak — the avatar will sign your message."
    }

    private var accessibilitySummary: String {
        var parts: [String] = []
        if isWatching {
            if !signTranscript.isEmpty { parts.append("Signed. \(signTranscript)") }
            if !glossEnglishSentence.isEmpty { parts.append("English. \(glossEnglishSentence)") }
        } else {
            if !heardText.isEmpty { parts.append("Heard. \(heardText)") }
            if !avatarSigning.isEmpty { parts.append("Signing. \(avatarSigning)") }
        }
        if let errorMessage { parts.append(errorMessage) }
        return parts.isEmpty ? "No captions yet." : parts.joined(separator: " ")
    }

    private enum RowStyle {
        case primary
        case secondary
    }

    private func labeledRow(label: String, text: String, style: RowStyle) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.white.opacity(0.55))
            Text(text)
                .font(style == .primary ? .headline : .subheadline)
                .foregroundStyle(style == .primary ? .white : .white.opacity(0.72))
                .lineLimit(2)
        }
    }
}

#Preview("Listening") {
    ZStack {
        Color.black.ignoresSafeArea()
        ConversationCaptionCard(
            heardText: "How are you today?",
            avatarSigning: "HOW YOU",
            signTranscript: "",
            errorMessage: nil,
            isLive: true,
            isListening: true,
            isWatching: false
        )
        .padding()
    }
}

#Preview("Watching") {
    ZStack {
        Color.black.ignoresSafeArea()
        ConversationCaptionCard(
            heardText: "",
            signTranscript: "ME PAIN THROAT",
            glossEnglishSentence: "I have a sore throat.",
            errorMessage: nil,
            isLive: true,
            isListening: false,
            isWatching: true
        )
        .padding()
    }
}
