//
//  DebugMediaPipePreview.swift
//  sema
//
//  Debug-only sheet: pick a token from the recognition set and watch the
//  bundled MediaPipe landmark template play back as a 2D skeleton. Used to
//  eyeball the reference clip the PoseTemplateMatcher compares the live
//  camera feed against — same projection as the DebugCameraLandmarkPiP, so
//  the user can mentally overlay "what the template looks like" on top of
//  "what my camera is producing".
//

import SwiftUI

struct DebugMediaPipePreview: View {
    let token: String
    let database: PoseDatabase?

    @Environment(\.dismiss) private var dismiss

    @State private var frames: [NormalizedFrame] = []
    @State private var fps: Float = 24
    @State private var landmarkSource: String?
    @State private var currentIndex: Int = 0
    @State private var isPlaying = true
    @State private var loadError: String?
    @State private var playbackTask: Task<Void, Never>?

    var body: some View {
        NavigationStack {
            content
                .navigationTitle(token)
                .navigationBarTitleDisplayMode(.inline)
                .toolbar { toolbar }
                .task { await load() }
                .onDisappear { playbackTask?.cancel() }
        }
    }

    @ViewBuilder
    private var content: some View {
        if let loadError {
            ContentUnavailableView {
                Label("Couldn't load template", systemImage: "exclamationmark.triangle.fill")
            } description: {
                Text(loadError)
            }
        } else if frames.isEmpty {
            ProgressView("Loading template…")
        } else {
            VStack(spacing: 16) {
                stage
                controls
            }
            .padding()
        }
    }

    private var stage: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(Color.black)

            // Reference template uses the un-mirrored projection so the
            // reader sees the avatar's left/right, not a selfie mirror.
            DebugLandmarkSkeletonOverlay(frame: frames[currentIndex], mirrored: false)

            VStack {
                Spacer()
                metaStrip
            }
            .padding(12)
        }
        .aspectRatio(3.0 / 4.0, contentMode: .fit)
    }

    private var metaStrip: some View {
        HStack(spacing: 8) {
            Label("\(currentIndex + 1)/\(frames.count)", systemImage: "film")
                .font(.caption.monospacedDigit())
            Spacer()
            if let landmarkSource {
                Text(landmarkSource.uppercased())
                    .font(.caption2.weight(.bold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(.thinMaterial, in: .capsule)
            }
            Text("\(Int(fps)) fps")
                .font(.caption.monospacedDigit())
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.black.opacity(0.5), in: .capsule)
    }

    private var controls: some View {
        VStack(spacing: 12) {
            Slider(
                value: Binding(
                    get: { Double(currentIndex) },
                    set: { newValue in
                        playbackTask?.cancel()
                        isPlaying = false
                        currentIndex = min(frames.count - 1, max(0, Int(newValue)))
                    }
                ),
                in: 0...Double(max(frames.count - 1, 0))
            )
            .accessibilityLabel("Frame scrubber")

            HStack(spacing: 24) {
                Button("Restart", systemImage: "backward.end.fill") {
                    restart()
                }
                Button(isPlaying ? "Pause" : "Play",
                       systemImage: isPlaying ? "pause.fill" : "play.fill") {
                    if isPlaying {
                        playbackTask?.cancel()
                        isPlaying = false
                    } else {
                        startPlayback(from: currentIndex)
                    }
                }
                .bold()
            }
            .labelStyle(.titleAndIcon)
            .buttonStyle(.bordered)
        }
    }

    @ToolbarContentBuilder
    private var toolbar: some ToolbarContent {
        ToolbarItem(placement: .topBarLeading) {
            Button("Close", action: dismiss.callAsFunction)
        }
    }

    // MARK: - Loading

    private func load() async {
        guard let database else {
            loadError = "Pose database isn't ready yet — start the conversation once before opening preview."
            return
        }
        do {
            guard let clip = try await database.lookup(token) else {
                loadError = "Token '\(token)' not in pose library."
                return
            }
            let normalized = (0..<clip.frameCount).map { i in
                Self.normalizedFrame(from: clip.frame(at: i), timestamp: Double(i) / Double(clip.fps))
            }
            self.frames = normalized
            self.fps = clip.fps
            startPlayback(from: 0)
        } catch {
            loadError = "\(error)"
        }
    }

    private func startPlayback(from index: Int) {
        playbackTask?.cancel()
        currentIndex = min(frames.count - 1, max(0, index))
        isPlaying = true
        let interval = UInt64(1_000_000_000 / max(UInt64(fps), 1))
        playbackTask = Task { [interval] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: interval)
                if Task.isCancelled { return }
                await MainActor.run {
                    let next = (currentIndex + 1) % max(frames.count, 1)
                    currentIndex = next
                }
            }
        }
    }

    private func restart() {
        startPlayback(from: 0)
    }

    /// Convert one (45, 3) flat clip frame into a `NormalizedFrame`.
    /// `NormalizedFrame.values` is stride 3 (x, y, z per joint, length 135)
    /// with a sibling 45-length `mask` array — NOT stride 4 with mask
    /// interleaved as a 4th channel. The earlier `dst = j * 4` packing
    /// produced 180 floats that downstream readers indexed as stride 3 →
    /// every joint past the first ended up reading the previous joint's
    /// mask byte as its x, scrambling the skeleton in this preview.
    private static func normalizedFrame(from xyz: [Float], timestamp: TimeInterval) -> NormalizedFrame {
        precondition(xyz.count == Landmark45.count * 3, "expected 45 joints × 3 coords")
        return NormalizedFrame(
            values: xyz,
            mask: Array(repeating: 1, count: Landmark45.count),
            timestamp: timestamp
        )
    }
}
