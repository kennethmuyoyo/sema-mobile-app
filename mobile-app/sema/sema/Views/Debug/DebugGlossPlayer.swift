//
//  DebugGlossPlayer.swift
//  sema
//
//  Debug-only: pick any token in the bundled PoseLibrary and either
//    1. play it on the avatar via PathBCoordinator.playGlossSequence
//       (bypasses ASR + Gemma — the controlled-generation surface), or
//    2. preview the bundled MediaPipe template as a 2D skeleton so the
//       reference is visible alongside the live camera PiP.
//
//  Reachable via a long-press on the LiveTopBar in ConversationScreenView.
//  Gated by DebugFeatures.glossPlayer so it doesn't ship to real users.
//

import SwiftUI

struct DebugGlossPlayer: View {
    let pathB: PathBCoordinator

    @Environment(\.dismiss) private var dismiss

    /// All tokens in the bundled index, alphabetical.
    @State private var tokens: [String] = []
    /// Tokens whose `source` is `single_gloss/recognition_set` — surfaced as
    /// their own section because that's the controlled corpus we want to
    /// eyeball under MediaPipe.
    @State private var recognitionSetTokens: [String] = []
    @State private var selected: [String] = []
    @State private var searchText: String = ""
    @State private var loadError: String?
    @State private var previewToken: PreviewToken?

    /// Local Identifiable wrapper so `.sheet(item:)` can drive off it without
    /// requiring a retroactive `String: Identifiable` conformance.
    private struct PreviewToken: Identifiable, Equatable {
        let token: String
        var id: String { token }
    }

    var body: some View {
        NavigationStack {
            content
                .navigationTitle("Vocabulary")
                .navigationBarTitleDisplayMode(.inline)
                .searchable(text: $searchText, prompt: "Filter tokens")
                .toolbar { toolbar }
                .task { await loadTokens() }
                .sheet(item: $previewToken) { entry in
                    DebugMediaPipePreview(token: entry.token, database: pathB.database)
                }
        }
    }

    @ViewBuilder
    private var content: some View {
        if let loadError {
            ContentUnavailableView {
                Label("Couldn't load pose library", systemImage: "exclamationmark.triangle.fill")
            } description: {
                Text(loadError)
            }
        } else if tokens.isEmpty {
            ProgressView("Loading tokens…")
        } else {
            tokenList
        }
    }

    private var tokenList: some View {
        List {
            if !selected.isEmpty {
                Section("Queued sequence") {
                    HStack {
                        Text(selected.joined(separator: " "))
                            .font(.body.monospaced())
                        Spacer()
                        Button("Clear", role: .destructive) { selected.removeAll() }
                            .buttonStyle(.borderless)
                    }
                }
            }

            if !recognitionSetTokens.isEmpty, searchText.isEmpty {
                Section {
                    ForEach(recognitionSetTokens, id: \.self) { token in
                        recognitionSetRow(token)
                    }
                } header: {
                    Text("MediaPipe preview · recognition set")
                } footer: {
                    Text("Plays the bundled MediaPipe template as a 2D skeleton — what the PoseTemplateMatcher compares the live camera feed against.")
                }
            }

            Section("All tokens (\(filtered.count) of \(tokens.count))") {
                ForEach(filtered, id: \.self) { token in
                    Button {
                        toggle(token)
                    } label: {
                        HStack {
                            Text(token)
                                .font(.body.monospaced())
                                .foregroundStyle(.primary)
                            Spacer()
                            if selected.contains(token) {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundStyle(Color.accentColor)
                            }
                        }
                    }
                }
            }
        }
        .listStyle(.insetGrouped)
    }

    private func recognitionSetRow(_ token: String) -> some View {
        HStack {
            Button {
                toggle(token)
            } label: {
                HStack {
                    Text(token)
                        .font(.body.monospaced())
                        .foregroundStyle(.primary)
                    if selected.contains(token) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(Color.accentColor)
                    }
                }
            }
            .buttonStyle(.plain)
            Spacer()
            Button("Preview", systemImage: "waveform.path") {
                previewToken = PreviewToken(token: token)
            }
            .labelStyle(.iconOnly)
            .buttonStyle(.bordered)
            .controlSize(.small)
            .accessibilityLabel("Preview \(token) on MediaPipe skeleton")
        }
    }

    @ToolbarContentBuilder
    private var toolbar: some ToolbarContent {
        ToolbarItem(placement: .topBarLeading) {
            Button("Close", action: dismiss.callAsFunction)
        }
        ToolbarItem(placement: .topBarTrailing) {
            Button("Play") {
                playSelected()
            }
            .bold()
            .disabled(selected.isEmpty)
        }
    }

    private var filtered: [String] {
        guard !searchText.isEmpty else { return tokens }
        let needle = searchText.uppercased()
        return tokens.filter { $0.contains(needle) }
    }

    private func toggle(_ token: String) {
        if let index = selected.firstIndex(of: token) {
            selected.remove(at: index)
        } else {
            selected.append(token)
        }
    }

    private func playSelected() {
        guard !selected.isEmpty else { return }
        pathB.playGlossSequence(selected)
        dismiss()
    }

    /// Reads BOTH bundled libraries — the curated demo set
    /// (`PoseLibrary/index.json`) and the v11-derived full set
    /// (`PoseLibraryFull/index_full.json`) — and merges them. Tokens with the
    /// `_full__` filename prefix in the full library are normalised back to
    /// their gloss name (e.g. `_full__BANK.npz` → token `"BANK"`) so the UI
    /// shows clean labels.
    private func loadTokens() async {
        do {
            var merged: [String: [String: Any]] = [:]
            for (resource, subdir) in [("index", "PoseLibrary"),
                                        ("index_full", "PoseLibraryFull")] {
                if let dict = try loadIndex(resource: resource, subdir: subdir) {
                    for (k, v) in dict {
                        merged[k] = v
                    }
                }
            }
            if merged.isEmpty {
                loadError = "no pose library index.json found in bundle"
                return
            }
            tokens = merged.keys.sorted()
            recognitionSetTokens = merged.compactMap { token, entry in
                guard let source = entry["source"] as? String,
                      source == "single_gloss/recognition_set"
                else { return nil }
                return token
            }.sorted()
        } catch {
            loadError = "\(error)"
        }
    }

    /// Best-effort: try the subdirectory first, then the bundle root in case
    /// Xcode 16's fileSystemSynchronizedGroup flattened the folder. Returns
    /// the decoded outer dictionary, or nil if the resource isn't present.
    private func loadIndex(resource: String, subdir: String) throws -> [String: [String: Any]]? {
        let url = Bundle.main.url(forResource: resource,
                                    withExtension: "json",
                                    subdirectory: subdir)
                ?? Bundle.main.url(forResource: resource, withExtension: "json")
        guard let url else { return nil }
        let data = try Data(contentsOf: url)
        let decoded = try JSONSerialization.jsonObject(with: data)
        return decoded as? [String: [String: Any]]
    }
}

#Preview {
    DebugGlossPlayer(pathB: PathBCoordinator())
}
