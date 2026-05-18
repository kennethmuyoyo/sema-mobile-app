//
//  ContentView.swift
//  sema
//

import SwiftUI

struct ContentView: View {
    @State private var orchestrator = ConversationOrchestrator()
    @State private var volumeShortcuts = VolumeShortcutDetector()

    var body: some View {
        ConversationScreenView(orchestrator: orchestrator)
            .background {
                HiddenVolumeHUDView()
                    .frame(width: 1, height: 1)
                    .allowsHitTesting(false)
            }
            .onAppear {
                guard !TestEnvironment.skipsPipelineStartup else { return }
                orchestrator.bootstrap()
                volumeShortcuts.start()
            }
            .onDisappear {
                volumeShortcuts.stop()
                orchestrator.pause()
            }
            .onReceive(NotificationCenter.default.publisher(for: .shortcutStartConversation)) { _ in
                if orchestrator.canStart, !orchestrator.isLive {
                    orchestrator.start()
                }
            }
    }
}

#Preview {
    ContentView()
}
