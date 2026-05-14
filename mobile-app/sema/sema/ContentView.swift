//
//  ContentView.swift
//  sema
//
//  Created by Bishal Jena on 13/05/26.
//

import SwiftUI

struct ContentView: View {
    @State private var coordinator = PipelineCoordinator()
    @State private var camera = CameraSessionController()

    var body: some View {
        CallScreenView(coordinator: coordinator, camera: camera)
        .onDisappear {
            coordinator.pause()
            camera.stop()
        }
    }
}

#Preview {
    ContentView()
}
