//
//  semaApp.swift
//  sema
//
//  Created by Bishal Jena on 13/05/26.
//

import SwiftUI

@main
struct semaApp: App {
    var body: some Scene {
        WindowGroup {
            if TestEnvironment.isActive {
                Color.black.ignoresSafeArea()
            } else {
                ContentView()
            }
        }
    }
}
