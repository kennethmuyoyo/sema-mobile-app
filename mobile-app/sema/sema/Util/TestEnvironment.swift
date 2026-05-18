import Foundation

enum TestEnvironment {
    /// True when the process is running under XCTest (unit test host).
    static var isActive: Bool {
        ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
    }

    /// True when Xcode is running SwiftUI Previews (`XCODE_RUNNING_FOR_PLAYGROUNDS`).
    static var isPreview: Bool {
        ProcessInfo.processInfo.environment["XCODE_RUNNING_FOR_PLAYGROUNDS"] == "1"
    }

    /// Skip model bootstrap, camera, and other pipeline startup (tests + previews).
    static var skipsPipelineStartup: Bool {
        isActive || isPreview
    }
}
