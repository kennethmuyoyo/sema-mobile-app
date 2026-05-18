import SwiftUI

/// Shared design tokens — spacing, corner radii, and brand colors — kept in
/// one place so the visual language stays consistent across tabs and components.
enum Design {
    enum Spacing {
        static let xs: CGFloat = 4
        static let s: CGFloat = 8
        static let m: CGFloat = 12
        static let l: CGFloat = 16
        static let xl: CGFloat = 20
        static let xxl: CGFloat = 28
    }

    enum Radius {
        static let chip: CGFloat = 12
        static let card: CGFloat = 20
        static let pill: CGFloat = 999
    }

    enum BrandColor {
        /// Warm teal/mint — picked to read as "communication" without
        /// competing with the live-red indicator or system blue.
        static let accent = Color(red: 0.20, green: 0.83, blue: 0.78)
        static let live = Color.red
        static let listening = Color(red: 0.30, green: 0.90, blue: 0.55)
    }
}
