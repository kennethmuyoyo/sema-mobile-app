# Frameworks

Binary frameworks linked by the sema target. Not committed (see root
`.gitignore`) because they're either large or fetched from upstream releases.

## `llama.xcframework`

Prebuilt llama.cpp runtime for iOS device + simulator (also macOS / tvOS /
xrOS, unused here). Used by `sema/ML/LlamaGemmaEngine.swift` to run the
on-device Gemma fine-tune for KSL ↔ EN/SW translation.

### To re-fetch

```sh
cd mobile-app/sema/Frameworks
gh release download --repo ggml-org/llama.cpp --pattern 'llama-*-xcframework.zip'
unzip llama-*-xcframework.zip
mv build-apple/llama.xcframework .
rm -rf build-apple llama-*-xcframework.zip
```

### To integrate in Xcode

1. Open `sema.xcworkspace`.
2. Select the `sema` project → `sema` target → **General** tab.
3. Under **Frameworks, Libraries, and Embedded Content**, click **+**.
4. **Add Other... → Add Files...** → choose `Frameworks/llama.xcframework`.
5. Set its embed setting to **Embed & Sign**.

The module map inside the xcframework declares `framework module llama`, so
once it's linked, Swift sees `import llama` automatically — no bridging
header needed.

### Bundle size

The xcframework adds ~200 MB to the unsigned artefact; per-slice
(`ios-arm64`) Apple's app-thinning trims it to ~50 MB shipped to a device.
