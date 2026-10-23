# TSE Edge Android Application

Android application for offline, real-time Target Speech Extraction (TSE) using INT8-quantised ONNX models.

## Pre-built APK

The pre-built, production-ready APK is available for download:
* **Release:** [GitHub Releases v2.0](https://github.com/rinolj-12/PJT1-TSE/releases/latest)
* **Direct Download:** [TSE-Edge-arm64.apk](https://github.com/rinolj-12/PJT1-TSE/releases/download/v2.0/TSE-Edge-arm64.apk) (~28 MB)
* **Repository copy:** [`app/TSE-Edge-arm64.apk`](../app/TSE-Edge-arm64.apk)

### System Requirements
* **Architecture:** ARM64-v8a (`arm64-v8a`)
* **Minimum Android version:** Android 7.0 (API level 24)
* **Target Android version:** Android 14 (API level 34)
* **Permissions:** Record Audio (for enrollment recording only; no internet permission required)

---

## Building from Source

### Prerequisites
* Android Studio Iguana (2023.2.1) or newer
* Android SDK 34
* JDK 17 (recommended for Gradle 8.x)

### Command-line Build

```bash
# Clone the repository
git clone https://github.com/rinolj-12/PJT1-TSE.git
cd PJT1-TSE/android-app

# Build debug APK
./gradlew assembleDebug

# Build release APK
./gradlew assembleRelease
```

The resulting release APK will be located under:
`app/build/outputs/apk/release/app-arm64-v8a-release.apk`

---

## Project Structure

```
android-app/
├── app/
│   ├── src/main/
│   │   ├── java/edu/vit/tsedemo/   # Activity, UI, Audio engine, and ONNX Runtime wrapper
│   │   ├── res/                     # Layouts, themes, strings, and icons
│   │   ├── assets/                  # INT8 ONNX models and bundled demo audio
│   │   └── AndroidManifest.xml      # App manifest (no internet permission)
│   ├── build.gradle                 # App-level build config (splits, proguard, signing)
│   └── proguard-rules.pro           # ProGuard / R8 rules for ONNX Runtime
├── build.gradle                     # Top-level build config
├── settings.gradle                  # Gradle project settings
└── uitest.sh                        # Automated UI & latency testing via ADB
```
