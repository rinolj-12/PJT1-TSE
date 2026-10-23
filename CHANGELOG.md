# Changelog

All notable changes to the Target Speech Extraction (TSE) on Edge Device project are documented in this file.

## [v2.0] - 2026-10-23

### Added
* **Android Release:** Packaged and published the official `TSE-Edge-arm64.apk` for 64-bit ARM Android devices.
* **Release Pipeline:** Created GitHub release with direct binary downloads attached to the release tag.
* **Documentation:**
  * Updated main `README.md` with release badges, direct download links, and installation guide.
  * Added `android-app/README.md` covering prerequisites, architecture, and step-by-step build instructions from source.

### Features
* **Multi-speaker Extraction:** Supports separating a target speaker from mixtures of 2, 3, or 4 simultaneous speakers.
* **On-device INT8 Inference:** Integrated quantised INT8 ONNX models (13.6 MB accurate / 8.6 MB compact) running via ONNX Runtime Mobile.
* **Real-time Streaming:** Low-latency mode (down to 0.4 s latency on compact model; 1.0 s latency on accurate model).
* **Zero Internet Dependency:** Operates 100% offline without network permission.
* **Built-in Benchmark:** On-device diagnostic tab to measure latency, real-time factor, and sample audio quality directly on user hardware.
