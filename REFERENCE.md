# Handy - Complete Codebase Reference

> Cross-platform desktop speech-to-text application with AMD GPU support via Vulkan

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Rust Backend](#rust-backend)
3. [React Frontend](#react-frontend)
4. [ML Models & Transcription](#ml-models--transcription)
5. [Audio Pipeline](#audio-pipeline)
6. [Build System](#build-system)
7. [Configuration](#configuration)
8. [Platform Notes](#platform-notes)

---

## Architecture Overview

Handy is built with **Tauri 2.x**, combining a Rust backend for performance-critical operations with a React/TypeScript frontend for the UI.

```
┌─────────────────────────────────────────────────────────────┐
│                      React Frontend                          │
│  (Settings UI, Model Selector, Onboarding, Overlay)         │
├─────────────────────────────────────────────────────────────┤
│                    Tauri Bridge (IPC)                        │
│              (Commands, Events, State)                       │
├─────────────────────────────────────────────────────────────┤
│                      Rust Backend                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │   Audio     │ │   Model     │ │   Transcription     │   │
│  │   Manager   │ │   Manager   │ │   Manager           │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Audio Toolkit                           │   │
│  │  (Recording, VAD, Resampling, Visualization)        │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              transcribe-rs                           │   │
│  │  (Whisper, Parakeet, Moonshine engines)             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Rust Backend

### Directory Structure

```
src-tauri/src/
├── lib.rs                 # Main entry, Tauri setup, manager init
├── main.rs                # Windows entry point
├── managers/
│   ├── mod.rs            # Module exports
│   ├── audio.rs          # Audio recording & device management
│   ├── model.rs          # Model downloading & management
│   ├── transcription.rs  # Speech-to-text processing
│   └── history.rs        # Transcription history (SQLite)
├── commands/
│   ├── mod.rs            # Command exports
│   ├── audio.rs          # Audio-related Tauri commands
│   ├── models.rs         # Model management commands
│   ├── transcription.rs  # Transcription commands
│   └── history.rs        # History commands
├── audio_toolkit/
│   ├── mod.rs            # Toolkit exports
│   ├── audio/
│   │   ├── device.rs     # Audio device enumeration
│   │   ├── recorder.rs   # Microphone recording
│   │   ├── resampler.rs  # Sample rate conversion
│   │   ├── utils.rs      # Audio utilities
│   │   └── visualizer.rs # Audio level visualization
│   ├── vad/
│   │   ├── silero.rs     # Silero VAD implementation
│   │   └── smoothed.rs   # Smoothed VAD wrapper
│   ├── text.rs           # Text processing, hallucination filtering
│   └── constants.rs      # Audio constants
├── settings.rs            # App settings management
├── shortcut/
│   ├── handler.rs        # Shortcut event handling
│   ├── handy_keys.rs     # Custom keyboard manager
│   └── tauri_impl.rs     # Tauri shortcut integration
├── overlay.rs             # Recording overlay window
├── tray.rs               # System tray icon & menu
├── tray_i18n.rs          # Tray internationalization
├── clipboard.rs          # Clipboard & paste operations
├── input.rs              # Input simulation (typing)
├── llm_client.rs         # LLM API client (optional)
├── signal_handle.rs      # Unix signal handling
└── utils.rs              # General utilities
```

### Core Managers

#### AudioManager (`managers/audio.rs`)
Handles microphone recording and audio device management.

```rust
pub struct AudioManager {
    app_handle: AppHandle,
    audio_state: Arc<Mutex<AudioState>>,
    is_recording: Arc<AtomicBool>,
}
```

**Key Functions:**
- `start_recording()` - Begin capturing from microphone
- `stop_recording()` - Stop capture, return audio samples
- `get_audio_devices()` - List available input devices
- `set_audio_device()` - Select recording device

#### ModelManager (`managers/model.rs`)
Manages model downloading, storage, and selection.

```rust
pub struct ModelManager {
    app_handle: AppHandle,
    models_dir: PathBuf,
    available_models: Mutex<HashMap<String, ModelInfo>>,
}
```

**Available Models:**
| ID | Name | Size | Engine | Notes |
|----|------|------|--------|-------|
| `small` | Whisper Small | 487MB | Whisper | Fast, fairly accurate |
| `medium` | Whisper Medium | 492MB | Whisper | Good accuracy |
| `turbo` | Whisper Turbo | 1.6GB | Whisper | Balanced |
| `large` | Whisper Large | 1.1GB | Whisper | Most accurate |
| `parakeet-tdt-0.6b-v2` | Parakeet V2 | 473MB | Parakeet | Best for English |
| `parakeet-tdt-0.6b-v3` | Parakeet V3 | 478MB | Parakeet | Fast & accurate |
| `moonshine-base` | Moonshine Base | 58MB | Moonshine | Very fast, English only |

#### TranscriptionManager (`managers/transcription.rs`)
Orchestrates the transcription pipeline.

```rust
pub struct TranscriptionManager {
    engine: Arc<Mutex<Option<LoadedEngine>>>,
    model_manager: Arc<ModelManager>,
    // ... idle timeout, loading state
}
```

**Key Functions:**
- `load_model(model_id)` - Load a transcription model
- `transcribe(audio)` - Convert audio samples to text
- `unload_model()` - Free model from memory

**Audio Quality Checks:**
- RMS energy threshold: `0.001`
- Peak amplitude threshold: `0.01`
- Zero crossing rate: `0.01 - 0.5`
- Minimum samples: `4000` (0.25s at 16kHz)

**Hallucination Detection:**
- Retries up to 2 times on suspected hallucination
- Exponential backoff: 100ms, 200ms, 400ms
- Filters common hallucination patterns

#### HistoryManager (`managers/history.rs`)
Stores transcription history in SQLite.

**Database:** `%APPDATA%/com.pais.handy/history.db`

**Schema:**
```sql
CREATE TABLE transcriptions (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    duration_ms INTEGER,
    model_id TEXT
);
```

### Audio Toolkit

#### Voice Activity Detection (VAD)
Uses Silero VAD v4 ONNX model for speech detection.

**Location:** `src-tauri/resources/models/silero_vad_v4.onnx`

```rust
// VAD configuration
pub const VAD_THRESHOLD: f32 = 0.5;
pub const MIN_SPEECH_DURATION_MS: u32 = 250;
pub const MIN_SILENCE_DURATION_MS: u32 = 300;
```

#### Resampling
Converts audio from device sample rate to 16kHz for models.

```rust
// Uses rubato crate for high-quality resampling
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32>
```

#### Text Processing (`audio_toolkit/text.rs`)
- `filter_transcription_output()` - Remove filler words
- `is_likely_hallucination()` - Detect model hallucinations
- `apply_custom_words()` - Apply user-defined word corrections

### Tauri Commands

Commands are exposed to the frontend via `#[tauri::command]`:

```rust
// Audio
#[tauri::command] fn start_recording()
#[tauri::command] fn stop_recording() -> Vec<f32>
#[tauri::command] fn get_audio_devices() -> Vec<AudioDevice>

// Models
#[tauri::command] fn get_available_models() -> Vec<ModelInfo>
#[tauri::command] fn download_model(model_id: String)
#[tauri::command] fn delete_model(model_id: String)

// Transcription
#[tauri::command] fn transcribe(audio: Vec<f32>) -> String
#[tauri::command] fn load_model(model_id: String)
#[tauri::command] fn unload_model()

// Settings
#[tauri::command] fn get_settings() -> Settings
#[tauri::command] fn set_settings(settings: Settings)
```

---

## React Frontend

### Directory Structure

```
src/
├── App.tsx               # Main app component, routing
├── main.tsx              # React entry point
├── bindings.ts           # Auto-generated Tauri type bindings
├── components/
│   ├── settings/         # Settings UI (35+ files)
│   │   ├── SettingsPanel.tsx
│   │   ├── GeneralSettings.tsx
│   │   ├── AudioSettings.tsx
│   │   ├── ShortcutSettings.tsx
│   │   └── ...
│   ├── model-selector/   # Model management UI
│   │   ├── ModelSelector.tsx
│   │   ├── ModelCard.tsx
│   │   └── DownloadProgress.tsx
│   ├── onboarding/       # First-run experience
│   │   ├── OnboardingFlow.tsx
│   │   ├── WelcomeStep.tsx
│   │   └── PermissionsStep.tsx
│   └── ui/               # shadcn/ui components
├── hooks/
│   ├── useSettings.ts    # Settings state hook
│   ├── useModels.ts      # Model management hook
│   └── useAudio.ts       # Audio state hook
├── stores/
│   └── settingsStore.ts  # Zustand settings store
├── overlay/              # Recording overlay window
│   ├── Overlay.tsx
│   └── overlay-main.tsx
├── i18n/
│   ├── index.ts          # i18n setup
│   ├── languages.ts      # Language metadata
│   └── locales/
│       ├── en/translation.json
│       ├── es/translation.json
│       ├── fr/translation.json
│       └── vi/translation.json
└── lib/
    ├── utils.ts          # Utility functions
    └── cn.ts             # Class name helper
```

### State Management

**Zustand Store (`stores/settingsStore.ts`):**
```typescript
interface SettingsState {
  selectedModel: string;
  selectedLanguage: string;
  translateToEnglish: boolean;
  customWords: string[];
  shortcut: string;
  pasteMethod: 'clipboard' | 'type';
  // ...
}
```

**React Query** for server state (model lists, download progress)

### Key Components

#### SettingsPanel
Main settings interface with tabs:
- General (language, translation)
- Audio (device selection, VAD sensitivity)
- Shortcuts (global hotkey configuration)
- Models (download, select, delete)
- Advanced (debug mode, experimental features)

#### ModelSelector
Model download and selection UI:
- Shows available models with accuracy/speed scores
- Download progress with resume support
- Model size and engine type indicators

#### Overlay
Recording indicator window:
- Shows during active recording
- Audio level visualization
- Displays selected model name

### Internationalization (i18n)

Uses `i18next` with React integration.

**Adding new text:**
1. Add key to `src/i18n/locales/en/translation.json`
2. Use in component: `const { t } = useTranslation(); t('key.path')`

**Supported languages:** English, Spanish, French, Vietnamese

---

## ML Models & Transcription

### Transcription Engines

#### 1. Whisper (via whisper-rs)
- OpenAI's Whisper model, compiled with whisper.cpp
- Supports multiple languages
- GGML format (quantized)
- GPU: Vulkan, Metal, CUDA

#### 2. Parakeet (via transcribe-rs)
- NVIDIA's Parakeet TDT models
- English-only, highly accurate
- ONNX format (int8 quantized)
- GPU: ONNX Runtime providers

#### 3. Moonshine (via transcribe-rs)
- Lightweight, fast model
- English-only, handles accents well
- ONNX format
- GPU: ONNX Runtime providers

### Model Storage

**Location:** `%APPDATA%/com.pais.handy/models/`

```
models/
├── ggml-small.bin           # Whisper Small
├── whisper-medium-q4_1.bin  # Whisper Medium (quantized)
├── ggml-large-v3-turbo.bin  # Whisper Turbo
├── parakeet-tdt-0.6b-v2-int8/  # Parakeet V2
│   ├── encoder.onnx
│   ├── decoder.onnx
│   └── tokenizer.json
└── moonshine-base/          # Moonshine Base
    ├── encoder_model.onnx
    ├── decoder_model_merged.onnx
    └── tokenizer.json
```

### GPU Acceleration

#### Vulkan (AMD/Intel/NVIDIA)
```bash
# Environment variables
VULKAN_SDK=C:\VulkanSDK\1.4.335.0
GGML_BACKEND=vulkan
```

#### Metal (macOS)
Automatically enabled on Apple Silicon.

#### CUDA (NVIDIA)
Requires CUDA toolkit, not enabled by default.

---

## Audio Pipeline

### Recording Flow

```
1. User presses shortcut (Ctrl+Win)
      ↓
2. AudioManager.start_recording()
      ↓
3. cpal captures from microphone (48kHz stereo)
      ↓
4. Resample to 16kHz mono
      ↓
5. VAD detects speech segments
      ↓
6. User releases shortcut
      ↓
7. AudioManager.stop_recording() → samples
      ↓
8. Audio quality validation
      ↓
9. TranscriptionManager.transcribe(samples)
      ↓
10. Post-processing (hallucination filter, custom words)
      ↓
11. Clipboard.copy() and paste to active window
```

### Audio Constants

```rust
pub const SAMPLE_RATE: u32 = 16000;      // Model input rate
pub const CHANNELS: u16 = 1;              // Mono
pub const MIN_AUDIO_ENERGY: f32 = 0.001;  // RMS threshold
pub const MIN_SPEECH_SAMPLES: usize = 4000; // 0.25s minimum
```

---

## Build System

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Rust | 1.75+ | Backend compilation |
| Bun | 1.0+ | Package manager, dev server |
| LLVM | 18+ | whisper-rs-sys compilation |
| Vulkan SDK | 1.3+ | AMD/Intel GPU support |
| CMake | 3.20+ | Native dependency builds |

### Commands

```bash
# Install dependencies
bun install

# Development mode
bun tauri dev

# Production build
bun tauri build

# Linting
bun run lint        # ESLint
bun run lint:fix    # ESLint with auto-fix
bun run format      # Prettier + cargo fmt
```

### Environment Variables

```bash
# Required for AMD GPU
VULKAN_SDK=C:\VulkanSDK\1.4.335.0
LIBCLANG_PATH=C:\Program Files\LLVM\bin
GGML_BACKEND=vulkan

# Optional
WHISPER_NO_AVX=ON   # Disable AVX (Linux)
WHISPER_NO_AVX2=ON  # Disable AVX2 (Linux)
```

---

## Configuration

### Cargo.toml Dependencies

```toml
[dependencies]
tauri = "2.9.1"
transcribe-rs = { version = "0.2.2", features = ["whisper", "parakeet", "moonshine"] }
cpal = "0.16.0"           # Audio capture
rubato = "0.16.2"         # Resampling
vad-rs = { git = "..." }  # Voice activity detection
rdev = { git = "..." }    # Global keyboard hooks
enigo = "0.6.1"           # Input simulation
rusqlite = "0.37"         # SQLite for history
```

### Tauri Configuration

**File:** `src-tauri/tauri.conf.json`

Key settings:
- `identifier`: `com.pais.handy`
- `windows.decorations`: false (frameless)
- `bundle.icon`: Custom app icons
- `plugins`: store, clipboard, shortcuts, autostart

### Package.json Scripts

```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "tauri": "tauri",
    "lint": "eslint src/",
    "format": "prettier --write . && cargo fmt"
  }
}
```

---

## Platform Notes

### Windows
- Vulkan acceleration for AMD GPUs
- Global shortcuts via `handy-keys` crate
- Paste via `Ctrl+V` or direct typing

### macOS
- Metal acceleration (automatic)
- Accessibility permissions required
- NSPanel for overlay

### Linux
- Vulkan + OpenBLAS acceleration
- Limited Wayland support
- Overlay disabled by default
- Requires: `libasound2-dev`, `libvulkan-dev`

---

## Debug Mode

Access: `Ctrl+Shift+D` (Windows/Linux) or `Cmd+Shift+D` (macOS)

Shows:
- Audio levels and VAD state
- Model loading status
- Transcription timing
- GPU backend info

---

## File Locations

| File | Location |
|------|----------|
| Models | `%APPDATA%/com.pais.handy/models/` |
| History DB | `%APPDATA%/com.pais.handy/history.db` |
| Settings | `%APPDATA%/com.pais.handy/settings.json` |
| Logs | `%APPDATA%/com.pais.handy/logs/` |
| VAD Model | `resources/models/silero_vad_v4.onnx` |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Quick start:**
1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes with tests
4. Run linting: `bun run lint && cargo clippy`
5. Submit pull request

---

*Generated for AMD 9070XT Vulkan configuration*
*Last updated: February 2026*
