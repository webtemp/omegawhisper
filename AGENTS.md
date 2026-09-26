# AGENTS.md/CLAUDE.md

## Critical Rules
1. **UI**: shadcn/ui (the React version) with Tailwind CSS 4, generated into `src/components/ui/`. This is a React project — never shadcn-vue.

## What this is
**Omegawhisper** — desktop speech-to-text, Tauri v2 + React 19, for macOS (Apple Silicon) and Linux. The Linux side is developed and tested on Arch with KDE Plasma 6 on Wayland and an AMD GPU.

**Local only.** Models (Whisper / Parakeet / Moonshine via `transcribe-rs`) run on this machine, buffer the whole recording, and transcribe once you stop. Nothing is sent anywhere. The hosted server and Deepgram were removed in 0.2.1.

**No main window.** The app is a menu-bar/tray agent: Rust does the work, and the only windows are `indicator` (the spectrogram, shown while dictating) and `settings` (opened from the tray).

## Commands
```bash
bun run tauri dev    # dev server + app
bun run tauri build  # production build
bun run dev          # frontend only
```
Vite port **1420**, strict. `nix-shell` or `flake.nix` gives a Nix dev env on Linux.

## One flow, all in Rust
```
F3 anywhere (macOS and X11: global key grab; Wayland: GlobalShortcuts portal;
             fallback: D-Bus / `omegawhisper transcribe toggle`)
                      |
      toggle_recording() -> start_recording_internal()
                      |
    audio capture thread (cpal) + transcription thread
                      |
    16 kHz -> buffer, transcribe once the recording stops,
    then type the text into whatever app has focus
                      |
    events to the indicator window; WAV saved to
    ~/Library/Application Support/omegawhisper/recordings/
    (Linux: ~/.local/share/omegawhisper/recordings/)
```

## Where things live
`src/`
- `main.tsx` — routes on `window.location.pathname`: `/settings`, `/indicator`
- `components/indicator.tsx` — spectrogram, live numbers, errors and startup warnings; drawn from the `mic-level` events Rust sends. It must never open the microphone itself
- `components/settings-page.tsx` — dictation key, microphone, models
- `lib/browser-settings.ts` — hands the settings the old main window kept in browser storage to Rust, once

`src-tauri/src/`
- `lib.rs` — `AudioState`, the event payloads, `run()` and the command list
- `main.rs` — `run()`, or the `transcribe toggle` CLI subcommand
- `settings.rs` — `Prefs`, the settings file, the one-time copy out of browser storage
- `recording.rs` — start/stop/toggle, the capture stream, the transcription thread
- `analysis.rs` — loudness, frequency bands, pitch, trimming, WAV bytes
- `microphone.rs` — listing input devices and picking one
- `indicator.rs` — showing, placing and hiding the indicator window
- `chime.rs` — the two sounds, worked out sample by sample through cpal
- `live.rs` — sentence splitter and the thread that types each sentence; the silence stop
- `history.rs` — the last 20 dictations, and the tray list that copies one on click
- `tray.rs` — the menu-bar icon and its frames; `shortcut.rs` — the dictation key
- `typing.rs` — putting text into other apps; `storage.rs` — the data folder and the log
- `vad.rs` — is there speech in the recording (Silero VAD)
- `linux.rs` — Linux only: the desktop file, the portal dictation key, the layer-shell indicator
- `models.rs` — the model list and download/delete; `tests.rs` — every test
- `managers/model.rs` — `AVAILABLE_MODELS`, download, delete, disk status
- `managers/transcription.rs` — loads a model, runs `transcribe-rs`
- `resampler.rs` — resample the microphone's rate down to 16 kHz

`src-tauri/icons/` — all committed, none generated. `icon.icns`/`icon.ico` and the sized PNGs are the app icon; `tray/` holds the menu-bar frames (`key-up`/`mid`/`down`, plus an unused `switch-*` set) as SVG source beside the 36x36 PNG that is compiled in. `TRAY_ICON` in `tray.rs` picks the set; `watch_tray_icon` plays the frames off the recording flag.

Read these in the code, not a copy here: `AudioState` at the top of `lib.rs`, `Prefs` in `settings.rs`, and the commands in `generate_handler!`.

## Notes
- Rust owns every setting. They live in `tray-prefs.json` next to the models and are saved the moment they change. The windows read them with `get_settings` and never keep their own copy — `localStorage` is not used for settings at all.
- One setting is not in `tray-prefs.json`: Start at login. The system holds it — on macOS the file `~/Library/LaunchAgents/Omegawhisper.plist`, written by `tauri-plugin-autostart` — and the user can delete it in System Settings, so a copy here could disagree. `get_start_at_login` asks the system every time. That file holds the app's full path and nothing checks it still leads anywhere, so `refresh_start_at_login` writes it again at startup; release builds only, or a `tauri dev` run would point login at the binary in `target/`.
- Two threads: capture (cpal; stereo to mono by averaging; F32/I16/U16) and transcription. Local transcription runs *after* the stop, which is why the app looks frozen for a moment. Linux adds a D-Bus thread.
- Two runtimes, two GPUs, two settings. Whisper runs on whisper.cpp and Metal; Parakeet and Moonshine run on ONNX Runtime and CoreML. They share nothing, and they want opposite answers. Measured on an M2 Pro over a minute of speech — Whisper Turbo 3.29 s on the GPU against 12.21 s off it, Parakeet 6.76 s on against 1.87 s off, Moonshine 2.59 s on against 1.66 s off. So `whisper_gpu` defaults on and `onnx_gpu` defaults off. Both features stay compiled in; the choice is made at runtime. `both_gpu_switches_change_where_the_model_runs` in `tests.rs` produced those numbers and re-measures them on any Mac, over every model that is downloaded.
- The GPU is chosen while a model is being built and cannot be changed afterwards, so `load_model` takes a `GpuChoice` and rebuilds when it differs from the one the loaded model was built with. Without that, moving a switch would do nothing until the next restart.
- Whisper needs `use_gpu` named explicitly in `WhisperLoadParams`. `WhisperLoadParams::default()` hardcodes it to true, and only `WhisperEngine::load` reads the library's global setting — which this app does not call, because it needs `flash_attn: false`.
- Typing into other apps uses `core-graphics` Unicode key events on macOS. On Linux `typing_tools` lists what is installed, best first (`wtype`, `ydotool`, `xdotool` on Wayland; `xdotool` first on X11), and each is tried until one succeeds. KWin does not give `wtype` the virtual keyboard protocol, so on KDE it fails and `ydotool` does the work. `ydotool` types from a fixed US keymap at 2 ms per key, so everything it is given is put on the clipboard and pasted with Ctrl+V, then the clipboard is put back. The clipboard is `arboard` with Wayland support, not the clipboard plugin, whose copy goes through XWayland. `ydotool key` takes one argument per key; the chord joined into one argument presses nothing and still exits 0, which is how a paste can "succeed" into nothing. Most terminals paste on Ctrl+Shift+V rather than Ctrl+V, so Cyrillic into a terminal depends on its bindings (this machine's kitty maps Ctrl+V). ASCII is typed at 2 ms per key whatever the length: a 1754-key burst at zero delay reached kitty intact but the program inside it read the burst as one lump and lost it.
- The dictation key on Linux depends on the session. Under Wayland an X11 key grab sees nothing, so `linux::watch_portal_shortcut` binds it through the `org.freedesktop.portal.GlobalShortcuts` portal (`ashpd`): the desktop shows a dialog once, then remembers the key under the app id and owns it from then on. Changing it is the desktop's job — `open_shortcut_settings` asks the portal to open that page, and `get_shortcut` returns the trigger the desktop reported. The portal needs an app id, which for an unsandboxed app means a `dev.omegawhisper.desktop` file: `write_desktop_file` keeps one in `~/.local/share/applications` pointing at the binary (release builds rewrite it every start, dev builds leave an existing one alone), and `register_host_app` claims the id over D-Bus. Under X11 the `tauri-plugin-global-shortcut` grab is used as on macOS. The D-Bus `toggle_recording` method and the `transcribe toggle` CLI stay as the fallback for compositors with no portal.
- Wayland does not let a window place itself or stay on top, so on Wayland the indicator is a layer-shell surface (`gtk-layer-shell`, overlay layer, anchored to the bottom, keyboard mode none). `float_indicator` must run before the window is realized, which is why it sits right after the indicator is built, while it is still hidden. `position_indicator` returns early on Wayland.
- The tray frames are black on clear for the macOS template icon. Linux panels draw the pixels as they are, so `tray_image` makes them white there.
- On Linux, Whisper runs on the GPU through Vulkan (`whisper-vulkan` feature, Linux target only). There is no GPU runtime for the ONNX models on Linux, so `onnx_gpu` changes nothing there; the settings page says so.
- `list_audio_devices` asks cpal, so the list is the same set of devices the recording can actually open. A chosen microphone that is unplugged falls back to the system default and says so in the log.
- Silence fed to Whisper makes it invent text. `vad::speech_seconds` is what stops that: Silero VAD (compiled in from `src-tauri/vad/`, written next to the models at first use) counts the seconds of speech in a copy scaled to a fixed peak, so the verdict does not depend on the microphone's gain, and under 0.3 s the recording never reaches the model. The old loudness gate `holds_speech` in `analysis.rs` is only the fallback if the detector cannot load; its fixed floor threw away every word from a headset at low gain. A peak under `DEAD_PEAK` is reported as a microphone delivering nothing. `what_the_speech_detector_makes_of_the_saved_recordings` in `tests.rs` (ignored) judges every saved recording on the machine, which is how the change was checked. `trim_quiet_edges` cuts the quiet start and end. Pauses over 2.2 s in the middle are shortened by default, except in the first 1.5 s after the first word. Energy-based voice detection was removed in 0.2.1 — it fed the model 3.2x the audio, chopped into fragments, and filtered nothing.
- Whisper keeps to the language of its initial prompt: the English style prompt turned Bulgarian speech into English. So `style_prompt` in `managers/transcription.rs` only returns a prompt for a language the user chose (English and Bulgarian have one), and the default "As spoken" setting sends no prompt and lets Whisper detect. `LANGUAGES` in `settings.rs` and `src/lib/languages.ts` list the same codes.
- History: `transcripts.json` next to the settings, newest first, 20 entries, saved before typing. Tray clicks copy through `typing::copy_to_clipboard`. Linux panels never redraw a changed submenu, so `tray::build_menu` rebuilds the whole menu and `set_menu` replaces it, on the main thread. A dictation started within 90 s of a silence stop is appended to the previous entry.
- Open Folder on Linux spawns `xdg-open` directly (`storage::open_folder`): the opener plugin double-forks it and on KDE nothing opens.
- Live typing (`live.rs`, on by default): Silero runs on the 16 kHz stream in `Segmenter` with hysteresis (start 0.5, hold 0.3); after 0.3 s of speech and a `live_pause_ms` pause (700 ms) the piece goes to a thread that transcribes and types it, in order. 0.5 s run-up and 0.3 s tail kept; frames scaled by the running peak (floor 0.03). Pause-shortening is skipped in this mode. Silence stop: twice the longest pause between words so far, floored at 2 s, capped at `silence_stop_ms`; before the first word the cap applies. Auto-resume (off by default, `auto_resume_ms` 6 s): after a silence stop with speech in it, `start_recording_internal(app, true)` listens again with no chime; speech continues the same history entry, nothing ends it quietly (`resume_pending` skips the done chime and the hide). `auto_enter` (off) presses Enter through `typing::press_enter` once the dictation is really over. `tidy_sentence_ends` (on) runs `live::tidy_end` on every piece: trailing dots, dangling fillers and the comma before them go. Both fall back to the plain recording if the detector cannot load.
- Sample rate comes from the input device, not hardcoded. 300 ms flush delay on `stop_recording`.
- `get_platform` tells the settings page what differs: which system, whether the desktop holds the key, and what each GPU switch turns on. The page changes its wording from that rather than guessing.
- Building on Linux without root: extract `webkit2gtk-4.1` (plus `enchant`, `libmanette`, `gtk-layer-shell`, `vulkan-headers`) into a prefix, point `PKG_CONFIG_PATH` at its `pkgconfig` dir with `prefix`/`libdir` rewritten, set `VULKAN_SDK` to the prefix, and run the result under `bwrap --overlay-src <prefix>/usr/lib --overlay-src /usr/lib --ro-overlay /usr/lib`, because WebKit spawns its helper processes from a compiled-in `/usr/lib/webkit2gtk-4.1` path.
