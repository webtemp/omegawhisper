<p align="center">
  <img src="logo.png" alt="Omegawhisper Logo" width="128" height="128">
</p>

<h1 align="center">Omegawhisper</h1>

<p align="center">
  Press one key anywhere, speak, and the text is typed into whatever app you are in.
</p>

---

## What it does

Omegawhisper sits in your menu bar or system tray. It has no Dock icon and no window in
your way. Press **F3**, speak, press **F3** again. A small spectrogram shows it is
listening, and the text is typed into the app you were already using. F3 is only the
default — pick any key.

Everything runs on your machine. Whisper, Parakeet and Moonshine models are downloaded
once and run locally, on the GPU where that helps. Nothing is sent anywhere, there is no
account and no server.

### Features

- One global shortcut, works in any app. **F3** by default
- macOS (Apple Silicon), Linux (Wayland or X11) and Windows 11
- Types into other apps: Unicode key events on macOS and Windows, `ydotool`/`wtype`/`xdotool` on Linux
- Local models on the GPU: Metal and CoreML on the Mac, Vulkan on Linux and Windows
- Spectrogram indicator window while you speak
- Recordings saved as WAV, and deletable from the tray menu
- Silence is never sent to the model, so it cannot invent text from a quiet room. Speech is recognised by Silero VAD, so a quiet microphone works as well as a loud one
- Dark theme

## Install (Linux)

Tested on Arch Linux with KDE Plasma 6 on Wayland and an AMD GPU. Other desktops work as
far as they offer the same freedesktop pieces, listed below.

### 1. Packages

```sh
sudo pacman -S --needed webkit2gtk-4.1 gtk-layer-shell libappindicator-gtk3 \
    ydotool wtype xdg-desktop-portal vulkan-headers shaderc cmake clang bun rustup
```

Or `./scripts/install-linux.sh`, which runs that, builds, installs into `~/.local/bin`,
and starts the `ydotool` service.

What each one is for:

| Package | Why |
|---|---|
| `webkit2gtk-4.1`, `libappindicator-gtk3` | The two small windows, and the tray icon |
| `gtk-layer-shell` | Puts the indicator at the bottom of the screen on Wayland. Without it Wayland decides where the window goes |
| `xdg-desktop-portal` plus your desktop's backend (`xdg-desktop-portal-kde`, `-gnome`, `-hyprland`…) | The dictation key on Wayland |
| `ydotool` | Types the text, on any compositor, through a virtual keyboard. Anything outside ASCII is pasted instead, since its keymap cannot type it |
| `wtype` | Tried first on Wayland: it types any character directly. KWin does not offer it the protocol it needs, so on KDE it is skipped |
| `vulkan-headers`, `shaderc` | Build Whisper for the GPU. Only at build time |
| `cmake`, `clang` | Build whisper.cpp |

On Debian or Ubuntu the names differ: `libwebkit2gtk-4.1-dev`, `libgtk-layer-shell-dev`,
`libayatana-appindicator3-dev`, `ydotool`, `wtype`, `libvulkan-dev`, `glslc`, plus `bun`
from [bun.sh](https://bun.sh) and Rust from [rustup.rs](https://rustup.rs).

### 2. Build and install

```sh
git clone https://github.com/webtemp/omegawhisper.git
cd omegawhisper
bun install
bun run tauri build --no-bundle
install -Dm755 src-tauri/target/release/omegawhisper ~/.local/bin/omegawhisper
```

The first build compiles whisper.cpp, its Vulkan shaders and ONNX Runtime and takes a
while. Later builds are much faster. `bun run tauri build` without `--no-bundle` also
produces `.deb`, `.rpm` and `.AppImage` files under `src-tauri/target/release/bundle/`.

### 3. Let it type

`ydotool` needs its service running and your user in the `input` group:

```sh
systemctl --user enable --now ydotool
sudo usermod -aG input "$USER"    # log out and in again afterwards
```

### 4. Run it and say yes to the key

```sh
omegawhisper
```

At startup the app writes `~/.local/share/applications/dev.omegawhisper.desktop`, so it
shows up in your launcher, and asks the desktop for **F3** through the GlobalShortcuts
portal. KDE and GNOME show a dialog once — *Omegawhisper wants to register the following
shortcut: Start or stop dictation, F3* — press **OK**. After that the desktop remembers
the key, under System Settings → Keyboard → Shortcuts → Omegawhisper, which is also where
you change it; the **Change...** button in Settings opens that page. On an X11 session
the app grabs the key itself and Settings → Dictation key → Change picks another.

Then open **Settings** from the tray icon, download **Whisper Turbo**, and press F3.

Start at login is the switch in Settings; it writes `~/.config/autostart/omegawhisper.desktop`.

### Without a portal

If your compositor has no GlobalShortcuts portal, bind this to a key in it:

```sh
omegawhisper transcribe toggle
```

It reaches the running app over D-Bus (`dev.omegawhisper` at `/dev/omegawhisper`, method
`toggle_recording`). Sway, i3 and everything else that can run a command on a key work
this way.

## Install (macOS)

There are no prebuilt macOS releases. You build it yourself. Tested on Apple Silicon.

### The short way

```sh
git clone https://github.com/webtemp/omegawhisper.git
cd omegawhisper
./scripts/install.sh
```

That does steps 1 to 4 below for you. It installs only what is missing, never replaces a
Rust or a Homebrew you already have, and stops twice to tell you what to click. You still
have to grant Accessibility yourself (step 5) — macOS does not let any script do that.

It also asks which offline model to download, defaulting to Whisper Turbo, and fetches it
in the background while everything else installs. Choose **None** to skip it and pick one
in Settings later. The model is usually the slowest part, so it waits for it at the end.

Or do it by hand, below. Six steps. Step 5 grants macOS permissions — the app cannot type
anything until you do it, so do not stop after the build.

### 1. Install the build tools

You need [Homebrew](https://brew.sh) first, since two of these come from it:

```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then:

```sh
xcode-select --install                                          # C/C++ compiler and linker
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh  # Rust (not Homebrew's rust)
brew install bun                                                # Bun
brew install cmake                                              # builds whisper.cpp
```

CMake is not optional. Without it the build fails while compiling `whisper-rs-sys`.

> [!IMPORTANT]
> **After installing Rust, open a new terminal.** Rustup does not add `cargo` to the
> terminal you are sitting in — it writes a line into your shell startup files, and only
> terminals opened afterwards read those. Carry on in the same window and the build fails
> with `failed to run 'cargo metadata'` and `No such file or directory (os error 2)`, which
> never mentions Rust. To fix the window you already have without opening a new one:
>
> ```sh
> source "$HOME/.cargo/env"
> ```

### 2. Get the code and the Tauri CLI

```sh
git clone https://github.com/webtemp/omegawhisper.git
cd omegawhisper
bun install
```

`bun install` is what installs the Tauri CLI — it is a devDependency, not something you
install globally. Check it worked before going on:

```sh
bun run tauri --version     # should print: tauri-cli 2.x.x
```

If that says "command not found", install it into the project by hand and try again:

```sh
bun add -D @tauri-apps/cli
```

### 3. Build

```sh
bun run tauri build --bundles app
```

The first build compiles whisper.cpp and ONNX Runtime from source and takes a while.
Later builds are much faster.

Result: `src-tauri/target/release/bundle/macos/Omegawhisper.app`

### 4. Copy it to Applications

```sh
cp -R src-tauri/target/release/bundle/macos/Omegawhisper.app /Applications/
open /Applications/Omegawhisper.app
```

**Replacing an existing install?** Quit the app first and use `rsync`, not `rm -rf`.
macOS App Management protection blocks deleting an `.app` folder in `/Applications` and
can leave it half-deleted:

```sh
rsync -a --delete src-tauri/target/release/bundle/macos/Omegawhisper.app/ /Applications/Omegawhisper.app/
```

### 5. Grant permissions — do not skip this

> [!IMPORTANT]
> **Without Accessibility, the app does nothing useful.** It will record you, transcribe
> you, and then fail to type a single character — with no error message. If you only do
> one thing from this whole page, do this.

**Accessibility** is what lets the app type into other apps. Turn it on:

```sh
open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
```

Then add `/Applications/Omegawhisper.app` to the list and switch the toggle on.

**Microphone** needs nothing from you now — macOS asks the first time you record. Say yes.

#### After every rebuild, do it again

These builds are unsigned, so macOS treats each new build as a different app and
**throws the Accessibility grant away**. The nasty part: the toggle still looks switched
on, so it appears fine while typing silently fails.

Every time you rebuild, run this and add the app back:

```sh
tccutil reset Accessibility dev.omegawhisper
open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
```

### 6. Pick a backend

Open **Settings** from the menu-bar icon.

To run offline, download a local model there. **Whisper Turbo** is the one to start with:
it is the best mix of speed and accuracy on the Mac GPU, and it handles any language.
Whisper Small is smaller and quicker if you are short of disk space; Whisper Large is more
accurate but noticeably slower. Parakeet and Moonshine are English only.

Models are 80 MB to 1.6 GB, so the first download takes a moment.

The microphone defaults to the system input; Settings → Microphone picks another.

## Install (Windows)

Windows 11, 64-bit. This is the one system with prebuilt releases: every
[GitHub release](https://github.com/webtemp/omegawhisper/releases) carries
`Omegawhisper_<version>_x64-setup.exe`.

1. Download the installer and run it. It installs for your user only, under
   `%LOCALAPPDATA%\Omegawhisper`, asks for no administrator password, adds a Start menu
   entry, and can be removed from Settings → Apps.
2. **SmartScreen.** The installer is not code-signed yet, so Windows says *Windows protected
   your PC*. Click **More info**, then **Run anyway**. The browser may also ask to keep the
   download.
3. WebView2 is part of Windows 11. If it is missing, the installer downloads it.
4. Start Omegawhisper from the Start menu. It sits in the system tray, under the `^` next
   to the clock. Open **Settings** from its icon, download **Whisper Turbo**, press **F3**.

Typing is done with Unicode key events (`SendInput`), so any language works whatever the
keyboard layout. Anything that refuses them is pasted with Ctrl+V instead. A program running
as administrator accepts no keys from one that is not: for those, paste by hand - the text is
on the clipboard whenever typing fails.

Whisper runs on the graphics card through Vulkan, with any current AMD, NVIDIA or Intel
driver. Parakeet and Moonshine run on the processor; the second switch changes nothing on
Windows.

Start at login writes the `Run` key under `HKEY_CURRENT_USER` in the registry.

At startup the app asks GitHub Releases whether there is a newer version and, if there is,
asks before installing it. That request is the only thing the app ever sends anywhere.

### For testers

- **The installer** is the `-setup.exe` on the
  [releases page](https://github.com/webtemp/omegawhisper/releases).
- **SmartScreen:** *More info* → *Run anyway*.
- **The log** is `%LOCALAPPDATA%\omegawhisper\omegawhisper.log`. Paste that into the
  Explorer address bar. It starts over when it reaches 5 MB, so copy it soon after a failure.
- **Recordings** are next to it, in `%LOCALAPPDATA%\omegawhisper\recordings`.
- **"F3 could not be registered"** on the indicator at startup means another program holds
  F3. Pick a different key in Settings, or close that program. Starting Omegawhisper a
  second time no longer does this: it opens Settings in the running copy instead.
- **When something fails,** send the log, what you pressed and said, the app you were typing
  into, your Windows version (`winver`) and graphics card. Switch on **Show debug stats** in
  the tray menu first: it puts a line of numbers for every dictation into the log.

## Using it

Press **F3** to start, speak, press **F3** to stop. That is the whole app.

To use a different key, open **Settings** → **Dictation key** → **Change**, then press the
combination you want. If it is already taken by another app it says so and keeps the old
one, so you can never end up with no shortcut.

### Starting it automatically

**Settings** → **Startup** → **Start when the computer starts**. After logging in the
menu-bar icon is there and the dictation key works; no window opens.

This writes `~/Library/LaunchAgents/Omegawhisper.plist`, which holds the full path to the
app. Move the app to another folder and that path is wrong, so the app writes the file
again at every startup, pointing at wherever it is being run from. Switching it off
deletes the file. macOS also lists it under System Settings → General → Login Items.
### Which models use the graphics card

**Settings** → **Graphics card**. Two switches, because the two kinds of model
answer differently.

One minute of speech, on an M2 Pro:

| Model | Setting | Graphics card | Processor |
|---|---|---|---|
| Whisper Turbo | **on** | **3.3 s** | 12.2 s |
| Parakeet v3 | **off** | 6.8 s | **1.9 s** |
| Moonshine Base | **off** | 2.6 s | **1.7 s** |

Whisper runs on whisper.cpp through Metal on the Mac and Vulkan on Linux. Parakeet and
Moonshine run on ONNX Runtime through CoreML on the Mac; the Linux build has no GPU
runtime for them, so there the second switch changes nothing. They are separate, so one
switch cannot serve both.

On Linux, an RX 9070 XT transcribed 13 seconds of speech with Whisper Turbo in 0.6 s.

The reason the second one loses: those models are quantised to 8-bit integers,
which CoreML handles poorly — it hands parts back to the processor and pays for
the crossing each time. Loading is slower too: Parakeet takes 5.7 seconds to
load on the graphics card against 0.6 on the processor.

These numbers are from one Mac, an M2 Pro. On yours, run:

```sh
cargo test --manifest-path src-tauri/Cargo.toml --release -- --ignored --nocapture both_gpu_switches
```

It times every model you have downloaded, both ways, and prints which won for
each. Either switch takes effect on the next dictation.

The menu-bar or tray icon has:

| Item | What it does |
|---|---|
| Language | The language spoken, "As spoken" lets Whisper detect it. Also in Settings |
| Recordings → Open Folder | `~/Library/Application Support/omegawhisper/recordings` on the Mac, `~/.local/share/omegawhisper/recordings` on Linux, `%LOCALAPPDATA%\omegawhisper\recordings` on Windows |
| Recordings → Delete Recordings | Deletes every saved WAV. Asks first |
| Show debug stats | Live microphone numbers, and a line of numbers under each result. Also in Settings |
| Settings | Dictation key, microphone, models, graphics card, startup |
| Quit | Quits |

## Troubleshooting

**Nothing happens when I press the key.** On the Mac, another app has taken it or
Accessibility is off. On Linux under Wayland, the desktop was told no in the dialog, or the
key is taken: System Settings → Keyboard → Shortcuts → Omegawhisper shows what it holds. On
Windows, another program holds the key; pick a different one in Settings.
Startup problems appear on the indicator as a message when the app starts.

**Text is transcribed but never typed.** On the Mac: Accessibility. If you rebuilt the app,
the grant is gone even though the checkbox still looks on: reset it (step 5). On Linux:
`ydotool` is missing or its service is not running. The text is put on the clipboard
instead, and the log says which tool failed and why. On Windows: the program you were typing
into runs as administrator, which blocks keys from ordinary programs; the text is on the
clipboard.

**The build says `failed to run 'cargo metadata'` / `No such file or directory (os error 2)`.**
That means the build cannot find `cargo`. Two different causes:

```sh
which cargo || ls ~/.cargo/bin/cargo
```

Nothing found at all — Rust is not installed, do step 1. Found in `~/.cargo/bin` but `which`
says nothing — it is installed and your terminal is just too old to see it, so open a new
one or run `source "$HOME/.cargo/env"`.

**"These seconds held no speech" while I was talking.** Speech is recognised by a small
model (Silero VAD) that listens to the shape of the sound, not its loudness, so a quiet
microphone is fine: a headset delivering a fiftieth of normal level is still heard. If
you get this anyway, the microphone is not reaching the app at all, or almost: check it
is the one chosen under Settings → Microphone, and that nothing else is holding it. The
Boost slider there multiplies the signal by 0.5 to 100 for a microphone that is merely
quiet or too hot; the app already scales speech to a normal level before Whisper hears it.

**I spoke Bulgarian and got English.** Settings → Language. "As spoken" lets Whisper detect the
language; choosing one tells Whisper outright and gets better punctuation in it.

**The app freezes for a second after I stop.** Expected with local models. They transcribe
after the recording ends, not during.

**Whisper writes text I never said.** Recordings with no speech in them are refused before
they reach the model, so this should not happen. If it does, the log line for that
dictation shows the loudness it measured.

**Anything else.** The log is at
`~/Library/Application Support/omegawhisper/omegawhisper.log` on the Mac,
`~/.local/share/omegawhisper/omegawhisper.log` on Linux and
`%LOCALAPPDATA%\omegawhisper\omegawhisper.log` on Windows. Switch on **Show debug
stats**, in the menu bar or in Settings, to get live microphone numbers and a line of
numbers per dictation.

## Development

```sh
bun install
bun run tauri dev # dev server + app
bun run test      # Rust tests + frontend tests
bun run test:rust # Rust only
bun run test:web  # frontend only
bun run dev       # frontend only, port 1420
```

```
src/                       React 19 frontend (two windows, no main window)
  components/indicator.tsx spectrogram, errors, startup warnings
  components/settings-page.tsx  dictation key, microphone, models
src-tauri/src/
  lib.rs                   app state, events, run(), the command list
  recording.rs             start/stop, capture stream, transcription thread
  analysis.rs              loudness, frequency bands, pitch, trimming, WAV
  settings.rs              the settings file; microphone.rs  input devices
  indicator.rs  chime.rs  tray.rs  shortcut.rs  typing.rs  storage.rs
  linux.rs                 desktop file, portal dictation key, layer-shell indicator
  update.rs                the startup check for a newer release, Windows for now
  managers/model.rs        model list, download, delete
  managers/transcription.rs  loads a model, runs transcribe-rs
src-tauri/icons/           app icon and menu-bar frames, all committed
```

**Tech stack:** React 19, TypeScript, Tailwind CSS 4, shadcn/ui, Rust, Tauri v2, cpal.

`flake.nix` and `shell.nix` predate the Linux work and have not been run since; the
packages listed under Install (Linux) are the tested path.

### Windows builds and releases

There is no Windows machine here; `.github/workflows/windows.yml` does the building. Every
push runs `cargo check`, clippy and the tests on a Windows runner. A tag `v*` builds the
NSIS installer, signs it for the updater and publishes a GitHub release with `latest.json`,
which running copies of the app read at startup.

The updater key pair was made with `bun run tauri signer generate`. The public half is in
`tauri.conf.json`; the private half is not in the repository and must never be. It lives
in `~/.tauri/omegawhisper.key`. Put it in the repository secret `TAURI_SIGNING_PRIVATE_KEY`
when the app should update itself: releases built without the secret are plain installers
that running copies never offer. Lose the key and no installed copy can ever be updated.

To cut a release:

```sh
# 1. same version in package.json, src-tauri/Cargo.toml, src-tauri/tauri.conf.json
# 2. commit, then
git tag v0.6.0
git push origin main v0.6.0
```

The workflow publishes the release a while later; the first build compiles whisper.cpp and
takes the longest. The tag must be `v` followed by the version in `tauri.conf.json`, or
the updater will announce one version and install another.

## License

[GPLv3](./LICENSE)

- Copyright (C) 2026 Ameya Shenoy &lt;shenoy.ameya@gmail.com&gt;
- Copyright (C) 2026 Deyan Danailov &lt;webtemp@gmail.com&gt;

Started in 2026 as a fork of Ameya Shenoy's hyperwhisper and rewritten since: local-only,
no main window, its own recording pipeline, and macOS and Linux desktop integration.
