#!/bin/bash
# One command from a clean Arch Linux desktop to a working Omegawhisper.
#
# Safe to run again. Packages already installed are left alone, and the binary
# in ~/.local/bin is replaced by the new build.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -t 1 ]; then
    BOLD=$(tput bold 2>/dev/null || true); BLUE=$(tput setaf 4 2>/dev/null || true)
    GREEN=$(tput setaf 2 2>/dev/null || true); YELLOW=$(tput setaf 3 2>/dev/null || true)
    RESET=$(tput sgr0 2>/dev/null || true)
else
    BOLD=""; BLUE=""; GREEN=""; YELLOW=""; RESET=""
fi
step() { printf "\n%s==> %s%s\n" "$BOLD$BLUE" "$1" "$RESET"; }
ok()   { printf "  %s+%s %s\n" "$GREEN" "$RESET" "$1"; }
warn() { printf "  %s!%s %s\n" "$YELLOW" "$RESET" "$1"; }

if ! command -v pacman >/dev/null 2>&1; then
    warn "This script knows pacman only. On another distribution, install the"
    warn "packages listed in README.md under Install (Linux) and run the build steps below."
    exit 1
fi

step "1/5  Packages"
sudo pacman -S --needed --noconfirm webkit2gtk-4.1 gtk-layer-shell libappindicator-gtk3 \
    ydotool wtype xdg-desktop-portal vulkan-headers shaderc cmake clang bun rustup
if ! command -v cargo >/dev/null 2>&1; then
    rustup default stable
fi
ok "installed"

step "2/5  Typing service"
# ydotool types through /dev/uinput, which belongs to the input group.
systemctl --user enable --now ydotool.service
if id -nG "$USER" | grep -qw input; then
    ok "ydotool running, and $USER is in the input group"
else
    sudo usermod -aG input "$USER"
    warn "Added $USER to the input group. Log out and in again before the first dictation."
fi

step "3/5  Project dependencies"
bun install
ok "installed, including the Tauri command line tool"

step "4/5  Building the app"
echo "    The first build compiles whisper.cpp and its Vulkan shaders. Later builds are quick."
bun run tauri build --no-bundle
ok "built"

step "5/5  Installing into ~/.local/bin"
install -Dm755 src-tauri/target/release/omegawhisper "$HOME/.local/bin/omegawhisper"
ok "installed at ~/.local/bin/omegawhisper"
case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) warn "~/.local/bin is not on your PATH. Start it with ~/.local/bin/omegawhisper." ;;
esac

printf "\n%sDone. Start it with%s omegawhisper\n" "$BOLD" "$RESET"
printf "
The desktop will ask once whether Omegawhisper may have %sF3%s. Say OK.
Then open Settings from the tray icon, download Whisper Turbo, press F3, speak, press F3.
" "$BOLD" "$RESET"
