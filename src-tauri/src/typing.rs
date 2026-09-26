// Putting the finished text into whatever app has focus.

use std::thread;
use std::time::Duration;
use tauri::AppHandle;

// How many UTF-16 units to put in one key event. A Unicode key event carries
// only a short string, so long text is sent in several events.
#[cfg(target_os = "macos")]
pub(crate) const CHUNK_UTF16_UNITS: usize = 20;

// Whether this app is allowed to control the computer (Accessibility).
// Without it the key events are created but the system drops them, so the
// text silently never appears.
#[cfg(target_os = "macos")]
pub(crate) fn accessibility_granted() -> bool {
    #[link(name = "ApplicationServices", kind = "framework")]
    extern "C" {
        fn AXIsProcessTrusted() -> bool;
    }
    unsafe { AXIsProcessTrusted() }
}

// After the dictation key: the key has to come up and the target window take
// focus back first, or the first characters land nowhere.
pub(crate) fn type_text_internal(app: &AppHandle, text: &str) -> Result<(), String> {
    type_text(app, text, Duration::from_millis(120))
}

// Mid-recording, no key involved: straight away.
pub(crate) fn type_text_now(app: &AppHandle, text: &str) -> Result<(), String> {
    type_text(app, text, Duration::ZERO)
}

// Enter in the focused app, for sending what was just typed.
#[allow(clippy::needless_return)]
pub(crate) fn press_enter(app: &AppHandle) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    let _ = app;
    // Let the target take in the last text first.
    thread::sleep(Duration::from_millis(80));

    #[cfg(target_os = "macos")]
    {
        use core_graphics::event::{CGEvent, CGEventFlags, CGEventTapLocation};
        use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};
        let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState)
            .map_err(|_| "Failed to create a keyboard event source".to_string())?;
        const RETURN: u16 = 36;
        for key_down in [true, false] {
            let event = CGEvent::new_keyboard_event(source.clone(), RETURN, key_down)
                .map_err(|_| "Failed to create a keyboard event".to_string())?;
            event.set_flags(CGEventFlags::CGEventFlagNull);
            event.post(CGEventTapLocation::HID);
        }
        return Ok(());
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = app;
        let mut failures = Vec::new();
        for tool in typing_tools() {
            let result = match tool.name {
                "ydotool" => run("ydotool", &["key", "28:1"], "28:0"),
                "xdotool" => run("xdotool", &["key"], "Return"),
                _ => run("wtype", &["-k"], "Return"),
            };
            match result {
                Ok(()) => return Ok(()),
                Err(e) => failures.push(format!("{} {}", tool.name, e)),
            }
        }
        Err(format!("Nothing could press Enter: {}", failures.join("; ")))
    }
}

// The return separates the macOS path from the Linux one below it.
#[allow(clippy::needless_return)]
fn type_text(app: &AppHandle, text: &str, settle: Duration) -> Result<(), String> {
    if text.is_empty() {
        return Ok(());
    }
    thread::sleep(settle);
    #[cfg(target_os = "macos")]
    let _ = app;

    #[cfg(target_os = "macos")]
    {
        // Post the text straight to the window server as Unicode key events.
        //
        // This replaces `osascript ... keystroke`, which spawned a process per
        // transcription, needed the Automation permission on top of
        // Accessibility, and typed through the current keyboard layout - so
        // any character the layout cannot produce (Cyrillic on a US layout)
        // came out wrong. Unicode key events do not use the layout.
        use core_graphics::event::{CGEvent, CGEventFlags, CGEventTapLocation};
        use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};

        if !accessibility_granted() {
            return Err(
                "Accessibility permission is not granted, so text cannot be typed. \
                 Add Omegawhisper in System Settings > Privacy & Security > Accessibility."
                    .to_string(),
            );
        }

        let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState)
            .map_err(|_| "Failed to create a keyboard event source".to_string())?;

        // One event carries only a short Unicode string, so send the text in
        // small pieces. Split on character boundaries, never inside a
        // surrogate pair, or the character is corrupted.
        let utf16: Vec<u16> = text.encode_utf16().collect();
        let mut start = 0;
        while start < utf16.len() {
            let mut end = std::cmp::min(start + CHUNK_UTF16_UNITS, utf16.len());
            // A leading surrogate at the end means the pair is split - keep it
            // with its trailing half in the next chunk.
            if end < utf16.len() && (0xD800..0xDC00).contains(&utf16[end - 1]) {
                end -= 1;
            }
            let chunk = String::from_utf16_lossy(&utf16[start..end]);

            for key_down in [true, false] {
                let event = CGEvent::new_keyboard_event(source.clone(), 0, key_down)
                    .map_err(|_| "Failed to create a keyboard event".to_string())?;
                // Events built from the live hardware state inherit whatever
                // modifiers are held right now. F3 sets the Fn modifier, and a
                // character carrying Fn (or Command) is read as a shortcut and
                // thrown away instead of typed. Send plain characters only.
                event.set_flags(CGEventFlags::CGEventFlagNull);
                event.set_string(&chunk);
                event.post(CGEventTapLocation::HID);
            }

            // Electron apps (Teams, VS Code) drop characters without a pause.
            thread::sleep(Duration::from_millis(2));
            start = end;
        }

        return Ok(());
    }

    #[cfg(not(target_os = "macos"))]
    {
        let tools = typing_tools();
        if tools.is_empty() {
            return Err(NO_TOOL_MESSAGE.to_string());
        }

        let mut failures = Vec::new();
        for tool in tools {
            let result = if needs_paste(tool.name, text) {
                paste_with(app, tool, text)
            } else {
                run(tool.name, tool.args, text)
            };
            match result {
                Ok(()) => return Ok(()),
                Err(e) => {
                    eprintln!("{}: {} ({})", tool.name, e, tool.hint);
                    failures.push(format!("{} {}", tool.name, e));
                }
            }
        }
        Err(format!(
            "Nothing could type the text: {}",
            failures.join("; ")
        ))
    }
}

#[cfg(not(target_os = "macos"))]
pub(crate) struct TypingTool {
    pub(crate) name: &'static str,
    args: &'static [&'static str],
    hint: &'static str,
}

#[cfg(not(target_os = "macos"))]
pub(crate) const NO_TOOL_MESSAGE: &str = "Text cannot be typed into other apps: none of \
    wtype, ydotool or xdotool is installed. ydotool works on any desktop once its \
    service is running.";

// wtype types any character but needs a protocol KWin does not offer. ydotool
// works anywhere, 2 ms per key so a long text is not one burst a program
// can lose, but from a US keymap. xdotool is for X11.
#[cfg(not(target_os = "macos"))]
const TOOLS: [TypingTool; 3] = [
    TypingTool {
        name: "wtype",
        args: &["--"],
        hint: "the compositor may not offer the virtual keyboard protocol it needs",
    },
    TypingTool {
        name: "ydotool",
        args: &["type", "--key-delay=2", "--"],
        hint: "is its service running? systemctl --user enable --now ydotool",
    },
    TypingTool {
        name: "xdotool",
        args: &["type", "--clearmodifiers", "--"],
        hint: "it only types into X11 windows",
    },
];

// Best first for the session: xdotool cannot type into Wayland windows, wtype
// cannot type into X11 ones.
#[cfg(not(target_os = "macos"))]
pub(crate) fn tool_order(wayland: bool) -> [&'static str; 3] {
    if wayland {
        ["wtype", "ydotool", "xdotool"]
    } else {
        ["xdotool", "ydotool", "wtype"]
    }
}

// Linux key codes for Ctrl+V: 29 is left Ctrl, 47 is V. ydotool wants one
// argument per key; joined into one it presses nothing and reports success.
#[cfg(not(target_os = "macos"))]
pub(crate) const PASTE_CHORD: [&str; 4] = ["29:1", "47:1", "47:0", "29:0"];

// ydotool types one key every 2 ms from a keymap with no Cyrillic or accents,
// so everything it is given goes through the clipboard and lands at once.
#[cfg(not(target_os = "macos"))]
pub(crate) fn needs_paste(tool: &str, _text: &str) -> bool {
    tool == "ydotool"
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn typing_tools() -> Vec<&'static TypingTool> {
    let wayland = std::env::var_os("WAYLAND_DISPLAY").is_some_and(|v| !v.is_empty());
    tool_order(wayland)
        .iter()
        .filter_map(|name| TOOLS.iter().find(|t| t.name == *name))
        .filter(|tool| installed(tool.name))
        .collect()
}

#[cfg(not(target_os = "macos"))]
fn run(program: &str, args: &[&str], text: &str) -> Result<(), String> {
    let status = std::process::Command::new(program)
        .args(args)
        .arg(text)
        .status()
        .map_err(|e| format!("could not be started: {}", e))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("failed with {}", status))
    }
}

// One clipboard for the life of the app: on Linux the contents go with the
// handle that set them.
#[cfg(not(target_os = "macos"))]
static CLIPBOARD: std::sync::Mutex<Option<arboard::Clipboard>> = std::sync::Mutex::new(None);

#[cfg(not(target_os = "macos"))]
fn with_clipboard<T>(
    work: impl FnOnce(&mut arboard::Clipboard) -> Result<T, String>,
) -> Result<T, String> {
    let mut guard = CLIPBOARD.lock().unwrap();
    if guard.is_none() {
        *guard = Some(arboard::Clipboard::new().map_err(|e| format!("no clipboard: {}", e))?);
    }
    work(guard.as_mut().unwrap())
}

// Put text on the clipboard and leave it there.
pub(crate) fn copy_to_clipboard(app: &AppHandle, text: &str) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        use tauri_plugin_clipboard_manager::ClipboardExt;
        app.clipboard()
            .write_text(text)
            .map_err(|e| format!("could not use the clipboard: {}", e))
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = app;
        with_clipboard(|clipboard| {
            clipboard
                .set_text(text)
                .map_err(|e| format!("could not use the clipboard: {}", e))
        })
    }
}

// Clipboard, Ctrl+V, then the clipboard put back.
#[cfg(not(target_os = "macos"))]
fn paste_with(_app: &AppHandle, tool: &TypingTool, text: &str) -> Result<(), String> {
    with_clipboard(|clipboard| {
        let before = clipboard.get_text().ok();
        clipboard
            .set_text(text)
            .map_err(|e| format!("could not use the clipboard: {}", e))?;
        thread::sleep(Duration::from_millis(50));
        let (last, first) = PASTE_CHORD.split_last().unwrap();
        let mut args = vec!["key"];
        args.extend(first);
        let pressed = run(tool.name, &args, last);
        // The target reads the clipboard on the key press; give it that long.
        thread::sleep(Duration::from_millis(200));
        if let Some(previous) = before {
            let _ = clipboard.set_text(previous);
        }
        pressed
    })
}

#[cfg(not(target_os = "macos"))]
fn installed(program: &str) -> bool {
    std::env::var_os("PATH")
        .is_some_and(|paths| std::env::split_paths(&paths).any(|dir| dir.join(program).is_file()))
}
