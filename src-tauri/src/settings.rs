// Everything the app remembers between runs, in one file next to the models.
// These used to live in the hidden window's browser storage, which only exists
// while that window does.
//
// The file is still called tray-prefs.json: renaming it would throw away the
// dictation key and the debug switch that are already saved in it.

use crate::AudioState;
use std::fs;
use std::path::PathBuf;
use tauri::{AppHandle, Emitter, Manager, State};

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct Prefs {
    /// Live microphone numbers and the per-dictation line. Off unless asked
    /// for; serde default keeps older settings files readable.
    #[serde(default)]
    pub(crate) debug_stats: bool,
    /// The key that starts and stops dictation, written the way Tauri parses
    /// it: "F3", "CommandOrControl+Shift+D".
    #[serde(default = "default_shortcut")]
    pub(crate) shortcut: String,
    #[serde(default)]
    pub(crate) active_local_model_id: Option<String>,
    /// Which microphone to record from, by the name the system gives it.
    /// None means whichever one the system has set as default.
    #[serde(default)]
    pub(crate) selected_microphone: Option<String>,
    /// Shorten long pauses in the middle of a recording before the model reads
    /// it. Off by default: it changes what the model hears, and the time it
    /// saves is under a second.
    #[serde(default)]
    pub(crate) pause_shortening: bool,
    /// How long a pause has to be before any of it is removed.
    #[serde(default = "default_pause_cutoff_ms")]
    pub(crate) pause_cutoff_ms: u32,
    /// Never shorten a pause in the first seconds after the first spoken word.
    /// On by default; the opening is what Whisper reads the language and the
    /// writing style from.
    #[serde(default = "default_true")]
    pub(crate) pause_protect_opening: bool,
    /// How much of the opening that covers. Remembered while the switch above
    /// is off, so turning it back on does not lose the number.
    #[serde(default = "default_pause_opening_ms")]
    pub(crate) pause_opening_ms: u32,
    /// Set once the settings held in the browser have been copied into here, so
    /// the copy happens exactly once and never overwrites a later change.
    #[serde(default)]
    pub(crate) migrated_from_browser: bool,
}

pub(crate) fn default_shortcut() -> String {
    "F3".to_string()
}

// 2.2 seconds. Shorter than this and it starts editing the breaths between
// sentences, which is where Whisper gets its full stops from.
pub(crate) fn default_pause_cutoff_ms() -> u32 {
    2200
}

// 3 seconds: long enough to cover the first sentence, which is what Whisper
// settles the language and the writing style from.
pub(crate) fn default_pause_opening_ms() -> u32 {
    3000
}

fn default_true() -> bool {
    true
}

impl Default for Prefs {
    fn default() -> Self {
        Self {
            debug_stats: false,
            shortcut: default_shortcut(),
            active_local_model_id: None,
            selected_microphone: None,
            pause_shortening: false,
            pause_cutoff_ms: default_pause_cutoff_ms(),
            pause_protect_opening: true,
            pause_opening_ms: default_pause_opening_ms(),
            migrated_from_browser: false,
        }
    }
}

pub(crate) fn prefs_path() -> Option<PathBuf> {
    dirs::data_local_dir().map(|d| d.join("omegawhisper").join("tray-prefs.json"))
}

pub(crate) fn load_prefs() -> Prefs {
    let Some(path) = prefs_path() else {
        return Prefs::default();
    };
    match fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str(&text).unwrap_or_else(|e| {
            eprintln!("Ignoring unreadable {}: {}", path.display(), e);
            Prefs::default()
        }),
        // Missing file just means nothing has been chosen yet.
        Err(_) => Prefs::default(),
    }
}

pub(crate) fn save_prefs(prefs: &Prefs) {
    let Some(path) = prefs_path() else {
        return;
    };
    if let Some(parent) = path.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            eprintln!("Could not create {}: {}", parent.display(), e);
            return;
        }
    }
    match serde_json::to_string_pretty(prefs) {
        Ok(text) => {
            if let Err(e) = fs::write(&path, text) {
                eprintln!("Could not save {}: {}", path.display(), e);
            }
        }
        Err(e) => eprintln!("Could not encode settings: {}", e),
    }
}

// Everything the settings window shows. Read from here, not from the browser's
// own storage, so there is one answer to what a setting is.
#[tauri::command]
pub(crate) fn get_settings(state: State<'_, AudioState>) -> Prefs {
    state.prefs()
}

// What is left of the settings the hidden window kept in browser storage.
// Optional: a missing value means that window never saved it.
#[derive(serde::Deserialize)]
pub(crate) struct BrowserSettings {
    pub(crate) active_local_model_id: Option<String>,
}

// Copy the settings out of the window that used to hold them, once. Rust
// cannot read browser storage itself, so a window has to hand it over.
//
// Runs only while `migrated_from_browser` is false, and only fills in settings
// the browser actually had, so it can never wipe a later change made here.
pub(crate) fn apply_browser_settings(prefs: &mut Prefs, from: BrowserSettings) -> bool {
    if prefs.migrated_from_browser {
        return false;
    }
    prefs.migrated_from_browser = true;

    if let Some(id) = from.active_local_model_id.filter(|s| !s.is_empty()) {
        prefs.active_local_model_id = Some(id);
    }
    true
}

// Async so writing the settings file cannot freeze the screen: a plain command
// runs on the thread that draws it.
#[tauri::command]
pub(crate) async fn migrate_browser_settings(
    state: State<'_, AudioState>,
    values: BrowserSettings,
) -> Result<bool, String> {
    let mut migrated = false;
    state.update_prefs(|p| migrated = apply_browser_settings(p, values));
    if migrated {
        eprintln!("Settings copied out of the browser and saved to disk.");
    }
    Ok(migrated)
}

// The one place the debug line is switched, so the tray tick, the settings
// switch, the saved file and both windows can never disagree.
pub(crate) fn set_debug_stats_everywhere(app: &AppHandle, enabled: bool) {
    let state = app.state::<AudioState>();
    state.update_prefs(|p| p.debug_stats = enabled);
    if let Some(item) = state.debug_menu_item.lock().unwrap().as_ref() {
        let _ = item.set_checked(enabled);
    }
    let _ = app.emit("debug-stats-changed", enabled);
}

#[tauri::command]
pub(crate) fn set_debug_stats(app: AppHandle, enabled: bool) {
    set_debug_stats_everywhere(&app, enabled);
}

// Asked by each window when it opens; after that the tray sends
// "debug-stats-changed" when it is switched.
#[tauri::command]
pub(crate) fn get_debug_stats(state: State<'_, AudioState>) -> bool {
    state.prefs().debug_stats
}

// The three pause-shortening settings. Async so writing the settings file
// cannot freeze the window.
#[tauri::command]
pub(crate) async fn set_pause_shortening(
    state: State<'_, AudioState>,
    enabled: bool,
) -> Result<(), String> {
    state.update_prefs(|p| p.pause_shortening = enabled);
    Ok(())
}

// The window already limits what can be typed; this is the same limit again,
// because a settings file edited by hand reaches here too. Below 500 ms there
// is nothing left to remove once the 300 ms gap is kept.
#[tauri::command]
pub(crate) async fn set_pause_cutoff_ms(
    state: State<'_, AudioState>,
    milliseconds: u32,
) -> Result<u32, String> {
    let milliseconds = milliseconds.clamp(500, 30_000);
    state.update_prefs(|p| p.pause_cutoff_ms = milliseconds);
    Ok(milliseconds)
}

#[tauri::command]
pub(crate) async fn set_pause_protect_opening(
    state: State<'_, AudioState>,
    enabled: bool,
) -> Result<(), String> {
    state.update_prefs(|p| p.pause_protect_opening = enabled);
    Ok(())
}

#[tauri::command]
pub(crate) async fn set_pause_opening_ms(
    state: State<'_, AudioState>,
    milliseconds: u32,
) -> Result<u32, String> {
    let milliseconds = milliseconds.min(30_000);
    state.update_prefs(|p| p.pause_opening_ms = milliseconds);
    Ok(milliseconds)
}
