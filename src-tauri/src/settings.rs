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
    /// Multiply the microphone's signal by this before anything looks at it,
    /// 0.5 to 100. 1 leaves it alone. For a microphone so quiet that the speech
    /// check throws the recording away, or one so hot that it clips.
    #[serde(default = "default_mic_boost")]
    pub(crate) mic_boost: f32,
    /// How the indicator draws the sound. One of `VISUALISATIONS`.
    #[serde(default = "default_visualisation")]
    pub(crate) visualisation: String,
    /// The language spoken, as a Whisper code, or "auto" to let it decide.
    #[serde(default = "default_language")]
    pub(crate) language: String,
    /// Shorten long pauses in the middle of a recording before the model reads
    /// it. On by default, with a cutoff long enough to leave the breaths
    /// between sentences alone.
    #[serde(default = "default_true")]
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
    /// Type each sentence the moment the pause after it is heard, instead of
    /// the whole text once the recording stops.
    #[serde(default = "default_true")]
    pub(crate) live_typing: bool,
    /// How long a pause has to be to end a sentence.
    #[serde(default = "default_live_pause_ms")]
    pub(crate) live_pause_ms: u32,
    /// End the recording by itself once nothing has been said for a while.
    #[serde(default = "default_true")]
    pub(crate) silence_stop: bool,
    #[serde(default = "default_silence_stop_ms")]
    pub(crate) silence_stop_ms: u32,
    /// After a silence stop, keep listening a moment: speech carries on the
    /// same dictation without the key.
    #[serde(default)]
    pub(crate) auto_resume: bool,
    #[serde(default = "default_auto_resume_ms")]
    pub(crate) auto_resume_ms: u32,
    /// Press Enter in the target app once the dictation is over and typed.
    #[serde(default)]
    pub(crate) auto_enter: bool,
    /// Drop the "uh, uh..." the model writes when a piece ends mid-thought.
    #[serde(default = "default_true")]
    pub(crate) tidy_sentence_ends: bool,
    /// Run the ONNX models - Parakeet, Moonshine - on the GPU, through CoreML.
    ///
    /// Off by default, which is the opposite of what it sounds like it should
    /// be. On an M2 Pro, over a minute of speech, Parakeet took 6.76 s with it
    /// and 1.87 s without, and its load went from 0.63 s to 5.72 s. Moonshine
    /// took 2.59 s against 1.66 s. The models are quantised to 8-bit integers,
    /// which CoreML handles poorly - it hands parts back to the processor and
    /// pays for the crossing each time.
    ///
    /// Kept as a switch rather than removed: that is one Mac, and another one,
    /// or a later ONNX Runtime, could well be faster with it on.
    #[serde(default)]
    pub(crate) onnx_gpu: bool,
    /// Run Whisper on the GPU, through Metal. A separate runtime from the one
    /// above, and a separate switch, because the answer is the opposite: same
    /// Mac, same minute of speech, Whisper Turbo took 3.29 s with it and
    /// 12.21 s without. On unless it is turned off.
    ///
    /// `both_gpu_switches_change_where_the_model_runs` in `tests.rs` is where
    /// all four numbers come from, and re-measures them on any Mac.
    #[serde(default = "default_true")]
    pub(crate) whisper_gpu: bool,
    /// Set once the settings held in the browser have been copied into here, so
    /// the copy happens exactly once and never overwrites a later change.
    #[serde(default)]
    pub(crate) migrated_from_browser: bool,
}

pub(crate) fn default_shortcut() -> String {
    "F3".to_string()
}

pub(crate) fn default_mic_boost() -> f32 {
    1.0
}

// The pictures the indicator can draw. The drawing is in
// src/components/visualisations.ts; this list is what the setting may hold.
pub(crate) const VISUALISATIONS: [&str; 6] =
    ["waterfall", "bars", "mirror", "ring", "dots", "curve"];

pub(crate) fn default_visualisation() -> String {
    VISUALISATIONS[0].to_string()
}

// Whisper language codes the settings window offers. "auto" is detection.
pub(crate) const LANGUAGES: [(&str, &str); 9] = [
    ("auto", "As spoken"),
    ("en", "English"),
    ("bg", "Bulgarian"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
    ("ru", "Russian"),
];

pub(crate) fn default_language() -> String {
    "auto".to_string()
}

// What the model is told, None meaning it decides.
pub(crate) fn whisper_language(setting: &str) -> Option<String> {
    (setting != "auto" && LANGUAGES.iter().any(|(code, _)| *code == setting))
        .then(|| setting.to_string())
}

// Half to a hundred times, in tenths.
pub(crate) fn clamp_mic_boost(boost: f32) -> f32 {
    if boost.is_finite() {
        (boost.clamp(0.5, 100.0) * 10.0).round() / 10.0
    } else {
        1.0
    }
}

// 2.2 seconds. Shorter than this and it starts editing the breaths between
// sentences, which is where Whisper gets its full stops from.
pub(crate) fn default_pause_cutoff_ms() -> u32 {
    2200
}

// 1.5 seconds: enough to cover the first words, which is what Whisper
// settles the language and the writing style from.
pub(crate) fn default_pause_opening_ms() -> u32 {
    1500
}

// 0.7 s: the breath between two sentences, not the one between two words.
pub(crate) fn default_live_pause_ms() -> u32 {
    700
}

// 3.5 seconds without a word before the recording ends by itself.
pub(crate) fn default_silence_stop_ms() -> u32 {
    3500
}

// 6 seconds of listening after a silence stop.
pub(crate) fn default_auto_resume_ms() -> u32 {
    6000
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
            mic_boost: default_mic_boost(),
            visualisation: default_visualisation(),
            language: default_language(),
            pause_shortening: true,
            pause_cutoff_ms: default_pause_cutoff_ms(),
            pause_protect_opening: true,
            pause_opening_ms: default_pause_opening_ms(),
            live_typing: true,
            live_pause_ms: default_live_pause_ms(),
            silence_stop: true,
            silence_stop_ms: default_silence_stop_ms(),
            auto_resume: false,
            auto_resume_ms: default_auto_resume_ms(),
            auto_enter: false,
            tidy_sentence_ends: true,
            onnx_gpu: false,
            whisper_gpu: true,
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

// The indicator hears about it at once, so the change shows on the next
// dictation without reopening anything.
#[tauri::command]
pub(crate) async fn set_visualisation(app: AppHandle, name: String) -> Result<(), String> {
    if !VISUALISATIONS.contains(&name.as_str()) {
        return Err(format!("\"{}\" is not a visualisation.", name));
    }
    app.state::<AudioState>()
        .update_prefs(|p| p.visualisation = name.clone());
    let _ = app.emit("visualisation-changed", name);
    Ok(())
}

#[tauri::command]
pub(crate) async fn set_language(state: State<'_, AudioState>, code: String) -> Result<(), String> {
    if !LANGUAGES.iter().any(|(c, _)| *c == code) {
        return Err(format!("\"{}\" is not a language this app offers.", code));
    }
    state.update_prefs(|p| p.language = code);
    Ok(())
}

// Takes effect at the next recording: the running capture reads it once.
#[tauri::command]
pub(crate) async fn set_mic_boost(state: State<'_, AudioState>, boost: f32) -> Result<f32, String> {
    let boost = clamp_mic_boost(boost);
    state.update_prefs(|p| p.mic_boost = boost);
    eprintln!("Microphone boost: {}x", boost);
    Ok(boost)
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

// Live typing and the silence stop. The limits are the window's again, for a
// file edited by hand.
#[tauri::command]
pub(crate) async fn set_live_typing(
    state: State<'_, AudioState>,
    enabled: bool,
) -> Result<(), String> {
    state.update_prefs(|p| p.live_typing = enabled);
    Ok(())
}

#[tauri::command]
pub(crate) async fn set_live_pause_ms(
    state: State<'_, AudioState>,
    milliseconds: u32,
) -> Result<u32, String> {
    let milliseconds = milliseconds.clamp(300, 10_000);
    state.update_prefs(|p| p.live_pause_ms = milliseconds);
    Ok(milliseconds)
}

#[tauri::command]
pub(crate) async fn set_silence_stop(
    state: State<'_, AudioState>,
    enabled: bool,
) -> Result<(), String> {
    state.update_prefs(|p| p.silence_stop = enabled);
    Ok(())
}

#[tauri::command]
pub(crate) async fn set_silence_stop_ms(
    state: State<'_, AudioState>,
    milliseconds: u32,
) -> Result<u32, String> {
    let milliseconds = milliseconds.clamp(1000, 60_000);
    state.update_prefs(|p| p.silence_stop_ms = milliseconds);
    Ok(milliseconds)
}

#[tauri::command]
pub(crate) async fn set_auto_resume(
    state: State<'_, AudioState>,
    enabled: bool,
) -> Result<(), String> {
    state.update_prefs(|p| p.auto_resume = enabled);
    Ok(())
}

#[tauri::command]
pub(crate) async fn set_auto_resume_ms(
    state: State<'_, AudioState>,
    milliseconds: u32,
) -> Result<u32, String> {
    let milliseconds = milliseconds.clamp(1000, 30_000);
    state.update_prefs(|p| p.auto_resume_ms = milliseconds);
    Ok(milliseconds)
}

#[tauri::command]
pub(crate) async fn set_auto_enter(
    state: State<'_, AudioState>,
    enabled: bool,
) -> Result<(), String> {
    state.update_prefs(|p| p.auto_enter = enabled);
    Ok(())
}

#[tauri::command]
pub(crate) async fn set_tidy_sentence_ends(
    state: State<'_, AudioState>,
    enabled: bool,
) -> Result<(), String> {
    state.update_prefs(|p| p.tidy_sentence_ends = enabled);
    Ok(())
}

// Start the app when the computer starts.
//
// This one setting is not in tray-prefs.json. The system holds it: on macOS it
// is the file ~/Library/LaunchAgents/Omegawhisper.plist, which the user can
// also delete from System Settings. A copy here could say "on" while the file
// is gone, so the system is asked every time instead.

#[tauri::command]
pub(crate) fn get_start_at_login(app: AppHandle) -> Result<bool, String> {
    use tauri_plugin_autostart::ManagerExt;
    app.autolaunch().is_enabled().map_err(|e| e.to_string())
}

#[tauri::command]
pub(crate) fn set_start_at_login(app: AppHandle, enabled: bool) -> Result<(), String> {
    use tauri_plugin_autostart::ManagerExt;
    let manager = app.autolaunch();
    let result = if enabled {
        manager.enable()
    } else {
        manager.disable()
    };
    match result {
        Ok(()) => {
            eprintln!("Start at login: {}", if enabled { "on" } else { "off" });
            Ok(())
        }
        Err(e) => {
            eprintln!("Could not change start at login: {}", e);
            Err(e.to_string())
        }
    }
}

// The saved login entry holds the full path to the app, and nothing checks
// that the path still leads anywhere. Move the app to another folder and it
// silently stops starting. Writing the entry again at every startup points it
// at wherever the app is being run from now.
//
// Only in a release build. A `bun run tauri dev` run would otherwise point the
// login entry at the development binary in `target/`.
pub(crate) fn refresh_start_at_login(app: &AppHandle) {
    if cfg!(debug_assertions) {
        return;
    }
    use tauri_plugin_autostart::ManagerExt;
    let manager = app.autolaunch();
    match manager.is_enabled() {
        Ok(true) => {
            if let Err(e) = manager.enable() {
                eprintln!(
                    "Could not point start at login at this copy of the app: {}",
                    e
                );
            }
        }
        Ok(false) => {}
        Err(e) => eprintln!("Could not read start at login: {}", e),
    }
}

// The two GPU switches. Saving is not enough on its own: a model already in
// memory was built the old way and would go on running that way. The next
// dictation rebuilds it, because `load_model` compares the switches against
// the ones the loaded model was built with.
#[tauri::command]
pub(crate) async fn set_onnx_gpu(
    state: State<'_, AudioState>,
    enabled: bool,
) -> Result<(), String> {
    state.update_prefs(|p| p.onnx_gpu = enabled);
    eprintln!(
        "Parakeet and Moonshine will run on the {} from the next dictation.",
        if enabled { "GPU" } else { "processor" }
    );
    Ok(())
}

#[tauri::command]
pub(crate) async fn set_whisper_gpu(
    state: State<'_, AudioState>,
    enabled: bool,
) -> Result<(), String> {
    state.update_prefs(|p| p.whisper_gpu = enabled);
    eprintln!(
        "Whisper will run on the {} from the next dictation.",
        if enabled { "GPU" } else { "processor" }
    );
    Ok(())
}
