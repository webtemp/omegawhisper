// The last few dictations, kept so one that landed in the wrong window can be
// fetched back from the tray. Newest first, in transcripts.json next to the
// settings, written the moment a dictation ends.

use crate::AudioState;
use chrono::Local;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use tauri::menu::{MenuItem, Submenu};
use tauri::{AppHandle, Manager, Wry};

pub(crate) const HISTORY_LIMIT: usize = 20;
// A dictation started this soon after silence ended the last one continues it.
pub(crate) const SESSION_GAP: std::time::Duration = std::time::Duration::from_secs(90);
pub(crate) const HISTORY_MENU_ID: &str = "copy_transcript_";
const LABEL_CHARS: usize = 48;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Transcript {
    // Wall-clock time as "YYYY-MM-DD HH:MM", enough to tell entries apart.
    pub when: String,
    pub text: String,
}

pub(crate) fn history_path() -> Option<PathBuf> {
    dirs::data_local_dir().map(|d| d.join("omegawhisper").join("transcripts.json"))
}

pub(crate) fn load_history() -> Vec<Transcript> {
    let Some(path) = history_path() else {
        return Vec::new();
    };
    let Ok(text) = fs::read_to_string(&path) else {
        return Vec::new();
    };
    match serde_json::from_str::<Vec<Transcript>>(&text) {
        Ok(list) => list,
        Err(e) => {
            eprintln!("Could not read {}: {}", path.display(), e);
            Vec::new()
        }
    }
}

pub(crate) fn save_history(list: &[Transcript]) {
    let Some(path) = history_path() else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    match serde_json::to_string_pretty(list) {
        Ok(text) => {
            if let Err(e) = fs::write(&path, text) {
                eprintln!("Could not write {}: {}", path.display(), e);
            }
        }
        Err(e) => eprintln!("Could not encode the transcript history: {}", e),
    }
}

// Newest first, capped at the limit. A continued session is added to the
// newest entry. The same text twice in a row is one entry: a retried
// dictation should not push an older one off the list.
pub(crate) fn remember(list: &mut Vec<Transcript>, entry: Transcript, continues: bool, limit: usize) {
    match list.first_mut() {
        Some(last) if continues => {
            last.text.push(' ');
            last.text.push_str(&entry.text);
        }
        Some(last) if last.text == entry.text => *last = entry,
        _ => list.insert(0, entry),
    }
    list.truncate(limit);
}

pub(crate) fn transcript_now(text: &str) -> Transcript {
    Transcript {
        when: Local::now().format("%Y-%m-%d %H:%M").to_string(),
        text: text.to_string(),
    }
}

// What the menu shows for one entry: the time and the first line of the
// text, cut short. `&` marks a mnemonic in a menu label, so it is doubled.
pub(crate) fn menu_label(entry: &Transcript) -> String {
    let one_line: String = entry
        .text
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let chars = one_line.chars().count();
    let shown: String = if chars > LABEL_CHARS {
        let mut s: String = one_line.chars().take(LABEL_CHARS - 1).collect();
        s.push('…');
        s
    } else {
        one_line
    };
    let when = chrono::NaiveDateTime::parse_from_str(&entry.when, "%Y-%m-%d %H:%M")
        .map(|t| t.format("%d %b %H:%M").to_string())
        .unwrap_or_else(|_| entry.when.clone());
    format!("{}  {}", when, shown).replace('&', "&&")
}

// Add a finished dictation and redraw the tray's list.
pub(crate) fn record_transcript(app: &AppHandle, text: &str) {
    let state = app.state::<AudioState>();
    let continues = std::mem::take(&mut *state.continue_session.lock().unwrap());
    {
        let mut history = state.history.lock().unwrap();
        remember(&mut history, transcript_now(text), continues, HISTORY_LIMIT);
        save_history(&history);
    }
    refresh_history_menu(app);
}

// Menus belong to the main thread on every desktop, so the rebuild is sent
// there. The transcription thread calls this.
pub(crate) fn refresh_history_menu(app: &AppHandle) {
    let handle = app.clone();
    let _ = app.run_on_main_thread(move || match crate::tray::rebuild_menu(&handle) {
        Ok(()) => eprintln!("Tray menu rebuilt with the latest transcript"),
        Err(e) => eprintln!("Could not update the transcript menu: {}", e),
    });
}

// The submenu, filled from the history as it is now.
pub(crate) fn build_history_menu(app: &AppHandle) -> tauri::Result<Submenu<Wry>> {
    let history = app.state::<AudioState>().history.lock().unwrap().clone();
    let menu = Submenu::with_id(app, "transcripts", "Last Transcripts", true)?;
    if history.is_empty() {
        let empty = MenuItem::with_id(app, "no_transcripts", "Nothing yet", false, None::<&str>)?;
        menu.append(&empty)?;
        return Ok(menu);
    }
    for (i, entry) in history.iter().enumerate() {
        let item = MenuItem::with_id(
            app,
            format!("{}{}", HISTORY_MENU_ID, i),
            menu_label(entry),
            true,
            None::<&str>,
        )?;
        menu.append(&item)?;
    }
    Ok(menu)
}

// A click on one of the entries: put its text on the clipboard.
pub(crate) fn handle_history_click(app: &AppHandle, id: &str) -> bool {
    let Some(index) = id.strip_prefix(HISTORY_MENU_ID) else {
        return false;
    };
    let Ok(index) = index.parse::<usize>() else {
        return false;
    };
    let text = app
        .state::<AudioState>()
        .history
        .lock()
        .unwrap()
        .get(index)
        .map(|t| t.text.clone());
    match text {
        Some(text) => match crate::typing::copy_to_clipboard(app, &text) {
            Ok(()) => eprintln!("Copied transcript {} to the clipboard", index),
            Err(e) => eprintln!("Could not copy transcript {}: {}", index, e),
        },
        None => eprintln!("No transcript at {}", index),
    }
    true
}
