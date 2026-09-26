// The folder the app keeps its recordings, models and settings in.

use chrono::Local;
use std::fs;
use std::path::PathBuf;

// Get the recordings directory, creating it if necessary
pub(crate) fn get_recordings_dir() -> Result<PathBuf, String> {
    let data_dir =
        dirs::data_local_dir().ok_or_else(|| "Could not find local data directory".to_string())?;
    let recordings_dir = data_dir.join("omegawhisper").join("recordings");

    if !recordings_dir.exists() {
        fs::create_dir_all(&recordings_dir)
            .map_err(|e| format!("Failed to create recordings directory: {}", e))?;
    }

    Ok(recordings_dir)
}

// Show a folder in the file manager. The opener plugin's detached launch
// double-forks and disowns xdg-open, and on KDE that xdg-open opens nothing
// while reporting success; a plain spawn that is waited on does open Dolphin.
pub(crate) fn open_folder(dir: &std::path::Path) -> Result<(), String> {
    #[cfg(target_os = "linux")]
    {
        use std::process::{Command, Stdio};
        let mut child = Command::new("xdg-open")
            .arg(dir)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("xdg-open could not start: {}", e))?;
        std::thread::spawn(move || {
            let _ = child.wait();
        });
        Ok(())
    }
    #[cfg(not(target_os = "linux"))]
    tauri_plugin_opener::open_path(dir, None::<&str>).map_err(|e| e.to_string())
}

// Deletes every recording in a folder. Only .wav files, so anything else that
// happens to be in there survives. Returns how many went.
pub(crate) fn delete_recordings_in(dir: &std::path::Path) -> Result<usize, String> {
    let entries =
        fs::read_dir(dir).map_err(|e| format!("Could not read {}: {}", dir.display(), e))?;
    let mut deleted = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("wav") {
            match fs::remove_file(&path) {
                Ok(()) => deleted += 1,
                Err(e) => eprintln!("Could not delete {}: {}", path.display(), e),
            }
        }
    }
    Ok(deleted)
}

// Everything the app prints goes to one file, whatever started it - Finder,
// the tray, or a terminal. Launched from Finder there is no terminal to print
// to, so a dictation that went wrong used to leave no trace at all.
#[cfg(unix)]
pub(crate) fn redirect_output_to_log() {
    let Some(dir) = dirs::data_local_dir() else {
        return;
    };
    let path = dir.join("omegawhisper").join("omegawhisper.log");
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    // Start over once the file gets big rather than growing without end.
    if fs::metadata(&path)
        .map(|m| m.len() > 5_000_000)
        .unwrap_or(false)
    {
        let _ = fs::remove_file(&path);
    }

    let Ok(file) = fs::OpenOptions::new().create(true).append(true).open(&path) else {
        return;
    };
    use std::os::unix::io::AsRawFd;
    unsafe {
        libc::dup2(file.as_raw_fd(), libc::STDOUT_FILENO);
        libc::dup2(file.as_raw_fd(), libc::STDERR_FILENO);
    }
    // The file must outlive this function: the two descriptors above now
    // point at it and closing it here would close them too.
    std::mem::forget(file);

    eprintln!(
        "\n===== started {} =====",
        Local::now().format("%F %H:%M:%S")
    );
}
