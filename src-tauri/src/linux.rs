// Linux only: the desktop file the portal knows the app by, the dictation key
// through the GlobalShortcuts portal on Wayland, and the indicator as a
// layer-shell overlay.

use crate::recording::toggle_recording;
use crate::AudioState;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

pub(crate) const APP_ID: &str = "dev.omegawhisper";
const SHORTCUT_ID: &str = "toggle";
const SHORTCUT_DESCRIPTION: &str = "Start or stop dictation";
const NOT_REGISTERED: &str = "The dictation key is not registered with the desktop.";

pub(crate) fn on_wayland() -> bool {
    std::env::var_os("WAYLAND_DISPLAY").is_some_and(|v| !v.is_empty())
}

// An X11 key grab sees nothing under Wayland, so the desktop holds the key.
pub(crate) fn shortcut_set_by_system() -> bool {
    on_wayland()
}

pub(crate) enum PortalCommand {
    OpenSettings,
}

#[derive(Default)]
pub(crate) struct Portal {
    // The key as the desktop describes it. None until it has answered.
    pub(crate) trigger: Mutex<Option<String>>,
    commands: Mutex<Option<UnboundedSender<PortalCommand>>>,
}

fn desktop_file_path() -> Option<PathBuf> {
    dirs::data_dir().map(|d| d.join("applications").join(format!("{APP_ID}.desktop")))
}

pub(crate) fn desktop_entry(exe: &str) -> String {
    format!(
        "[Desktop Entry]\n\
         Type=Application\n\
         Name=Omegawhisper\n\
         Comment=Press a key, speak, and the text is typed into the app you are using\n\
         Exec={exe}\n\
         Icon=omegawhisper\n\
         Terminal=false\n\
         Categories=Utility;AudioVideo;\n"
    )
}

// Release builds keep Exec= pointing at themselves. A dev build leaves an
// existing entry alone rather than point the launcher at target/.
pub(crate) fn write_desktop_file() {
    let Some(path) = desktop_file_path() else {
        return;
    };
    if cfg!(debug_assertions) && path.exists() {
        return;
    }
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let wanted = desktop_entry(&exe.display().to_string());
    if fs::read_to_string(&path).ok().as_deref() == Some(wanted.as_str()) {
        return;
    }
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    match fs::write(&path, wanted) {
        Ok(()) => eprintln!("Wrote {}", path.display()),
        Err(e) => eprintln!("Could not write {}: {}", path.display(), e),
    }
}

// Tauri's "CommandOrControl+Shift+D" in the portal's "CTRL+SHIFT+d".
pub(crate) fn xdg_trigger(accelerator: &str) -> String {
    accelerator
        .split('+')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| match part.to_ascii_lowercase().as_str() {
            "commandorcontrol" | "cmdorctrl" | "control" | "ctrl" => "CTRL".to_string(),
            "shift" => "SHIFT".to_string(),
            "alt" | "option" => "ALT".to_string(),
            "super" | "command" | "cmd" | "meta" | "win" => "LOGO".to_string(),
            "space" => "space".to_string(),
            "enter" | "return" => "Return".to_string(),
            key => {
                let key = key.strip_prefix("key").unwrap_or(key);
                let key = key.strip_prefix("digit").unwrap_or(key);
                if key.chars().count() == 1 {
                    key.to_string()
                } else {
                    part.to_string()
                }
            }
        })
        .collect::<Vec<_>>()
        .join("+")
}

pub(crate) fn watch_portal_shortcut(app: AppHandle) {
    let (tx, rx) = unbounded_channel();
    *app.state::<AudioState>().portal.commands.lock().unwrap() = Some(tx);
    tauri::async_runtime::spawn(async move {
        if let Err(e) = portal_shortcut(&app, rx).await {
            app.state::<AudioState>().warn(format!(
                "The dictation key could not be registered with the desktop ({e})."
            ));
        }
    });
}

async fn portal_shortcut(
    app: &AppHandle,
    mut commands: UnboundedReceiver<PortalCommand>,
) -> ashpd::Result<()> {
    use ashpd::desktop::global_shortcuts::{
        BindShortcutsOptions, ConfigureShortcutsOptions, GlobalShortcuts, NewShortcut, Shortcut,
    };
    use ashpd::desktop::CreateSessionOptions;
    use futures_util::StreamExt;

    ashpd::register_host_app(APP_ID.parse()?).await?;
    let portal = GlobalShortcuts::new().await?;
    let session = portal
        .create_session(CreateSessionOptions::default())
        .await?;

    let remember = |bound: &[Shortcut]| {
        let trigger = bound
            .iter()
            .find(|s| s.id() == SHORTCUT_ID)
            .map(|s| s.trigger_description().to_string());
        eprintln!(
            "Dictation key from the desktop: {}",
            trigger.as_deref().unwrap_or("none")
        );
        *app.state::<AudioState>().portal.trigger.lock().unwrap() = trigger;
        let _ = app.emit("shortcut-changed", ());
    };

    let mut changed = portal.receive_shortcuts_changed().await?;
    let mut activated = portal.receive_activated().await?;

    // The desktop asks the user once. A refusal leaves the app running with
    // no key; Settings can open the desktop's shortcut page later.
    let wanted = xdg_trigger(&app.state::<AudioState>().prefs().shortcut);
    let shortcuts =
        [NewShortcut::new(SHORTCUT_ID, SHORTCUT_DESCRIPTION).preferred_trigger(wanted.as_str())];
    match portal
        .bind_shortcuts(&session, &shortcuts, None, BindShortcutsOptions::default())
        .await
        .and_then(|request| request.response())
    {
        Ok(bound) => remember(bound.shortcuts()),
        Err(e) => app.state::<AudioState>().warn(format!(
            "The desktop did not give Omegawhisper a dictation key ({e}). \
             Open Settings to choose one."
        )),
    }

    loop {
        tokio::select! {
            Some(event) = activated.next() => {
                if event.shortcut_id() == SHORTCUT_ID {
                    eprintln!("[{}] shortcut pressed", crate::now());
                    toggle_recording(app);
                }
            }
            Some(event) = changed.next() => remember(event.shortcuts()),
            Some(PortalCommand::OpenSettings) = commands.recv() => {
                if let Err(e) = portal
                    .configure_shortcuts(&session, None, ConfigureShortcutsOptions::default())
                    .await
                {
                    eprintln!("Could not open the desktop's shortcut settings: {e}");
                }
            }
            else => break,
        }
    }
    Ok(())
}

pub(crate) fn open_shortcut_settings(app: &AppHandle) -> Result<(), String> {
    app.state::<AudioState>()
        .portal
        .commands
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|tx| tx.send(PortalCommand::OpenSettings).ok())
        .ok_or_else(|| NOT_REGISTERED.to_string())
}

// Wayland lets no window place itself or stay on top; a layer-shell overlay
// anchored to the bottom does both, and never takes the keyboard. Must run
// before the window is realized.
pub(crate) fn float_indicator(window: &tauri::WebviewWindow) {
    use gtk::prelude::*;
    use gtk_layer_shell::{Edge, KeyboardMode, Layer, LayerShell};

    if !on_wayland() {
        return;
    }
    if !gtk_layer_shell::is_supported() {
        eprintln!("indicator: no layer shell here, the compositor places the window");
        return;
    }
    let gtk_window = match window.gtk_window() {
        Ok(w) => w,
        Err(e) => {
            eprintln!("indicator: no GTK window to place ({e})");
            return;
        }
    };
    if gtk_window.is_realized() {
        eprintln!("indicator: window already realized, cannot become a layer surface");
        return;
    }
    gtk_window.init_layer_shell();
    gtk_window.set_namespace("omegawhisper-indicator");
    gtk_window.set_layer(Layer::Overlay);
    gtk_window.set_anchor(Edge::Bottom, true);
    gtk_window.set_layer_shell_margin(Edge::Bottom, 90);
    gtk_window.set_keyboard_mode(KeyboardMode::None);
}
