// The startup check for a newer release on GitHub, and the install once the
// user has said yes.

use crate::AudioState;
use std::thread;
use tauri::{AppHandle, Manager};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};
use tauri_plugin_updater::UpdaterExt;

// Release builds only: a dev run is never the version on GitHub. Its own
// thread, because the question is a blocking dialog.
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn check_on_startup(app: AppHandle) {
    if cfg!(debug_assertions) {
        return;
    }
    thread::spawn(move || {
        if let Err(e) = tauri::async_runtime::block_on(check(&app)) {
            eprintln!("Update check: {}", e);
        }
    });
}

async fn check(app: &AppHandle) -> Result<(), String> {
    let handle = app.clone();
    let updater = app
        .updater_builder()
        // The installer ends the process through exit(), which skips the
        // RunEvent::Exit teardown in run(); let the model go here instead.
        .on_before_exit(move || {
            handle
                .state::<AudioState>()
                .transcription_manager
                .unload_model()
        })
        .build()
        .map_err(|e| e.to_string())?;
    let Some(update) = updater.check().await.map_err(|e| e.to_string())? else {
        eprintln!(
            "Update check: {} is the latest release.",
            app.package_info().version
        );
        return Ok(());
    };
    eprintln!(
        "Update check: {} is available, this is {}.",
        update.version, update.current_version
    );

    let wanted = app
        .dialog()
        .message(format!(
            "Omegawhisper {} is available; this is {}.\n\nInstall it now? \
             The app restarts when it is done.",
            update.version, update.current_version
        ))
        .title("Update available")
        .kind(MessageDialogKind::Info)
        .buttons(MessageDialogButtons::OkCancelCustom(
            "Install".to_string(),
            "Later".to_string(),
        ))
        .blocking_show();
    if !wanted {
        eprintln!("Update check: declined, asked again at the next start.");
        return Ok(());
    }

    update
        .download_and_install(|_, _| {}, || eprintln!("Update: downloaded, installing."))
        .await
        .map_err(|e| e.to_string())?;
    // Windows never gets here: the installer exits the app and starts the
    // new one. Anywhere else the new build has to be started by hand.
    app.restart()
}
