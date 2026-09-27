// Which microphone to record from, and what format it will give us.

use crate::AudioState;
use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{Device, SampleFormat, SupportedStreamConfig};
use tauri::State;

/// One microphone the app could record from.
#[derive(Clone, serde::Serialize)]
pub struct AudioDevice {
    /// What the recording is opened with, and what is saved.
    pub name: String,
    /// What the settings window shows.
    pub label: String,
    /// True for the one the system would use if nothing were chosen.
    pub is_default: bool,
}

// Every microphone the app can open. From cpal on a Mac. On Linux cpal lists
// every ALSA plugin it can name, dozens of them, so PipeWire is asked instead.
#[tauri::command]
pub(crate) fn list_audio_devices() -> Result<Vec<AudioDevice>, String> {
    #[cfg(target_os = "linux")]
    return linux_sources();
    #[cfg(not(target_os = "linux"))]
    cpal_devices()
}

#[cfg(target_os = "linux")]
fn linux_sources() -> Result<Vec<AudioDevice>, String> {
    let out = std::process::Command::new("pactl")
        .args(["--format=json", "list", "sources"])
        .output()
        .map_err(|e| format!("Could not ask PipeWire for microphones (pactl: {})", e))?;
    if !out.status.success() {
        return Err("Could not ask PipeWire for microphones.".to_string());
    }
    let default = std::process::Command::new("pactl")
        .arg("get-default-source")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    parse_sources(&out.stdout, &default)
}

// pactl's JSON. Monitors of outputs are left out: nobody dictates through
// their own speakers.
#[cfg(target_os = "linux")]
pub(crate) fn parse_sources(json: &[u8], default: &str) -> Result<Vec<AudioDevice>, String> {
    let sources: Vec<serde_json::Value> = serde_json::from_slice(json)
        .map_err(|e| format!("Could not read the microphone list: {}", e))?;
    Ok(sources
        .iter()
        .filter_map(|s| {
            let name = s.get("name")?.as_str()?.to_string();
            if name.ends_with(".monitor") {
                return None;
            }
            let label = s
                .get("description")
                .and_then(|d| d.as_str())
                .filter(|d| !d.is_empty())
                .unwrap_or(&name)
                .to_string();
            Some(AudioDevice {
                is_default: name == default,
                name,
                label,
            })
        })
        .collect())
}

#[cfg(not(target_os = "linux"))]
fn cpal_devices() -> Result<Vec<AudioDevice>, String> {
    let host = cpal::default_host();
    let default_name = host.default_input_device().and_then(|d| d.name().ok());

    let devices = host
        .input_devices()
        .map_err(|e| format!("Could not ask the system for microphones: {}", e))?;

    Ok(devices
        .filter_map(|device| device.name().ok())
        .map(|name| AudioDevice {
            is_default: Some(&name) == default_name.as_ref(),
            label: name.clone(),
            name,
        })
        .collect())
}

#[tauri::command]
pub(crate) fn set_selected_device(state: State<'_, AudioState>, name: Option<String>) {
    let name = name.filter(|n| !n.trim().is_empty());
    state.update_prefs(|p| p.selected_microphone = name);
}

// The microphone to record from: the one chosen in Settings, or the system's
// own choice when nothing is chosen or the chosen one has been unplugged.
pub(crate) fn get_input_device(wanted: Option<&str>) -> Result<Device, String> {
    let host = cpal::default_host();

    // The "pipewire" ALSA device opens the source PIPEWIRE_NODE names, or
    // the default when it is unset.
    #[cfg(target_os = "linux")]
    {
        let connected = wanted
            .filter(|name| linux_sources().is_ok_and(|list| list.iter().any(|d| d.name == *name)));
        match (wanted, connected) {
            (Some(name), Some(_)) => {
                std::env::set_var("PIPEWIRE_NODE", name);
                eprintln!("microphone: {}", name);
            }
            (Some(name), None) => {
                std::env::remove_var("PIPEWIRE_NODE");
                eprintln!(
                    "microphone: \"{}\" is not connected, using the system's own choice instead",
                    name
                );
            }
            (None, _) => std::env::remove_var("PIPEWIRE_NODE"),
        }
        if let Ok(devices) = host.input_devices() {
            for device in devices {
                if device.name().is_ok_and(|name| name == "pipewire") {
                    if wanted.is_none() {
                        eprintln!("microphone: pipewire (follows the system default)");
                    }
                    return Ok(device);
                }
            }
        }
    }

    if let Some(wanted) = wanted {
        if let Ok(devices) = host.input_devices() {
            for device in devices {
                if device.name().is_ok_and(|name| name == wanted) {
                    eprintln!("microphone: {}", wanted);
                    return Ok(device);
                }
            }
        }
        // Said out loud rather than swapped silently: a recording that comes
        // back from the wrong microphone is otherwise impossible to explain.
        eprintln!(
            "microphone: \"{}\" is not connected, using the system's own choice instead",
            wanted
        );
    }

    // Linux: "pipewire" follows whatever WirePlumber has set as default and
    // handles Bluetooth better than the raw ALSA devices behind it.
    if let Ok(devices) = host.input_devices() {
        for device in devices {
            if device.name().is_ok_and(|name| name == "pipewire") {
                eprintln!("microphone: pipewire (follows the system default)");
                return Ok(device);
            }
        }
    }

    let device = host
        .default_input_device()
        .ok_or_else(|| "No microphone found. Check System Settings > Sound > Input.".to_string())?;
    if let Ok(name) = device.name() {
        eprintln!("microphone: {} (the system's own choice)", name);
    }
    Ok(device)
}

// Get a safe stream config that works with Bluetooth devices
// Bluetooth audio on Linux (especially with PipeWire) can crash GNOME when using
// certain buffer sizes or sample rates. This function tries to find a safer config.
pub(crate) fn get_safe_input_config(device: &Device) -> Result<SupportedStreamConfig, String> {
    // First, try to get supported configs and find one that's known to work well
    if let Ok(configs) = device.supported_input_configs() {
        let configs: Vec<_> = configs.collect();

        // Prefer 48000 Hz or 44100 Hz with F32 format - these are most compatible
        let preferred_rates = [48000u32, 44100, 16000, 32000, 96000];

        for rate in preferred_rates {
            for config in &configs {
                if config.min_sample_rate().0 <= rate
                    && config.max_sample_rate().0 >= rate
                    && config.sample_format() == SampleFormat::F32
                {
                    return Ok((*config).with_sample_rate(cpal::SampleRate(rate)));
                }
            }
            // If F32 not available at this rate, try I16
            for config in &configs {
                if config.min_sample_rate().0 <= rate
                    && config.max_sample_rate().0 >= rate
                    && config.sample_format() == SampleFormat::I16
                {
                    return Ok((*config).with_sample_rate(cpal::SampleRate(rate)));
                }
            }
        }
    }

    // Fall back to default config if no preferred config found
    device
        .default_input_config()
        .map_err(|e| format!("Failed to get input config: {}", e))
}
