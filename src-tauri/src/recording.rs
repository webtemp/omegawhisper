// Recording: opening the microphone, keeping what it hears, and handing it to
// the model when the recording stops.

use crate::analysis::{
    audio_stats, boost_quiet_audio, boost_samples, chunk_level, detect_pitch, frequency_bands,
    i16_to_f32, keep_recent, mix_to_mono, shorten_long_pauses, speech_level, to_wav_bytes,
    trim_quiet_edges, u16_to_f32, PauseRules, FFT_SIZE,
};
use crate::chime::{play_chime, START_CHIME};
use crate::indicator::show_indicator;
use crate::managers::GpuChoice;
use crate::microphone::{get_input_device, get_safe_input_config};
use crate::resampler::AudioResampler;
use crate::storage::get_recordings_dir;
use crate::history::record_transcript;
use crate::live::{LiveState, Segmenter, SharedLiveState};
use crate::typing::{copy_to_clipboard, press_enter, type_text_internal};
use crate::vad::{self, Heard, Silero};
use crate::AudioState;
use crate::{now, CompleteOnDrop, DictationStats, MicLevel, TranscriptionEvent};
use chrono::Utc;
use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{Device, SampleFormat};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager};

// What the capture callback writes into.
pub(crate) struct CaptureTargets {
    is_recording: Arc<Mutex<bool>>,
    recorded_samples: Arc<Mutex<Vec<f32>>>,
    mic_level: Arc<Mutex<(f32, f32)>>,
    mic_recent: Arc<Mutex<Vec<f32>>>,
    audio_tx: std::sync::mpsc::Sender<Vec<f32>>,
    channels: u16,
    boost: f32,
}

// A microphone problem the user has to be told about. Clearing the flag matters
// as much as the message: leaving it set made the next shortcut press fail with
// "Already recording", with nothing on screen to explain it.
pub(crate) fn report_microphone_failure(
    app: &AppHandle,
    is_recording: &Arc<Mutex<bool>>,
    message: String,
) {
    eprintln!("{}", message);
    if let Ok(mut flag) = is_recording.lock() {
        *flag = false;
    }
    let _ = app.emit("transcription-error", message);
}

// Runs on the audio thread: store and hand off, never wait.
pub(crate) fn build_capture_stream<T>(
    device: &Device,
    config: &cpal::StreamConfig,
    targets: CaptureTargets,
    on_broken: impl Fn(String) + Send + 'static,
    to_f32: fn(T) -> f32,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: cpal::SizedSample + 'static,
{
    device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            if !*targets.is_recording.lock().unwrap() {
                return;
            }
            let buffer: Vec<f32> = data.iter().copied().map(to_f32).collect();
            let mut buffer = mix_to_mono(buffer, targets.channels);
            boost_samples(&mut buffer, targets.boost);
            // Store for the WAV file
            targets
                .recorded_samples
                .lock()
                .unwrap()
                .extend_from_slice(&buffer);
            *targets.mic_level.lock().unwrap() = chunk_level(&buffer);
            keep_recent(&targets.mic_recent, &buffer);
            // Hand to the transcription thread
            let _ = targets.audio_tx.send(buffer);
        },
        // The microphone died mid-recording: unplugged, or taken by another app.
        // The sound from here on was never captured and cannot be recovered, so
        // say so and stop, which leaves what was already captured to transcribe.
        move |err| on_broken(format!("{}", err)),
        None,
    )
}

// Where the resampled audio goes while recording: the full buffer always,
// and the sentence splitter when live typing or the silence stop is on.
struct Intake {
    all_audio: Vec<f32>,
    segmenter: Option<Segmenter<Silero>>,
    sentence_tx: Option<std::sync::mpsc::Sender<Vec<f32>>>,
    silence_stop_ms: Option<u32>,
    stop_asked: bool,
    live_state: SharedLiveState,
    // Listening after a silence stop: nothing said within this ends it.
    armed_ms: Option<u32>,
    auto_resume: bool,
}

impl Intake {
    fn push(&mut self, app: &AppHandle, samples: Vec<f32>) {
        if let Some(segmenter) = self.segmenter.as_mut() {
            for sentence in segmenter.feed(&samples) {
                if let Some(tx) = &self.sentence_tx {
                    let _ = tx.send(sentence);
                }
            }
            if self.sentence_tx.is_some() {
                self.live_state.lock().unwrap().pause = segmenter.pause_progress();
            }
            if let Some(limit) = self.silence_stop_ms {
                let silent = segmenter.silence_seconds();
                let allowed = match (segmenter.heard_speech(), self.armed_ms) {
                    (false, Some(armed)) => armed as f32 / 1000.0,
                    (false, None) => limit as f32 / 1000.0,
                    (true, _) => segmenter.stop_after_seconds(limit as f32 / 1000.0),
                };
                if !self.stop_asked && silent >= allowed {
                    self.stop_asked = true;
                    let resume = self.auto_resume && segmenter.heard_speech();
                    eprintln!(
                        "[{}] nothing said for {:.1} s (limit {:.1} s), stopping the recording{}",
                        now(),
                        silent,
                        allowed,
                        if resume { ", listening for more" } else { "" }
                    );
                    let state = app.state::<AudioState>();
                    *state.session_open_until.lock().unwrap() =
                        Some(std::time::Instant::now() + crate::history::SESSION_GAP);
                    *state.resume_pending.lock().unwrap() = resume;
                    // Blocking, and it must not wait for this thread.
                    let app = app.clone();
                    thread::spawn(move || {
                        if let Err(e) = stop_recording_internal(&app) {
                            eprintln!("silence stop: {}", e);
                            return;
                        }
                        if resume {
                            if let Err(e) = start_recording_internal(&app, true) {
                                eprintln!("could not listen for more: {}", e);
                            }
                        }
                    });
                }
            }
        }
        self.all_audio.extend(samples);
    }
}

// Save the exact audio handed to the model, at 16kHz, so it can be listened
// to afterwards. This is not the same as the saved recording: it is after
// resampling and after voice detection has cut pieces out, which is the
// whole point - it is what the model heard, not what the microphone heard.
pub(crate) fn save_model_input(samples: &[f32]) -> Option<PathBuf> {
    let dir = get_recordings_dir().ok()?;
    let path = dir.join(format!(
        "{}-model-input.wav",
        Utc::now().format("%Y-%m-%d_%H-%M-%S")
    ));
    fs::write(&path, to_wav_bytes(samples, 16_000, 1)).ok()?;
    Some(path)
}

#[tauri::command]
pub(crate) async fn start_recording(app_handle: AppHandle) -> Result<(), String> {
    tokio::task::spawn_blocking(move || start_recording_internal(&app_handle, false))
        .await
        .map_err(|e| format!("Task join error: {}", e))?
}

// The whole of starting a recording, as a plain function, so the dictation key
// can call it as directly as the button does. Blocking: it opens the
// microphone, so it must never run on the thread that draws the screen.
// `armed`: started by the app after a silence stop, to hear whether the
// speaker goes on. Quiet if nothing comes.
pub(crate) fn start_recording_internal(app_handle: &AppHandle, armed: bool) -> Result<(), String> {
    let state = app_handle.state::<AudioState>();
    let app_handle = app_handle.clone();

    // Check recording state
    {
        let is_recording = state.is_recording.lock().unwrap();
        if *is_recording {
            return Err("Already recording".to_string());
        }
    }
    eprintln!(
        "[{}] {}",
        now(),
        if armed { "listening for more" } else { "recording started" }
    );

    // Picked up again soon after silence ended the last one: same session.
    let continues = armed
        || state
            .session_open_until
            .lock()
            .unwrap()
            .take()
            .is_some_and(|until| std::time::Instant::now() < until);
    *state.continue_session.lock().unwrap() = continues;

    let prefs = state.prefs();
    let auto_resume = prefs.auto_resume;
    let armed_ms = armed.then_some(prefs.auto_resume_ms);
    let auto_enter = prefs.auto_enter;
    let tidy = prefs.tidy_sentence_ends;

    let model_id = prefs
        .active_local_model_id
        .clone()
        .ok_or_else(|| "No model chosen. Open Settings and pick one to download.".to_string())?;
    if !state.model_manager.is_model_downloaded(&model_id) {
        return Err(format!(
            "The {} model is not downloaded. Open Settings and download it.",
            model_id
        ));
    }

    // Clone transcription manager for the thread
    let transcription_manager = state.transcription_manager.0.clone();

    // Read once here: the transcription thread cannot reach the app state.
    let gpu = GpuChoice {
        onnx: prefs.onnx_gpu,
        whisper: prefs.whisper_gpu,
    };

    let language = crate::settings::whisper_language(&prefs.language);
    let live_typing = prefs.live_typing;
    let live_pause_ms = prefs.live_pause_ms;
    let silence_stop_ms = prefs.silence_stop.then_some(prefs.silence_stop_ms);
    let shorten_pauses = prefs.pause_shortening.then_some(PauseRules {
        cutoff_ms: prefs.pause_cutoff_ms,
        protect_opening_ms: prefs
            .pause_protect_opening
            .then_some(prefs.pause_opening_ms),
    });

    // Cleared, then given room below once the sample rate is known.
    state.recorded_samples.lock().unwrap().clear();

    let device = get_input_device(prefs.selected_microphone.as_deref())?;
    let mic_boost = prefs.mic_boost;
    if mic_boost != 1.0 {
        eprintln!("microphone boost: {}x", mic_boost);
    }
    let config = get_safe_input_config(&device)?;

    let sample_format = config.sample_format();
    let sample_rate = config.sample_rate().0;
    let channels = config.channels();

    // Room for five minutes, set aside now. The audio callback appends to this
    // list and must never be slow: growing it there means copying every sample
    // recorded so far into a bigger block, which at two minutes is 23 MB, in the
    // one place that cannot afford to wait. The memory is handed back when the
    // recording stops.
    const RESERVE_SECONDS: usize = 300;
    state
        .recorded_samples
        .lock()
        .unwrap()
        .reserve(sample_rate as usize * RESERVE_SECONDS);

    // Use default buffer size - fixed sizes can cause issues with Bluetooth on PipeWire
    let stream_config: cpal::StreamConfig = config.into();

    // Store sample rate
    *state.sample_rate.lock().unwrap() = Some(sample_rate);

    let is_recording_arc = state.is_recording.clone();
    let recorded_samples_arc = state.recorded_samples.clone();

    // Set recording flag
    *state.is_recording.lock().unwrap() = true;

    // Create channel for stop signal
    let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
    *state.stop_signal.lock().unwrap() = Some(stop_tx);

    // Channel for sending audio chunks to transcription thread
    let (audio_tx, audio_rx) = std::sync::mpsc::channel::<Vec<f32>>();

    // What the indicator shows about live typing.
    let live_state: SharedLiveState = Arc::new(Mutex::new(LiveState {
        armed,
        ..LiveState::default()
    }));

    // Everything is recorded first and transcribed once the recording stops,
    // which is why the app looks frozen for a moment afterwards.
    let live_state_ws = live_state.clone();
    let app_handle_ws = app_handle.clone();
    let is_recording_ws = is_recording_arc.clone();
    let stop_signal_ws = state.stop_signal.clone();

    thread::spawn(move || {
        // Helper to stop recording on error
        let stop_recording_on_error =
            |is_recording: &Arc<Mutex<bool>>,
             stop_signal: &Arc<Mutex<Option<std::sync::mpsc::Sender<()>>>>| {
                *is_recording.lock().unwrap() = false;
                if let Some(stop_tx) = stop_signal.lock().unwrap().take() {
                    let _ = stop_tx.send(());
                }
            };

        // Load the model if not already loaded
        {
            let mut manager = match transcription_manager.lock() {
                Ok(m) => m,
                Err(e) => {
                    let _ = app_handle_ws.emit(
                        "transcription-error",
                        format!("Failed to access transcription engine: {}", e),
                    );
                    stop_recording_on_error(&is_recording_ws, &stop_signal_ws);
                    return;
                }
            };

            // load_model decides for itself whether anything has to be built:
            // a different model, or the same one with the GPU switches moved
            // since it was loaded.
            if let Err(e) = manager.load_model(&model_id, gpu) {
                let _ = app_handle_ws.emit(
                    "transcription-error",
                    format!("Failed to load model: {}", e),
                );
                stop_recording_on_error(&is_recording_ws, &stop_signal_ws);
                return;
            }
        }

        // Create resampler to convert device sample rate to 16kHz
        let mut resampler = match AudioResampler::new(sample_rate) {
            Ok(r) => r,
            Err(e) => {
                let _ = app_handle_ws.emit(
                    "transcription-error",
                    format!("Failed to create resampler: {}", e),
                );
                stop_recording_on_error(&is_recording_ws, &stop_signal_ws);
                return;
            }
        };

        // The sentence splitter listens along when a sentence is to be typed
        // as soon as it ends, or when silence is to end the recording. If the
        // detector cannot load, both fall back to the plain recording.
        let segmenter = if live_typing || silence_stop_ms.is_some() {
            match Silero::new() {
                Ok(detector) => Some(Segmenter::new(detector, live_pause_ms)),
                Err(e) => {
                    eprintln!("{} The text will be typed when the recording stops.", e);
                    None
                }
            }
        } else {
            None
        };
        let live = live_typing && segmenter.is_some();

        // Sentences go to their own thread, so a slow model run never holds
        // up the audio coming in. The channel keeps them in order.
        let (sentence_tx, sentence_rx) = std::sync::mpsc::channel::<Vec<f32>>();
        let sentence_thread = live.then(|| {
            let app = app_handle_ws.clone();
            let manager = transcription_manager.clone();
            let language = language.clone();
            let live_state = live_state_ws.clone();
            thread::spawn(move || {
                crate::live::type_sentences(app, manager, language, sentence_rx, live_state, tidy)
            })
        });

        let mut intake = Intake {
            all_audio: Vec::new(),
            segmenter,
            sentence_tx: live.then_some(sentence_tx),
            silence_stop_ms,
            stop_asked: false,
            live_state: live_state_ws,
            armed_ms,
            auto_resume,
        };

        loop {
            // Check if we should stop
            if !*is_recording_ws.lock().unwrap() {
                break;
            }

            // Receive audio data with timeout
            match audio_rx.recv_timeout(Duration::from_millis(50)) {
                Ok(samples) => {
                    // Resample to 16kHz
                    if let Ok(resampled) = resampler.process(&samples) {
                        intake.push(&app_handle_ws, resampled);
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    // No data, continue checking
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    // Channel closed - process remaining audio
                    break;
                }
            }
        }

        // Transcription happens here after loop exits (either from stop signal or channel disconnect)

        // Take whatever is still waiting in the channel. The loop above
        // leaves the moment recording stops, so without this the last
        // chunks captured - and any backlog, if resampling fell behind -
        // would never reach the model.
        while let Ok(samples) = audio_rx.try_recv() {
            if let Ok(resampled) = resampler.process(&samples) {
                intake.push(&app_handle_ws, resampled);
            }
        }

        // Emit processing state
        let _ = app_handle_ws.emit("transcription-processing", ());
        let _complete = CompleteOnDrop(app_handle_ws.clone());

        // Flush resampler
        if let Ok(final_samples) = resampler.flush() {
            intake.push(&app_handle_ws, final_samples);
        }

        let Intake {
            mut all_audio,
            segmenter,
            sentence_tx,
            ..
        } = intake;

        // Enter goes only when the dictation is really over: not before a
        // listen for more, and after one that heard nothing.
        let send_enter = {
            let app = app_handle_ws.clone();
            move |typed_now: bool| {
                let resuming = *app.state::<AudioState>().resume_pending.lock().unwrap();
                if auto_enter && !resuming && (typed_now || armed) {
                    match press_enter(&app) {
                        Ok(()) => eprintln!("Enter pressed"),
                        Err(e) => eprintln!("{}", e),
                    }
                }
            }
        };

        // Live typing: the last sentence goes out, then the sentence thread
        // is waited for, and what it typed is reported as one dictation.
        if let Some(sentence_thread) = sentence_thread {
            let audio_seconds = all_audio.len() as f32 / 16_000.0;
            if let Some(last) = segmenter.and_then(|s| s.finish()) {
                if let Some(tx) = &sentence_tx {
                    let _ = tx.send(last);
                }
            }
            drop(sentence_tx);
            let outcome = match sentence_thread.join() {
                Ok(outcome) => outcome,
                Err(_) => {
                    let _ = app_handle_ws.emit(
                        "transcription-error",
                        "The sentence thread crashed. Whatever was typed before that stands."
                            .to_string(),
                    );
                    return;
                }
            };
            let text = outcome.text();
            let model_id = transcription_manager
                .lock()
                .ok()
                .and_then(|m| m.get_loaded_model_id().map(|s| s.to_string()))
                .unwrap_or_else(|| "none".to_string());
            let heard = outcome.audio.len() as f32 / 16_000.0;
            let saved_input = (!outcome.audio.is_empty()).then(|| save_model_input(&outcome.audio));
            eprintln!(
                "[{}] dictation: model={} audio={:.1}s sentences={} heard={:.1}s took={:.1}s chars={}",
                now(),
                model_id,
                audio_seconds,
                outcome.texts.len(),
                heard,
                outcome.took,
                text.chars().count()
            );
            if let Some(Some(path)) = saved_input {
                eprintln!("  what the model heard: {}", path.display());
            }
            let _ = app_handle_ws.emit(
                "dictation-stats",
                DictationStats {
                    model: model_id,
                    seconds: heard,
                    trimmed: (audio_seconds - heard).max(0.0),
                    shortened: 0.0,
                    level_before: speech_level(&all_audio),
                    level_after: speech_level(&outcome.audio),
                    gain: 1.0,
                    took: outcome.took,
                    chars: text.chars().count(),
                },
            );
            if text.is_empty() {
                if !armed {
                    let _ = app_handle_ws.emit(
                        "transcription-error",
                        format!(
                            "These {:.0} seconds held no speech, so nothing was typed.",
                            audio_seconds
                        ),
                    );
                }
                send_enter(false);
                return;
            }
            record_transcript(&app_handle_ws, &text);
            if outcome.typing_failed.is_none() {
                send_enter(true);
            }
            if let Some(e) = outcome.typing_failed {
                let message = match copy_to_clipboard(&app_handle_ws, &text) {
                    Ok(()) => format!(
                        "{} The whole text is on the clipboard - press {}+V to paste it.",
                        e,
                        if cfg!(target_os = "macos") { "Cmd" } else { "Ctrl" }
                    ),
                    Err(clip) => format!("{} The clipboard could not be used either ({}).", e, clip),
                };
                let _ = app_handle_ws.emit("transcription-error", message);
            }
            let _ = app_handle_ws.emit(
                "transcription",
                TranscriptionEvent {
                    text,
                    is_final: true,
                },
            );
            return;
        }

        // Transcribe all accumulated audio at once
        if !all_audio.is_empty() {
            // Transcription only starts once recording stops, so this time
            // is exactly how long the app looks frozen after pressing the
            // shortcut. Loading a model the first time is counted here too.
            let audio_seconds = all_audio.len() as f32 / 16_000.0;
            let (peak, rms, silence) = audio_stats(&all_audio);
            let level_before = speech_level(&all_audio);

            // Silence makes Whisper invent sentences, so it never reaches
            // the model. Silero decides from the shape of the sound.
            let skipped = match vad::judge(&all_audio, peak, level_before) {
                Heard::Speech(Some(seconds)) => {
                    eprintln!(
                        "speech detector: {:.1}s of speech in {:.1}s",
                        seconds, audio_seconds
                    );
                    None
                }
                Heard::Speech(None) => None,
                Heard::Dead => Some((
                    "the microphone delivered nothing",
                    format!(
                        "The microphone delivered nothing at all for these {:.0} seconds. \
                         Another app may be holding it (Teams, Zoom), or it is muted in the \
                         system's sound settings.",
                        audio_seconds
                    ),
                )),
                Heard::Silent => Some((
                    "no speech in it",
                    format!(
                        "These {:.0} seconds held no speech, so nothing was typed. If you \
                         were speaking, the microphone is barely reaching the app: turn up \
                         Boost under Settings > Microphone, or the input volume in the \
                         system's sound settings.",
                        audio_seconds
                    ),
                )),
            };
            if let Some((why, message)) = skipped {
                eprintln!(
                    "dictation: skipped {:.1}s, {} (speech={:.4} peak={:.3})",
                    audio_seconds, why, level_before, peak
                );
                if armed {
                    send_enter(false);
                    return;
                }
                let _ = app_handle_ws.emit(
                    "dictation-stats",
                    DictationStats {
                        model: "skipped - no speech".to_string(),
                        seconds: audio_seconds,
                        trimmed: 0.0,
                        shortened: 0.0,
                        level_before,
                        level_after: level_before,
                        gain: 1.0,
                        took: 0.0,
                        chars: 0,
                    },
                );
                let _ = app_handle_ws.emit("transcription-error", message);
                return;
            }

            // Raise the level first, so the trimming below sees a normal
            // recording whatever the microphone's gain was. Then drop the
            // quiet head and tail, shorten the long pauses left in the middle
            // if that is switched on, and save, so the saved file is exactly
            // what the model was given.
            let gain = boost_quiet_audio(&mut all_audio);
            let trimmed = trim_quiet_edges(&mut all_audio);
            let shortened = match shorten_pauses {
                Some(rules) => shorten_long_pauses(&mut all_audio, rules),
                None => 0.0,
            };
            let level_after = speech_level(&all_audio);
            let saved_input = save_model_input(&all_audio);

            let started = std::time::Instant::now();
            // Panicking here would skip "transcription-complete" and hang both windows.
            let (result, model_id) = match transcription_manager.lock() {
                Ok(mut manager) => {
                    let id = manager.get_loaded_model_id().unwrap_or("none").to_string();
                    let r = manager.transcribe(&all_audio, language.clone());
                    (r, id)
                }
                Err(e) => {
                    let _ = app_handle_ws.emit(
                        "transcription-error",
                        format!(
                            "The transcription engine is in a broken state after an earlier \
                             failure ({}). Restart Omegawhisper.",
                            e
                        ),
                    );
                    return;
                }
            };
            let took = started.elapsed().as_secs_f32();

            // The same numbers as the log line below, sent to the main
            // window, so a result can be judged without opening a log.
            let _ = app_handle_ws.emit(
                "dictation-stats",
                DictationStats {
                    model: model_id.clone(),
                    seconds: audio_seconds,
                    trimmed,
                    shortened,
                    level_before,
                    level_after,
                    gain,
                    took,
                    chars: result
                        .as_ref()
                        .map(|t| t.trim().chars().count())
                        .unwrap_or(0),
                },
            );

            // Everything that differs between a good result and a bad one,
            // on one line, so two dictations can be compared directly.
            eprintln!(
                "[{}] dictation: model={} audio={:.1}s \
                 peak={:.2} rms={:.3} silence={:.0}% trimmed={:.1}s pauses={:.1}s \
                 speech={:.4}->{:.4} gain={:.1}x took={:.1}s",
                now(),
                model_id,
                audio_seconds,
                peak,
                rms,
                silence * 100.0,
                trimmed,
                shortened,
                level_before,
                level_after,
                gain,
                took
            );
            if let Some(path) = saved_input {
                eprintln!("  what the model heard: {}", path.display());
            }

            // A microphone this quiet cannot be rescued by raising the
            // level, and the model will return little or nothing. Say so,
            // otherwise a long dictation just disappears with no reason
            // given. 0.02 is about a quarter of normal speech loudness.
            if level_after < 0.02 {
                let _ = app_handle_ws.emit(
                    "transcription-error",
                    format!(
                        "The microphone was almost silent for these {:.0} seconds, so some \
                         of the speech may have been lost. Turn up Boost under Settings > \
                         Microphone, or the input volume in the system's sound settings, and \
                         speak closer to the microphone.",
                        audio_seconds
                    ),
                );
            }

            match result {
                Ok(text) => {
                    let text = if tidy {
                        crate::live::tidy_end(text.trim())
                    } else {
                        text.trim().to_string()
                    };
                    if text.is_empty() {
                        eprintln!("Transcription returned no text; nothing to type.");
                    }
                    if !text.is_empty() {
                        eprintln!(
                            "Auto-typing {} characters: {:?}",
                            text.chars().count(),
                            text
                        );
                        // Kept before typing, so a dictation that lands in
                        // the wrong window can be fetched back from the tray.
                        record_transcript(&app_handle_ws, &text);
                        if let Err(e) = type_text_internal(&app_handle_ws, &format!("{} ", text)) {
                            eprintln!("Auto-type failed: {}", e);
                            // A minute of speech must never end up nowhere.
                            // Done here rather than in a window: the window
                            // that used to do it is normally hidden, and the
                            // browser refuses the clipboard without focus.
                            let message = match copy_to_clipboard(&app_handle_ws, &text) {
                                Ok(()) => format!(
                                    "{} The text is on the clipboard - press {}+V to paste it.",
                                    e,
                                    if cfg!(target_os = "macos") {
                                        "Cmd"
                                    } else {
                                        "Ctrl"
                                    }
                                ),
                                Err(clip) => {
                                    eprintln!("Clipboard also failed: {}", clip);
                                    format!("{} The clipboard could not be used either ({}), so the text is only in the Omegawhisper window.", e, clip)
                                }
                            };
                            let _ = app_handle_ws.emit("transcription-error", message);
                        }
                        let event = TranscriptionEvent {
                            text,
                            is_final: true,
                        };
                        let _ = app_handle_ws.emit("transcription", event);
                        send_enter(true);
                    }
                }
                Err(e) => {
                    let _ = app_handle_ws.emit("transcription-error", e);
                }
            }
        }

        // "transcription-complete" is sent by _complete when this thread ends.
    });

    // Newest microphone level and the most recent samples, written by the
    // capture callback and read by the thread below. The callback must not
    // wait on anything, so it only stores data here and never talks to the UI
    // itself, and never does the frequency maths.
    let mic_level = Arc::new(Mutex::new((0.0f32, 0.0f32)));
    let mic_recent = Arc::new(Mutex::new(Vec::<f32>::new()));

    // Everything the windows draw comes from here, 20 times a second. The
    // windows used to open the microphone themselves to draw with, which is
    // what made macOS wind the recording's gain up over the first seconds.
    {
        let mic_level = mic_level.clone();
        let mic_recent = mic_recent.clone();
        let is_recording = is_recording_arc.clone();
        let live_state = live_state.clone();
        let app_handle_level = app_handle.clone();
        thread::spawn(move || {
            let mut planner = rustfft::FftPlanner::<f32>::new();
            let fft = planner.plan_fft_forward(FFT_SIZE);
            let started = std::time::Instant::now();
            let mut tick = 0u32;
            let mut pitch = 0.0f32;
            // The loudest the microphone has been lately, falling slowly. The
            // bars are drawn from the sound scaled against it, so a quiet
            // microphone fills the window as a loud one does. The numbers
            // on the debug line stay raw: they are there to show the truth.
            let mut loudest = 0.0f32;

            while *is_recording.lock().unwrap() {
                let (peak, rms) = *mic_level.lock().unwrap();
                let recent = mic_recent.lock().unwrap().clone();

                loudest = (loudest * 0.98).max(peak).max(0.005);
                let display_gain = (0.8 / loudest).min(100.0);
                let scaled: Vec<f32> = recent.iter().map(|s| s * display_gain).collect();
                let bands = frequency_bands(&scaled, &fft, sample_rate);
                // Pitch costs more than the rest put together, so it runs 5
                // times a second rather than 20.
                if tick.is_multiple_of(4) {
                    pitch = detect_pitch(&scaled, sample_rate);
                }
                tick += 1;

                let live = *live_state.lock().unwrap();
                let _ = app_handle_level.emit(
                    "mic-level",
                    MicLevel {
                        peak,
                        rms,
                        seconds: started.elapsed().as_secs_f32(),
                        pitch,
                        bands,
                        pause: live.pause,
                        typing: live.typing,
                        sentences: live.sentences,
                        armed: live.armed,
                    },
                );
                thread::sleep(Duration::from_millis(50));
            }
        });
    }

    // Spawn audio recording thread
    let app_handle_audio = app_handle.clone();
    let is_recording_audio = is_recording_arc.clone();
    thread::spawn(move || {
        // The microphone would not open at all.
        let fail = |message: String| {
            report_microphone_failure(&app_handle_audio, &is_recording_audio, message);
        };
        // It opened, then died part-way through: unplugged, or taken by another
        // app. The sound from that moment on was never captured and cannot be
        // recovered, so stop and keep what was already recorded.
        let broken = {
            let app = app_handle_audio.clone();
            let flag = is_recording_audio.clone();
            move |err: String| {
                report_microphone_failure(
                    &app,
                    &flag,
                    format!(
                        "The microphone stopped during the recording ({}). Whatever was \
                         recorded before it stopped has been kept.",
                        err
                    ),
                );
            }
        };
        let targets = CaptureTargets {
            is_recording: is_recording_arc.clone(),
            recorded_samples: recorded_samples_arc.clone(),
            mic_level: mic_level.clone(),
            mic_recent: mic_recent.clone(),
            audio_tx: audio_tx.clone(),
            channels,
            boost: mic_boost,
        };
        let stream_result = match sample_format {
            SampleFormat::F32 => {
                build_capture_stream(&device, &stream_config, targets, broken, |s: f32| s)
            }
            SampleFormat::I16 => {
                build_capture_stream(&device, &stream_config, targets, broken, i16_to_f32)
            }
            SampleFormat::U16 => {
                build_capture_stream(&device, &stream_config, targets, broken, u16_to_f32)
            }
            _ => {
                fail(format!(
                    "This microphone sends audio in a format the app cannot read ({:?}).",
                    sample_format
                ));
                return;
            }
        };

        let stream = match stream_result {
            Ok(s) => s,
            Err(e) => {
                fail(format!(
                    "The microphone could not be opened: {}. Check that no other app \
                     is holding it, and that Omegawhisper is allowed in System \
                     Settings > Privacy & Security > Microphone.",
                    e
                ));
                return;
            }
        };

        if let Err(e) = stream.play() {
            fail(format!("The microphone opened but would not start: {}.", e));
            return;
        }

        // Keep stream alive until stop signal
        let _ = stop_rx.recv();
    });

    // On screen and audible only once the recording is really going, so a
    // failure above is never announced as a success.
    show_indicator(&app_handle);
    if !armed {
        play_chime(&START_CHIME);
    }

    Ok(())
}

#[tauri::command]
pub(crate) async fn stop_recording(app_handle: AppHandle) -> Result<(), String> {
    tokio::task::spawn_blocking(move || stop_recording_internal(&app_handle))
        .await
        .map_err(|e| format!("Task join error: {}", e))?
}

// Blocking, like its opposite: it waits for the last audio to arrive.
pub(crate) fn stop_recording_internal(app_handle: &AppHandle) -> Result<(), String> {
    let state = app_handle.state::<AudioState>();
    {
        let is_recording = state.is_recording.lock().unwrap();
        if !*is_recording {
            return Err("Not recording".to_string());
        }
    }

    eprintln!("[{}] recording stopped", now());

    // Stop recording
    *state.is_recording.lock().unwrap() = false;

    // Send stop signal
    let stop_tx = state.stop_signal.lock().unwrap().take();
    if let Some(stop_tx) = stop_tx {
        let _ = stop_tx.send(());
    }

    // Wait for buffers to flush.
    thread::sleep(Duration::from_millis(300));

    // Get recorded samples
    let samples = state.recorded_samples.lock().unwrap().clone();
    if samples.is_empty() {
        return Err("No audio data recorded".to_string());
    }

    // Convert to WAV
    let sample_rate = state.sample_rate.lock().unwrap().unwrap_or(48000);
    let wav_bytes = to_wav_bytes(&samples, sample_rate, 1);

    // Save to disk
    let recordings_dir = get_recordings_dir()?;
    let timestamp = Utc::now().format("%Y%m%d-%H%M%SUTC").to_string();
    let file_name = format!("{}.wav", timestamp);
    let file_path = recordings_dir.join(&file_name);

    fs::write(&file_path, &wav_bytes).map_err(|e| format!("Failed to save recording: {}", e))?;

    // Clear recorded samples
    *state.recorded_samples.lock().unwrap() = Vec::new();

    eprintln!("recording saved: {}", file_path.display());
    Ok(())
}

// One press of the dictation key. On its own thread because both halves block
// and this is called from the thread that handles the key, which must not wait.
pub(crate) fn toggle_recording(app: &AppHandle) {
    let app = app.clone();
    thread::spawn(move || {
        let recording = *app.state::<AudioState>().is_recording.lock().unwrap();
        let result = if recording {
            stop_recording_internal(&app)
        } else {
            start_recording_internal(&app, false)
        };
        // Said where it can be seen. A key that quietly does nothing is the
        // whole reason this moved out of a hidden window.
        if let Err(e) = result {
            eprintln!("dictation key: {}", e);
            let _ = app.emit("transcription-error", e);
        }
    });
}
