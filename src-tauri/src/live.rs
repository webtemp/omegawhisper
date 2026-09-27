// Typing while you talk: the recording is cut into sentences at the pauses
// between them, and each one goes to the model and into the focused app while
// the next is still being spoken. Also where the recording notices that the
// speaker has stopped.

use crate::analysis::{boost_quiet_audio, trim_quiet_edges};
use crate::managers::TranscriptionManager;
use crate::typing::type_text_now;
use crate::vad::{seconds_of, FRAME, MIN_SPEECH_SECONDS, NORMALIZED_PEAK};
use crate::{now, TranscriptionEvent};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter};

pub(crate) trait SpeechDetector {
    fn is_speech(&mut self, frame: &[f32]) -> bool;
}

// Speech starts above the first and lasts while above the second.
pub(crate) const SPEECH_START: f32 = 0.5;
pub(crate) const SPEECH_HOLD: f32 = 0.3;

pub(crate) fn still_speech(in_speech: bool, probability: f32) -> bool {
    probability >= if in_speech { SPEECH_HOLD } else { SPEECH_START }
}

// What the indicator shows about live typing, written here and read by the
// thread that sends the microphone numbers.
#[derive(Clone, Copy, Default)]
pub(crate) struct LiveState {
    // How far the current pause is towards the cut, 0 to 1.
    pub(crate) pause: f32,
    pub(crate) typing: bool,
    pub(crate) sentences: u32,
    pub(crate) armed: bool,
}

pub(crate) type SharedLiveState = Arc<Mutex<LiveState>>;

// Kept in front of the first word of a sentence, so it does not start cut.
const LEAD_IN_FRAMES: usize = 16;
// Kept after the last word, so it does not end cut either.
const TAIL_FRAMES: usize = 10;
// The adaptive silence stop never goes under this.
pub(crate) const STOP_FLOOR_SECONDS: f32 = 2.0;
// Below this the running peak is not trusted: the detector would be fed
// amplified noise before the first word.
const PEAK_FLOOR: f32 = 0.03;

pub(crate) struct Segmenter<D: SpeechDetector> {
    detector: D,
    // Samples short of a whole frame, waiting for the next chunk.
    partial: Vec<f32>,
    // Audio since the last sentence went out.
    pending: Vec<f32>,
    // Frames of speech in pending.
    speech_frames: usize,
    // Frames at the end of pending since its last speech frame.
    trailing: usize,
    // Frames since the last speech frame anywhere in the recording.
    silence_run: usize,
    heard_speech: bool,
    // The longest pause between words so far, in frames.
    longest_pause: usize,
    peak: f32,
    pause_frames: usize,
}

impl<D: SpeechDetector> Segmenter<D> {
    pub(crate) fn new(detector: D, pause_ms: u32) -> Self {
        Self {
            detector,
            partial: Vec::new(),
            pending: Vec::new(),
            speech_frames: 0,
            trailing: 0,
            silence_run: 0,
            heard_speech: false,
            longest_pause: 0,
            peak: 0.0,
            pause_frames: (pause_ms as usize * 16_000 / 1000).div_ceil(FRAME).max(1),
        }
    }

    // Take in 16 kHz audio. Returns every sentence finished by it, in order.
    pub(crate) fn feed(&mut self, samples: &[f32]) -> Vec<Vec<f32>> {
        let mut sentences = Vec::new();
        self.partial.extend_from_slice(samples);
        let whole = self.partial.len() / FRAME * FRAME;
        let frames: Vec<f32> = self.partial.drain(..whole).collect();
        for frame in frames.as_chunks::<FRAME>().0 {
            self.peak = self.peak.max(frame.iter().fold(0.0f32, |m, s| m.max(s.abs())));
            let gain = NORMALIZED_PEAK / self.peak.max(PEAK_FLOOR);
            let scaled: Vec<f32> = frame.iter().map(|s| s * gain).collect();
            let speech = self.detector.is_speech(&scaled);
            self.pending.extend_from_slice(frame);
            if speech {
                if self.heard_speech {
                    self.longest_pause = self.longest_pause.max(self.silence_run);
                }
                self.heard_speech = true;
                self.speech_frames += 1;
                self.trailing = 0;
                self.silence_run = 0;
            } else {
                self.trailing += 1;
                self.silence_run += 1;
            }
            if self.speech_frames == 0 {
                self.keep_lead_in();
            } else if self.trailing >= self.pause_frames {
                if let Some(sentence) = self.take_sentence() {
                    sentences.push(sentence);
                }
            }
        }
        sentences
    }

    // Nothing said yet: keep only a short run-up to the first word.
    fn keep_lead_in(&mut self) {
        let keep = LEAD_IN_FRAMES * FRAME;
        if self.pending.len() > keep {
            let extra = self.pending.len() - keep;
            self.pending.drain(..extra);
        }
        self.trailing = self.trailing.min(LEAD_IN_FRAMES);
    }

    // The pending audio up to a little after its last word. The rest of the
    // pause stays behind as the run-up to the next sentence.
    fn take_sentence(&mut self) -> Option<Vec<f32>> {
        let drop = self.trailing.saturating_sub(TAIL_FRAMES) * FRAME;
        let end = self.pending.len().saturating_sub(drop);
        let sentence: Vec<f32> = self.pending.drain(..end).collect();
        let long_enough = seconds_of(self.speech_frames) >= MIN_SPEECH_SECONDS;
        self.speech_frames = 0;
        self.trailing = self.pending.len() / FRAME;
        self.keep_lead_in();
        long_enough.then_some(sentence)
    }

    // The recording is over: whatever was said since the last cut.
    pub(crate) fn finish(mut self) -> Option<Vec<f32>> {
        self.pending.append(&mut self.partial);
        (seconds_of(self.speech_frames) >= MIN_SPEECH_SECONDS).then_some(self.pending)
    }

    pub(crate) fn silence_seconds(&self) -> f32 {
        seconds_of(self.silence_run)
    }

    pub(crate) fn heard_speech(&self) -> bool {
        self.heard_speech
    }

    // Silence that ends the dictation: twice the longest pause so far, never
    // under the floor, never over the setting.
    pub(crate) fn stop_after_seconds(&self, max_seconds: f32) -> f32 {
        (2.0 * seconds_of(self.longest_pause)).clamp(STOP_FLOOR_SECONDS.min(max_seconds), max_seconds)
    }

    // 0 until something was said, then how far the pause is towards the cut.
    pub(crate) fn pause_progress(&self) -> f32 {
        if self.speech_frames == 0 {
            0.0
        } else {
            (self.trailing as f32 / self.pause_frames as f32).min(1.0)
        }
    }
}

// Hesitations the model writes out when a piece ends mid-thought.
const FILLERS: [&str; 7] = ["uh", "um", "umm", "hmm", "erm", "ah", "eh"];

// A piece cut at a hesitation comes back as "and then, uh, uh..." Drop the
// trailing dots, the dangling fillers and the comma left behind.
pub(crate) fn tidy_end(text: &str) -> String {
    let mut text = text.trim_end().to_string();
    loop {
        let before = text.len();
        for dots in ["...", "…"] {
            if let Some(rest) = text.strip_suffix(dots) {
                text = rest.trim_end().to_string();
            }
        }
        while let Some(rest) = text.strip_suffix(',').or_else(|| text.strip_suffix(';')) {
            text = rest.trim_end().to_string();
        }
        let last = text
            .rsplit(|c: char| c.is_whitespace() || c == ',')
            .next()
            .unwrap_or("")
            .trim_end_matches(['.', '!', '?'])
            .to_lowercase();
        if !last.is_empty() && FILLERS.contains(&last.as_str()) {
            let cut = text.len() - text.rsplit(|c: char| c.is_whitespace() || c == ',').next().unwrap_or("").len();
            text.truncate(cut);
            text = text.trim_end().to_string();
        }
        if text.len() == before {
            return text;
        }
    }
}

// What the sentence thread did, for the log line and the history.
#[derive(Default)]
pub(crate) struct LiveOutcome {
    pub(crate) texts: Vec<String>,
    // Every sentence's audio as the model heard it, in order.
    pub(crate) audio: Vec<f32>,
    pub(crate) took: f32,
    pub(crate) typing_failed: Option<String>,
}

impl LiveOutcome {
    pub(crate) fn text(&self) -> String {
        self.texts.join(" ")
    }
}

// Runs until the sender is dropped: each sentence through the model and into
// the focused app, in the order they were spoken.
pub(crate) fn type_sentences(
    app: AppHandle,
    manager: Arc<Mutex<TranscriptionManager>>,
    language: Option<String>,
    sentences: Receiver<Vec<f32>>,
    live_state: SharedLiveState,
    tidy: bool,
) -> LiveOutcome {
    let mut outcome = LiveOutcome::default();
    for (n, mut audio) in sentences.iter().enumerate() {
        let gain = boost_quiet_audio(&mut audio);
        trim_quiet_edges(&mut audio);
        let seconds = audio.len() as f32 / 16_000.0;
        let started = std::time::Instant::now();
        let result = match manager.lock() {
            Ok(mut manager) => manager.transcribe(&audio, language.clone()),
            Err(e) => Err(format!("The transcription engine is in a broken state ({})", e)),
        };
        let took = started.elapsed().as_secs_f32();
        outcome.took += took;
        outcome.audio.extend(audio);
        let text = match result {
            Ok(text) if tidy => tidy_end(text.trim()),
            Ok(text) => text.trim().to_string(),
            Err(e) => {
                eprintln!("sentence {}: {}", n + 1, e);
                let _ = app.emit("transcription-error", e);
                continue;
            }
        };
        eprintln!(
            "[{}] sentence {}: audio={:.1}s gain={:.1}x took={:.1}s chars={}: {:?}",
            now(),
            n + 1,
            seconds,
            gain,
            took,
            text.chars().count(),
            text
        );
        if text.is_empty() {
            continue;
        }
        live_state.lock().unwrap().typing = true;
        if let Err(e) = type_text_now(&app, &format!("{} ", text)) {
            eprintln!("Auto-type failed: {}", e);
            outcome.typing_failed.get_or_insert(e);
        }
        {
            let mut state = live_state.lock().unwrap();
            state.typing = false;
            state.sentences += 1;
        }
        outcome.texts.push(text);
        let _ = app.emit(
            "transcription",
            TranscriptionEvent {
                text: outcome.text(),
                is_final: false,
            },
        );
    }
    outcome
}
