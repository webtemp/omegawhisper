// Whether a recording holds speech, decided by Silero VAD from the shape of
// the sound rather than its loudness, so a quiet microphone is judged like a
// loud one. The loudness gate it replaced threw away every word from a
// headset at low gain.

use crate::analysis;
use std::fs;
use std::path::PathBuf;
use transcribe_rs::vad::{SileroVad, Vad};

// Compiled in: 1.8 MB, nothing to download.
const MODEL: &[u8] = include_bytes!("../vad/silero_vad_v4.onnx");
const MODEL_FILE: &str = "silero_vad_v4.onnx";

// 30 ms at 16 kHz, the only frame the model accepts.
pub(crate) const FRAME: usize = 480;
const THRESHOLD: f32 = 0.5;
// The model sees a copy scaled to this peak, never the microphone's own level.
pub(crate) const NORMALIZED_PEAK: f32 = 0.9;

// Below this peak nothing came through the microphone.
pub(crate) const DEAD_PEAK: f32 = 0.001;
// A short word. Stray frames flagged in noise never add up to it.
pub(crate) const MIN_SPEECH_SECONDS: f32 = 0.3;

#[derive(Debug, PartialEq)]
pub(crate) enum Heard {
    // Seconds of speech, or None when the loudness fallback decided.
    Speech(Option<f32>),
    Silent,
    Dead,
}

pub(crate) fn seconds_of(frames: usize) -> f32 {
    frames as f32 * FRAME as f32 / 16_000.0
}

// The runtime reads the model from a file, so the compiled-in copy is written
// next to the other models once.
pub(crate) fn model_path() -> Result<PathBuf, String> {
    let dir = dirs::data_local_dir()
        .ok_or_else(|| "Could not find local data directory".to_string())?
        .join("omegawhisper")
        .join("models");
    fs::create_dir_all(&dir).map_err(|e| format!("Could not create {}: {}", dir.display(), e))?;
    let path = dir.join(MODEL_FILE);
    let present = fs::metadata(&path).is_ok_and(|m| m.len() == MODEL.len() as u64);
    if !present {
        fs::write(&path, MODEL)
            .map_err(|e| format!("Could not write {}: {}", path.display(), e))?;
    }
    Ok(path)
}

pub(crate) fn normalized(samples: &[f32]) -> Vec<f32> {
    let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    if peak <= 0.0 {
        return samples.to_vec();
    }
    let gain = NORMALIZED_PEAK / peak;
    samples.iter().map(|s| s * gain).collect()
}

// Seconds of speech the model finds in 16 kHz mono audio.
pub(crate) fn speech_seconds(samples: &[f32]) -> Result<f32, String> {
    let mut vad = SileroVad::new(model_path()?, THRESHOLD)
        .map_err(|e| format!("The speech detector could not be loaded: {}", e))?;
    let mut frames = 0usize;
    for frame in normalized(samples).chunks_exact(FRAME) {
        if vad.is_speech(frame).map_err(|e| e.to_string())? {
            frames += 1;
        }
    }
    Ok(seconds_of(frames))
}

// The detector as the live splitter uses it, one frame at a time, with
// hysteresis so a soft syllable does not read as a pause.
pub(crate) struct Silero {
    vad: SileroVad,
    in_speech: bool,
}

impl Silero {
    pub(crate) fn new() -> Result<Self, String> {
        let vad = SileroVad::new(model_path()?, THRESHOLD)
            .map_err(|e| format!("The speech detector could not be loaded: {}", e))?;
        Ok(Self {
            vad,
            in_speech: false,
        })
    }
}

impl crate::live::SpeechDetector for Silero {
    fn is_speech(&mut self, frame: &[f32]) -> bool {
        let probability = self.vad.speech_probability(frame).unwrap_or(0.0);
        self.in_speech = crate::live::still_speech(self.in_speech, probability);
        self.in_speech
    }
}

pub(crate) fn holds_speech(speech_seconds: f32) -> bool {
    speech_seconds >= MIN_SPEECH_SECONDS
}

// What to do with a finished recording. The loudness check only decides if
// the model cannot be loaded.
pub(crate) fn judge(samples: &[f32], peak: f32, speech_level: f32) -> Heard {
    if peak < DEAD_PEAK {
        return Heard::Dead;
    }
    match speech_seconds(samples) {
        Ok(seconds) if holds_speech(seconds) => Heard::Speech(Some(seconds)),
        Ok(_) => Heard::Silent,
        Err(e) => {
            eprintln!("{} Falling back to the loudness check.", e);
            if analysis::holds_speech(speech_level, peak) {
                Heard::Speech(None)
            } else {
                Heard::Silent
            }
        }
    }
}
