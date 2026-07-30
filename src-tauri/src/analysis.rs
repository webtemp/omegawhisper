// Plain maths on lists of numbers: how loud a recording is, what frequencies
// are in it, where the speech starts and stops. No microphone, no model.

use std::sync::{Arc, Mutex};

// Loudness of what was sent to the model: the loudest sample, the average
// level, and how much of it sits near silence. Quiet or mostly-silent audio
// is the usual reason a dictation comes back wrong or invented.
pub(crate) fn audio_stats(samples: &[f32]) -> (f32, f32, f32) {
    if samples.is_empty() {
        return (0.0, 0.0, 1.0);
    }
    let mut peak = 0.0f32;
    let mut sum_squares = 0.0f64;
    let mut quiet_samples = 0usize;
    for &s in samples {
        let level = s.abs();
        if level > peak {
            peak = level;
        }
        sum_squares += (s as f64) * (s as f64);
        // about -40 dBFS, below which speech is unlikely to be understood
        if level < 0.01 {
            quiet_samples += 1;
        }
    }
    let rms = (sum_squares / samples.len() as f64).sqrt() as f32;
    (peak, rms, quiet_samples as f32 / samples.len() as f32)
}

// How many samples the drawing maths looks at, and how many bars come out.
pub(crate) const FFT_SIZE: usize = 1024;

pub(crate) const BAND_COUNT: usize = 64;

// Loudness per frequency band, for the bars the windows draw.
//
// The bands are spaced logarithmically, so the voice range fills the width
// instead of being squeezed into the left edge, and the result is in decibels,
// because that is how loudness is heard. -90 dB comes out as 0 and -20 dB as 1.
pub(crate) fn frequency_bands(
    samples: &[f32],
    fft: &std::sync::Arc<dyn rustfft::Fft<f32>>,
) -> Vec<f32> {
    use rustfft::num_complex::Complex;

    if samples.len() < FFT_SIZE {
        return vec![0.0; BAND_COUNT];
    }
    let start = samples.len() - FFT_SIZE;

    // A Hann window: without it the ends of the slice act like a step change
    // and smear energy across every band.
    let mut buffer: Vec<Complex<f32>> = samples[start..]
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let w =
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (FFT_SIZE - 1) as f32).cos();
            Complex::new(s * w, 0.0)
        })
        .collect();
    fft.process(&mut buffer);

    let bins = FFT_SIZE / 2;
    let min_bin = 2.0f32;
    let max_bin = (bins as f32) * 0.5;

    (0..BAND_COUNT)
        .map(|i| {
            let pos = i as f32 / (BAND_COUNT - 1) as f32;
            let bin = (min_bin * (max_bin / min_bin).powf(pos)).round() as usize;
            let bin = bin.min(bins - 1);
            let magnitude = buffer[bin].norm() / (bins as f32);
            let db = 20.0 * (magnitude + 1e-9).log10();
            ((db + 90.0) / 70.0).clamp(0.0, 1.0)
        })
        .collect()
}

// Base frequency of the voice, found by looking for the shortest delay after
// which the wave repeats. 0 when the sound is too quiet or too noisy to tell -
// a wrong number is worse than none.
// Anything from 1.02 to 1.06 works; outside that it gets worse in one direction
// or the other. Do not raise it thinking bigger is safer.
pub(crate) const PITCH_MARGIN: f32 = 1.05;

pub(crate) fn detect_pitch(samples: &[f32], sample_rate: u32) -> f32 {
    let size = samples.len().min(2048);
    if size < 512 {
        return 0.0;
    }
    let window = &samples[samples.len() - size..];

    let energy: f32 = window.iter().map(|s| s * s).sum();
    let rms = (energy / size as f32).sqrt();
    if rms < 0.004 {
        return 0.0;
    }

    let min_lag = (sample_rate as f32 / 400.0) as usize; // highest voice
    let max_lag = ((sample_rate as f32 / 70.0) as usize).min(size - 1); // lowest voice
    let mut best_lag = 0usize;
    let mut best = 0.0f32;
    for lag in min_lag..=max_lag {
        let mut sum = 0.0f32;
        for i in 0..(size - lag) {
            sum += window[i] * window[i + lag];
        }
        let score = sum / (size - lag) as f32;
        // A delay of two or three repeats matches as well as one repeat, so a later
        // delay has to win clearly, not by rounding. Measured over 816 voices:
        // 1.0 is wrong 363 times, 1.05 is wrong 7, 1.1 is wrong 56 the other way.
        if score > best * PITCH_MARGIN {
            best = score;
            best_lag = lag;
        }
    }
    // The repeat has to be at least a third as strong as the sound itself.
    if best_lag == 0 || best < rms * rms * 0.33 {
        return 0.0;
    }
    sample_rate as f32 / best_lag as f32
}

// Keep the newest few thousand samples for the drawing maths. Called from the
// audio callback, so it does no more than a copy: enough for the frequency
// bands and for finding the pitch, and nothing older.
pub(crate) fn keep_recent(store: &Arc<Mutex<Vec<f32>>>, chunk: &[f32]) {
    const KEEP: usize = 4096;
    let Ok(mut recent) = store.lock() else { return };
    recent.extend_from_slice(chunk);
    if recent.len() > KEEP {
        let drop = recent.len() - KEEP;
        recent.drain(0..drop);
    }
}

// Loudest sample and average level of one chunk, for the live meter.
pub(crate) fn chunk_level(samples: &[f32]) -> (f32, f32) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let mut peak = 0.0f32;
    let mut sum_squares = 0.0f32;
    for &s in samples {
        peak = peak.max(s.abs());
        sum_squares += s * s;
    }
    (peak, (sum_squares / samples.len() as f32).sqrt())
}

pub(crate) fn i16_to_f32(sample: i16) -> f32 {
    sample as f32 / i16::MAX as f32
}

// Unsigned samples put silence in the middle of the range, so shift as well as scale.
pub(crate) fn u16_to_f32(sample: u16) -> f32 {
    (sample as f32 / u16::MAX as f32) * 2.0 - 1.0
}

// Stereo arrives as left, right, left, right.
pub(crate) fn mix_to_mono(buffer: Vec<f32>, channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return buffer;
    }
    let mut mono = Vec::with_capacity(buffer.len().div_ceil(channels as usize));
    for frame in buffer.chunks(channels as usize) {
        mono.push(frame.iter().sum::<f32>() / frame.len() as f32);
    }
    mono
}

// How loud the spoken parts are, ignoring pauses and one-off bangs.
//
// The recording is cut into 30 ms pieces, each piece's loudness is measured,
// and the value 10% from the top is returned. Pauses sit at the bottom and a
// single door slam sits in that top 10%, so neither decides the answer.
// Normal close speech lands around 0.08 on this scale.
pub(crate) fn speech_level(samples: &[f32]) -> f32 {
    const FRAME: usize = 480; // 30 ms at 16 kHz

    let mut levels: Vec<f32> = samples
        .chunks(FRAME)
        .filter(|c| c.len() == FRAME)
        .map(|c| {
            let sum: f32 = c.iter().map(|s| s * s).sum();
            (sum / c.len() as f32).sqrt()
        })
        .collect();

    if levels.is_empty() {
        return 0.0;
    }

    levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Clamp, not wrap: wrapping would return the quietest frame.
    let index = ((levels.len() as f32 * 0.9) as usize).min(levels.len() - 1);
    levels[index]
}

// Below these two, a recording holds no speech and must not be sent to the
// model. Measured on the microphone audio, before any level boost. Speech
// close to the microphone reaches a speech level of 0.08 and a peak near 0.5;
// an empty room measured 0.0014 and 0.02.
pub(crate) const MIN_SPEECH_LEVEL: f32 = 0.005;

pub(crate) const MIN_SPEECH_PEAK: f32 = 0.05;

// Whisper invents sentences from silence, and auto-type types them.
pub(crate) fn holds_speech(speech_level: f32, peak: f32) -> bool {
    speech_level >= MIN_SPEECH_LEVEL && peak >= MIN_SPEECH_PEAK
}

// Cut the quiet head and tail off, keeping everything in between.
//
// Pressing the shortcut and starting to speak takes a few seconds, and those
// seconds arrive as room noise. Whisper reads the recording as one block and
// a long quiet opening makes it lose the first words. Only the two ends are
// touched: a pause in the middle of a sentence is never cut, because cutting
// there would join words that were seconds apart.
//
// Returns how many seconds were removed from the front.
pub(crate) fn trim_quiet_edges(samples: &mut Vec<f32>) -> f32 {
    const FRAME: usize = 480; // 30 ms at 16 kHz
    const KEEP: usize = 16; // ~0.5 s of margin, so no word loses its start

    let level = speech_level(samples);
    if level <= 0.0 {
        return 0.0;
    }
    // A quarter of the speaking level: quiet enough to catch a soft word,
    // loud enough to ignore room noise.
    let threshold = (level * 0.25).max(0.004);

    let loud: Vec<bool> = samples
        .chunks(FRAME)
        .map(|c| {
            let sum: f32 = c.iter().map(|s| s * s).sum();
            (sum / c.len() as f32).sqrt() > threshold
        })
        .collect();

    let Some(first) = loud.iter().position(|&x| x) else {
        return 0.0;
    };
    let last = loud.iter().rposition(|&x| x).unwrap_or(loud.len() - 1);

    let start = first.saturating_sub(KEEP) * FRAME;
    let end = ((last + KEEP + 1) * FRAME).min(samples.len());
    if start >= end {
        return 0.0;
    }

    let cut_seconds = start as f32 / 16_000.0;
    *samples = samples[start..end].to_vec();
    cut_seconds
}

// How much of a shortened pause is left behind. Whisper decides where a
// sentence ends from the pauses it hears, so the gap has to survive - only
// its length changes.
const KEEP_PAUSE_FRAMES: usize = 10; // ~0.3 s, split evenly across the join

// Samples blended across the join. A hard cut leaves a step in the wave, which
// is heard as a click and is exactly the kind of thing that confuses the model.
const JOIN_FADE: usize = 80; // 5 ms at 16 kHz

// What Settings says pause-shortening should do.
#[derive(Clone, Copy)]
pub(crate) struct PauseRules {
    // A pause has to last at least this long before any of it is removed.
    //
    // Worth keeping high. Over 32 real recordings a 1 s limit shortened 14 of
    // them and saved 8%; 2.2 s shortens 3 and saves 4%. The 11 extra were all
    // ordinary breathing between sentences, which is what Whisper uses to
    // decide where the full stops go.
    pub(crate) cutoff_ms: u32,
    // Leave this long after the first spoken word alone, or None to allow
    // cutting anywhere. Whisper reads the recording as one block and settles
    // on a language and a writing style from the first words it hears, and
    // the rest of the text follows that decision.
    pub(crate) protect_opening_ms: Option<u32>,
}

// Where the long pauses are, as (first quiet frame, first loud frame after it).
//
// Only stretches with speech on both sides count, so the quiet head and tail
// are never in here - those belong to trim_quiet_edges.
pub(crate) fn long_pauses(samples: &[f32], rules: PauseRules) -> Vec<(usize, usize)> {
    const FRAME: usize = 480; // 30 ms at 16 kHz

    // Never shorter than what is kept: below that the two halves of a pause
    // would overlap and the copying further down would read past its end.
    let min_pause_frames = (rules.cutoff_ms as usize / 30).max(KEEP_PAUSE_FRAMES + 1);

    let level = speech_level(samples);
    if level <= 0.0 {
        return Vec::new();
    }
    // The same bar trim_quiet_edges uses: a quarter of how loudly this person
    // spoke, so it works on any microphone in any room.
    let threshold = (level * 0.25).max(0.004);

    let loud: Vec<bool> = samples
        .chunks(FRAME)
        .map(|c| {
            let sum: f32 = c.iter().map(|s| s * s).sum();
            (sum / c.len() as f32).sqrt() > threshold
        })
        .collect();

    let Some(first) = loud.iter().position(|&x| x) else {
        return Vec::new();
    };
    let last = loud.iter().rposition(|&x| x).unwrap_or(first);

    // The opening is counted from the first spoken word, not from the start of
    // the file, so a long walk to the microphone does not use it up.
    let opening_ends = match rules.protect_opening_ms {
        Some(ms) => first + ms as usize / 30,
        None => 0,
    };

    let mut pauses: Vec<(usize, usize)> = Vec::new();
    let mut run_start: Option<usize> = None;
    for (frame, &is_loud) in loud.iter().enumerate().take(last + 1).skip(first) {
        if is_loud {
            if let Some(start) = run_start.take() {
                if frame - start >= min_pause_frames && start >= opening_ends {
                    pauses.push((start, frame));
                }
            }
        } else if run_start.is_none() {
            run_start = Some(frame);
        }
    }
    pauses
}

// Shorten the long pauses in the middle of a recording, without removing them.
//
// Thinking for three seconds mid-sentence means three seconds the model has to
// read through. This keeps about 0.3 s of that pause - enough for Whisper to
// still hear a sentence break - and drops the rest.
//
// Runs over the finished recording, never a live stream, so every frame is a
// full frame and the speaking level is already known. The old voice detection
// worked the other way and padded half-frames with zeros, which turned one
// second of speech into three.
//
// Returns how many seconds were removed.
pub(crate) fn shorten_long_pauses(samples: &mut Vec<f32>, rules: PauseRules) -> f32 {
    const FRAME: usize = 480; // 30 ms at 16 kHz
    const HALF_KEEP: usize = KEEP_PAUSE_FRAMES / 2;

    let pauses = long_pauses(samples, rules);
    if pauses.is_empty() {
        return 0.0;
    }

    let mut out: Vec<f32> = Vec::with_capacity(samples.len());
    let mut copied = 0usize;
    for (start, end) in pauses {
        let head_end = (start + HALF_KEEP) * FRAME;
        let tail_start = (end - HALF_KEEP) * FRAME;
        out.extend_from_slice(&samples[copied..head_end]);

        // Fade one side into the other across the join. Both sides are room
        // noise from this same recording, so the result stays room noise.
        let joined = out.len();
        for i in 0..JOIN_FADE {
            let mix = (i + 1) as f32 / (JOIN_FADE + 1) as f32;
            let from = out[joined - JOIN_FADE + i];
            out[joined - JOIN_FADE + i] = from * (1.0 - mix) + samples[tail_start + i] * mix;
        }
        copied = tail_start + JOIN_FADE;
    }
    out.extend_from_slice(&samples[copied..]);

    let removed = (samples.len() - out.len()) as f32 / 16_000.0;
    *samples = out;
    removed
}

// Bring quiet recordings up to a level the model can work with.
//
// Whisper reads audio in 30 second windows and drops a whole window when it
// is not sure the window holds speech. A quiet microphone makes that happen
// again and again, so a long dictation comes back as a few sentences or as
// nothing. Scaling the whole recording up avoids it. The saved microphone
// recording is untouched; only what goes to the model is changed.
//
// The gain is chosen from the speech level, not the loudest sample, so long
// pauses do not shrink it and one loud bang does not cancel it. The peak
// still caps the gain, so nothing clips.
//
// Returns the gain applied, 1.0 meaning the audio was already loud enough.
pub(crate) fn boost_quiet_audio(samples: &mut [f32]) -> f32 {
    // Loudness of normal speech close to the microphone.
    const TARGET_LEVEL: f32 = 0.08;
    // Leave headroom so the loudest sample does not hit the ceiling.
    const MAX_PEAK: f32 = 0.95;
    // Past this the recording is being turned into something it was not.
    // 40x used to be allowed, and it raised an empty room to speech loudness,
    // which Whisper then read as sentences in Dutch.
    const MAX_GAIN: f32 = 10.0;

    let level = speech_level(samples);
    let peak = samples.iter().fold(0.0f32, |max, s| max.max(s.abs()));
    if level < MIN_SPEECH_LEVEL || peak <= 0.0 {
        return 1.0;
    }

    let gain = (TARGET_LEVEL / level).min(MAX_PEAK / peak).min(MAX_GAIN);
    if gain <= 1.0 {
        return 1.0;
    }

    for s in samples.iter_mut() {
        *s = (*s * gain).clamp(-1.0, 1.0);
    }
    gain
}

pub(crate) fn to_wav_bytes(samples: &[f32], sample_rate: u32, channels: u16) -> Vec<u8> {
    let mut wav_data = Vec::new();

    let bytes_per_sample: u16 = 2; // 16-bit
    let byte_rate = sample_rate * channels as u32 * bytes_per_sample as u32;
    let block_align: u16 = channels * bytes_per_sample;
    let data_size = samples.len() * bytes_per_sample as usize;
    let file_size = 36 + data_size as u32;

    // RIFF header
    wav_data.extend_from_slice(b"RIFF");
    wav_data.extend_from_slice(&file_size.to_le_bytes());
    wav_data.extend_from_slice(b"WAVE");

    // fmt chunk
    wav_data.extend_from_slice(b"fmt ");
    wav_data.extend_from_slice(&16u32.to_le_bytes());
    wav_data.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav_data.extend_from_slice(&channels.to_le_bytes());
    wav_data.extend_from_slice(&sample_rate.to_le_bytes());
    wav_data.extend_from_slice(&byte_rate.to_le_bytes());
    wav_data.extend_from_slice(&block_align.to_le_bytes());
    wav_data.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

    // data chunk
    wav_data.extend_from_slice(b"data");
    wav_data.extend_from_slice(&(data_size as u32).to_le_bytes());

    // Convert f32 samples to i16
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let i16_sample = (clamped * i16::MAX as f32) as i16;
        wav_data.extend_from_slice(&i16_sample.to_le_bytes());
    }

    wav_data
}
