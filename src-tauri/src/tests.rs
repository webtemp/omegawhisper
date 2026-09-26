// Tests for the maths the app depends on: turning what the microphone sends
// into numbers, finding the speech in a recording, and the settings file.
// No microphone, no model, no windows.
use crate::analysis::*;
use crate::chime::*;
use crate::history::*;
use crate::live::*;
use crate::managers::transcription::{needs_load, onnx_accelerator, GpuChoice};
use crate::settings::*;
use crate::storage::delete_recordings_in;
use crate::*;
use std::fs;

fn near(got: f32, want: f32, tolerance: f32, what: &str) {
    assert!(
        (got - want).abs() <= tolerance,
        "{what}: got {got}, wanted {want} (allowed {tolerance})"
    );
}

// Blocks of 480 samples (30 ms at 16 kHz), alternating +/- amplitude so the
// measured loudness is exactly that amplitude.
fn blocks(spec: &[(usize, f32)]) -> Vec<f32> {
    let mut out = Vec::new();
    for &(count, amplitude) in spec {
        for _ in 0..count {
            for i in 0..480 {
                out.push(if i % 2 == 0 { amplitude } else { -amplitude });
            }
        }
    }
    out
}

fn sine(count: usize, hz: f32, sample_rate: f32, amplitude: f32) -> Vec<f32> {
    (0..count)
        .map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / sample_rate).sin() * amplitude)
        .collect()
}

// Repeatable stand-in for noise, so a failure can be reproduced.
fn noise(count: usize, amplitude: f32) -> Vec<f32> {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = (state >> 32) as f32 / u32::MAX as f32;
            (unit * 2.0 - 1.0) * amplitude
        })
        .collect()
}

fn peak_of(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0f32, |max, s| max.max(s.abs()))
}

// ---- turning what the device sends into numbers between -1 and 1 -------

#[test]
fn sixteen_bit_signed_samples_convert() {
    // (raw value from the device, expected, what this case is for)
    let cases: &[(i16, f32, &str)] = &[
        (0, 0.0, "silence sits at zero"),
        (i16::MAX, 1.0, "the loudest positive value reaches 1.0"),
        (-i16::MAX, -1.0, "the loudest negative value reaches -1.0"),
        (16383, 0.5, "half loudness"),
        (-16383, -0.5, "half loudness, negative"),
    ];
    for &(raw, want, what) in cases {
        near(i16_to_f32(raw), want, 0.0001, what);
    }
    // One value below the loudest negative. It is allowed to land slightly
    // past -1.0; the check that follows every boost keeps it in range.
    assert!(i16_to_f32(i16::MIN) >= -1.001);
}

#[test]
fn sixteen_bit_unsigned_samples_convert() {
    let cases: &[(u16, f32, &str)] = &[
        (32768, 0.0, "silence sits in the middle, not at zero"),
        (0, -1.0, "the bottom of the range is the loudest negative"),
        (
            u16::MAX,
            1.0,
            "the top of the range is the loudest positive",
        ),
        (49151, 0.5, "half loudness"),
        (16384, -0.5, "half loudness, negative"),
    ];
    for &(raw, want, what) in cases {
        near(u16_to_f32(raw), want, 0.0001, what);
    }
}

#[test]
fn silence_from_an_unsigned_device_stays_silent() {
    // Scaling without shifting would turn a silent room into a steady tone.
    let quiet: Vec<f32> = vec![32768u16; 4800].into_iter().map(u16_to_f32).collect();
    near(peak_of(&quiet), 0.0, 0.0001, "silence must stay silent");
    assert!(!holds_speech(speech_level(&quiet), peak_of(&quiet)));
}

// ---- mixing several channels down to one ------------------------------

#[test]
fn channels_are_averaged_into_one() {
    // (channels, what the device sent, what should come out, why)
    let cases: &[(u16, &[f32], &[f32], &str)] = &[
        (
            1,
            &[0.1, 0.2, 0.3],
            &[0.1, 0.2, 0.3],
            "one channel passes through",
        ),
        (
            0,
            &[0.1, 0.2],
            &[0.1, 0.2],
            "a nonsense channel count changes nothing",
        ),
        (
            2,
            &[1.0, 0.0, 0.5, 0.5],
            &[0.5, 0.5],
            "two channels are averaged",
        ),
        (
            2,
            &[1.0, -1.0],
            &[0.0],
            "opposite channels cancel, they are not just dropped",
        ),
        (3, &[0.3, 0.6, 0.9], &[0.6], "three channels are averaged"),
        (
            2,
            &[1.0, 0.0, 1.0],
            &[0.5, 1.0],
            "a half-finished frame at the end is kept",
        ),
        (2, &[], &[], "nothing in, nothing out"),
    ];
    for &(channels, input, want, what) in cases {
        let got = mix_to_mono(input.to_vec(), channels);
        assert_eq!(got.len(), want.len(), "{what}: wrong length");
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            near(g, w, 0.0001, &format!("{what}, sample {i}"));
        }
    }
}

// ---- how loud is the speech -------------------------------------------

#[test]
fn speech_loudness_ignores_pauses_and_single_bangs() {
    // (recording, expected loudness, why this case exists)
    let cases: &[(Vec<f32>, f32, &str)] = &[
        (Vec::new(), 0.0, "nothing recorded"),
        (vec![0.5; 100], 0.0, "shorter than one 30 ms block"),
        (
            blocks(&[(1, 0.3)]),
            0.3,
            "exactly one block reports its own loudness",
        ),
        (
            blocks(&[(50, 0.08)]),
            0.08,
            "steady speech reports its loudness",
        ),
        (
            blocks(&[(99, 0.002), (1, 0.9)]),
            0.002,
            "one door slam in a quiet minute does not count as speech",
        ),
        (
            blocks(&[(80, 0.002), (20, 0.08)]),
            0.08,
            "real speech among pauses does count",
        ),
    ];
    for (recording, want, what) in cases {
        near(speech_level(recording), *want, 0.0001, what);
    }
}

#[test]
fn a_single_block_does_not_wrap_to_the_quietest_one() {
    near(
        speech_level(&blocks(&[(1, 0.42)])),
        0.42,
        0.0001,
        "one block",
    );
}

// ---- the rule that silence must never reach the model -----------------

#[test]
fn only_real_speech_is_sent_to_the_model() {
    // (speech loudness, loudest sample, may it be sent, why)
    let cases: &[(f32, f32, bool, &str)] = &[
        (0.0, 0.0, false, "digital silence"),
        (
            0.0014,
            0.02,
            false,
            "an empty room, as measured on this Mac",
        ),
        (
            0.004,
            0.60,
            false,
            "one click: loud peak, but nothing is being said",
        ),
        (0.08, 0.02, false, "a steady hum: high average, no peaks"),
        (0.005, 0.05, true, "exactly on both thresholds"),
        (0.08, 0.50, true, "normal speech close to the microphone"),
        (0.30, 0.99, true, "loud speech"),
    ];
    for &(level, peak, want, what) in cases {
        assert_eq!(holds_speech(level, peak), want, "{what}");
    }
}

// ---- raising the level of a quiet recording ---------------------------

#[test]
fn quiet_recordings_are_raised_to_normal_however_quiet() {
    // (recording, expected gain, why)
    // A gain of 1.0 means the recording was left exactly as it was. The
    // speech detector runs before this, so an empty room never gets here.
    let cases: &[(Vec<f32>, f32, &str)] = &[
        (Vec::new(), 1.0, "nothing recorded"),
        (
            blocks(&[(10, 0.0)]),
            1.0,
            "digital silence cannot be raised",
        ),
        (
            blocks(&[(10, 0.002)]),
            40.0,
            "a headset at low gain is brought up to normal",
        ),
        (
            blocks(&[(10, 0.02)]),
            4.0,
            "quiet speech is brought up to normal",
        ),
        (
            blocks(&[(10, 0.0005)]),
            100.0,
            "the gain stops at a hundred times",
        ),
        (
            blocks(&[(10, 0.2)]),
            1.0,
            "speech that is already loud is left alone",
        ),
    ];
    for (recording, want, what) in cases {
        let mut audio = recording.clone();
        near(boost_quiet_audio(&mut audio), *want, 0.01, what);
    }
}

#[test]
fn raising_the_level_never_pushes_a_sample_past_the_limit() {
    let recordings: &[(Vec<f32>, &str)] = &[
        (blocks(&[(10, 0.006)]), "very quiet speech"),
        (blocks(&[(10, 0.02)]), "quiet speech"),
        (
            blocks(&[(99, 0.006), (1, 0.9)]),
            "quiet speech with one loud bang in it",
        ),
        (sine(48_000, 220.0, 16_000.0, 0.01), "a quiet steady tone"),
        (noise(48_000, 0.01), "quiet noise"),
    ];
    for (recording, what) in recordings {
        let mut audio = recording.clone();
        boost_quiet_audio(&mut audio);
        let peak = peak_of(&audio);
        assert!(peak <= 1.0, "{what}: loudest sample reached {peak}");
    }
}

#[test]
fn the_level_after_raising_matches_the_gain_reported() {
    let mut audio = blocks(&[(10, 0.02)]);
    let before = speech_level(&audio);
    let gain = boost_quiet_audio(&mut audio);
    let after = speech_level(&audio);
    near(
        after,
        before * gain,
        0.001,
        "reported gain matches the result",
    );
    near(
        after,
        0.08,
        0.001,
        "quiet speech ends up at normal speech loudness",
    );
}

// ---- cutting the quiet start and end off ------------------------------

#[test]
fn only_the_quiet_start_and_end_are_cut() {
    // 40 blocks of silence, 40 of speech, 40 of silence.
    let mut audio = blocks(&[(40, 0.0), (40, 0.1), (40, 0.0)]);
    let speech_samples_before = audio.iter().filter(|s| s.abs() > 0.05).count();

    let cut = trim_quiet_edges(&mut audio);

    near(cut, 0.72, 0.01, "seconds removed from the front");
    assert_eq!(audio.len(), 34_560, "length after trimming");
    assert_eq!(
        audio.iter().filter(|s| s.abs() > 0.05).count(),
        speech_samples_before,
        "every spoken sample survived the trim"
    );
}

#[test]
fn a_pause_in_the_middle_of_a_sentence_is_never_cut() {
    // Cutting here would join words that were seconds apart.
    let mut audio = blocks(&[(20, 0.1), (40, 0.0), (20, 0.1)]);
    let length_before = audio.len();

    let cut = trim_quiet_edges(&mut audio);

    near(cut, 0.0, 0.0001, "nothing cut from the front");
    assert_eq!(audio.len(), length_before, "nothing cut at all");
    assert_eq!(
        audio.iter().filter(|s| s.abs() < 0.05).count(),
        40 * 480,
        "the pause is still there, at its full length"
    );
}

#[test]
fn trimming_leaves_a_recording_with_no_speech_alone() {
    // (recording, why)
    let cases: &[(Vec<f32>, &str)] = &[
        (blocks(&[(20, 0.0)]), "digital silence"),
        (blocks(&[(20, 0.002)]), "an empty room"),
    ];
    for (recording, what) in cases {
        let mut audio = recording.clone();
        let length_before = audio.len();
        near(trim_quiet_edges(&mut audio), 0.0, 0.0001, what);
        assert_eq!(audio.len(), length_before, "{what}: nothing should be cut");
    }
}

#[test]
fn a_quiet_tail_is_cut_without_touching_the_front() {
    let mut audio = blocks(&[(40, 0.1), (40, 0.0)]);
    let cut = trim_quiet_edges(&mut audio);
    near(cut, 0.0, 0.0001, "nothing cut from the front");
    assert_eq!(audio.len(), 26_880, "the tail was cut");
}

// ---- shortening the long pauses in the middle -------------------------

// The biggest step between two neighbouring samples. A big step is a click.
fn biggest_step(samples: &[f32]) -> f32 {
    samples
        .windows(2)
        .fold(0.0f32, |max, w| max.max((w[1] - w[0]).abs()))
}

// 5 s of speech, so the pause that follows is past the protected opening.
// Anything shorter than this is left alone whatever else it is.
const PAST_THE_OPENING: usize = 170;

// What Settings starts out saying: 2.2 s, first 1.5 s protected.
fn rules() -> PauseRules {
    PauseRules {
        cutoff_ms: default_pause_cutoff_ms(),
        protect_opening_ms: Some(default_pause_opening_ms()),
    }
}

#[test]
fn a_long_pause_is_shortened_to_about_a_third_of_a_second() {
    // 5.1 s of speech, 3 s of thinking, 0.6 s of speech.
    let mut audio = blocks(&[(PAST_THE_OPENING, 0.1), (100, 0.0), (20, 0.1)]);
    let speech_samples_before = audio.iter().filter(|s| s.abs() > 0.05).count();
    let opening: Vec<f32> = audio[..PAST_THE_OPENING * 480].to_vec();
    let last_block: Vec<f32> = audio[audio.len() - 20 * 480..].to_vec();

    let removed = shorten_long_pauses(&mut audio, rules());

    near(removed, 2.705, 0.001, "seconds removed");
    let quiet = audio.iter().filter(|s| s.abs() < 0.05).count();
    near(
        quiet as f32 / 16_000.0,
        0.295,
        0.001,
        "what is left of the pause",
    );
    assert_eq!(
        audio.iter().filter(|s| s.abs() > 0.05).count(),
        speech_samples_before,
        "every spoken sample survived"
    );
    assert_eq!(
        audio[..PAST_THE_OPENING * 480],
        opening[..],
        "the speech before the cut"
    );
    assert_eq!(
        audio[audio.len() - 20 * 480..],
        last_block[..],
        "the speech after the cut"
    );
}

#[test]
fn two_long_pauses_are_both_shortened() {
    let mut audio = blocks(&[
        (PAST_THE_OPENING, 0.1),
        (100, 0.0),
        (20, 0.1),
        (100, 0.0),
        (20, 0.1),
    ]);
    let removed = shorten_long_pauses(&mut audio, rules());
    near(removed, 5.41, 0.001, "2.705 s taken out of each pause");
    assert_eq!(audio.len(), 110_240);
}

#[test]
fn a_recording_without_a_long_pause_comes_back_untouched() {
    // (recording, why)
    let cases: &[(Vec<f32>, &str)] = &[
        (blocks(&[(40, 0.1)]), "speech with no pause in it at all"),
        (
            blocks(&[(PAST_THE_OPENING, 0.1), (20, 0.0), (20, 0.1)]),
            "a 0.6 s pause: normal breathing, left alone",
        ),
        (
            blocks(&[(PAST_THE_OPENING, 0.1), (50, 0.0), (20, 0.1)]),
            "1.5 s: still a breath between sentences, and Whisper needs it",
        ),
        (
            blocks(&[(PAST_THE_OPENING, 0.1), (72, 0.0), (20, 0.1)]),
            "2.16 s, one frame under the 2.2 s the settings start at",
        ),
        (
            blocks(&[(100, 0.0), (20, 0.1), (100, 0.0)]),
            "a quiet start and end belong to trim_quiet_edges, not here",
        ),
    ];
    for (recording, what) in cases {
        let mut audio = recording.clone();
        near(shorten_long_pauses(&mut audio, rules()), 0.0, 0.0001, what);
        assert_eq!(&audio, recording, "{what}: not one sample may change");
    }
}

// The first seconds decide the language and the writing style for the whole
// recording, so nothing is cut near them.
#[test]
fn a_pause_in_the_opening_seconds_is_left_alone() {
    // (recording, why)
    let cases: &[(Vec<f32>, &str)] = &[
        (
            blocks(&[(20, 0.1), (100, 0.0), (20, 0.1)]),
            "a 3 s pause 0.6 s in: too close to the first words",
        ),
        (
            blocks(&[(49, 0.1), (100, 0.0), (20, 0.1)]),
            "one frame short of the opening being over",
        ),
        (
            blocks(&[(100, 0.0), (20, 0.1), (100, 0.0), (20, 0.1)]),
            "the opening is counted from the first word, not from the file",
        ),
    ];
    for (recording, what) in cases {
        let mut audio = recording.clone();
        near(shorten_long_pauses(&mut audio, rules()), 0.0, 0.0001, what);
        assert_eq!(&audio, recording, "{what}: not one sample may change");
    }

    // One frame later and it is shortened, so the line is where it says it is.
    let mut audio = blocks(&[(50, 0.1), (100, 0.0), (20, 0.1)]);
    assert!(
        shorten_long_pauses(&mut audio, rules()) > 0.0,
        "a pause starting just after the opening is shortened"
    );
}

#[test]
fn switching_the_opening_off_lets_an_early_pause_be_shortened() {
    let early = blocks(&[(20, 0.1), (100, 0.0), (20, 0.1)]);

    let mut protected = early.clone();
    near(
        shorten_long_pauses(&mut protected, rules()),
        0.0,
        0.0001,
        "protected: left alone",
    );

    let mut unprotected = early;
    near(
        shorten_long_pauses(
            &mut unprotected,
            PauseRules {
                protect_opening_ms: None,
                ..rules()
            },
        ),
        2.705,
        0.001,
        "unprotected: the same pause is shortened",
    );
}

#[test]
fn the_opening_length_decides_how_much_is_protected() {
    // Speech for 3 s, then a 3 s pause. (protected opening, is it shortened)
    let cases: &[(u32, bool, &str)] = &[
        (5000, false, "a 5 s opening covers a pause 3 s in"),
        (3000, true, "exactly on the line"),
        (1000, true, "a 1 s opening does not reach it"),
        (0, true, "no opening at all"),
    ];
    for &(opening_ms, expected, what) in cases {
        let mut audio = blocks(&[(100, 0.1), (100, 0.0), (20, 0.1)]);
        let removed = shorten_long_pauses(
            &mut audio,
            PauseRules {
                protect_opening_ms: Some(opening_ms),
                ..rules()
            },
        );
        assert_eq!(removed > 0.0, expected, "{what}");
    }
}

#[test]
fn the_cutoff_decides_which_pauses_are_touched() {
    // A 2 s pause. (cutoff in milliseconds, is it shortened, why)
    let cases: &[(u32, bool, &str)] = &[
        (2200, false, "the default leaves a 2 s pause alone"),
        (2000, true, "exactly on the cutoff"),
        (1000, true, "a low cutoff reaches it"),
        (30_000, false, "a cutoff longer than any real pause"),
    ];
    for &(cutoff_ms, expected, what) in cases {
        let mut audio = blocks(&[(PAST_THE_OPENING, 0.1), (67, 0.0), (20, 0.1)]);
        let removed = shorten_long_pauses(
            &mut audio,
            PauseRules {
                cutoff_ms,
                ..rules()
            },
        );
        assert_eq!(removed > 0.0, expected, "{what}");
    }
}

// A settings file edited by hand can hold anything, and this runs on the
// transcription thread where a panic loses the whole dictation.
#[test]
fn a_nonsense_cutoff_does_not_panic() {
    for cutoff_ms in [0, 1, 299, 300, u32::MAX] {
        let mut audio = blocks(&[(PAST_THE_OPENING, 0.1), (100, 0.0), (20, 0.1)]);
        let length_before = audio.len();
        let removed = shorten_long_pauses(
            &mut audio,
            PauseRules {
                cutoff_ms,
                ..rules()
            },
        );
        assert!(
            audio.len() + (removed * 16_000.0) as usize == length_before,
            "cutoff {cutoff_ms}: the length and the seconds reported must agree"
        );
    }
}

#[test]
fn a_recording_with_nothing_in_it_does_not_panic() {
    let cases: &[(Vec<f32>, &str)] = &[
        (Vec::new(), "nothing recorded"),
        (vec![0.5; 100], "shorter than one 30 ms block"),
        (blocks(&[(50, 0.0)]), "digital silence"),
        (blocks(&[(50, 0.002)]), "an empty room"),
    ];
    for (recording, what) in cases {
        let mut audio = recording.clone();
        near(shorten_long_pauses(&mut audio, rules()), 0.0, 0.0001, what);
        assert_eq!(&audio, recording, "{what}: nothing should change");
    }
}

#[test]
fn the_join_does_not_leave_a_click() {
    // A steady 55 Hz hum stands in for room noise. Cutting it at two different
    // points in its wave is exactly how a click gets made.
    let mut audio = blocks(&[(PAST_THE_OPENING, 0.1)]);
    audio.extend(sine(100 * 480, 55.0, 16_000.0, 0.005));
    audio.extend(blocks(&[(20, 0.1)]));

    // What a straight cut would have jumped by, with nothing blended.
    let hard_cut = (audio[265 * 480] - audio[175 * 480 - 1]).abs();
    assert!(
        hard_cut > 0.002,
        "this test is only meaningful if a straight cut would jump: {hard_cut}"
    );

    shorten_long_pauses(&mut audio, rules());

    // The speech blocks jump by 0.2 every sample by design, so only the quiet
    // middle is measured.
    let middle = &audio[PAST_THE_OPENING * 480..audio.len() - 20 * 480];
    let step = biggest_step(middle);
    assert!(
        step < hard_cut / 10.0,
        "the join jumps by {step}, a straight cut would jump by {hard_cut}"
    );
}

// ---- the numbers shown while recording --------------------------------

#[test]
fn chunk_loudness_reports_peak_and_average() {
    // (chunk, expected loudest, expected average, why)
    let cases: &[(&[f32], f32, f32, &str)] = &[
        (&[], 0.0, 0.0, "nothing recorded"),
        (&[0.0; 8], 0.0, 0.0, "silence"),
        (&[0.5, -0.5, 0.5, -0.5], 0.5, 0.5, "a steady tone"),
        (&[1.0, 0.0, 0.0, 0.0], 1.0, 0.5, "one spike among silence"),
        (
            &[-0.8, 0.1],
            0.8,
            0.5701,
            "the loudest sample can be negative",
        ),
    ];
    for &(chunk, want_peak, want_rms, what) in cases {
        let (peak, rms) = chunk_level(chunk);
        near(peak, want_peak, 0.001, &format!("{what}: loudest"));
        near(rms, want_rms, 0.001, &format!("{what}: average"));
    }
}

#[test]
fn recording_statistics_report_how_much_was_near_silence() {
    // (recording, loudest, average, share near silence, why)
    let cases: &[(Vec<f32>, f32, f32, f32, &str)] = &[
        (
            Vec::new(),
            0.0,
            0.0,
            1.0,
            "nothing recorded counts as all silence",
        ),
        (vec![0.0; 100], 0.0, 0.0, 1.0, "silence"),
        (vec![0.5; 100], 0.5, 0.5, 0.0, "a steady loud signal"),
        (
            [vec![0.5; 50], vec![0.001; 50]].concat(),
            0.5,
            0.3536,
            0.5,
            "half loud, half near silence",
        ),
    ];
    for (recording, want_peak, want_rms, want_quiet, what) in cases {
        let (peak, rms, quiet) = audio_stats(recording);
        near(peak, *want_peak, 0.001, &format!("{what}: loudest"));
        near(rms, *want_rms, 0.001, &format!("{what}: average"));
        near(
            quiet,
            *want_quiet,
            0.001,
            &format!("{what}: share near silence"),
        );
    }
}

#[test]
fn the_recent_sample_store_keeps_the_newest_and_stays_bounded() {
    let store = Arc::new(Mutex::new(Vec::<f32>::new()));

    // Less than the cap: everything is kept.
    keep_recent(&store, &[1.0, 2.0, 3.0]);
    assert_eq!(store.lock().unwrap().len(), 3);

    // Well past the cap, fed in pieces the way the microphone delivers it.
    let total = 6000usize;
    store.lock().unwrap().clear();
    for start in (0..total).step_by(1000) {
        let chunk: Vec<f32> = (start..start + 1000).map(|i| i as f32).collect();
        keep_recent(&store, &chunk);
    }
    let kept = store.lock().unwrap().clone();
    assert_eq!(kept.len(), 4096, "the store must not grow without limit");
    near(kept[4095], 5999.0, 0.5, "the newest sample is kept");
    near(
        kept[0],
        (total - 4096) as f32,
        0.5,
        "the oldest ones are dropped",
    );
}

// ---- the bars and the pitch the windows draw --------------------------

#[test]
fn frequency_bands_stay_in_range_and_follow_the_tone() {
    let mut planner = rustfft::FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);

    // Too short to measure: a full set of empty bars, not a crash.
    let short = frequency_bands(&[0.1; 100], &fft, 16_000);
    assert_eq!(short.len(), BAND_COUNT);
    assert!(
        short.iter().all(|&b| b == 0.0),
        "too-short input gives no bars"
    );

    // Silence: every bar empty.
    let silent = frequency_bands(&vec![0.0; FFT_SIZE], &fft, 16_000);
    assert!(silent.iter().all(|&b| b == 0.0), "silence gives no bars");

    // A single tone lights up one region, and no bar leaves the 0-to-1 range.
    let tone = sine(FFT_SIZE, 440.0, 16_000.0, 0.5);
    let bands = frequency_bands(&tone, &fft, 16_000);
    assert_eq!(bands.len(), BAND_COUNT);
    assert!(
        bands.iter().all(|&b| (0.0..=1.0).contains(&b)),
        "a bar outside 0 to 1 would draw off the window"
    );
    let loudest = bands
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert!(
        (18..=30).contains(&loudest),
        "a 440 Hz tone sits a third of the way across an 80 Hz to 8 kHz display, not at bar {loudest}"
    );
    assert!(
        bands[0] < bands[loudest] * 0.5,
        "the lowest bar should stay quiet"
    );

    // The display covers up to 8 kHz whatever the rate, so a high tone lands
    // on the right at 48 kHz just as at 16 kHz.
    let high = sine(FFT_SIZE, 6_000.0, 48_000.0, 0.5);
    let bands = frequency_bands(&high, &fft, 48_000);
    let loudest = bands
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert!(
        loudest >= 54,
        "a 6 kHz tone should light up the right edge, not bar {loudest}"
    );
}

#[test]
fn pitch_is_refused_when_it_cannot_be_told() {
    // 0.0 means "cannot be told" - better than a wrong number on screen.
    let cases: &[(Vec<f32>, u32, &str)] = &[
        (vec![0.0; 2048], 48_000, "silence"),
        (vec![0.1; 200], 48_000, "too little audio to measure"),
        (
            sine(2048, 200.0, 48_000.0, 0.001),
            48_000,
            "too quiet to measure",
        ),
        (noise(2048, 0.2), 48_000, "noise is not a voice"),
        (noise(2048, 0.5), 16_000, "loud noise is still not a voice"),
    ];
    for (audio, sample_rate, what) in cases {
        near(detect_pitch(audio, *sample_rate), 0.0, 0.0, what);
    }
}

#[test]
fn pitch_is_found_for_a_voice() {
    // (audio, sample rate, expected pitch, allowed error, why)
    // The ignored test below covers the ones this gets wrong.
    let cases: &[(Vec<f32>, u32, f32, f32, &str)] = &[
        (
            sine(2048, 80.0, 48_000.0, 0.3),
            48_000,
            80.0,
            5.0,
            "a very low voice",
        ),
        (
            sine(2048, 120.0, 48_000.0, 0.3),
            48_000,
            120.0,
            6.0,
            "a low voice",
        ),
        (
            sine(2048, 200.0, 48_000.0, 0.3),
            48_000,
            200.0,
            10.0,
            "an average voice",
        ),
        (
            sine(2048, 400.0, 48_000.0, 0.3),
            48_000,
            400.0,
            20.0,
            "a high voice",
        ),
        (
            sine(2048, 120.0, 16_000.0, 0.3),
            16_000,
            120.0,
            6.0,
            "a low voice at 16 kHz",
        ),
        (
            sine(2048, 200.0, 16_000.0, 0.3),
            16_000,
            200.0,
            10.0,
            "an average voice at 16 kHz",
        ),
    ];
    for (audio, sample_rate, want, tolerance, what) in cases {
        near(detect_pitch(audio, *sample_rate), *want, *tolerance, what);
    }
}

#[test]
fn pitch_should_never_be_reported_an_octave_too_low() {
    for hz in [150.0f32, 220.0, 250.0, 280.0, 300.0, 330.0, 350.0] {
        for sample_rate in [48_000u32, 16_000] {
            let audio = sine(2048, hz, sample_rate as f32, 0.3);
            let got = detect_pitch(&audio, sample_rate);
            assert!(
                (got - hz).abs() / hz < 0.06,
                "{hz} Hz at {sample_rate} Hz was reported as {got} Hz"
            );
        }
    }
}

// ---- writing the audio out --------------------------------------------

#[test]
fn wav_files_have_a_correct_header() {
    let wav = to_wav_bytes(&[0.0; 10], 16_000, 1);
    assert_eq!(wav.len(), 44 + 20, "44-byte header plus 2 bytes per sample");
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(&wav[12..16], b"fmt ");
    assert_eq!(&wav[36..40], b"data");
    assert_eq!(u32::from_le_bytes(wav[4..8].try_into().unwrap()), 36 + 20);
    assert_eq!(
        u16::from_le_bytes(wav[22..24].try_into().unwrap()),
        1,
        "channels"
    );
    assert_eq!(
        u32::from_le_bytes(wav[24..28].try_into().unwrap()),
        16_000,
        "sample rate"
    );
    assert_eq!(
        u16::from_le_bytes(wav[34..36].try_into().unwrap()),
        16,
        "bits per sample"
    );
    assert_eq!(
        u32::from_le_bytes(wav[40..44].try_into().unwrap()),
        20,
        "data size"
    );

    // An empty recording still produces a valid, empty file.
    assert_eq!(to_wav_bytes(&[], 16_000, 1).len(), 44);
}

#[test]
fn samples_outside_the_allowed_range_are_pulled_back_in() {
    // (sample, expected 16-bit value, why)
    let cases: &[(f32, i16, &str)] = &[
        (0.0, 0, "silence"),
        (1.0, i16::MAX, "the loudest allowed value"),
        (-1.0, -i16::MAX, "the quietest allowed value"),
        (
            2.0,
            i16::MAX,
            "past the top: pulled back, not wrapped around",
        ),
        (
            -2.0,
            -i16::MAX,
            "past the bottom: pulled back, not wrapped around",
        ),
    ];
    for &(sample, want, what) in cases {
        let wav = to_wav_bytes(&[sample], 16_000, 1);
        let got = i16::from_le_bytes(wav[44..46].try_into().unwrap());
        assert_eq!(got, want, "{what}");
    }
}

// ---- settings that have to survive a restart --------------------------

#[test]
fn tray_settings_survive_a_restart() {
    // (stored text, expected debug line, expected key, why)
    let cases: &[(&str, bool, &str, &str)] = &[
        (
            r#"{"debug_stats":true,"shortcut":"CommandOrControl+Shift+D"}"#,
            true,
            "CommandOrControl+Shift+D",
            "both saved",
        ),
        (
            r#"{"debug_stats":false}"#,
            false,
            "F3",
            "a file from before the key could be changed",
        ),
        (
            r#"{"language":"en","debug_stats":true}"#,
            true,
            "F3",
            "a file from when the language menu existed still loads",
        ),
        (
            r#"{}"#,
            false,
            "F3",
            "an empty settings file loads as defaults",
        ),
    ];
    for &(stored, want_debug, want_key, what) in cases {
        let prefs: Prefs = serde_json::from_str(stored).expect(what);
        assert_eq!(prefs.debug_stats, want_debug, "{what}: debug line");
        assert_eq!(prefs.shortcut, want_key, "{what}: key");
    }

    let text = serde_json::to_string(&Prefs {
        debug_stats: true,
        shortcut: "Alt+Space".to_string(),
        ..Prefs::default()
    })
    .unwrap();
    let loaded: Prefs = serde_json::from_str(&text).unwrap();
    assert!(loaded.debug_stats, "written out and read back");
    assert_eq!(loaded.shortcut, "Alt+Space");

    // A damaged file is rejected rather than half-read, so the caller can
    // fall back to the defaults.
    assert!(serde_json::from_str::<Prefs>("not json at all").is_err());
}

// ---- the settings that used to live in the hidden window ---------------

#[test]
fn a_settings_file_from_before_the_move_keeps_its_defaults() {
    // Only the two settings that were ever written by the old app.
    let old = r#"{"debug_stats":true,"shortcut":"Alt+Space"}"#;
    let prefs: Prefs = serde_json::from_str(old).expect("an old file still loads");

    assert!(prefs.debug_stats);
    assert_eq!(prefs.shortcut, "Alt+Space");
    assert!(prefs.active_local_model_id.is_none());
    assert!(
        prefs.pause_shortening,
        "pause-shortening is on unless it has been switched off"
    );
    // A file written before these existed must not read as "cut everything,
    // and cut it at the start too".
    assert_eq!(
        prefs.pause_cutoff_ms, 2200,
        "the cutoff falls back to 2.2 s"
    );
    assert!(
        prefs.pause_protect_opening,
        "and the opening is protected until that is switched off"
    );
    assert_eq!(
        prefs.pause_opening_ms, 1500,
        "the opening falls back to 1.5 s"
    );
    assert!(
        !prefs.migrated_from_browser,
        "the browser settings have not been copied over yet"
    );
}

#[test]
fn every_setting_survives_being_written_and_read_back() {
    let saved = Prefs {
        debug_stats: true,
        shortcut: "CommandOrControl+Shift+D".to_string(),
        active_local_model_id: Some("whisper-small".to_string()),
        selected_microphone: Some("MacBook Pro Microphone".to_string()),
        mic_boost: 1.8,
        visualisation: "ring".to_string(),
        language: "bg".to_string(),
        pause_shortening: true,
        pause_cutoff_ms: 3500,
        pause_protect_opening: false,
        pause_opening_ms: 8000,
        live_typing: true,
        live_pause_ms: 700,
        silence_stop: true,
        silence_stop_ms: 9000,
        auto_resume: true,
        auto_resume_ms: 4000,
        auto_enter: true,
        tidy_sentence_ends: false,
        onnx_gpu: true,
        whisper_gpu: false,
        migrated_from_browser: true,
    };
    let text = serde_json::to_string(&saved).unwrap();
    let loaded: Prefs = serde_json::from_str(&text).unwrap();

    assert_eq!(loaded.shortcut, saved.shortcut);
    assert!(loaded.pause_shortening);
    assert_eq!(loaded.pause_cutoff_ms, 3500);
    assert!(!loaded.pause_protect_opening);
    assert_eq!(
        loaded.pause_opening_ms, 8000,
        "the length is remembered even with the switch off"
    );
    assert!(loaded.live_typing);
    assert_eq!(loaded.live_pause_ms, 700);
    assert!(loaded.silence_stop);
    assert_eq!(loaded.silence_stop_ms, 9000);
    assert!(loaded.auto_resume);
    assert_eq!(loaded.auto_resume_ms, 4000);
    assert!(loaded.auto_enter);
    assert!(!loaded.tidy_sentence_ends);
    assert_eq!(
        loaded.active_local_model_id.as_deref(),
        Some("whisper-small")
    );
    assert_eq!(
        loaded.selected_microphone.as_deref(),
        Some("MacBook Pro Microphone")
    );
    assert!(loaded.migrated_from_browser);
    assert_eq!(loaded.mic_boost, 1.8);
    assert_eq!(loaded.visualisation, "ring");
    assert_eq!(loaded.language, "bg");
    assert!(
        loaded.onnx_gpu,
        "the ONNX GPU switch is written and read back"
    );
    assert!(
        !loaded.whisper_gpu,
        "and so is the Whisper one, including when it is off"
    );
}

// ---- which runtime gets the GPU ---------------------------------------
//
// The two switches point opposite ways by default, which is the whole reason
// they are two switches. Measured on an M2 Pro, 12.2 s of speech, second run:
//
//   Whisper Turbo    GPU 1.74 s    CPU 7.20 s     -> GPU, by 4.1x
//   Parakeet v3      GPU 642 ms    CPU 360 ms     -> CPU, by 1.8x
//   Moonshine base   GPU 400 ms    CPU 302 ms     -> CPU, by 1.3x
//
// On 70 s of speech Parakeet took 12-24 s on the GPU against 2.3 s on the
// processor, so the gap grows with the length of the dictation.

#[test]
fn the_gpu_switches_start_pointing_opposite_ways() {
    let fresh = Prefs::default();
    assert!(
        !fresh.onnx_gpu,
        "Parakeet and Moonshine start on the processor, which measured faster"
    );
    assert!(
        fresh.whisper_gpu,
        "Whisper starts on the GPU, where it is four times faster"
    );
}

#[test]
fn a_settings_file_from_before_the_gpu_switches_keeps_the_measured_defaults() {
    // Anyone updating has a file with neither field in it. It must not read
    // as "both off", which would make Whisper four times slower, nor as
    // "both on", which is the slow arrangement this change exists to fix.
    let old = r#"{"debug_stats":true,"shortcut":"F3","pause_shortening":true}"#;
    let prefs: Prefs = serde_json::from_str(old).expect("an old file still loads");

    assert!(!prefs.onnx_gpu, "ONNX falls back to the processor");
    assert!(prefs.whisper_gpu, "Whisper falls back to the GPU");
    assert!(prefs.pause_shortening, "and the rest of the file is intact");
}

#[test]
fn the_onnx_switch_picks_the_right_provider() {
    use transcribe_rs::OrtAccelerator;
    assert_eq!(
        onnx_accelerator(false),
        OrtAccelerator::CpuOnly,
        "off means the processor, not the library's own choice"
    );
    assert_eq!(
        onnx_accelerator(true),
        OrtAccelerator::Auto,
        "on hands the choice back to the library, which picks CoreML on a Mac"
    );
}

#[test]
fn moving_a_gpu_switch_makes_the_next_dictation_build_the_model_again() {
    let cpu = GpuChoice {
        onnx: false,
        whisper: true,
    };
    let onnx_on = GpuChoice {
        onnx: true,
        whisper: true,
    };
    let whisper_off = GpuChoice {
        onnx: false,
        whisper: false,
    };

    assert!(
        needs_load(None, "parakeet-v3-int8", cpu),
        "nothing loaded yet"
    );
    assert!(
        !needs_load(Some(("parakeet-v3-int8", cpu)), "parakeet-v3-int8", cpu),
        "same model, same switches: the one in memory is used again"
    );
    assert!(
        needs_load(Some(("parakeet-v3-int8", cpu)), "whisper-turbo", cpu),
        "a different model always means building"
    );

    // The point of the whole test: a model in memory was built with the
    // switches as they were. Moving one has to throw it away, or the switch
    // would look dead until the next restart.
    assert!(
        needs_load(Some(("parakeet-v3-int8", cpu)), "parakeet-v3-int8", onnx_on),
        "same model, ONNX switch moved"
    );
    assert!(
        needs_load(Some(("whisper-turbo", cpu)), "whisper-turbo", whisper_off),
        "same model, Whisper switch moved"
    );
}

#[test]
fn a_settings_file_with_the_old_microphone_number_still_loads() {
    // The microphone used to be saved as a WirePlumber number, which meant
    // nothing on a Mac. The name replaced it; the old number is ignored
    // rather than making the whole file unreadable.
    let prefs: Prefs = serde_json::from_str(
        r#"{"selected_audio_device_id":57,"active_local_model_id":"whisper-turbo"}"#,
    )
    .expect("an old file still loads");
    assert_eq!(prefs.selected_microphone, None, "back to the system's own");
    assert_eq!(
        prefs.active_local_model_id.as_deref(),
        Some("whisper-turbo"),
        "and everything beside it survives"
    );
}

// ---- copying the settings out of the browser, once ---------------------

// Browser storage keeps everything as text, so every value arrives as text.
fn browser(pairs: &[(&str, &str)]) -> BrowserSettings {
    let get = |key: &str| {
        pairs
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| v.to_string())
    };
    BrowserSettings {
        active_local_model_id: get("active_local_model_id"),
    }
}

#[test]
fn settings_saved_in_the_browser_are_carried_over() {
    let mut prefs = Prefs::default();
    let copied = apply_browser_settings(
        &mut prefs,
        browser(&[("active_local_model_id", "parakeet-v3-int8")]),
    );

    assert!(copied);
    assert_eq!(
        prefs.active_local_model_id.as_deref(),
        Some("parakeet-v3-int8")
    );
    assert!(prefs.migrated_from_browser, "and it is marked as done");
}

#[test]
fn the_copy_happens_once_and_never_undoes_a_later_change() {
    let mut prefs = Prefs::default();
    apply_browser_settings(
        &mut prefs,
        browser(&[("active_local_model_id", "whisper-small")]),
    );
    assert_eq!(
        prefs.active_local_model_id.as_deref(),
        Some("whisper-small")
    );

    // The user then picks a different model in Settings.
    prefs.active_local_model_id = Some("whisper-turbo".to_string());

    // The browser still holds the old value, but it must not win again.
    let copied = apply_browser_settings(
        &mut prefs,
        browser(&[("active_local_model_id", "whisper-small")]),
    );
    assert!(!copied, "the second attempt does nothing");
    assert_eq!(
        prefs.active_local_model_id.as_deref(),
        Some("whisper-turbo")
    );
}

#[test]
fn empty_browser_values_are_left_behind() {
    let mut prefs = Prefs {
        active_local_model_id: Some("already-here".to_string()),
        ..Prefs::default()
    };
    apply_browser_settings(&mut prefs, browser(&[("active_local_model_id", "")]));

    assert_eq!(
        prefs.active_local_model_id.as_deref(),
        Some("already-here"),
        "a blank name must not wipe a real one"
    );
}

// ---- the two chimes ---------------------------------------------------

#[test]
fn a_chime_starts_and_ends_in_silence() {
    // A note that jumps straight to full loudness clicks. These are the
    // moments that must be silent for it not to.
    for notes in [&START_CHIME, &DONE_CHIME] {
        near(
            chime_at(notes, 0.0),
            0.0,
            0.0001,
            "silent at the very start",
        );
        let after = chime_seconds(notes) + 0.05;
        near(
            chime_at(notes, after),
            0.0,
            0.0001,
            "silent once it is over",
        );
        near(
            chime_at(notes, -0.1),
            0.0,
            0.0001,
            "silent before it begins",
        );
    }
}

#[test]
fn a_chime_never_gets_loud_enough_to_distort() {
    // Both notes overlap, so their loudness adds up. Anything past 1.0
    // would be clipped by the speakers into a buzz.
    for notes in [&START_CHIME, &DONE_CHIME] {
        let steps = 20_000;
        let length = chime_seconds(notes);
        let mut loudest = 0.0f32;
        for step in 0..steps {
            let value = chime_at(notes, length * step as f32 / steps as f32);
            loudest = loudest.max(value.abs());
        }
        assert!(loudest > 0.05, "it has to be audible: got {loudest}");
        assert!(loudest < 1.0, "it must not distort: got {loudest}");
    }
}

#[test]
fn one_note_fades_in_quickly_and_out_slowly() {
    // (how far into the note, what the loudness should be, why)
    let cases: &[(f32, f32, &str)] = &[
        (-0.01, 0.0, "before it starts"),
        (0.0, 0.0, "silent at the moment it starts"),
        (0.015, 0.06, "half way through the fade in"),
        (0.03, 0.12, "at its loudest once faded in"),
        (0.215, 0.06, "half way through the fade out"),
        (0.4, 0.0, "silent at the moment it ends"),
        (0.5, 0.0, "after it has ended"),
    ];
    for &(age, want, what) in cases {
        near(note_loudness(age, 0.4), want, 0.005, what);
    }
}

#[test]
fn a_chime_lasts_until_its_last_note_has_finished() {
    // The second note starts late, so the first one ending is not the end.
    near(
        chime_seconds(&START_CHIME),
        0.5,
        0.0001,
        "0.08s in plus 0.42s long",
    );
    near(
        chime_seconds(&DONE_CHIME),
        0.6,
        0.0001,
        "0.08s in plus 0.52s",
    );
}

// ---- deleting recordings ----------------------------------------------

#[test]
fn deleting_recordings_removes_only_recordings() {
    let dir = std::env::temp_dir().join(format!("omegawhisper-delete-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    // (file name, should it survive, why)
    let files: &[(&str, bool, &str)] = &[
        ("2026-01-01.wav", false, "a recording"),
        ("2026-01-01-model-input.wav", false, "what the model heard"),
        ("notes.txt", true, "someone else's file"),
        ("tray-prefs.json", true, "a settings file"),
        ("recording.wav.bak", true, "a backup, not a recording"),
        ("WAV", true, "no extension at all"),
    ];
    for (name, _, _) in files {
        fs::write(dir.join(name), b"x").unwrap();
    }
    // A folder must survive too - only files are considered.
    fs::create_dir(dir.join("old")).unwrap();
    fs::write(dir.join("old").join("kept.wav"), b"x").unwrap();

    let deleted = delete_recordings_in(&dir).unwrap();
    assert_eq!(deleted, 2, "only the two recordings should go");

    for (name, survives, why) in files {
        assert_eq!(dir.join(name).exists(), *survives, "{name}: {why}");
    }
    assert!(
        dir.join("old").join("kept.wav").exists(),
        "subfolders untouched"
    );
    assert!(dir.exists(), "the folder itself must stay");

    // Running it again on an empty folder is not an error.
    assert_eq!(delete_recordings_in(&dir).unwrap(), 0);

    let _ = fs::remove_dir_all(&dir);
}

// ---- pause-shortening against real recordings --------------------------
//
// Both of these read the recordings this Mac has saved, so they are skipped
// unless asked for by name. To run them:
//
//   cargo test --manifest-path src-tauri/Cargo.toml -- --ignored --nocapture

// One of the 16 kHz, mono, 16-bit files the app writes. 44-byte header.
fn read_model_input(path: &std::path::Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("{}: {}", path.display(), e));
    bytes[44..]
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / i16::MAX as f32)
        .collect()
}

// Every "-model-input.wav" in the recordings folder: exactly what the model
// was given on a real dictation, in the order they were recorded.
fn corpus() -> Vec<(String, Vec<f32>)> {
    let dir = crate::storage::get_recordings_dir().expect("no recordings folder");
    let mut paths: Vec<std::path::PathBuf> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {}", dir.display(), e))
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with("-model-input.wav"))
        })
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "no recordings in {}", dir.display());

    paths
        .into_iter()
        .map(|path| {
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            let name = name.trim_end_matches("-model-input.wav").to_string();
            (name, read_model_input(&path))
        })
        .collect()
}

// ---- quitting with a model still loaded --------------------------------
//
// Whisper keeps GPU buffers inside the loaded model. The app ends through C's
// exit(), which runs Metal's own teardown but not Rust's, so the model has to
// be let go first - `RunEvent::Exit` in lib.rs - or the teardown finds buffers
// still held and aborts the process:
//
//   ggml-metal-device.m: GGML_ASSERT([rsets->data count] == 0) failed
//
// Nothing is lost when that happens, which is exactly why it needs a test:
// take the line out of lib.rs and everything still appears to work. Only the
// exit code changes, from 0 to 134.
//
// It takes two processes, because the failure is the process dying. The parent
// runs the child below twice and compares how each one ended.

// Whisper specifically: the buffers that trip the assertion are Metal's, and
// only Whisper is on Metal.
fn a_downloaded_whisper_model() -> Option<&'static str> {
    let models = crate::managers::ModelManager::new().ok()?;
    crate::managers::AVAILABLE_MODELS
        .iter()
        .find(|m| m.id.starts_with("whisper") && models.is_model_downloaded(m.id))
        .map(|m| m.id)
}

#[test]
#[ignore = "starts two more processes, one of which is meant to abort"]
fn quitting_with_a_model_loaded_does_not_abort() {
    let Some(_) = a_downloaded_whisper_model() else {
        println!("no Whisper model downloaded, nothing to test");
        return;
    };
    let exe = std::env::current_exe().expect("no test binary");

    // (let the model go first, should the process end cleanly, why)
    let cases: &[(&str, bool, &str)] = &[
        (
            "0",
            false,
            "a model still in memory at exit aborts - the fault this guards",
        ),
        ("1", true, "letting it go first is what makes exit clean"),
    ];

    for &(drop_first, want_clean, why) in cases {
        let status = std::process::Command::new(&exe)
            .args(["--exact", "--ignored", "--nocapture", "tests::quit_child"])
            .env("OMEGAWHISPER_QUIT_CHILD", "1")
            .env("OMEGAWHISPER_QUIT_DROP", drop_first)
            .output()
            .expect("could not start the child");

        assert_eq!(
            status.status.success(),
            want_clean,
            "{}: ended with {:?}",
            why,
            status.status
        );
    }
}

// The other half of the test above, and only useful when driven by it: it ends
// its own process on purpose. Run on its own it does nothing.
#[test]
#[ignore = "the second half of quitting_with_a_model_loaded_does_not_abort"]
fn quit_child() {
    if std::env::var("OMEGAWHISPER_QUIT_CHILD").is_err() {
        return;
    }
    let model_id = a_downloaded_whisper_model().expect("no Whisper model");
    let models = Arc::new(crate::managers::ModelManager::new().expect("no model folder"));
    let mut engine = crate::managers::TranscriptionManager::new(models);

    engine
        .load_model(
            model_id,
            GpuChoice {
                onnx: false,
                whisper: true,
            },
        )
        .expect("could not load");

    // Something has to run for the GPU buffers to be handed out.
    engine
        .transcribe(&vec![0.0f32; 16_000], None)
        .expect("transcribe failed");

    if std::env::var("OMEGAWHISPER_QUIT_DROP").as_deref() == Ok("1") {
        engine.unload_model();
    }

    // What the tray's Quit does, and what RunEvent::Exit runs just before.
    std::process::exit(0);
}

// The two GPU switches, timed against each other on every model on this Mac
// and a real recording. This is what the defaults in `Prefs` are set from; run
// it on any Mac to find out whether they are right for that machine.
//
//   cargo test --manifest-path src-tauri/Cargo.toml --release -- --ignored \
//       --nocapture both_gpu_switches
//
// It asserts only that both settings produce text. Speed is printed, not
// asserted: a timing test that fails on a busy machine is worse than no test.
#[test]
#[ignore = "loads every model on this Mac twice over and times them"]
fn both_gpu_switches_change_where_the_model_runs() {
    let models = Arc::new(crate::managers::ModelManager::new().expect("no model folder"));

    // Every model on this Mac, not only the chosen one, so a single run covers
    // both switches: Whisper answers to Metal, the rest to CoreML.
    let downloaded: Vec<&str> = crate::managers::AVAILABLE_MODELS
        .iter()
        .map(|m| m.id)
        .filter(|id| models.is_model_downloaded(id))
        .collect();
    assert!(!downloaded.is_empty(), "download a model in Settings first");

    // The longest recording on this Mac, cut to a minute. The gap between
    // processor and GPU grows with the length of the audio, so the shortest
    // recording would hide it - but the longest one here was a whole meeting,
    // and transcribing it four times took eighteen minutes. A minute is long
    // enough to show the difference and short enough to sit through, and it
    // stays under the 64 seconds Moonshine refuses to go past.
    const MOST_SECONDS: usize = 60;
    let (name, mut samples) = corpus()
        .into_iter()
        .max_by_key(|(_, samples)| samples.len())
        .expect("no recordings");
    let full_seconds = samples.len() as f32 / 16_000.0;
    samples.truncate(MOST_SECONDS * 16_000);
    println!(
        "\naudio: {} ({:.1}s of {:.1}s)\nmodels: {}",
        name,
        samples.len() as f32 / 16_000.0,
        full_seconds,
        downloaded.join(", ")
    );

    for model_id in downloaded {
        // Whisper is the only one on Metal; everything else is on CoreML. Only
        // the switch that model actually answers to is moved.
        let on_metal = model_id.starts_with("whisper");
        println!(
            "\n{} - {}",
            model_id,
            if on_metal {
                "whisper.cpp, so the Whisper switch"
            } else {
                "ONNX Runtime, so the Parakeet and Moonshine switch"
            }
        );

        let mut timings = Vec::new();
        for gpu_on in [false, true] {
            let gpu = GpuChoice {
                onnx: !on_metal && gpu_on,
                whisper: on_metal && gpu_on,
            };

            // A fresh manager each time, so nothing built the other way is
            // left in memory - the mistake that made an earlier hand
            // measurement of this show no difference at all.
            let mut engine = crate::managers::TranscriptionManager::new(models.clone());

            let started = std::time::Instant::now();
            engine.load_model(model_id, gpu).expect("could not load");
            let load = started.elapsed();

            // Twice: the first run pays one-off setup a later dictation would
            // not, and it is the later dictations that matter.
            let mut run = std::time::Duration::ZERO;
            let mut text = String::new();
            let mut refused = None;
            for _ in 0..2 {
                let started = std::time::Instant::now();
                match engine.transcribe(&samples, None) {
                    Ok(got) => {
                        text = got;
                        run = started.elapsed();
                    }
                    Err(e) => {
                        refused = Some(e);
                        break;
                    }
                }
            }
            if let Some(e) = refused {
                println!("  refused the audio: {}", e);
                break;
            }

            assert!(
                !text.trim().is_empty(),
                "{} produced no text with the GPU {}",
                model_id,
                if gpu_on { "on" } else { "off" }
            );
            println!(
                "  GPU {:3}   load {:>9.2?}   transcribe {:>9.2?}",
                if gpu_on { "on" } else { "off" },
                load,
                run
            );
            timings.push(run);
        }

        // Only when both halves ran. A model that refused the audio has one.
        if let [cpu, gpu] = timings[..] {
            let (faster, ratio) = if cpu <= gpu {
                ("the processor", gpu.as_secs_f64() / cpu.as_secs_f64())
            } else {
                ("the GPU", cpu.as_secs_f64() / gpu.as_secs_f64())
            };
            println!("  -> {} wins, by {:.1}x", faster, ratio);
        }
    }
}

// Where every long pause sits in every real recording, so the rule about
// leaving the opening alone can be set from evidence rather than a guess.
#[test]
#[ignore = "reads the recordings saved on this Mac"]
fn where_the_long_pauses_are_in_real_recordings() {
    for (name, samples) in corpus() {
        let seconds = samples.len() as f32 / 16_000.0;
        let pauses = long_pauses(&samples, rules());
        if pauses.is_empty() {
            continue;
        }
        let list: Vec<String> = pauses
            .iter()
            .map(|(start, end)| {
                format!(
                    "{:.1}s..{:.1}s ({:.1}s long)",
                    *start as f32 * 0.03,
                    *end as f32 * 0.03,
                    (end - start) as f32 * 0.03
                )
            })
            .collect();
        println!("{:<22} {:>5.1}s total   {}", name, seconds, list.join(", "));
    }
    println!();
}

#[test]
#[ignore = "reads the recordings saved on this Mac"]
fn how_much_shorter_pause_shortening_makes_real_recordings() {
    println!(
        "\n{:<22} {:>9} {:>9} {:>8}",
        "recording", "before", "after", "saved"
    );
    let (mut total_before, mut total_after) = (0.0f32, 0.0f32);

    for (name, samples) in corpus() {
        let before = samples.len() as f32 / 16_000.0;
        let mut shortened = samples;
        shorten_long_pauses(&mut shortened, rules());
        let after = shortened.len() as f32 / 16_000.0;
        total_before += before;
        total_after += after;
        println!(
            "{:<22} {:>8.1}s {:>8.1}s {:>7.0}%",
            name,
            before,
            after,
            (before - after) / before * 100.0
        );
    }

    println!(
        "{:<22} {:>8.1}s {:>8.1}s {:>7.0}%   <- all {} recordings\n",
        "TOTAL",
        total_before,
        total_after,
        (total_before - total_after) / total_before * 100.0,
        corpus().len()
    );
}

// Every recording, transcribed four times: twice untouched and twice with the
// pauses shortened.
//
// Two runs of the untouched audio are what make this readable. The model does
// not give the same text twice from the same samples, so without that control
// every difference would look like damage done by pause-shortening. The
// recordings it changes nothing in - no pause long enough - are the measuring
// stick: whatever they disagree by is the model on its own.
#[test]
#[ignore = "loads a model and transcribes every recording four times"]
fn pause_shortening_does_not_change_what_the_model_types() {
    let model_id = load_prefs()
        .active_local_model_id
        .expect("choose a model in Settings first");
    let models = Arc::new(crate::managers::ModelManager::new().expect("no model folder"));
    let mut engine = crate::managers::TranscriptionManager::new(models);
    // The saved switches, so this measures the model the way the app runs it.
    let saved = load_prefs();
    engine
        .load_model(
            &model_id,
            GpuChoice {
                onnx: saved.onnx_gpu,
                whisper: saved.whisper_gpu,
            },
        )
        .expect("could not load model");
    println!("\nmodel: {}\n", model_id);

    // (recordings, of those where the two untouched runs disagreed, of those
    // where neither untouched run matched either shortened run)
    let mut untouched_audio = (0usize, 0usize, 0usize);
    let mut shortened_audio = (0usize, 0usize, 0usize);
    // Every recording, whether it differed or not, so nothing is hidden.
    let mut rows: Vec<serde_json::Value> = Vec::new();

    for (name, samples) in corpus() {
        let plain = [
            engine
                .transcribe(&samples, None)
                .expect("transcribe failed"),
            engine
                .transcribe(&samples, None)
                .expect("transcribe failed"),
        ];
        let mut shortened = samples;
        let removed = shorten_long_pauses(&mut shortened, rules());
        let cut = [
            engine
                .transcribe(&shortened, None)
                .expect("transcribe failed"),
            engine
                .transcribe(&shortened, None)
                .expect("transcribe failed"),
        ];

        let plain: Vec<&str> = plain.iter().map(|t| t.trim()).collect();
        let cut: Vec<&str> = cut.iter().map(|t| t.trim()).collect();
        let model_disagreed = plain[0] != plain[1];
        // Nothing in common at all: not one of the four runs came back the
        // same, so the shortening cannot be excused as the model wandering.
        let no_overlap = !cut.iter().any(|c| plain.contains(c));

        let counters = if removed > 0.0 {
            &mut shortened_audio
        } else {
            &mut untouched_audio
        };
        counters.0 += 1;
        counters.1 += usize::from(model_disagreed);
        counters.2 += usize::from(no_overlap);

        println!(
            "--- {} ({:.1}s removed{})",
            name,
            removed,
            if removed > 0.0 {
                ""
            } else {
                ", audio unchanged"
            }
        );
        println!("    untouched 1: {:?}", plain[0]);
        println!("    untouched 2: {:?}", plain[1]);
        println!("    shortened 1: {:?}", cut[0]);
        println!("    shortened 2: {:?}", cut[1]);

        rows.push(serde_json::json!({
            "name": name,
            "removed": removed,
            "untouched": plain,
            "shortened": cut,
        }));
    }

    println!(
        "\n{:<34} {:>7} {:>12} {:>12}",
        "", "count", "model split", "no overlap"
    );
    for (what, c) in [
        ("audio unchanged (nothing to shorten)", untouched_audio),
        ("audio shortened", shortened_audio),
    ] {
        println!("{:<34} {:>7} {:>12} {:>12}", what, c.0, c.1, c.2);
    }
    println!(
        "\n\"model split\" = the two untouched runs disagreed with each other.\n\
         \"no overlap\" = neither shortened run matched either untouched run.\n"
    );

    // The same rows again as one line of JSON, so the four texts per recording
    // can be put side by side somewhere they are readable.
    let path = std::env::temp_dir().join("omegawhisper-pause-shortening-runs.json");
    match serde_json::to_string(&rows).map(|text| fs::write(&path, text)) {
        Ok(Ok(())) => println!("every run, as JSON: {}\n", path.display()),
        _ => println!("could not write {}\n", path.display()),
    }
}

#[test]
fn deleting_from_a_missing_folder_is_an_error_not_a_panic() {
    let missing = std::env::temp_dir().join("omegawhisper-does-not-exist-at-all");
    let _ = fs::remove_dir_all(&missing);
    assert!(delete_recordings_in(&missing).is_err());
}

#[cfg(target_os = "linux")]
mod linux_only {
    use crate::linux::{desktop_entry, xdg_trigger, APP_ID};

    #[test]
    fn the_dictation_key_is_spelled_the_way_the_portal_reads_it() {
        assert_eq!(xdg_trigger("F3"), "F3");
        assert_eq!(xdg_trigger("CommandOrControl+Shift+D"), "CTRL+SHIFT+d");
        assert_eq!(xdg_trigger("Control+KeyA"), "CTRL+a");
        assert_eq!(xdg_trigger("Alt+Space"), "ALT+space");
        assert_eq!(xdg_trigger("Super+Digit1"), "LOGO+1");
        assert_eq!(xdg_trigger("Shift+Enter"), "SHIFT+Return");
    }

    // The portal finds the app through this file, named after the app id.
    #[test]
    fn the_desktop_file_points_at_the_binary_it_was_written_by() {
        let entry = desktop_entry("/opt/omegawhisper/omegawhisper");
        assert!(entry.starts_with("[Desktop Entry]\n"));
        assert!(entry.contains("\nExec=/opt/omegawhisper/omegawhisper\n"));
        assert!(entry.contains("\nName=Omegawhisper\n"));
        assert_eq!(APP_ID, "dev.omegawhisper");
    }
}

#[test]
fn the_microphone_boost_multiplies_and_clips() {
    let mut samples = vec![0.1, -0.2, 0.3, -0.5];
    boost_samples(&mut samples, 2.5);
    assert_eq!(samples, vec![0.25, -0.5, 0.75, -1.0]);

    let mut untouched = vec![0.1, -0.2];
    boost_samples(&mut untouched, 1.0);
    assert_eq!(untouched, vec![0.1, -0.2]);
}

#[test]
fn the_microphone_boost_stays_between_half_and_a_hundred() {
    assert_eq!(clamp_mic_boost(0.0), 0.5);
    assert_eq!(clamp_mic_boost(1.7), 1.7);
    assert_eq!(clamp_mic_boost(1.23), 1.2);
    assert_eq!(clamp_mic_boost(50.0), 50.0);
    assert_eq!(clamp_mic_boost(500.0), 100.0);
    assert_eq!(clamp_mic_boost(f32::NAN), 1.0);
}

#[test]
fn a_settings_file_from_before_the_boost_has_it_off() {
    let old = r#"{"debug_stats":true,"shortcut":"F3"}"#;
    let prefs: Prefs = serde_json::from_str(old).expect("an old file still loads");
    assert_eq!(prefs.mic_boost, 1.0);
    assert_eq!(Prefs::default().mic_boost, 1.0);
}

// A 16 kHz mono WAV as samples, whatever its sample width; 48 kHz files are
// brought down to 16 kHz the way a recording is.
fn wav_samples_16k(path: &std::path::Path) -> Vec<f32> {
    let bytes = fs::read(path).expect("wav readable");
    let rate = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    let channels = u16::from_le_bytes([bytes[22], bytes[23]]);
    let bits = u16::from_le_bytes([bytes[34], bytes[35]]);
    let format = u16::from_le_bytes([bytes[20], bytes[21]]);
    let mut pos = 12;
    let data = loop {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        if id == b"data" {
            break &bytes[pos + 8..(pos + 8 + len).min(bytes.len())];
        }
        pos += 8 + len;
    };
    let samples: Vec<f32> = match (format, bits) {
        (3, 32) => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        (1, 16) => data
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect(),
        other => panic!("unsupported wav format {:?}", other),
    };
    let mono = mix_to_mono(samples, channels);
    if rate == 16_000 {
        return mono;
    }
    let mut resampler = crate::resampler::AudioResampler::new(rate).expect("resampler");
    let mut out = resampler.process(&mono).expect("resample");
    out.extend(resampler.flush().unwrap_or_default());
    out
}

#[test]
fn the_speech_detector_hears_speech_in_a_recording_however_quiet() {
    let jfk = wav_samples_16k(std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests-data/jfk.wav"
    )));
    let loud = crate::vad::speech_seconds(&jfk).expect("detector runs");
    assert!(
        loud > 5.0,
        "11 s of a speech should be mostly speech, got {loud}"
    );

    // The same words at a fortieth of the level, which is what a headset at
    // low gain delivered and the old gate threw away.
    let quiet: Vec<f32> = jfk.iter().map(|s| s * 0.01).collect();
    assert!(
        speech_level(&quiet) < MIN_SPEECH_LEVEL,
        "quiet enough to fail the old gate"
    );
    let heard = crate::vad::speech_seconds(&quiet).expect("detector runs");
    assert!(
        (heard - loud).abs() < 1.0,
        "quiet {heard} against loud {loud}"
    );
    assert!(crate::vad::holds_speech(heard));
}

#[test]
fn the_speech_detector_hears_nothing_in_noise_or_a_tone() {
    let hiss = noise(16_000 * 4, 0.3);
    let heard = crate::vad::speech_seconds(&hiss).expect("detector runs");
    assert!(
        !crate::vad::holds_speech(heard),
        "noise counted as {heard}s of speech"
    );

    let tone = sine(16_000 * 4, 440.0, 16_000.0, 0.5);
    let heard = crate::vad::speech_seconds(&tone).expect("detector runs");
    assert!(
        !crate::vad::holds_speech(heard),
        "a tone counted as {heard}s of speech"
    );

    let nothing = vec![0.0f32; 16_000 * 3];
    assert_eq!(crate::vad::speech_seconds(&nothing).unwrap(), 0.0);
}

// Every saved recording on this machine, judged by the detector. Run with
// --ignored --nocapture to see what it makes of real takes.
#[test]
#[ignore]
fn what_the_speech_detector_makes_of_the_saved_recordings() {
    let dir = crate::storage::get_recordings_dir().expect("recordings folder");
    let mut paths: Vec<_> = fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "wav"))
        .collect();
    paths.sort();
    for path in paths {
        let samples = wav_samples_16k(&path);
        let (peak, _, _) = audio_stats(&samples);
        let level = speech_level(&samples);
        let heard = crate::vad::speech_seconds(&samples).expect("detector runs");
        println!(
            "{:<45} {:5.1}s  peak {:.3}  level {:.4}  old gate {:<5}  speech {:4.1}s  new gate {}",
            path.file_name().unwrap().to_string_lossy(),
            samples.len() as f32 / 16_000.0,
            peak,
            level,
            holds_speech(level, peak),
            heard,
            crate::vad::holds_speech(heard)
        );
    }
}

#[test]
fn a_settings_file_from_before_the_visualisations_draws_the_waterfall() {
    let old = r#"{"debug_stats":true,"shortcut":"F3"}"#;
    let prefs: Prefs = serde_json::from_str(old).expect("an old file still loads");
    assert_eq!(prefs.visualisation, "waterfall");
    assert!(VISUALISATIONS.contains(&prefs.visualisation.as_str()));
}

#[test]
fn the_detector_sees_every_recording_at_the_same_level() {
    let quiet = vec![0.0, 0.01, -0.02, 0.005];
    let scaled = crate::vad::normalized(&quiet);
    let peak = scaled.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    near(peak, 0.9, 1e-6, "peak after scaling");
    assert_eq!(crate::vad::normalized(&[0.0, 0.0]), vec![0.0, 0.0]);
}

#[test]
fn a_dead_microphone_and_a_silent_room_are_told_apart() {
    use crate::vad::{judge, Heard};
    let nothing = vec![0.0f32; 16_000 * 2];
    assert_eq!(judge(&nothing, 0.0, 0.0), Heard::Dead);

    let hiss = noise(16_000 * 3, 0.2);
    let (peak, _, _) = audio_stats(&hiss);
    assert_eq!(judge(&hiss, peak, speech_level(&hiss)), Heard::Silent);

    let jfk = wav_samples_16k(std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests-data/jfk.wav"
    )));
    let (peak, _, _) = audio_stats(&jfk);
    match judge(&jfk, peak, speech_level(&jfk)) {
        Heard::Speech(Some(seconds)) => assert!(seconds > 5.0),
        other => panic!("a speech should be heard as speech, not {other:?}"),
    }
}

#[cfg(not(target_os = "macos"))]
mod typing_tools {
    use crate::typing::{needs_paste, tool_order};

    #[test]
    fn the_tool_for_the_session_comes_first() {
        assert_eq!(tool_order(true), ["wtype", "ydotool", "xdotool"]);
        assert_eq!(tool_order(false), ["xdotool", "ydotool", "wtype"]);
    }

    #[test]
    fn the_paste_chord_is_one_argument_per_key_and_releases_what_it_pressed() {
        use crate::typing::PASTE_CHORD;
        assert_eq!(PASTE_CHORD.len(), 4);
        for key in PASTE_CHORD {
            let (code, state) = key.split_once(':').expect("code:state");
            assert!(
                code.parse::<u16>().is_ok() && matches!(state, "0" | "1"),
                "{key}"
            );
        }
        assert_eq!(PASTE_CHORD[0], "29:1", "Ctrl goes down first");
        assert_eq!(PASTE_CHORD[3], "29:0", "and comes up last");
    }

    #[test]
    fn everything_ydotool_is_given_is_pasted() {
        assert!(needs_paste("ydotool", "plain ascii, any length"));
        assert!(needs_paste("ydotool", "Здравей"));
        assert!(needs_paste("ydotool", "wörld"));
        assert!(!needs_paste("wtype", "Здравей"));
        assert!(!needs_paste("xdotool", "Здравей"));
    }
}

#[cfg(target_os = "linux")]
mod microphone_list {
    use crate::microphone::parse_sources;

    const PACTL: &str = r#"[
        {"index": 70, "name": "alsa_output.usb-Speaker.monitor", "description": "Monitor of Speakers"},
        {"index": 74, "name": "alsa_input.usb-Generic.Mic", "description": "USB Audio Microphone"},
        {"index": 76, "name": "alsa_input.usb-HyperX.mono", "description": "HyperX Cloud Alpha S Mono"},
        {"index": 77, "name": "alsa_input.unnamed", "description": ""}
    ]"#;

    #[test]
    fn monitors_are_left_out_and_the_default_is_marked() {
        let list = parse_sources(PACTL.as_bytes(), "alsa_input.usb-HyperX.mono").unwrap();
        let names: Vec<&str> = list.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "alsa_input.usb-Generic.Mic",
                "alsa_input.usb-HyperX.mono",
                "alsa_input.unnamed"
            ]
        );
        assert_eq!(list[1].label, "HyperX Cloud Alpha S Mono");
        assert!(list[1].is_default);
        assert!(!list[0].is_default);
    }

    #[test]
    fn a_source_without_a_description_is_shown_by_name() {
        let list = parse_sources(PACTL.as_bytes(), "").unwrap();
        assert_eq!(list[2].label, "alsa_input.unnamed");
    }

    #[test]
    fn a_broken_answer_is_an_error_not_an_empty_menu() {
        assert!(parse_sources(b"not json", "").is_err());
    }
}

#[test]
fn the_style_prompt_follows_the_chosen_language_and_stays_out_of_detection() {
    use crate::managers::transcription::style_prompt;
    assert!(style_prompt(Some("en")).unwrap().starts_with("Hello."));
    assert!(style_prompt(Some("bg")).unwrap().starts_with("Здравей."));
    assert_eq!(style_prompt(Some("de")), None, "no prompt written for it");
    assert_eq!(
        style_prompt(None),
        None,
        "a prompt would pull detection into its language"
    );
}

#[test]
fn the_language_setting_becomes_what_whisper_is_told() {
    assert_eq!(whisper_language("auto"), None);
    assert_eq!(whisper_language("bg"), Some("bg".to_string()));
    assert_eq!(
        whisper_language("xx"),
        None,
        "an unknown code is left to detection"
    );
    let old = r#"{"debug_stats":true,"shortcut":"F3"}"#;
    let prefs: Prefs = serde_json::from_str(old).expect("an old file still loads");
    assert_eq!(prefs.language, "auto");
}

// ---- the last transcripts in the tray -----------------------------------

fn transcript(text: &str) -> Transcript {
    Transcript {
        when: "2026-09-26 14:32".to_string(),
        text: text.to_string(),
    }
}

#[test]
fn the_newest_transcript_is_first_and_the_list_stays_at_twenty() {
    let mut list = Vec::new();
    for i in 0..25 {
        remember(&mut list, transcript(&format!("dictation {i}")), false, HISTORY_LIMIT);
    }
    assert_eq!(list.len(), HISTORY_LIMIT);
    assert_eq!(list[0].text, "dictation 24", "newest first");
    assert_eq!(list[19].text, "dictation 5", "the oldest five are gone");
}

#[test]
fn the_same_text_twice_in_a_row_is_one_entry() {
    let mut list = Vec::new();
    remember(&mut list, transcript("hello"), false, HISTORY_LIMIT);
    remember(&mut list, transcript("hello"), false, HISTORY_LIMIT);
    assert_eq!(list.len(), 1);
    remember(&mut list, transcript("again"), false, HISTORY_LIMIT);
    remember(&mut list, transcript("hello"), false, HISTORY_LIMIT);
    assert_eq!(list.len(), 3, "the same text later on is a new entry");
}

#[test]
fn a_dictation_picked_up_after_a_silence_stop_joins_the_previous_entry() {
    let mut list = Vec::new();
    remember(&mut list, transcript("First part."), false, HISTORY_LIMIT);
    remember(&mut list, transcript("Second part."), true, HISTORY_LIMIT);
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].text, "First part. Second part.");
    remember(&mut list, transcript("Second part."), true, HISTORY_LIMIT);
    assert_eq!(list[0].text, "First part. Second part. Second part.", "a continuation is never folded as a repeat");
    remember(&mut list, transcript("Only part."), true, HISTORY_LIMIT);
    assert_eq!(list.len(), 1, "with nothing before it, continuing is just adding");
    let mut empty = Vec::new();
    remember(&mut empty, transcript("Only part."), true, HISTORY_LIMIT);
    assert_eq!(empty.len(), 1);
}

#[test]
fn a_menu_label_is_one_short_line_with_the_time() {
    let long = "word ".repeat(40);
    let label = menu_label(&transcript(&long));
    assert!(label.starts_with("26 Sep 14:32  word word"), "{label}");
    assert!(label.ends_with('…'), "{label}");
    assert!(label.chars().count() <= 14 + 48, "{label}");

    let label = menu_label(&transcript("line one\nline two\ttabbed"));
    assert_eq!(label, "26 Sep 14:32  line one line two tabbed");

    let label = menu_label(&transcript("this & that"));
    assert_eq!(label, "26 Sep 14:32  this && that", "& is a mnemonic in a menu label");

    let label = menu_label(&transcript("Здравей, свят"));
    assert_eq!(label, "26 Sep 14:32  Здравей, свят");
}

#[test]
fn the_transcript_history_survives_being_written_and_read_back() {
    let list = vec![transcript("first"), transcript("second & third")];
    let text = serde_json::to_string(&list).unwrap();
    let loaded: Vec<Transcript> = serde_json::from_str(&text).unwrap();
    assert_eq!(loaded, list);
    assert!(serde_json::from_str::<Vec<Transcript>>("not json").is_err());
}


// ---- typing while you talk ----------------------------------------------

// Stands in for Silero: a frame is speech when it is loud enough. The
// splitter scales every frame to a fixed peak first, so the threshold is on
// the scaled copy.
struct Loud;

impl SpeechDetector for Loud {
    fn is_speech(&mut self, frame: &[f32]) -> bool {
        frame.iter().any(|s| s.abs() > 0.3)
    }
}

fn splitter(pause_ms: u32) -> Segmenter<Loud> {
    Segmenter::new(Loud, pause_ms)
}

#[test]
fn live_settings_start_switched_on() {
    let prefs = Prefs::default();
    assert!(prefs.live_typing);
    assert_eq!(prefs.live_pause_ms, 700);
    assert!(prefs.silence_stop);
    assert_eq!(prefs.silence_stop_ms, 3500);
    assert!(!prefs.auto_resume, "listening on after a stop is opt-in");
    assert_eq!(prefs.auto_resume_ms, 6000);
    assert!(!prefs.auto_enter);
    assert!(prefs.tidy_sentence_ends);
    let old: Prefs = serde_json::from_str(r#"{"pause_shortening":true}"#).unwrap();
    assert!(old.live_typing, "a file from before live typing gets it too");
    assert!(old.silence_stop);
    assert_eq!(old.silence_stop_ms, 3500);
}

#[test]
fn a_soft_syllable_does_not_end_speech_but_a_soft_noise_does_not_start_it() {
    assert!(!still_speech(false, 0.4), "below the start line: not speech");
    assert!(still_speech(false, 0.5), "on it: speech starts");
    assert!(still_speech(true, 0.35), "once started, a softer frame keeps it");
    assert!(!still_speech(true, 0.25), "until it drops under the hold line");
}

#[test]
fn the_pause_bar_fills_from_the_last_word_to_the_cut() {
    let mut splitter = splitter(1000);
    splitter.feed(&blocks(&[(40, 0.0)]));
    near(splitter.pause_progress(), 0.0, 1e-6, "nothing said yet");
    splitter.feed(&blocks(&[(33, 0.1)]));
    near(splitter.pause_progress(), 0.0, 1e-6, "still talking");
    splitter.feed(&blocks(&[(17, 0.0)]));
    near(splitter.pause_progress(), 0.5, 0.02, "half way");
    splitter.feed(&blocks(&[(17, 0.0)]));
    near(splitter.pause_progress(), 0.0, 1e-6, "cut, and the bar starts over");
}

#[test]
fn the_silence_stop_follows_the_speakers_longest_pause() {
    let mut splitter = splitter(700);
    splitter.feed(&blocks(&[(40, 0.0), (33, 0.1)]));
    near(splitter.stop_after_seconds(3.5), 2.0, 1e-6, "no pause yet: the floor");
    // A 0.6 s pause, then more words.
    splitter.feed(&blocks(&[(20, 0.0), (33, 0.1)]));
    near(splitter.stop_after_seconds(3.5), 2.0, 1e-6, "twice 0.6 s is under the floor");
    // A 1.5 s pause, then more words.
    splitter.feed(&blocks(&[(50, 0.0), (33, 0.1)]));
    near(splitter.stop_after_seconds(3.5), 3.0, 0.05, "twice the longest pause");
    // A 4 s pause: the setting caps it.
    splitter.feed(&blocks(&[(134, 0.0), (33, 0.1)]));
    near(splitter.stop_after_seconds(3.5), 3.5, 1e-6, "never over the setting");
    near(splitter.stop_after_seconds(1.5), 1.5, 1e-6, "a setting under the floor wins");

    let mut fresh = self::splitter(700);
    fresh.feed(&blocks(&[(200, 0.0), (33, 0.1)]));
    near(fresh.stop_after_seconds(3.5), 2.0, 1e-6, "the run-up before the first word is not a pause");
}

#[test]
fn a_piece_cut_at_a_hesitation_loses_its_dots_and_fillers() {
    let cases: &[(&str, &str)] = &[
        ("and then, uh, uh...", "and then"),
        ("and then...", "and then"),
        ("and then…", "and then"),
        ("So I went there, um", "So I went there"),
        ("It works. Uh...", "It works."),
        ("What is happening here now is that", "What is happening here now is that"),
        ("Done.", "Done."),
        ("Really?", "Really?"),
        ("uh...", ""),
        ("Здравей, ъъъ...", "Здравей, ъъъ"),
    ];
    for &(given, want) in cases {
        assert_eq!(tidy_end(given), want, "{given:?}");
    }
}

#[test]
fn a_sentence_goes_out_once_the_pause_after_it_is_long_enough() {
    let mut splitter = splitter(1000);
    // 1 s of speech, then silence one frame short of a second.
    let speech = blocks(&[(33, 0.1)]);
    assert!(splitter.feed(&speech).is_empty(), "still talking");
    let short = blocks(&[(33, 0.0)]);
    assert!(splitter.feed(&short).is_empty(), "the pause is not over yet");
    let out = splitter.feed(&blocks(&[(1, 0.0)]));
    assert_eq!(out.len(), 1, "one more frame and the sentence is out");
    // The speech and a 0.3 s tail, not the whole second of silence.
    let frames = out[0].len() / 480;
    assert_eq!(frames, 33 + 10, "speech plus the kept tail");
}

#[test]
fn audio_arriving_in_odd_sized_chunks_is_split_the_same() {
    let audio = blocks(&[(33, 0.1), (40, 0.0), (33, 0.1), (40, 0.0)]);
    let mut whole = splitter(1000);
    let mut out_whole = whole.feed(&audio);
    let mut pieces = splitter(1000);
    let mut out_pieces = Vec::new();
    for chunk in audio.chunks(1000) {
        out_pieces.extend(pieces.feed(chunk));
    }
    assert_eq!(out_whole.len(), 2);
    assert_eq!(out_pieces.len(), 2, "two sentences either way");
    assert_eq!(out_whole.pop().unwrap().len(), out_pieces.pop().unwrap().len());
}

#[test]
fn the_run_up_before_a_sentence_is_short_and_a_lone_click_is_not_a_sentence() {
    let mut splitter = splitter(1000);
    // Five seconds of nothing, then a word.
    assert!(splitter.feed(&blocks(&[(170, 0.0)])).is_empty());
    let out = splitter.feed(&blocks(&[(33, 0.1), (40, 0.0)]));
    assert_eq!(out.len(), 1);
    let frames = out[0].len() / 480;
    assert!(
        frames <= 16 + 33 + 10,
        "at most half a second of run-up is kept, got {frames} frames"
    );

    // One 30 ms blip and a long pause: under 0.3 s of speech, so nothing.
    let mut splitter = self::splitter(1000);
    assert!(splitter.feed(&blocks(&[(1, 0.1), (60, 0.0)])).is_empty());
}

#[test]
fn what_is_left_at_the_end_is_the_last_sentence() {
    let mut splitter = splitter(1000);
    assert!(splitter.feed(&blocks(&[(33, 0.1), (40, 0.0)])).len() == 1);
    assert!(splitter.feed(&blocks(&[(20, 0.1), (5, 0.0)])).is_empty());
    let last = splitter.finish().expect("the unfinished sentence comes out");
    assert!(last.len() / 480 >= 20, "with all of its speech");

    let mut splitter = self::splitter(1000);
    assert!(splitter.feed(&blocks(&[(33, 0.1), (40, 0.0)])).len() == 1);
    assert!(splitter.finish().is_none(), "nothing was said after the last cut");
}

#[test]
fn silence_is_counted_from_the_last_word_and_from_the_start() {
    let mut splitter = splitter(1000);
    splitter.feed(&blocks(&[(100, 0.0)]));
    near(splitter.silence_seconds(), 3.0, 0.05, "quiet from the start");
    splitter.feed(&blocks(&[(10, 0.1)]));
    near(splitter.silence_seconds(), 0.0, 0.001, "a word resets it");
    splitter.feed(&blocks(&[(200, 0.0)]));
    near(splitter.silence_seconds(), 6.0, 0.05, "and it grows again after");
}

#[test]
fn a_quiet_microphone_is_split_like_a_loud_one() {
    // The same sentence at a fifth of the level, a headset at low gain. The
    // detector sees a scaled copy, so the cut lands in the same place.
    let loud = blocks(&[(33, 0.1), (40, 0.0)]);
    let quiet: Vec<f32> = loud.iter().map(|s| s * 0.2).collect();
    let mut a = splitter(1000);
    let mut b = splitter(1000);
    let out_loud = a.feed(&loud);
    let out_quiet = b.feed(&quiet);
    assert_eq!(out_loud.len(), 1);
    assert_eq!(out_quiet.len(), 1);
    assert_eq!(out_loud[0].len(), out_quiet[0].len());
}

// Run with --ignored --nocapture: where the live splitter would cut each of
// the saved recordings, fed in 20 ms chunks as the microphone delivers them.
#[test]
#[ignore]
fn where_the_sentence_splitter_cuts_the_saved_recordings() {
    let dir = crate::storage::get_recordings_dir().expect("recordings folder");
    let mut paths: Vec<_> = fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "wav"))
        .filter(|p| !p.to_string_lossy().contains("model-input"))
        .collect();
    paths.sort();
    for path in paths.iter().rev().take(6) {
        let samples = wav_samples_16k(path);
        let mut splitter = Segmenter::new(crate::vad::Silero::new().unwrap(), 1000);
        let mut cuts = Vec::new();
        for chunk in samples.chunks(320) {
            for sentence in splitter.feed(chunk) {
                cuts.push(sentence.len() as f32 / 16_000.0);
            }
        }
        let last = splitter.finish().map(|s| s.len() as f32 / 16_000.0);
        println!(
            "{:<32} {:5.1}s  sentences {:?}  last {:?}",
            path.file_name().unwrap().to_string_lossy(),
            samples.len() as f32 / 16_000.0,
            cuts.iter().map(|s| format!("{s:.1}")).collect::<Vec<_>>(),
            last.map(|s| format!("{s:.1}"))
        );
    }
}
