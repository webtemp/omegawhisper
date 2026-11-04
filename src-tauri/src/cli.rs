// Transcribing a file from the command line: one audio file in, text out, then
// exit. It runs as its own process. If the app is running it is not touched:
// no tray, no windows, no dictation key here. The settings file is only read.
//
// The audio is split at pauses by the same speech detector the app uses, and
// each piece goes to the model on its own. While that runs, the terminal shows
// the loudness of the whole file, how far it has got, and the piece the model
// is working on. The model reports nothing while it works on a piece, so the
// movement inside a piece is an estimate from the speed of the earlier pieces.

use crate::live::Segmenter;
use crate::managers::{EngineType, GpuChoice, ModelInfo, ModelManager, TranscriptionManager, AVAILABLE_MODELS};
use crate::settings::{load_prefs, whisper_language, Prefs, LANGUAGES};
use std::fs::{self, File};
use std::io::{IsTerminal, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const RATE: usize = 16_000;
// Pieces longer than this are cut at their quietest moment. Moonshine refuses
// more than 64 s, so the ONNX models get 30 s. Whisper copes with any length,
// but a shorter piece keeps the progress on screen moving.
const LONGEST_PIECE_ONNX: f32 = 30.0;
const LONGEST_PIECE_WHISPER: f32 = 90.0;

pub struct Job {
    pub file: PathBuf,
    pub output: Option<PathBuf>,
    pub force: bool,
    pub language: Option<String>,
    pub model: Option<String>,
    pub timestamps: bool,
    pub pause_ms: u32,
    pub cpu: bool,
    pub quiet: bool,
    pub visual: bool,
}

// ---------------------------------------------------------------------------
// The two output streams.
//
// whisper.cpp, ONNX Runtime and the app's own model code print their own
// messages while a model loads. Those would break the drawing and mix into the
// text, so the real stdout and stderr are kept aside for this file's own
// writing, and what everything else prints goes nowhere.

struct Streams {
    out: Box<dyn Write + Send>,
    err: Box<dyn Write + Send>,
    // A second handle on stderr for the panic message.
    err_copy: Option<File>,
    out_terminal: bool,
    err_terminal: bool,
}

fn take_streams() -> Streams {
    let out_terminal = std::io::stdout().is_terminal();
    let err_terminal = std::io::stderr().is_terminal();
    prepare_console();
    match keep_streams_aside() {
        Some((out, err)) => Streams {
            err_copy: err.try_clone().ok(),
            out: Box::new(out),
            err: Box::new(err),
            out_terminal,
            err_terminal,
        },
        None => Streams {
            out: Box::new(std::io::stdout()),
            err: Box::new(std::io::stderr()),
            err_copy: None,
            out_terminal,
            err_terminal,
        },
    }
}

#[cfg(unix)]
fn keep_streams_aside() -> Option<(File, File)> {
    use std::os::unix::io::FromRawFd;
    unsafe {
        let out = libc::dup(libc::STDOUT_FILENO);
        let err = libc::dup(libc::STDERR_FILENO);
        if out < 0 || err < 0 {
            return None;
        }
        let null = libc::open(c"/dev/null".as_ptr(), libc::O_WRONLY);
        if null >= 0 {
            libc::dup2(null, libc::STDOUT_FILENO);
            libc::dup2(null, libc::STDERR_FILENO);
            libc::close(null);
        }
        Some((File::from_raw_fd(out), File::from_raw_fd(err)))
    }
}

#[cfg(windows)]
fn keep_streams_aside() -> Option<(File, File)> {
    use std::os::windows::io::{FromRawHandle, IntoRawHandle};
    use windows_sys::Win32::Foundation::{
        DuplicateHandle, DUPLICATE_SAME_ACCESS, HANDLE, INVALID_HANDLE_VALUE,
    };
    use windows_sys::Win32::System::Console::{
        GetStdHandle, SetStdHandle, STD_ERROR_HANDLE, STD_OUTPUT_HANDLE,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    // Copies of the handles, so the two files never close the same one.
    unsafe fn copy(handle: HANDLE) -> Option<HANDLE> {
        if handle.is_null() || handle == INVALID_HANDLE_VALUE {
            return None;
        }
        let mut copy: HANDLE = std::ptr::null_mut();
        let process = GetCurrentProcess();
        let ok = DuplicateHandle(process, handle, process, &mut copy, 0, 0, DUPLICATE_SAME_ACCESS);
        (ok != 0).then_some(copy)
    }

    unsafe {
        let out = copy(GetStdHandle(STD_OUTPUT_HANDLE))?;
        let err = copy(GetStdHandle(STD_ERROR_HANDLE))?;
        if let Ok(null) = fs::OpenOptions::new().write(true).open("NUL") {
            let handle = null.into_raw_handle();
            SetStdHandle(STD_OUTPUT_HANDLE, handle as HANDLE);
            SetStdHandle(STD_ERROR_HANDLE, handle as HANDLE);
            // whisper.cpp prints through the C runtime, which keeps its own.
            let fd = libc::open_osfhandle(handle as isize, 0);
            if fd >= 0 {
                libc::dup2(fd, 1);
                libc::dup2(fd, 2);
            }
        }
        Some((File::from_raw_handle(out as _), File::from_raw_handle(err as _)))
    }
}

#[cfg(not(any(unix, windows)))]
fn keep_streams_aside() -> Option<(File, File)> {
    None
}

// Windows: the console has to be told to show UTF-8 and to obey the colour and
// cursor codes. Elsewhere terminals already do both.
// The console is shared with the shell that started this, so its code page is
// put back on the way out.
#[cfg(windows)]
static OLD_CODE_PAGE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

#[cfg(windows)]
fn prepare_console() {
    use windows_sys::Win32::System::Console::{
        GetConsoleMode, GetConsoleOutputCP, GetStdHandle, SetConsoleMode, SetConsoleOutputCP,
        ENABLE_VIRTUAL_TERMINAL_PROCESSING, STD_ERROR_HANDLE, STD_OUTPUT_HANDLE,
    };
    unsafe {
        OLD_CODE_PAGE.store(GetConsoleOutputCP(), Ordering::Relaxed);
        SetConsoleOutputCP(65001);
        for which in [STD_OUTPUT_HANDLE, STD_ERROR_HANDLE] {
            let handle = GetStdHandle(which);
            let mut mode = 0;
            if GetConsoleMode(handle, &mut mode) != 0 {
                SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
            }
        }
    }
}

#[cfg(not(windows))]
fn prepare_console() {}

#[cfg(windows)]
fn restore_console() {
    let old = OLD_CODE_PAGE.load(Ordering::Relaxed);
    if old != 0 {
        unsafe {
            windows_sys::Win32::System::Console::SetConsoleOutputCP(old);
        }
    }
}

#[cfg(not(windows))]
fn restore_console() {}

#[cfg(unix)]
fn terminal_width() -> usize {
    let width = File::open("/dev/tty").ok().and_then(|tty| {
        use std::os::unix::io::AsRawFd;
        let mut size: libc::winsize = unsafe { std::mem::zeroed() };
        let ok = unsafe { libc::ioctl(tty.as_raw_fd(), libc::TIOCGWINSZ, &mut size) } == 0;
        (ok && size.ws_col > 0).then_some(size.ws_col as usize)
    });
    width.or_else(columns_from_env).unwrap_or(80)
}

#[cfg(windows)]
fn terminal_width() -> usize {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::System::Console::{GetConsoleScreenBufferInfo, CONSOLE_SCREEN_BUFFER_INFO};
    let width = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("CONOUT$")
        .ok()
        .and_then(|console| {
            let mut info: CONSOLE_SCREEN_BUFFER_INFO = unsafe { std::mem::zeroed() };
            let ok = unsafe { GetConsoleScreenBufferInfo(console.as_raw_handle() as _, &mut info) } != 0;
            let width = (info.srWindow.Right - info.srWindow.Left + 1) as i32;
            (ok && width > 0).then_some(width as usize)
        });
    width.or_else(columns_from_env).unwrap_or(80)
}

#[cfg(not(any(unix, windows)))]
fn terminal_width() -> usize {
    columns_from_env().unwrap_or(80)
}

fn columns_from_env() -> Option<usize> {
    std::env::var("COLUMNS").ok()?.parse().ok().filter(|&c| c > 0)
}

// ---------------------------------------------------------------------------
// The screen. With a terminal on stderr a block of lines at the bottom is
// redrawn ten times a second; finished text is printed above it and scrolls
// up. Without one, only plain lines are printed.

const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
const LABEL: usize = 13;
const MIN_WIDTH: usize = 30;

const GREEN: &str = "32";
const YELLOW: &str = "33";
const GREY: &str = "90";
const RED: &str = "31";
const BOLD: &str = "1";

enum View {
    Blank,
    Status {
        label: String,
        fraction: Option<f32>,
        since: Instant,
    },
    Transcribe(Box<Progress>),
}

struct Progress {
    // Loudness of the whole file, 0 to 1, in many narrow columns.
    overview: Vec<f32>,
    total_seconds: f32,
    // Start and end of every piece, in seconds.
    pieces: Vec<(f32, f32)>,
    done: usize,
    // The piece the model has now: its loudness and when it was handed over.
    current: Option<(Vec<f32>, Instant)>,
    audio_done: f32,
    compute_done: f32,
    started: Instant,
}

impl Progress {
    fn piece_seconds(&self) -> f32 {
        self.pieces.iter().map(|(a, b)| b - a).sum::<f32>().max(0.001)
    }

    // Seconds of computing per second of audio, from the pieces done so far.
    // Before the first one, a guess.
    fn cost(&self) -> f32 {
        if self.audio_done > 0.0 {
            self.compute_done / self.audio_done
        } else {
            0.15
        }
    }

    // How far the model is into the current piece, by the estimate. Held
    // under the end, so the drawing never shows a piece done before its text.
    fn current_fraction(&self) -> f32 {
        let Some((_, since)) = &self.current else {
            return 0.0;
        };
        let (start, end) = self.pieces[self.done];
        let expected = ((end - start) * self.cost()).max(0.05);
        (since.elapsed().as_secs_f32() / expected).min(0.95)
    }

    fn fraction(&self) -> f32 {
        let finished: f32 = self.pieces[..self.done].iter().map(|(a, b)| b - a).sum();
        let current = if self.current.is_some() {
            let (a, b) = self.pieces[self.done];
            (b - a) * self.current_fraction()
        } else {
            0.0
        };
        ((finished + current) / self.piece_seconds()).min(1.0)
    }

    // Everything before this second is done.
    fn position(&self) -> f32 {
        self.pieces.get(self.done).map_or(self.total_seconds, |p| p.0)
    }
}

struct Screen {
    err: Box<dyn Write + Send>,
    live: bool,
    quiet: bool,
    color: bool,
    // How many characters each line of the block took when it was drawn.
    drawn: Vec<usize>,
    tick: usize,
    view: View,
}

impl Screen {
    fn paint(&self, text: &str, code: &str) -> String {
        if self.color {
            format!("\x1b[{code}m{text}\x1b[0m")
        } else {
            text.to_string()
        }
    }

    // Back to the top of the block and wipe it. A terminal made narrower since
    // the drawing wraps its lines, so the rows are counted at today's width.
    fn clear(&mut self) -> String {
        if self.drawn.is_empty() {
            return String::new();
        }
        let width = terminal_width().max(1);
        let rows: usize = self.drawn.iter().map(|w| w.div_ceil(width).max(1)).sum();
        self.drawn.clear();
        format!("\x1b[{rows}F\x1b[J")
    }

    fn redraw(&mut self) {
        if !self.live {
            return;
        }
        let width = terminal_width();
        // Too narrow to draw anything readable: the text lines still come.
        let lines = if width < MIN_WIDTH { Vec::new() } else { self.render(width) };
        let mut text = self.clear();
        for line in &lines {
            let line = fit(line, width - 1);
            self.drawn.push(visible_len(&line));
            text.push_str(&line);
            text.push_str("\x1b[K\n");
        }
        let _ = self.err.write_all(text.as_bytes());
        let _ = self.err.flush();
    }

    fn line(&mut self, text: &str) {
        let mut out = if self.live { self.clear() } else { String::new() };
        out.push_str(text);
        out.push('\n');
        let _ = self.err.write_all(out.as_bytes());
        let _ = self.err.flush();
        self.redraw();
    }

    fn render(&self, width: usize) -> Vec<String> {
        let spin = SPINNER[self.tick % SPINNER.len()];
        let bar_width = width.saturating_sub(LABEL + 6).clamp(10, 120);
        match &self.view {
            View::Blank => Vec::new(),
            View::Status { label, fraction, since } => {
                let head = format!("{spin} {label}   {}", clock(since.elapsed().as_secs_f32()));
                let mut lines = vec![format!(" {}", cut(&head, width.saturating_sub(2)))];
                if let Some(f) = fraction {
                    lines.push(format!(
                        "   {} {:3.0}%",
                        self.paint(&fill_bar(*f, bar_width + LABEL - 3), GREEN),
                        f * 100.0
                    ));
                }
                lines
            }
            View::Transcribe(p) => self.render_progress(p, width, bar_width, spin),
        }
    }

    fn render_progress(&self, p: &Progress, width: usize, bar_width: usize, spin: char) -> Vec<String> {
        let pad = " ".repeat(LABEL);
        let position = p.position();
        let current = p.current.as_ref().map(|_| p.pieces[p.done]);

        // The whole file, coloured by what is done, under way and still to come.
        let columns = squeeze(&p.overview, bar_width);
        let mut whole = String::new();
        let mut run = String::new();
        let mut run_color = "";
        for (i, level) in columns.iter().enumerate() {
            let t = (i as f32 + 0.5) / bar_width as f32 * p.total_seconds;
            let color = match current {
                Some((a, b)) if t >= a && t < b => YELLOW,
                _ if t < position => GREEN,
                _ => GREY,
            };
            if color != run_color && !run.is_empty() {
                whole.push_str(&self.paint(&run, run_color));
                run.clear();
            }
            run_color = color;
            run.push(bar_char(*level));
        }
        whole.push_str(&self.paint(&run, run_color));

        let fraction = p.fraction();
        let mut lines = vec![
            format!(" {:<w$}{whole}", "Whole file", w = LABEL - 1),
            format!("{pad}{} {:3.0}%", self.paint(&fill_bar(fraction, bar_width), GREEN), fraction * 100.0),
        ];

        // The piece the model is working on.
        match (&p.current, current) {
            (Some((levels, _)), Some((a, b))) => {
                let head = format!(
                    "{spin} Piece {} of {}   {} - {}   {} s",
                    p.done + 1,
                    p.pieces.len(),
                    clock(a),
                    clock(b),
                    (b - a) as u32
                );
                lines.push(format!(" {:<w$}{}", "Now", cut(&head, bar_width + 5), w = LABEL - 1));
                let shown = squeeze(levels, bar_width);
                let lit = (p.current_fraction() * shown.len() as f32) as usize;
                let (done, rest): (String, String) = (
                    shown[..lit].iter().map(|l| bar_char(*l)).collect(),
                    shown[lit..].iter().map(|l| bar_char(*l)).collect(),
                );
                lines.push(format!("{pad}{}{}", self.paint(&done, GREEN), self.paint(&rest, GREY)));
            }
            _ => {
                let head = format!("{spin} Piece {} of {}", p.done.min(p.pieces.len()), p.pieces.len());
                lines.push(format!(" {:<w$}{head}", "Now", w = LABEL - 1));
                lines.push(String::new());
            }
        }

        let elapsed = p.started.elapsed().as_secs_f32();
        let left_audio: f32 = p.piece_seconds() * (1.0 - fraction);
        let left = if p.audio_done > 0.0 {
            format!("~{}", clock(left_audio * p.cost()))
        } else {
            "measuring".to_string()
        };
        let speed = if p.compute_done > 0.0 {
            format!("{:.1}x real time", p.audio_done / p.compute_done)
        } else {
            "-".to_string()
        };
        let footer = format!(" Elapsed {}   Left {}   Model speed {}", clock(elapsed), left, speed);
        lines.push(self.paint(&cut(&footer, width.saturating_sub(1)), GREY));
        lines
    }
}

// Set while a self-test check runs, so a caught panic is reported as the
// check's, not as the end of the program.
static IN_CHECK: AtomicBool = AtomicBool::new(false);

struct Ui {
    screen: Arc<Mutex<Screen>>,
    stop: Arc<AtomicBool>,
    ticker: Option<JoinHandle<()>>,
}

impl Ui {
    fn start(err: Box<dyn Write + Send>, err_copy: Option<File>, live: bool, quiet: bool) -> Self {
        let color = std::env::var_os("NO_COLOR").is_none() && live;
        DRAWING.store(live, Ordering::SeqCst);
        let screen = Arc::new(Mutex::new(Screen {
            err,
            live,
            quiet,
            color,
            drawn: Vec::new(),
            tick: 0,
            view: View::Blank,
        }));
        if let Some(copy) = err_copy {
            // The block is wiped before the message so the two do not mix.
            // try_lock: the panic may have happened with the screen locked.
            let screen = screen.clone();
            let copy = Mutex::new(copy);
            std::panic::set_hook(Box::new(move |info| {
                let in_check = IN_CHECK.load(Ordering::Relaxed);
                let wipe = |s: &mut Screen| {
                    // The program is ending: no more drawing after the message.
                    if !in_check {
                        s.live = false;
                    }
                    s.clear()
                };
                let wipe = match screen.try_lock() {
                    Ok(mut s) => wipe(&mut s),
                    Err(std::sync::TryLockError::Poisoned(p)) => wipe(&mut p.into_inner()),
                    Err(std::sync::TryLockError::WouldBlock) => String::new(),
                };
                if !in_check {
                    restore_console();
                }
                if let Ok(mut f) = copy.lock() {
                    let what = if in_check {
                        "a self-test check stopped on an internal error"
                    } else {
                        "omegawhisper stopped on an internal error"
                    };
                    let _ = writeln!(f, "{wipe}{what}: {info}");
                }
            }));
        }
        let stop = Arc::new(AtomicBool::new(false));
        let ticker = live.then(|| {
            let screen = screen.clone();
            let stop = stop.clone();
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    // A panic caught in the self-test marks the lock as broken;
                    // the screen itself is fine, so drawing goes on.
                    let mut s = screen.lock().unwrap_or_else(|e| e.into_inner());
                    s.tick += 1;
                    s.redraw();
                    drop(s);
                    std::thread::sleep(Duration::from_millis(100));
                }
            })
        });
        Self { screen, stop, ticker }
    }

    fn with<T>(&self, f: impl FnOnce(&mut Screen) -> T) -> T {
        let mut screen = self.screen.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut screen)
    }

    fn live(&self) -> bool {
        self.with(|s| s.live)
    }

    fn paint(&self, text: &str, code: &str) -> String {
        self.with(|s| s.paint(text, code))
    }

    // A line of information; not shown with --quiet.
    fn say(&self, text: &str) {
        self.with(|s| {
            if !s.quiet {
                s.line(text)
            }
        });
    }

    // Shown whatever the flags: something was left out.
    fn warn(&self, text: &str) {
        self.with(|s| {
            let text = s.paint(&format!("Warning: {text}"), YELLOW);
            s.line(&text)
        });
    }

    // Shown whatever the flags.
    fn error(&self, text: &str) {
        self.with(|s| {
            let text = s.paint(&format!("Error: {text}"), RED);
            s.line(&text)
        });
    }

    fn status(&self, label: &str, fraction: Option<f32>) {
        self.with(|s| {
            let since = match &s.view {
                View::Status { label: old, since, .. } if old == label => *since,
                _ => Instant::now(),
            };
            s.view = View::Status {
                label: label.to_string(),
                fraction,
                since,
            };
        });
    }

    fn blank(&self) {
        self.with(|s| {
            s.view = View::Blank;
            s.redraw();
        });
    }

    fn progress(&self, f: impl FnOnce(&mut Progress)) {
        self.with(|s| {
            if let View::Transcribe(p) = &mut s.view {
                f(p);
            }
        });
    }

    fn finish(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.ticker.take() {
            let _ = t.join();
        }
        self.with(|s| {
            s.view = View::Blank;
            s.redraw();
            let _ = s.err.flush();
        });
        restore_console();
    }
}

// ---------------------------------------------------------------------------
// Small helpers for the drawing.

fn clock(seconds: f32) -> String {
    let s = seconds.max(0.0) as u64;
    if s >= 3600 {
        format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
    } else {
        format!("{}:{:02}", s / 60, s % 60)
    }
}

// The time mark before each piece in the text.
fn stamp(seconds: f32, long: bool) -> String {
    let s = seconds.max(0.0) as u64;
    if long {
        format!("[{}:{:02}:{:02}]", s / 3600, s / 60 % 60, s % 60)
    } else {
        format!("[{:02}:{:02}]", s / 60, s % 60)
    }
}

fn cut(text: &str, width: usize) -> String {
    text.chars().take(width).collect()
}

// Characters a line takes on screen, leaving out the colour codes.
fn visible_len(line: &str) -> usize {
    let mut count = 0;
    let mut in_code = false;
    for c in line.chars() {
        match (in_code, c) {
            (false, '\x1b') => in_code = true,
            (false, _) => count += 1,
            (true, c) if c.is_ascii_alphabetic() => in_code = false,
            _ => {}
        }
    }
    count
}

// A line cut to `width` characters on screen, colour codes kept and closed.
fn fit(line: &str, width: usize) -> String {
    if visible_len(line) <= width {
        return line.to_string();
    }
    let mut out = String::new();
    let mut count = 0;
    let mut in_code = false;
    for c in line.chars() {
        if in_code {
            out.push(c);
            in_code = !c.is_ascii_alphabetic();
        } else if c == '\x1b' {
            out.push(c);
            in_code = true;
        } else if count < width {
            out.push(c);
            count += 1;
        }
    }
    out.push_str("\x1b[0m");
    out
}

fn bar_char(level: f32) -> char {
    BARS[((level.clamp(0.0, 1.0) * 7.0).round() as usize).min(7)]
}

fn fill_bar(fraction: f32, width: usize) -> String {
    const PARTS: [char; 8] = [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉'];
    let filled = fraction.clamp(0.0, 1.0) * width as f32;
    let whole = filled as usize;
    let mut bar = "█".repeat(whole);
    if whole < width {
        let part = ((filled - whole as f32) * 8.0) as usize;
        bar.push(if part == 0 { '░' } else { PARTS[part] });
        bar.push_str(&"░".repeat(width - whole - 1));
    }
    bar
}

// Loudness of each column, 0 to 1, on a 45 dB scale under the loudest column.
fn loudness_columns(samples: &[f32], columns: usize) -> Vec<f32> {
    if samples.is_empty() || columns == 0 {
        return vec![0.0; columns];
    }
    let len = samples.len();
    let rms: Vec<f32> = (0..columns)
        .map(|c| {
            let a = c * len / columns;
            let b = ((c + 1) * len / columns).max(a + 1).min(len);
            let part = &samples[a.min(len - 1)..b];
            (part.iter().map(|s| s * s).sum::<f32>() / part.len() as f32).sqrt()
        })
        .collect();
    let loudest = rms.iter().cloned().fold(0.0f32, f32::max);
    if loudest <= 0.0 {
        return vec![0.0; columns];
    }
    rms.iter()
        .map(|r| ((20.0 * (r / loudest).max(1e-6).log10() + 45.0) / 45.0).clamp(0.0, 1.0))
        .collect()
}

// Many columns into fewer, keeping the loudest of each group.
fn squeeze(levels: &[f32], width: usize) -> Vec<f32> {
    if levels.is_empty() || width == 0 {
        return vec![0.0; width];
    }
    (0..width)
        .map(|c| {
            let a = c * levels.len() / width;
            let b = ((c + 1) * levels.len() / width).max(a + 1).min(levels.len());
            levels[a.min(levels.len() - 1)..b].iter().cloned().fold(0.0, f32::max)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Reading the audio.

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// The samples of a WAV file, all channels averaged into one, at the file's
/// own rate. Reads RF64 too, and files whose data size was never filled in
/// (a recording cut off before it was closed): their data runs to the end.
pub(crate) fn read_wav(bytes: &[u8]) -> Result<(Vec<f32>, u32), String> {
    let riff = bytes.len() >= 12 && (&bytes[0..4] == b"RIFF" || &bytes[0..4] == b"RF64");
    if !riff || &bytes[8..12] != b"WAVE" {
        return Err("not a WAV file".into());
    }
    let mut format = None;
    let mut data = None;
    // RF64 keeps the real data size in its ds64 block.
    let mut size64: Option<usize> = None;
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32_at(bytes, pos + 4) as usize;
        let start = pos + 8;
        let end = start.saturating_add(len).min(bytes.len());
        match id {
            b"fmt " => {
                if end - start < 16 {
                    return Err("the WAV format block is too short".into());
                }
                let b = &bytes[start..end];
                let mut code = u16_at(b, 0);
                // WAVE_FORMAT_EXTENSIBLE keeps the real code in its sub-format.
                if code == 0xFFFE && b.len() >= 26 {
                    code = u16_at(b, 24);
                }
                format = Some((code, u16_at(b, 2), u32_at(b, 4), u16_at(b, 14)));
            }
            b"ds64" if end - start >= 16 => {
                let b = &bytes[start..end];
                size64 = Some(u64::from_le_bytes(b[8..16].try_into().unwrap()) as usize);
            }
            b"data" => {
                let real = match (len, size64) {
                    (l, Some(big)) if l == u32::MAX as usize => start.saturating_add(big).min(bytes.len()),
                    // Never filled in: the data runs to the end of the file.
                    (0, _) => bytes.len(),
                    (l, None) if l == u32::MAX as usize => bytes.len(),
                    _ => end,
                };
                data = Some(&bytes[start..real]);
                if real == bytes.len() {
                    break;
                }
            }
            _ => {}
        }
        if format.is_some() && data.is_some() {
            break;
        }
        pos = start.saturating_add(len).saturating_add(len & 1);
    }
    let (code, channels, rate, bits) = format.ok_or("the WAV file has no format block")?;
    let data = data.ok_or("the WAV file has no audio data")?;
    if channels == 0 {
        return Err("the WAV file says it has no channels".into());
    }
    if !(1_000..=768_000).contains(&rate) {
        return Err(format!("the WAV file says its sample rate is {rate} Hz, which is not a real rate"));
    }

    let decode: fn(&[u8]) -> f32 = match (code, bits) {
        (1, 8) => |c| (c[0] as f32 - 128.0) / 128.0,
        (1, 16) => |c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0,
        (1, 24) => |c| (i32::from_le_bytes([0, c[0], c[1], c[2]]) >> 8) as f32 / 8_388_608.0,
        (1, 32) => |c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 / 2_147_483_648.0,
        (3, 32) => |c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
        (3, 64) => |c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32,
        _ => return Err(format!("this WAV encoding is not supported (format {code}, {bits} bits)")),
    };

    // Straight to one channel, so a long stereo file is never held as stereo.
    let width = bits as usize / 8;
    let channels = channels as usize;
    let mono = data
        .chunks_exact(width * channels)
        .map(|frame| frame.chunks_exact(width).map(decode).sum::<f32>() / channels as f32)
        .collect();
    Ok((mono, rate))
}

/// The file types `-f` reads, named in the errors.
const SUPPORTED_TYPES: &str = "WAV, MP3, M4A (AAC or Apple Lossless), MP4 (its sound), FLAC, OGG (Vorbis)";

/// What `read_audio` gives back.
#[derive(Debug)]
pub(crate) struct Decoded {
    /// All channels averaged into one, at the file's own rate.
    pub(crate) samples: Vec<f32>,
    pub(crate) rate: u32,
    /// Set when part of the file could not be read and was left out.
    pub(crate) warning: Option<String>,
}

/// The samples of an audio file. WAV goes to `read_wav`; every other type is
/// read with symphonia. `name` is the file's name or path: its extension
/// tells symphonia which format to try first.
pub(crate) fn read_audio(bytes: &[u8], name: &str) -> Result<Decoded, String> {
    let wav = bytes.len() >= 12 && (&bytes[0..4] == b"RIFF" || &bytes[0..4] == b"RF64") && &bytes[8..12] == b"WAVE";
    if wav {
        let (samples, rate) = read_wav(bytes)?;
        return Ok(Decoded { samples, rate, warning: None });
    }
    if bytes.is_empty() {
        return Err("the file is empty".into());
    }
    read_compressed(bytes, name)
}

/// A readable name for the codec of a sound track, for the error when it
/// cannot be decoded.
fn codec_name(params: &symphonia::core::codecs::audio::AudioCodecParameters) -> String {
    use symphonia::core::codecs::audio::well_known::profiles::{CODEC_PROFILE_AAC_HE, CODEC_PROFILE_AAC_HE_V2};
    use symphonia::core::codecs::audio::well_known::*;
    use symphonia::core::codecs::audio::CODEC_ID_NULL_AUDIO;
    let name = match params.codec {
        CODEC_ID_AAC => match params.profile {
            Some(CODEC_PROFILE_AAC_HE) => "HE-AAC",
            Some(CODEC_PROFILE_AAC_HE_V2) => "HE-AAC v2",
            // Symphonia decodes only plain AAC (AAC-LC), mono or stereo.
            _ => "AAC of a kind other than plain AAC-LC (for example HE-AAC, or more than 2 channels)",
        },
        CODEC_ID_ALAC => "Apple Lossless (ALAC)",
        CODEC_ID_OPUS => "Opus",
        CODEC_ID_AC3 => "Dolby Digital (AC-3)",
        CODEC_ID_EAC3 => "Dolby Digital Plus (E-AC-3)",
        CODEC_ID_AC4 => "Dolby AC-4",
        CODEC_ID_DCA => "DTS",
        CODEC_ID_TRUEHD => "Dolby TrueHD",
        CODEC_ID_WMA => "Windows Media Audio",
        CODEC_ID_SPEEX => "Speex",
        CODEC_ID_MP1 => "MPEG Layer 1 (MP1)",
        CODEC_ID_MP2 => "MPEG Layer 2 (MP2)",
        CODEC_ID_MP3 => "MP3",
        CODEC_ID_VORBIS => "Vorbis",
        CODEC_ID_FLAC => "FLAC",
        CODEC_ID_WAVPACK => "WavPack",
        CODEC_ID_MONKEYS_AUDIO => "Monkey's Audio",
        CODEC_ID_NULL_AUDIO => "an unknown codec",
        other => return format!("codec {other}"),
    };
    name.into()
}

/// The first sound track a decoder can be made for, and that decoder. The
/// video tracks of an MP4 are passed over.
fn open_sound_track(
    format: &dyn symphonia::core::formats::FormatReader,
) -> Result<(u32, Box<dyn symphonia::core::codecs::audio::AudioDecoder>), String> {
    use symphonia::core::codecs::audio::well_known::{CODEC_ID_MP1, CODEC_ID_MP2, CODEC_ID_MP3};
    use symphonia::core::codecs::audio::AudioDecoderOptions;
    let mut refused = None;
    for track in format.tracks() {
        let Some(params) = track.codec_params.as_ref().and_then(|p| p.audio()) else {
            continue;
        };
        // For MPEG audio the decoder's own trimming is off: it also drops
        // every frame past the length the file states, and that length is
        // only a guess for an MP3 without a header, or wrong for two MP3s
        // joined into one. The start delay is cut after decoding instead.
        let mpeg = [CODEC_ID_MP1, CODEC_ID_MP2, CODEC_ID_MP3].contains(&params.codec);
        let options = AudioDecoderOptions::default().gapless(!mpeg);
        match symphonia::default::get_codecs().make_audio_decoder(params, &options) {
            Ok(decoder) => return Ok((track.id, decoder)),
            Err(_) => {
                refused.get_or_insert_with(|| codec_name(params));
            }
        }
    }
    Err(match refused {
        Some(codec) => format!("the sound in this file is {codec}, which cannot be read here. Supported: {SUPPORTED_TYPES}"),
        None => "the file has no sound track".into(),
    })
}

/// Appends `samples`, at `from` Hz, to `out` at `to` Hz, by drawing a straight
/// line between neighbouring samples. Only for the rare file whose sample rate
/// changes partway through; the conversion to 16 kHz is done later by `to_16k`.
fn append_at_rate(out: &mut Vec<f32>, samples: &[f32], from: u32, to: u32) {
    let Some(&last) = samples.last() else {
        return;
    };
    let count = (samples.len() as u64 * to as u64 / from as u64) as usize;
    let step = from as f64 / to as f64;
    out.extend((0..count).map(|i| {
        let at = i as f64 * step;
        let j = at as usize;
        let a = samples.get(j).copied().unwrap_or(last);
        let b = samples.get(j + 1).copied().unwrap_or(last);
        a + (b - a) * (at - j as f64) as f32
    }));
}

/// MP3, M4A, MP4, FLAC and OGG, read with symphonia. A panic inside symphonia
/// ends only the reading of this file, with an error; its message is kept
/// off the screen while the file is read.
fn read_compressed(bytes: &[u8], name: &str) -> Result<Decoded, String> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| decode_compressed(bytes, name)));
    std::panic::set_hook(hook);
    result.unwrap_or_else(|panic| {
        let why = panic
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "no reason given".into());
        Err(format!("the file could not be read (the decoder failed: {why})"))
    })
}

fn decode_compressed(bytes: &[u8], name: &str) -> Result<Decoded, String> {
    use symphonia::core::errors::Error;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::io::MediaSourceStream;

    let extension = Path::new(name).extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase);
    let mut hint = Hint::new();
    if let Some(extension) = &extension {
        hint.with_extension(extension);
    }
    // Reads straight from the bytes already in memory; nothing is copied.
    let source = MediaSourceStream::new(Box::new(std::io::Cursor::new(bytes)), Default::default());
    let mut format = symphonia::default::get_probe()
        .probe(&hint, source, Default::default(), Default::default())
        .map_err(|e| match (e, extension.as_deref()) {
            (Error::LimitError(what), _) => format!("the file could not be read: it goes over a safety limit ({what})"),
            (_, Some(ext @ ("mp3" | "m4a" | "mp4" | "m4v" | "aac" | "flac" | "ogg" | "oga" | "mov"))) => format!(
                "the file is named .{ext} but no audio in a known format was found in it. It may be damaged or of another type. Supported: {SUPPORTED_TYPES}"
            ),
            _ => format!("this file type is not supported. Supported: {SUPPORTED_TYPES}"),
        })?;

    let (mut track_id, mut decoder) = open_sound_track(format.as_ref())?;
    let track = format.tracks().iter().find(|t| t.id == track_id);
    let params = track.and_then(|t| t.codec_params.as_ref()).and_then(|p| p.audio()).cloned();
    let track_gaps = (track.and_then(|t| t.delay), track.and_then(|t| t.padding));

    // Room for the whole track up front when its length is known. The length
    // is only an estimate for some files (an MP3 without a header that
    // counts its frames), and one sample over it would double the memory, so
    // 2% and a second more are added. Capped at 16 samples per byte of file,
    // more than any real compressed file holds, so a wrong length in a
    // damaged header cannot ask for gigabytes.
    let mut mono: Vec<f32> = Vec::new();
    // The length the file states, in frames at its own rate.
    let mut stated = track.and_then(|t| t.num_frames);
    if let Some(frames) = stated {
        let second = params.as_ref().and_then(|p| p.sample_rate).unwrap_or(48_000) as u64;
        let room = frames + frames / 50 + second;
        let _ = mono.try_reserve(room.min(bytes.len() as u64 * 16) as usize);
    }
    // The rate of `mono`: that of the first decoded packet.
    let mut rate: Option<u32> = None;
    // Audio at another rate than `mono`, kept apart until it is converted.
    let mut other: Vec<f32> = Vec::new();
    let mut other_rate = 0;
    // One packet's samples, all channels side by side; reused for every packet.
    let mut interleaved: Vec<f32> = Vec::new();
    let mut damaged = 0usize;
    // Frames as the decoder handed them out, before any rate conversion.
    let mut decoded = 0u64;
    // Why the reading stopped before the end of the file, if it did.
    let mut stopped: Option<String> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            // The list of tracks changed (one OGG file joined after another):
            // pick the sound track again with a new decoder.
            Err(Error::ResetRequired) => {
                (track_id, decoder) = open_sound_track(format.as_ref())?;
                stated = None;
                continue;
            }
            Err(e) => {
                if mono.is_empty() && other.is_empty() {
                    return Err(format!("the file could not be read. It may be damaged or cut short ({e})"));
                }
                // Cut off or damaged after some sound was read: keep what
                // was read, and say so.
                stopped = Some(match e {
                    Error::IoError(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => "the file is cut off".into(),
                    e => e.to_string(),
                });
                break;
            }
        };
        if packet.track_id != track_id {
            continue;
        }
        let buffer = match decoder.decode(&packet) {
            Ok(buffer) => buffer,
            // One damaged packet, or one using a feature the decoder lacks
            // (some rare parts of AAC): skip it and go on with the next.
            Err(Error::DecodeError(_) | Error::IoError(_) | Error::Unsupported(_)) => {
                damaged += 1;
                continue;
            }
            Err(Error::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(e) => return Err(format!("the sound could not be decoded: {e}")),
        };
        if buffer.frames() == 0 {
            continue;
        }
        // Channels and rate are read from every packet, as they may change
        // partway through a file.
        let channels = buffer.spec().channels().count().max(1);
        let packet_rate = buffer.spec().rate();
        if !(1_000..=768_000).contains(&packet_rate) {
            return Err(format!("the file says its sample rate is {packet_rate} Hz, which is not a real rate"));
        }
        decoded += buffer.frames() as u64;
        buffer.copy_to_vec_interleaved(&mut interleaved);
        let frames = interleaved.chunks_exact(channels).map(|frame| frame.iter().sum::<f32>() / channels as f32);
        let to = *rate.get_or_insert(packet_rate);
        if packet_rate == to {
            append_at_rate(&mut mono, &other, other_rate, to);
            other.clear();
            mono.extend(frames);
        } else {
            if packet_rate != other_rate {
                append_at_rate(&mut mono, &other, other_rate, to);
                other.clear();
                other_rate = packet_rate;
            }
            other.extend(frames);
        }
    }
    let rate = rate.unwrap_or(0);
    append_at_rate(&mut mono, &other, other_rate, rate);
    if mono.is_empty() {
        return Err(if damaged > 0 {
            format!("none of the sound in the file could be decoded ({damaged} damaged packets)")
        } else {
            "the file holds no audio".into()
        });
    }

    use symphonia::core::codecs::audio::well_known::{CODEC_ID_AAC, CODEC_ID_MP1, CODEC_ID_MP2, CODEC_ID_MP3};
    let aac = params.as_ref().is_some_and(|p| p.codec == CODEC_ID_AAC);
    let mpeg = params.as_ref().is_some_and(|p| [CODEC_ID_MP1, CODEC_ID_MP2, CODEC_ID_MP3].contains(&p.codec));
    // The start delay and end padding an MP3's LAME header states. With no
    // such header the stated length is not trusted for the "cut short"
    // check: for a header-less MP3 it is only a guess.
    let (delay, padding) = track_gaps;
    if mpeg && delay.is_none() {
        stated = None;
    }
    let plain = params.as_ref().is_some_and(plain_aac);

    // The reader can also stop without an error at a cut: an MP3 cut in
    // half ends cleanly after its last whole frame. When the file states its
    // length, more than a second or 2% short of it is reported as a cut.
    // Only an estimate for some MP3 files, hence the margin. Not for HE-AAC:
    // decoded without its SBR part, it may come out at half the stated count.
    if let (None, Some(stated)) = (&stopped, stated) {
        let margin = (rate as u64).max(stated / 50);
        if decoded == mono.len() as u64 && decoded + margin < stated && (plain || !aac) {
            stopped = Some(format!("the file says it holds {} of sound", clock(stated as f32 / rate as f32)));
        }
    }

    // An MP3 with a LAME header: cut the start delay it states, and the end
    // padding when the file is exactly as long as it says (so nothing was
    // joined on after it).
    if mpeg && decoded == mono.len() as u64 {
        if let Some(delay) = delay {
            if let Some(frames) = stated {
                let padding = padding.unwrap_or(0) as u64;
                if decoded == frames + delay as u64 + padding {
                    mono.truncate((frames + delay as u64) as usize);
                }
            }
            mono.drain(..(delay as usize).min(mono.len()));
        }
    }

    // AAC starts with silence the encoder put there; the decoder hands it
    // out as sound. Cut it, so the time marks match the original.
    if aac && decoded == mono.len() as u64 {
        let tags = itunes_gapless(&mut *format);
        let (start, length) = aac_priming(bytes, tags, plain, decoded, rate);
        mono.drain(..start.min(mono.len()));
        if let Some(length) = length {
            mono.truncate(length);
        }
    }

    let seconds = mono.len() as f32 / rate as f32;
    let warning = match (stopped, damaged) {
        (Some(why), _) => Some(format!(
            "The file ends early or is damaged at {}: {why}. The text stops there.",
            clock(seconds)
        )),
        (None, 1) => Some("1 damaged piece of the file was skipped.".into()),
        (None, n) if n > 1 => Some(format!("{n} damaged pieces of the file were skipped.")),
        _ => None,
    };
    Ok(Decoded { samples: mono, rate, warning })
}

// The silence AAC-LC encoders put before the sound, in samples: 2112 from
// Apple's encoders (afconvert, iTunes, Music) and from Nero and FDK. Used only
// when the file does not state its own.
const AAC_PRIMING: usize = 2112;

/// Whether the AAC track is plain AAC-LC. Its first 5 bits name the kind of
/// AAC, 2 for plain, and a longer setup block means an extension (the SBR of
/// HE-AAC) is signalled after it. HE-AAC is decoded here without its SBR
/// part, at half the rate, so the usual 2112 does not apply to it.
fn plain_aac(params: &symphonia::core::codecs::audio::AudioCodecParameters) -> bool {
    match params.extra_data.as_deref() {
        Some(setup) => setup.len() == 2 && setup[0] >> 3 == 2,
        None => false,
    }
}

/// The silence before the sound and the length of the sound, as an iTunes
/// "iTunSMPB" tag states them: (before, length, after), in samples.
fn itunes_gapless(format: &mut dyn symphonia::core::formats::FormatReader) -> Option<(u64, u64, u64)> {
    use symphonia::core::meta::RawValue;
    let mut metadata = format.metadata();
    let revision = metadata.skip_to_latest()?;
    let tag = revision.media.tags.iter().find(|t| t.raw.key.ends_with("iTunSMPB"))?;
    let RawValue::String(text) = &tag.raw.value else {
        return None;
    };
    // " 00000000 00000840 00000040 000000000002AF80 ...": a zero, the samples
    // before, the samples after, and the length, all in hexadecimal.
    let fields: Vec<u64> = text.split_whitespace().map(|f| u64::from_str_radix(f, 16)).collect::<Result<_, _>>().ok()?;
    match fields[..] {
        [_, before, after, length, ..] => Some((before, length, after)),
        _ => None,
    }
}

/// Samples to cut from the front of `decoded` AAC samples at `rate` Hz, and
/// the length to keep if the file states it. In this order: the iTunSMPB tag
/// when its numbers add up to what was decoded (at half the count for HE-AAC
/// decoded without SBR); the edit list of an MP4 file; for plain AAC-LC in an
/// MP4 file stating neither, the usual 2112. ADTS (.aac) files state nothing
/// and nothing is cut.
fn aac_priming(
    bytes: &[u8],
    tags: Option<(u64, u64, u64)>,
    plain: bool,
    decoded: u64,
    rate: u32,
) -> (usize, Option<usize>) {
    if let Some((before, length, after)) = tags {
        let total = before + length + after;
        if total == decoded {
            return (before as usize, Some(length as usize));
        }
        if total == decoded * 2 {
            return ((before / 2) as usize, Some((length / 2) as usize));
        }
    }
    let mp4 = bytes.len() >= 8 && &bytes[4..8] == b"ftyp";
    if !mp4 {
        return (0, None);
    }
    match mp4_edit_start(bytes) {
        Some(seconds) => ((seconds * rate as f64).round() as usize, None),
        None if plain => (AAC_PRIMING, None),
        None => (0, None),
    }
}

/// The boxes inside `data`, an MP4 box body or the whole file, as (name, body).
fn mp4_boxes(data: &[u8]) -> impl Iterator<Item = (&[u8], &[u8])> {
    let mut at = 0usize;
    std::iter::from_fn(move || {
        let head = data.get(at..at + 8)?;
        let name = &head[4..8];
        let (size, header) = match u32::from_be_bytes(head[..4].try_into().ok()?) {
            // The size is in the next 8 bytes.
            1 => (u64::from_be_bytes(data.get(at + 8..at + 16)?.try_into().ok()?) as usize, 16),
            // The box runs to the end.
            0 => (data.len() - at, 8),
            size => (size as usize, 8),
        };
        if size < header {
            return None;
        }
        let end = at.checked_add(size)?.min(data.len());
        let body = data.get(at + header..end)?;
        at = end;
        Some((name, body))
    })
}

fn mp4_child<'a>(data: &'a [u8], name: &[u8]) -> Option<&'a [u8]> {
    mp4_boxes(data).find(|(n, _)| *n == name).map(|(_, body)| body)
}

/// Seconds of the first sound track an MP4 file's edit list says to skip
/// at the start: what the encoder put before the sound. None without an
/// edit list.
fn mp4_edit_start(bytes: &[u8]) -> Option<f64> {
    let moov = mp4_child(bytes, b"moov")?;
    let sound = mp4_boxes(moov).filter(|(n, _)| *n == b"trak").map(|(_, t)| t).find(|trak| {
        mp4_child(trak, b"mdia")
            .and_then(|mdia| mp4_child(mdia, b"hdlr"))
            .and_then(|hdlr| hdlr.get(8..12))
            == Some(b"soun".as_slice())
    })?;
    let mdhd = mp4_child(mp4_child(sound, b"mdia")?, b"mdhd")?;
    // Version 1 has 64-bit times, so the timescale sits further in.
    let at = if mdhd.first() == Some(&1) { 20 } else { 12 };
    let timescale = u32::from_be_bytes(mdhd.get(at..at + 4)?.try_into().ok()?);
    let elst = mp4_child(mp4_child(sound, b"edts")?, b"elst")?;
    let wide = elst.first() == Some(&1);
    let count = u32::from_be_bytes(elst.get(4..8)?.try_into().ok()?) as usize;
    let entry = if wide { 20 } else { 12 };
    // An entry at -1 is an empty edit: a pause before the track, not a skip.
    let media_time = (0..count.min(16)).find_map(|i| {
        let at = 8 + i * entry;
        let time = if wide {
            i64::from_be_bytes(elst.get(at + 8..at + 16)?.try_into().ok()?)
        } else {
            i32::from_be_bytes(elst.get(at + 4..at + 8)?.try_into().ok()?) as i64
        };
        (time >= 0).then_some(time)
    })?;
    (timescale > 0).then(|| media_time as f64 / timescale as f64)
}

/// Down (or up) to the 16 kHz the models read, telling `progress` how far it got.
fn to_16k(samples: &[f32], rate: u32, mut progress: impl FnMut(f32)) -> Result<Vec<f32>, String> {
    if rate as usize == RATE {
        return Ok(samples.to_vec());
    }
    let mut resampler = crate::resampler::AudioResampler::new(rate)?;
    let wanted = samples.len() * RATE / rate as usize;
    let mut out = Vec::with_capacity(wanted + RATE);
    let step = rate as usize * 10;
    for (i, chunk) in samples.chunks(step).enumerate() {
        out.extend(resampler.process(chunk)?);
        progress(((i + 1) * step).min(samples.len()) as f32 / samples.len().max(1) as f32);
    }
    // The resampler hands its output back late by a fixed amount. That much is
    // cut from the front, and silence is pushed through to get the end out, so
    // the time marks match the file.
    // The resampler holds back up to one block of its input, and at some
    // rates a block is long, so silence is pushed until all of it is out.
    let delay = resampler.output_delay();
    let silence = vec![0.0; rate as usize / 10];
    for _ in 0..200 {
        if out.len() >= wanted + delay {
            break;
        }
        out.extend(resampler.process(&silence)?);
    }
    if out.len() < wanted + delay {
        return Err("the audio could not be converted to 16 kHz".into());
    }
    let mut out = out.split_off(delay);
    out.truncate(wanted);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Splitting at pauses.

/// The file cut into pieces, as sample ranges into `samples`. The app's own
/// speech detector finds the speech and the pauses of `pause_ms` or longer.
///
/// The pauses are not thrown away. The detector misses soft words - "ask
/// not", a quiet "yes" - and those sit in the pauses. So a pause up to
/// `FILL_GAP` seconds stays whole, split between the pieces on either side at
/// its quietest point. A longer pause keeps `EDGE` seconds on each side; its
/// middle becomes a piece of its own only if the detector, looking at it
/// alone, hears speech in it. Long silence, hiss and short noises (a door,
/// clicks) never reach the model, which writes invented sentences for them.
///
/// Pieces over `longest` seconds are cut again at quiet moments.
fn find_pieces(
    samples: &[f32],
    pause_ms: u32,
    longest: f32,
    progress: impl FnMut(f32),
) -> Result<Vec<Range<usize>>, String> {
    let speech = detect_speech(samples, pause_ms, progress)?;
    let longest = (longest * RATE as f32) as usize;
    // Inside a long pause the detector looks again with a fresh start, so a
    // cough there becomes a short piece, not the whole pause.
    let pieces = cover(samples, speech, |part| detect_speech(part, pause_ms, |_| {}).unwrap_or_default());
    Ok(join_short(pieces, longest)
        .into_iter()
        .flat_map(|p| split_long(samples, p, longest))
        .collect())
}

// The detector's settings for a file. It scores speech lower than it should
// when the rest of the sound is close to digital zero (the silences of a
// decoded MP3, everything above 4 kHz in 8 kHz audio), and lower again after
// a long stretch of earlier speech, so a sentence in the middle of a long file
// scored 0.3-0.45 where the same sentence at the start scored 0.7. A noise
// floor 80 dB under the peak and lower thresholds than dictation's 0.5 and 0.3
// bring those back. Hiss and silence still score under 0.1.
const DETECT_START: f32 = 0.4;
const DETECT_HOLD: f32 = 0.25;
const DETECT_NOISE_FLOOR: f32 = 1e-4;

/// The stretches of speech the app's detector hears, cut at pauses of
/// `pause_ms`, short ones kept.
fn detect_speech(
    samples: &[f32],
    pause_ms: u32,
    mut progress: impl FnMut(f32),
) -> Result<Vec<Range<usize>>, String> {
    let detector = crate::vad::Silero::new()?.with_thresholds(DETECT_START, DETECT_HOLD);
    let mut segmenter = Segmenter::new(detector, pause_ms)
        .keeping_short_sentences()
        .with_noise_floor(DETECT_NOISE_FLOOR);
    let mut speech = Vec::new();
    for (i, chunk) in samples.chunks(crate::vad::FRAME).enumerate() {
        let sentences = segmenter.feed(chunk);
        for (piece, &start) in sentences.iter().zip(segmenter.sentence_starts()) {
            speech.push(start..start + piece.len());
        }
        if i % 2000 == 0 {
            progress(i as f32 * crate::vad::FRAME as f32 / samples.len().max(1) as f32);
        }
    }
    let start = segmenter.pending_start();
    if let Some(piece) = segmenter.finish() {
        speech.push(start..start + piece.len());
    }
    Ok(speech)
}

// A pause up to this long is kept whole.
const FILL_GAP: f32 = 6.0;
// Of a longer pause, this much stays with the speech on each side.
const EDGE: f32 = 1.0;

/// The speech pieces widened over the pauses, as `find_pieces` describes.
/// `speech_in` finds speech in the middle of a long pause, as ranges into
/// the part it is given; each is kept with `EDGE` on both sides.
fn cover(
    samples: &[f32],
    speech: Vec<Range<usize>>,
    speech_in: impl Fn(&[f32]) -> Vec<Range<usize>>,
) -> Vec<Range<usize>> {
    if speech.is_empty() {
        return Vec::new();
    }
    let fill = (FILL_GAP * RATE as f32) as usize;
    let edge = (EDGE * RATE as f32) as usize;
    let len = samples.len();
    let mut out: Vec<Range<usize>> = Vec::new();
    let middle = |from: usize, to: usize, out: &mut Vec<Range<usize>>| {
        if to <= from {
            return;
        }
        for r in speech_in(&samples[from..to]) {
            let piece = (from + r.start).saturating_sub(edge).max(from)..(from + r.end + edge).min(to);
            match out.last_mut() {
                Some(last) if last.end >= piece.start && last.start >= from => last.end = last.end.max(piece.end),
                _ => out.push(piece),
            }
        }
    };

    // Where the piece being built starts.
    let first = speech[0].start;
    let mut open = if first <= fill {
        Some(0)
    } else {
        middle(0, first - edge, &mut out);
        Some(first - edge)
    };
    for pair in speech.windows(2) {
        let (end, next) = (pair[0].end.min(pair[1].start), pair[1].start);
        let start = open.take().unwrap_or(pair[0].start);
        if next - end <= fill {
            let cut = quietest(samples, end, next);
            out.push(start..cut);
            open = Some(cut);
        } else {
            out.push(start..end + edge);
            middle(end + edge, next - edge, &mut out);
            open = Some(next - edge);
        }
    }
    let last = speech[speech.len() - 1].end.min(len);
    let start = open.unwrap_or(0);
    if len - last <= fill {
        out.push(start..len);
    } else {
        out.push(start..last + edge);
        middle(last + edge, len, &mut out);
    }
    out.retain(|p| p.end > p.start);
    out
}

/// The middle of the quietest 0.3 s between `from` and `to`, in 30 ms steps.
/// A pause shorter than that is cut in its middle.
fn quietest(samples: &[f32], from: usize, to: usize) -> usize {
    let frame = crate::vad::FRAME;
    let span = frame * 10;
    let to = to.min(samples.len());
    if to < from + span {
        return (from + to) / 2;
    }
    // Average per frame, so a short last frame does not look quieter.
    let energy: Vec<f32> = samples[from..to]
        .chunks(frame)
        .map(|c| c.iter().map(|s| s * s).sum::<f32>() / c.len() as f32)
        .collect();
    let windows: Vec<f32> = energy.windows(10).map(|w| w.iter().sum::<f32>()).collect();
    let least = windows.iter().cloned().fold(f32::MAX, f32::min);
    // Equally quiet windows (digital silence is all zeros): the middle one,
    // so the silence is shared out and not all given to the next piece.
    let quiet: Vec<usize> = (0..windows.len()).filter(|&i| windows[i] <= least * 1.0001).collect();
    let best = quiet.get(quiet.len() / 2).copied().unwrap_or(0);
    from + best * frame + span / 2
}

// Pieces under this many seconds are joined to a piece next to them.
const SHORT_PIECE: f32 = 2.0;
// But only across a gap up to this long: a word on its own after a long
// silence stays on its own.
const JOIN_GAP: f32 = 3.0;

/// A short piece - "yes", "ask not" - is joined to the closer of the pieces
/// next to it, so the model hears it with its sentence and not alone. Nothing
/// is joined across more than `JOIN_GAP` or past `longest`.
fn join_short(pieces: Vec<Range<usize>>, longest: usize) -> Vec<Range<usize>> {
    let short = (SHORT_PIECE * RATE as f32) as usize;
    let gap_limit = (JOIN_GAP * RATE as f32) as usize;
    let gap = |a: &Range<usize>, b: &Range<usize>| b.start.saturating_sub(a.end);
    let fits = |a: &Range<usize>, b: &Range<usize>| gap(a, b) <= gap_limit && b.end - a.start <= longest;
    let mut v = pieces;
    let mut i = 0;
    while i < v.len() {
        if v[i].len() >= short {
            i += 1;
            continue;
        }
        let before = (i > 0 && fits(&v[i - 1], &v[i])).then(|| gap(&v[i - 1], &v[i]));
        let after = (i + 1 < v.len() && fits(&v[i], &v[i + 1])).then(|| gap(&v[i], &v[i + 1]));
        let into_before = match (before, after) {
            (Some(b), Some(a)) => b <= a,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => {
                i += 1;
                continue;
            }
        };
        // Joined pieces are checked again: two short ones can make one short one.
        let (a, b) = if into_before { (i - 1, i) } else { (i, i + 1) };
        v[a].end = v[b].end;
        v.remove(b);
        i = a;
    }
    v
}

/// A piece over `longest` samples, cut into nearly equal parts, each cut in
/// the middle of the quietest 100 ms near where it would fall, searched in
/// 10 ms steps. 100 ms because the silences inside a word - before the "k" in
/// "scarf" - are shorter than that and can be quieter than a breath between
/// sentences; a shorter window chose them and cut the word in two. No part is over `longest`, and
/// none is left tiny at the end.
fn split_long(samples: &[f32], piece: Range<usize>, longest: usize) -> Vec<Range<usize>> {
    let window = RATE / 10;
    let step = RATE / 100;
    let mut out = Vec::new();
    let mut start = piece.start;
    while piece.end - start > longest {
        let left = piece.end - start;
        let ideal = left / left.div_ceil(longest);
        let from = start + ideal / 2;
        let to = (start + (ideal * 5 / 4).min(longest)).saturating_sub(window).max(from + 1);
        let energy = |at: usize| samples[at..(at + window).min(samples.len())].iter().map(|s| s * s).sum::<f32>();
        let spots: Vec<(usize, f32)> = (from..to).step_by(step).map(|at| (at, energy(at))).collect();
        let least = spots.iter().map(|s| s.1).fold(f32::MAX, f32::min);
        // Equally quiet spots (digital silence): the middle one, not the first,
        // or every part comes out half as long as it could be.
        let quiet: Vec<usize> = spots.iter().filter(|s| s.1 <= least * 1.0001).map(|s| s.0).collect();
        let cut = quiet
            .get(quiet.len() / 2)
            .map_or(start + ideal, |at| at + window / 2)
            .min(start + longest);
        out.push(start..cut);
        start = cut;
    }
    out.push(start..piece.end);
    out
}

// ---------------------------------------------------------------------------
// Choosing the model and the language.

fn pick_model(models: &ModelManager, wanted: Option<&str>, prefs: &Prefs) -> Result<&'static ModelInfo, String> {
    let ids = || AVAILABLE_MODELS.iter().map(|m| m.id).collect::<Vec<_>>().join(", ");
    if let Some(id) = wanted {
        let model = AVAILABLE_MODELS
            .iter()
            .find(|m| m.id == id)
            .ok_or_else(|| format!("there is no model \"{id}\". The models are: {}", ids()))?;
        if !models.is_model_downloaded(id) {
            return Err(format!(
                "{id} is not downloaded. Download it in Settings, or see --list-models"
            ));
        }
        return Ok(model);
    }
    let active = prefs.active_local_model_id.as_deref();
    AVAILABLE_MODELS
        .iter()
        .find(|m| Some(m.id) == active && models.is_model_downloaded(m.id))
        .or_else(|| AVAILABLE_MODELS.iter().find(|m| models.is_model_downloaded(m.id)))
        .ok_or_else(|| "no model is downloaded. Download one in Settings first".to_string())
}

/// What the model is told: None lets it work out the language itself.
fn pick_language(wanted: Option<&str>, prefs: &Prefs) -> Result<Option<String>, String> {
    let Some(typed) = wanted else {
        return Ok(whisper_language(&prefs.language));
    };
    let code = typed.trim().to_lowercase();
    if code == "auto" {
        return Ok(None);
    }
    // The settings list only the common languages; Whisper knows these.
    let known = LANGUAGES.iter().any(|(c, _)| *c == code) || WHISPER_LANGUAGES.contains(&code.as_str());
    if known {
        Ok(Some(code))
    } else {
        let codes = LANGUAGES.iter().map(|(c, _)| *c).collect::<Vec<_>>().join(", ");
        Err(format!("\"{typed}\" is not a language code. Use one like: {codes}"))
    }
}

// Every language code Whisper accepts.
const WHISPER_LANGUAGES: [&str; 100] = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha",
    "ba", "jw", "su", "yue",
];

/// The pause shortening the settings ask for. The opening is not protected
/// here: the app keeps it so Whisper hears the first words of a dictation in
/// one run, but a piece of a file is not the start of anything, and a short
/// first word followed by seconds of silence is exactly what to shorten.
fn pause_rules(prefs: &Prefs) -> Option<crate::analysis::PauseRules> {
    prefs.pause_shortening.then_some(crate::analysis::PauseRules {
        cutoff_ms: prefs.pause_cutoff_ms,
        protect_opening_ms: None,
    })
}

fn gpu_choice(prefs: &Prefs, cpu: bool) -> GpuChoice {
    GpuChoice {
        onnx: prefs.onnx_gpu && !cpu,
        whisper: prefs.whisper_gpu && !cpu,
    }
}

fn runs_on(model: &ModelInfo, gpu: GpuChoice) -> &'static str {
    let on = match model.engine_type {
        EngineType::Whisper => gpu.whisper,
        _ => gpu.onnx,
    };
    if on {
        "GPU"
    } else {
        "processor"
    }
}

fn longest_piece(model: &ModelInfo) -> f32 {
    match model.engine_type {
        EngineType::Whisper => LONGEST_PIECE_WHISPER,
        _ => LONGEST_PIECE_ONNX,
    }
}

/// `-f` with only a file name: the current folder first, then the app's
/// recordings folder.
fn find_input(path: &Path) -> Result<PathBuf, String> {
    if path.is_dir() {
        return Err(format!("{} is a folder. Give -f an audio file", path.display()));
    }
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    if path.exists() {
        return Err(format!("{} is not a regular file. Give -f an audio file", path.display()));
    }
    let bare = path.components().count() == 1;
    if bare {
        if let Ok(dir) = crate::storage::get_recordings_dir() {
            let there = dir.join(path);
            if there.is_file() {
                return Ok(there);
            }
        }
    }
    Err(format!(
        "file not found: {}{}",
        path.display(),
        if bare { " (also looked in the recordings folder)" } else { "" }
    ))
}

// ---------------------------------------------------------------------------
// The transcription itself, shared by -f and the self-test.

struct Done {
    pieces: usize,
    with_text: usize,
    failed: usize,
    audio: f32,
    took: f32,
    text: String,
}

struct Pass<'a> {
    samples: &'a [f32],
    pieces: Vec<Range<usize>>,
    language: Option<String>,
    timestamps: bool,
    tidy: bool,
    // Cut the silent start and end off each piece, keeping 0.3 s. Moonshine
    // loses a short word with more than about 1.5 s of silence before it, and
    // Whisper repeats its style prompt as text before one.
    cut_silence: bool,
    // The app's own pause shortening, from the settings. Whisper fed seconds
    // of silence before a short word repeats its style prompt as text.
    shorten: Option<crate::analysis::PauseRules>,
    // Drop the stock phrases models write for a click or a cough ("Thank
    // you.", "Yeah.", "Mm.") from a short piece the detector hears almost no
    // speech in.
    drop_stock_phrases: bool,
}

/// Seconds from the start of `piece` to its first 30 ms that is at least a
/// tenth as loud as its loudest 30 ms. Against the piece's own loudest part,
/// so room noise does not count as the start, and digital silence (all zeros)
/// does not hide it.
fn sound_start(piece: &[f32]) -> f32 {
    let frame = crate::vad::FRAME;
    let level: Vec<f32> = piece
        .chunks(frame)
        .map(|c| (c.iter().map(|s| s * s).sum::<f32>() / c.len() as f32).sqrt())
        .collect();
    let loudest = level.iter().cloned().fold(0.0f32, f32::max);
    let threshold = (loudest * 0.1).max(0.001);
    level.iter().position(|&l| l >= threshold).unwrap_or(0) as f32 * frame as f32 / RATE as f32
}

/// `piece` from 0.3 s before its first sound to 0.3 s after its last, by the
/// same measure as `sound_start`.
fn without_silent_edges(piece: Vec<f32>) -> Vec<f32> {
    let frame = crate::vad::FRAME;
    let level: Vec<f32> = piece
        .chunks(frame)
        .map(|c| (c.iter().map(|s| s * s).sum::<f32>() / c.len() as f32).sqrt())
        .collect();
    let loudest = level.iter().cloned().fold(0.0f32, f32::max);
    let threshold = (loudest * 0.1).max(0.001);
    let (Some(first), Some(last)) = (
        level.iter().position(|&l| l >= threshold),
        level.iter().rposition(|&l| l >= threshold),
    ) else {
        return piece;
    };
    let margin = RATE * 3 / 10;
    let start = (first * frame).saturating_sub(margin);
    let end = ((last + 1) * frame + margin).min(piece.len());
    piece[start..end].to_vec()
}

// What Whisper writes for a sound that is not speech. It learned these from
// the ends of videos.
// Sounds, not words: dropped from a piece with under NO_WORD_SECONDS of speech.
const FILLERS: [&str; 5] = ["mm", "hmm", "mhm", "uh", "um"];

// Real words that are also what models write for a click. "You" and "Oh"
// are left out: they are common one-word answers, and a real "Oh." is so
// short the detector hears almost nothing in it. Dropped only with under
// NO_SOUND_SECONDS of speech.
const STOCK_PHRASES: [&str; 10] = [
    "yeah",
    "thank you",
    "thanks",
    "thank you for watching",
    "thanks for watching",
    "thank you very much",
    "bye",
    "subscribe",
    "благодаря",
    "абонирайте се",
];

// Under this much speech a piece holds no word. A short real word ("Bye.")
// can score under the 0.3 s used elsewhere; a click scores near nothing.
const NO_WORD_SECONDS: f32 = 0.15;
// Two 30 ms frames: under this the detector heard nothing at all.
const NO_SOUND_SECONDS: f32 = 0.06;

/// Whether Whisper's `text` for `piece` is one of its stock phrases with
/// next to no speech under it. A real "thank you" has speech under it and
/// stays.
fn invented(text: &str, piece: &[f32]) -> bool {
    let plain = words(text).join(" ");
    let limit = if FILLERS.contains(&plain.as_str()) {
        NO_WORD_SECONDS
    } else if STOCK_PHRASES.contains(&plain.as_str()) {
        NO_SOUND_SECONDS
    } else {
        return false;
    };
    piece.len() < RATE * 4 && heard_seconds(piece).unwrap_or(1.0) < limit
}

/// Seconds of speech the detector hears in `piece`, judged the way
/// `detect_speech` judges: the same thresholds, the same gain limit, and the
/// same noise floor added after the scaling, so 8 kHz audio is judged like
/// any other and a silent piece stays silent.
fn heard_seconds(piece: &[f32]) -> Result<f32, String> {
    use crate::live::SpeechDetector;
    let mut detector = crate::vad::Silero::new()?.with_thresholds(DETECT_START, DETECT_HOLD);
    // Scaled like `Segmenter` scales: to a fixed peak, but by no more than
    // its gain limit, so a soft click is not raised to full volume.
    let peak = piece.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    let gain = crate::vad::NORMALIZED_PEAK / peak.max(crate::live::PEAK_FLOOR);
    let mut copy: Vec<f32> = piece.iter().map(|s| s * gain).collect();
    for (s, n) in copy.iter_mut().zip(noise(piece.len(), DETECT_NOISE_FLOOR)) {
        *s += n;
    }
    let frames = copy.as_chunks::<{ crate::vad::FRAME }>().0.iter().filter(|f| detector.is_speech(&f[..])).count();
    Ok(crate::vad::seconds_of(frames))
}

/// Every piece through the model, in order. A piece the model fails on is
/// reported and marked in the text, and the rest still go through.
fn transcribe_pieces(
    engine: &mut TranscriptionManager,
    pass: Pass,
    ui: &Ui,
    sink: &mut dyn FnMut(&str) -> Result<(), String>,
) -> Result<Done, String> {
    let total = pass.samples.len() as f32 / RATE as f32;
    let long = total >= 3600.0;
    let seconds = |r: &Range<usize>| (r.start as f32 / RATE as f32, r.end as f32 / RATE as f32);
    let started = Instant::now();
    ui.with(|s| {
        s.view = View::Transcribe(Box::new(Progress {
            overview: loudness_columns(pass.samples, 1000),
            total_seconds: total,
            pieces: pass.pieces.iter().map(seconds).collect(),
            done: 0,
            current: None,
            audio_done: 0.0,
            compute_done: 0.0,
            started,
        }))
    });
    let live = ui.live();
    let mut text = String::new();
    let mut audio = 0.0;
    let mut failed = 0;
    let mut with_text = 0;
    for range in &pass.pieces {
        // Quiet speech raised to a normal level, as the app does. The edges
        // are cut further down with `without_silent_edges`, which measures
        // against the piece's loudest part and keeps 0.3 s, so a soft word
        // there stays; the app's own `trim_quiet_edges` is stricter.
        let mut piece = pass.samples[range.clone()].to_vec();
        crate::analysis::boost_quiet_audio(&mut piece);
        // The piece starts in a pause; the time mark goes where the sound does.
        let at = stamp(seconds(range).0 + sound_start(&piece), long);
        if pass.cut_silence {
            piece = without_silent_edges(piece);
        }
        if let Some(rules) = pass.shorten {
            crate::analysis::shorten_long_pauses(&mut piece, rules);
        }
        let levels = loudness_columns(&piece, 200);
        ui.progress(|p| p.current = Some((levels, Instant::now())));
        let began = Instant::now();
        let heard = if piece.len() < RATE * 3 / 10 {
            Ok(String::new())
        } else if piece.len() < RATE * 3 {
            // Too short for the style prompt to help, long enough for Whisper
            // to repeat it instead.
            engine.transcribe_without_prompt(&piece, pass.language.clone())
        } else {
            engine.transcribe(&piece, pass.language.clone())
        };
        let took = began.elapsed().as_secs_f32();
        let length = range.len() as f32 / RATE as f32;
        // What really went to the model, silent edges and long pauses left out.
        audio += piece.len() as f32 / RATE as f32;
        ui.progress(|p| {
            p.current = None;
            p.done += 1;
            p.audio_done += length;
            p.compute_done += took;
        });

        let line = match heard {
            Ok(heard) => {
                let heard = if pass.tidy {
                    crate::live::tidy_end(heard.trim())
                } else {
                    heard.trim().to_string()
                };
                if heard.is_empty() || (pass.drop_stock_phrases && invented(&heard, &piece)) {
                    continue;
                }
                with_text += 1;
                if pass.timestamps {
                    format!("{at} {heard}")
                } else {
                    heard
                }
            }
            Err(e) => {
                failed += 1;
                ui.error(&format!("the piece at {at} was not transcribed: {e}"));
                if pass.timestamps {
                    format!("{at} (this piece could not be transcribed)")
                } else {
                    "(this piece could not be transcribed)".to_string()
                }
            }
        };
        if live {
            let mark = ui.paint(&line, BOLD);
            ui.with(|s| s.line(&format!("{mark}\n")));
        }
        sink(&line)?;
        text.push_str(&line);
        text.push_str("\n\n");
    }
    ui.blank();
    Ok(Done {
        pieces: pass.pieces.len(),
        with_text,
        failed,
        audio,
        took: started.elapsed().as_secs_f32(),
        text,
    })
}

// Read and convert the file, with progress on screen. The file's bytes are
// let go as soon as they are read, so a long recording is not held twice.
fn prepare_audio(bytes: Vec<u8>, name: &Path, ui: &Ui) -> Result<Vec<f32>, String> {
    ui.status("Reading the audio", Some(0.0));
    let Decoded { samples: raw, rate, warning } = read_audio(&bytes, &name.to_string_lossy())?;
    drop(bytes);
    if let Some(warning) = warning {
        ui.warn(&warning);
    }
    if raw.is_empty() {
        return Err("the file holds no audio".into());
    }
    to_16k(&raw, rate, |f| ui.status("Reading the audio", Some(f)))
}

fn split_audio(samples: &[f32], pause_ms: u32, longest: f32, ui: &Ui) -> Vec<Range<usize>> {
    ui.status("Finding the speech", Some(0.0));
    match find_pieces(samples, pause_ms, longest, |f| ui.status("Finding the speech", Some(f))) {
        Ok(pieces) => pieces,
        Err(e) => {
            // No detector: the whole file in even pieces, silence and all.
            ui.say(&format!("The speech detector did not load ({e}). Splitting into fixed pieces."));
            split_long(samples, 0..samples.len(), (longest * RATE as f32) as usize)
        }
    }
}

fn load_engine(model: &ModelInfo, gpu: GpuChoice, ui: &Ui) -> Result<TranscriptionManager, String> {
    let models = Arc::new(ModelManager::new()?);
    let mut engine = TranscriptionManager::new(models);
    ui.status(&format!("Loading {} on the {}", model.id, runs_on(model, gpu)), None);
    engine.load_model(model.id, gpu)?;
    Ok(engine)
}

// A path made absolute without needing the file to exist yet.
fn absolute(path: &Path) -> Option<PathBuf> {
    if let Ok(full) = path.canonicalize() {
        return Some(full);
    }
    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.canonicalize().ok()?,
        _ => std::env::current_dir().ok()?,
    };
    Some(parent.join(path.file_name()?))
}

// The text is written to a hidden `.<name>.<process id>.part` file next to
// the output while the work runs, and takes the output's name when the run
// reaches the end - with all its text, or none if no speech was found. A run
// that stops early - an error, a panic, Ctrl+C, a closed terminal, kill, or an
// abort() in a model library - leaves an old output as it was and removes its
// own .part file, and only that one. Only kill -9 leaves the .part behind.
struct Output {
    path: PathBuf,
    part: PathBuf,
    file: Option<File>,
    finished: bool,
    force: bool,
}

impl Output {
    fn write(&mut self, text: &str) -> Result<(), String> {
        let file = self.file.as_mut().ok_or("the output file is closed")?;
        file.write_all(text.as_bytes())
            .and_then(|_| file.flush())
            .map_err(|e| format!("cannot write the output file: {e}"))
    }

    fn finish(&mut self) -> Result<(), String> {
        drop(self.file.take());
        if self.force {
            fs::rename(&self.part, &self.path)
        } else {
            // Without --force the output must still not exist: another run may
            // have made it since this one started. A hard link fails if it
            // does, in one step, so two runs cannot both win.
            match fs::hard_link(&self.part, &self.path) {
                // The output is in place; a .part that will not go away is
                // only clutter, not a failed run.
                Ok(()) => {
                    let _ = fs::remove_file(&self.part);
                    Ok(())
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Err(e),
                // Drives without hard links (FAT, some network shares): check,
                // then rename. Not in one step, but the best they allow.
                Err(_) if self.path.exists() => Err(std::io::ErrorKind::AlreadyExists.into()),
                Err(_) => fs::rename(&self.part, &self.path),
            }
        }
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::AlreadyExists => format!(
                "{} was made by something else while this ran. Add --force to overwrite it",
                self.path.display()
            ),
            _ => format!("cannot write {}: {e}", self.path.display()),
        })?;
        self.finished = true;
        forget_part_file();
        Ok(())
    }
}

impl Drop for Output {
    fn drop(&mut self) {
        drop(self.file.take());
        if !self.finished {
            let _ = fs::remove_file(&self.part);
            forget_part_file();
        }
    }
}

fn open_output(path: &Path, input: &Path, force: bool) -> Result<Output, String> {
    if path.is_dir() {
        return Err(format!("{} is a folder. Give a file name for -o", path.display()));
    }
    if absolute(path).is_some() && absolute(path) == absolute(input) {
        return Err("the output file is the audio file. Give -o a different name".into());
    }
    if path.exists() && !force {
        return Err(format!("{} already exists. Add --force to overwrite it", path.display()));
    }
    let name = path.file_name().ok_or("-o needs a file name")?.to_string_lossy();
    let part = path.with_file_name(format!(".{name}.{}.part", std::process::id()));
    if absolute(&part) == absolute(input) {
        return Err("the output's temporary name is the audio file. Give -o a different name".into());
    }
    // create_new: never open a file that is already there. Only once it is
    // ours does the Ctrl+C handler learn its name, so it can never remove a
    // file this run did not make.
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&part)
        .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    remember_part_file(&part);
    Ok(Output {
        path: path.to_path_buf(),
        part,
        file: Some(file),
        finished: false,
        force,
    })
}

// ---------------------------------------------------------------------------
// Ctrl+C. The program is stopped at once, as without this, but first its
// .part file is removed and, on Windows, the console's code page put back.

#[cfg(unix)]
static PART_FILE: std::sync::atomic::AtomicPtr<libc::c_char> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

#[cfg(unix)]
fn remember_part_file(path: &Path) {
    use std::os::unix::ffi::OsStrExt;
    if let Ok(c) = std::ffi::CString::new(path.as_os_str().as_bytes()) {
        // Leaked on purpose: the signal handler may read it at any moment.
        PART_FILE.store(c.into_raw(), Ordering::SeqCst);
    }
}

#[cfg(unix)]
fn forget_part_file() {
    PART_FILE.store(std::ptr::null_mut(), Ordering::SeqCst);
}

#[cfg(unix)]
extern "C" fn on_interrupt(signal: libc::c_int) {
    // Only calls that are safe inside a signal handler: unlink, write, _exit.
    let part = PART_FILE.load(Ordering::SeqCst);
    unsafe {
        if !part.is_null() {
            libc::unlink(part);
        }
        let note: &[u8] = if DRAWING.load(Ordering::SeqCst) {
            b"\x1b[0m\nStopped.\n"
        } else {
            b"Stopped.\n"
        };
        libc::write(ERR_FD.load(Ordering::SeqCst), note.as_ptr() as *const _, note.len());
        libc::_exit(128 + signal);
    }
}

#[cfg(unix)]
static ERR_FD: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(2);

// Whether colour and cursor codes are going to stderr, for the Ctrl+C note.
static DRAWING: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
fn handle_interrupt(err: Option<&File>) {
    use std::os::unix::io::AsRawFd;
    if let Some(f) = err {
        ERR_FD.store(f.as_raw_fd(), Ordering::SeqCst);
    }
    // Ctrl+C, kill, a closed terminal, Ctrl+\, and an abort() inside
    // whisper.cpp or ONNX Runtime. A signal the parent set to be ignored
    // (nohup, a script's background job) stays ignored.
    unsafe {
        let handler = on_interrupt as extern "C" fn(libc::c_int) as libc::sighandler_t;
        for signal in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP, libc::SIGQUIT, libc::SIGABRT] {
            if libc::signal(signal, handler) == libc::SIG_IGN {
                libc::signal(signal, libc::SIG_IGN);
            }
        }
    }
}

#[cfg(windows)]
static PART_FILE: Mutex<Option<PathBuf>> = Mutex::new(None);

#[cfg(windows)]
fn remember_part_file(path: &Path) {
    *PART_FILE.lock().unwrap_or_else(|e| e.into_inner()) = Some(path.to_path_buf());
}

#[cfg(windows)]
fn forget_part_file() {
    *PART_FILE.lock().unwrap_or_else(|e| e.into_inner()) = None;
}

#[cfg(windows)]
unsafe extern "system" fn on_interrupt(_kind: u32) -> windows_sys::core::BOOL {
    // Windows runs this on its own thread, so ordinary calls are fine.
    if let Some(part) = PART_FILE.lock().unwrap_or_else(|e| e.into_inner()).take() {
        let _ = fs::remove_file(part);
    }
    restore_console();
    std::process::exit(130);
}

#[cfg(windows)]
fn handle_interrupt(_err: Option<&File>) {
    unsafe {
        windows_sys::Win32::System::Console::SetConsoleCtrlHandler(Some(on_interrupt), 1);
    }
}

#[cfg(not(any(unix, windows)))]
fn remember_part_file(_: &Path) {}
#[cfg(not(any(unix, windows)))]
fn forget_part_file() {}
#[cfg(not(any(unix, windows)))]
fn handle_interrupt(_: Option<&File>) {}

// ---------------------------------------------------------------------------
// The commands.

pub fn transcribe_file(job: Job) -> i32 {
    let streams = take_streams();
    let live = job.visual && !job.quiet && streams.err_terminal;
    handle_interrupt(streams.err_copy.as_ref());
    let ui = Ui::start(streams.err, streams.err_copy, live, job.quiet);
    let mut output = None;
    let result = run_job(&job, &ui, streams.out, streams.out_terminal, &mut output);
    let code = match result {
        Ok(0) => 0,
        Ok(failed) => {
            let what = if failed == 1 { "1 piece" } else { "pieces" };
            let count = if failed == 1 { String::new() } else { format!("{failed} ") };
            ui.error(&format!("{count}{what} could not be transcribed; marked in the text"));
            1
        }
        Err(e) => {
            ui.blank();
            ui.error(&e);
            1
        }
    };
    // Not finished: dropping it removes its .part file and keeps an old output.
    drop(output);
    ui.finish();
    code
}

/// The -f command. Returns how many pieces failed.
fn run_job(
    job: &Job,
    ui: &Ui,
    mut out: Box<dyn Write + Send>,
    out_terminal: bool,
    output: &mut Option<Output>,
) -> Result<usize, String> {
    let prefs = load_prefs();
    let models = ModelManager::new()?;
    let model = pick_model(&models, job.model.as_deref(), &prefs)?;
    let language = pick_language(job.language.as_deref(), &prefs)?;
    let gpu = gpu_choice(&prefs, job.cpu);
    let input = find_input(&job.file)?;

    // Made before the long work starts, so a wrong path is found out at once.
    if let Some(path) = &job.output {
        *output = Some(open_output(path, &input, job.force)?);
    }

    let bytes = fs::read(&input).map_err(|e| format!("cannot read {}: {e}", input.display()))?;
    let samples = prepare_audio(bytes, &input, ui)?;
    let total = samples.len() as f32 / RATE as f32;

    let language_note = match (&language, model.engine_type) {
        (_, EngineType::Whisper) => language.clone().unwrap_or_else(|| "as spoken".into()),
        (Some(_), _) => "chosen by the model (only Whisper takes a language)".into(),
        (None, _) => "chosen by the model".into(),
    };
    ui.say(&format!(
        "{}  {} on the {}   Language {}   Audio {}   {}",
        ui.paint("Omegawhisper", BOLD),
        model.id,
        runs_on(model, gpu),
        language_note,
        clock(total),
        input.file_name().unwrap_or_default().to_string_lossy()
    ));

    let pieces = split_audio(&samples, job.pause_ms, longest_piece(model), ui);
    let done = if pieces.is_empty() {
        ui.blank();
        ui.say("No speech was found in the file.");
        None
    } else {
        let mut engine = load_engine(model, gpu, ui)?;

        // Where the text goes. With a terminal on both, the drawing already
        // shows it, and writing it to stdout as well would print it twice.
        let to_stdout = output.is_none() && !(out_terminal && ui.live());
        // A blank line between pieces, none after the last.
        let mut first = true;
        let mut sink = |line: &str| -> Result<(), String> {
            let entry = if first { format!("{line}\n") } else { format!("\n{line}\n") };
            first = false;
            if let Some(o) = output.as_mut() {
                o.write(&entry)?;
            } else if to_stdout {
                // The reader went away (`| head`): no point going on.
                write!(out, "{entry}")
                    .and_then(|_| out.flush())
                    .map_err(|e| format!("the output was closed ({e})"))?;
            }
            Ok(())
        };
        let pass = Pass {
            samples: &samples,
            pieces,
            language: language.filter(|_| model.engine_type == EngineType::Whisper),
            timestamps: job.timestamps,
            tidy: prefs.tidy_sentence_ends,
            cut_silence: true,
            shorten: pause_rules(&prefs),
            drop_stock_phrases: true,
        };
        let done = transcribe_pieces(&mut engine, pass, ui, &mut sink)?;
        engine.unload_model();
        Some(done)
    };

    // The run reached the end: the .part file takes the output's name.
    if let Some(o) = output.as_mut() {
        o.finish()?;
    }
    let Some(done) = done else {
        return Ok(0);
    };
    let target = job
        .output
        .as_ref()
        .map(|p| format!(" -> {}", p.display()))
        .unwrap_or_default();
    ui.say(&format!(
        "Done: {} {}{}, {} of speech in {} ({:.1}x real time){target}",
        done.pieces,
        if done.pieces == 1 { "piece" } else { "pieces" },
        if done.with_text == done.pieces { String::new() } else { format!(" ({} with text)", done.with_text) },
        clock(done.audio),
        clock(done.took),
        done.audio / done.took.max(0.001)
    ));
    Ok(done.failed)
}

pub fn list_models() -> i32 {
    let models = match ModelManager::new() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error: {e}");
            return 1;
        }
    };
    let prefs = load_prefs();
    let active = prefs.active_local_model_id.as_deref();
    println!("{:<20} {:<28} {:>8}  Downloaded", "ID", "Name", "Size");
    for m in AVAILABLE_MODELS.iter() {
        let downloaded = if models.is_model_downloaded(m.id) { "yes" } else { "no" };
        println!(
            "{:<20} {:<28} {:>8}  {}{}",
            m.id,
            cut(m.name, 28),
            size(m.total_size_bytes),
            downloaded,
            if Some(m.id) == active { "  (in Settings)" } else { "" }
        );
    }
    0
}

fn size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1e9)
    } else {
        format!("{:.0} MB", bytes as f64 / 1e6)
    }
}

// ---------------------------------------------------------------------------
// The self-test: every part the -f command uses, checked on known audio.

// John F. Kennedy, 11 seconds, the usual first test of a speech model.
const JFK: &[u8] = include_bytes!("../tests-data/jfk.wav");
// The same speech encoded as MP3 and as AAC in an M4A.
const JFK_MP3: &[u8] = include_bytes!("../tests-data/jfk.mp3");
const JFK_M4A: &[u8] = include_bytes!("../tests-data/jfk.m4a");
const JFK_TEXT: &str = "And so my fellow Americans, ask not what your country can do for you, \
                        ask what you can do for your country.";

fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| w.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Share of the expected words heard, in order (longest common subsequence).
fn word_match(heard: &str, expected: &str) -> f32 {
    let (a, b) = (words(heard), words(expected));
    if b.is_empty() {
        return 1.0;
    }
    let mut row = vec![0usize; b.len() + 1];
    for x in &a {
        let mut diagonal = 0;
        for (j, y) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = if x == y { diagonal + 1 } else { row[j + 1].max(row[j]) };
            diagonal = above;
        }
    }
    row[b.len()] as f32 / b.len() as f32
}

fn sine(seconds: f32, hz: f32, rate: u32) -> Vec<f32> {
    (0..(seconds * rate as f32) as usize)
        .map(|i| 0.5 * (2.0 * std::f32::consts::PI * hz * i as f32 / rate as f32).sin())
        .collect()
}

fn noise(samples: usize, amplitude: f32) -> Vec<f32> {
    let mut x: u32 = 12345;
    (0..samples)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (x as f32 / u32::MAX as f32 * 2.0 - 1.0) * amplitude
        })
        .collect()
}

fn jfk_16k() -> Result<Vec<f32>, String> {
    let (samples, rate) = read_wav(JFK)?;
    to_16k(&samples, rate, |_| {})
}

type Check = Box<dyn FnOnce(&Ui) -> Result<String, String>>;

pub fn self_test(model: Option<String>, cpu: bool, visual: bool) -> i32 {
    let streams = take_streams();
    let live = visual && streams.err_terminal;
    handle_interrupt(streams.err_copy.as_ref());
    let ui = Ui::start(streams.err, streams.err_copy, live, false);
    let prefs = load_prefs();
    let gpu = gpu_choice(&prefs, cpu);

    let mut checks: Vec<(String, Check)> = Vec::new();

    checks.push((
        "Settings file".into(),
        Box::new(|_| {
            let path = crate::settings::prefs_path().ok_or("no settings folder on this system")?;
            let prefs = load_prefs();
            Ok(format!(
                "{} {} · model {} · language {}",
                path.display(),
                if path.exists() { "" } else { "(not made yet, defaults used)" },
                prefs.active_local_model_id.as_deref().unwrap_or("none"),
                prefs.language
            ))
        }),
    ));

    checks.push((
        "WAV reader and resampler".into(),
        Box::new(|_| {
            // 220 Hz at 48 kHz, stereo, written and read back as 16 kHz mono.
            let tone = sine(2.0, 220.0, 48_000);
            let stereo: Vec<f32> = tone.iter().flat_map(|s| [*s, *s]).collect();
            let wav = crate::analysis::to_wav_bytes(&stereo, 48_000, 2);
            let (mono, rate) = read_wav(&wav)?;
            if rate != 48_000 || mono.len() != tone.len() {
                return Err(format!("read {} samples at {rate} Hz, wanted {} at 48000", mono.len(), tone.len()));
            }
            let down = to_16k(&mono, rate, |_| {})?;
            // Whole cycles: each time the wave climbs from clearly below zero to
            // clearly above it. A plain sign change also counts the tiny
            // ripples near zero and reads too high.
            let mut cycles = 0;
            let mut low = false;
            for &x in &down {
                if x < -0.1 {
                    low = true;
                } else if x > 0.1 && low {
                    low = false;
                    cycles += 1;
                }
            }
            let hz = cycles as f32 / (down.len() as f32 / RATE as f32);
            if (down.len() as i64 - 32_000).abs() > 320 || (hz - 220.0).abs() > 3.0 {
                return Err(format!("got {} samples at {hz:.1} Hz, wanted 32000 at 220 Hz", down.len()));
            }
            Ok(format!("48 kHz stereo -> {} samples at 16 kHz, tone {hz:.1} Hz", down.len()))
        }),
    ));

    checks.push((
        "MP3 and M4A reader".into(),
        Box::new(|_| {
            let wav = jfk_16k()?;
            // Loudness in 10 ms steps, to compare the decoded sound with the
            // WAV without caring about the small shift an encoder adds.
            let step = RATE / 100;
            let loudness = |x: &[f32]| -> Vec<f32> {
                x.chunks(step).map(|c| (c.iter().map(|s| s * s).sum::<f32>() / c.len() as f32).sqrt()).collect()
            };
            let wanted = loudness(&wav);
            let mut report = Vec::new();
            for (name, bytes) in [("MP3", JFK_MP3), ("M4A", JFK_M4A)] {
                let sound = read_audio(bytes, &format!("jfk.{}", name.to_lowercase()))?;
                let down = to_16k(&sound.samples, sound.rate, |_| {})?;
                let off = (down.len() as f32 - wav.len() as f32) / RATE as f32;
                if off.abs() > 0.02 {
                    return Err(format!("{name}: {:.2} s long, the WAV {:.2} s", down.len() as f32 / RATE as f32, wav.len() as f32 / RATE as f32));
                }
                // The best match of the two loudness lines, moved up to 0.3 s
                // against each other (Pearson correlation, 1.0 is the same shape).
                let got = loudness(&down);
                let correlation = |shift: usize| -> f32 {
                    let pairs: Vec<(f32, f32)> = wanted.iter().zip(got.iter().skip(shift)).map(|(a, b)| (*a, *b)).collect();
                    let n = pairs.len() as f32;
                    let (ma, mb) = (pairs.iter().map(|p| p.0).sum::<f32>() / n, pairs.iter().map(|p| p.1).sum::<f32>() / n);
                    let cov: f32 = pairs.iter().map(|(a, b)| (a - ma) * (b - mb)).sum();
                    let va: f32 = pairs.iter().map(|(a, _)| (a - ma).powi(2)).sum();
                    let vb: f32 = pairs.iter().map(|(_, b)| (b - mb).powi(2)).sum();
                    cov / (va * vb).sqrt().max(1e-12)
                };
                let (shift, best) = (0..=30).map(|s| (s, correlation(s))).fold((0, f32::MIN), |a, b| if b.1 > a.1 { b } else { a });
                if best < 0.95 {
                    return Err(format!("{name}: loudness matches the WAV only {:.0}%", best * 100.0));
                }
                // The encoder's silence in front is cut, so the sound starts
                // where the WAV's does.
                if shift > 1 {
                    return Err(format!("{name}: starts {} ms later than the WAV", shift * 10));
                }
                report.push(format!("{name} {:.1}% alike, {} ms late", best * 100.0, shift * 10));
            }
            Ok(report.join(" · "))
        }),
    ));

    checks.push((
        "Speech detector".into(),
        Box::new(|_| {
            let jfk = jfk_16k()?;
            let speech = crate::vad::speech_seconds(&jfk)?;
            let quiet = crate::vad::speech_seconds(&vec![0.0; RATE * 3])?;
            let hiss = crate::vad::speech_seconds(&noise(RATE * 3, 0.2))?;
            if speech < 5.0 || quiet > 0.0 || hiss >= crate::vad::MIN_SPEECH_SECONDS {
                return Err(format!("speech {speech:.1} s, silence {quiet:.1} s, noise {hiss:.1} s"));
            }
            Ok(format!("speech {speech:.1} s · silence {quiet:.1} s · noise {hiss:.1} s"))
        }),
    ));

    checks.push((
        "Splitting at pauses".into(),
        Box::new(|_| {
            let jfk = jfk_16k()?;
            let mut audio = jfk.clone();
            let gap = audio.len()..audio.len() + RATE * 3;
            audio.extend(vec![0.0; RATE * 3]);
            audio.extend(&jfk);
            let pieces = find_pieces(&audio, 1200, LONGEST_PIECE_WHISPER, |_| {})?;
            // A long quiet hiss between the two: none of it in the middle may
            // reach the model, or it writes sentences nobody said.
            let mut hissy = jfk.clone();
            let hiss = hissy.len()..hissy.len() + RATE * 60;
            hissy.extend(noise(RATE * 60, 0.003));
            hissy.extend(&jfk);
            let around = find_pieces(&hissy, 1200, LONGEST_PIECE_WHISPER, |_| {})?;
            let kept: usize = around
                .iter()
                .map(|p| p.end.min(hiss.end).saturating_sub(p.start.max(hiss.start)))
                .sum();
            if kept > RATE * 3 {
                return Err(format!("{:.0} s of a 60 s silence would go to the model", kept as f32 / RATE as f32));
            }
            // Nothing left out: the pieces run end to end over the whole file.
            let whole = pieces.first().map(|p| p.start) == Some(0)
                && pieces.last().map(|p| p.end) == Some(audio.len())
                && pieces.windows(2).all(|w| w[0].end == w[1].start);
            if !whole {
                return Err("the pieces leave part of the audio out".into());
            }
            // And one cut falls in the silence, not in a word.
            if !pieces.iter().any(|p| p.end > gap.start && p.end < gap.end) {
                return Err(format!("{} pieces, none cut in the 3 s silence", pieces.len()));
            }
            let short = split_long(&audio, 0..audio.len(), RATE * 10);
            if short.iter().any(|p| p.len() > RATE * 10) || short.last().map(|p| p.end) != Some(audio.len()) {
                return Err("cutting a long piece went wrong".into());
            }
            Ok(format!(
                "3 s pause kept whole ({} pieces) · 60 s silence: {:.0} s kept",
                pieces.len(),
                kept as f32 / RATE as f32
            ))
        }),
    ));

    let manager = ModelManager::new();
    let to_test: Vec<&'static ModelInfo> = match (&manager, &model) {
        (Ok(m), Some(id)) => AVAILABLE_MODELS.iter().filter(|x| x.id == id && m.is_model_downloaded(x.id)).collect(),
        (Ok(m), None) => AVAILABLE_MODELS.iter().filter(|x| m.is_model_downloaded(x.id)).collect(),
        _ => Vec::new(),
    };
    let wanted_missing = model.clone().filter(|_| to_test.is_empty());
    checks.push((
        "Models folder".into(),
        Box::new(move |_| {
            let m = ModelManager::new()?;
            if let Some(id) = wanted_missing {
                return Err(format!("{id} is unknown or not downloaded"));
            }
            let found: Vec<&str> = AVAILABLE_MODELS.iter().filter(|x| m.is_model_downloaded(x.id)).map(|x| x.id).collect();
            if found.is_empty() {
                return Err("no model is downloaded".into());
            }
            Ok(found.join(", "))
        }),
    ));

    for info in &to_test {
        let info: &'static ModelInfo = info;
        checks.push((
            format!("Model {}", info.id),
            Box::new(move |ui| {
                let jfk = jfk_16k()?;
                let began = Instant::now();
                let mut engine = load_engine(info, gpu, ui)?;
                let load = began.elapsed().as_secs_f32();
                ui.status(&format!("Transcribing with {}", info.id), None);
                let began = Instant::now();
                let language = (info.engine_type == EngineType::Whisper).then(|| "en".to_string());
                let heard = engine.transcribe(&jfk, language)?;
                let run = began.elapsed().as_secs_f32();
                engine.unload_model();
                let score = word_match(&heard, JFK_TEXT);
                let report = format!(
                    "on the {} · load {load:.1} s · 11 s of speech in {run:.2} s · {:.0}% of the words",
                    runs_on(info, gpu),
                    score * 100.0
                );
                if score < 0.8 {
                    return Err(format!("{report}. Heard: \"{}\"", heard.trim()));
                }
                Ok(report)
            }),
        ));
    }

    // The whole -f command on a real file: written, read back, split,
    // transcribed with the drawing on screen, and saved.
    let whole_model = to_test
        .iter()
        .find(|m| Some(m.id) == prefs.active_local_model_id.as_deref())
        .or(to_test.first())
        .copied();
    checks.push((
        "Whole command, file to text".into(),
        Box::new(move |ui| {
            let model = whole_model.ok_or("no model to run it with")?;
            let jfk = jfk_16k()?;
            let mut audio = jfk.clone();
            audio.extend(vec![0.0; RATE * 2]);
            audio.extend(&jfk);
            let dir = std::env::temp_dir().join(format!("omegawhisper-self-test-{}", std::process::id()));
            fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            let wav = dir.join("speech.wav");
            fs::write(&wav, crate::analysis::to_wav_bytes(&audio, RATE as u32, 1)).map_err(|e| e.to_string())?;

            let result = (|| {
                let bytes = fs::read(&wav).map_err(|e| e.to_string())?;
                let samples = prepare_audio(bytes, &wav, ui)?;
                let pieces = split_audio(&samples, 1200, longest_piece(model), ui);
                let mut engine = load_engine(model, gpu, ui)?;
                let pass = Pass {
                    samples: &samples,
                    pieces,
                    language: (model.engine_type == EngineType::Whisper).then(|| "en".to_string()),
                    timestamps: true,
                    tidy: true,
                    cut_silence: true,
                    shorten: pause_rules(&load_prefs()),
                    drop_stock_phrases: true,
                };
                let done = transcribe_pieces(&mut engine, pass, ui, &mut |_| Ok(()))?;
                engine.unload_model();
                Ok::<_, String>(done)
            })();
            let _ = fs::remove_dir_all(&dir);
            let done = result?;
            let text: String = done.text.lines().filter(|l| !l.is_empty()).map(|l| l.split_once("] ").map_or(l, |x| x.1)).collect::<Vec<_>>().join(" ");
            let twice = format!("{JFK_TEXT} {JFK_TEXT}");
            let score = word_match(&text, &twice);
            let stamps = done.text.lines().filter(|l| l.starts_with('[')).count();
            if score < 0.8 || stamps < 2 {
                return Err(format!("{} pieces, {stamps} with a time mark, {:.0}% of the words", done.pieces, score * 100.0));
            }
            Ok(format!(
                "{} · {} pieces · {:.0}% of the words · {:.1}x real time",
                model.id,
                done.pieces,
                score * 100.0,
                done.audio / done.took.max(0.001)
            ))
        }),
    ));

    let count = checks.len();
    ui.say(&format!("{}  self-test, {count} checks", ui.paint("Omegawhisper", BOLD)));
    let started = Instant::now();
    let mut failed = 0;
    for (i, (name, check)) in checks.into_iter().enumerate() {
        let label = format!("Check {} of {count}: {name}", i + 1);
        ui.status(&label, Some(i as f32 / count as f32));
        let began = Instant::now();
        IN_CHECK.store(true, Ordering::Relaxed);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| check(&ui)))
            .unwrap_or_else(|_| Err("stopped on an internal error".into()));
        IN_CHECK.store(false, Ordering::Relaxed);
        let took = began.elapsed().as_secs_f32();
        ui.status(&label, Some((i + 1) as f32 / count as f32));
        let (mark, detail) = match outcome {
            Ok(detail) => (ui.paint("✓", GREEN), detail),
            Err(e) => {
                failed += 1;
                (ui.paint("✗", RED), ui.paint(&e, RED))
            }
        };
        ui.say(&format!(" {mark} {name:<30} {took:6.2} s   {detail}"));
    }
    ui.blank();
    let total = clock(started.elapsed().as_secs_f32());
    if failed == 0 {
        ui.say(&ui.paint(&format!("Self-test passed: {count} of {count} checks in {total}"), GREEN));
    } else {
        ui.say(&ui.paint(&format!("Self-test failed: {failed} of {count} checks in {total}"), RED));
    }
    ui.finish();
    if failed == 0 {
        0
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_match_counts_words_in_order() {
        assert_eq!(word_match("ask not what", "ask not what"), 1.0);
        assert_eq!(word_match("Ask, NOT what!", "ask not what"), 1.0);
        assert!((word_match("ask what", "ask not what") - 2.0 / 3.0).abs() < 1e-6);
        assert_eq!(word_match("", "ask"), 0.0);
    }

    #[test]
    fn a_wav_reads_back_as_written() {
        let samples = vec![0.0, 0.5, -0.5, 0.25];
        let (back, rate) = read_wav(&crate::analysis::to_wav_bytes(&samples, 22_050, 1)).unwrap();
        assert_eq!(rate, 22_050);
        for (a, b) in samples.iter().zip(&back) {
            assert!((a - b).abs() < 1e-3);
        }
        assert!(read_wav(b"not a wav file at all").is_err());
    }

    #[test]
    fn mp3_and_m4a_read_as_long_as_the_wav() {
        let (wav, wav_rate) = read_wav(JFK).unwrap();
        let wav_seconds = wav.len() as f32 / wav_rate as f32;
        for (name, bytes) in [("jfk.mp3", JFK_MP3), ("jfk.m4a", JFK_M4A)] {
            let Decoded { samples, rate, warning } = read_audio(bytes, name).unwrap();
            assert_eq!(rate, wav_rate, "{name}");
            assert!(warning.is_none(), "{name}: {warning:?}");
            let seconds = samples.len() as f32 / rate as f32;
            assert!((seconds - wav_seconds).abs() < 0.02, "{name}: {seconds} s against {wav_seconds} s");
            assert!(samples.iter().all(|x| x.is_finite() && x.abs() <= 2.0), "{name}");
            // Read the same with no name to go by.
            assert_eq!(read_audio(bytes, "").unwrap().samples.len(), samples.len(), "{name}");
        }
    }

    #[test]
    fn two_mp3_files_joined_are_read_to_the_end() {
        // The first file's header states its own length; the decoder must
        // not stop there and drop the second file.
        let one = read_audio(JFK_MP3, "jfk.mp3").unwrap().samples.len();
        let joined = [JFK_MP3, JFK_MP3].concat();
        let Decoded { samples, warning, .. } = read_audio(&joined, "joined.mp3").unwrap();
        assert!(warning.is_none(), "{warning:?}");
        assert!(samples.len() > one * 19 / 10, "{} samples against {one} for one file", samples.len());
    }

    // The first 3 s of jfk.wav at 44.1 kHz, in the left channel only, the
    // right one silent: as MP3 and as AAC in an M4A.
    const LEFT_MP3: &[u8] = include_bytes!("../tests-data/jfk_left_44k.mp3");
    const LEFT_M4A: &[u8] = include_bytes!("../tests-data/jfk_left_44k.m4a");

    #[test]
    fn stereo_44k_is_read_as_mono_in_time() {
        let wav = jfk_16k().unwrap();
        let wav = &wav[..RATE * 3];
        let rms = |x: &[f32]| (x.iter().map(|s| s * s).sum::<f32>() / x.len() as f32).sqrt();
        for (name, bytes) in [("left.mp3", LEFT_MP3), ("left.m4a", LEFT_M4A)] {
            let sound = read_audio(bytes, name).unwrap();
            assert_eq!(sound.rate, 44_100, "{name}");
            assert!(sound.warning.is_none(), "{name}: {:?}", sound.warning);
            let seconds = sound.samples.len() as f32 / 44_100.0;
            assert!((seconds - 3.0).abs() < 0.03, "{name}: {seconds} s");
            let down = to_16k(&sound.samples, sound.rate, |_| {}).unwrap();
            // One channel holds the speech and the other silence, so the mix
            // is the speech at half its level.
            let ratio = rms(&down[RATE / 2..RATE * 5 / 2]) / rms(&wav[RATE / 2..RATE * 5 / 2]);
            assert!((ratio - 0.5).abs() < 0.05, "{name}: level {ratio} of the original");
            // No encoder silence left in front: the sound lines up with the
            // WAV to the millisecond.
            let shift = (-800i64..=800)
                .map(|shift| {
                    let sum: f32 = (RATE / 2..RATE * 5 / 2)
                        .map(|i| wav[i] * down.get((i as i64 + shift) as usize).copied().unwrap_or(0.0))
                        .sum();
                    (shift, sum)
                })
                .fold((0, f32::MIN), |a, b| if b.1 > a.1 { b } else { a })
                .0;
            assert!(shift.abs() <= 16, "{name}: {shift} samples late");
        }
    }

    #[test]
    fn aac_priming_comes_from_the_file() {
        // An iTunSMPB tag whose numbers add up to what was decoded.
        assert_eq!(aac_priming(&[], Some((2112, 176_000, 64)), true, 178_176, 16_000), (2112, Some(176_000)));
        // The same at half the count: HE-AAC decoded without its SBR part.
        assert_eq!(aac_priming(&[], Some((2112, 176_000, 64)), false, 89_088, 16_000), (1056, Some(88_000)));
        // Not an MP4 file and no tag: nothing is cut.
        assert_eq!(aac_priming(b"ADTS data", None, true, 1000, 16_000), (0, None));
        // An MP4 file whose sound track's edit list skips 1024 samples at
        // 16 kHz, after an empty edit; the HE-AAC track is decoded at 8 kHz.
        let mp4_box = |name: &[u8], body: &[u8]| [&(body.len() as u32 + 8).to_be_bytes()[..], name, body].concat();
        let mdhd = mp4_box(b"mdhd", &[&[0u8; 12][..], &16_000u32.to_be_bytes(), &[0; 8]].concat());
        let hdlr = mp4_box(b"hdlr", &[&[0u8; 8][..], b"soun", &[0; 13]].concat());
        let entry = |time: i32| [&1000u32.to_be_bytes()[..], &time.to_be_bytes(), &[0, 1, 0, 0]].concat();
        let elst = mp4_box(b"elst", &[&[0u8; 4][..], &2u32.to_be_bytes(), &entry(-1), &entry(1024)].concat());
        let trak = mp4_box(b"trak", &[mp4_box(b"edts", &elst), mp4_box(b"mdia", &[mdhd, hdlr].concat())].concat());
        let file = [mp4_box(b"ftyp", b"M4A \0\0\0\0"), mp4_box(b"moov", &trak)].concat();
        assert_eq!(mp4_edit_start(&file), Some(1024.0 / 16_000.0));
        assert_eq!(aac_priming(&file, None, false, 100_000, 8_000), (512, None));
        // An MP4 file with no edit list: 2112 for plain AAC, nothing else.
        assert_eq!(mp4_edit_start(JFK_M4A), None);
        assert_eq!(aac_priming(JFK_M4A, None, true, 100_000, 16_000), (AAC_PRIMING, None));
        assert_eq!(aac_priming(JFK_M4A, None, false, 100_000, 16_000), (0, None));
        assert_eq!(mp4_edit_start(JFK_MP3), None);
    }

    #[test]
    fn damaged_and_empty_files_are_errors() {
        assert!(read_audio(&[], "empty.mp3").is_err());
        assert!(read_audio(&[], "empty.m4a").is_err());
        let garbage: Vec<u8> = noise(50_000, 1.0).iter().map(|x| (x * 127.0) as i8 as u8).collect();
        assert!(read_audio(&garbage, "garbage.mp3").is_err());
        assert!(read_audio(&garbage, "garbage.m4a").is_err());
        assert!(read_audio(b"just some text, not sound", "notes.mp3").is_err());
        let err = read_audio(b"just some text, not sound", "notes.txt").unwrap_err();
        assert!(err.contains("not supported"), "{err}");
        // A file cut short is read up to the cut, or refused, and never panics.
        for bytes in [JFK_MP3, JFK_M4A] {
            for cut in [10, 100, 1_000, bytes.len() / 2] {
                let _ = read_audio(&bytes[..cut], "cut.m4a");
                let _ = read_audio(&bytes[..cut], "cut.mp3");
            }
        }
        let half = read_audio(&JFK_MP3[..JFK_MP3.len() / 2], "half.mp3").unwrap();
        assert!(half.samples.len() > RATE * 4, "{} samples from half an MP3", half.samples.len());
    }

    #[test]
    fn rate_changes_partway_are_converted() {
        let mut out = Vec::new();
        append_at_rate(&mut out, &[0.0, 1.0, 0.0, 1.0], 8_000, 16_000);
        assert_eq!(out.len(), 8);
        assert!((out[1] - 0.5).abs() < 1e-6);
        append_at_rate(&mut out, &[], 8_000, 16_000);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn long_pieces_are_cut_under_the_limit() {
        let audio = noise(RATE * 100, 0.1);
        let pieces = split_long(&audio, 0..audio.len(), RATE * 30);
        assert!(pieces.iter().all(|p| p.len() <= RATE * 30));
        assert_eq!(pieces.first().unwrap().start, 0);
        assert_eq!(pieces.last().unwrap().end, audio.len());
        assert!(pieces.windows(2).all(|w| w[0].end == w[1].start));
    }

    #[test]
    fn pieces_cover_the_whole_file() {
        let never = |_: &[f32]| Vec::new();
        let silence = vec![0.0; 100];
        assert_eq!(cover(&silence, vec![10..20, 40..50], never), vec![0..30, 30..100]);
        let one = |r: Range<usize>| std::iter::once(r).collect::<Vec<_>>();
        assert_eq!(cover(&silence, one(10..20), never), one(0..100));
        assert!(cover(&silence, vec![], never).is_empty());
        // A long pause keeps a second on each side; its middle goes only if it
        // holds speech.
        let sec = |x: f32| (x * RATE as f32) as usize;
        let long = vec![0.0; sec(40.0)];
        assert_eq!(
            cover(&long, vec![sec(10.0)..sec(12.0), sec(30.0)..sec(31.0)], never),
            vec![sec(9.0)..sec(13.0), sec(29.0)..sec(32.0)]
        );
        assert_eq!(
            cover(&long, vec![sec(10.0)..sec(12.0), sec(30.0)..sec(31.0)], |p| std::iter::once(0..p.len()).collect()),
            vec![
                sec(0.0)..sec(9.0),
                sec(9.0)..sec(13.0),
                sec(13.0)..sec(29.0),
                sec(29.0)..sec(32.0),
                sec(32.0)..sec(40.0)
            ]
        );
        // A soft word in the middle of the pause: the cut goes around it.
        let mut audio = noise(RATE * 4, 0.3);
        for x in &mut audio[RATE..RATE * 3] {
            *x *= 0.01;
        }
        for x in &mut audio[RATE * 19 / 10..RATE * 21 / 10] {
            *x *= 30.0;
        }
        let pieces = cover(&audio, vec![0..RATE, RATE * 3..RATE * 4], |_| Vec::new());
        let cut = pieces[0].end;
        assert!(!(RATE * 19 / 10..=RATE * 21 / 10).contains(&cut), "cut at {cut}, inside the word");
    }

    #[test]
    fn a_cough_in_a_long_silence_is_a_short_piece() {
        let sec = |x: f32| (x * RATE as f32) as usize;
        let audio = vec![0.0; sec(600.0)];
        // Speech at 0-5 s and 590-600 s; something at 300-300.5 s.
        let pieces = cover(&audio, vec![0..sec(5.0), sec(590.0)..sec(600.0)], |p| {
            let at = sec(300.0) - sec(6.0);
            if p.len() > at { std::iter::once(at..at + sec(0.5)).collect() } else { Vec::new() }
        });
        assert!(pieces.iter().all(|p| p.len() <= sec(12.0)), "{pieces:?}");
        assert!(pieces.iter().any(|p| p.contains(&sec(300.2))));
    }

    #[test]
    fn short_pieces_join_a_close_neighbour() {
        let s = |a: f32, b: f32| (a * RATE as f32) as usize..(b * RATE as f32) as usize;
        let long = RATE * 90;
        // "ask not" 1 s after a sentence joins it.
        assert_eq!(join_short(vec![s(0.0, 2.4), s(3.6, 4.0), s(5.2, 11.0)], long), vec![s(0.0, 4.0), s(5.2, 11.0)]);
        // Two long pieces stay apart.
        assert_eq!(join_short(vec![s(0.0, 5.0), s(6.0, 12.0)], long), vec![s(0.0, 5.0), s(6.0, 12.0)]);
        // A short word after a long silence stays alone.
        assert_eq!(join_short(vec![s(0.0, 5.0), s(20.0, 20.5)], long), vec![s(0.0, 5.0), s(20.0, 20.5)]);
        // A short piece goes to the closer side.
        assert_eq!(join_short(vec![s(0.0, 5.0), s(7.5, 8.0), s(8.5, 15.0)], long), vec![s(0.0, 5.0), s(7.5, 15.0)]);
        // Never past the longest piece.
        assert_eq!(join_short(vec![s(0.0, 1.0), s(2.0, 3.0)], RATE * 2), vec![s(0.0, 1.0), s(2.0, 3.0)]);
    }

    #[test]
    fn languages_are_checked() {
        let prefs: Prefs = serde_json::from_str(r#"{"debug_stats":true,"shortcut":"F3"}"#).unwrap();
        assert_eq!(pick_language(Some("auto"), &prefs).unwrap(), None);
        assert_eq!(pick_language(Some("BG"), &prefs).unwrap(), Some("bg".into()));
        assert!(pick_language(Some("bulgarian"), &prefs).is_err());
        assert!(pick_language(Some("xx"), &prefs).is_err());
        assert_eq!(pick_language(Some("ja"), &prefs).unwrap(), Some("ja".into()));
    }

    #[test]
    fn the_time_mark_goes_where_the_sound_starts() {
        let mut piece = vec![0.0; RATE * 10];
        piece.extend(noise(RATE, 0.3));
        assert!((sound_start(&piece) - 10.0).abs() < 0.05);
        let mut hiss = noise(RATE * 3, 0.002);
        hiss.extend(noise(RATE, 0.3));
        assert!((sound_start(&hiss) - 3.0).abs() < 0.05);
    }

    #[test]
    fn silent_edges_are_cut_with_a_margin() {
        let mut piece = vec![0.0; RATE * 3];
        piece.extend(noise(RATE / 2, 0.3));
        piece.extend(vec![0.0; RATE * 2]);
        let cut = without_silent_edges(piece);
        let seconds = cut.len() as f32 / RATE as f32;
        assert!((seconds - 1.1).abs() < 0.05, "{seconds}");
        assert!(without_silent_edges(vec![0.0; 100]).len() == 100);
    }

    #[test]
    fn a_stock_phrase_on_a_click_is_dropped() {
        let mut click = vec![0.0; RATE];
        click.extend(noise(RATE / 5, 0.3));
        click.extend(vec![0.0; RATE]);
        assert!(invented("Thank you.", &click));
        assert!(!invented("The report is due on Friday.", &click));
        let jfk = jfk_16k().unwrap();
        assert!(!invented("Thank you.", &jfk[..RATE * 3]));
    }

    #[test]
    fn time_marks_and_bars() {
        assert_eq!(stamp(75.0, false), "[01:15]");
        assert_eq!(stamp(3725.0, true), "[1:02:05]");
        assert_eq!(clock(59.9), "0:59");
        assert_eq!(fill_bar(0.5, 4).chars().count(), 4);
        assert_eq!(fill_bar(1.0, 4), "████");
        assert_eq!(squeeze(&[0.1, 0.9, 0.2, 0.3], 2), vec![0.9, 0.3]);
    }
}
