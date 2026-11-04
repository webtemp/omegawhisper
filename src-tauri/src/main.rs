// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use clap::{CommandFactory, Parser, Subcommand};
use std::path::PathBuf;

const ABOUT: &str = "Omegawhisper - local speech to text";

const LONG_ABOUT: &str = "Omegawhisper - local speech to text

With no options, starts the app (tray, dictation key).
With -f, transcribes one audio file and exits. A running app is not touched:
this is a separate process that only reads the settings.";

const EXAMPLES: &str = "Examples:
  omegawhisper -f meeting.wav -l bg -o ~/Desktop/meeting.txt
  omegawhisper -f 20261104-090129UTC.wav          (also looks in the recordings folder)
  omegawhisper -f talk.mp3 -m parakeet-v3-int8 --no-timestamps > talk.txt
  omegawhisper -f lecture.mp4 -o lecture.txt          (reads the sound of a video)
  omegawhisper --list-models

Audio files: WAV, MP3, M4A (AAC or Apple Lossless), MP4 (its sound), FLAC,
OGG (Vorbis). Read inside the program, so no ffmpeg is needed.

While it runs, the text goes to a hidden .<name>.<id>.part file next to the -o file,
which gets the -o name when the run reaches the end. Ctrl+C keeps an old -o file.
Windows: use -o for non-English text. PowerShell's > can change the letters.";

#[derive(Parser, Debug)]
#[command(name = "omegawhisper", version, about = ABOUT, long_about = LONG_ABOUT, after_help = EXAMPLES)]
#[command(args_conflicts_with_subcommands = true)]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Audio file to transcribe: WAV, MP3, M4A, MP4, FLAC or OGG
    #[arg(short = 'f', long = "file", value_name = "PATH",
          visible_short_alias = 'i', visible_alias = "input")]
    file: Option<PathBuf>,

    /// Write the text to this file. Without it, print to the screen
    #[arg(short, long, value_name = "PATH", requires = "file")]
    output: Option<PathBuf>,

    /// Overwrite the output file if it already exists
    #[arg(long, requires = "output")]
    force: bool,

    /// bg, en, ... or "auto" for As spoken [default: the language in Settings]
    #[arg(short, long, value_name = "CODE")]
    language: Option<String>,

    /// Model to use, for example whisper-turbo, parakeet-v3-int8 [default: the model in Settings]
    #[arg(short, long, value_name = "ID")]
    model: Option<String>,

    /// Show the models and which ones are downloaded, then exit
    #[arg(long, conflicts_with_all = ["file", "test_self", "model", "language", "cpu", "quiet", "no_visual", "pause_ms", "output", "force", "no_timestamps"])]
    list_models: bool,

    /// Do not put [mm:ss] before each piece
    #[arg(long, requires = "file")]
    no_timestamps: bool,

    /// A pause this long starts a new piece
    #[arg(long, value_name = "MS", default_value_t = 1200,
          value_parser = clap::value_parser!(u32).range(200..=10_000))]
    pause_ms: u32,

    /// Do not use the GPU
    #[arg(long)]
    cpu: bool,

    /// No moving progress drawing, only plain lines
    #[arg(long)]
    no_visual: bool,

    /// Do not print progress, only the text and errors
    #[arg(short, long)]
    quiet: bool,

    /// Check every part of the -f command on built-in audio
    #[arg(long, hide = true, conflicts_with_all = ["file", "language", "quiet", "pause_ms", "output", "force", "no_timestamps"])]
    test_self: bool,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Manage transcription
    Transcribe {
        #[command(subcommand)]
        action: TranscribeAction,
    },
}

#[derive(Subcommand, Debug)]
enum TranscribeAction {
    /// Toggle transcription on/off (requires running Omegawhisper instance)
    Toggle,
}

fn main() {
    // A release build on Windows has no console of its own. Anything given on
    // the command line means a terminal started it, so its console is used.
    #[cfg(windows)]
    if std::env::args_os().len() > 1 {
        use windows_sys::Win32::System::Console::{AttachConsole, ATTACH_PARENT_PROCESS};
        unsafe {
            AttachConsole(ATTACH_PARENT_PROCESS);
        }
    }

    // Parse CLI arguments - clap handles --version and --help automatically
    let args = Args::parse();

    if let Some(Commands::Transcribe { action }) = args.command {
        match action {
            TranscribeAction::Toggle => {
                if let Err(e) = toggle_recording() {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        return;
    }

    if args.list_models {
        std::process::exit(omegawhisper_lib::cli::list_models());
    }
    if args.test_self {
        std::process::exit(omegawhisper_lib::cli::self_test(args.model, args.cpu, !args.no_visual));
    }
    if let Some(file) = args.file {
        let job = omegawhisper_lib::cli::Job {
            file,
            output: args.output,
            force: args.force,
            language: args.language,
            model: args.model,
            timestamps: !args.no_timestamps,
            pause_ms: args.pause_ms,
            cpu: args.cpu,
            quiet: args.quiet,
            visual: !args.no_visual,
        };
        std::process::exit(omegawhisper_lib::cli::transcribe_file(job));
    }

    // Flags that only mean something with -f, given without it.
    if std::env::args_os().len() > 1 {
        Args::command()
            .error(
                clap::error::ErrorKind::MissingRequiredArgument,
                "these options need a file: add -f <PATH>",
            )
            .exit();
    }

    // Run the Tauri application
    omegawhisper_lib::run()
}

/// Toggle recording via D-Bus
#[cfg(unix)]
fn toggle_recording() -> Result<(), Box<dyn std::error::Error>> {
    // Use blocking D-Bus call since we're in a simple CLI context
    let connection = zbus::blocking::Connection::session()?;

    let proxy = zbus::blocking::Proxy::new(
        &connection,
        "dev.omegawhisper",
        "/dev/omegawhisper",
        "dev.omegawhisper",
    )?;

    let _result: bool = proxy.call("ToggleRecording", &())?;

    Ok(())
}

// No D-Bus here: the dictation key is the app's own on Windows.
#[cfg(not(unix))]
fn toggle_recording() -> Result<(), Box<dyn std::error::Error>> {
    Err("transcribe toggle is only for Linux; press the dictation key instead".into())
}
