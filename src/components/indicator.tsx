import { useEffect, useRef, useState, type CSSProperties } from "react";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { db } from "@/lib/audio-level";
import { handOverBrowserSettings } from "@/lib/browser-settings";
import { DEFAULT_VISUALISATION, visualisation, type MicLevel } from "@/components/visualisations";

const BINS = 64;
const ROWS = 30;

// Everything drawn here comes from Rust, measured on the audio the recording
// actually gets. This window must never open the microphone itself: WebKit
// ignores the "raw stream" constraints and puts the device into processed
// mode, and macOS then winds its gain up over the first seconds - the first
// words of every dictation came out nearly inaudible.

// What one finished dictation did. Sent once, after the model returns, and the
// same numbers as the "dictation:" line in the log.
type DictationStats = {
  model: string;
  seconds: number;
  trimmed: number;
  shortened: number;
  level_before: number;
  level_after: number;
  gain: number;
  took: number;
  chars: number;
};

// Two short lines, not one long one: the window is 460px wide and a single
// line ran off the end of it.
//
// The first says what happened, the second what was done to the audio. The
// loudness boost is left out unless it did something - it is off for anyone
// who is not speaking quietly, and "0.145 -> 0.145  gain 1.0x" reads as a
// broken number rather than as "nothing needed doing".
function describeDictation(s: DictationStats): string[] {
  const boosted = s.gain > 1.005;
  return [
    `${s.model}  ${s.seconds.toFixed(1)}s  took ${s.took.toFixed(1)}s  ${s.chars} chars`,
    `trim ${s.trimmed.toFixed(1)}s  pauses ${s.shortened.toFixed(1)}s  speech ` +
      (boosted
        ? `${s.level_before.toFixed(3)}→${s.level_after.toFixed(3)} (gain ${s.gain.toFixed(1)}x)`
        : s.level_after.toFixed(3)),
  ];
}

// A ring of bars pointing outwards, their lengths running around the circle in
// a wave. Shown while the model is working, where there is no live sound to
// draw, so the wave is made from three sine waves at different speeds instead:
// no two moments look the same and it never sits still.
//
// Short bars are blue, long bars white, through cyan in between, so the shape
// reads as movement rather than a rotating ring.
function drawWaveRing(ctx: CanvasRenderingContext2D, w: number, h: number, t: number) {
  const BARS = 84;
  const cx = w / 2;
  const cy = h / 2;
  const size = Math.min(w, h);
  const inner = size * 0.14;
  const reach = size * 0.13;

  ctx.clearRect(0, 0, w, h);
  ctx.lineCap = "round";

  // Faint circle the bars stand on, so the shape holds together when most
  // bars are short.
  ctx.beginPath();
  ctx.arc(cx, cy, inner - 3, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(120, 200, 255, 0.18)";
  ctx.lineWidth = 1;
  ctx.stroke();

  for (let i = 0; i < BARS; i++) {
    const angle = (i / BARS) * Math.PI * 2 + t * 0.22;
    const wave =
      (Math.sin(i * 0.42 - t * 3.1) +
        Math.sin(i * 0.17 + t * 2.0) +
        Math.sin(i * 0.91 + t * 1.3)) /
      3;
    const level = (wave + 1) / 2; // 0 to 1
    const len = inner * 0.12 + reach * level;

    // blue -> cyan -> white as the bar gets longer
    let r: number, g: number, b: number;
    if (level < 0.5) {
      const k = level * 2;
      r = 37 + (56 - 37) * k;
      g = 99 + (189 - 99) * k;
      b = 235 + (248 - 235) * k;
    } else {
      const k = (level - 0.5) * 2;
      r = 56 + (255 - 56) * k;
      g = 189 + (255 - 189) * k;
      b = 248 + (255 - 248) * k;
    }
    const color = `rgb(${r | 0}, ${g | 0}, ${b | 0})`;

    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    ctx.beginPath();
    ctx.moveTo(cx + cos * inner, cy + sin * inner);
    ctx.lineTo(cx + cos * (inner + len), cy + sin * (inner + len));
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.4;
    ctx.shadowColor = color;
    ctx.shadowBlur = 6 + level * 6;
    ctx.stroke();
  }
  ctx.shadowBlur = 0;
}

// Isometric waterfall spectrogram shown (in its own window) while recording.
// The mic is opened only while the window is active, driven by "indicator-active".
export function Indicator() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const waveRef = useRef<HTMLCanvasElement | null>(null);
  const timerRef = useRef<HTMLSpanElement | null>(null);
  const statsRef = useRef<HTMLDivElement | null>(null);
  const textRef = useRef<HTMLDivElement | null>(null);
  // True between the stop and the finished text: the model is running and the
  // whole app is unresponsive, so this has to be obvious.
  const [transcribing, setTranscribing] = useState(false);
  // Something went wrong. Shown here because this is the window that is
  // actually on screen when it happens - the main window is normally hidden.
  const [errorText, setErrorText] = useState<string | null>(null);
  // The drawing loop is set up once and cannot read React state.
  const errorRef = useRef(false);
  // Which picture the drawing loop makes of the sound. Chosen in Settings.
  const vizRef = useRef(DEFAULT_VISUALISATION);

  // Anything Rust found wrong at startup: a dictation key it could not
  // register, a permission it was not given. This is the only window the user
  // ever sees, so a warning shown anywhere else is a warning nobody reads.
  const [startupWarning, setStartupWarning] = useState<string | null>(null);

  // The line of live numbers is off unless switched on in the tray menu.
  const [showStats, setShowStats] = useState(false);
  // The drawing loop cannot read React state, so it reads this.
  const showStatsRef = useRef(false);
  // What the last dictation did. Cleared when the next one starts, so an old
  // line is never read as the new one.
  const [lastDictation, setLastDictation] = useState<DictationStats | null>(null);

  // The settings the deleted main window kept in browser storage. This window
  // loads on every launch, shown or not, so it is the one that can hand them
  // over without the user having to open anything. Rust copies them once.
  useEffect(() => {
    handOverBrowserSettings().catch((err) =>
      console.error("Could not hand the old settings to Rust:", err)
    );
  }, []);

  // Asked for rather than pushed: Rust finds these before this window exists
  // to hear about them. Rust puts the window on screen once it is told there
  // is something to show.
  useEffect(() => {
    invoke<string[]>("get_startup_warnings")
      .then((warnings) => {
        if (warnings.length === 0) return;
        setStartupWarning(warnings.join(" "));
        invoke("show_startup_warning").catch(() => {});
      })
      .catch(() => {});
  }, []);

  // The ring only draws while the model is working.
  useEffect(() => {
    if (!transcribing) return;
    let raf = 0;
    let start = 0;

    const draw = (now: number) => {
      if (!start) start = now;
      const canvas = waveRef.current;
      if (canvas) {
        const dpr = window.devicePixelRatio || 1;
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
          canvas.width = w * dpr;
          canvas.height = h * dpr;
        }
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          drawWaveRing(ctx, w, h, (now - start) / 1000);
        }
      }
      if (timerRef.current) {
        timerRef.current.textContent = `${((now - start) / 1000).toFixed(1)}s`;
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);

    return () => cancelAnimationFrame(raf);
  }, [transcribing]);

  useEffect(() => {
    invoke<{ visualisation: string }>("get_settings")
      .then((s) => {
        vizRef.current = s.visualisation;
      })
      .catch(() => {});
    const unlisten = listen<string>("visualisation-changed", (e) => {
      vizRef.current = e.payload;
    });
    return () => {
      unlisten.then((fn) => fn()).catch(() => {});
    };
  }, []);

  useEffect(() => {
    const apply = (on: boolean) => {
      showStatsRef.current = on;
      setShowStats(on);
    };
    invoke<boolean>("get_debug_stats")
      .then(apply)
      .catch(() => {});
    const unlisten = listen<boolean>("debug-stats-changed", (e) =>
      apply(e.payload)
    );

    return () => {
      unlisten.then((fn) => fn()).catch(() => {});
    };
  }, []);

  useEffect(() => {
    let raf = 0;
    let frame = 0;
    // Newest numbers from the recording itself; empty until the first arrives.
    const quiet = (): MicLevel => ({
      peak: 0, rms: 0, seconds: 0, pitch: 0, bands: [], pause: 0, typing: false, sentences: 0,
      armed: false,
    });
    let mic: MicLevel = quiet();
    // Live typing: a bar along the bottom fills as the pause runs towards the
    // cut, and flashes when a sentence has gone out.
    let sentencesSeen = 0;
    let flashUntil = 0;

    function drawLiveState(ctx: CanvasRenderingContext2D, w: number, h: number) {
      const now = performance.now();
      if (mic.armed && mic.sentences === 0 && !mic.typing) {
        ctx.font = "600 11px ui-sans-serif, system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillStyle = "rgba(150, 220, 255, 0.85)";
        ctx.shadowColor = "rgba(0, 0, 0, 0.9)";
        ctx.shadowBlur = 4;
        ctx.fillText("still listening", w / 2, h - 8);
        ctx.shadowBlur = 0;
      }
      if (mic.sentences > sentencesSeen) {
        sentencesSeen = mic.sentences;
        flashUntil = now + 450;
      }
      const flash = Math.max(0, (flashUntil - now) / 450);
      if (flash > 0) {
        ctx.fillStyle = `rgba(140, 255, 200, ${0.9 * flash})`;
        ctx.fillRect(0, h - 3, w, 3);
        return;
      }
      if (mic.typing) {
        ctx.fillStyle = "rgba(140, 255, 200, 0.6)";
        ctx.fillRect(0, h - 3, w, 3);
        return;
      }
      if (mic.pause > 0) {
        ctx.fillStyle = "rgba(255, 255, 255, 0.12)";
        ctx.fillRect(0, h - 3, w, 3);
        ctx.fillStyle = `rgba(150, 220, 255, ${0.35 + 0.55 * mic.pause})`;
        ctx.fillRect(0, h - 3, w * mic.pause, 3);
      }
    }

    const history: number[][] = Array.from({ length: ROWS }, () =>
      new Array(BINS).fill(0)
    );

    // One row of the waterfall, taken from the bands Rust sends. Smoothed
    // against the previous row so the surface flows instead of flickering.
    function pushRow() {
      const previous = history[0];
      const row = new Array(BINS).fill(0);
      for (let i = 0; i < BINS; i++) {
        const value = mic.bands[i] ?? 0;
        row[i] = previous[i] * 0.45 + value * 0.55;
      }
      history.pop();
      history.unshift(row);
    }

    // Numbers above the spectrogram. The level and peak come from Rust, so
    // they show what the recording gets, not what this window hears - that is
    // the pair that matters when a dictation comes back empty.
    function updateStats() {
      const el = statsRef.current;
      if (!el || !showStatsRef.current) return;

      // A quiet microphone is fine: speech is recognised by its shape and
      // raised afterwards. Only nothing at all, or clipping, is a problem.
      let state = "ok";
      let color = "rgba(226, 248, 255, 0.75)";
      if (mic.peak > 0.98) {
        state = "TOO LOUD";
        color = "rgb(255, 138, 128)";
      } else if (mic.seconds > 1 && mic.peak < 0.001) {
        state = "NO SIGNAL";
        color = "rgb(255, 196, 100)";
      } else if (mic.rms < 0.005) {
        state = "quiet";
      }

      // Every field is padded to a fixed width. Without that the numbers
      // change length as you speak and the whole line shifts sideways, which
      // makes it unreadable.
      el.style.color = color;
      el.textContent =
        `${mic.seconds.toFixed(1).padStart(5)}s  ` +
        `rec ${db(mic.rms).padStart(4)} dB  ` +
        `peak ${db(mic.peak).padStart(4)} dB  ` +
        `pitch ${(mic.pitch > 0 ? mic.pitch.toFixed(0) : "--").padStart(3)} Hz  ` +
        state.padEnd(9);
    }

    function draw() {
      const canvas = canvasRef.current;
      if (!canvas) {
        raf = requestAnimationFrame(draw);
        return;
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        // Without this the loop ends for good and the window stays blank.
        raf = requestAnimationFrame(draw);
        return;
      }

      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr;
        canvas.height = h * dpr;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      frame++;
      if (frame % 2 === 0) pushRow();
      // Every frame is unreadable; ~6 times a second is not.
      if (frame % 10 === 0) updateStats();

      visualisation(vizRef.current).draw(ctx, w, h, history, mic, frame);
      drawLiveState(ctx, w, h);

      raf = requestAnimationFrame(draw);
    }

    draw();

    // Nothing to start or stop any more - the numbers arrive from Rust while
    // it records. Going inactive only clears what is on screen.
    const setActive = (active: boolean) => {
      if (!active) {
        mic = quiet();
        sentencesSeen = 0;
        flashUntil = 0;
        setTranscribing(false);
        errorRef.current = false;
        setErrorText(null);
        setStartupWarning(null);
        if (textRef.current) textRef.current.textContent = "";
        for (const row of history) row.fill(0);
      }
    };
    const unlisteners: Promise<() => void>[] = [];
    try {
      unlisteners.push(
        listen<boolean>("indicator-active", (e) => setActive(!!e.payload)),
        listen<MicLevel>("mic-level", (e) => {
          mic = e.payload;
        }),
        // Only the streaming backends send text while you speak; local models
        // send one event at the end, which shows up here for a moment.
        listen<{ text: string }>("transcription", (e) => {
          if (textRef.current) textRef.current.textContent = e.payload.text;
        }),
        listen<string>("transcription-error", (e) => {
          errorRef.current = true;
          setErrorText(e.payload);
          setTranscribing(false);
        }),
        listen("transcription-processing", () => {
          setTranscribing(true);
        }),
        // Sent once the model returns, including when the recording was
        // skipped for holding no speech.
        listen<DictationStats>("dictation-stats", (e) => {
          setLastDictation(e.payload);
        }),
        listen("transcription-complete", () => {
          setTranscribing(false);
        }),
        listen<boolean>("mic-level", () => {
          // Sound is arriving again, so the last error is history.
          if (errorRef.current) {
            errorRef.current = false;
            setErrorText(null);
          }
          // And so is the last dictation's line. These only arrive while
          // recording, which is the one moment that means a new dictation has
          // started - clearing on "transcription-processing" instead left the
          // previous run's numbers on screen for the whole of the next one.
          // Returning the old value unchanged costs nothing when it is
          // already empty, which it is 20 times a second.
          setLastDictation((previous) => (previous ? null : previous));
        })
      );
    } catch {
      // events unavailable outside the app: the numbers just stay at zero
    }

    const onVisibility = () => setActive(!document.hidden);
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      cancelAnimationFrame(raf);
      document.removeEventListener("visibilitychange", onVisibility);
      unlisteners.forEach((u) => u.then((fn) => fn()).catch(() => {}));
    };
  }, []);

  const overlay: CSSProperties = {
    position: "absolute",
    left: 0,
    right: 0,
    textAlign: "center",
    fontFamily:
      "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
    fontSize: 10,
    letterSpacing: 0.2,
    whiteSpace: "nowrap",
    pointerEvents: "none",
    // The window has no background, so the text needs its own outline to stay
    // readable over whatever is behind it.
    textShadow: "0 1px 3px rgba(0, 0, 0, 0.9), 0 0 8px rgba(0, 0, 0, 0.7)",
  };

  return (
    <div
      style={{
        position: "relative",
        width: "100vw",
        height: "100vh",
        background: "transparent",
        overflow: "hidden",
      }}
    >
      <canvas
        ref={canvasRef}
        style={{
          width: "100%",
          height: "100%",
          // The spectrogram is frozen while the model runs, so fade it out and
          // let the spinner have the window.
          opacity: transcribing ? 0.15 : 1,
          transition: "opacity 150ms",
        }}
      />
      {/* A dark plate behind the numbers. On a transparent window over light
          content the shadow alone was not enough to read them. */}
      <div
        style={{
          position: "absolute",
          top: 2,
          left: 0,
          right: 0,
          display: showStats ? "flex" : "none",
          flexDirection: "column",
          alignItems: "center",
          gap: 3,
          pointerEvents: "none",
        }}
      >
        <div
          ref={statsRef}
          style={{
            fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
            fontSize: 11,
            whiteSpace: "pre",
            padding: "3px 10px",
            borderRadius: 7,
            background: "rgba(10, 14, 20, 0.82)",
            border: "1px solid rgba(255, 255, 255, 0.12)",
          }}
        />
        {/* What the dictation that just finished did. The window is held on
            screen for a few seconds while this is on, so it can be read. */}
        {lastDictation && (
          <div
            style={{
              fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
              fontSize: 10,
              lineHeight: 1.5,
              textAlign: "center",
              // Wraps rather than being cut off. Nothing here is worth hiding.
              overflowWrap: "anywhere",
              padding: "3px 10px",
              borderRadius: 7,
              background: "rgba(10, 14, 20, 0.82)",
              border: "1px solid rgba(255, 255, 255, 0.12)",
              // A dictation that typed nothing is the one worth noticing.
              color:
                lastDictation.chars === 0
                  ? "rgb(255, 196, 100)"
                  : "rgba(226, 248, 255, 0.75)",
            }}
          >
            {describeDictation(lastDictation).map((line) => (
              <div key={line}>{line}</div>
            ))}
          </div>
        )}
      </div>
      <div
        ref={textRef}
        style={{
          ...overlay,
          bottom: 4,
          fontSize: 11,
          color: "rgba(226, 248, 255, 0.85)",
          overflow: "hidden",
          textOverflow: "ellipsis",
          paddingLeft: 8,
          paddingRight: 8,
        }}
      />
      {(errorText ?? startupWarning) && (
        <div
          className="absolute inset-0 flex items-center justify-center px-4"
          style={{ pointerEvents: "none" }}
        >
          <div
            style={{
              maxHeight: "100%",
              overflowY: "auto",
              padding: "10px 14px",
              borderRadius: 10,
              background: "rgba(24, 10, 10, 0.94)",
              border: "1px solid rgba(255, 138, 128, 0.45)",
              color: "rgb(255, 190, 184)",
              fontFamily: "ui-sans-serif, system-ui, -apple-system, sans-serif",
              fontSize: 12,
              lineHeight: 1.45,
              textAlign: "left",
            }}
          >
            {errorText ?? startupWarning}
          </div>
        </div>
      )}
      {transcribing && !errorText && !startupWarning && (
        <div
          className="absolute inset-0 flex flex-col items-center justify-center gap-3"
          style={{ pointerEvents: "none" }}
        >
          <canvas
            ref={waveRef}
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
            }}
          />
          <div
            style={{
              position: "absolute",
              left: 0,
              right: 0,
              top: "50%",
              marginTop: 62,
              textAlign: "center",
              fontFamily: "ui-sans-serif, system-ui, -apple-system, sans-serif",
              fontSize: 16,
              fontWeight: 600,
              letterSpacing: 0.6,
              color: "rgb(240, 252, 255)",
              textShadow:
                "0 1px 4px rgba(0, 0, 0, 0.95), 0 0 14px rgba(0, 0, 0, 0.8)",
            }}
          >
            Transcribing
            <span className="inline-block animate-pulse">...</span>
            {/* Counts up while the model runs, so a long wait is visibly a
                wait and not a freeze. */}
            <span
              ref={timerRef}
              style={{
                marginLeft: 8,
                fontVariantNumeric: "tabular-nums",
                fontWeight: 500,
                color: "rgba(150, 220, 255, 0.95)",
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
