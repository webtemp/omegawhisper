use crate::managers::model::{EngineType, ModelManager, AVAILABLE_MODELS};
use std::sync::{Arc, Mutex};
use transcribe_rs::onnx::moonshine::{MoonshineModel, MoonshineParams, MoonshineVariant};
use transcribe_rs::onnx::parakeet::{ParakeetModel, ParakeetParams};
use transcribe_rs::onnx::Quantization;
use transcribe_rs::whisper_cpp::{WhisperEngine, WhisperInferenceParams, WhisperLoadParams};

/// Which of the two runtimes may use the GPU. Whisper runs on whisper.cpp and
/// Metal; the rest run on ONNX Runtime and CoreML. They share nothing, so they
/// are answered separately.
///
/// Both are settled while a model is being built and cannot be changed while
/// it is in memory, which is what `needs_load` below is for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GpuChoice {
    /// CoreML for Parakeet and Moonshine.
    pub onnx: bool,
    /// Metal for Whisper.
    pub whisper: bool,
}

/// What the ONNX switch means to the transcription library. `Auto` is its own
/// pick of the best provider, which on a Mac is CoreML.
pub fn onnx_accelerator(gpu: bool) -> transcribe_rs::OrtAccelerator {
    if gpu {
        transcribe_rs::OrtAccelerator::Auto
    } else {
        transcribe_rs::OrtAccelerator::CpuOnly
    }
}

/// Whether a dictation has to build the model, given what is already in memory.
///
/// The switches count as much as the name: a model built while a switch was on
/// goes on using it. Comparing the name alone would leave a moved switch doing
/// nothing until the app was restarted or another model chosen.
pub fn needs_load(loaded: Option<(&str, GpuChoice)>, model_id: &str, gpu: GpuChoice) -> bool {
    loaded != Some((model_id, gpu))
}

/// Loaded transcription engine
enum LoadedEngine {
    Whisper(WhisperEngine),
    Parakeet(ParakeetModel),
    Moonshine(MoonshineModel),
}

// Whisper writes in whichever style it starts in, so long recordings often come
// back with no capitals and no punctuation. It copies the style of this text
// instead, and reads it as background it never types out.
//
// Deliberately about nothing: any subject in here becomes words Whisper expects
// to hear. And only in the language chosen: Whisper also keeps to the language
// of its prompt, so an English prompt turned Bulgarian speech into English.
// With the language left to detection there is no prompt.
const STYLE_PROMPTS: [(&str, &str); 2] = [
    (
        "en",
        "Hello. This is an ordinary sentence, written the normal way, with commas \
         where they belong and a full stop at the end. On Monday I told Maria that \
         the work would be done by January. Do you see how it reads? Yes, exactly \
         like that.",
    ),
    (
        "bg",
        "Здравей. Това е обикновено изречение, написано по нормалния начин, със \
         запетаи, където им е мястото, и точка в края. В понеделник казах на Мария, \
         че работата ще бъде готова до януари. Виждаш ли как се чете? Да, точно така.",
    ),
];

pub fn style_prompt(language: Option<&str>) -> Option<String> {
    let language = language?;
    STYLE_PROMPTS
        .iter()
        .find(|(code, _)| *code == language)
        .map(|(_, prompt)| prompt.to_string())
}

impl LoadedEngine {
    /// Transcribe audio samples (expects 16kHz mono f32 audio).
    /// Samples go straight to the engine - no temp WAV file needed.
    fn transcribe(&mut self, samples: &[f32], language: Option<String>) -> Result<String, String> {
        let result = match self {
            LoadedEngine::Whisper(engine) => {
                // no_speech_thold is whisper.cpp's own default of 0.6, not the
                // 0.2 that transcribe-rs sets. Whisper reads audio in 30 second
                // windows and throws a whole window away when the chance of it
                // being speech is below this number. At 0.2 a quiet microphone
                // loses most windows, which is why a long dictation came back
                // as only its last few sentences, or as nothing at all.
                let params = WhisperInferenceParams {
                    initial_prompt: style_prompt(language.as_deref()),
                    language,
                    no_speech_thold: 0.6,
                    ..Default::default()
                };
                engine
                    .transcribe_with(samples, &params)
                    .map_err(|e| format!("Whisper transcription error: {}", e))
            }
            LoadedEngine::Parakeet(model) => {
                // transcribe_with prepends 250ms of silence itself: Parakeet's
                // mel spectrogram preprocessor weakens the start of the audio,
                // which drops the first words without that padding.
                model
                    .transcribe_with(samples, &ParakeetParams::default())
                    .map_err(|e| format!("Parakeet transcription error: {}", e))
            }
            LoadedEngine::Moonshine(model) => model
                .transcribe_with(samples, &MoonshineParams::default())
                .map_err(|e| format!("Moonshine transcription error: {}", e)),
        };

        result.map(|r| r.text)
    }
}

/// Transcription manager handles loading and using transcription models
pub struct TranscriptionManager {
    loaded_engine: Option<LoadedEngine>,
    current_model_id: Option<String>,
    /// Which runtime had the GPU when the model in memory was built.
    loaded_with: Option<GpuChoice>,
    model_manager: Arc<ModelManager>,
}

impl TranscriptionManager {
    /// Create a new transcription manager
    pub fn new(model_manager: Arc<ModelManager>) -> Self {
        Self {
            loaded_engine: None,
            current_model_id: None,
            loaded_with: None,
            model_manager,
        }
    }

    /// What is in memory now: the model and the switches it was built with.
    fn loaded(&self) -> Option<(&str, GpuChoice)> {
        self.current_model_id.as_deref().zip(self.loaded_with)
    }

    /// Check if a model is currently loaded
    /// Get the currently loaded model ID
    pub fn get_loaded_model_id(&self) -> Option<&str> {
        self.current_model_id.as_deref()
    }

    /// Load a model by ID
    pub fn load_model(&mut self, model_id: &str, gpu: GpuChoice) -> Result<(), String> {
        // Already in memory, and built the way the settings ask for.
        if !needs_load(self.loaded(), model_id, gpu) {
            return Ok(());
        }

        // Unload current model first
        self.unload_model();

        // Has to be set before the model is built: each ONNX session picks its
        // provider as it is created, and keeps it. Whisper ignores this and is
        // told separately below.
        transcribe_rs::set_ort_accelerator(onnx_accelerator(gpu.onnx));

        // Get model info
        let model_info = AVAILABLE_MODELS
            .iter()
            .find(|m| m.id == model_id)
            .ok_or_else(|| format!("Unknown model: {}", model_id))?;

        // Check if model is downloaded
        if !self.model_manager.is_model_downloaded(model_id) {
            return Err(format!("Model {} is not downloaded", model_id));
        }

        let model_path = self.model_manager.get_model_path(model_id);

        // Load the appropriate engine based on model type
        let engine = match model_info.engine_type {
            EngineType::Whisper => {
                let files = model_info.get_files();
                let model_file = files.first().ok_or("No model file defined")?;
                let full_path = model_path.join(model_file.filename);

                // Load explicitly instead of WhisperEngine::load, whose defaults
                // turn flash attention on. Flash attention was off before the
                // transcribe-rs 0.3 upgrade and turning it on made Whisper
                // output repeated nonsense, sometimes in the wrong language.
                //
                // use_gpu has to be named here. WhisperLoadParams::default()
                // sets it to true whatever the settings say - only
                // WhisperEngine::load reads them, and that is the call this
                // does not use, because of the flash attention note above.
                let params = WhisperLoadParams {
                    flash_attn: false,
                    use_gpu: gpu.whisper,
                    ..Default::default()
                };
                let whisper = WhisperEngine::load_with_params(&full_path, params)
                    .map_err(|e| format!("Failed to load Whisper model: {}", e))?;

                LoadedEngine::Whisper(whisper)
            }
            EngineType::Parakeet => {
                // Int8 resolves to encoder-model.int8.onnx / decoder_joint-model.int8.onnx,
                // which is what we download.
                let model = ParakeetModel::load(&model_path, &Quantization::Int8)
                    .map_err(|e| format!("Failed to load Parakeet model: {}", e))?;

                LoadedEngine::Parakeet(model)
            }
            EngineType::Moonshine => {
                // FP32 means no quantization suffix: encoder_model.onnx /
                // decoder_model_merged.onnx, matching the downloaded files.
                let model =
                    MoonshineModel::load(&model_path, MoonshineVariant::Base, &Quantization::FP32)
                        .map_err(|e| format!("Failed to load Moonshine model: {}", e))?;

                LoadedEngine::Moonshine(model)
            }
        };

        self.loaded_engine = Some(engine);
        self.current_model_id = Some(model_id.to_string());
        self.loaded_with = Some(gpu);
        // Only the switch this model answers to. Naming the other one as well
        // reads as though it were in use.
        let on_gpu = match model_info.engine_type {
            EngineType::Whisper => gpu.whisper,
            _ => gpu.onnx,
        };
        eprintln!(
            "Loaded {} on the {}",
            model_id,
            if on_gpu { "GPU" } else { "processor" }
        );

        Ok(())
    }

    /// Unload the current model
    pub fn unload_model(&mut self) {
        if let Some(model_id) = self.current_model_id.take() {
            eprintln!("Unloading model {}", model_id);
        }
        self.loaded_engine = None;
        self.loaded_with = None;
    }

    /// Transcribe 16kHz mono f32 audio. `language` None = auto-detect.
    pub fn transcribe(
        &mut self,
        samples: &[f32],
        language: Option<String>,
    ) -> Result<String, String> {
        let engine = self.loaded_engine.as_mut().ok_or("No model loaded")?;

        engine.transcribe(samples, language)
    }
}

/// Thread-safe wrapper for TranscriptionManager
pub struct SharedTranscriptionManager(pub Arc<Mutex<TranscriptionManager>>);

impl SharedTranscriptionManager {
    pub fn new(model_manager: Arc<ModelManager>) -> Self {
        Self(Arc::new(Mutex::new(TranscriptionManager::new(
            model_manager,
        ))))
    }

    pub fn get_loaded_model_id(&self) -> Option<String> {
        self.0
            .lock()
            .ok()
            .and_then(|m| m.get_loaded_model_id().map(|s| s.to_string()))
    }

    pub fn unload_model(&self) {
        if let Ok(mut manager) = self.0.lock() {
            manager.unload_model();
        }
    }
}
