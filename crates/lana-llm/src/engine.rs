//! [`LlmEngine`] — streaming chat over candle + a quantized LFM2 GGUF
//! (Luth-LFM2: French-specialised Liquid LFM2).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_lfm2::ModelWeights;
use candle_transformers::utils::apply_repeat_penalty;
use hf_hub::Repo;
use hf_hub::api::sync::ApiBuilder;
use parking_lot::Mutex;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::error::LlmError;
use crate::token_stream::TokenOutputStream;

const TOKEN_CHANNEL_CAPACITY: usize = 256;
/// End-of-turn token: the `<|im_end|>` marker, `LFM2`/`Luth` eos.
const EOS_TOKEN: &str = "<|im_end|>";

/// Default model: French-specialised Luth-LFM2-1.2B, fetched from the Hub
/// (cached) so nothing is downloaded by hand. `Q8_0` (~1.25 GB) is the
/// smallest published quant and near-lossless for a 1.2 B model.
const MODEL_REPO: &str = "kurakurai/Luth-LFM2-1.2B-GGUF";
const MODEL_FILE: &str = "Luth-LFM2-1.2B-Q8_0.gguf";
/// Tokenizer lives in the base (non-GGUF) repo.
const TOKENIZER_REPO: &str = "kurakurai/Luth-LFM2-1.2B";
const TOKENIZER_FILE: &str = "tokenizer.json";

/// Resolve the GGUF weights and tokenizer paths.
///
/// An explicit local path (`model_env` / `tok_env`, from `LANA_MODEL_PATH`
/// / `LANA_TOKENIZER_PATH`) wins; otherwise the default Luth-LFM2 assets
/// are downloaded from the Hugging Face Hub and cached (no HF token
/// required — public repos). Performs blocking network I/O on first run;
/// call it off the async runtime (e.g. via `spawn_blocking`).
///
/// # Errors
///
/// Returns [`LlmError::Load`] if the Hub API or a download fails.
pub fn resolve_model_assets(
    model_env: Option<String>,
    tok_env: Option<String>,
) -> Result<(PathBuf, PathBuf), LlmError> {
    let api = ApiBuilder::new()
        .with_token(std::env::var("HF_TOKEN").ok())
        .build()
        .map_err(|e| LlmError::Load(format!("hf-hub api: {e}")))?;

    let fetch = |repo: &str, file: &str| -> Result<PathBuf, LlmError> {
        info!(
            repo,
            file, "resolving model asset from Hugging Face (cached)"
        );
        api.repo(Repo::model(repo.to_owned()))
            .get(file)
            .map_err(|e| LlmError::Load(format!("download {repo}/{file}: {e}")))
    };

    let model_path = match model_env {
        Some(p) => PathBuf::from(p),
        None => fetch(MODEL_REPO, MODEL_FILE)?,
    };
    let tokenizer_path = match tok_env {
        Some(p) => PathBuf::from(p),
        None => fetch(TOKENIZER_REPO, TOKENIZER_FILE)?,
    };
    Ok((model_path, tokenizer_path))
}

/// Static configuration of a [`LlmEngine`].
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Absolute path to the GGUF file (e.g. `Qwen3-1.7B-Q5_K_M.gguf`).
    pub model_path: PathBuf,
    /// Absolute path to the tokenizer.json for the same model.
    pub tokenizer_path: PathBuf,
    /// System prompt prepended to every request.
    pub system_prompt: String,
    /// Generation hyper-parameters.
    pub generation: GenerationConfig,
}

/// Tunable sampling parameters.
#[derive(Debug, Clone, Copy)]
pub struct GenerationConfig {
    /// Sampling temperature. `0.0` triggers greedy/argmax decoding.
    pub temperature: f64,
    /// Top-p nucleus sampling cutoff, when set.
    pub top_p: Option<f64>,
    /// Top-k truncation, when set.
    pub top_k: Option<usize>,
    /// Hard cap on generated tokens per request.
    pub max_tokens: usize,
    /// Repetition penalty (`1.0` disables it).
    pub repeat_penalty: f32,
    /// Window of recent tokens considered for the repetition penalty.
    pub repeat_last_n: usize,
    /// Deterministic seed for the sampler.
    pub seed: u64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: Some(0.9),
            top_k: None,
            max_tokens: 1024,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: 299_792_458,
        }
    }
}

/// Who authored a conversation turn (the system prompt is configured
/// separately on the engine, so it is not represented here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// The human speaker.
    User,
    /// Lana's own prior replies (fed back so she has memory).
    Assistant,
}

impl Role {
    const fn tag(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

/// One conversation turn in the running history.
#[derive(Debug, Clone)]
pub struct Message {
    /// Turn author.
    pub role: Role,
    /// Turn text.
    pub content: String,
}

impl Message {
    /// A user turn.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    /// An assistant (Lana) turn.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// One streamed delta of generated text plus a terminal flag.
#[derive(Debug, Clone)]
pub struct TokenChunk {
    /// Text emitted in this chunk (may span multiple tokens).
    pub text: String,
    /// `true` on the last chunk of a response.
    pub is_final: bool,
}

/// Streaming chat engine backed by candle on Metal.
pub struct LlmEngine {
    model: Arc<Mutex<ModelWeights>>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    eos_token: u32,
    system_prompt: Arc<str>,
    generation: GenerationConfig,
}

impl std::fmt::Debug for LlmEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmEngine")
            .field("device", &self.device)
            .field("system_prompt", &self.system_prompt)
            .field("generation", &self.generation)
            .finish_non_exhaustive()
    }
}

impl LlmEngine {
    /// Load the GGUF weights and tokenizer into a Metal-resident engine.
    ///
    /// Heavy I/O and tensor allocation is performed on a blocking task so the
    /// async runtime stays responsive.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::Load`] if the GGUF file, tokenizer, or Metal
    /// device cannot be initialised.
    pub async fn new(config: EngineConfig) -> Result<Self, LlmError> {
        let model_path = config.model_path.clone();
        let tokenizer_path = config.tokenizer_path.clone();

        let loaded =
            tokio::task::spawn_blocking(move || load_weights(&model_path, &tokenizer_path))
                .await
                .map_err(|e| LlmError::Load(format!("blocking join: {e}")))??;

        let engine = Self {
            model: Arc::new(Mutex::new(loaded.model)),
            tokenizer: Arc::new(loaded.tokenizer),
            device: loaded.device,
            eos_token: loaded.eos_token,
            system_prompt: Arc::from(config.system_prompt),
            generation: config.generation,
        };

        engine.warmup().await?;
        Ok(engine)
    }

    /// Run one tiny forward pass to compile Metal kernels ahead of the first
    /// user prompt. Subsequent inferences hit the warm cache and see TTFT in
    /// the low hundreds of milliseconds instead of multi-second cold starts.
    async fn warmup(&self) -> Result<(), LlmError> {
        info!("warming up metal kernels");
        let model = Arc::clone(&self.model);
        let device = self.device.clone();
        let started = std::time::Instant::now();
        tokio::task::spawn_blocking(move || -> Result<(), LlmError> {
            let mut weights = model.lock();
            let input = Tensor::new(&[0_u32, 1_u32, 2_u32], &device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| LlmError::Load(format!("warmup tensor: {e}")))?;
            // index_pos = 0 with a multi-token input resets LFM2 state
            // (attention KV ignored, short-conv recomputed) — no explicit
            // cache-clear API and none needed.
            let _ = weights
                .forward(&input, 0)
                .map_err(|e| LlmError::Load(format!("warmup forward: {e}")))?;
            drop(weights);
            Ok(())
        })
        .await
        .map_err(|e| LlmError::Load(format!("warmup join: {e}")))??;
        info!(elapsed_s = started.elapsed().as_secs_f32(), "warmup done");
        Ok(())
    }

    /// Send the full conversation `history` (oldest → newest, ending with
    /// the latest user turn) and receive a stream of [`TokenChunk`]s. The
    /// caller owns the history so Lana has memory across turns.
    ///
    /// A terminal chunk with `is_final == true` is always sent before the
    /// channel closes (even on mid-stream error). Each call clears the KV
    /// cache and re-encodes the whole history (stateless completion).
    #[must_use]
    pub fn stream(&self, history: &[Message]) -> mpsc::Receiver<TokenChunk> {
        let (tx, rx) = mpsc::channel(TOKEN_CHANNEL_CAPACITY);
        let model = Arc::clone(&self.model);
        let tokenizer = Arc::clone(&self.tokenizer);
        let device = self.device.clone();
        let eos = self.eos_token;
        let system_prompt = Arc::clone(&self.system_prompt);
        let history = history.to_vec();
        let generation = self.generation;

        tokio::task::spawn_blocking(move || {
            run_completion(
                &model,
                &tokenizer,
                &device,
                eos,
                &system_prompt,
                &history,
                generation,
                &tx,
            );
            let _ = tx.blocking_send(TokenChunk {
                text: String::new(),
                is_final: true,
            });
        });

        rx
    }
}

struct LoadedModel {
    model: ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
    eos_token: u32,
}

fn load_weights(model_path: &Path, tokenizer_path: &Path) -> Result<LoadedModel, LlmError> {
    info!(
        model = %model_path.display(),
        tokenizer = %tokenizer_path.display(),
        "loading gguf + tokenizer",
    );

    let device = Device::new_metal(0).map_err(|e| LlmError::Load(format!("metal device: {e}")))?;

    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| LlmError::Load(format!("tokenizer: {e}")))?;

    let eos_token = tokenizer
        .get_vocab(true)
        .get(EOS_TOKEN)
        .copied()
        .ok_or_else(|| LlmError::Load(format!("eos token `{EOS_TOKEN}` not in vocab")))?;

    let mut file =
        std::fs::File::open(model_path).map_err(|e| LlmError::Load(format!("open gguf: {e}")))?;
    let content = gguf_file::Content::read(&mut file)
        .map_err(|e| LlmError::Load(format!("read gguf: {e}")))?;
    let model = ModelWeights::from_gguf(content, &mut file, &device)
        .map_err(|e| LlmError::Load(format!("build model: {e}")))?;

    info!("model ready");
    Ok(LoadedModel {
        model,
        tokenizer,
        device,
        eos_token,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "explicit args keep the blocking-task call site obvious"
)]
#[expect(
    clippy::too_many_lines,
    reason = "the inference loop is genuinely linear; splitting would obscure control flow"
)]
fn run_completion(
    model: &Mutex<ModelWeights>,
    tokenizer: &Tokenizer,
    device: &Device,
    eos_token: u32,
    system_prompt: &str,
    history: &[Message],
    params: GenerationConfig,
    tx: &mpsc::Sender<TokenChunk>,
) {
    // LFM2 / Luth chat template (from the model's chat_template.jinja):
    // a leading BOS, then ChatML turns, then the assistant open. No
    // thinking tags — LFM2/Luth is a non-reasoning model. The full string
    // (BOS + specials included) is tokenized with add_special_tokens=false,
    // exactly as a chat template is meant to be applied.
    let mut prompt = format!("<|startoftext|><|im_start|>system\n{system_prompt}<|im_end|>\n");
    for msg in history {
        prompt.push_str("<|im_start|>");
        prompt.push_str(msg.role.tag());
        prompt.push('\n');
        prompt.push_str(&msg.content);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");

    let encoded = match tokenizer.encode(prompt, false) {
        Ok(e) => e,
        Err(e) => {
            warn!(error = %e, "tokenize failed");
            return;
        }
    };
    let prompt_tokens: Vec<u32> = encoded.get_ids().to_vec();

    let mut logits_processor = LogitsProcessor::from_sampling(params.seed, build_sampling(&params));
    let mut tos = TokenOutputStream::new(tokenizer);

    let mut weights = model.lock();
    // No explicit cache reset: the first forward below is the full prompt
    // at index_pos = 0, which resets LFM2 state (attention KV ignored,
    // short-conv recomputed) between independent completions.

    let prompt_tensor =
        match Tensor::new(prompt_tokens.as_slice(), device).and_then(|t| t.unsqueeze(0)) {
            Ok(t) => t,
            Err(e) => {
                warn!(error = %e, "prompt tensor");
                return;
            }
        };
    let prompt_logits = match weights
        .forward(&prompt_tensor, 0)
        .and_then(|l| l.squeeze(0))
    {
        Ok(l) => l,
        Err(e) => {
            warn!(error = %e, "prompt forward");
            return;
        }
    };
    let mut next = match logits_processor.sample(&prompt_logits) {
        Ok(t) => t,
        Err(e) => {
            warn!(error = %e, "initial sample");
            return;
        }
    };

    let mut generated: Vec<u32> = Vec::with_capacity(params.max_tokens);
    generated.push(next);
    if !emit(&mut tos, next, tx) {
        return;
    }
    if next == eos_token {
        return;
    }

    let prompt_len = prompt_tokens.len();
    let budget = params.max_tokens.saturating_sub(1);

    for step in 0..budget {
        let input = match Tensor::new(&[next], device).and_then(|t| t.unsqueeze(0)) {
            Ok(t) => t,
            Err(e) => {
                warn!(error = %e, "decode tensor");
                return;
            }
        };
        let pos = prompt_len.saturating_add(step);
        let raw = match weights.forward(&input, pos).and_then(|l| l.squeeze(0)) {
            Ok(l) => l,
            Err(e) => {
                warn!(error = %e, "decode forward");
                return;
            }
        };

        let logits = if params.repeat_penalty > 1.0 {
            let start = generated.len().saturating_sub(params.repeat_last_n);
            match apply_repeat_penalty(&raw, params.repeat_penalty, &generated[start..]) {
                Ok(l) => l,
                Err(e) => {
                    warn!(error = %e, "repeat penalty");
                    return;
                }
            }
        } else {
            raw
        };

        next = match logits_processor.sample(&logits) {
            Ok(t) => t,
            Err(e) => {
                warn!(error = %e, "sample");
                return;
            }
        };
        generated.push(next);

        if !emit(&mut tos, next, tx) {
            return;
        }
        if next == eos_token {
            break;
        }
    }
    drop(weights);

    if let Ok(Some(rest)) = tos.decode_rest()
        && !rest.is_empty()
    {
        let _ = tx.blocking_send(TokenChunk {
            text: rest,
            is_final: false,
        });
    }
}

fn build_sampling(params: &GenerationConfig) -> Sampling {
    if params.temperature <= 0.0 {
        return Sampling::ArgMax;
    }
    match (params.top_k, params.top_p) {
        (None, None) => Sampling::All {
            temperature: params.temperature,
        },
        (Some(k), None) => Sampling::TopK {
            k,
            temperature: params.temperature,
        },
        (None, Some(p)) => Sampling::TopP {
            p,
            temperature: params.temperature,
        },
        (Some(k), Some(p)) => Sampling::TopKThenTopP {
            k,
            p,
            temperature: params.temperature,
        },
    }
}

/// Detokenise one sampled token and push the delta downstream. Returns
/// `false` if the receiver has been dropped.
fn emit(tos: &mut TokenOutputStream<'_>, token: u32, tx: &mpsc::Sender<TokenChunk>) -> bool {
    let text = match tos.next_token(token) {
        Ok(Some(text)) => text,
        Ok(None) => return true,
        Err(e) => {
            warn!(error = %e, "detokenize");
            return true;
        }
    };
    if text.is_empty() {
        return true;
    }
    tx.blocking_send(TokenChunk {
        text,
        is_final: false,
    })
    .is_ok()
}
