//! [`LlmEngine`] — streaming chat over candle + quantized Qwen3 GGUF.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_qwen3::ModelWeights;
use candle_transformers::utils::apply_repeat_penalty;
use parking_lot::Mutex;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::error::LlmError;
use crate::think_filter::ThinkFilter;
use crate::token_stream::TokenOutputStream;

const TOKEN_CHANNEL_CAPACITY: usize = 256;
const QWEN3_EOS: &str = "<|im_end|>";

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
            weights.clear_kv_cache();
            let input = Tensor::new(&[0_u32, 1_u32, 2_u32], &device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| LlmError::Load(format!("warmup tensor: {e}")))?;
            let _ = weights
                .forward(&input, 0)
                .map_err(|e| LlmError::Load(format!("warmup forward: {e}")))?;
            weights.clear_kv_cache();
            drop(weights);
            Ok(())
        })
        .await
        .map_err(|e| LlmError::Load(format!("warmup join: {e}")))??;
        info!(elapsed_s = started.elapsed().as_secs_f32(), "warmup done");
        Ok(())
    }

    /// Send a user prompt and receive a stream of [`TokenChunk`]s.
    ///
    /// A terminal chunk with `is_final == true` is always sent before the
    /// channel closes (even on mid-stream error). Each call clears the KV
    /// cache and runs a single completion.
    #[must_use]
    pub fn stream(&self, user_prompt: &str) -> mpsc::Receiver<TokenChunk> {
        let (tx, rx) = mpsc::channel(TOKEN_CHANNEL_CAPACITY);
        let model = Arc::clone(&self.model);
        let tokenizer = Arc::clone(&self.tokenizer);
        let device = self.device.clone();
        let eos = self.eos_token;
        let system_prompt = Arc::clone(&self.system_prompt);
        let user_prompt: Arc<str> = Arc::from(user_prompt);
        let generation = self.generation;

        tokio::task::spawn_blocking(move || {
            run_completion(
                &model,
                &tokenizer,
                &device,
                eos,
                &system_prompt,
                &user_prompt,
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
        .get(QWEN3_EOS)
        .copied()
        .ok_or_else(|| LlmError::Load(format!("eos token `{QWEN3_EOS}` not in vocab")))?;

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
    user_prompt: &str,
    params: GenerationConfig,
    tx: &mpsc::Sender<TokenChunk>,
) {
    let prompt = format!(
        "<|im_start|>system\n{system_prompt}<|im_end|>\n\
         <|im_start|>user\n{user_prompt}<|im_end|>\n\
         <|im_start|>assistant\n",
    );

    let encoded = match tokenizer.encode(prompt, true) {
        Ok(e) => e,
        Err(e) => {
            warn!(error = %e, "tokenize failed");
            return;
        }
    };
    let prompt_tokens: Vec<u32> = encoded.get_ids().to_vec();

    let mut logits_processor = LogitsProcessor::from_sampling(params.seed, build_sampling(&params));
    let mut tos = TokenOutputStream::new(tokenizer);
    let mut filter = ThinkFilter::new();

    let mut weights = model.lock();
    weights.clear_kv_cache();

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
    if !emit(&mut tos, &mut filter, next, tx) {
        return;
    }
    if next == eos_token {
        flush_filter(&mut filter, tx);
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

        if !emit(&mut tos, &mut filter, next, tx) {
            return;
        }
        if next == eos_token {
            break;
        }
    }
    drop(weights);

    if let Ok(Some(rest)) = tos.decode_rest() {
        let text = filter.push(&rest);
        if !text.is_empty() {
            let _ = tx.blocking_send(TokenChunk {
                text,
                is_final: false,
            });
        }
    }
    flush_filter(&mut filter, tx);
}

/// Flush whatever the think filter still holds and forward it downstream.
fn flush_filter(filter: &mut ThinkFilter, tx: &mpsc::Sender<TokenChunk>) {
    let tail = filter.flush();
    if !tail.is_empty() {
        let _ = tx.blocking_send(TokenChunk {
            text: tail,
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

/// Detokenise one sampled token, run it through the think filter, and push
/// the user-visible delta if any. Returns `false` if the downstream receiver
/// has been dropped.
fn emit(
    tos: &mut TokenOutputStream<'_>,
    filter: &mut ThinkFilter,
    token: u32,
    tx: &mpsc::Sender<TokenChunk>,
) -> bool {
    let raw = match tos.next_token(token) {
        Ok(Some(text)) => text,
        Ok(None) => return true,
        Err(e) => {
            warn!(error = %e, "detokenize");
            return true;
        }
    };
    let visible = filter.push(&raw);
    if visible.is_empty() {
        return true;
    }
    tx.blocking_send(TokenChunk {
        text: visible,
        is_final: false,
    })
    .is_ok()
}
