//! Streaming detokeniser.
//!
//! BPE tokenisers cannot decode each token in isolation: many tokens are
//! sub-bytes of a UTF-8 character, others adjust spacing relative to the
//! previous token. This wrapper batches enough state to emit user-visible
//! text only when a full decodable suffix is available.
//!
//! Adapted from `candle-examples/src/token_output_stream.rs`. We expose
//! `Result<_, String>` instead of `candle::Result` to keep the pub(crate)lic API
//! free of candle types.

use tokenizers::Tokenizer;

/// Incremental detokeniser keeping enough history to emit clean UTF-8 chunks.
#[derive(Debug)]
pub(crate) struct TokenOutputStream<'a> {
    tokenizer: &'a Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl<'a> TokenOutputStream<'a> {
    pub(crate) const fn new(tokenizer: &'a Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    fn decode(&self, ids: &[u32]) -> Result<String, String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| format!("decode: {e}"))
    }

    /// Push one freshly sampled token and return the user-visible delta if a
    /// clean alphanumeric boundary has just been crossed. Otherwise the token
    /// is buffered for the next call.
    pub(crate) fn next_token(&mut self, token: u32) -> Result<Option<String>, String> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            self.decode(&self.tokens[self.prev_index..self.current_index])?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().is_some_and(char::is_alphanumeric) {
            let (_, suffix) = text.split_at(prev_text.len());
            let out = suffix.to_owned();
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(out))
        } else {
            Ok(None)
        }
    }

    /// Flush any remaining buffered tokens at end-of-stream.
    pub(crate) fn decode_rest(&self) -> Result<Option<String>, String> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            self.decode(&self.tokens[self.prev_index..self.current_index])?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let (_, suffix) = text.split_at(prev_text.len());
            Ok(Some(suffix.to_owned()))
        } else {
            Ok(None)
        }
    }
}
