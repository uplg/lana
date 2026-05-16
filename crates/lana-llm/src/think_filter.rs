//! Streaming filter that strips `<think>...</think>` blocks from the model
//! output before they reach the consumer.
//!
//! Qwen3 emits its chain-of-thought inside `<think>...</think>` tags by
//! default. For a voice assistant this is noise — the TTS would read the
//! thinking aloud and the latency before the first user-visible token would
//! balloon. We let the model keep thinking internally and drop the wrapping
//! tags from the stream.
//!
//! The filter is buffered: a tag may be split across multiple input chunks
//! (e.g. `"<thi"` then `"nk>"`), so we always hold back the last few bytes
//! until they could no longer complete a tag.

/// Opening tag of a Qwen3 chain-of-thought block.
const OPEN: &str = "<think>";
/// Closing tag of a Qwen3 chain-of-thought block.
const CLOSE: &str = "</think>";
/// Number of trailing bytes withheld from output between chunks, sufficient
/// to recognise either tag once subsequent bytes arrive.
const KEEP_BACK: usize = CLOSE.len();

/// State machine that emits only the user-visible portion of a streamed
/// completion.
#[derive(Debug, Default)]
pub(crate) struct ThinkFilter {
    buffer: String,
    in_think: bool,
}

impl ThinkFilter {
    pub(crate) const fn new() -> Self {
        Self {
            buffer: String::new(),
            in_think: false,
        }
    }

    /// Feed one input chunk; return the text to emit downstream (may be
    /// empty, e.g. while inside a `<think>` block).
    pub(crate) fn push(&mut self, chunk: &str) -> String {
        self.buffer.push_str(chunk);
        self.drain(KEEP_BACK)
    }

    /// Flush at end of stream; emits anything still buffered outside of a
    /// think block, dropping the rest.
    pub(crate) fn flush(&mut self) -> String {
        self.drain(0)
    }

    fn drain(&mut self, keep_back: usize) -> String {
        let mut out = String::new();
        loop {
            if self.in_think {
                if let Some(idx) = self.buffer.find(CLOSE) {
                    let after = idx.saturating_add(CLOSE.len());
                    self.buffer.drain(..after);
                    self.in_think = false;
                    continue;
                }
                // No close tag yet: discard everything but the holdback so
                // we don't miss a tag split across chunks.
                let drop_until =
                    floor_char_boundary(&self.buffer, self.buffer.len().saturating_sub(keep_back));
                self.buffer.drain(..drop_until);
                break;
            }

            if let Some(idx) = self.buffer.find(OPEN) {
                if idx > 0 {
                    out.push_str(&self.buffer[..idx]);
                }
                let after = idx.saturating_add(OPEN.len());
                self.buffer.drain(..after);
                self.in_think = true;
                continue;
            }

            // No open tag in sight: emit everything except the holdback,
            // which might still start a tag with the next chunk.
            let emit_until =
                floor_char_boundary(&self.buffer, self.buffer.len().saturating_sub(keep_back));
            if emit_until > 0 {
                out.push_str(&self.buffer[..emit_until]);
                self.buffer.drain(..emit_until);
            }
            break;
        }
        out
    }
}

/// Largest byte index `<= n` that lies on a UTF-8 char boundary inside `s`.
fn floor_char_boundary(s: &str, n: usize) -> usize {
    let mut i = n.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i = i.saturating_sub(1);
    }
    i
}

#[cfg(test)]
mod tests {
    use super::ThinkFilter;

    fn run(chunks: &[&str]) -> String {
        let mut f = ThinkFilter::new();
        let mut out = String::new();
        for c in chunks {
            out.push_str(&f.push(c));
        }
        out.push_str(&f.flush());
        out
    }

    #[test]
    fn passes_through_without_tags() {
        assert_eq!(run(&["Hello", " ", "world"]), "Hello world");
    }

    #[test]
    fn strips_single_think_block() {
        assert_eq!(run(&["<think>internal reasoning</think>answer"]), "answer");
    }

    #[test]
    fn strips_think_across_chunks() {
        assert_eq!(
            run(&["<think>step ", "one ", "step two</think>", "real text"]),
            "real text"
        );
    }

    #[test]
    fn open_tag_split_across_chunks() {
        assert_eq!(run(&["<th", "ink>secret</think>", "answer"]), "answer");
    }

    #[test]
    fn close_tag_split_across_chunks() {
        assert_eq!(run(&["<think>secret</thi", "nk>answer"]), "answer");
    }

    #[test]
    fn preserves_text_around_block() {
        assert_eq!(
            run(&["prefix <think>noise</think> suffix"]),
            "prefix  suffix"
        );
    }

    #[test]
    fn handles_multibyte_holdback() {
        // The trailing accented chars span multiple bytes; the filter must
        // not slice them.
        assert_eq!(run(&["Réponse simple à toi"]), "Réponse simple à toi");
    }

    #[test]
    fn flush_inside_unterminated_think_is_dropped() {
        assert_eq!(run(&["<think>still thinking..."]), "");
    }
}
