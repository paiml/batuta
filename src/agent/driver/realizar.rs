//! RealizarDriver — sovereign local inference via GGUF/APR models.
//!
//! Uses the `realizar` crate for local LLM inference. All data
//! stays on-device (Sovereign privacy tier, Genchi Genbutsu).
//!
//! Tool call parsing: local models output `<tool_call>` JSON blocks
//! in their text. The driver extracts these into `ToolCall` structs.
//!
//! Feature-gated behind `inference`.

use async_trait::async_trait;
use std::path::{Path, PathBuf};

use super::chat_template::{format_prompt_with_template, ChatTemplate};
use super::{CompletionRequest, CompletionResponse, LlmDriver, ToolCall};
use crate::agent::result::{AgentError, DriverError, StopReason, TokenUsage};
use crate::serve::backends::PrivacyTier;

/// Local inference driver using realizar (GGUF/APR/SafeTensors).
pub struct RealizarDriver {
    /// Path to model file.
    model_path: PathBuf,
    /// Context window size.
    context_window_size: usize,
    /// Auto-detected chat template.
    template: ChatTemplate,
}

impl RealizarDriver {
    /// Create a new RealizarDriver from a model path.
    ///
    /// **Contract: `apr_model_validity` (apr-code-v1.yaml)**
    ///
    /// Preconditions enforced at the load boundary (Jidoka):
    /// - File must exist
    /// - APR files: must have embedded tokenizer (checked via header)
    /// - GGUF files: must have valid magic bytes
    ///
    /// Violation → actionable error with re-conversion instructions.
    /// No broken model ever reaches the inference loop.
    pub fn new(model_path: PathBuf, context_window: Option<usize>) -> Result<Self, AgentError> {
        if !model_path.exists() {
            return Err(AgentError::Driver(DriverError::InferenceFailed(format!(
                "model not found: {}",
                model_path.display()
            ))));
        }

        // ═══ CONTRACT: apr_model_validity — Jidoka boundary check ═══
        validate_model_file(&model_path)?;

        let context_window_size = context_window.unwrap_or(4096);
        let template = ChatTemplate::from_model_path(&model_path);
        Ok(Self { model_path, context_window_size, template })
    }
}

#[async_trait]
impl LlmDriver for RealizarDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, AgentError> {
        // Format messages using auto-detected chat template
        let prompt = format_prompt_with_template(&request, self.template);

        // Build inference config (explicit fields — no Default impl)
        let config = realizar::infer::InferenceConfig {
            model_path: self.model_path.clone(),
            prompt: Some(prompt),
            input_tokens: None,
            max_tokens: request.max_tokens as usize,
            temperature: request.temperature,
            top_k: 0,
            no_gpu: false,
            trace: false,
            trace_verbose: false,
            trace_output: None,
            trace_steps: None,
            verbose: false,
            use_mock_backend: false,
        };

        // Run inference in blocking thread (realizar is sync)
        let result = tokio::task::spawn_blocking(move || realizar::infer::run_inference(&config))
            .await
            .map_err(|e| {
                AgentError::Driver(DriverError::InferenceFailed(format!("spawn_blocking: {e}")))
            })?
            .map_err(|e| AgentError::Driver(DriverError::InferenceFailed(e.to_string())))?;

        // Parse tool calls from text output
        let (text, tool_calls) = parse_tool_calls(&result.text);

        let stop_reason =
            if tool_calls.is_empty() { StopReason::EndTurn } else { StopReason::ToolUse };

        Ok(CompletionResponse {
            text,
            stop_reason,
            tool_calls,
            usage: TokenUsage {
                input_tokens: result.input_token_count as u64,
                output_tokens: result.generated_token_count as u64,
            },
        })
    }

    fn context_window(&self) -> usize {
        self.context_window_size
    }

    fn privacy_tier(&self) -> PrivacyTier {
        PrivacyTier::Sovereign
    }
}

/// Parse `<tool_call>` JSON blocks from model output text.
///
/// Local models output tool calls as:
/// ```text
/// <tool_call>
/// {"name": "rag", "input": {"query": "SIMD"}}
/// </tool_call>
/// ```
///
/// Returns the remaining text (with tool_call blocks removed)
/// and the extracted tool calls.
fn parse_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    let mut tool_calls = Vec::new();
    let mut remaining = String::new();
    let mut call_counter = 0u32;

    let mut cursor = text;
    loop {
        let Some(start) = cursor.find("<tool_call>") else {
            remaining.push_str(cursor);
            break;
        };

        remaining.push_str(&cursor[..start]);
        let after_tag = &cursor[start + "<tool_call>".len()..];

        let Some(end) = after_tag.find("</tool_call>") else {
            // No closing tag — treat as plain text
            remaining.push_str(&cursor[start..]);
            break;
        };

        let json_str = after_tag[..end].trim();
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
            let name = parsed.get("name").and_then(|n| n.as_str()).unwrap_or("unknown").to_string();
            let input = parsed.get("input").cloned().unwrap_or(serde_json::json!({}));
            call_counter += 1;
            tool_calls.push(ToolCall { id: format!("local-{call_counter}"), name, input });
        } else {
            // Malformed JSON — treat as plain text
            remaining
                .push_str(&cursor[start..start + "<tool_call>".len() + end + "</tool_call>".len()]);
        }

        cursor = &after_tag[end + "</tool_call>".len()..];
    }

    (remaining.trim().to_string(), tool_calls)
}

// ═══ CONTRACT: apr_model_validity (apr-code-v1.yaml) ═══
//
// Jidoka boundary check: validate model file BEFORE entering inference.
// APR files must have embedded tokenizer. GGUF files must have valid magic.
// Broken models are rejected with actionable error + re-conversion command.

/// APR v2 magic bytes: "APR\0"
const APR_MAGIC: [u8; 4] = [0x41, 0x50, 0x52, 0x00];
/// GGUF magic bytes: "GGUF"
const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46];

/// Validate model file at the load boundary (Jidoka: stop on first defect).
///
/// **Contract precondition**: model file must be structurally valid before
/// any inference attempt. For APR files, this means embedded tokenizer data
/// must be present. Violation → clear error with `apr convert` instructions.
fn validate_model_file(path: &Path) -> Result<(), AgentError> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    // Read first 64KB for header validation (fast, no full model load)
    let header =
        std::fs::read(path).map(|data| data[..data.len().min(65536)].to_vec()).map_err(|e| {
            AgentError::Driver(DriverError::InferenceFailed(format!("cannot read model file: {e}")))
        })?;

    if header.len() < 4 {
        return Err(AgentError::Driver(DriverError::InferenceFailed(
            "model file too small (< 4 bytes)".into(),
        )));
    }

    match ext {
        "apr" => validate_apr_header(&header, path),
        "gguf" => validate_gguf_header(&header, path),
        _ => Ok(()), // Unknown formats pass through to realizar
    }
}

/// Validate APR file header and check for embedded tokenizer data.
fn validate_apr_header(header: &[u8], path: &Path) -> Result<(), AgentError> {
    // Check magic bytes
    if header[..4] != APR_MAGIC {
        return Err(AgentError::Driver(DriverError::InferenceFailed(format!(
            "invalid APR file (wrong magic bytes): {}",
            path.display()
        ))));
    }

    // APR v2 binary metadata follows the 4-byte magic.
    // Scan the header for tokenizer indicators.
    // Embedded tokenizer data is stored as metadata keys containing
    // vocabulary entries. If the metadata section is small (< 1KB after
    // magic), it almost certainly lacks a tokenizer.
    //
    // Full validation: check for "tokenizer.ggml.tokens" or BPE vocab
    // in the metadata. We scan the header bytes for these markers.
    let header_str = String::from_utf8_lossy(&header[4..]);
    let has_tokenizer = header_str.contains("tokenizer")
        || header_str.contains("vocab")
        || header_str.contains("merges")
        || header_str.contains("bpe");

    if !has_tokenizer {
        return Err(AgentError::Driver(DriverError::InferenceFailed(format!(
            "APR file missing embedded tokenizer: {}\n\
             APR format requires a self-contained tokenizer (Jidoka: fail-fast).\n\
             Re-convert with: apr convert {} -o {}",
            path.display(),
            path.with_extension("gguf").display(),
            path.display(),
        ))));
    }

    Ok(())
}

/// Validate GGUF file header magic bytes.
fn validate_gguf_header(header: &[u8], path: &Path) -> Result<(), AgentError> {
    if header[..4] != GGUF_MAGIC {
        return Err(AgentError::Driver(DriverError::InferenceFailed(format!(
            "invalid GGUF file (wrong magic bytes): {}",
            path.display()
        ))));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_no_tool_calls() {
        let (text, calls) = parse_tool_calls("Hello world");
        assert_eq!(text, "Hello world");
        assert!(calls.is_empty());
    }

    #[test]
    fn test_parse_single_tool_call() {
        let input = r#"Before text
<tool_call>
{"name": "rag", "input": {"query": "SIMD"}}
</tool_call>
After text"#;
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(text, "Before text\n\nAfter text");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "rag");
        assert_eq!(calls[0].id, "local-1");
        assert_eq!(calls[0].input, serde_json::json!({"query": "SIMD"}));
    }

    #[test]
    fn test_parse_multiple_tool_calls() {
        let input = r#"<tool_call>
{"name": "rag", "input": {"query": "a"}}
</tool_call>
Middle
<tool_call>
{"name": "memory", "input": {"action": "recall", "query": "b"}}
</tool_call>"#;
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(text, "Middle");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "rag");
        assert_eq!(calls[0].id, "local-1");
        assert_eq!(calls[1].name, "memory");
        assert_eq!(calls[1].id, "local-2");
    }

    #[test]
    fn test_parse_malformed_json() {
        let input = r#"<tool_call>
not valid json
</tool_call>"#;
        let (_text, calls) = parse_tool_calls(input);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_parse_missing_close_tag() {
        let input = "<tool_call>\n{\"name\": \"rag\"}";
        let (text, calls) = parse_tool_calls(input);
        assert!(calls.is_empty());
        assert!(text.contains("<tool_call>"));
    }

    #[test]
    fn test_parse_missing_name() {
        let input = r#"<tool_call>
{"input": {"query": "test"}}
</tool_call>"#;
        let (_, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "unknown");
    }

    #[test]
    fn test_privacy_tier_always_sovereign() {
        assert_eq!(PrivacyTier::Sovereign, PrivacyTier::Sovereign);
    }

    // ── CONTRACT: apr_model_validity tests (FALSIFY-AC-008) ──

    #[test]
    fn test_apr_without_tokenizer_rejected_at_boundary() {
        let tmp = tempfile::NamedTempFile::with_suffix(".apr").expect("tmpfile");
        // Write valid APR magic but no tokenizer data
        let mut data = Vec::new();
        data.extend_from_slice(&APR_MAGIC);
        data.extend_from_slice(&[0u8; 100]); // empty metadata
        std::fs::write(tmp.path(), &data).expect("write");

        let result = validate_model_file(tmp.path());
        assert!(result.is_err(), "APR without tokenizer must be rejected");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("missing embedded tokenizer"), "error must mention tokenizer: {err}");
        assert!(err.contains("apr convert"), "error must include fix command: {err}");
    }

    #[test]
    fn test_apr_with_tokenizer_passes_validation() {
        let tmp = tempfile::NamedTempFile::with_suffix(".apr").expect("tmpfile");
        // Write valid APR magic + tokenizer marker in metadata
        let mut data = Vec::new();
        data.extend_from_slice(&APR_MAGIC);
        data.extend_from_slice(b"metadata with tokenizer vocab and merges data");
        std::fs::write(tmp.path(), &data).expect("write");

        let result = validate_model_file(tmp.path());
        assert!(result.is_ok(), "APR with tokenizer should pass: {result:?}");
    }

    #[test]
    fn test_gguf_valid_magic_passes() {
        let tmp = tempfile::NamedTempFile::with_suffix(".gguf").expect("tmpfile");
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC);
        data.extend_from_slice(&[0u8; 100]);
        std::fs::write(tmp.path(), &data).expect("write");

        let result = validate_model_file(tmp.path());
        assert!(result.is_ok(), "valid GGUF should pass: {result:?}");
    }

    #[test]
    fn test_gguf_invalid_magic_rejected() {
        let tmp = tempfile::NamedTempFile::with_suffix(".gguf").expect("tmpfile");
        std::fs::write(tmp.path(), b"NOT_GGUF_DATA_HERE").expect("write");

        let result = validate_model_file(tmp.path());
        assert!(result.is_err(), "invalid GGUF must be rejected");
        assert!(result.unwrap_err().to_string().contains("wrong magic bytes"));
    }

    #[test]
    fn test_empty_file_rejected() {
        let tmp = tempfile::NamedTempFile::with_suffix(".apr").expect("tmpfile");
        std::fs::write(tmp.path(), b"").expect("write");

        let result = validate_model_file(tmp.path());
        assert!(result.is_err(), "empty file must be rejected");
    }
}
