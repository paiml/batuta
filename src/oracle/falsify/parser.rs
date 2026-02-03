//! Specification Parser
//!
//! Parses markdown specifications into structured requirements.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Parsed specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedSpec {
    /// Specification name
    pub name: String,
    /// Module/crate name
    pub module: String,
    /// Extracted requirements
    pub requirements: Vec<ParsedRequirement>,
    /// Types mentioned
    pub types: Vec<String>,
    /// Functions mentioned
    pub functions: Vec<String>,
    /// Numerical tolerances if specified
    pub tolerances: Option<ToleranceSpec>,
}

/// A parsed requirement from the spec
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedRequirement {
    /// Requirement ID (from spec or generated)
    pub id: String,
    /// Description
    pub description: String,
    /// Category hint (boundary, invariant, numerical, etc.)
    pub category_hint: Option<String>,
    /// Input type hint
    pub input_type: Option<String>,
    /// Output type hint
    pub output_type: Option<String>,
    /// Is this a critical requirement?
    pub critical: bool,
}

/// Numerical tolerance specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToleranceSpec {
    /// Absolute tolerance
    pub atol: Option<f64>,
    /// Relative tolerance
    pub rtol: Option<f64>,
}

/// Specification parser
#[derive(Debug)]
pub struct SpecParser {
    // Configuration options could go here
}

impl SpecParser {
    /// Create a new parser
    pub fn new() -> Self {
        Self {}
    }

    /// Parse a specification file
    pub fn parse_file(&self, path: &Path) -> anyhow::Result<ParsedSpec> {
        let content = std::fs::read_to_string(path)?;
        self.parse(&content, path)
    }

    /// Parse specification content
    pub fn parse(&self, content: &str, source: &Path) -> anyhow::Result<ParsedSpec> {
        let name = source
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unnamed".to_string());

        let module = self.extract_module(content).unwrap_or_else(|| name.clone());
        let requirements = self.extract_requirements(content);
        let types = self.extract_types(content);
        let functions = self.extract_functions(content);
        let tolerances = self.extract_tolerances(content);

        Ok(ParsedSpec {
            name,
            module,
            requirements,
            types,
            functions,
            tolerances,
        })
    }

    /// Extract module name from spec
    fn extract_module(&self, content: &str) -> Option<String> {
        // Look for patterns like "module: foo" or "crate: foo"
        for line in content.lines() {
            let lower = line.to_lowercase();
            if lower.starts_with("module:") || lower.starts_with("crate:") {
                return line.split(':').nth(1).map(|s| s.trim().to_string());
            }
        }

        // Try to extract from code blocks
        for line in content.lines() {
            if line.contains("use ") && line.contains("::") {
                if let Some(start) = line.find("use ") {
                    let rest = &line[start + 4..];
                    if let Some(end) = rest.find("::") {
                        return Some(rest[..end].trim().to_string());
                    }
                }
            }
        }

        None
    }

    /// Extract requirements from spec
    fn extract_requirements(&self, content: &str) -> Vec<ParsedRequirement> {
        let mut requirements = Vec::new();
        let mut current_section = String::new();
        let mut req_counter = 0;

        for line in content.lines() {
            // Track sections (headers)
            if line.starts_with('#') {
                current_section = line.trim_start_matches('#').trim().to_lowercase();
                continue;
            }

            // Look for requirement patterns:
            // - Bullet points that look like requirements
            // - "MUST", "SHALL", "SHOULD" keywords
            // - Numbered items
            if let Some(req) = self.parse_requirement_line(line, &current_section, &mut req_counter)
            {
                requirements.push(req);
            }
        }

        requirements
    }

    /// Parse a potential requirement line
    fn parse_requirement_line(
        &self,
        line: &str,
        section: &str,
        counter: &mut u32,
    ) -> Option<ParsedRequirement> {
        let trimmed = line.trim();

        // Skip empty lines and non-requirement lines
        if trimmed.is_empty() || trimmed.starts_with("```") {
            return None;
        }

        // Check for requirement keywords
        let upper = trimmed.to_uppercase();
        let is_requirement = upper.contains("MUST")
            || upper.contains("SHALL")
            || upper.contains("SHOULD")
            || upper.contains("REQUIRE")
            || (trimmed.starts_with("- ") && trimmed.len() > 10);

        if !is_requirement {
            return None;
        }

        *counter += 1;

        // Determine category hint from section name
        let category_hint = self.infer_category(section, trimmed);

        // Check if critical
        let critical = upper.contains("CRITICAL")
            || upper.contains("MUST NOT")
            || upper.contains("SHALL NOT");

        Some(ParsedRequirement {
            id: format!("REQ-{:03}", counter),
            description: trimmed
                .trim_start_matches('-')
                .trim_start_matches('*')
                .trim()
                .to_string(),
            category_hint,
            input_type: self.extract_type_hint(trimmed, "input"),
            output_type: self.extract_type_hint(trimmed, "output"),
            critical,
        })
    }

    /// Infer category from section name and content
    fn infer_category(&self, section: &str, content: &str) -> Option<String> {
        let lower_section = section.to_lowercase();
        let lower_content = content.to_lowercase();

        if lower_section.contains("boundary")
            || lower_content.contains("empty")
            || lower_content.contains("null")
            || lower_content.contains("limit")
        {
            return Some("boundary".to_string());
        }

        if lower_section.contains("invariant")
            || lower_content.contains("idempotent")
            || lower_content.contains("commutative")
        {
            return Some("invariant".to_string());
        }

        if lower_section.contains("numeric")
            || lower_content.contains("precision")
            || lower_content.contains("floating")
        {
            return Some("numerical".to_string());
        }

        if lower_section.contains("concurren")
            || lower_content.contains("thread")
            || lower_content.contains("race")
        {
            return Some("concurrency".to_string());
        }

        if lower_section.contains("resource")
            || lower_content.contains("memory")
            || lower_content.contains("exhaust")
        {
            return Some("resource".to_string());
        }

        if lower_section.contains("parity")
            || lower_content.contains("reference")
            || lower_content.contains("match")
        {
            return Some("parity".to_string());
        }

        None
    }

    /// Extract type hint from content
    fn extract_type_hint(&self, content: &str, kind: &str) -> Option<String> {
        // Look for patterns like "input: Vec<f64>" or "returns f64"
        let lower = content.to_lowercase();

        if kind == "input" {
            if lower.contains("vec") || lower.contains("array") || lower.contains("list") {
                return Some("Vec<T>".to_string());
            }
            if lower.contains("string") || lower.contains("str") {
                return Some("String".to_string());
            }
        }

        if kind == "output" {
            if lower.contains("returns") {
                if lower.contains("bool") {
                    return Some("bool".to_string());
                }
                if lower.contains("result") || lower.contains("error") {
                    return Some("Result<T, E>".to_string());
                }
            }
        }

        None
    }

    /// Extract types mentioned in spec
    fn extract_types(&self, content: &str) -> Vec<String> {
        let mut types = Vec::new();

        // Look for struct/enum definitions
        for line in content.lines() {
            if line.contains("struct ") {
                if let Some(name) = line.split("struct ").nth(1) {
                    if let Some(name) = name.split(|c: char| !c.is_alphanumeric() && c != '_').next()
                    {
                        types.push(name.to_string());
                    }
                }
            }
            if line.contains("enum ") {
                if let Some(name) = line.split("enum ").nth(1) {
                    if let Some(name) = name.split(|c: char| !c.is_alphanumeric() && c != '_').next()
                    {
                        types.push(name.to_string());
                    }
                }
            }
        }

        types.sort();
        types.dedup();
        types
    }

    /// Extract functions mentioned in spec
    fn extract_functions(&self, content: &str) -> Vec<String> {
        let mut functions = Vec::new();

        for line in content.lines() {
            // Look for fn definitions
            if line.contains("fn ") {
                if let Some(name) = line.split("fn ").nth(1) {
                    if let Some(name) = name.split('(').next() {
                        functions.push(name.trim().to_string());
                    }
                }
            }
            // Look for method calls mentioned
            if line.contains("`.") {
                // Markdown code inline
                let parts: Vec<&str> = line.split('`').collect();
                for (i, part) in parts.iter().enumerate() {
                    if i % 2 == 1 && part.contains('.') {
                        if let Some(method) = part.split('.').last() {
                            if let Some(name) = method.split('(').next() {
                                functions.push(name.to_string());
                            }
                        }
                    }
                }
            }
        }

        functions.sort();
        functions.dedup();
        functions
    }

    /// Extract tolerance specifications
    fn extract_tolerances(&self, content: &str) -> Option<ToleranceSpec> {
        let mut atol = None;
        let mut rtol = None;

        for line in content.lines() {
            let lower = line.to_lowercase();

            // Look for tolerance specifications
            if lower.contains("atol") || lower.contains("absolute") {
                if let Some(val) = self.extract_number(line) {
                    atol = Some(val);
                }
            }
            if lower.contains("rtol") || lower.contains("relative") {
                if let Some(val) = self.extract_number(line) {
                    rtol = Some(val);
                }
            }
            if lower.contains("tolerance") && lower.contains("1e-") {
                if let Some(val) = self.extract_number(line) {
                    if atol.is_none() {
                        atol = Some(val);
                    }
                }
            }
        }

        if atol.is_some() || rtol.is_some() {
            Some(ToleranceSpec { atol, rtol })
        } else {
            None
        }
    }

    /// Extract a number from a line
    fn extract_number(&self, line: &str) -> Option<f64> {
        // Use regex-like pattern matching for scientific notation
        let mut best_num: Option<f64> = None;

        // Look for patterns like 1e-5, 0.001, 1.5e-10
        let chars: Vec<char> = line.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            if chars[i].is_ascii_digit() {
                // Found start of potential number
                let mut num_str = String::new();
                while i < chars.len() {
                    let c = chars[i];
                    if c.is_ascii_digit() || c == '.' {
                        num_str.push(c);
                        i += 1;
                    } else if (c == 'e' || c == 'E') && i + 1 < chars.len() {
                        num_str.push(c);
                        i += 1;
                        // Handle optional sign after e
                        if i < chars.len() && (chars[i] == '-' || chars[i] == '+') {
                            num_str.push(chars[i]);
                            i += 1;
                        }
                    } else {
                        break;
                    }
                }

                if let Ok(val) = num_str.parse::<f64>() {
                    best_num = Some(val);
                    break; // Return first valid number
                }
            } else {
                i += 1;
            }
        }

        best_num
    }
}

impl Default for SpecParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let _parser = SpecParser::new();
        assert!(true); // Parser created successfully
    }

    #[test]
    fn test_parse_simple_spec() {
        let parser = SpecParser::new();
        let content = r#"
# Test Spec

module: test_module

## Requirements

- MUST handle empty input
- SHOULD return error on invalid input
- The function MUST NOT panic
"#;
        let spec = parser
            .parse(content, Path::new("test-spec.md"))
            .unwrap();
        assert_eq!(spec.name, "test-spec");
        assert_eq!(spec.module, "test_module");
        assert!(!spec.requirements.is_empty());
    }

    #[test]
    fn test_extract_tolerances() {
        let parser = SpecParser::new();
        // Use format that matches the parser's expectations
        let content = r#"
Use a tolerance of 1e-5 for comparisons
"#;
        let tolerances = parser.extract_tolerances(content);
        assert!(tolerances.is_some(), "Should extract tolerance from '1e-5'");
        let tol = tolerances.unwrap();
        assert!(tol.atol.is_some(), "atol should be extracted");
        assert!((tol.atol.unwrap() - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_infer_category() {
        let parser = SpecParser::new();

        assert_eq!(
            parser.infer_category("boundary conditions", "handle empty input"),
            Some("boundary".to_string())
        );

        assert_eq!(
            parser.infer_category("numerical", "floating point precision"),
            Some("numerical".to_string())
        );

        assert_eq!(
            parser.infer_category("other", "thread safety"),
            Some("concurrency".to_string())
        );
    }
}
