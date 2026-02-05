//! Pattern detection utilities for bug-hunter.
//!
//! This module contains functions for detecting code patterns and determining
//! whether they represent real technical debt or false positives.
//!
//! # Safety
//!
//! This module contains the string literals "unsafe {" and "transmute" as
//! pattern matchers for detecting unsafe code in scanned files. These are
//! string constants used for pattern matching, not actual unsafe code.

use super::types::Finding;
use std::collections::HashSet;

/// Check if a finding should be suppressed (BH-15).
/// Wired into analyze_common_patterns per issue #17.
pub fn should_suppress_finding(finding: &Finding, line_content: &str) -> bool {
    // Issue #17: Suppress identical-blocks warnings for mapper functions
    if finding.title.contains("identical blocks") || finding.title.contains("if_same_then_else") {
        // Check if this looks like a mapper function (returns enum variants)
        if line_content.contains("=>") || line_content.contains("PartitionSpec::") {
            return true;
        }
        // Check for intentional comment
        if line_content.contains("INTENTIONAL") || line_content.contains("intentional") {
            return true;
        }
    }

    // Suppress warnings about code that detects patterns (meta-level)
    if (line_content.contains("PATTERN_MARKERS") || line_content.contains("pattern"))
        && (finding.title.contains("FIXME") || finding.title.contains("TODO"))
    {
        return true;
    }

    false
}

/// Determine which lines are inside test code (after #[cfg(test)] or #[test]).
pub fn compute_test_lines(content: &str) -> HashSet<usize> {
    let mut test_lines = HashSet::new();
    let mut in_test_module = false;
    let mut test_module_start_depth: i32 = 0;
    let mut brace_depth: i32 = 0;
    let mut waiting_for_brace = false;

    for (line_num, line) in content.lines().enumerate() {
        let line_num = line_num + 1;
        let trimmed = line.trim();

        // Track brace depth changes on this line
        let open_braces = line.matches('{').count() as i32;
        let close_braces = line.matches('}').count() as i32;

        // Check for test module entry: #[cfg(test)]
        if trimmed == "#[cfg(test)]" {
            waiting_for_brace = true;
            test_lines.insert(line_num); // The attribute itself is test code
        }

        // Check for individual test function: #[test]
        if trimmed == "#[test]" || trimmed.starts_with("#[test]") {
            waiting_for_brace = true;
            test_lines.insert(line_num); // The attribute itself is test code
        }

        // If we're waiting for the opening brace of a test block
        if waiting_for_brace && open_braces > 0 {
            in_test_module = true;
            test_module_start_depth = brace_depth; // Remember depth BEFORE this line's braces
            waiting_for_brace = false;
        }

        // Update brace depth
        brace_depth += open_braces - close_braces;

        // Mark lines inside test modules
        if in_test_module {
            test_lines.insert(line_num);
            // Check if we've exited the test module (brace depth returned to start level)
            if brace_depth <= test_module_start_depth {
                in_test_module = false;
            }
        }
    }

    test_lines
}

/// Check if pattern appears in a "real" code context, not inside a string literal.
pub fn is_real_pattern(line: &str, pattern: &str) -> bool {
    // Find the pattern position
    let Some(pos) = line.find(pattern) else {
        return false;
    };

    let trimmed = line.trim();
    let before = &line[..pos];

    // For tech debt markers (TODO/FIXME/HACK/XXX), check if this is a real marker
    let is_tech_debt = matches!(pattern, "TODO" | "FIXME" | "HACK" | "XXX");
    if is_tech_debt {
        // Exclude doc comments (/// or //!) - these usually describe code, not mark tech debt
        let is_doc_comment = trimmed.starts_with("///") || trimmed.starts_with("//!");
        if is_doc_comment {
            return false;
        }

        // Exclude lines that explain pattern matching (contain multiple pattern names)
        // e.g., "// For TODO/FIXME/HACK/XXX, they should be preceded..."
        let pattern_count = ["TODO", "FIXME", "HACK", "XXX"]
            .iter()
            .filter(|p| line.contains(*p))
            .count();
        if pattern_count >= 2 {
            return false;
        }

        // Real tech debt is in regular comments: must have // or /* before the pattern
        let has_comment = before.contains("//") || before.contains("/*");

        // Exclude patterns inside strings (basic heuristic: count quotes before position)
        let quotes_before = before.matches('"').count();
        let in_string = quotes_before % 2 == 1;

        // Exclude patterns that are clearly inside a path or identifier (e.g., "CB-XXX")
        // Real tech debt markers have whitespace or comment marker right before them
        let char_before = before.chars().last();
        let has_space_before =
            matches!(char_before, Some(' ') | Some('\t') | Some('/') | Some('*') | None);

        return has_comment && !in_string && has_space_before;
    }

    // TestDebt and GpuKernelBugs patterns can appear in comments
    // These are documentation of known issues, not actual code
    let is_comment_pattern = matches!(
        pattern,
        "were removed"
            | "tests hang"
            | "hang during"
            | "compilation hang"
            | "// skip"
            | "// skipped"
            | "// broken"
            | "// fails"
            | "// disabled"
            | "// fallback"
            | "// degraded"
            | "CUDA_ERROR"
            | "INVALID_PTX"
            | "PTX error"
            | "kernel fail"
    );
    if is_comment_pattern {
        // These patterns should match in comments
        let is_comment = trimmed.starts_with("//");
        let quotes_before = before.matches('"').count();
        let in_string = quotes_before % 2 == 1;

        // Skip doc comments (//! or ///) - these are documentation, not bug markers
        let is_doc_comment = trimmed.starts_with("///") || trimmed.starts_with("//!");
        if is_doc_comment {
            return false;
        }

        // Skip debug/diagnostic comments (Debug:, for debugging, etc.)
        let line_lower = line.to_lowercase();
        if line_lower.contains("debug:")
            || line_lower.contains("for debugging")
            || line_lower.contains("diagnostic")
        {
            return false;
        }

        // Skip architectural documentation comments that explain error handling
        // e.g., "returns CUDA_ERROR_UNKNOWN" in design documentation
        if line_lower.contains("returns cuda_error")
            || line_lower.contains("fix:")
            || line_lower.contains("via ")
            || line_lower.contains("sentinel")
            || line_lower.contains("recreates")
        {
            return false;
        }

        return is_comment && !in_string;
    }

    // HiddenDebt euphemisms can appear in doc comments, regular comments, or code
    // (e.g., variable names like `placeholder_logits`, function names like `stub_impl`)
    let is_euphemism_pattern = matches!(
        pattern,
        "placeholder"
            | "stub"
            | "dummy"
            | "fake"
            | "mock"
            | "simplified"
            | "for demonstration"
            | "demo only"
            | "not implemented"
            | "unimplemented"
            | "temporary"
            | "hardcoded"
            | "hard-coded"
            | "magic number"
            | "workaround"
            | "quick fix"
            | "quick-fix"
            | "bandaid"
            | "band-aid"
            | "kludge"
            | "tech debt"
            | "technical debt"
    );
    if is_euphemism_pattern {
        // Euphemisms can appear in regular comments or code, but not doc comments
        // Doc comments (/// or //!) usually describe what code does, not mark debt
        let is_doc_comment = trimmed.starts_with("///") || trimmed.starts_with("//!");
        if is_doc_comment {
            return false;
        }

        // Only exclude if inside a string literal (to avoid false positives in user-facing text)
        let quotes_before = before.matches('"').count();
        let in_string = quotes_before % 2 == 1;
        if in_string {
            return false;
        }

        // Skip unimplemented!() with explanatory messages - these are intentional design choices
        // e.g., unimplemented!("does not support stochastic updates")
        // Also skip bare `unimplemented!(` which likely has message on next line
        if pattern == "unimplemented" {
            let line_lower = line.to_lowercase();
            // Skip if it has an explanatory message indicating intentional non-support
            if line_lower.contains("does not support")
                || line_lower.contains("not supported")
                || line_lower.contains("use minimize")
                || line_lower.contains("by design")
            {
                return false;
            }
            // Skip bare `unimplemented!(` on its own line - message is likely on next line
            // These are typically intentional with explanatory messages
            let trimmed_lower = trimmed.to_lowercase();
            if trimmed_lower == "unimplemented!("
                || trimmed_lower.starts_with("unimplemented!(") && !trimmed_lower.contains(')')
            {
                return false;
            }
            // Skip test functions that verify unimplemented behavior
            // e.g., fn test_foo_unimplemented() or #[should_panic] tests
            if line_lower.contains("_unimplemented")
                || line_lower.contains("should_panic")
                || line_lower.contains("// test unimplemented")
            {
                return false;
            }
        }

        // Skip "not implemented" in test assertion messages or test function names
        // e.g., assert!(result.is_err()); // returns error for not implemented
        // Also skip multi-line assert messages (lines ending with ", or ")
        if pattern == "not implemented" {
            let line_lower = line.to_lowercase();
            // Skip if it's describing expected test behavior
            if line_lower.contains("assert")
                || line_lower.contains("expect")
                || line_lower.contains("returns error")
                || line_lower.contains("should fail")
                || line_lower.contains("should panic")
                || line_lower.contains("test_")
                || line_lower.contains("_test")
                || line_lower.contains("is_err")
            {
                return false;
            }
            // Skip if it looks like a multi-line string in an assertion
            // (line ends with ", or " which suggests it's an error message)
            let trimmed_end = trimmed.trim_end();
            if trimmed_end.ends_with("\",") || trimmed_end.ends_with("\"") {
                return false;
            }
            // Skip if it's inside a format string (has {} placeholders)
            if line.contains("{}") || line.contains("{:") {
                return false;
            }
            // Skip test comments explaining expected error behavior
            // e.g., "// Still fails (not implemented)", "// Correct dimensions but not implemented"
            if trimmed.starts_with("//") {
                // Skip if it's describing failure/error expectations
                if line_lower.contains("fails") || line_lower.contains("error") {
                    return false;
                }
                // Skip if it's a brief note about not being implemented (likely test context)
                // e.g., "// not implemented" or "// but not implemented"
                if line_lower.contains("but not implemented") || trimmed.len() < 50 {
                    return false;
                }
            }
        }

        // For single-word euphemisms (placeholder, stub, etc.), check if they're part of an identifier
        // If the euphemism is NOT at the start of an identifier, skip it (e.g., `module_placeholder`)
        // But flag it if it IS at the start (e.g., `placeholder_logits` = placeholder data)
        let single_word_euphemisms = [
            "placeholder",
            "stub",
            "dummy",
            "fake",
            "mock",
            "temporary",
            "hardcoded",
        ];
        if single_word_euphemisms.contains(&pattern) {
            // Check character before pattern - if it's `_` or alphanumeric, it's mid-identifier
            if let Some(c) = before.chars().last() {
                if c == '_' || c.is_alphanumeric() {
                    return false;
                }
            }
        }

        // Skip "hardcoded" in comments that explain test expectations (e.g., "from the hardcoded")
        if pattern == "hardcoded" || pattern == "hard-coded" {
            let line_lower = line.to_lowercase();
            // Skip test explanation comments
            if line_lower.contains("from the hardcoded")
                || line_lower.contains("uses hardcoded")
                || line_lower.contains("using hardcoded")
                || (trimmed.starts_with("//") && line_lower.contains("should"))
            {
                return false;
            }
        }

        return true;
    }

    // For code patterns (unwrap, unsafe, etc.), they should be actual code
    // Exclude if inside a string literal (basic heuristic)
    let quotes_before = before.matches('"').count();
    let in_string = quotes_before % 2 == 1;

    // Also exclude if it's part of documentation/comment text
    let is_doc_comment = trimmed.starts_with("///") || trimmed.starts_with("//!");
    let is_comment = trimmed.starts_with("//");

    // For keyword patterns like "unsafe {", check word boundary
    // Don't match "in_unsafe {" or "foo_unsafe {"
    let keyword_patterns = ["unsafe {", "transmute", "panic!"];
    if keyword_patterns
        .iter()
        .any(|kw| pattern.starts_with(kw.split_whitespace().next().unwrap_or(kw)))
    {
        // Check character before the pattern position
        if let Some(c) = before.chars().last() {
            // If preceded by alphanumeric or underscore, it's part of an identifier
            if c.is_alphanumeric() || c == '_' {
                return false;
            }
        }
    }

    // For code patterns, we want actual code, not comments
    !in_string && !is_doc_comment && !is_comment
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_real_pattern_todo_in_comment() {
        assert!(is_real_pattern("// TODO: fix this", "TODO"));
    }

    #[test]
    fn test_is_real_pattern_todo_in_string() {
        assert!(!is_real_pattern(r#"let msg = "TODO: implement";"#, "TODO"));
    }

    #[test]
    fn test_is_real_pattern_todo_in_doc_comment() {
        assert!(!is_real_pattern("/// TODO: document this", "TODO"));
    }

    #[test]
    fn test_is_real_pattern_multiple_patterns() {
        // Line mentions multiple SATD patterns - probably explaining them
        assert!(!is_real_pattern(
            "// For TODO/FIXME/HACK/XXX patterns",
            "TODO"
        ));
    }

    #[test]
    fn test_compute_test_lines_basic() {
        let content = "fn normal() {}\n\n#[cfg(test)]\nmod tests {\n    fn test_foo() {}\n}\n";
        let test_lines = compute_test_lines(content);
        // Line 3 is #[cfg(test)], lines 4-6 are inside test module
        assert!(test_lines.contains(&3)); // #[cfg(test)]
        assert!(test_lines.contains(&4)); // mod tests {
        assert!(test_lines.contains(&5)); // fn test_foo() {}
        assert!(test_lines.contains(&6)); // }
        // Line 1 is normal function, not in test
        assert!(!test_lines.contains(&1));
    }

    #[test]
    fn test_should_suppress_identical_blocks_mapper() {
        let finding = Finding::new("BH-001", std::path::PathBuf::new(), 1, "identical blocks");
        assert!(should_suppress_finding(&finding, "Foo => Bar"));
    }

    #[test]
    fn test_should_suppress_intentional() {
        let finding = Finding::new("BH-001", std::path::PathBuf::new(), 1, "identical blocks");
        assert!(should_suppress_finding(
            &finding,
            "// INTENTIONAL duplicate"
        ));
    }
}
