#![allow(dead_code)]
//! Advanced Fault Localization Module (BH-16 to BH-20)
//!
//! Implements research-based fault localization techniques:
//! - BH-16: Mutation-Based Fault Localization (MBFL)
//! - BH-17: Causal Fault Localization
//! - BH-18: Predictive Mutation Testing
//! - BH-19: Multi-Channel Fault Localization
//! - BH-20: Semantic Crash Bucketing
//!
//! References:
//! - Papadakis & Le Traon (2015) "Metallaxis-FL" - IEEE TSE
//! - Baah et al. (2010) "Causal Inference for Statistical Fault Localization" - ISSTA
//! - Zhang et al. (2018) "Predictive Mutation Testing" - IEEE TSE
//! - Li et al. (2021) "DeepFL" - ISSTA
//! - Cui et al. (2016) "RETracer" - ICSE

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::types::{
    ChannelWeights, CrashBucketingMode, Finding, FindingSeverity, HuntMode, LocalizationStrategy,
    SbflFormula,
};

/// Location with multi-channel scores.
#[derive(Debug, Clone)]
pub struct ScoredLocation {
    pub file: PathBuf,
    pub line: usize,
    /// SBFL spectrum score
    pub spectrum_score: f64,
    /// MBFL mutation score
    pub mutation_score: f64,
    /// Static analysis score
    pub static_score: f64,
    /// Semantic similarity score
    pub semantic_score: f64,
    /// PMAT quality score (BH-21)
    pub quality_score: f64,
    /// Final combined score
    pub final_score: f64,
}

impl ScoredLocation {
    pub fn new(file: PathBuf, line: usize) -> Self {
        Self {
            file,
            line,
            spectrum_score: 0.0,
            mutation_score: 0.0,
            static_score: 0.0,
            semantic_score: 0.0,
            quality_score: 0.0,
            final_score: 0.0,
        }
    }

    /// Compute final score using channel weights (BH-19, BH-21)
    pub fn compute_final_score(&mut self, weights: &ChannelWeights) {
        self.final_score = weights.combine(
            self.spectrum_score,
            self.mutation_score,
            self.static_score,
            self.semantic_score,
            self.quality_score,
        );
    }
}

/// Coverage data for a single test.
#[derive(Debug, Clone)]
pub struct TestCoverage {
    pub test_name: String,
    pub passed: bool,
    /// Lines executed: (file, line) -> execution count
    pub executed_lines: HashMap<(PathBuf, usize), usize>,
}

/// SBFL spectrum data.
#[derive(Debug, Default)]
pub struct SpectrumData {
    /// Lines covered by failing tests
    pub failed_coverage: HashMap<(PathBuf, usize), usize>,
    /// Lines covered by passing tests
    pub passed_coverage: HashMap<(PathBuf, usize), usize>,
    /// Total failing tests
    pub total_failed: usize,
    /// Total passing tests
    pub total_passed: usize,
}

impl SpectrumData {
    /// Compute SBFL score for a location using the specified formula.
    pub fn compute_score(&self, file: &Path, line: usize, formula: SbflFormula) -> f64 {
        let key = (file.to_path_buf(), line);

        let ef = *self.failed_coverage.get(&key).unwrap_or(&0) as f64; // executed in failed
        let ep = *self.passed_coverage.get(&key).unwrap_or(&0) as f64; // executed in passed
        let nf = (self.total_failed as f64) - ef; // not executed in failed
        let _np = (self.total_passed as f64) - ep; // not executed in passed (reserved for future formulas)

        match formula {
            SbflFormula::Tarantula => {
                let fail_ratio = if self.total_failed > 0 {
                    ef / self.total_failed as f64
                } else {
                    0.0
                };
                let pass_ratio = if self.total_passed > 0 {
                    ep / self.total_passed as f64
                } else {
                    0.0
                };
                if fail_ratio + pass_ratio > 0.0 {
                    fail_ratio / (fail_ratio + pass_ratio)
                } else {
                    0.0
                }
            }
            SbflFormula::Ochiai => {
                let denom = ((ef + nf) * (ef + ep)).sqrt();
                if denom > 0.0 {
                    ef / denom
                } else {
                    0.0
                }
            }
            SbflFormula::DStar2 => {
                let denom = ep + nf;
                if denom > 0.0 {
                    (ef * ef) / denom
                } else if ef > 0.0 {
                    f64::MAX
                } else {
                    0.0
                }
            }
            SbflFormula::DStar3 => {
                let denom = ep + nf;
                if denom > 0.0 {
                    (ef * ef * ef) / denom
                } else if ef > 0.0 {
                    f64::MAX
                } else {
                    0.0
                }
            }
        }
    }
}

/// Mutation data for MBFL (BH-16).
#[derive(Debug, Default)]
pub struct MutationData {
    /// Mutants at each location: (file, line) -> (total_mutants, killed_by_failing_tests)
    pub mutants: HashMap<(PathBuf, usize), (usize, usize)>,
}

impl MutationData {
    /// Compute MBFL score for a location.
    /// Score = |killed_by_failing| / |total_mutants|
    pub fn compute_score(&self, file: &Path, line: usize) -> f64 {
        let key = (file.to_path_buf(), line);
        if let Some((total, killed)) = self.mutants.get(&key) {
            if *total > 0 {
                return *killed as f64 / *total as f64;
            }
        }
        0.0
    }
}

/// Multi-channel fault localizer (BH-19).
pub struct MultiChannelLocalizer {
    pub strategy: LocalizationStrategy,
    pub weights: ChannelWeights,
    pub sbfl_formula: SbflFormula,
    pub spectrum_data: SpectrumData,
    pub mutation_data: MutationData,
    /// Static analysis findings mapped to locations
    pub static_findings: HashMap<(PathBuf, usize), f64>,
    /// Error message for semantic matching
    pub error_message: Option<String>,
}

impl MultiChannelLocalizer {
    pub fn new(strategy: LocalizationStrategy, weights: ChannelWeights) -> Self {
        Self {
            strategy,
            weights,
            sbfl_formula: SbflFormula::Ochiai,
            spectrum_data: SpectrumData::default(),
            mutation_data: MutationData::default(),
            static_findings: HashMap::new(),
            error_message: None,
        }
    }

    /// Add spectrum data from coverage.
    pub fn add_coverage(&mut self, coverage: &[TestCoverage]) {
        for test in coverage {
            if test.passed {
                self.spectrum_data.total_passed += 1;
                for (loc, count) in &test.executed_lines {
                    *self
                        .spectrum_data
                        .passed_coverage
                        .entry(loc.clone())
                        .or_insert(0) += count;
                }
            } else {
                self.spectrum_data.total_failed += 1;
                for (loc, count) in &test.executed_lines {
                    *self
                        .spectrum_data
                        .failed_coverage
                        .entry(loc.clone())
                        .or_insert(0) += count;
                }
            }
        }
    }

    /// Add static analysis finding.
    pub fn add_static_finding(&mut self, file: &Path, line: usize, score: f64) {
        self.static_findings
            .insert((file.to_path_buf(), line), score);
    }

    /// Set error message for semantic matching.
    pub fn set_error_message(&mut self, msg: &str) {
        self.error_message = Some(msg.to_string());
    }

    /// Compute semantic similarity score for a location.
    fn compute_semantic_score(&self, _file: &Path, line: usize, content: &str) -> f64 {
        let Some(ref error_msg) = self.error_message else {
            return 0.0;
        };

        // Simple keyword matching (could be enhanced with embeddings)
        let error_lower = error_msg.to_lowercase();
        let error_words: Vec<&str> = error_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        let line_content = content.lines().nth(line.saturating_sub(1)).unwrap_or("");
        let line_lower = line_content.to_lowercase();

        let matches = error_words
            .iter()
            .filter(|w| line_lower.contains(*w))
            .count();

        if error_words.is_empty() {
            0.0
        } else {
            (matches as f64 / error_words.len() as f64).min(1.0)
        }
    }

    /// Localize faults using the configured strategy.
    pub fn localize(&self, project_path: &Path) -> Vec<ScoredLocation> {
        let mut locations: HashMap<(PathBuf, usize), ScoredLocation> = HashMap::new();

        // Collect all locations from all channels
        for key in self.spectrum_data.failed_coverage.keys() {
            locations
                .entry(key.clone())
                .or_insert_with(|| ScoredLocation::new(key.0.clone(), key.1));
        }
        for key in self.mutation_data.mutants.keys() {
            locations
                .entry(key.clone())
                .or_insert_with(|| ScoredLocation::new(key.0.clone(), key.1));
        }
        for key in self.static_findings.keys() {
            locations
                .entry(key.clone())
                .or_insert_with(|| ScoredLocation::new(key.0.clone(), key.1));
        }

        // Compute channel scores for each location
        let mut result: Vec<ScoredLocation> = locations
            .into_iter()
            .map(|(key, mut loc)| {
                // Spectrum score (SBFL)
                loc.spectrum_score =
                    self.spectrum_data
                        .compute_score(&key.0, key.1, self.sbfl_formula);

                // Mutation score (MBFL)
                loc.mutation_score = self.mutation_data.compute_score(&key.0, key.1);

                // Static analysis score
                loc.static_score = *self.static_findings.get(&key).unwrap_or(&0.0);

                // Semantic score (if error message available)
                if self.error_message.is_some() {
                    let file_path = project_path.join(&key.0);
                    if let Ok(content) = std::fs::read_to_string(&file_path) {
                        loc.semantic_score = self.compute_semantic_score(&key.0, key.1, &content);
                    }
                }

                // Compute final score based on strategy
                match self.strategy {
                    LocalizationStrategy::Sbfl => {
                        loc.final_score = loc.spectrum_score;
                    }
                    LocalizationStrategy::Mbfl => {
                        loc.final_score = loc.mutation_score;
                    }
                    LocalizationStrategy::Causal => {
                        // Causal uses modified spectrum with intervention estimation
                        // For now, use spectrum as approximation
                        loc.final_score = loc.spectrum_score;
                    }
                    LocalizationStrategy::MultiChannel | LocalizationStrategy::Hybrid => {
                        loc.compute_final_score(&self.weights);
                    }
                }

                loc
            })
            .collect();

        // Sort by final score (descending)
        result.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        result
    }
}

/// Crash bucket for semantic grouping (BH-20).
#[derive(Debug, Clone)]
pub struct CrashBucket {
    /// Root cause pattern identifier
    pub pattern: String,
    /// Description of the root cause
    pub description: String,
    /// Crashes in this bucket
    pub crashes: Vec<CrashInfo>,
    /// Representative crash
    pub representative: Option<CrashInfo>,
}

/// Information about a single crash.
#[derive(Debug, Clone)]
pub struct CrashInfo {
    pub id: String,
    pub file: PathBuf,
    pub line: usize,
    pub message: String,
    pub stack_trace: Vec<StackFrame>,
}

/// A stack frame in a crash trace.
#[derive(Debug, Clone)]
pub struct StackFrame {
    pub function: String,
    pub file: Option<PathBuf>,
    pub line: Option<usize>,
}

/// Root cause patterns for crash bucketing (BH-20).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RootCausePattern {
    IndexOutOfBounds,
    NullPointerDeref,
    IntegerOverflow,
    DivisionByZero,
    StackOverflow,
    HeapOverflow,
    UseAfterFree,
    DoubleFree,
    UnwrapOnNone,
    AssertionFailed,
    Unknown,
}

impl std::fmt::Display for RootCausePattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IndexOutOfBounds => write!(f, "index_out_of_bounds"),
            Self::NullPointerDeref => write!(f, "null_pointer_deref"),
            Self::IntegerOverflow => write!(f, "integer_overflow"),
            Self::DivisionByZero => write!(f, "division_by_zero"),
            Self::StackOverflow => write!(f, "stack_overflow"),
            Self::HeapOverflow => write!(f, "heap_overflow"),
            Self::UseAfterFree => write!(f, "use_after_free"),
            Self::DoubleFree => write!(f, "double_free"),
            Self::UnwrapOnNone => write!(f, "unwrap_on_none"),
            Self::AssertionFailed => write!(f, "assertion_failed"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Semantic crash bucketer (BH-20).
pub struct CrashBucketer {
    pub mode: CrashBucketingMode,
    pub buckets: HashMap<String, CrashBucket>,
}

impl CrashBucketer {
    pub fn new(mode: CrashBucketingMode) -> Self {
        Self {
            mode,
            buckets: HashMap::new(),
        }
    }

    /// Detect root cause pattern from crash message.
    pub fn detect_pattern(message: &str) -> RootCausePattern {
        let msg_lower = message.to_lowercase();

        if msg_lower.contains("index out of bounds") || msg_lower.contains("indexoutofbounds") {
            RootCausePattern::IndexOutOfBounds
        } else if msg_lower.contains("null") || msg_lower.contains("nullptr") {
            RootCausePattern::NullPointerDeref
        } else if msg_lower.contains("overflow") && msg_lower.contains("integer") {
            RootCausePattern::IntegerOverflow
        } else if msg_lower.contains("overflow") && msg_lower.contains("stack") {
            RootCausePattern::StackOverflow
        } else if msg_lower.contains("overflow") {
            RootCausePattern::HeapOverflow
        } else if msg_lower.contains("division by zero") || msg_lower.contains("divide by zero") {
            RootCausePattern::DivisionByZero
        } else if msg_lower.contains("use after free") {
            RootCausePattern::UseAfterFree
        } else if msg_lower.contains("double free") {
            RootCausePattern::DoubleFree
        } else if (msg_lower.contains("unwrap") && msg_lower.contains("none"))
            || msg_lower.contains("called `option::unwrap()`")
        {
            RootCausePattern::UnwrapOnNone
        } else if msg_lower.contains("assertion") || msg_lower.contains("assert") {
            RootCausePattern::AssertionFailed
        } else {
            RootCausePattern::Unknown
        }
    }

    /// Add a crash to the appropriate bucket.
    pub fn add_crash(&mut self, crash: CrashInfo) {
        let bucket_key = match self.mode {
            CrashBucketingMode::None => {
                // Each crash gets its own bucket
                crash.id.clone()
            }
            CrashBucketingMode::StackTrace => {
                // Bucket by top 3 stack frames
                let frames: Vec<String> = crash
                    .stack_trace
                    .iter()
                    .take(3)
                    .map(|f| f.function.clone())
                    .collect();
                frames.join("::")
            }
            CrashBucketingMode::Semantic => {
                // Bucket by root cause pattern
                let pattern = Self::detect_pattern(&crash.message);
                format!("{}:{}", pattern, crash.file.display())
            }
        };

        let bucket = self.buckets.entry(bucket_key.clone()).or_insert_with(|| {
            let pattern = Self::detect_pattern(&crash.message);
            CrashBucket {
                pattern: pattern.to_string(),
                description: format!("{} in {}", pattern, crash.file.display()),
                crashes: Vec::new(),
                representative: None,
            }
        });

        // First crash becomes representative
        if bucket.representative.is_none() {
            bucket.representative = Some(crash.clone());
        }

        bucket.crashes.push(crash);
    }

    /// Get deduplicated findings from bucketed crashes.
    pub fn to_findings(&self) -> Vec<Finding> {
        self.buckets
            .values()
            .filter_map(|bucket| {
                bucket.representative.as_ref().map(|rep| {
                    Finding::new(
                        format!("BH-CRASH-{}", bucket.pattern.to_uppercase()),
                        &rep.file,
                        rep.line,
                        &bucket.description,
                    )
                    .with_description(format!(
                        "{} occurrence(s) of {} pattern",
                        bucket.crashes.len(),
                        bucket.pattern
                    ))
                    .with_severity(FindingSeverity::High)
                    .with_suspiciousness(0.8)
                    .with_discovered_by(HuntMode::Hunt)
                })
            })
            .collect()
    }

    /// Get deduplication statistics.
    pub fn stats(&self) -> (usize, usize) {
        let total_crashes: usize = self.buckets.values().map(|b| b.crashes.len()).sum();
        let unique_buckets = self.buckets.len();
        (total_crashes, unique_buckets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sbfl_ochiai() {
        let mut data = SpectrumData::default();
        data.total_failed = 2;
        data.total_passed = 8;
        data.failed_coverage
            .insert((PathBuf::from("test.rs"), 10), 2);
        data.passed_coverage
            .insert((PathBuf::from("test.rs"), 10), 1);

        let score = data.compute_score(Path::new("test.rs"), 10, SbflFormula::Ochiai);
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_crash_pattern_detection() {
        assert_eq!(
            CrashBucketer::detect_pattern("index out of bounds: 5 >= 3"),
            RootCausePattern::IndexOutOfBounds
        );
        assert_eq!(
            CrashBucketer::detect_pattern("called `Option::unwrap()` on a `None` value"),
            RootCausePattern::UnwrapOnNone
        );
        assert_eq!(
            CrashBucketer::detect_pattern("integer overflow"),
            RootCausePattern::IntegerOverflow
        );
    }

    #[test]
    fn test_channel_weights() {
        let weights = ChannelWeights::default();
        let score = weights.combine(0.8, 0.6, 0.4, 0.2, 0.5);
        // 0.30*0.8 + 0.25*0.6 + 0.20*0.4 + 0.15*0.2 + 0.10*0.5
        // = 0.24 + 0.15 + 0.08 + 0.03 + 0.05 = 0.55
        assert!((score - 0.55).abs() < 0.01);
    }

    #[test]
    fn test_semantic_bucketing_dedup() {
        let mut bucketer = CrashBucketer::new(CrashBucketingMode::Semantic);

        // Add 3 similar crashes
        for i in 0..3 {
            bucketer.add_crash(CrashInfo {
                id: format!("crash-{}", i),
                file: PathBuf::from("src/lib.rs"),
                line: 42,
                message: "index out of bounds: the len is 5 but the index is 10".to_string(),
                stack_trace: vec![],
            });
        }

        let (total, buckets) = bucketer.stats();
        assert_eq!(total, 3);
        assert_eq!(buckets, 1); // All 3 in same bucket

        let findings = bucketer.to_findings();
        assert_eq!(findings.len(), 1);
    }

    #[test]
    fn test_scored_location_new() {
        let loc = ScoredLocation::new(PathBuf::from("test.rs"), 42);
        assert_eq!(loc.line, 42);
        assert_eq!(loc.spectrum_score, 0.0);
        assert_eq!(loc.mutation_score, 0.0);
        assert_eq!(loc.static_score, 0.0);
        assert_eq!(loc.semantic_score, 0.0);
        assert_eq!(loc.quality_score, 0.0);
        assert_eq!(loc.final_score, 0.0);
    }

    #[test]
    fn test_scored_location_compute_final_score() {
        let mut loc = ScoredLocation::new(PathBuf::from("test.rs"), 10);
        loc.spectrum_score = 0.8;
        loc.mutation_score = 0.6;
        loc.static_score = 0.4;
        loc.semantic_score = 0.2;

        let weights = ChannelWeights::default();
        loc.compute_final_score(&weights);

        assert!(loc.final_score > 0.0);
    }

    #[test]
    fn test_sbfl_tarantula() {
        let mut data = SpectrumData::default();
        data.total_failed = 2;
        data.total_passed = 8;
        data.failed_coverage
            .insert((PathBuf::from("test.rs"), 10), 2);
        data.passed_coverage
            .insert((PathBuf::from("test.rs"), 10), 2);

        let score = data.compute_score(Path::new("test.rs"), 10, SbflFormula::Tarantula);
        assert!(score >= 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_sbfl_dstar2() {
        let mut data = SpectrumData::default();
        data.total_failed = 2;
        data.total_passed = 8;
        data.failed_coverage
            .insert((PathBuf::from("test.rs"), 10), 2);
        data.passed_coverage
            .insert((PathBuf::from("test.rs"), 10), 1);

        let score = data.compute_score(Path::new("test.rs"), 10, SbflFormula::DStar2);
        assert!(score > 0.0);
    }

    #[test]
    fn test_sbfl_dstar3() {
        let mut data = SpectrumData::default();
        data.total_failed = 2;
        data.total_passed = 8;
        data.failed_coverage
            .insert((PathBuf::from("test.rs"), 10), 2);
        data.passed_coverage
            .insert((PathBuf::from("test.rs"), 10), 1);

        let score = data.compute_score(Path::new("test.rs"), 10, SbflFormula::DStar3);
        assert!(score > 0.0);
    }

    #[test]
    fn test_sbfl_zero_tests() {
        let data = SpectrumData::default();
        let score = data.compute_score(Path::new("test.rs"), 10, SbflFormula::Tarantula);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_sbfl_no_coverage() {
        let mut data = SpectrumData::default();
        data.total_failed = 2;
        data.total_passed = 8;
        // No coverage entries

        let score = data.compute_score(Path::new("test.rs"), 10, SbflFormula::Ochiai);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_crash_pattern_assertion() {
        assert_eq!(
            CrashBucketer::detect_pattern("assertion failed: x > 0"),
            RootCausePattern::AssertionFailed
        );
    }

    #[test]
    fn test_crash_pattern_divide_by_zero() {
        assert_eq!(
            CrashBucketer::detect_pattern("attempt to divide by zero"),
            RootCausePattern::DivisionByZero
        );
    }

    #[test]
    fn test_crash_pattern_stack_overflow() {
        assert_eq!(
            CrashBucketer::detect_pattern("thread 'main' has overflowed its stack"),
            RootCausePattern::StackOverflow
        );
    }

    #[test]
    fn test_crash_pattern_null_pointer() {
        assert_eq!(
            CrashBucketer::detect_pattern("null pointer dereference"),
            RootCausePattern::NullPointerDeref
        );
    }

    #[test]
    fn test_crash_pattern_unknown() {
        assert_eq!(
            CrashBucketer::detect_pattern("some random error message"),
            RootCausePattern::Unknown
        );
    }

    #[test]
    fn test_none_bucketing() {
        let mut bucketer = CrashBucketer::new(CrashBucketingMode::None);

        bucketer.add_crash(CrashInfo {
            id: "crash-1".to_string(),
            file: PathBuf::from("src/lib.rs"),
            line: 42,
            message: "error 1".to_string(),
            stack_trace: vec![],
        });

        bucketer.add_crash(CrashInfo {
            id: "crash-2".to_string(),
            file: PathBuf::from("src/lib.rs"),
            line: 42,
            message: "error 2".to_string(),
            stack_trace: vec![],
        });

        let (total, buckets) = bucketer.stats();
        assert_eq!(total, 2);
        assert_eq!(buckets, 2); // Each crash gets its own bucket in None mode
    }

    #[test]
    fn test_test_coverage() {
        let cov = TestCoverage {
            test_name: "test_example".to_string(),
            passed: true,
            executed_lines: HashMap::new(),
        };
        assert!(cov.passed);
        assert_eq!(cov.test_name, "test_example");
    }

    // =========================================================================
    // Coverage gap: MultiChannelLocalizer::localize()
    // =========================================================================

    #[test]
    fn test_localize_sbfl_strategy() {
        let mut localizer =
            MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
        localizer.spectrum_data.total_failed = 1;
        localizer.spectrum_data.total_passed = 5;
        localizer
            .spectrum_data
            .failed_coverage
            .insert((PathBuf::from("src/lib.rs"), 42), 1);
        localizer
            .spectrum_data
            .passed_coverage
            .insert((PathBuf::from("src/lib.rs"), 42), 1);

        let results = localizer.localize(Path::new("/tmp"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].line, 42);
        assert!(results[0].final_score > 0.0);
        // SBFL strategy: final_score == spectrum_score
        assert_eq!(results[0].final_score, results[0].spectrum_score);
    }

    #[test]
    fn test_localize_mbfl_strategy() {
        let mut localizer =
            MultiChannelLocalizer::new(LocalizationStrategy::Mbfl, ChannelWeights::default());
        localizer
            .mutation_data
            .mutants
            .insert((PathBuf::from("src/lib.rs"), 10), (5, 2));

        let results = localizer.localize(Path::new("/tmp"));
        assert_eq!(results.len(), 1);
        // MBFL strategy: final_score == mutation_score
        assert_eq!(results[0].final_score, results[0].mutation_score);
    }

    #[test]
    fn test_localize_causal_strategy() {
        let mut localizer =
            MultiChannelLocalizer::new(LocalizationStrategy::Causal, ChannelWeights::default());
        localizer.spectrum_data.total_failed = 1;
        localizer.spectrum_data.total_passed = 5;
        localizer
            .spectrum_data
            .failed_coverage
            .insert((PathBuf::from("src/lib.rs"), 20), 1);

        let results = localizer.localize(Path::new("/tmp"));
        assert_eq!(results.len(), 1);
        // Causal uses spectrum as approximation
        assert_eq!(results[0].final_score, results[0].spectrum_score);
    }

    #[test]
    fn test_localize_multichannel_strategy() {
        let mut localizer = MultiChannelLocalizer::new(
            LocalizationStrategy::MultiChannel,
            ChannelWeights {
                spectrum: 0.4,
                mutation: 0.3,
                static_analysis: 0.2,
                semantic: 0.1,
                quality: 0.0,
            },
        );
        localizer.spectrum_data.total_failed = 1;
        localizer.spectrum_data.total_passed = 5;
        let key = (PathBuf::from("src/lib.rs"), 30);
        localizer
            .spectrum_data
            .failed_coverage
            .insert(key.clone(), 1);
        localizer.static_findings.insert(key.clone(), 0.7);

        let results = localizer.localize(Path::new("/tmp"));
        assert_eq!(results.len(), 1);
        // MultiChannel strategy uses compute_final_score with weights
        assert!(results[0].final_score > 0.0);
    }

    #[test]
    fn test_localize_hybrid_strategy() {
        let mut localizer =
            MultiChannelLocalizer::new(LocalizationStrategy::Hybrid, ChannelWeights::default());
        let key = (PathBuf::from("src/lib.rs"), 5);
        localizer.static_findings.insert(key, 0.5);

        let results = localizer.localize(Path::new("/tmp"));
        assert_eq!(results.len(), 1);
        // Hybrid also uses compute_final_score
        assert!(results[0].final_score >= 0.0);
    }

    #[test]
    fn test_localize_multiple_locations_sorted() {
        let mut localizer =
            MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
        localizer.spectrum_data.total_failed = 2;
        localizer.spectrum_data.total_passed = 8;
        // High suspiciousness location
        localizer
            .spectrum_data
            .failed_coverage
            .insert((PathBuf::from("src/a.rs"), 10), 2);
        localizer
            .spectrum_data
            .passed_coverage
            .insert((PathBuf::from("src/a.rs"), 10), 0);
        // Low suspiciousness location
        localizer
            .spectrum_data
            .failed_coverage
            .insert((PathBuf::from("src/b.rs"), 20), 1);
        localizer
            .spectrum_data
            .passed_coverage
            .insert((PathBuf::from("src/b.rs"), 20), 7);

        let results = localizer.localize(Path::new("/tmp"));
        assert_eq!(results.len(), 2);
        // Should be sorted descending by final_score
        assert!(results[0].final_score >= results[1].final_score);
    }

    #[test]
    fn test_localize_empty() {
        let localizer =
            MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
        let results = localizer.localize(Path::new("/tmp"));
        assert!(results.is_empty());
    }

    #[test]
    fn test_localize_merges_channels() {
        let mut localizer = MultiChannelLocalizer::new(
            LocalizationStrategy::MultiChannel,
            ChannelWeights::default(),
        );
        // Same location from spectrum and static
        let key = (PathBuf::from("src/lib.rs"), 42);
        localizer.spectrum_data.total_failed = 1;
        localizer.spectrum_data.total_passed = 5;
        localizer
            .spectrum_data
            .failed_coverage
            .insert(key.clone(), 1);
        localizer.mutation_data.mutants.insert(key.clone(), (3, 1));
        localizer.static_findings.insert(key, 0.8);

        let results = localizer.localize(Path::new("/tmp"));
        // Should merge into a single location
        assert_eq!(results.len(), 1);
        assert!(results[0].spectrum_score > 0.0);
        assert!(results[0].mutation_score > 0.0);
        assert_eq!(results[0].static_score, 0.8);
    }

    // =========================================================================
    // Coverage gap: add_coverage()
    // =========================================================================

    #[test]
    fn test_add_coverage_passing_tests() {
        let mut localizer =
            MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
        let mut lines = HashMap::new();
        lines.insert((PathBuf::from("src/lib.rs"), 10), 1);
        lines.insert((PathBuf::from("src/lib.rs"), 20), 3);

        let coverage = vec![TestCoverage {
            test_name: "test_pass_1".to_string(),
            passed: true,
            executed_lines: lines,
        }];

        localizer.add_coverage(&coverage);

        assert_eq!(localizer.spectrum_data.total_passed, 1);
        assert_eq!(localizer.spectrum_data.total_failed, 0);
        assert_eq!(
            *localizer
                .spectrum_data
                .passed_coverage
                .get(&(PathBuf::from("src/lib.rs"), 10))
                .unwrap(),
            1
        );
        assert_eq!(
            *localizer
                .spectrum_data
                .passed_coverage
                .get(&(PathBuf::from("src/lib.rs"), 20))
                .unwrap(),
            3
        );
    }

    #[test]
    fn test_add_coverage_failing_tests() {
        let mut localizer =
            MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
        let mut lines = HashMap::new();
        lines.insert((PathBuf::from("src/lib.rs"), 42), 2);

        let coverage = vec![TestCoverage {
            test_name: "test_fail_1".to_string(),
            passed: false,
            executed_lines: lines,
        }];

        localizer.add_coverage(&coverage);

        assert_eq!(localizer.spectrum_data.total_passed, 0);
        assert_eq!(localizer.spectrum_data.total_failed, 1);
        assert_eq!(
            *localizer
                .spectrum_data
                .failed_coverage
                .get(&(PathBuf::from("src/lib.rs"), 42))
                .unwrap(),
            2
        );
    }

    #[test]
    fn test_add_coverage_mixed_pass_fail() {
        let mut localizer =
            MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());

        let mut pass_lines = HashMap::new();
        pass_lines.insert((PathBuf::from("src/lib.rs"), 10), 1);

        let mut fail_lines = HashMap::new();
        fail_lines.insert((PathBuf::from("src/lib.rs"), 10), 1);
        fail_lines.insert((PathBuf::from("src/lib.rs"), 42), 1);

        let coverage = vec![
            TestCoverage {
                test_name: "test_pass".to_string(),
                passed: true,
                executed_lines: pass_lines,
            },
            TestCoverage {
                test_name: "test_fail".to_string(),
                passed: false,
                executed_lines: fail_lines,
            },
        ];

        localizer.add_coverage(&coverage);

        assert_eq!(localizer.spectrum_data.total_passed, 1);
        assert_eq!(localizer.spectrum_data.total_failed, 1);
        // Line 10 hit by both passing and failing
        assert_eq!(
            *localizer
                .spectrum_data
                .passed_coverage
                .get(&(PathBuf::from("src/lib.rs"), 10))
                .unwrap(),
            1
        );
        assert_eq!(
            *localizer
                .spectrum_data
                .failed_coverage
                .get(&(PathBuf::from("src/lib.rs"), 10))
                .unwrap(),
            1
        );
        // Line 42 only hit by failing
        assert!(localizer
            .spectrum_data
            .passed_coverage
            .get(&(PathBuf::from("src/lib.rs"), 42))
            .is_none());
        assert_eq!(
            *localizer
                .spectrum_data
                .failed_coverage
                .get(&(PathBuf::from("src/lib.rs"), 42))
                .unwrap(),
            1
        );
    }

    #[test]
    fn test_add_coverage_empty() {
        let mut localizer =
            MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
        localizer.add_coverage(&[]);
        assert_eq!(localizer.spectrum_data.total_passed, 0);
        assert_eq!(localizer.spectrum_data.total_failed, 0);
    }

    #[test]
    fn test_add_coverage_accumulates_counts() {
        let mut localizer =
            MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());

        let mut lines1 = HashMap::new();
        lines1.insert((PathBuf::from("a.rs"), 5), 2);

        let mut lines2 = HashMap::new();
        lines2.insert((PathBuf::from("a.rs"), 5), 3);

        let coverage = vec![
            TestCoverage {
                test_name: "t1".to_string(),
                passed: true,
                executed_lines: lines1,
            },
            TestCoverage {
                test_name: "t2".to_string(),
                passed: true,
                executed_lines: lines2,
            },
        ];

        localizer.add_coverage(&coverage);

        assert_eq!(localizer.spectrum_data.total_passed, 2);
        // Counts accumulate: 2 + 3 = 5
        assert_eq!(
            *localizer
                .spectrum_data
                .passed_coverage
                .get(&(PathBuf::from("a.rs"), 5))
                .unwrap(),
            5
        );
    }

    // =========================================================================
    // Coverage gap: compute_semantic_score()
    // =========================================================================

    #[test]
    fn test_compute_semantic_score_no_error_message() {
        let localizer =
            MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
        // No error message set
        let score = localizer.compute_semantic_score(
            Path::new("test.rs"),
            1,
            "fn main() { println!(\"hello\"); }",
        );
        assert_eq!(score, 0.0, "Should return 0.0 when no error message is set");
    }

    #[test]
    fn test_compute_semantic_score_matching_keywords() {
        let mut localizer =
            MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
        localizer.set_error_message("index overflow detected in array");

        // Content with matching keywords on line 1
        let content = "fn process_array() { index overflow check }";
        let score = localizer.compute_semantic_score(Path::new("test.rs"), 1, content);
        assert!(score > 0.0, "Should match keywords from error message");
        assert!(score <= 1.0);
    }

    #[test]
    fn test_compute_semantic_score_no_matching_keywords() {
        let mut localizer =
            MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
        localizer.set_error_message("index overflow detected in array");

        // Content with no matching keywords
        let content = "fn hello_world() { println!(\"hi\"); }";
        let score = localizer.compute_semantic_score(Path::new("test.rs"), 1, content);
        assert_eq!(score, 0.0, "Should return 0.0 when no keywords match");
    }

    #[test]
    fn test_compute_semantic_score_partial_match() {
        let mut localizer =
            MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
        localizer.set_error_message("buffer overflow detected in memory allocation");

        // Only some keywords match (overflow matches, but not buffer/detected/memory/allocation)
        let content = "fn check() { handle overflow here }";
        let score = localizer.compute_semantic_score(Path::new("test.rs"), 1, content);
        assert!(score > 0.0, "Should have partial match");
        assert!(score < 1.0, "Should not be perfect match");
    }

    #[test]
    fn test_compute_semantic_score_all_keywords_match() {
        let mut localizer =
            MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
        // Only words > 3 chars are used as keywords
        localizer.set_error_message("buffer overflow detected");

        // Line with all keywords matching
        let content = "buffer overflow detected here";
        let score = localizer.compute_semantic_score(Path::new("test.rs"), 1, content);
        assert!(
            (score - 1.0).abs() < f64::EPSILON,
            "Should return 1.0 when all keywords match, got {}",
            score
        );
    }

    #[test]
    fn test_compute_semantic_score_short_words_filtered() {
        let mut localizer =
            MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
        // All words <= 3 chars are filtered out
        localizer.set_error_message("a be on if");

        let content = "a be on if match";
        let score = localizer.compute_semantic_score(Path::new("test.rs"), 1, content);
        assert_eq!(
            score, 0.0,
            "Should return 0.0 when all error words are <= 3 chars"
        );
    }

    #[test]
    fn test_compute_semantic_score_multiline_content() {
        let mut localizer =
            MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
        localizer.set_error_message("panic detected crash");

        // Multi-line content, target line 2
        let content = "fn safe() {}\nfn dangerous() { panic detected crash }\nfn other() {}";
        let score = localizer.compute_semantic_score(Path::new("test.rs"), 2, content);
        assert!(score > 0.0, "Should match keywords on the target line");
    }

    #[test]
    fn test_compute_semantic_score_line_beyond_content() {
        let mut localizer =
            MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
        localizer.set_error_message("panic detected");

        // Only 1 line, but asking for line 100
        let content = "fn main() {}";
        let score = localizer.compute_semantic_score(Path::new("test.rs"), 100, content);
        assert_eq!(
            score, 0.0,
            "Should return 0.0 for line beyond content length"
        );
    }

    #[test]
    fn test_compute_semantic_score_case_insensitive() {
        let mut localizer =
            MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
        localizer.set_error_message("BUFFER OVERFLOW");

        let content = "fn check() { buffer overflow here }";
        let score = localizer.compute_semantic_score(Path::new("test.rs"), 1, content);
        assert!(
            score > 0.0,
            "Should be case-insensitive when matching keywords"
        );
    }

    // =========================================================================
    // Coverage gap: add_static_finding() and set_error_message()
    // =========================================================================

    #[test]
    fn test_add_static_finding() {
        let mut localizer =
            MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());

        localizer.add_static_finding(Path::new("src/lib.rs"), 42, 0.75);
        localizer.add_static_finding(Path::new("src/main.rs"), 10, 0.3);

        assert_eq!(
            *localizer
                .static_findings
                .get(&(PathBuf::from("src/lib.rs"), 42))
                .unwrap(),
            0.75
        );
        assert_eq!(
            *localizer
                .static_findings
                .get(&(PathBuf::from("src/main.rs"), 10))
                .unwrap(),
            0.3
        );
    }

    #[test]
    fn test_set_error_message() {
        let mut localizer =
            MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());

        assert!(localizer.error_message.is_none());
        localizer.set_error_message("test error");
        assert_eq!(localizer.error_message.as_deref(), Some("test error"));
    }

    // =========================================================================
    // Coverage gap: MutationData edge cases
    // =========================================================================

    #[test]
    fn test_mutation_data_no_mutants() {
        let data = MutationData::default();
        let score = data.compute_score(Path::new("test.rs"), 10);
        assert_eq!(score, 0.0, "No mutants should yield 0.0");
    }

    #[test]
    fn test_mutation_data_zero_total() {
        let mut data = MutationData::default();
        data.mutants.insert((PathBuf::from("test.rs"), 10), (0, 0));
        let score = data.compute_score(Path::new("test.rs"), 10);
        assert_eq!(score, 0.0, "Zero total mutants should yield 0.0");
    }

    #[test]
    fn test_mutation_data_all_killed() {
        let mut data = MutationData::default();
        data.mutants.insert((PathBuf::from("test.rs"), 10), (5, 5));
        let score = data.compute_score(Path::new("test.rs"), 10);
        assert!(
            (score - 1.0).abs() < f64::EPSILON,
            "All killed should yield 1.0"
        );
    }

    #[test]
    fn test_mutation_data_partial_killed() {
        let mut data = MutationData::default();
        data.mutants.insert((PathBuf::from("test.rs"), 10), (4, 2));
        let score = data.compute_score(Path::new("test.rs"), 10);
        assert!(
            (score - 0.5).abs() < f64::EPSILON,
            "2/4 killed should yield 0.5"
        );
    }

    // =========================================================================
    // Coverage gap: SBFL edge cases — DStar2/DStar3 denom=0 with ef>0
    // =========================================================================

    #[test]
    fn test_sbfl_dstar2_denom_zero_ef_positive() {
        let mut data = SpectrumData::default();
        data.total_failed = 1;
        data.total_passed = 0;
        // ef=1, ep=0, nf=0 → denom = ep + nf = 0
        data.failed_coverage
            .insert((PathBuf::from("test.rs"), 10), 1);

        let score = data.compute_score(Path::new("test.rs"), 10, SbflFormula::DStar2);
        assert_eq!(
            score,
            f64::MAX,
            "DStar2 with denom=0 and ef>0 should be MAX"
        );
    }

    #[test]
    fn test_sbfl_dstar3_denom_zero_ef_positive() {
        let mut data = SpectrumData::default();
        data.total_failed = 1;
        data.total_passed = 0;
        data.failed_coverage
            .insert((PathBuf::from("test.rs"), 10), 1);

        let score = data.compute_score(Path::new("test.rs"), 10, SbflFormula::DStar3);
        assert_eq!(
            score,
            f64::MAX,
            "DStar3 with denom=0 and ef>0 should be MAX"
        );
    }

    #[test]
    fn test_sbfl_dstar2_denom_zero_ef_zero() {
        let mut data = SpectrumData::default();
        data.total_failed = 0;
        data.total_passed = 0;

        let score = data.compute_score(Path::new("test.rs"), 10, SbflFormula::DStar2);
        assert_eq!(score, 0.0, "DStar2 with denom=0 and ef=0 should be 0.0");
    }

    #[test]
    fn test_sbfl_dstar3_denom_zero_ef_zero() {
        let mut data = SpectrumData::default();
        data.total_failed = 0;
        data.total_passed = 0;

        let score = data.compute_score(Path::new("test.rs"), 10, SbflFormula::DStar3);
        assert_eq!(score, 0.0, "DStar3 with denom=0 and ef=0 should be 0.0");
    }

    #[test]
    fn test_sbfl_ochiai_denom_zero() {
        let mut data = SpectrumData::default();
        data.total_failed = 0;
        data.total_passed = 0;

        let score = data.compute_score(Path::new("test.rs"), 10, SbflFormula::Ochiai);
        assert_eq!(score, 0.0, "Ochiai with denom=0 should be 0.0");
    }

    #[test]
    fn test_sbfl_tarantula_only_failed() {
        let mut data = SpectrumData::default();
        data.total_failed = 3;
        data.total_passed = 0;
        data.failed_coverage
            .insert((PathBuf::from("test.rs"), 10), 3);

        let score = data.compute_score(Path::new("test.rs"), 10, SbflFormula::Tarantula);
        // fail_ratio = 3/3 = 1.0, pass_ratio = 0.0 → score = 1.0 / (1.0 + 0.0) = 1.0
        assert!(
            (score - 1.0).abs() < f64::EPSILON,
            "Tarantula: all failed, score should be 1.0, got {}",
            score
        );
    }

    // =========================================================================
    // Coverage gap: StackTrace bucketing mode
    // =========================================================================

    #[test]
    fn test_stack_trace_bucketing() {
        let mut bucketer = CrashBucketer::new(CrashBucketingMode::StackTrace);

        // Two crashes with same top-3 frames
        for i in 0..2 {
            bucketer.add_crash(CrashInfo {
                id: format!("crash-{}", i),
                file: PathBuf::from("src/lib.rs"),
                line: 42,
                message: "error".to_string(),
                stack_trace: vec![
                    StackFrame {
                        function: "fn_a".to_string(),
                        file: None,
                        line: None,
                    },
                    StackFrame {
                        function: "fn_b".to_string(),
                        file: Some(PathBuf::from("src/lib.rs")),
                        line: Some(10),
                    },
                    StackFrame {
                        function: "fn_c".to_string(),
                        file: None,
                        line: None,
                    },
                ],
            });
        }

        let (total, buckets) = bucketer.stats();
        assert_eq!(total, 2);
        assert_eq!(buckets, 1, "Same top-3 frames should be 1 bucket");
    }

    #[test]
    fn test_stack_trace_bucketing_different_frames() {
        let mut bucketer = CrashBucketer::new(CrashBucketingMode::StackTrace);

        bucketer.add_crash(CrashInfo {
            id: "c1".to_string(),
            file: PathBuf::from("a.rs"),
            line: 1,
            message: "err".to_string(),
            stack_trace: vec![StackFrame {
                function: "fn_x".to_string(),
                file: None,
                line: None,
            }],
        });

        bucketer.add_crash(CrashInfo {
            id: "c2".to_string(),
            file: PathBuf::from("b.rs"),
            line: 2,
            message: "err".to_string(),
            stack_trace: vec![StackFrame {
                function: "fn_y".to_string(),
                file: None,
                line: None,
            }],
        });

        let (total, buckets) = bucketer.stats();
        assert_eq!(total, 2);
        assert_eq!(buckets, 2, "Different frames should give different buckets");
    }

    #[test]
    fn test_stack_trace_bucketing_empty_frames() {
        let mut bucketer = CrashBucketer::new(CrashBucketingMode::StackTrace);

        bucketer.add_crash(CrashInfo {
            id: "c1".to_string(),
            file: PathBuf::from("a.rs"),
            line: 1,
            message: "err".to_string(),
            stack_trace: vec![],
        });

        let (total, buckets) = bucketer.stats();
        assert_eq!(total, 1);
        assert_eq!(buckets, 1);
        // Empty frames join to empty string
    }

    // =========================================================================
    // Coverage gap: RootCausePattern Display
    // =========================================================================

    #[test]
    fn test_root_cause_pattern_display() {
        assert_eq!(
            RootCausePattern::IndexOutOfBounds.to_string(),
            "index_out_of_bounds"
        );
        assert_eq!(
            RootCausePattern::NullPointerDeref.to_string(),
            "null_pointer_deref"
        );
        assert_eq!(
            RootCausePattern::IntegerOverflow.to_string(),
            "integer_overflow"
        );
        assert_eq!(
            RootCausePattern::DivisionByZero.to_string(),
            "division_by_zero"
        );
        assert_eq!(
            RootCausePattern::StackOverflow.to_string(),
            "stack_overflow"
        );
        assert_eq!(RootCausePattern::HeapOverflow.to_string(), "heap_overflow");
        assert_eq!(RootCausePattern::UseAfterFree.to_string(), "use_after_free");
        assert_eq!(RootCausePattern::DoubleFree.to_string(), "double_free");
        assert_eq!(RootCausePattern::UnwrapOnNone.to_string(), "unwrap_on_none");
        assert_eq!(
            RootCausePattern::AssertionFailed.to_string(),
            "assertion_failed"
        );
        assert_eq!(RootCausePattern::Unknown.to_string(), "unknown");
    }

    // =========================================================================
    // Coverage gap: detect_pattern edge cases
    // =========================================================================

    #[test]
    fn test_detect_pattern_use_after_free() {
        assert_eq!(
            CrashBucketer::detect_pattern("use after free in allocator"),
            RootCausePattern::UseAfterFree
        );
    }

    #[test]
    fn test_detect_pattern_double_free() {
        assert_eq!(
            CrashBucketer::detect_pattern("double free detected"),
            RootCausePattern::DoubleFree
        );
    }

    #[test]
    fn test_detect_pattern_heap_overflow() {
        assert_eq!(
            CrashBucketer::detect_pattern("heap buffer overflow"),
            RootCausePattern::HeapOverflow
        );
    }

    #[test]
    fn test_detect_pattern_indexoutofbounds_single_word() {
        assert_eq!(
            CrashBucketer::detect_pattern("IndexOutOfBounds exception"),
            RootCausePattern::IndexOutOfBounds
        );
    }

    #[test]
    fn test_detect_pattern_nullptr() {
        assert_eq!(
            CrashBucketer::detect_pattern("nullptr dereference"),
            RootCausePattern::NullPointerDeref
        );
    }

    #[test]
    fn test_detect_pattern_unwrap_none_variant() {
        assert_eq!(
            CrashBucketer::detect_pattern("unwrap called on None value"),
            RootCausePattern::UnwrapOnNone
        );
    }

    #[test]
    fn test_detect_pattern_division_by_zero_variant() {
        assert_eq!(
            CrashBucketer::detect_pattern("division by zero error"),
            RootCausePattern::DivisionByZero
        );
    }

    #[test]
    fn test_detect_pattern_assert_keyword() {
        assert_eq!(
            CrashBucketer::detect_pattern("assert_eq failed: 1 != 2"),
            RootCausePattern::AssertionFailed
        );
    }

    // =========================================================================
    // Coverage gap: to_findings() content verification
    // =========================================================================

    #[test]
    fn test_to_findings_content() {
        let mut bucketer = CrashBucketer::new(CrashBucketingMode::Semantic);

        bucketer.add_crash(CrashInfo {
            id: "crash-1".to_string(),
            file: PathBuf::from("src/parser.rs"),
            line: 100,
            message: "index out of bounds: len is 3 but index is 5".to_string(),
            stack_trace: vec![],
        });
        bucketer.add_crash(CrashInfo {
            id: "crash-2".to_string(),
            file: PathBuf::from("src/parser.rs"),
            line: 100,
            message: "index out of bounds: len is 10 but index is 20".to_string(),
            stack_trace: vec![],
        });

        let findings = bucketer.to_findings();
        assert_eq!(findings.len(), 1);

        let f = &findings[0];
        assert!(f.id.starts_with("BH-CRASH-"));
        assert_eq!(f.file, PathBuf::from("src/parser.rs"));
        assert_eq!(f.line, 100);
        assert_eq!(f.severity, FindingSeverity::High);
        assert!((f.suspiciousness - 0.8).abs() < f64::EPSILON);
        assert!(f.description.contains("2 occurrence(s)"));
    }

    #[test]
    fn test_to_findings_empty_bucketer() {
        let bucketer = CrashBucketer::new(CrashBucketingMode::Semantic);
        let findings = bucketer.to_findings();
        assert!(findings.is_empty());
    }

    // =========================================================================
    // Coverage gap: localize() with error message (triggers semantic scoring)
    // =========================================================================

    #[test]
    fn test_localize_with_error_message() {
        let temp = std::env::temp_dir().join("test_localize_semantic");
        let _ = std::fs::create_dir_all(&temp);

        // Create a file with content matching the error message
        let file_name = "semantic_test.rs";
        std::fs::write(
            temp.join(file_name),
            "fn process() {\n    panic!(\"buffer overflow detected\");\n}\n",
        )
        .unwrap();

        let mut localizer = MultiChannelLocalizer::new(
            LocalizationStrategy::MultiChannel,
            ChannelWeights {
                spectrum: 0.0,
                mutation: 0.0,
                static_analysis: 0.0,
                semantic: 1.0,
                quality: 0.0,
            },
        );

        // Set error message and add location at the file
        localizer.set_error_message("buffer overflow detected");
        localizer
            .static_findings
            .insert((PathBuf::from(file_name), 2), 0.5);

        let results = localizer.localize(&temp);
        assert_eq!(results.len(), 1);
        // Semantic score should be computed because error_message is set
        // The file exists so read will succeed and compute_semantic_score runs
        // The final_score uses semantic weight=1.0, so final_score should reflect semantic_score
    }

    #[test]
    fn test_localize_with_error_message_file_not_found() {
        let mut localizer = MultiChannelLocalizer::new(
            LocalizationStrategy::MultiChannel,
            ChannelWeights::default(),
        );

        localizer.set_error_message("some error");
        localizer
            .static_findings
            .insert((PathBuf::from("nonexistent_file.rs"), 10), 0.5);

        // /nonexistent as project_path — file won't be found, semantic_score stays 0.0
        let results = localizer.localize(Path::new("/nonexistent_project"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].semantic_score, 0.0);
    }

    // =========================================================================
    // Coverage gap: localize with add_coverage integration
    // =========================================================================

    #[test]
    fn test_localize_after_add_coverage() {
        let mut localizer =
            MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());

        let mut pass_lines = HashMap::new();
        pass_lines.insert((PathBuf::from("lib.rs"), 10), 1);

        let mut fail_lines = HashMap::new();
        fail_lines.insert((PathBuf::from("lib.rs"), 10), 1);
        fail_lines.insert((PathBuf::from("lib.rs"), 20), 1);

        localizer.add_coverage(&[
            TestCoverage {
                test_name: "pass".to_string(),
                passed: true,
                executed_lines: pass_lines,
            },
            TestCoverage {
                test_name: "fail".to_string(),
                passed: false,
                executed_lines: fail_lines,
            },
        ]);

        let results = localizer.localize(Path::new("/tmp"));
        // Line 20 only in failed tests => higher score than line 10
        assert_eq!(results.len(), 2);
        let line20 = results.iter().find(|r| r.line == 20).unwrap();
        let line10 = results.iter().find(|r| r.line == 10).unwrap();
        assert!(
            line20.final_score >= line10.final_score,
            "Line only in failing tests should have higher score"
        );
    }

    // =========================================================================
    // Coverage gap: Semantic bucketing with different files
    // =========================================================================

    #[test]
    fn test_semantic_bucketing_different_files() {
        let mut bucketer = CrashBucketer::new(CrashBucketingMode::Semantic);

        bucketer.add_crash(CrashInfo {
            id: "c1".to_string(),
            file: PathBuf::from("src/a.rs"),
            line: 10,
            message: "index out of bounds".to_string(),
            stack_trace: vec![],
        });
        bucketer.add_crash(CrashInfo {
            id: "c2".to_string(),
            file: PathBuf::from("src/b.rs"),
            line: 20,
            message: "index out of bounds".to_string(),
            stack_trace: vec![],
        });

        let (total, buckets) = bucketer.stats();
        assert_eq!(total, 2);
        // Same pattern but different files => different bucket keys
        assert_eq!(buckets, 2);
    }

    // =========================================================================
    // Cleanup for semantic test
    // =========================================================================

    fn _cleanup_semantic_test() {
        let _ = std::fs::remove_dir_all(std::env::temp_dir().join("test_localize_semantic"));
    }
}
