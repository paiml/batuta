//! Pipeline Library Analysis
//!
//! Provides analysis of Python ML library usage for conversion guidance.
//! Extracts NumPy, scikit-learn, and PyTorch operations to map to Rust equivalents.

use anyhow::Result;
use std::path::Path;

#[cfg(feature = "native")]
use tracing::info;

#[cfg(feature = "native")]
use walkdir::WalkDir;

use crate::numpy_converter::{NumPyConverter, NumPyOp};
use crate::pytorch_converter::{PyTorchConverter, PyTorchOperation};
use crate::sklearn_converter::{SklearnAlgorithm, SklearnConverter};

/// Analyzer for ML library usage in Python code
pub struct LibraryAnalyzer {
    numpy_converter: NumPyConverter,
    sklearn_converter: SklearnConverter,
    pytorch_converter: PyTorchConverter,
}

impl Default for LibraryAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl LibraryAnalyzer {
    /// Create a new library analyzer
    pub fn new() -> Self {
        Self {
            numpy_converter: NumPyConverter::new(),
            sklearn_converter: SklearnConverter::new(),
            pytorch_converter: PyTorchConverter::new(),
        }
    }

    /// Analyze Python source for NumPy usage and provide conversion guidance
    #[cfg(feature = "native")]
    pub fn analyze_numpy_usage(&self, input_path: &Path) -> Result<Vec<String>> {
        let converter = &self.numpy_converter;
        analyze_library(input_path, &["import numpy", "from numpy"], "NumPy", |path, content| {
            let operations = [
                ("np.add", NumPyOp::Add),
                ("np.subtract", NumPyOp::Subtract),
                ("np.multiply", NumPyOp::Multiply),
                ("np.dot", NumPyOp::Dot),
                ("np.sum", NumPyOp::Sum),
                ("np.array", NumPyOp::Array),
            ];
            operations.iter().filter_map(|(pattern, op)| {
                if content.contains(pattern) {
                    converter.convert(op).map(|r| format!("{}: {} → {}", path.display(), pattern, r.code_template))
                } else {
                    None
                }
            }).collect()
        })
    }

    /// Stub for WASM build
    #[cfg(not(feature = "native"))]
    pub fn analyze_numpy_usage(&self, _input_path: &Path) -> Result<Vec<String>> {
        Ok(Vec::new())
    }

    /// Analyze Python source for sklearn usage and provide conversion guidance
    #[cfg(feature = "native")]
    pub fn analyze_sklearn_usage(&self, input_path: &Path) -> Result<Vec<String>> {
        let converter = &self.sklearn_converter;
        analyze_library(input_path, &["import sklearn", "from sklearn"], "sklearn", |path, content| {
            let algorithms = [
                ("LinearRegression", SklearnAlgorithm::LinearRegression),
                ("LogisticRegression", SklearnAlgorithm::LogisticRegression),
                ("KMeans", SklearnAlgorithm::KMeans),
                ("DecisionTreeClassifier", SklearnAlgorithm::DecisionTreeClassifier),
                ("RandomForestClassifier", SklearnAlgorithm::RandomForestClassifier),
                ("StandardScaler", SklearnAlgorithm::StandardScaler),
                ("train_test_split", SklearnAlgorithm::TrainTestSplit),
            ];
            algorithms.iter().filter_map(|(pattern, alg)| {
                if content.contains(pattern) {
                    converter.convert(alg).map(|r| format!("{}: {} ({}) → {}", path.display(), pattern, alg.sklearn_module(), r.code_template))
                } else {
                    None
                }
            }).collect()
        })
    }

    /// Stub for WASM build
    #[cfg(not(feature = "native"))]
    pub fn analyze_sklearn_usage(&self, _input_path: &Path) -> Result<Vec<String>> {
        Ok(Vec::new())
    }

    /// Analyze Python source for PyTorch usage and provide conversion guidance
    #[cfg(feature = "native")]
    pub fn analyze_pytorch_usage(&self, input_path: &Path) -> Result<Vec<String>> {
        let converter = &self.pytorch_converter;
        analyze_library(input_path, &["import torch", "from torch", "from transformers"], "PyTorch", |path, content| {
            let operations = [
                ("torch.load", PyTorchOperation::LoadModel),
                ("from_pretrained", PyTorchOperation::LoadModel),
                ("AutoTokenizer", PyTorchOperation::LoadTokenizer),
                (".forward(", PyTorchOperation::Forward),
                (".generate(", PyTorchOperation::Generate),
                ("nn.Linear", PyTorchOperation::Linear),
                ("MultiheadAttention", PyTorchOperation::Attention),
                ("tokenizer.encode", PyTorchOperation::Encode),
                ("tokenizer.decode", PyTorchOperation::Decode),
            ];
            operations.iter().filter_map(|(pattern, op)| {
                if content.contains(pattern) {
                    converter.convert(op).map(|r| format!("{}: {} ({}) → {}", path.display(), pattern, op.pytorch_module(), r.code_template))
                } else {
                    None
                }
            }).collect()
        })
    }

    /// Stub for WASM build
    #[cfg(not(feature = "native"))]
    pub fn analyze_pytorch_usage(&self, _input_path: &Path) -> Result<Vec<String>> {
        Ok(Vec::new())
    }
}

/// Shared helper: walk Python files matching import patterns and apply conversion logic
#[cfg(feature = "native")]
fn analyze_library<F>(
    input_path: &Path,
    import_patterns: &[&str],
    lib_name: &str,
    match_content: F,
) -> Result<Vec<String>>
where
    F: Fn(&Path, &str) -> Vec<String>,
{
    let mut recommendations = Vec::new();
    for entry in WalkDir::new(input_path)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let Some(ext) = entry.path().extension() else { continue };
        if ext != "py" { continue; }
        let Ok(content) = std::fs::read_to_string(entry.path()) else { continue };
        if !import_patterns.iter().any(|p| content.contains(p)) { continue; }
        info!("  Found {} usage in: {}", lib_name, entry.path().display());
        recommendations.extend(match_content(entry.path(), &content));
    }
    Ok(recommendations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_library_analyzer_creation() {
        let analyzer = LibraryAnalyzer::new();
        // Just verify it can be created
        assert!(true, "LibraryAnalyzer created successfully");
        let _ = analyzer;
    }

    #[test]
    fn test_library_analyzer_default() {
        let analyzer = LibraryAnalyzer::default();
        let _ = analyzer;
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_analyze_numpy_nonexistent_path() {
        let analyzer = LibraryAnalyzer::new();
        let path = PathBuf::from("/nonexistent/path");
        // Should return empty vec for nonexistent path
        let result = analyzer.analyze_numpy_usage(&path);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_analyze_sklearn_nonexistent_path() {
        let analyzer = LibraryAnalyzer::new();
        let path = PathBuf::from("/nonexistent/path");
        let result = analyzer.analyze_sklearn_usage(&path);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_analyze_pytorch_nonexistent_path() {
        let analyzer = LibraryAnalyzer::new();
        let path = PathBuf::from("/nonexistent/path");
        let result = analyzer.analyze_pytorch_usage(&path);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }
}
