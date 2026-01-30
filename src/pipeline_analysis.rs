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
        let mut recommendations = Vec::new();

        // Walk Python files looking for NumPy imports
        for entry in WalkDir::new(input_path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if let Some(ext) = entry.path().extension() {
                if ext == "py" {
                    // Read file and check for numpy imports
                    if let Ok(content) = std::fs::read_to_string(entry.path()) {
                        if content.contains("import numpy") || content.contains("from numpy") {
                            info!("  Found NumPy usage in: {}", entry.path().display());

                            // Analyze common NumPy operations
                            let operations = vec![
                                ("np.add", NumPyOp::Add),
                                ("np.subtract", NumPyOp::Subtract),
                                ("np.multiply", NumPyOp::Multiply),
                                ("np.dot", NumPyOp::Dot),
                                ("np.sum", NumPyOp::Sum),
                                ("np.array", NumPyOp::Array),
                            ];

                            for (pattern, op) in operations {
                                if content.contains(pattern) {
                                    if let Some(trueno_op) = self.numpy_converter.convert(&op) {
                                        recommendations.push(format!(
                                            "{}: {} → {}",
                                            entry.path().display(),
                                            pattern,
                                            trueno_op.code_template
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(recommendations)
    }

    /// Stub for WASM build
    #[cfg(not(feature = "native"))]
    pub fn analyze_numpy_usage(&self, _input_path: &Path) -> Result<Vec<String>> {
        Ok(Vec::new())
    }

    /// Analyze Python source for sklearn usage and provide conversion guidance
    #[cfg(feature = "native")]
    pub fn analyze_sklearn_usage(&self, input_path: &Path) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();

        // Walk Python files looking for sklearn imports
        for entry in WalkDir::new(input_path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if let Some(ext) = entry.path().extension() {
                if ext == "py" {
                    // Read file and check for sklearn imports
                    if let Ok(content) = std::fs::read_to_string(entry.path()) {
                        if content.contains("import sklearn") || content.contains("from sklearn") {
                            info!("  Found sklearn usage in: {}", entry.path().display());

                            // Analyze common sklearn algorithms
                            let algorithms = vec![
                                ("LinearRegression", SklearnAlgorithm::LinearRegression),
                                ("LogisticRegression", SklearnAlgorithm::LogisticRegression),
                                ("KMeans", SklearnAlgorithm::KMeans),
                                (
                                    "DecisionTreeClassifier",
                                    SklearnAlgorithm::DecisionTreeClassifier,
                                ),
                                (
                                    "RandomForestClassifier",
                                    SklearnAlgorithm::RandomForestClassifier,
                                ),
                                ("StandardScaler", SklearnAlgorithm::StandardScaler),
                                ("train_test_split", SklearnAlgorithm::TrainTestSplit),
                            ];

                            for (pattern, alg) in algorithms {
                                if content.contains(pattern) {
                                    if let Some(aprender_alg) = self.sklearn_converter.convert(&alg)
                                    {
                                        recommendations.push(format!(
                                            "{}: {} ({}) → {}",
                                            entry.path().display(),
                                            pattern,
                                            alg.sklearn_module(),
                                            aprender_alg.code_template
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(recommendations)
    }

    /// Stub for WASM build
    #[cfg(not(feature = "native"))]
    pub fn analyze_sklearn_usage(&self, _input_path: &Path) -> Result<Vec<String>> {
        Ok(Vec::new())
    }

    /// Analyze Python source for PyTorch usage and provide conversion guidance
    #[cfg(feature = "native")]
    pub fn analyze_pytorch_usage(&self, input_path: &Path) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();

        // Walk Python files looking for PyTorch/transformers imports
        for entry in WalkDir::new(input_path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if let Some(ext) = entry.path().extension() {
                if ext == "py" {
                    // Read file and check for PyTorch imports
                    if let Ok(content) = std::fs::read_to_string(entry.path()) {
                        if content.contains("import torch")
                            || content.contains("from torch")
                            || content.contains("from transformers")
                        {
                            info!("  Found PyTorch usage in: {}", entry.path().display());

                            // Analyze common PyTorch operations
                            let operations = vec![
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

                            for (pattern, op) in operations {
                                if content.contains(pattern) {
                                    if let Some(realizar_op) = self.pytorch_converter.convert(&op) {
                                        recommendations.push(format!(
                                            "{}: {} ({}) → {}",
                                            entry.path().display(),
                                            pattern,
                                            op.pytorch_module(),
                                            realizar_op.code_template
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(recommendations)
    }

    /// Stub for WASM build
    #[cfg(not(feature = "native"))]
    pub fn analyze_pytorch_usage(&self, _input_path: &Path) -> Result<Vec<String>> {
        Ok(Vec::new())
    }
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
