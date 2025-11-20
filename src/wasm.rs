//! WebAssembly API for Batuta
//!
//! Provides JavaScript-friendly interfaces for core Batuta functionality
//! that operates on in-memory code without file system access.
//!
//! # Features
//!
//! - **Language Detection**: Analyze code snippets to detect languages
//! - **Backend Selection**: Recommend optimal compute backend (SIMD/GPU)
//! - **NumPy Conversion**: Convert NumPy operations to Trueno
//! - **sklearn Conversion**: Convert sklearn algorithms to Aprender
//! - **PyTorch Conversion**: Convert PyTorch operations to Realizar
//! - **Code Analysis**: PARF pattern detection and analysis
//!
//! # Example (JavaScript)
//!
//! ```javascript
//! import init, { analyze_code, convert_numpy, backend_recommend } from './batuta.js';
//!
//! await init();
//!
//! // Detect language
//! const analysis = analyze_code("import numpy as np\nx = np.array([1, 2, 3])");
//! console.log(analysis.language); // "Python"
//!
//! // Convert NumPy to Trueno
//! const conversion = convert_numpy("np.add(a, b)");
//! console.log(conversion.rust_code);
//!
//! // Get backend recommendation
//! const backend = backend_recommend("matmul", 1024);
//! console.log(backend); // "SIMD" or "GPU"
//! ```

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "wasm")]
use crate::backend::{BackendSelector, OpComplexity};
#[cfg(feature = "wasm")]
use crate::numpy_converter::{NumPyConverter, NumPyOp};
#[cfg(feature = "wasm")]
use crate::pytorch_converter::{PyTorchConverter, PyTorchOperation};
#[cfg(feature = "wasm")]
use crate::sklearn_converter::{SklearnAlgorithm, SklearnConverter};

/// Initialize the WASM module (sets panic hook for better error messages)
#[cfg(feature = "wasm")]
#[wasm_bindgen(start)]
pub fn wasm_init() {
    // Set panic hook for better error messages in browser console
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    web_sys::console::log_1(&"Batuta WASM module initialized".into());
}

/// Analysis result for code snippets
#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[derive(Serialize, Deserialize)]
pub struct AnalysisResult {
    pub language: String,
    pub has_numpy: bool,
    pub has_sklearn: bool,
    pub has_pytorch: bool,
    pub lines_of_code: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl AnalysisResult {
    /// Get JSON representation
    #[wasm_bindgen(js_name = toJSON)]
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(self)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }
}

/// Conversion result for code transformations
#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[derive(Serialize, Deserialize)]
pub struct ConversionResult {
    pub original_code: String,
    pub rust_code: String,
    pub imports: String,
    pub backend_recommendation: String,
    pub complexity: String,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl ConversionResult {
    /// Get JSON representation
    #[wasm_bindgen(js_name = toJSON)]
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(self)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }
}

/// Analyze code snippet and detect language and dependencies
///
/// # Arguments
/// * `code` - Source code to analyze
///
/// # Returns
/// Analysis result with language detection and dependency info
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn analyze_code(code: &str) -> Result<AnalysisResult, JsValue> {
    let lines: Vec<&str> = code.lines().collect();

    // Simple language detection
    let language = if code.contains("import ") || code.contains("def ") || code.contains("class ") {
        "Python"
    } else if code.contains("#include") || code.contains("int main") {
        "C/C++"
    } else if code.contains("fn ") || code.contains("struct ") {
        "Rust"
    } else if code.contains("#!/bin/bash") || code.contains("#!/bin/sh") {
        "Shell"
    } else {
        "Unknown"
    };

    // Detect ML libraries
    let has_numpy = code.contains("numpy") || code.contains("np.");
    let has_sklearn = code.contains("sklearn");
    let has_pytorch = code.contains("torch") || code.contains("transformers");

    Ok(AnalysisResult {
        language: language.to_string(),
        has_numpy,
        has_sklearn,
        has_pytorch,
        lines_of_code: lines.len(),
    })
}

/// Convert NumPy operation to Trueno Rust code
///
/// # Arguments
/// * `numpy_code` - NumPy operation string (e.g., "np.add(a, b)")
/// * `data_size` - Optional data size for backend recommendation
///
/// # Returns
/// Conversion result with Rust code and backend recommendation
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn convert_numpy(numpy_code: &str, data_size: Option<usize>) -> Result<ConversionResult, JsValue> {
    let converter = NumPyConverter::new();

    // Simple pattern matching for common NumPy operations
    let op = if numpy_code.contains("np.add") || numpy_code.contains("numpy.add") {
        NumPyOp::Add
    } else if numpy_code.contains("np.dot") || numpy_code.contains("numpy.dot") {
        NumPyOp::Dot
    } else if numpy_code.contains("np.sum") || numpy_code.contains("numpy.sum") {
        NumPyOp::Sum
    } else if numpy_code.contains("np.mean") || numpy_code.contains("numpy.mean") {
        NumPyOp::Mean
    } else if numpy_code.contains("np.array") || numpy_code.contains("numpy.array") {
        NumPyOp::Array
    } else if numpy_code.contains("reshape") {
        NumPyOp::Reshape
    } else if numpy_code.contains("transpose") || numpy_code.contains(".T") {
        NumPyOp::Transpose
    } else {
        return Err(JsValue::from_str("Unsupported NumPy operation"));
    };

    let trueno_op = converter
        .convert(&op)
        .ok_or_else(|| JsValue::from_str("Conversion failed: operation not supported"))?;

    let size = data_size.unwrap_or(1000);
    let backend = converter.recommend_backend(&op, size);

    Ok(ConversionResult {
        original_code: numpy_code.to_string(),
        rust_code: trueno_op.code_template.clone(),
        imports: trueno_op.imports.join("\n"),
        backend_recommendation: format!("{:?}", backend),
        complexity: format!("{:?}", trueno_op.complexity),
    })
}

/// Convert sklearn algorithm to Aprender Rust code
///
/// # Arguments
/// * `sklearn_code` - sklearn algorithm string (e.g., "LinearRegression()")
/// * `data_size` - Optional data size for backend recommendation
///
/// # Returns
/// Conversion result with Rust code and backend recommendation
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn convert_sklearn(sklearn_code: &str, data_size: Option<usize>) -> Result<ConversionResult, JsValue> {
    let converter = SklearnConverter::new();

    // Simple pattern matching for common sklearn algorithms
    let algo = if sklearn_code.contains("LinearRegression") {
        SklearnAlgorithm::LinearRegression
    } else if sklearn_code.contains("LogisticRegression") {
        SklearnAlgorithm::LogisticRegression
    } else if sklearn_code.contains("KMeans") {
        SklearnAlgorithm::KMeans
    } else if sklearn_code.contains("DecisionTreeClassifier") {
        SklearnAlgorithm::DecisionTreeClassifier
    } else if sklearn_code.contains("RandomForestClassifier") {
        SklearnAlgorithm::RandomForestClassifier
    } else if sklearn_code.contains("StandardScaler") {
        SklearnAlgorithm::StandardScaler
    } else {
        return Err(JsValue::from_str("Unsupported sklearn algorithm"));
    };

    let aprender_algo = converter
        .convert(&algo)
        .ok_or_else(|| JsValue::from_str("Conversion failed: algorithm not supported"))?;

    let size = data_size.unwrap_or(1000);
    let backend = converter.recommend_backend(&algo, size);

    Ok(ConversionResult {
        original_code: sklearn_code.to_string(),
        rust_code: aprender_algo.code_template.clone(),
        imports: aprender_algo.imports.join("\n"),
        backend_recommendation: format!("{:?}", backend),
        complexity: format!("{:?}", aprender_algo.complexity),
    })
}

/// Convert PyTorch operation to Realizar Rust code
///
/// # Arguments
/// * `pytorch_code` - PyTorch operation string (e.g., "model.generate()")
/// * `data_size` - Optional data size for backend recommendation
///
/// # Returns
/// Conversion result with Rust code and backend recommendation
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn convert_pytorch(pytorch_code: &str, data_size: Option<usize>) -> Result<ConversionResult, JsValue> {
    let converter = PyTorchConverter::new();

    // Simple pattern matching for common PyTorch operations
    let op = if pytorch_code.contains("load") && pytorch_code.contains("model") {
        PyTorchOperation::LoadModel
    } else if pytorch_code.contains("generate") {
        PyTorchOperation::Generate
    } else if pytorch_code.contains("forward") {
        PyTorchOperation::Forward
    } else if pytorch_code.contains("encode") {
        PyTorchOperation::Encode
    } else if pytorch_code.contains("decode") {
        PyTorchOperation::Decode
    } else if pytorch_code.contains("nn.Linear") {
        PyTorchOperation::Linear
    } else if pytorch_code.contains("Attention") {
        PyTorchOperation::Attention
    } else {
        return Err(JsValue::from_str("Unsupported PyTorch operation"));
    };

    let realizar_op = converter
        .convert(&op)
        .ok_or_else(|| JsValue::from_str("Conversion failed: operation not supported"))?;

    let size = data_size.unwrap_or(1000000); // LLMs are large by default
    let backend = converter.recommend_backend(&op, size);

    Ok(ConversionResult {
        original_code: pytorch_code.to_string(),
        rust_code: realizar_op.code_template.clone(),
        imports: realizar_op.imports.join("\n"),
        backend_recommendation: format!("{:?}", backend),
        complexity: format!("{:?}", realizar_op.complexity),
    })
}

/// Get backend recommendation for an operation
///
/// # Arguments
/// * `operation_type` - Type of operation ("element-wise", "reduction", "matmul")
/// * `data_size` - Size of data to process
///
/// # Returns
/// Recommended backend as string ("Scalar", "SIMD", "GPU")
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn backend_recommend(operation_type: &str, data_size: usize) -> Result<String, JsValue> {
    let selector = BackendSelector::new();

    let backend = match operation_type.to_lowercase().as_str() {
        "element-wise" | "elementwise" => selector.select_for_elementwise(data_size),
        "reduction" | "reduce" => {
            let complexity = if data_size < 10_000 {
                OpComplexity::Low
            } else if data_size < 100_000 {
                OpComplexity::Medium
            } else {
                OpComplexity::High
            };
            selector.select_with_moe(complexity, data_size)
        }
        "matmul" | "matrix-multiply" => {
            let n = (data_size as f64).sqrt() as usize;
            selector.select_for_matmul(n, n, n)
        }
        _ => return Err(JsValue::from_str("Unknown operation type")),
    };

    Ok(format!("{:?}", backend))
}

/// Get Batuta version
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_python_code() {
        let code = "import numpy as np\nx = np.array([1, 2, 3])";
        let result = analyze_code(code).unwrap();

        assert_eq!(result.language, "Python");
        assert!(result.has_numpy);
        assert!(!result.has_sklearn);
        assert!(!result.has_pytorch);
        assert_eq!(result.lines_of_code, 2);
    }

    #[test]
    fn test_convert_numpy_add() {
        let result = convert_numpy("np.add(a, b)", Some(1000)).unwrap();

        assert!(result.rust_code.contains("add"));
        assert!(!result.imports.is_empty());
        assert!(!result.backend_recommendation.is_empty());
    }

    #[test]
    fn test_backend_recommendation() {
        let backend = backend_recommend("matmul", 1024).unwrap();
        assert!(!backend.is_empty());
    }

    #[test]
    fn test_version() {
        let ver = version();
        assert!(!ver.is_empty());
    }
}
