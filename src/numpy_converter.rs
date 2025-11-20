//! NumPy to Trueno conversion module (BATUTA-008)
//!
//! Converts Python NumPy operations to Rust Trueno operations with
//! automatic backend selection via MoE routing.
//!
//! # Conversion Strategy
//!
//! NumPy operations are mapped to equivalent Trueno operations:
//! - `np.array(...)` → `Vector::from_slice(...)` or `Matrix::from_slice(...)`
//! - `np.add(a, b)` → `a.add(&b)`
//! - `np.dot(a, b)` → `a.dot(&b)` or `a.matmul(&b)`
//! - `np.sum(a)` → `a.sum()`
//! - Element-wise ops automatically use MoE routing
//!
//! # Example
//!
//! ```python
//! # Python NumPy code
//! import numpy as np
//! a = np.array([1.0, 2.0, 3.0])
//! b = np.array([4.0, 5.0, 6.0])
//! c = np.add(a, b)
//! ```
//!
//! Converts to:
//!
//! ```rust,ignore
//! use trueno::Vector;
//! let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
//! let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
//! let c = a.add(&b).unwrap();
//! ```

use std::collections::HashMap;

/// NumPy operation types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NumPyOp {
    /// Array creation: np.array, np.zeros, np.ones
    Array,
    /// Element-wise addition: np.add, a + b
    Add,
    /// Element-wise subtraction: np.subtract, a - b
    Subtract,
    /// Element-wise multiplication: np.multiply, a * b
    Multiply,
    /// Element-wise division: np.divide, a / b
    Divide,
    /// Dot product / matrix multiply: np.dot, np.matmul, a @ b
    Dot,
    /// Sum reduction: np.sum
    Sum,
    /// Mean reduction: np.mean
    Mean,
    /// Max reduction: np.max
    Max,
    /// Min reduction: np.min
    Min,
    /// Reshape: np.reshape
    Reshape,
    /// Transpose: np.transpose, a.T
    Transpose,
}

impl NumPyOp {
    /// Get the operation complexity for MoE routing
    pub fn complexity(&self) -> crate::backend::OpComplexity {
        use crate::backend::OpComplexity;
        
        match self {
            // Element-wise operations are Low complexity (memory-bound)
            NumPyOp::Add | NumPyOp::Subtract | NumPyOp::Multiply | NumPyOp::Divide => {
                OpComplexity::Low
            }
            // Reductions are Medium complexity
            NumPyOp::Sum | NumPyOp::Mean | NumPyOp::Max | NumPyOp::Min => {
                OpComplexity::Medium
            }
            // Dot product and matrix ops are High complexity
            NumPyOp::Dot => OpComplexity::High,
            // Structural operations don't need backend selection
            NumPyOp::Array | NumPyOp::Reshape | NumPyOp::Transpose => OpComplexity::Low,
        }
    }
}

/// Trueno equivalent operation
#[derive(Debug, Clone)]
pub struct TruenoOp {
    /// Rust code template for the operation
    pub code_template: String,
    /// Required imports
    pub imports: Vec<String>,
    /// Operation complexity
    pub complexity: crate::backend::OpComplexity,
}

/// NumPy to Trueno converter
pub struct NumPyConverter {
    /// Operation mapping
    op_map: HashMap<NumPyOp, TruenoOp>,
    /// Backend selector for MoE routing
    backend_selector: crate::backend::BackendSelector,
}

impl Default for NumPyConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl NumPyConverter {
    /// Create a new NumPy converter with default mappings
    pub fn new() -> Self {
        let mut op_map = HashMap::new();

        // Array operations
        op_map.insert(
            NumPyOp::Array,
            TruenoOp {
                code_template: "Vector::from_slice(&[{values}])".to_string(),
                imports: vec!["use trueno::Vector;".to_string()],
                complexity: crate::backend::OpComplexity::Low,
            },
        );

        // Element-wise operations
        op_map.insert(
            NumPyOp::Add,
            TruenoOp {
                code_template: "{lhs}.add(&{rhs}).unwrap()".to_string(),
                imports: vec!["use trueno::Vector;".to_string()],
                complexity: crate::backend::OpComplexity::Low,
            },
        );

        op_map.insert(
            NumPyOp::Subtract,
            TruenoOp {
                code_template: "{lhs}.sub(&{rhs}).unwrap()".to_string(),
                imports: vec!["use trueno::Vector;".to_string()],
                complexity: crate::backend::OpComplexity::Low,
            },
        );

        op_map.insert(
            NumPyOp::Multiply,
            TruenoOp {
                code_template: "{lhs}.mul(&{rhs}).unwrap()".to_string(),
                imports: vec!["use trueno::Vector;".to_string()],
                complexity: crate::backend::OpComplexity::Low,
            },
        );

        // Reductions
        op_map.insert(
            NumPyOp::Sum,
            TruenoOp {
                code_template: "{array}.sum()".to_string(),
                imports: vec!["use trueno::Vector;".to_string()],
                complexity: crate::backend::OpComplexity::Medium,
            },
        );

        op_map.insert(
            NumPyOp::Dot,
            TruenoOp {
                code_template: "{lhs}.dot(&{rhs}).unwrap()".to_string(),
                imports: vec!["use trueno::Vector;".to_string()],
                complexity: crate::backend::OpComplexity::High,
            },
        );

        Self {
            op_map,
            backend_selector: crate::backend::BackendSelector::new(),
        }
    }

    /// Convert a NumPy operation to Trueno
    pub fn convert(&self, op: &NumPyOp) -> Option<&TruenoOp> {
        self.op_map.get(op)
    }

    /// Get recommended backend for an operation
    pub fn recommend_backend(&self, op: &NumPyOp, data_size: usize) -> crate::backend::Backend {
        self.backend_selector.select_with_moe(op.complexity(), data_size)
    }

    /// Get all available conversions
    pub fn available_ops(&self) -> Vec<&NumPyOp> {
        self.op_map.keys().collect()
    }

    /// Generate conversion report
    pub fn conversion_report(&self) -> String {
        let mut report = String::from("NumPy → Trueno Conversion Map\n");
        report.push_str("================================\n\n");

        for (op, trueno_op) in &self.op_map {
            report.push_str(&format!("{:?}:\n", op));
            report.push_str(&format!("  Complexity: {:?}\n", trueno_op.complexity));
            report.push_str(&format!("  Template: {}\n", trueno_op.code_template));
            report.push_str(&format!("  Imports: {}\n\n", trueno_op.imports.join(", ")));
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_converter_creation() {
        let converter = NumPyConverter::new();
        assert!(!converter.available_ops().is_empty());
    }

    #[test]
    fn test_operation_complexity() {
        assert_eq!(NumPyOp::Add.complexity(), crate::backend::OpComplexity::Low);
        assert_eq!(NumPyOp::Sum.complexity(), crate::backend::OpComplexity::Medium);
        assert_eq!(NumPyOp::Dot.complexity(), crate::backend::OpComplexity::High);
    }

    #[test]
    fn test_add_conversion() {
        let converter = NumPyConverter::new();
        let trueno_op = converter.convert(&NumPyOp::Add).unwrap();
        assert!(trueno_op.code_template.contains("add"));
        assert!(trueno_op.imports.iter().any(|i| i.contains("Vector")));
    }

    #[test]
    fn test_backend_recommendation() {
        let converter = NumPyConverter::new();
        
        // Small element-wise operation should use Scalar
        let backend = converter.recommend_backend(&NumPyOp::Add, 100);
        assert_eq!(backend, crate::backend::Backend::Scalar);

        // Large element-wise should use SIMD
        let backend = converter.recommend_backend(&NumPyOp::Add, 2_000_000);
        assert_eq!(backend, crate::backend::Backend::SIMD);

        // Large matrix operation should use GPU
        let backend = converter.recommend_backend(&NumPyOp::Dot, 50_000);
        assert_eq!(backend, crate::backend::Backend::GPU);
    }

    #[test]
    fn test_conversion_report() {
        let converter = NumPyConverter::new();
        let report = converter.conversion_report();
        assert!(report.contains("NumPy → Trueno"));
        assert!(report.contains("Add"));
        assert!(report.contains("Complexity"));
    }
}
