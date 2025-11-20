//! sklearn to Aprender conversion module (BATUTA-009)
//!
//! Converts Python scikit-learn (sklearn) algorithms to Rust Aprender equivalents
//! with automatic backend selection and ergonomic API mapping.
//!
//! # Conversion Strategy
//!
//! sklearn algorithms are mapped to equivalent Aprender algorithms:
//! - `LinearRegression()` → `LinearRegression::new()`
//! - `KMeans(n_clusters=3)` → `KMeans::new(3)`
//! - `DecisionTreeClassifier()` → `DecisionTreeClassifier::new()`
//! - `train_test_split()` → `train_test_split()`
//! - Model methods automatically use MoE routing for optimal performance
//!
//! # Example
//!
//! ```python
//! # Python sklearn code
//! from sklearn.linear_model import LinearRegression
//! from sklearn.model_selection import train_test_split
//!
//! X_train, X_test, y_train, y_test = train_test_split(X, y)
//! model = LinearRegression()
//! model.fit(X_train, y_train)
//! predictions = model.predict(X_test)
//! ```
//!
//! Converts to:
//!
//! ```rust,ignore
//! use aprender::linear_model::LinearRegression;
//! use aprender::model_selection::train_test_split;
//! use aprender::Estimator;
//!
//! let (X_train, X_test, y_train, y_test) = train_test_split(&X, &y, 0.25)?;
//! let mut model = LinearRegression::new();
//! model.fit(&X_train, &y_train)?;
//! let predictions = model.predict(&X_test)?;
//! ```

use std::collections::HashMap;

/// sklearn algorithm types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SklearnAlgorithm {
    // Linear Models
    LinearRegression,
    Ridge,
    Lasso,
    LogisticRegression,

    // Clustering
    KMeans,
    DBSCAN,

    // Tree Models
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
    RandomForestRegressor,

    // Preprocessing
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,

    // Model Selection
    TrainTestSplit,
    CrossValidation,

    // Metrics
    Accuracy,
    Precision,
    Recall,
    F1Score,
    MeanSquaredError,
    R2Score,
}

impl SklearnAlgorithm {
    /// Get the computational complexity for MoE routing
    pub fn complexity(&self) -> crate::backend::OpComplexity {
        use crate::backend::OpComplexity;

        match self {
            // Preprocessing operations are Low complexity
            SklearnAlgorithm::StandardScaler
            | SklearnAlgorithm::MinMaxScaler
            | SklearnAlgorithm::LabelEncoder
            | SklearnAlgorithm::TrainTestSplit => OpComplexity::Low,

            // Linear models and metrics are Medium complexity
            SklearnAlgorithm::LinearRegression
            | SklearnAlgorithm::Ridge
            | SklearnAlgorithm::Lasso
            | SklearnAlgorithm::LogisticRegression
            | SklearnAlgorithm::Accuracy
            | SklearnAlgorithm::Precision
            | SklearnAlgorithm::Recall
            | SklearnAlgorithm::F1Score
            | SklearnAlgorithm::MeanSquaredError
            | SklearnAlgorithm::R2Score => OpComplexity::Medium,

            // Tree models, ensemble methods, and clustering are High complexity
            SklearnAlgorithm::DecisionTreeClassifier
            | SklearnAlgorithm::DecisionTreeRegressor
            | SklearnAlgorithm::RandomForestClassifier
            | SklearnAlgorithm::RandomForestRegressor
            | SklearnAlgorithm::KMeans
            | SklearnAlgorithm::DBSCAN
            | SklearnAlgorithm::CrossValidation => OpComplexity::High,
        }
    }

    /// Get the sklearn module path
    pub fn sklearn_module(&self) -> &str {
        match self {
            SklearnAlgorithm::LinearRegression
            | SklearnAlgorithm::Ridge
            | SklearnAlgorithm::Lasso
            | SklearnAlgorithm::LogisticRegression => "sklearn.linear_model",

            SklearnAlgorithm::KMeans | SklearnAlgorithm::DBSCAN => "sklearn.cluster",

            SklearnAlgorithm::DecisionTreeClassifier
            | SklearnAlgorithm::DecisionTreeRegressor => "sklearn.tree",

            SklearnAlgorithm::RandomForestClassifier
            | SklearnAlgorithm::RandomForestRegressor => "sklearn.ensemble",

            SklearnAlgorithm::StandardScaler
            | SklearnAlgorithm::MinMaxScaler
            | SklearnAlgorithm::LabelEncoder => "sklearn.preprocessing",

            SklearnAlgorithm::TrainTestSplit | SklearnAlgorithm::CrossValidation => {
                "sklearn.model_selection"
            }

            SklearnAlgorithm::Accuracy
            | SklearnAlgorithm::Precision
            | SklearnAlgorithm::Recall
            | SklearnAlgorithm::F1Score
            | SklearnAlgorithm::MeanSquaredError
            | SklearnAlgorithm::R2Score => "sklearn.metrics",
        }
    }
}

/// Aprender equivalent algorithm
#[derive(Debug, Clone)]
pub struct AprenderAlgorithm {
    /// Rust code template for the algorithm
    pub code_template: String,
    /// Required imports
    pub imports: Vec<String>,
    /// Computational complexity
    pub complexity: crate::backend::OpComplexity,
    /// Typical usage pattern
    pub usage_pattern: String,
}

/// sklearn to Aprender converter
pub struct SklearnConverter {
    /// Algorithm mapping
    algorithm_map: HashMap<SklearnAlgorithm, AprenderAlgorithm>,
    /// Backend selector for MoE routing
    backend_selector: crate::backend::BackendSelector,
}

impl Default for SklearnConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl SklearnConverter {
    /// Create a new sklearn converter with default mappings
    pub fn new() -> Self {
        let mut algorithm_map = HashMap::new();

        // Linear Models
        algorithm_map.insert(
            SklearnAlgorithm::LinearRegression,
            AprenderAlgorithm {
                code_template: "LinearRegression::new()".to_string(),
                imports: vec![
                    "use aprender::linear_model::LinearRegression;".to_string(),
                    "use aprender::Estimator;".to_string(),
                ],
                complexity: crate::backend::OpComplexity::Medium,
                usage_pattern: "let mut model = LinearRegression::new();\nmodel.fit(&X_train, &y_train)?;\nlet predictions = model.predict(&X_test)?;".to_string(),
            },
        );

        algorithm_map.insert(
            SklearnAlgorithm::LogisticRegression,
            AprenderAlgorithm {
                code_template: "LogisticRegression::new()".to_string(),
                imports: vec![
                    "use aprender::classification::LogisticRegression;".to_string(),
                    "use aprender::Estimator;".to_string(),
                ],
                complexity: crate::backend::OpComplexity::Medium,
                usage_pattern: "let mut model = LogisticRegression::new();\nmodel.fit(&X_train, &y_train)?;\nlet predictions = model.predict(&X_test)?;".to_string(),
            },
        );

        // Clustering
        algorithm_map.insert(
            SklearnAlgorithm::KMeans,
            AprenderAlgorithm {
                code_template: "KMeans::new({n_clusters})".to_string(),
                imports: vec![
                    "use aprender::cluster::KMeans;".to_string(),
                    "use aprender::UnsupervisedEstimator;".to_string(),
                ],
                complexity: crate::backend::OpComplexity::High,
                usage_pattern: "let mut model = KMeans::new(3);\nmodel.fit(&X)?;\nlet labels = model.predict(&X)?;".to_string(),
            },
        );

        // Tree Models
        algorithm_map.insert(
            SklearnAlgorithm::DecisionTreeClassifier,
            AprenderAlgorithm {
                code_template: "DecisionTreeClassifier::new()".to_string(),
                imports: vec![
                    "use aprender::tree::DecisionTreeClassifier;".to_string(),
                    "use aprender::Estimator;".to_string(),
                ],
                complexity: crate::backend::OpComplexity::High,
                usage_pattern: "let mut model = DecisionTreeClassifier::new();\nmodel.fit(&X_train, &y_train)?;\nlet predictions = model.predict(&X_test)?;".to_string(),
            },
        );

        // Preprocessing
        algorithm_map.insert(
            SklearnAlgorithm::StandardScaler,
            AprenderAlgorithm {
                code_template: "StandardScaler::new()".to_string(),
                imports: vec![
                    "use aprender::preprocessing::StandardScaler;".to_string(),
                    "use aprender::Transformer;".to_string(),
                ],
                complexity: crate::backend::OpComplexity::Low,
                usage_pattern: "let mut scaler = StandardScaler::new();\nscaler.fit(&X_train)?;\nlet X_train_scaled = scaler.transform(&X_train)?;".to_string(),
            },
        );

        // Model Selection
        algorithm_map.insert(
            SklearnAlgorithm::TrainTestSplit,
            AprenderAlgorithm {
                code_template: "train_test_split(&X, &y, {test_size})".to_string(),
                imports: vec!["use aprender::model_selection::train_test_split;".to_string()],
                complexity: crate::backend::OpComplexity::Low,
                usage_pattern: "let (X_train, X_test, y_train, y_test) = train_test_split(&X, &y, 0.25)?;".to_string(),
            },
        );

        // Metrics
        algorithm_map.insert(
            SklearnAlgorithm::Accuracy,
            AprenderAlgorithm {
                code_template: "accuracy_score(&y_true, &y_pred)".to_string(),
                imports: vec!["use aprender::metrics::accuracy_score;".to_string()],
                complexity: crate::backend::OpComplexity::Medium,
                usage_pattern: "let acc = accuracy_score(&y_true, &y_pred)?;".to_string(),
            },
        );

        algorithm_map.insert(
            SklearnAlgorithm::MeanSquaredError,
            AprenderAlgorithm {
                code_template: "mean_squared_error(&y_true, &y_pred)".to_string(),
                imports: vec!["use aprender::metrics::mean_squared_error;".to_string()],
                complexity: crate::backend::OpComplexity::Medium,
                usage_pattern: "let mse = mean_squared_error(&y_true, &y_pred)?;".to_string(),
            },
        );

        Self {
            algorithm_map,
            backend_selector: crate::backend::BackendSelector::new(),
        }
    }

    /// Convert a sklearn algorithm to Aprender
    pub fn convert(&self, algorithm: &SklearnAlgorithm) -> Option<&AprenderAlgorithm> {
        self.algorithm_map.get(algorithm)
    }

    /// Get recommended backend for an algorithm
    pub fn recommend_backend(
        &self,
        algorithm: &SklearnAlgorithm,
        data_size: usize,
    ) -> crate::backend::Backend {
        self.backend_selector
            .select_with_moe(algorithm.complexity(), data_size)
    }

    /// Get all available conversions
    pub fn available_algorithms(&self) -> Vec<&SklearnAlgorithm> {
        self.algorithm_map.keys().collect()
    }

    /// Generate conversion report
    pub fn conversion_report(&self) -> String {
        let mut report = String::from("sklearn → Aprender Conversion Map\n");
        report.push_str("===================================\n\n");

        // Group by module
        let mut by_module: HashMap<&str, Vec<(&SklearnAlgorithm, &AprenderAlgorithm)>> =
            HashMap::new();

        for (alg, aprender_alg) in &self.algorithm_map {
            by_module
                .entry(alg.sklearn_module())
                .or_default()
                .push((alg, aprender_alg));
        }

        for (module, algorithms) in by_module.iter() {
            report.push_str(&format!("## {}\n\n", module));

            for (alg, aprender_alg) in algorithms {
                report.push_str(&format!("{:?}:\n", alg));
                report.push_str(&format!("  Template: {}\n", aprender_alg.code_template));
                report.push_str(&format!("  Complexity: {:?}\n", aprender_alg.complexity));
                report.push_str(&format!("  Imports: {}\n", aprender_alg.imports.join(", ")));
                report.push_str(&format!("  Usage:\n    {}\n\n",
                    aprender_alg.usage_pattern.replace('\n', "\n    ")));
            }
            report.push('\n');
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_converter_creation() {
        let converter = SklearnConverter::new();
        assert!(!converter.available_algorithms().is_empty());
    }

    #[test]
    fn test_algorithm_complexity() {
        assert_eq!(
            SklearnAlgorithm::LinearRegression.complexity(),
            crate::backend::OpComplexity::Medium
        );
        assert_eq!(
            SklearnAlgorithm::StandardScaler.complexity(),
            crate::backend::OpComplexity::Low
        );
        assert_eq!(
            SklearnAlgorithm::KMeans.complexity(),
            crate::backend::OpComplexity::High
        );
    }

    #[test]
    fn test_linear_regression_conversion() {
        let converter = SklearnConverter::new();
        let aprender_alg = converter
            .convert(&SklearnAlgorithm::LinearRegression)
            .unwrap();
        assert!(aprender_alg.code_template.contains("LinearRegression"));
        assert!(aprender_alg
            .imports
            .iter()
            .any(|i| i.contains("linear_model")));
    }

    #[test]
    fn test_kmeans_conversion() {
        let converter = SklearnConverter::new();
        let aprender_alg = converter.convert(&SklearnAlgorithm::KMeans).unwrap();
        assert!(aprender_alg.code_template.contains("KMeans"));
        assert!(aprender_alg.imports.iter().any(|i| i.contains("cluster")));
    }

    #[test]
    fn test_backend_recommendation() {
        let converter = SklearnConverter::new();

        // Small dataset with preprocessing should use Scalar
        let backend = converter.recommend_backend(&SklearnAlgorithm::StandardScaler, 100);
        assert_eq!(backend, crate::backend::Backend::Scalar);

        // Large dataset with linear model should use SIMD
        let backend = converter.recommend_backend(&SklearnAlgorithm::LinearRegression, 50_000);
        assert_eq!(backend, crate::backend::Backend::SIMD);

        // Large dataset with clustering should use GPU
        let backend = converter.recommend_backend(&SklearnAlgorithm::KMeans, 100_000);
        assert_eq!(backend, crate::backend::Backend::GPU);
    }

    #[test]
    fn test_sklearn_module_paths() {
        assert_eq!(
            SklearnAlgorithm::LinearRegression.sklearn_module(),
            "sklearn.linear_model"
        );
        assert_eq!(
            SklearnAlgorithm::KMeans.sklearn_module(),
            "sklearn.cluster"
        );
        assert_eq!(
            SklearnAlgorithm::StandardScaler.sklearn_module(),
            "sklearn.preprocessing"
        );
    }

    #[test]
    fn test_conversion_report() {
        let converter = SklearnConverter::new();
        let report = converter.conversion_report();
        assert!(report.contains("sklearn → Aprender"));
        assert!(report.contains("LinearRegression"));
        assert!(report.contains("Complexity"));
    }
}
