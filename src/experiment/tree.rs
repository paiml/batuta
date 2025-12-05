//! Experiment Tracking Frameworks Tree Visualization
//!
//! Displays comparison tree of Python experiment tracking frameworks
//! (MLflow, Weights & Biases, Neptune, etc.) and their PAIML Rust replacements.
//!
//! Core principle: Python experiment tracking is replaced by sovereign Rust alternatives.
//! No Python runtime dependencies permitted in production.

use serde::{Deserialize, Serialize};

/// Experiment tracking framework being compared
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExperimentFramework {
    /// MLflow - Open source experiment tracking
    MLflow,
    /// Weights & Biases - Commercial experiment tracking
    WandB,
    /// Neptune.ai - Commercial ML metadata store
    Neptune,
    /// Comet ML - Commercial experiment tracking
    CometML,
    /// Sacred - Academic experiment tracking
    Sacred,
    /// DVC - Data Version Control with experiment tracking
    Dvc,
}

impl ExperimentFramework {
    /// Get the display name
    pub fn name(&self) -> &'static str {
        match self {
            ExperimentFramework::MLflow => "MLflow",
            ExperimentFramework::WandB => "Weights & Biases",
            ExperimentFramework::Neptune => "Neptune.ai",
            ExperimentFramework::CometML => "Comet ML",
            ExperimentFramework::Sacred => "Sacred",
            ExperimentFramework::Dvc => "DVC",
        }
    }

    /// Get the PAIML replacement
    pub fn replacement(&self) -> &'static str {
        match self {
            ExperimentFramework::MLflow => "Entrenar + Batuta",
            ExperimentFramework::WandB => "Entrenar + Trueno-Viz",
            ExperimentFramework::Neptune => "Entrenar",
            ExperimentFramework::CometML => "Entrenar + Presentar",
            ExperimentFramework::Sacred => "Entrenar",
            ExperimentFramework::Dvc => "Batuta + Trueno-DB",
        }
    }

    /// Get all frameworks
    pub fn all() -> Vec<ExperimentFramework> {
        vec![
            ExperimentFramework::MLflow,
            ExperimentFramework::WandB,
            ExperimentFramework::Neptune,
            ExperimentFramework::CometML,
            ExperimentFramework::Sacred,
            ExperimentFramework::Dvc,
        ]
    }
}

/// Integration type - all are replacements (Python eliminated)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntegrationType {
    /// PAIML component fully replaces Python equivalent
    Replaces,
}

impl IntegrationType {
    /// Get the display code
    pub fn code(&self) -> &'static str {
        match self {
            IntegrationType::Replaces => "REP",
        }
    }
}

/// A component within a framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkComponent {
    /// Component name
    pub name: String,
    /// Description
    pub description: String,
    /// PAIML replacement
    pub replacement: String,
    /// Sub-components
    pub sub_components: Vec<String>,
}

/// A category of components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkCategory {
    /// Category name
    pub name: String,
    /// Components in this category
    pub components: Vec<FrameworkComponent>,
}

/// Integration mapping between Python and Rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationMapping {
    /// PAIML component
    pub paiml_component: String,
    /// Python component being replaced
    pub python_component: String,
    /// Integration type
    pub integration_type: IntegrationType,
    /// Category
    pub category: String,
}

/// Experiment tracking tree structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentTree {
    /// Framework name
    pub framework: String,
    /// PAIML replacement
    pub replacement: String,
    /// Categories
    pub categories: Vec<FrameworkCategory>,
}

/// Build MLflow tree
pub fn build_mlflow_tree() -> ExperimentTree {
    ExperimentTree {
        framework: "MLflow".to_string(),
        replacement: "Entrenar + Batuta".to_string(),
        categories: vec![
            FrameworkCategory {
                name: "Experiment Tracking".to_string(),
                components: vec![
                    FrameworkComponent {
                        name: "mlflow.start_run()".to_string(),
                        description: "Run lifecycle management".to_string(),
                        replacement: "Entrenar::ExperimentRun::new()".to_string(),
                        sub_components: vec![
                            "log_param".to_string(),
                            "log_metric".to_string(),
                            "log_artifact".to_string(),
                        ],
                    },
                    FrameworkComponent {
                        name: "mlflow.log_params()".to_string(),
                        description: "Hyperparameter logging".to_string(),
                        replacement: "ExperimentRun::log_param()".to_string(),
                        sub_components: vec![],
                    },
                    FrameworkComponent {
                        name: "mlflow.log_metrics()".to_string(),
                        description: "Metric logging".to_string(),
                        replacement: "ExperimentRun::log_metric()".to_string(),
                        sub_components: vec![],
                    },
                    FrameworkComponent {
                        name: "mlflow.set_tags()".to_string(),
                        description: "Run tagging".to_string(),
                        replacement: "ExperimentRun::tags".to_string(),
                        sub_components: vec![],
                    },
                ],
            },
            FrameworkCategory {
                name: "Model Registry".to_string(),
                components: vec![
                    FrameworkComponent {
                        name: "mlflow.register_model()".to_string(),
                        description: "Model versioning".to_string(),
                        replacement: "SovereignDistribution".to_string(),
                        sub_components: vec![
                            "stage_transitions".to_string(),
                            "model_versions".to_string(),
                        ],
                    },
                    FrameworkComponent {
                        name: "mlflow.pyfunc.log_model()".to_string(),
                        description: "Model artifact storage".to_string(),
                        replacement: "SovereignArtifact".to_string(),
                        sub_components: vec![],
                    },
                ],
            },
            FrameworkCategory {
                name: "Artifact Storage".to_string(),
                components: vec![
                    FrameworkComponent {
                        name: "mlflow.log_artifact()".to_string(),
                        description: "File storage".to_string(),
                        replacement: "SovereignArtifact".to_string(),
                        sub_components: vec![],
                    },
                    FrameworkComponent {
                        name: "S3/Azure/GCS backends".to_string(),
                        description: "Cloud storage".to_string(),
                        replacement: "Batuta deploy (sovereign)".to_string(),
                        sub_components: vec![],
                    },
                ],
            },
            FrameworkCategory {
                name: "Search & Query".to_string(),
                components: vec![
                    FrameworkComponent {
                        name: "mlflow.search_runs()".to_string(),
                        description: "Run search".to_string(),
                        replacement: "ExperimentStorage::list_runs()".to_string(),
                        sub_components: vec![],
                    },
                    FrameworkComponent {
                        name: "mlflow.search_experiments()".to_string(),
                        description: "Experiment search".to_string(),
                        replacement: "ExperimentStorage trait".to_string(),
                        sub_components: vec![],
                    },
                ],
            },
            FrameworkCategory {
                name: "GenAI / LLM".to_string(),
                components: vec![
                    FrameworkComponent {
                        name: "mlflow.tracing".to_string(),
                        description: "LLM trace capture".to_string(),
                        replacement: "Realizar::trace()".to_string(),
                        sub_components: vec![
                            "OpenAI".to_string(),
                            "LangChain".to_string(),
                            "LlamaIndex".to_string(),
                        ],
                    },
                    FrameworkComponent {
                        name: "mlflow.evaluate()".to_string(),
                        description: "LLM evaluation".to_string(),
                        replacement: "Entrenar::evaluate()".to_string(),
                        sub_components: vec![],
                    },
                    FrameworkComponent {
                        name: "Prompt Registry".to_string(),
                        description: "Prompt versioning".to_string(),
                        replacement: "Realizar::PromptTemplate".to_string(),
                        sub_components: vec![],
                    },
                ],
            },
            FrameworkCategory {
                name: "Deployment".to_string(),
                components: vec![
                    FrameworkComponent {
                        name: "mlflow models serve".to_string(),
                        description: "Model serving".to_string(),
                        replacement: "Batuta serve (GGUF)".to_string(),
                        sub_components: vec![],
                    },
                    FrameworkComponent {
                        name: "MLflow Gateway".to_string(),
                        description: "LLM gateway".to_string(),
                        replacement: "Batuta serve + SpilloverRouter".to_string(),
                        sub_components: vec![],
                    },
                ],
            },
        ],
    }
}

/// Build Weights & Biases tree
pub fn build_wandb_tree() -> ExperimentTree {
    ExperimentTree {
        framework: "Weights & Biases".to_string(),
        replacement: "Entrenar + Trueno-Viz".to_string(),
        categories: vec![
            FrameworkCategory {
                name: "Experiment Tracking".to_string(),
                components: vec![
                    FrameworkComponent {
                        name: "wandb.init()".to_string(),
                        description: "Run initialization".to_string(),
                        replacement: "Entrenar::ExperimentRun::new()".to_string(),
                        sub_components: vec![],
                    },
                    FrameworkComponent {
                        name: "wandb.log()".to_string(),
                        description: "Metric logging".to_string(),
                        replacement: "ExperimentRun::log_metric()".to_string(),
                        sub_components: vec![],
                    },
                    FrameworkComponent {
                        name: "wandb.config".to_string(),
                        description: "Hyperparameter config".to_string(),
                        replacement: "ExperimentRun::hyperparameters".to_string(),
                        sub_components: vec![],
                    },
                ],
            },
            FrameworkCategory {
                name: "Visualization".to_string(),
                components: vec![
                    FrameworkComponent {
                        name: "wandb.plot".to_string(),
                        description: "Custom plots".to_string(),
                        replacement: "Trueno-Viz::Chart".to_string(),
                        sub_components: vec![],
                    },
                    FrameworkComponent {
                        name: "Tables".to_string(),
                        description: "Data tables".to_string(),
                        replacement: "Trueno-Viz::DataGrid".to_string(),
                        sub_components: vec![],
                    },
                    FrameworkComponent {
                        name: "Media logging".to_string(),
                        description: "Images/audio/video".to_string(),
                        replacement: "Presentar::MediaView".to_string(),
                        sub_components: vec![],
                    },
                ],
            },
            FrameworkCategory {
                name: "Sweeps".to_string(),
                components: vec![
                    FrameworkComponent {
                        name: "wandb.sweep()".to_string(),
                        description: "Hyperparameter search".to_string(),
                        replacement: "Entrenar::HyperparameterSearch".to_string(),
                        sub_components: vec![
                            "grid".to_string(),
                            "random".to_string(),
                            "bayes".to_string(),
                        ],
                    },
                ],
            },
            FrameworkCategory {
                name: "Artifacts".to_string(),
                components: vec![
                    FrameworkComponent {
                        name: "wandb.Artifact".to_string(),
                        description: "Dataset/model versioning".to_string(),
                        replacement: "SovereignArtifact".to_string(),
                        sub_components: vec![],
                    },
                ],
            },
        ],
    }
}

/// Build Neptune tree
pub fn build_neptune_tree() -> ExperimentTree {
    ExperimentTree {
        framework: "Neptune.ai".to_string(),
        replacement: "Entrenar".to_string(),
        categories: vec![
            FrameworkCategory {
                name: "Experiment Tracking".to_string(),
                components: vec![
                    FrameworkComponent {
                        name: "neptune.init_run()".to_string(),
                        description: "Run initialization".to_string(),
                        replacement: "Entrenar::ExperimentRun::new()".to_string(),
                        sub_components: vec![],
                    },
                    FrameworkComponent {
                        name: "run[\"metric\"].log()".to_string(),
                        description: "Metric logging".to_string(),
                        replacement: "ExperimentRun::log_metric()".to_string(),
                        sub_components: vec![],
                    },
                ],
            },
            FrameworkCategory {
                name: "Metadata Store".to_string(),
                components: vec![
                    FrameworkComponent {
                        name: "Namespace hierarchy".to_string(),
                        description: "Nested metadata".to_string(),
                        replacement: "ExperimentRun::hyperparameters (JSON)".to_string(),
                        sub_components: vec![],
                    },
                    FrameworkComponent {
                        name: "System metrics".to_string(),
                        description: "CPU/GPU/memory".to_string(),
                        replacement: "EnergyMetrics + ComputeDevice".to_string(),
                        sub_components: vec![],
                    },
                ],
            },
        ],
    }
}

/// Build DVC tree
pub fn build_dvc_tree() -> ExperimentTree {
    ExperimentTree {
        framework: "DVC".to_string(),
        replacement: "Batuta + Trueno-DB".to_string(),
        categories: vec![
            FrameworkCategory {
                name: "Data Versioning".to_string(),
                components: vec![
                    FrameworkComponent {
                        name: "dvc add".to_string(),
                        description: "Track data files".to_string(),
                        replacement: "Trueno-DB::DatasetVersion".to_string(),
                        sub_components: vec![],
                    },
                    FrameworkComponent {
                        name: "dvc push/pull".to_string(),
                        description: "Remote storage".to_string(),
                        replacement: "SovereignDistribution".to_string(),
                        sub_components: vec![],
                    },
                ],
            },
            FrameworkCategory {
                name: "Experiment Tracking".to_string(),
                components: vec![
                    FrameworkComponent {
                        name: "dvc exp run".to_string(),
                        description: "Run experiments".to_string(),
                        replacement: "Batuta orchestrate".to_string(),
                        sub_components: vec![],
                    },
                    FrameworkComponent {
                        name: "dvc exp show".to_string(),
                        description: "Compare experiments".to_string(),
                        replacement: "Batuta experiment tree".to_string(),
                        sub_components: vec![],
                    },
                    FrameworkComponent {
                        name: "dvc metrics".to_string(),
                        description: "Metric tracking".to_string(),
                        replacement: "ExperimentRun::metrics".to_string(),
                        sub_components: vec![],
                    },
                ],
            },
            FrameworkCategory {
                name: "Pipelines".to_string(),
                components: vec![
                    FrameworkComponent {
                        name: "dvc.yaml".to_string(),
                        description: "Pipeline definition".to_string(),
                        replacement: "batuta.toml".to_string(),
                        sub_components: vec![],
                    },
                    FrameworkComponent {
                        name: "dvc repro".to_string(),
                        description: "Reproduce pipeline".to_string(),
                        replacement: "Batuta transpile + validate".to_string(),
                        sub_components: vec![],
                    },
                ],
            },
        ],
    }
}

/// Build all integration mappings
pub fn build_integration_mappings() -> Vec<IntegrationMapping> {
    vec![
        // Experiment Tracking
        IntegrationMapping {
            paiml_component: "Entrenar::ExperimentRun".to_string(),
            python_component: "mlflow.start_run()".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "Experiment Tracking".to_string(),
        },
        IntegrationMapping {
            paiml_component: "ExperimentRun::log_metric()".to_string(),
            python_component: "mlflow.log_metrics()".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "Experiment Tracking".to_string(),
        },
        IntegrationMapping {
            paiml_component: "ExperimentRun::log_param()".to_string(),
            python_component: "mlflow.log_params()".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "Experiment Tracking".to_string(),
        },
        IntegrationMapping {
            paiml_component: "ExperimentRun::tags".to_string(),
            python_component: "mlflow.set_tags()".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "Experiment Tracking".to_string(),
        },
        IntegrationMapping {
            paiml_component: "Entrenar::ExperimentRun".to_string(),
            python_component: "wandb.init()".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "Experiment Tracking".to_string(),
        },
        IntegrationMapping {
            paiml_component: "Entrenar::ExperimentRun".to_string(),
            python_component: "neptune.init_run()".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "Experiment Tracking".to_string(),
        },
        // Model Registry
        IntegrationMapping {
            paiml_component: "SovereignDistribution".to_string(),
            python_component: "mlflow.register_model()".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "Model Registry".to_string(),
        },
        IntegrationMapping {
            paiml_component: "SovereignArtifact".to_string(),
            python_component: "mlflow.pyfunc.log_model()".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "Model Registry".to_string(),
        },
        IntegrationMapping {
            paiml_component: "SovereignArtifact".to_string(),
            python_component: "wandb.Artifact".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "Model Registry".to_string(),
        },
        // Cost & Energy
        IntegrationMapping {
            paiml_component: "EnergyMetrics".to_string(),
            python_component: "System metrics (wandb/neptune)".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "Cost & Energy".to_string(),
        },
        IntegrationMapping {
            paiml_component: "CostMetrics".to_string(),
            python_component: "N/A (not in MLflow)".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "Cost & Energy".to_string(),
        },
        IntegrationMapping {
            paiml_component: "CostPerformanceBenchmark".to_string(),
            python_component: "N/A (Pareto frontier)".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "Cost & Energy".to_string(),
        },
        IntegrationMapping {
            paiml_component: "ComputeDevice".to_string(),
            python_component: "mlflow.system_metrics".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "Cost & Energy".to_string(),
        },
        // Visualization
        IntegrationMapping {
            paiml_component: "Trueno-Viz::Chart".to_string(),
            python_component: "wandb.plot".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "Visualization".to_string(),
        },
        IntegrationMapping {
            paiml_component: "Trueno-Viz::DataGrid".to_string(),
            python_component: "wandb.Table".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "Visualization".to_string(),
        },
        IntegrationMapping {
            paiml_component: "Presentar::Dashboard".to_string(),
            python_component: "MLflow UI".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "Visualization".to_string(),
        },
        // LLM / GenAI
        IntegrationMapping {
            paiml_component: "Realizar::trace()".to_string(),
            python_component: "mlflow.tracing".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "LLM / GenAI".to_string(),
        },
        IntegrationMapping {
            paiml_component: "Entrenar::evaluate()".to_string(),
            python_component: "mlflow.evaluate()".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "LLM / GenAI".to_string(),
        },
        IntegrationMapping {
            paiml_component: "Realizar::PromptTemplate".to_string(),
            python_component: "MLflow Prompt Registry".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "LLM / GenAI".to_string(),
        },
        // Deployment
        IntegrationMapping {
            paiml_component: "Batuta serve".to_string(),
            python_component: "mlflow models serve".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "Deployment".to_string(),
        },
        IntegrationMapping {
            paiml_component: "SpilloverRouter".to_string(),
            python_component: "MLflow Gateway".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "Deployment".to_string(),
        },
        IntegrationMapping {
            paiml_component: "SovereignDistribution".to_string(),
            python_component: "MLflow Docker/K8s".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "Deployment".to_string(),
        },
        // Data Versioning
        IntegrationMapping {
            paiml_component: "Trueno-DB::DatasetVersion".to_string(),
            python_component: "dvc add".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "Data Versioning".to_string(),
        },
        IntegrationMapping {
            paiml_component: "Batuta orchestrate".to_string(),
            python_component: "dvc repro".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "Data Versioning".to_string(),
        },
        // Academic / Research
        IntegrationMapping {
            paiml_component: "ResearchArtifact".to_string(),
            python_component: "N/A (ORCID/CRediT)".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "Academic / Research".to_string(),
        },
        IntegrationMapping {
            paiml_component: "CitationMetadata".to_string(),
            python_component: "N/A (BibTeX/CFF)".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "Academic / Research".to_string(),
        },
        IntegrationMapping {
            paiml_component: "PreRegistration".to_string(),
            python_component: "N/A (reproducibility)".to_string(),
            integration_type: IntegrationType::Replaces,
            category: "Academic / Research".to_string(),
        },
    ]
}

/// Format a single framework tree as ASCII
pub fn format_framework_tree(tree: &ExperimentTree) -> String {
    let mut output = String::new();
    output.push_str(&format!(
        "{} (Python) → {} (Rust)\n",
        tree.framework, tree.replacement
    ));

    for (cat_idx, category) in tree.categories.iter().enumerate() {
        let is_last_cat = cat_idx == tree.categories.len() - 1;
        let cat_prefix = if is_last_cat { "└──" } else { "├──" };
        let cat_continuation = if is_last_cat { "    " } else { "│   " };

        output.push_str(&format!("{} {}\n", cat_prefix, category.name));

        for (comp_idx, component) in category.components.iter().enumerate() {
            let is_last_comp = comp_idx == category.components.len() - 1;
            let comp_prefix = if is_last_comp { "└──" } else { "├──" };

            output.push_str(&format!(
                "{}{} {} → {}\n",
                cat_continuation, comp_prefix, component.name, component.replacement
            ));

            if !component.sub_components.is_empty() {
                let sub_continuation = if is_last_comp {
                    format!("{}    ", cat_continuation)
                } else {
                    format!("{}│   ", cat_continuation)
                };

                for (sub_idx, sub) in component.sub_components.iter().enumerate() {
                    let is_last_sub = sub_idx == component.sub_components.len() - 1;
                    let sub_prefix = if is_last_sub { "└──" } else { "├──" };
                    output.push_str(&format!("{}{} {}\n", sub_continuation, sub_prefix, sub));
                }
            }
        }
    }

    output
}

/// Format all frameworks
pub fn format_all_frameworks() -> String {
    let mut output = String::new();
    output.push_str("EXPERIMENT TRACKING FRAMEWORKS ECOSYSTEM\n");
    output.push_str("========================================\n\n");

    output.push_str(&format_framework_tree(&build_mlflow_tree()));
    output.push('\n');
    output.push_str(&format_framework_tree(&build_wandb_tree()));
    output.push('\n');
    output.push_str(&format_framework_tree(&build_neptune_tree()));
    output.push('\n');
    output.push_str(&format_framework_tree(&build_dvc_tree()));

    output.push_str(&format!(
        "\nSummary: {} Python frameworks replaced by sovereign Rust stack\n",
        ExperimentFramework::all().len()
    ));

    output
}

/// Format integration mappings
pub fn format_integration_mappings() -> String {
    let mappings = build_integration_mappings();
    let mut output = String::new();

    output.push_str("PAIML REPLACEMENTS FOR PYTHON EXPERIMENT TRACKING\n");
    output.push_str("=================================================\n\n");

    // Group by category
    let categories = [
        "Experiment Tracking",
        "Model Registry",
        "Cost & Energy",
        "Visualization",
        "LLM / GenAI",
        "Deployment",
        "Data Versioning",
        "Academic / Research",
    ];

    for category in categories {
        let cat_mappings: Vec<_> = mappings
            .iter()
            .filter(|m| m.category == category)
            .collect();

        if cat_mappings.is_empty() {
            continue;
        }

        output.push_str(&format!("{}\n", category.to_uppercase()));

        for (idx, mapping) in cat_mappings.iter().enumerate() {
            let is_last = idx == cat_mappings.len() - 1;
            let prefix = if is_last { "└──" } else { "├──" };

            output.push_str(&format!(
                "{} [{}] {} ← {}\n",
                prefix,
                mapping.integration_type.code(),
                mapping.paiml_component,
                mapping.python_component
            ));
        }
        output.push('\n');
    }

    output.push_str("Legend: [REP]=Replaces (Python eliminated)\n\n");
    output.push_str(&format!(
        "Summary: {} Python components replaced by sovereign Rust alternatives\n",
        mappings.len()
    ));
    output.push_str("         Zero Python dependencies in production\n");

    output
}

/// Format as JSON
pub fn format_json(framework: Option<ExperimentFramework>, integration: bool) -> String {
    if integration {
        let mappings = build_integration_mappings();
        serde_json::to_string_pretty(&mappings).unwrap_or_default()
    } else {
        match framework {
            Some(ExperimentFramework::MLflow) => {
                serde_json::to_string_pretty(&build_mlflow_tree()).unwrap_or_default()
            }
            Some(ExperimentFramework::WandB) => {
                serde_json::to_string_pretty(&build_wandb_tree()).unwrap_or_default()
            }
            Some(ExperimentFramework::Neptune) => {
                serde_json::to_string_pretty(&build_neptune_tree()).unwrap_or_default()
            }
            Some(ExperimentFramework::Dvc) => {
                serde_json::to_string_pretty(&build_dvc_tree()).unwrap_or_default()
            }
            Some(ExperimentFramework::CometML) | Some(ExperimentFramework::Sacred) => {
                // Minimal trees for these
                let tree = ExperimentTree {
                    framework: framework.unwrap().name().to_string(),
                    replacement: framework.unwrap().replacement().to_string(),
                    categories: vec![],
                };
                serde_json::to_string_pretty(&tree).unwrap_or_default()
            }
            None => {
                let trees = vec![
                    build_mlflow_tree(),
                    build_wandb_tree(),
                    build_neptune_tree(),
                    build_dvc_tree(),
                ];
                serde_json::to_string_pretty(&trees).unwrap_or_default()
            }
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn test_EXP_TREE_001_framework_names() {
        assert_eq!(ExperimentFramework::MLflow.name(), "MLflow");
        assert_eq!(ExperimentFramework::WandB.name(), "Weights & Biases");
        assert_eq!(ExperimentFramework::Neptune.name(), "Neptune.ai");
        assert_eq!(ExperimentFramework::Dvc.name(), "DVC");
    }

    #[test]
    fn test_EXP_TREE_002_framework_replacements() {
        assert_eq!(ExperimentFramework::MLflow.replacement(), "Entrenar + Batuta");
        assert_eq!(ExperimentFramework::WandB.replacement(), "Entrenar + Trueno-Viz");
        assert_eq!(ExperimentFramework::Neptune.replacement(), "Entrenar");
        assert_eq!(ExperimentFramework::Dvc.replacement(), "Batuta + Trueno-DB");
    }

    #[test]
    fn test_EXP_TREE_003_all_frameworks() {
        let all = ExperimentFramework::all();
        assert_eq!(all.len(), 6);
    }

    #[test]
    fn test_EXP_TREE_004_integration_type_code() {
        assert_eq!(IntegrationType::Replaces.code(), "REP");
    }

    #[test]
    fn test_EXP_TREE_005_mlflow_tree_structure() {
        let tree = build_mlflow_tree();
        assert_eq!(tree.framework, "MLflow");
        assert_eq!(tree.replacement, "Entrenar + Batuta");
        assert!(!tree.categories.is_empty());

        // Check experiment tracking category exists
        let exp_tracking = tree.categories.iter().find(|c| c.name == "Experiment Tracking");
        assert!(exp_tracking.is_some());
    }

    #[test]
    fn test_EXP_TREE_006_mlflow_has_genai() {
        let tree = build_mlflow_tree();
        let genai = tree.categories.iter().find(|c| c.name == "GenAI / LLM");
        assert!(genai.is_some());

        let genai = genai.unwrap();
        assert!(genai.components.iter().any(|c| c.name == "mlflow.tracing"));
    }

    #[test]
    fn test_EXP_TREE_007_wandb_tree_structure() {
        let tree = build_wandb_tree();
        assert_eq!(tree.framework, "Weights & Biases");
        assert_eq!(tree.replacement, "Entrenar + Trueno-Viz");

        // Check visualization category
        let viz = tree.categories.iter().find(|c| c.name == "Visualization");
        assert!(viz.is_some());
    }

    #[test]
    fn test_EXP_TREE_008_neptune_tree_structure() {
        let tree = build_neptune_tree();
        assert_eq!(tree.framework, "Neptune.ai");
        assert_eq!(tree.replacement, "Entrenar");
    }

    #[test]
    fn test_EXP_TREE_009_dvc_tree_structure() {
        let tree = build_dvc_tree();
        assert_eq!(tree.framework, "DVC");
        assert_eq!(tree.replacement, "Batuta + Trueno-DB");

        // Check data versioning
        let data_ver = tree.categories.iter().find(|c| c.name == "Data Versioning");
        assert!(data_ver.is_some());
    }

    #[test]
    fn test_EXP_TREE_010_integration_mappings_count() {
        let mappings = build_integration_mappings();
        assert!(mappings.len() >= 20);
    }

    #[test]
    fn test_EXP_TREE_011_integration_mappings_categories() {
        let mappings = build_integration_mappings();
        let categories: std::collections::HashSet<_> =
            mappings.iter().map(|m| m.category.as_str()).collect();

        assert!(categories.contains("Experiment Tracking"));
        assert!(categories.contains("Model Registry"));
        assert!(categories.contains("Visualization"));
        assert!(categories.contains("LLM / GenAI"));
    }

    #[test]
    fn test_EXP_TREE_012_format_framework_tree() {
        let tree = build_mlflow_tree();
        let output = format_framework_tree(&tree);

        assert!(output.contains("MLflow (Python) → Entrenar + Batuta (Rust)"));
        assert!(output.contains("Experiment Tracking"));
        assert!(output.contains("mlflow.start_run()"));
    }

    #[test]
    fn test_EXP_TREE_013_format_all_frameworks() {
        let output = format_all_frameworks();

        assert!(output.contains("EXPERIMENT TRACKING FRAMEWORKS ECOSYSTEM"));
        assert!(output.contains("MLflow"));
        assert!(output.contains("Weights & Biases"));
        assert!(output.contains("Neptune.ai"));
        assert!(output.contains("DVC"));
        assert!(output.contains("Summary:"));
    }

    #[test]
    fn test_EXP_TREE_014_format_integration_mappings() {
        let output = format_integration_mappings();

        assert!(output.contains("PAIML REPLACEMENTS"));
        assert!(output.contains("[REP]"));
        assert!(output.contains("Entrenar::ExperimentRun"));
        assert!(output.contains("mlflow.start_run()"));
        assert!(output.contains("Legend:"));
    }

    #[test]
    fn test_EXP_TREE_015_json_output_single() {
        let json = format_json(Some(ExperimentFramework::MLflow), false);
        assert!(json.contains("MLflow"));
        assert!(json.contains("Entrenar"));

        // Verify it's valid JSON
        let parsed: Result<ExperimentTree, _> = serde_json::from_str(&json);
        assert!(parsed.is_ok());
    }

    #[test]
    fn test_EXP_TREE_016_json_output_all() {
        let json = format_json(None, false);

        // Verify it's valid JSON array
        let parsed: Result<Vec<ExperimentTree>, _> = serde_json::from_str(&json);
        assert!(parsed.is_ok());
        assert_eq!(parsed.unwrap().len(), 4);
    }

    #[test]
    fn test_EXP_TREE_017_json_output_integration() {
        let json = format_json(None, true);

        // Verify it's valid JSON array
        let parsed: Result<Vec<IntegrationMapping>, _> = serde_json::from_str(&json);
        assert!(parsed.is_ok());
        assert!(parsed.unwrap().len() >= 20);
    }

    #[test]
    fn test_EXP_TREE_018_all_mappings_have_replacements() {
        let mappings = build_integration_mappings();
        for mapping in &mappings {
            assert_eq!(mapping.integration_type, IntegrationType::Replaces);
            assert!(!mapping.paiml_component.is_empty());
            assert!(!mapping.python_component.is_empty());
        }
    }

    #[test]
    fn test_EXP_TREE_019_academic_features() {
        let mappings = build_integration_mappings();
        let academic: Vec<_> = mappings
            .iter()
            .filter(|m| m.category == "Academic / Research")
            .collect();

        assert!(academic.len() >= 3);
        assert!(academic.iter().any(|m| m.paiml_component.contains("ResearchArtifact")));
        assert!(academic.iter().any(|m| m.paiml_component.contains("CitationMetadata")));
    }

    #[test]
    fn test_EXP_TREE_020_cost_energy_features() {
        let mappings = build_integration_mappings();
        let cost_energy: Vec<_> = mappings
            .iter()
            .filter(|m| m.category == "Cost & Energy")
            .collect();

        assert!(cost_energy.len() >= 3);
        assert!(cost_energy.iter().any(|m| m.paiml_component.contains("EnergyMetrics")));
        assert!(cost_energy.iter().any(|m| m.paiml_component.contains("CostMetrics")));
    }
}
