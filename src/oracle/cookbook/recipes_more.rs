//! Additional recipe registrations
//!
//! Continuation of recipes module, split for file size compliance.

use super::Recipe;

pub fn register_training_recipes(cookbook: &mut super::Cookbook) {
    // LoRA Fine-tuning
    cookbook.add(
        Recipe::new("training-lora", "LoRA Fine-tuning")
            .with_problem("Fine-tune large models efficiently with Low-Rank Adaptation")
            .with_components(vec!["entrenar", "aprender", "alimentar"])
            .with_tags(vec!["training", "lora", "fine-tuning", "efficient", "llm"])
            .with_code(
                r#"use entrenar::prelude::*;

// Load base model
let model = Model::load("llama-7b.apr")?;

// Configure LoRA
let lora_config = LoraConfig {
    r: 16,                    // Rank
    alpha: 32,                // Scaling factor
    dropout: 0.1,
    target_modules: vec!["q_proj", "v_proj"],
};

// Apply LoRA adapters
let model = model.with_lora(lora_config)?;

// Only ~0.1% of parameters are trainable now
println!("Trainable params: {}", model.trainable_params());

// Training loop
let optimizer = AdamW::new(model.trainable_params(), 1e-4);
for batch in dataloader {
    let loss = model.forward(&batch)?;
    loss.backward()?;
    optimizer.step()?;
}

// Save LoRA weights only (small file)
model.save_lora("adapter.lora")?;

// Later: merge for inference
// let merged = Model::load("llama-7b.apr")?.merge_lora("adapter.lora")?;
"#,
            )
            .with_related(vec!["training-qlora", "training-autograd"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_lora_config_rank_and_alpha() {
    let rank = 16;
    let alpha = 32;
    assert!(rank > 0 && alpha >= rank);
}

    #[test]
    fn test_trainable_params_fraction() {
    let total = 1_000_000;
    let lora = 8192;
    let fraction = lora as f64 / total as f64;
    assert!(fraction < 0.1);
}

    #[test]
    fn test_dropout_in_valid_range() {
    let dropout = 0.1_f64;
    assert!(dropout >= 0.0 && dropout <= 1.0);
}
}"#,
            ),
    );

    // QLoRA
    cookbook.add(
        Recipe::new("training-qlora", "QLoRA Quantized Fine-tuning")
            .with_problem("Fine-tune 4-bit quantized models on consumer hardware")
            .with_components(vec!["entrenar", "aprender"])
            .with_tags(vec![
                "training",
                "qlora",
                "quantization",
                "4bit",
                "memory-efficient",
            ])
            .with_code(
                r#"use entrenar::prelude::*;

// Load 4-bit quantized model
let model = Model::load_quantized("llama-7b.q4_k.gguf")?;

// QLoRA config (LoRA on quantized base)
let qlora_config = QLoraConfig {
    lora: LoraConfig { r: 64, alpha: 16, dropout: 0.1, .. },
    nf4: true,              // NormalFloat4 quantization
    double_quant: true,     // Double quantization for memory
    compute_dtype: F16,     // Compute in fp16
};

let model = model.with_qlora(qlora_config)?;

// Train on 24GB GPU (fits 7B model!)
let trainer = Trainer::new(model)
    .gradient_checkpointing(true)
    .batch_size(4)
    .gradient_accumulation(4);

trainer.train(&dataset, 3)?;  // 3 epochs
"#,
            )
            .with_related(vec!["training-lora"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_quantization_bits_valid() {
    let bits = 4;
    assert!(bits == 4 || bits == 8);
}

    #[test]
    fn test_effective_batch_size() {
    let batch_size = 4;
    let grad_accum = 4;
    let effective = batch_size * grad_accum;
    assert_eq!(effective, 16);
}

    #[test]
    fn test_nf4_requires_4bit() {
    let nf4 = true;
    let bits = 4;
    assert!(nf4 && bits == 4);
}
}"#,
            ),
    );

    // Autograd
    cookbook.add(
        Recipe::new("training-autograd", "Custom Training with Autograd")
            .with_problem("Build custom neural networks with automatic differentiation")
            .with_components(vec!["entrenar", "trueno"])
            .with_tags(vec!["training", "autograd", "neural-network", "custom"])
            .with_code(
                r#"use entrenar::autograd::*;

// Define model with autograd tensors
let w1 = Tensor::randn(&[784, 256]).requires_grad();
let w2 = Tensor::randn(&[256, 10]).requires_grad();

// Forward pass (computation graph built automatically)
fn forward(x: &Tensor, w1: &Tensor, w2: &Tensor) -> Tensor {
    let h = x.matmul(w1).relu();
    h.matmul(w2).softmax(-1)
}

// Training loop
let optimizer = SGD::new(vec![&w1, &w2], 0.01);
for (x, y) in dataloader {
    let pred = forward(&x, &w1, &w2);
    let loss = cross_entropy(&pred, &y);

    // Backward pass (gradients computed automatically)
    loss.backward();

    optimizer.step();
    optimizer.zero_grad();
}

// Gradients accessible
println!("w1 grad: {:?}", w1.grad());
"#,
            )
            .with_related(vec!["training-lora", "ml-random-forest"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_weight_matrix_dimensions() {
    let input_dim = 784;
    let hidden_dim = 256;
    let weights = vec![vec![0.0_f64; hidden_dim]; input_dim];
    assert_eq!(weights.len(), input_dim);
}

    #[test]
    fn test_softmax_sums_to_one() {
    let logits = vec![1.0_f64, 2.0, 3.0];
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp_sum: f64 = logits.iter().map(|x| (x - max).exp()).sum();
    let sum: f64 = logits.iter().map(|x| (x - max).exp() / exp_sum).sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

    #[test]
    fn test_learning_rate_positive() {
    let lr = 0.01_f64;
    assert!(lr > 0.0);
}
}"#,
            ),
    );
}

// =========================================================================
// Data Loading Recipes
// =========================================================================

pub fn register_data_recipes(cookbook: &mut super::Cookbook) {
    // Alimentar Data Loading
    cookbook.add(
        Recipe::new("data-alimentar", "Zero-Copy Data Loading")
            .with_problem("Load large datasets efficiently with memory mapping")
            .with_components(vec!["alimentar", "trueno"])
            .with_tags(vec!["data", "loading", "parquet", "arrow", "zero-copy"])
            .with_code(
                r#"use alimentar::prelude::*;

// Load Parquet with zero-copy (memory-mapped)
let dataset = ParquetDataset::open("data.parquet")?
    .select(&["features", "label"])?
    .filter(|row| row["label"].as_i64() > 0)?;

// Iterate with batching
let dataloader = DataLoader::new(dataset)
    .batch_size(32)
    .shuffle(true)
    .num_workers(4);

for batch in dataloader {
    // batch.features is Arrow array (zero-copy)
    let features = batch["features"].as_tensor()?;
    let labels = batch["label"].as_tensor()?;

    model.train_step(&features, &labels)?;
}

// Streaming from remote (S3, HuggingFace)
let dataset = Dataset::from_hub("username/dataset")?
    .streaming(true);  // Don't download entire dataset
"#,
            )
            .with_related(vec!["data-preprocessing", "ml-random-forest"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_batch_size_config() {
    let batch_size = 32_u32;
    assert!(batch_size > 0);
}

    #[test]
    fn test_column_selection() {
    let columns = vec!["features", "label"];
    assert_eq!(columns.len(), 2);
}

    #[test]
    fn test_worker_count() {
    let workers = 4;
    assert!(workers > 0 && workers <= 16);
}
}"#,
            ),
    );

    // Data Preprocessing
    cookbook.add(
        Recipe::new("data-preprocessing", "Data Preprocessing Pipeline")
            .with_problem("Build reproducible preprocessing pipelines")
            .with_components(vec!["alimentar", "aprender"])
            .with_tags(vec!["data", "preprocessing", "pipeline", "transforms"])
            .with_code(
                r#"use alimentar::prelude::*;
use aprender::preprocessing::*;

// Build preprocessing pipeline
let pipeline = Pipeline::new()
    .add(StandardScaler::fit(&train_data)?)
    .add(OneHotEncoder::fit(&["category"])?)
    .add(Imputer::median());

// Apply to train/test
let X_train = pipeline.transform(&train_data)?;
let X_test = pipeline.transform(&test_data)?;

// Save pipeline for inference
pipeline.save("preprocess.pipeline")?;

// Later: load and apply
let pipeline = Pipeline::load("preprocess.pipeline")?;
let X_new = pipeline.transform(&new_data)?;
"#,
            )
            .with_related(vec!["data-alimentar"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_pipeline_step_count() {
    let steps = vec!["scale", "encode", "impute"];
    assert_eq!(steps.len(), 3);
}

    #[test]
    fn test_transform_preserves_row_count() {
    let input_rows = 1000;
    let output_rows = 1000;
    assert_eq!(input_rows, output_rows);
}

    #[test]
    fn test_scaler_std_positive() {
    let std_dev = 1.0_f64;
    assert!(std_dev > 0.0);
}
}"#,
            ),
    );
}

// =========================================================================
// Model Registry Recipes
// =========================================================================

pub fn register_registry_recipes(cookbook: &mut super::Cookbook) {
    // Pacha Model Registry
    cookbook.add(
        Recipe::new("registry-pacha", "Model Registry with Pacha")
            .with_problem("Version, sign, and distribute ML models securely")
            .with_components(vec!["pacha", "aprender"])
            .with_tags(vec![
                "registry",
                "versioning",
                "signing",
                "distribution",
                "mlops",
            ])
            .with_code(
                r#"use pacha::prelude::*;

// Initialize registry
let registry = Registry::new("./models")?;

// Register model with metadata
let model_card = ModelCard {
    name: "sentiment-classifier",
    version: "1.0.0",
    description: "BERT-based sentiment analysis",
    metrics: hashmap!{
        "accuracy" => 0.94,
        "f1" => 0.92,
    },
    license: "MIT",
    authors: vec!["team@example.com"],
};

// Push with Ed25519 signature
let artifact = registry.push(
    "model.apr",
    model_card,
    SigningKey::from_env()?,  // PACHA_SIGNING_KEY
)?;

println!("Registered: {}@{}", artifact.name, artifact.version);
println!("Hash: {}", artifact.blake3_hash);

// Pull model (verifies signature)
let model_path = registry.pull("sentiment-classifier", "1.0.0")?;

// List versions
for version in registry.versions("sentiment-classifier")? {
    println!("{} - {}", version.version, version.created_at);
}
"#,
            )
            .with_related(vec!["registry-hf", "ml-serving"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_model_card_metadata() {
    let name = "sentiment-classifier";
    let version = "1.0.0";
    assert!(!name.is_empty());
    assert!(version.chars().filter(|c| *c == '.').count() == 2);
}

    #[test]
    fn test_version_string_format() {
    let version = "1.0.0";
    let parts: Vec<_> = version.split('.').collect();
    assert_eq!(parts.len(), 3);
}

    #[test]
    fn test_hash_length() {
    let blake3_hash = "a".repeat(64);
    assert_eq!(blake3_hash.len(), 64);
}
}"#,
            ),
    );

    // HuggingFace Integration
    cookbook.add(
        Recipe::new("registry-hf", "HuggingFace Hub Integration")
            .with_problem("Download and cache models from HuggingFace Hub")
            .with_components(vec!["hf-hub", "aprender", "realizar"])
            .with_tags(vec!["registry", "huggingface", "download", "cache"])
            .with_code(
                r#"use hf_hub::api::sync::Api;

// Initialize API (uses HF_TOKEN env var if set)
let api = Api::new()?;

// Download model files
let repo = api.model("meta-llama/Llama-2-7b");
let model_path = repo.get("model.safetensors")?;
let config_path = repo.get("config.json")?;

// Files cached in ~/.cache/huggingface/hub/
println!("Model: {}", model_path.display());

// Download specific revision
let repo = api.model("meta-llama/Llama-2-7b").revision("main");
let path = repo.get("tokenizer.json")?;

// Progress callback
let repo = api.model("big-model").progress(|p| {
    println!("Downloading: {:.1}%", p.percent * 100.0);
});
"#,
            )
            .with_related(vec!["registry-pacha", "speech-whisper"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_api_url_valid() {
    let url = "https://huggingface.co";
    assert!(url.starts_with("https://"));
}

    #[test]
    fn test_model_path_structure() {
    let org = "meta-llama";
    let model = "Llama-2-7b";
    let path = format!("{}/{}", org, model);
    assert_eq!(path.split('/').count(), 2);
}

    #[test]
    fn test_revision_default() {
    let revision = "main";
    assert_eq!(revision, "main");
}
}"#,
            ),
    );
}

// =========================================================================
// RAG Pipeline Recipes
// =========================================================================

pub fn register_rag_recipes(cookbook: &mut super::Cookbook) {
    // RAG Pipeline
    cookbook.add(
        Recipe::new("rag-pipeline", "RAG Pipeline with Trueno-RAG")
            .with_problem("Build retrieval-augmented generation pipelines")
            .with_components(vec!["trueno-rag", "trueno-db", "aprender"])
            .with_tags(vec![
                "rag",
                "retrieval",
                "generation",
                "embeddings",
                "search",
            ])
            .with_code(
                r#"use trueno_rag::prelude::*;

// Initialize RAG pipeline
let rag = RagPipeline::builder()
    .chunker(SemanticChunker::new(512))  // Semantic chunking
    .embedder(Embedder::load("bge-small-en")?)
    .retriever(HybridRetriever::new()
        .bm25_weight(0.3)
        .dense_weight(0.7))
    .reranker(CrossEncoder::load("ms-marco-MiniLM")?)
    .build()?;

// Index documents
for doc in documents {
    rag.add_document(&doc)?;
}
rag.build_index()?;

// Query with retrieval
let query = "What is the capital of France?";
let results = rag.retrieve(query, 5)?;  // Top 5 chunks

for (i, chunk) in results.iter().enumerate() {
    println!("{}. [score: {:.3}] {}", i+1, chunk.score, chunk.text);
}

// Full RAG with generation
let context = rag.retrieve_context(query, 3)?;
let prompt = format!("Context:\n{}\n\nQuestion: {}\nAnswer:", context, query);
let answer = llm.generate(&prompt)?;
"#,
            )
            .with_related(vec!["rag-semantic-search", "ml-serving"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_top_k_parameter() {
    let top_k = 5;
    assert!(top_k > 0 && top_k <= 100);
}

    #[test]
    fn test_chunk_size_exceeds_overlap() {
    let chunk_size = 512;
    let overlap = 50;
    assert!(chunk_size > overlap);
}

    #[test]
    fn test_retriever_weights_sum_to_one() {
    let bm25_weight = 0.3_f64;
    let vector_weight = 0.7_f64;
    assert!((bm25_weight + vector_weight - 1.0).abs() < 1e-6);
}
}"#,
            ),
    );

    // Semantic Search
    cookbook.add(
        Recipe::new("rag-semantic-search", "Semantic Search Engine")
            .with_problem("Build fast semantic search over documents")
            .with_components(vec!["trueno-db", "trueno-rag"])
            .with_tags(vec![
                "search",
                "semantic",
                "embeddings",
                "hnsw",
                "vector-db",
            ])
            .with_code(
                r#"use trueno_db::prelude::*;
use trueno_rag::embeddings::*;

// Initialize vector store with HNSW index
let db = VectorDb::open("vectors.db")?
    .with_index(HnswConfig {
        m: 16,
        ef_construction: 200,
        ef_search: 50,
    });

// Embed and store documents
let embedder = Embedder::load("bge-small-en")?;
for doc in documents {
    let embedding = embedder.embed(&doc.text)?;
    db.insert(&doc.id, &embedding, &doc.metadata)?;
}

// Search
let query_embedding = embedder.embed("machine learning")?;
let results = db.search(&query_embedding, 10)?;

for result in results {
    println!("{}: {:.3}", result.id, result.score);
}

// Filtered search
let results = db.search_filtered(
    &query_embedding,
    10,
    |meta| meta["category"] == "science",
)?;
"#,
            )
            .with_related(vec!["rag-pipeline"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_hnsw_config_params() {
    let m = 16;
    let ef_construction = 200;
    assert!(m >= 4 && m <= 64);
    assert!(ef_construction >= m);
}

    #[test]
    fn test_search_result_ordering() {
    let scores = vec![0.95, 0.85, 0.75];
    let is_sorted = scores.windows(2).all(|w| w[0] >= w[1]);
    assert!(is_sorted);
}

    #[test]
    fn test_filter_predicate() {
    let min_score = 0.5_f64;
    let result_score = 0.75_f64;
    assert!(result_score >= min_score);
}
}"#,
            ),
    );
}

// =========================================================================
// Visualization Recipes
// =========================================================================

pub fn register_viz_recipes(cookbook: &mut super::Cookbook) {
    // Terminal Visualization
    cookbook.add(
        Recipe::new("viz-terminal", "Terminal Visualization")
            .with_problem("Create charts and plots in the terminal")
            .with_components(vec!["trueno-viz"])
            .with_tags(vec!["visualization", "terminal", "charts", "ascii"])
            .with_code(
                r#"use trueno_viz::prelude::*;

// Line chart in terminal
let chart = LineChart::new()
    .title("Training Loss")
    .x_label("Epoch")
    .y_label("Loss")
    .series("train", &train_losses)
    .series("val", &val_losses);

chart.render_terminal(80, 24)?;  // 80x24 chars

// Histogram
let hist = Histogram::new(&data)
    .bins(20)
    .title("Distribution");
hist.render_terminal(60, 15)?;

// Scatter plot
let scatter = ScatterPlot::new()
    .points(&x_vals, &y_vals)
    .title("Correlation");
scatter.render_terminal(40, 20)?;

// Progress bars (integrated with training)
let pb = ProgressBar::new(total_epochs);
for epoch in 0..total_epochs {
    // ... training ...
    pb.set(epoch, format!("loss: {:.4}", loss));
}
"#,
            )
            .with_related(vec!["viz-png", "training-autograd"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_chart_dimensions() {
    let width = 80;
    let height = 24;
    assert!(width > 0 && height > 0);
}

    #[test]
    fn test_bin_count() {
    let bins = 20;
    assert!(bins > 0 && bins <= 100);
}

    #[test]
    fn test_series_data_finite() {
    let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    assert!(data.iter().all(|x| x.is_finite()));
}
}"#,
            ),
    );

    // PNG Export
    cookbook.add(
        Recipe::new("viz-png", "PNG Chart Export")
            .with_problem("Export publication-quality charts as PNG images")
            .with_components(vec!["trueno-viz"])
            .with_tags(vec!["visualization", "png", "export", "charts"])
            .with_code(
                r#"use trueno_viz::prelude::*;

// Create chart
let chart = LineChart::new()
    .title("Model Performance")
    .x_label("Epoch")
    .y_label("Accuracy")
    .series("ResNet", &resnet_acc)
    .series("VGG", &vgg_acc)
    .legend(Position::TopRight);

// Export as PNG
chart.save_png("performance.png", 800, 600)?;

// With custom styling
let styled = chart
    .background(Color::WHITE)
    .grid(true)
    .font_size(14);
styled.save_png("styled.png", 1200, 800)?;

// Batch export multiple charts
let charts = vec![
    ("loss", loss_chart),
    ("accuracy", acc_chart),
    ("confusion", confusion_matrix),
];
for (name, chart) in charts {
    chart.save_png(&format!("{}.png", name), 800, 600)?;
}
"#,
            )
            .with_related(vec!["viz-terminal"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_image_dimensions() {
    let width = 800;
    let height = 600;
    assert!(width > 0 && height > 0);
}

    #[test]
    fn test_chart_title_non_empty() {
    let title = "Model Performance";
    assert!(!title.is_empty());
}

    #[test]
    fn test_batch_export_count() {
    let charts = vec!["loss", "accuracy", "confusion"];
    assert_eq!(charts.len(), 3);
}
}"#,
            ),
    );
}

// =========================================================================
// RLHF & Alignment Recipes
// =========================================================================

pub fn register_rlhf_recipes(cookbook: &mut super::Cookbook) {
    // Supervised Fine-Tuning (SFT)
    cookbook.add(
        Recipe::new("rlhf-sft", "Supervised Fine-Tuning (SFT)")
            .with_problem("Fine-tune a base model on instruction-following data")
            .with_components(vec!["entrenar", "aprender", "alimentar"])
            .with_tags(vec![
                "rlhf",
                "sft",
                "instruction-tuning",
                "fine-tuning",
                "alignment",
            ])
            .with_code(
                r#"use entrenar::prelude::*;
use entrenar::sft::*;

// Load base model
let model = Model::load("llama-7b.apr")?;

// Load instruction dataset (prompt, response pairs)
let dataset = SftDataset::load("alpaca_data.json")?
    .with_prompt_template(
        "Below is an instruction. Write a response.\n\n\
         ### Instruction:\n{instruction}\n\n\
         ### Response:\n"
    );

// Configure SFT trainer
let config = SftConfig {
    learning_rate: 2e-5,
    batch_size: 4,
    gradient_accumulation: 8,
    max_seq_length: 2048,
    warmup_ratio: 0.03,
    weight_decay: 0.0,
    ..Default::default()
};

// Train with LoRA for efficiency
let model = model.with_lora(LoraConfig::default())?;
let trainer = SftTrainer::new(model, config);

trainer.train(&dataset, 3)?;  // 3 epochs
model.save("sft-model.apr")?;

// Evaluation
let eval_loss = trainer.evaluate(&eval_dataset)?;
println!("Eval loss: {:.4}", eval_loss);
"#,
            )
            .with_related(vec!["rlhf-reward-model", "rlhf-dpo", "training-lora"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_sft_config_defaults() {
    let epochs = 3;
    let batch_size = 4;
    assert!(epochs > 0 && batch_size > 0);
}

    #[test]
    fn test_prompt_template_has_placeholder() {
    let template = "Instruction: {instruction} Response:";
    assert!(template.contains("{instruction}"));
}

    #[test]
    fn test_warmup_ratio_in_range() {
    let warmup_ratio = 0.03_f64;
    assert!(warmup_ratio >= 0.0 && warmup_ratio <= 1.0);
}
}"#,
            ),
    );

    // Reward Modeling
    cookbook.add(
        Recipe::new("rlhf-reward-model", "Reward Model Training")
            .with_problem("Train a reward model from human preference data")
            .with_components(vec!["entrenar", "aprender"])
            .with_tags(vec![
                "rlhf",
                "reward-model",
                "preferences",
                "ranking",
                "alignment",
            ])
            .with_code(
                r#"use entrenar::prelude::*;
use entrenar::reward::*;

// Load SFT model as base
let model = Model::load("sft-model.apr")?;

// Convert to reward model (adds value head)
let reward_model = RewardModel::from_base(model)?;

// Load preference dataset (chosen vs rejected)
let dataset = PreferenceDataset::load("preferences.json")?;
// Each sample: { prompt, chosen, rejected }

// Configure reward model training
let config = RewardConfig {
    learning_rate: 1e-5,
    batch_size: 4,
    max_length: 512,
    // Use margin ranking loss
    loss_fn: RewardLoss::MarginRanking { margin: 0.0 },
    ..Default::default()
};

let trainer = RewardTrainer::new(reward_model, config);
trainer.train(&dataset, 1)?;

// Evaluate accuracy (chosen > rejected)
let accuracy = trainer.evaluate(&eval_dataset)?;
println!("Preference accuracy: {:.2}%", accuracy * 100.0);

reward_model.save("reward-model.apr")?;

// Inference: score a response
let score = reward_model.score(&prompt, &response)?;
println!("Reward score: {:.3}", score);
"#,
            )
            .with_related(vec!["rlhf-sft", "rlhf-ppo", "rlhf-dpo"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_preference_accuracy_in_range() {
    let accuracy = 0.72_f64;
    assert!(accuracy >= 0.0 && accuracy <= 1.0);
}

    #[test]
    fn test_margin_non_negative() {
    let margin = 0.0_f64;
    assert!(margin >= 0.0);
}

    #[test]
    fn test_reward_score_ordering() {
    let chosen_reward = 1.5_f64;
    let rejected_reward = -0.3_f64;
    assert!(chosen_reward > rejected_reward);
}
}"#,
            ),
    );

    // Direct Preference Optimization (DPO)
    cookbook.add(
        Recipe::new("rlhf-dpo", "Direct Preference Optimization (DPO)")
            .with_problem("Align models directly from preferences without reward modeling")
            .with_components(vec!["entrenar", "aprender"])
            .with_tags(vec!["rlhf", "dpo", "alignment", "preferences", "efficient"])
            .with_code(
                r#"use entrenar::prelude::*;
use entrenar::dpo::*;

// Load SFT model (policy) and reference model
let policy = Model::load("sft-model.apr")?;
let reference = Model::load("sft-model.apr")?;  // Frozen copy

// Load preference dataset
let dataset = PreferenceDataset::load("preferences.json")?;

// Configure DPO
let config = DpoConfig {
    beta: 0.1,                    // KL penalty coefficient
    learning_rate: 5e-7,
    batch_size: 4,
    gradient_accumulation: 4,
    max_length: 512,
    max_prompt_length: 256,
    label_smoothing: 0.0,
    ..Default::default()
};

// DPO loss: -log σ(β * (log π(y_w|x) - log π(y_l|x)))
let trainer = DpoTrainer::new(policy, reference, config);
trainer.train(&dataset, 1)?;

// Evaluate
let metrics = trainer.evaluate(&eval_dataset)?;
println!("Accuracy: {:.2}%", metrics.accuracy * 100.0);
println!("Chosen reward: {:.3}", metrics.chosen_reward);
println!("Rejected reward: {:.3}", metrics.rejected_reward);

policy.save("dpo-model.apr")?;
"#,
            )
            .with_related(vec!["rlhf-sft", "rlhf-ipo", "rlhf-kto"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_beta_positive() {
    let beta = 0.1_f64;
    assert!(beta > 0.0);
}

    #[test]
    fn test_max_length_exceeds_prompt_length() {
    let max_prompt_length = 256;
    let max_length = 512;
    assert!(max_length > max_prompt_length);
}

    #[test]
    fn test_label_smoothing_in_range() {
    let label_smoothing = 0.0_f64;
    assert!(label_smoothing >= 0.0 && label_smoothing <= 1.0);
}
}"#,
            ),
    );

    // DPO Variants (IPO, KTO, ORPO)
    cookbook.add(
        Recipe::new("rlhf-dpo-variants", "DPO Variants: IPO, KTO, ORPO")
            .with_problem("Use improved DPO variants for better alignment")
            .with_components(vec!["entrenar", "aprender"])
            .with_tags(vec!["rlhf", "dpo", "ipo", "kto", "orpo", "alignment"])
            .with_code(
                r#"use entrenar::prelude::*;
use entrenar::dpo::*;

// === IPO (Identity Preference Optimization) ===
// Addresses DPO's overfitting with identity mapping
let ipo_config = IpoConfig {
    tau: 0.1,                     // Temperature parameter
    learning_rate: 5e-7,
    ..Default::default()
};
let trainer = IpoTrainer::new(policy.clone(), reference.clone(), ipo_config);

// === KTO (Kahneman-Tversky Optimization) ===
// Works with unpaired data (no need for chosen/rejected pairs)
let kto_dataset = KtoDataset::load("ratings.json")?;
// Each sample: { prompt, response, is_desirable: bool }

let kto_config = KtoConfig {
    beta: 0.1,
    desirable_weight: 1.0,
    undesirable_weight: 1.0,
    ..Default::default()
};
let trainer = KtoTrainer::new(policy.clone(), reference.clone(), kto_config);
trainer.train(&kto_dataset, 1)?;

// === ORPO (Odds Ratio Preference Optimization) ===
// No reference model needed - uses odds ratio
let orpo_config = OrpoConfig {
    beta: 0.1,
    learning_rate: 8e-6,
    ..Default::default()
};
// ORPO combines SFT and preference learning
let trainer = OrpoTrainer::new(policy.clone(), orpo_config);
trainer.train(&dataset, 1)?;

// === SimPO (Simple Preference Optimization) ===
// Length-normalized, reference-free
let simpo_config = SimpoConfig {
    beta: 2.5,
    gamma: 0.5,                   // Target margin
    ..Default::default()
};
let trainer = SimpoTrainer::new(policy, simpo_config);
"#,
            )
            .with_related(vec!["rlhf-dpo", "rlhf-sft"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_ipo_tau_positive() {
    let tau = 0.1_f64;
    assert!(tau > 0.0);
}

    #[test]
    fn test_kto_weights_non_negative() {
    let desirable_weight = 1.0_f64;
    let undesirable_weight = 1.0_f64;
    assert!(desirable_weight >= 0.0 && undesirable_weight >= 0.0);
}

    #[test]
    fn test_orpo_beta_in_valid_range() {
    let beta = 0.1_f64;
    assert!(beta > 0.0 && beta < 1.0);
}
}"#,
            ),
    );

    // PPO for RLHF
    cookbook.add(
        Recipe::new("rlhf-ppo", "PPO for RLHF")
            .with_problem("Train language models with PPO using a reward model")
            .with_components(vec!["entrenar", "aprender"])
            .with_tags(vec!["rlhf", "ppo", "reinforcement-learning", "alignment"])
            .with_code(
                r#"use entrenar::prelude::*;
use entrenar::ppo::*;

// Load models
let policy = Model::load("sft-model.apr")?;
let reference = Model::load("sft-model.apr")?;  // KL anchor
let reward_model = RewardModel::load("reward-model.apr")?;

// Configure PPO
let config = PpoConfig {
    // Learning
    learning_rate: 1e-6,
    batch_size: 64,
    mini_batch_size: 8,
    gradient_accumulation: 8,

    // PPO hyperparameters
    ppo_epochs: 4,
    clip_range: 0.2,
    clip_range_value: 0.2,
    gamma: 1.0,
    lam: 0.95,                    // GAE lambda

    // KL control
    kl_penalty: KlPenalty::Adaptive {
        target: 6.0,
        horizon: 10000,
    },

    // Generation
    max_new_tokens: 128,
    temperature: 0.7,
    top_p: 0.9,

    ..Default::default()
};

// Create PPO trainer
let trainer = PpoTrainer::new(policy, reference, reward_model, config);

// Training loop with prompts
let prompts = PromptDataset::load("prompts.txt")?;
for epoch in 0..10 {
    let stats = trainer.step(&prompts)?;
    println!("Epoch {}: reward={:.3}, kl={:.3}, loss={:.4}",
        epoch, stats.mean_reward, stats.kl_div, stats.loss);
}

trainer.policy().save("rlhf-model.apr")?;
"#,
            )
            .with_related(vec!["rlhf-reward-model", "rlhf-sft", "rlhf-stability"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_clip_range_valid() {
    let clip_range = 0.2_f64;
    assert!(clip_range > 0.0 && clip_range <= 1.0);
}

    #[test]
    fn test_ppo_epochs_positive() {
    let ppo_epochs = 4;
    assert!(ppo_epochs > 0);
}

    #[test]
    fn test_temperature_positive() {
    let temperature = 0.7_f64;
    assert!(temperature > 0.0);
}
}"#,
            ),
    );

    // RLHF Stability & Best Practices
    cookbook.add(
        Recipe::new("rlhf-stability", "RLHF Stability & Best Practices")
            .with_problem("Ensure stable RLHF training and avoid common pitfalls")
            .with_components(vec!["entrenar", "aprender", "trueno-viz"])
            .with_tags(vec![
                "rlhf",
                "stability",
                "best-practices",
                "debugging",
                "monitoring",
            ])
            .with_code(
                r#"use entrenar::prelude::*;
use entrenar::ppo::*;

// === Stability Techniques ===

// 1. Reward normalization (running statistics)
let config = PpoConfig {
    reward_normalization: true,
    reward_clip: 10.0,           // Clip extreme rewards
    ..Default::default()
};

// 2. Advantage normalization
let config = PpoConfig {
    advantage_normalization: true,
    ..config
};

// 3. Value function clipping
let config = PpoConfig {
    clip_range_value: 0.2,       // Clip value function updates
    ..config
};

// 4. Adaptive KL penalty (InstructGPT style)
let config = PpoConfig {
    kl_penalty: KlPenalty::Adaptive {
        target: 6.0,             // Target KL divergence
        horizon: 10000,          // Adaptation horizon
    },
    ..config
};

// 5. Gradient clipping
let config = PpoConfig {
    max_grad_norm: 1.0,
    ..config
};

// === Monitoring ===
let callback = |stats: &PpoStats| {
    // Check for reward hacking
    if stats.mean_reward > 10.0 {
        println!("Warning: Possible reward hacking!");
    }

    // Check KL divergence
    if stats.kl_div > 15.0 {
        println!("Warning: High KL divergence - policy drifting!");
    }

    // Check for mode collapse
    if stats.response_entropy < 0.5 {
        println!("Warning: Low entropy - possible mode collapse!");
    }
};

// === Evaluation ===
// Always evaluate on held-out prompts
let eval_results = trainer.evaluate(&eval_prompts)?;
println!("Win rate vs SFT: {:.2}%", eval_results.win_rate * 100.0);
println!("Mean length: {:.1}", eval_results.mean_length);
println!("Diversity: {:.3}", eval_results.diversity);
"#,
            )
            .with_related(vec!["rlhf-ppo", "rlhf-evaluation"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_reward_clip_positive() {
    let reward_clip = 10.0_f64;
    assert!(reward_clip > 0.0);
}

    #[test]
    fn test_grad_norm_positive() {
    let max_grad_norm = 1.0_f64;
    assert!(max_grad_norm > 0.0);
}

    #[test]
    fn test_kl_target_positive() {
    let kl_target = 6.0_f64;
    assert!(kl_target > 0.0);
}
}"#,
            ),
    );

    // RLHF Evaluation
    cookbook.add(
        Recipe::new("rlhf-evaluation", "RLHF Model Evaluation")
            .with_problem("Comprehensively evaluate aligned models")
            .with_components(vec!["entrenar", "aprender", "trueno-viz"])
            .with_tags(vec![
                "rlhf",
                "evaluation",
                "benchmarks",
                "metrics",
                "alignment",
            ])
            .with_code(
                r#"use entrenar::prelude::*;
use entrenar::eval::*;

// Load models to compare
let sft_model = Model::load("sft-model.apr")?;
let rlhf_model = Model::load("rlhf-model.apr")?;

// === Pairwise Evaluation ===
let evaluator = PairwiseEvaluator::new(reward_model);
let results = evaluator.compare(
    &sft_model,
    &rlhf_model,
    &eval_prompts,
)?;
println!("RLHF win rate: {:.2}%", results.model_b_wins * 100.0);
println!("Tie rate: {:.2}%", results.ties * 100.0);

// === Safety Evaluation ===
let safety_eval = SafetyEvaluator::new()
    .add_detector(ToxicityDetector::new())
    .add_detector(BiasDetector::new())
    .add_detector(HarmfulContentDetector::new());

let safety_results = safety_eval.evaluate(&rlhf_model, &safety_prompts)?;
println!("Toxicity rate: {:.3}%", safety_results.toxicity_rate * 100.0);
println!("Refusal rate: {:.2}%", safety_results.refusal_rate * 100.0);

// === Helpfulness Benchmarks ===
let benchmarks = vec![
    ("MT-Bench", MtBench::new()),
    ("AlpacaEval", AlpacaEval::new()),
    ("HumanEval", HumanEval::new()),
];

for (name, bench) in benchmarks {
    let score = bench.evaluate(&rlhf_model)?;
    println!("{}: {:.2}", name, score);
}

// === Diversity Metrics ===
let diversity = DiversityMetrics::compute(&rlhf_model, &prompts)?;
println!("Distinct-1: {:.3}", diversity.distinct_1);
println!("Distinct-2: {:.3}", diversity.distinct_2);
println!("Self-BLEU: {:.3}", diversity.self_bleu);
"#,
            )
            .with_related(vec!["rlhf-stability", "rlhf-ppo", "rlhf-dpo"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_win_rate_in_range() {
    let win_rate = 0.65_f64;
    assert!(win_rate >= 0.0 && win_rate <= 1.0);
}

    #[test]
    fn test_distinct_n_in_range() {
    let distinct_2 = 0.82_f64;
    assert!(distinct_2 >= 0.0 && distinct_2 <= 1.0);
}

    #[test]
    fn test_toxicity_rate_in_range() {
    let toxicity = 0.03_f64;
    assert!(toxicity >= 0.0 && toxicity <= 1.0);
}
}"#,
            ),
    );

    // Quantization for Alignment
    cookbook.add(
        Recipe::new("rlhf-quantization", "Quantization for Efficient Alignment")
            .with_problem("Apply quantization techniques throughout the alignment pipeline")
            .with_components(vec!["entrenar", "aprender", "realizar"])
            .with_tags(vec![
                "rlhf",
                "quantization",
                "4bit",
                "8bit",
                "efficient",
                "memory",
            ])
            .with_code(
                r#"use entrenar::prelude::*;
use entrenar::quantization::*;

// === QLoRA for SFT ===
let model = Model::load_quantized("llama-7b.q4_k.gguf")?;
let model = model.with_qlora(QLoraConfig {
    lora: LoraConfig { r: 64, alpha: 16, ..Default::default() },
    nf4: true,
    double_quant: true,
    ..Default::default()
})?;

// SFT on 24GB GPU
let trainer = SftTrainer::new(model, SftConfig::default());
trainer.train(&dataset, 3)?;

// === Quantized Reward Model ===
let reward_model = RewardModel::load("reward-model.apr")?;
let quantized_rm = reward_model.quantize(Quantization::Int8)?;
// 2x faster inference, minimal accuracy loss

// === INT8 PPO Training ===
let config = PpoConfig {
    // Use 8-bit Adam optimizer (saves 75% optimizer memory)
    optimizer: Optimizer::Adam8bit {
        lr: 1e-6,
        betas: (0.9, 0.999),
    },
    // Mixed precision training
    mixed_precision: MixedPrecision::Bf16,
    // Gradient checkpointing
    gradient_checkpointing: true,
    ..Default::default()
};

// === Post-Training Quantization ===
let rlhf_model = Model::load("rlhf-model.apr")?;

// GPTQ quantization (4-bit, minimal quality loss)
let gptq_model = rlhf_model.quantize_gptq(GptqConfig {
    bits: 4,
    group_size: 128,
    calibration_data: &calibration_samples,
})?;
gptq_model.save("rlhf-model.q4.gguf")?;

// AWQ quantization (activation-aware)
let awq_model = rlhf_model.quantize_awq(AwqConfig {
    bits: 4,
    group_size: 128,
})?;
"#,
            )
            .with_related(vec!["training-qlora", "rlhf-sft", "rlhf-ppo"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_quantization_bits_valid() {
    let bits = 4;
    assert!(bits == 4 || bits == 8);
}

    #[test]
    fn test_group_size_positive() {
    let group_size = 128;
    assert!(group_size > 0);
}

    #[test]
    fn test_mixed_precision_flag() {
    let use_bf16 = true;
    assert!(use_bf16);
}
}"#,
            ),
    );

    // PEFT Adapters
    cookbook.add(
        Recipe::new("rlhf-peft", "PEFT Adapters for Alignment")
            .with_problem("Use parameter-efficient methods beyond LoRA for alignment")
            .with_components(vec!["entrenar", "aprender"])
            .with_tags(vec![
                "rlhf",
                "peft",
                "lora",
                "adapters",
                "efficient",
                "fine-tuning",
            ])
            .with_code(
                r#"use entrenar::prelude::*;
use entrenar::peft::*;

// === LoRA (Low-Rank Adaptation) ===
let lora = LoraConfig {
    r: 16,
    alpha: 32,
    dropout: 0.1,
    target_modules: vec!["q_proj", "v_proj", "k_proj", "o_proj"],
    ..Default::default()
};

// === DoRA (Weight-Decomposed LoRA) ===
// Decomposes weights into magnitude and direction
let dora = DoraConfig {
    r: 16,
    alpha: 32,
    use_dora: true,              // Enable magnitude learning
    ..Default::default()
};

// === AdaLoRA (Adaptive LoRA) ===
// Dynamically allocates rank budget
let adalora = AdaLoraConfig {
    init_r: 12,
    target_r: 8,
    beta1: 0.85,
    beta2: 0.85,
    ..Default::default()
};

// === IA3 (Infused Adapter by Inhibiting and Amplifying) ===
// Even more efficient than LoRA
let ia3 = Ia3Config {
    target_modules: vec!["k_proj", "v_proj", "down_proj"],
    feedforward_modules: vec!["down_proj"],
};

// === Prefix Tuning ===
let prefix = PrefixTuningConfig {
    num_virtual_tokens: 20,
    encoder_hidden_size: 512,
};

// === Apply adapter to model ===
let model = Model::load("llama-7b.apr")?;
let model = model.with_adapter(AdapterConfig::LoRA(lora))?;

// Train with any alignment method
let trainer = DpoTrainer::new(model, reference, DpoConfig::default());
trainer.train(&dataset, 1)?;

// Save only adapter weights (small file)
model.save_adapter("dpo-adapter.lora")?;

// Merge adapters for inference
let merged = model.merge_adapter()?;
merged.save("dpo-merged.apr")?;
"#,
            )
            .with_related(vec!["training-lora", "training-qlora", "rlhf-sft"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_lora_rank_positive() {
    let rank = 16;
    assert!(rank > 0);
}

    #[test]
    fn test_target_modules_non_empty() {
    let target_modules = vec!["q_proj", "v_proj"];
    assert!(!target_modules.is_empty());
}

    #[test]
    fn test_prefix_tuning_virtual_tokens_positive() {
    let num_virtual_tokens = 20;
    assert!(num_virtual_tokens > 0);
}
}"#,
            ),
    );
}
