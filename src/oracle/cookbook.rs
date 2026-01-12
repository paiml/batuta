//! Cookbook - Practical recipes for common Sovereign AI Stack patterns
//!
//! Each recipe includes:
//! - Problem description
//! - Components involved
//! - Code example
//! - Related recipes

use serde::{Deserialize, Serialize};

/// A cookbook recipe for a common pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recipe {
    /// Unique recipe ID
    pub id: String,
    /// Recipe title
    pub title: String,
    /// Problem this recipe solves
    pub problem: String,
    /// Components used
    pub components: Vec<String>,
    /// Tags for discovery
    pub tags: Vec<String>,
    /// Complete code example
    pub code: String,
    /// Related recipe IDs
    pub related: Vec<String>,
}

impl Recipe {
    pub fn new(id: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            problem: String::new(),
            components: Vec::new(),
            tags: Vec::new(),
            code: String::new(),
            related: Vec::new(),
        }
    }

    pub fn with_problem(mut self, problem: impl Into<String>) -> Self {
        self.problem = problem.into();
        self
    }

    pub fn with_components(mut self, components: Vec<&str>) -> Self {
        self.components = components.into_iter().map(String::from).collect();
        self
    }

    pub fn with_tags(mut self, tags: Vec<&str>) -> Self {
        self.tags = tags.into_iter().map(String::from).collect();
        self
    }

    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = code.into();
        self
    }

    pub fn with_related(mut self, related: Vec<&str>) -> Self {
        self.related = related.into_iter().map(String::from).collect();
        self
    }
}

/// Cookbook containing all recipes
#[derive(Debug, Clone, Default)]
pub struct Cookbook {
    recipes: Vec<Recipe>,
}

impl Cookbook {
    /// Create a new empty cookbook
    pub fn new() -> Self {
        Self::default()
    }

    /// Create cookbook with all standard recipes
    pub fn standard() -> Self {
        let mut cookbook = Self::new();
        cookbook.register_wasm_recipes();
        cookbook.register_ml_recipes();
        cookbook.register_transpilation_recipes();
        cookbook.register_distributed_recipes();
        cookbook.register_quality_recipes();
        cookbook.register_speech_recipes();
        cookbook.register_training_recipes();
        cookbook.register_data_recipes();
        cookbook.register_registry_recipes();
        cookbook.register_rag_recipes();
        cookbook.register_viz_recipes();
        cookbook
    }

    /// Get all recipes
    pub fn recipes(&self) -> &[Recipe] {
        &self.recipes
    }

    /// Find recipes by tag
    pub fn find_by_tag(&self, tag: &str) -> Vec<&Recipe> {
        self.recipes
            .iter()
            .filter(|r| r.tags.iter().any(|t| t.eq_ignore_ascii_case(tag)))
            .collect()
    }

    /// Find recipes by component
    pub fn find_by_component(&self, component: &str) -> Vec<&Recipe> {
        self.recipes
            .iter()
            .filter(|r| {
                r.components
                    .iter()
                    .any(|c| c.eq_ignore_ascii_case(component))
            })
            .collect()
    }

    /// Get recipe by ID
    pub fn get(&self, id: &str) -> Option<&Recipe> {
        self.recipes.iter().find(|r| r.id == id)
    }

    /// Search recipes by keyword
    pub fn search(&self, query: &str) -> Vec<&Recipe> {
        let query_lower = query.to_lowercase();
        self.recipes
            .iter()
            .filter(|r| {
                r.title.to_lowercase().contains(&query_lower)
                    || r.problem.to_lowercase().contains(&query_lower)
                    || r.tags
                        .iter()
                        .any(|t| t.to_lowercase().contains(&query_lower))
            })
            .collect()
    }

    // =========================================================================
    // WASM Recipes
    // =========================================================================

    fn register_wasm_recipes(&mut self) {
        // Zero-JS WASM Architecture
        self.recipes.push(
            Recipe::new("wasm-zero-js", "Zero-JS WASM Application")
                .with_problem("Build interactive web apps with pure Rust/WASM, eliminating JavaScript entirely")
                .with_components(vec!["simular", "trueno", "web-sys", "wasm-bindgen"])
                .with_tags(vec!["wasm", "zero-js", "web", "canvas", "animation"])
                .with_code(r##"// Cargo.toml
[lib]
crate-type = ["cdylib"]

[dependencies]
wasm-bindgen = "0.2"
web-sys = { version = "0.3", features = [
    "Window", "Document", "HtmlCanvasElement",
    "CanvasRenderingContext2d", "Element"
]}
console_error_panic_hook = "0.1"

// src/lib.rs
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

struct AppState {
    canvas: web_sys::HtmlCanvasElement,
    ctx: web_sys::CanvasRenderingContext2d,
    frame: u32,
}

impl AppState {
    fn tick(&mut self) {
        self.frame += 1;
        self.render();
    }

    fn render(&self) {
        let w = self.canvas.width() as f64;
        let h = self.canvas.height() as f64;

        // Clear
        self.ctx.set_fill_style_str("#0a0a1a");
        self.ctx.fill_rect(0.0, 0.0, w, h);

        // Draw
        self.ctx.set_fill_style_str("#4ecdc4");
        self.ctx.begin_path();
        self.ctx.arc(w/2.0, h/2.0, 50.0, 0.0, std::f64::consts::PI * 2.0).unwrap();
        self.ctx.fill();
    }
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window().unwrap()
        .request_animation_frame(f.as_ref().unchecked_ref()).unwrap();
}

#[wasm_bindgen(js_name = initApp)]
pub fn init_app() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();

    let window = web_sys::window().expect("no window");
    let document = window.document().expect("no document");
    let canvas = document.get_element_by_id("canvas")
        .expect("no canvas")
        .dyn_into::<web_sys::HtmlCanvasElement>()?;

    let ctx = canvas.get_context("2d")?.unwrap()
        .dyn_into::<web_sys::CanvasRenderingContext2d>()?;

    let state = Rc::new(RefCell::new(AppState { canvas, ctx, frame: 0 }));

    // Animation loop
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = Rc::clone(&f);
    let state_clone = Rc::clone(&state);

    *g.borrow_mut() = Some(Closure::new(move || {
        state_clone.borrow_mut().tick();
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));

    request_animation_frame(g.borrow().as_ref().unwrap());
    Ok(())
}

// index.html - ONE LINE OF JAVASCRIPT
// <script type="module">import init, { initApp } from './pkg/app.js'; init().then(initApp);</script>
"##)
                .with_related(vec!["wasm-event-handling", "wasm-canvas-rendering"]),
        );

        // WASM Event Handling
        self.recipes.push(
            Recipe::new("wasm-event-handling", "WASM Event Handling")
                .with_problem("Handle DOM events (click, input, keypress) in pure Rust")
                .with_components(vec!["simular", "web-sys", "wasm-bindgen"])
                .with_tags(vec!["wasm", "events", "dom", "closure"])
                .with_code(r##"use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

fn setup_button<F>(document: &web_sys::Document, id: &str, mut callback: F) -> Result<(), JsValue>
where
    F: FnMut() + 'static,
{
    if let Some(btn) = document.get_element_by_id(id) {
        let closure = Closure::wrap(Box::new(move |_: web_sys::Event| {
            callback();
        }) as Box<dyn FnMut(_)>);

        btn.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
        closure.forget(); // Prevent closure from being dropped
    }
    Ok(())
}

fn setup_slider(document: &web_sys::Document, id: &str, state: Rc<RefCell<AppState>>) -> Result<(), JsValue> {
    if let Some(slider) = document.get_element_by_id(id) {
        let closure = Closure::wrap(Box::new(move |e: web_sys::Event| {
            if let Some(target) = e.target() {
                if let Some(input) = target.dyn_ref::<web_sys::HtmlInputElement>() {
                    if let Ok(value) = input.value().parse::<u32>() {
                        state.borrow_mut().set_speed(value);
                    }
                }
            }
        }) as Box<dyn FnMut(_)>);

        slider.add_event_listener_with_callback("input", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }
    Ok(())
}
"##)
                .with_related(vec!["wasm-zero-js", "wasm-canvas-rendering"]),
        );

        // WASM Canvas Rendering
        self.recipes.push(
            Recipe::new("wasm-canvas-rendering", "WASM Canvas 2D Rendering")
                .with_problem("Render graphics to HTML5 Canvas from Rust/WASM")
                .with_components(vec!["simular", "web-sys"])
                .with_tags(vec!["wasm", "canvas", "graphics", "rendering"])
                .with_code(
                    r##"use web_sys::CanvasRenderingContext2d;

fn render(ctx: &CanvasRenderingContext2d, w: f64, h: f64, trail: &[(f64, f64)]) {
    // Clear background
    ctx.set_fill_style_str("#0f0f23");
    ctx.fill_rect(0.0, 0.0, w, h);

    // Draw grid
    ctx.set_stroke_style_str("#1a1a2e");
    ctx.set_line_width(1.0);
    for i in 1..=10 {
        let x = w * (i as f64) / 10.0;
        ctx.begin_path();
        ctx.move_to(x, 0.0);
        ctx.line_to(x, h);
        ctx.stroke();
    }

    // Draw circle with glow
    ctx.set_fill_style_str("#ffd93d");
    ctx.begin_path();
    ctx.arc(w/2.0, h/2.0, 15.0, 0.0, std::f64::consts::PI * 2.0).unwrap();
    ctx.fill();

    // Semi-transparent glow
    ctx.set_global_alpha(0.3);
    ctx.begin_path();
    ctx.arc(w/2.0, h/2.0, 25.0, 0.0, std::f64::consts::PI * 2.0).unwrap();
    ctx.fill();
    ctx.set_global_alpha(1.0);

    // Draw trail with fading alpha
    ctx.begin_path();
    if let Some((x, y)) = trail.first() {
        ctx.move_to(*x, *y);
    }
    for (i, (x, y)) in trail.iter().enumerate().skip(1) {
        let alpha = (i as f64) / (trail.len() as f64) * 0.5;
        ctx.set_global_alpha(alpha);
        ctx.line_to(*x, *y);
    }
    ctx.set_global_alpha(1.0);
    ctx.stroke();

    // Text labels
    ctx.set_fill_style_str("#888888");
    ctx.set_font("12px monospace");
    ctx.fill_text("Label", 100.0, 100.0).unwrap();
}
"##,
                )
                .with_related(vec!["wasm-zero-js", "wasm-event-handling"]),
        );
    }

    // =========================================================================
    // ML Recipes
    // =========================================================================

    fn register_ml_recipes(&mut self) {
        // Random Forest Classification
        self.recipes.push(
            Recipe::new("ml-random-forest", "Random Forest Classification")
                .with_problem("Train a random forest classifier and export for serving")
                .with_components(vec!["aprender", "realizar", "alimentar"])
                .with_tags(vec!["ml", "classification", "random-forest", "supervised"])
                .with_code(
                    r##"use aprender::prelude::*;
use alimentar::CsvReader;

// Load data
let data = CsvReader::from_path("data.csv")?.load()?;
let (X, y) = data.split_target("label")?;

// Train-test split
let (X_train, X_test, y_train, y_test) = train_test_split(&X, &y, 0.2, Some(42))?;

// Train model
let model = RandomForestClassifier::new()
    .n_estimators(100)
    .max_depth(Some(10))
    .min_samples_split(5)
    .fit(&X_train, &y_train)?;

// Evaluate
let predictions = model.predict(&X_test)?;
let accuracy = accuracy_score(&y_test, &predictions);
println!("Accuracy: {:.2}%", accuracy * 100.0);

// Export for serving
model.save_apr("model.apr")?;

// Load in realizar for inference
// realizar serve --model model.apr --port 8080
"##,
                )
                .with_related(vec!["ml-serving", "ml-preprocessing"]),
        );

        // Model Serving
        self.recipes.push(
            Recipe::new("ml-serving", "Model Serving with Realizar")
                .with_problem("Deploy trained models as HTTP API or Lambda function")
                .with_components(vec!["realizar", "aprender"])
                .with_tags(vec!["ml", "serving", "inference", "api", "lambda"])
                .with_code(
                    r##"// Command line serving
// realizar serve --model model.apr --port 8080

// Programmatic serving
use realizar::prelude::*;

let model = Model::load("model.apr")?;
let server = Server::new(model)
    .port(8080)
    .batch_size(32)
    .workers(4);

server.run()?;

// Lambda deployment
// realizar package --model model.apr --output lambda.zip
// aws lambda create-function --function-name my-model \
//     --runtime provided.al2 --handler bootstrap \
//     --zip-file fileb://lambda.zip

// Request format
// POST /predict
// Content-Type: application/json
// {"features": [1.0, 2.0, 3.0, 4.0]}
"##,
                )
                .with_related(vec!["ml-random-forest", "distributed-inference"]),
        );
    }

    // =========================================================================
    // Transpilation Recipes
    // =========================================================================

    fn register_transpilation_recipes(&mut self) {
        // Python to Rust
        self.recipes.push(
            Recipe::new("transpile-python", "Python to Rust Migration")
                .with_problem("Convert Python ML code to Rust using depyler")
                .with_components(vec!["depyler", "aprender", "trueno", "batuta"])
                .with_tags(vec!["transpilation", "python", "migration"])
                .with_code(
                    r##"# Original Python code (sklearn_model.py)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

X = np.random.rand(100, 4)
y = np.random.randint(0, 2, 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Transpile with depyler
# batuta transpile sklearn_model.py --output src/model.rs

// Generated Rust code (src/model.rs)
use aprender::prelude::*;
use trueno::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let X = Tensor::rand(&[100, 4]);
    let y = Tensor::randint(0, 2, &[100]);

    let (X_train, X_test, y_train, y_test) = train_test_split(&X, &y, 0.2, None)?;
    let model = RandomForestClassifier::new()
        .n_estimators(100)
        .fit(&X_train, &y_train)?;

    Ok(())
}
"##,
                )
                .with_related(vec!["transpile-numpy", "quality-golden-trace"]),
        );

        // NumPy to Trueno
        self.recipes.push(
            Recipe::new("transpile-numpy", "NumPy to Trueno Conversion")
                .with_problem("Convert NumPy operations to SIMD-accelerated Trueno")
                .with_components(vec!["depyler", "trueno"])
                .with_tags(vec!["transpilation", "numpy", "simd", "tensors"])
                .with_code(
                    r##"# Python NumPy
import numpy as np
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
dot = np.dot(a, b)
matmul = np.matmul(X, W)

// Rust Trueno (SIMD-accelerated)
use trueno::prelude::*;

let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0]);
let dot = a.dot(&b);  // SIMD auto-vectorized
let matmul = X.matmul(&W);  // GPU if available
"##,
                )
                .with_related(vec!["transpile-python"]),
        );
    }

    // =========================================================================
    // Distributed Recipes
    // =========================================================================

    fn register_distributed_recipes(&mut self) {
        // Work-Stealing Distribution
        self.recipes.push(
            Recipe::new(
                "distributed-work-stealing",
                "Work-Stealing Task Distribution",
            )
            .with_problem("Distribute computation across CPU cores with work-stealing")
            .with_components(vec!["repartir", "trueno"])
            .with_tags(vec!["distributed", "parallel", "work-stealing", "cpu"])
            .with_code(
                r##"use repartir::prelude::*;

// Create pool with work-stealing scheduler
let pool = Pool::builder()
    .num_workers(num_cpus::get())
    .build()?;

// Submit tasks
let results: Vec<f64> = pool.map(data.chunks(1000), |chunk| {
    // Each chunk processed by a worker
    chunk.iter().map(|x| x * x).sum()
})?;

// Reduce results
let total: f64 = results.iter().sum();
"##,
            )
            .with_related(vec!["distributed-gpu", "distributed-remote"]),
        );

        // GPU Distribution
        self.recipes.push(
            Recipe::new("distributed-gpu", "GPU Compute Distribution")
                .with_problem("Offload computation to GPU using wgpu")
                .with_components(vec!["repartir", "trueno"])
                .with_tags(vec!["distributed", "gpu", "wgpu", "compute"])
                .with_code(
                    r##"use repartir::prelude::*;
use trueno::prelude::*;

// Create GPU executor
let executor = GpuExecutor::new()?;

// GPU-accelerated matrix multiplication
let result = executor.submit(|gpu| {
    let a = gpu.tensor(&matrix_a)?;
    let b = gpu.tensor(&matrix_b)?;
    a.matmul(&b)
})?;

// Hybrid CPU/GPU
let pool = Pool::builder()
    .add_gpu_executor(executor)
    .add_cpu_executor(CpuExecutor::new())
    .build()?;
"##,
                )
                .with_related(vec!["distributed-work-stealing"]),
        );
    }

    // =========================================================================
    // Quality Recipes
    // =========================================================================

    fn register_quality_recipes(&mut self) {
        // EDD Testing Pattern
        self.recipes.push(
            Recipe::new("quality-edd", "Equation-Driven Development")
                .with_problem("Implement simulations with verifiable governing equations")
                .with_components(vec!["simular", "probar", "certeza"])
                .with_tags(vec![
                    "quality",
                    "testing",
                    "edd",
                    "simulation",
                    "falsification",
                ])
                .with_code(
                    r##"use simular::prelude::*;

/// EDD Demo following the complete cycle:
/// 1. Equation - Define governing equation
/// 2. Failing Test - Test fails without implementation
/// 3. Implementation - Implement the simulation
/// 4. Verification - Test passes, equation verified
/// 5. Falsification - Demonstrate conditions that break

pub trait DemoEngine {
    /// Load from YAML configuration (YAML-first)
    fn from_yaml(yaml: &str) -> Result<Self, DemoError> where Self: Sized;

    /// Advance simulation by one timestep
    fn step(&mut self, dt: f64) -> StepResult;

    /// Verify governing equation invariants
    fn verify_invariants(&self) -> bool;

    /// Get falsification status
    fn falsification_status(&self) -> FalsificationStatus;
}

// Example: Harmonic Oscillator
// Governing equation: E = (1/2)kx² + (1/2)mv²
// Falsification: Energy drift > tolerance indicates integrator failure

impl DemoEngine for HarmonicOscillator {
    fn verify_invariants(&self) -> bool {
        let energy = self.kinetic_energy() + self.potential_energy();
        let error = (energy - self.initial_energy).abs() / self.initial_energy;
        error < 1e-9  // Energy conservation within tolerance
    }
}
"##,
                )
                .with_related(vec!["quality-probar", "quality-golden-trace"]),
        );

        // Probar Testing
        self.recipes.push(
            Recipe::new("quality-probar", "Probar Property-Based Testing")
                .with_problem("Validate WASM demos with property-based and GUI coverage testing")
                .with_components(vec!["probar", "simular", "certeza"])
                .with_tags(vec![
                    "quality",
                    "testing",
                    "probar",
                    "property-testing",
                    "gui-coverage",
                ])
                .with_code(
                    r##"use probar::prelude::*;

// Property-based tests
#[probar::property]
fn prop_tour_length_positive(cities: Vec<City>) -> bool {
    let tour = solve_tsp(&cities);
    tour.length() > 0.0
}

#[probar::property]
fn prop_energy_conserved(dt: f64) -> bool {
    let mut sim = OrbitSimulation::new();
    let e0 = sim.total_energy();
    sim.step(dt);
    let e1 = sim.total_energy();
    (e1 - e0).abs() / e0.abs() < 1e-9
}

// Metamorphic relations
#[probar::metamorphic]
fn mr_scale_invariance(scale: f64, cities: Vec<City>) {
    let tour1 = solve_tsp(&cities);
    let scaled = cities.iter().map(|c| c * scale).collect();
    let tour2 = solve_tsp(&scaled);
    assert!((tour2.length() / tour1.length() - scale).abs() < 1e-6);
}

// GUI coverage
#[probar::gui_coverage]
fn test_canvas_coverage(app: &mut App) {
    app.click_button("start");
    app.wait_frames(100);
    let coverage = app.pixel_coverage();
    assert!(coverage > 0.8, "Must render to >80% of canvas");
}
"##,
                )
                .with_related(vec!["quality-edd", "quality-certeza"]),
        );

        // Golden Trace Comparison
        self.recipes.push(
            Recipe::new("quality-golden-trace", "Golden Trace Validation")
                .with_problem("Validate transpiled code produces identical behavior to original")
                .with_components(vec!["renacer", "certeza"])
                .with_tags(vec!["quality", "validation", "trace", "transpilation"])
                .with_code(
                    r##"use renacer::prelude::*;

// Capture golden trace from Python
// renacer trace python sklearn_model.py --output golden.trace

// Capture Rust trace
// renacer trace ./target/release/model --output rust.trace

// Compare traces
let comparison = renacer::compare("golden.trace", "rust.trace")?;

assert!(comparison.semantically_equivalent(),
    "Transpiled code must produce identical results");
assert!(comparison.syscall_compatible(),
    "System calls must match (file I/O, network, etc.)");

// Performance comparison
println!("Python: {:.2}ms", comparison.baseline_time_ms());
println!("Rust: {:.2}ms", comparison.target_time_ms());
println!("Speedup: {:.1}x", comparison.speedup());
"##,
                )
                .with_related(vec!["transpile-python", "quality-edd"]),
        );
    }

    // =========================================================================
    // Speech Recognition Recipes
    // =========================================================================

    fn register_speech_recipes(&mut self) {
        // Whisper ASR
        self.recipes.push(
            Recipe::new("speech-whisper", "Whisper Speech Recognition")
                .with_problem("Transcribe audio to text using pure-Rust Whisper implementation")
                .with_components(vec!["whisper-apr", "aprender", "trueno"])
                .with_tags(vec!["speech", "asr", "whisper", "transcription", "wasm"])
                .with_code(
                    r##"use whisper_apr::prelude::*;

// Load model (downloads from HuggingFace on first run)
let model = WhisperModel::load("tiny.en")?;

// Transcribe audio file
let audio = Audio::load("recording.wav")?;
let result = model.transcribe(&audio)?;

println!("Transcription: {}", result.text);

// With timestamps
for segment in result.segments {
    println!("[{:.2}s - {:.2}s] {}",
        segment.start, segment.end, segment.text);
}

// Streaming transcription (real-time)
let mut stream = model.stream()?;
for chunk in audio_chunks {
    if let Some(text) = stream.process(&chunk)? {
        print!("{}", text);
    }
}

// WASM deployment
// wasm-pack build --target web --features wasm
// <script type="module">
//   import init, { transcribe } from './pkg/whisper_apr.js';
//   await init();
//   const text = await transcribe(audioBuffer);
// </script>
"##,
                )
                .with_related(vec!["speech-streaming", "ml-serving"]),
        );

        // Streaming Speech
        self.recipes.push(
            Recipe::new("speech-streaming", "Real-time Speech Streaming")
                .with_problem("Process audio in real-time with low latency")
                .with_components(vec!["whisper-apr", "trueno"])
                .with_tags(vec!["speech", "streaming", "real-time", "low-latency"])
                .with_code(
                    r##"use whisper_apr::streaming::*;

// Configure streaming decoder
let config = StreamConfig {
    chunk_size_ms: 500,      // Process every 500ms
    overlap_ms: 100,         // Overlap for continuity
    vad_enabled: true,       // Voice activity detection
    language: Some("en"),
};

let mut decoder = StreamingDecoder::new(model, config)?;

// Process microphone input
let mut mic = Microphone::open()?;
loop {
    let samples = mic.read_chunk()?;

    // VAD filters silence
    if let Some(result) = decoder.process(&samples)? {
        // Partial results for UI feedback
        if result.is_partial {
            print!("\r{}", result.text);
        } else {
            println!("\n[Final] {}", result.text);
        }
    }
}
"##,
                )
                .with_related(vec!["speech-whisper"]),
        );
    }

    // =========================================================================
    // Training Recipes
    // =========================================================================

    fn register_training_recipes(&mut self) {
        // LoRA Fine-tuning
        self.recipes.push(
            Recipe::new("training-lora", "LoRA Fine-tuning")
                .with_problem("Fine-tune large models efficiently with Low-Rank Adaptation")
                .with_components(vec!["entrenar", "aprender", "alimentar"])
                .with_tags(vec!["training", "lora", "fine-tuning", "efficient", "llm"])
                .with_code(
                    r##"use entrenar::prelude::*;

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
"##,
                )
                .with_related(vec!["training-qlora", "training-autograd"]),
        );

        // QLoRA
        self.recipes.push(
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
                    r##"use entrenar::prelude::*;

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
"##,
                )
                .with_related(vec!["training-lora"]),
        );

        // Autograd
        self.recipes.push(
            Recipe::new("training-autograd", "Custom Training with Autograd")
                .with_problem("Build custom neural networks with automatic differentiation")
                .with_components(vec!["entrenar", "trueno"])
                .with_tags(vec!["training", "autograd", "neural-network", "custom"])
                .with_code(
                    r##"use entrenar::autograd::*;

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
"##,
                )
                .with_related(vec!["training-lora", "ml-random-forest"]),
        );
    }

    // =========================================================================
    // Data Loading Recipes
    // =========================================================================

    fn register_data_recipes(&mut self) {
        // Alimentar Data Loading
        self.recipes.push(
            Recipe::new("data-alimentar", "Zero-Copy Data Loading")
                .with_problem("Load large datasets efficiently with memory mapping")
                .with_components(vec!["alimentar", "trueno"])
                .with_tags(vec!["data", "loading", "parquet", "arrow", "zero-copy"])
                .with_code(
                    r##"use alimentar::prelude::*;

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
"##,
                )
                .with_related(vec!["data-preprocessing", "ml-random-forest"]),
        );

        // Data Preprocessing
        self.recipes.push(
            Recipe::new("data-preprocessing", "Data Preprocessing Pipeline")
                .with_problem("Build reproducible preprocessing pipelines")
                .with_components(vec!["alimentar", "aprender"])
                .with_tags(vec!["data", "preprocessing", "pipeline", "transforms"])
                .with_code(
                    r##"use alimentar::prelude::*;
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
"##,
                )
                .with_related(vec!["data-alimentar"]),
        );
    }

    // =========================================================================
    // Model Registry Recipes
    // =========================================================================

    fn register_registry_recipes(&mut self) {
        // Pacha Model Registry
        self.recipes.push(
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
                    r##"use pacha::prelude::*;

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
"##,
                )
                .with_related(vec!["registry-hf", "ml-serving"]),
        );

        // HuggingFace Integration
        self.recipes.push(
            Recipe::new("registry-hf", "HuggingFace Hub Integration")
                .with_problem("Download and cache models from HuggingFace Hub")
                .with_components(vec!["hf-hub", "aprender", "realizar"])
                .with_tags(vec!["registry", "huggingface", "download", "cache"])
                .with_code(
                    r##"use hf_hub::api::sync::Api;

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
"##,
                )
                .with_related(vec!["registry-pacha", "speech-whisper"]),
        );
    }

    // =========================================================================
    // RAG Pipeline Recipes
    // =========================================================================

    fn register_rag_recipes(&mut self) {
        // RAG Pipeline
        self.recipes.push(
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
                    r##"use trueno_rag::prelude::*;

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
"##,
                )
                .with_related(vec!["rag-semantic-search", "ml-serving"]),
        );

        // Semantic Search
        self.recipes.push(
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
                    r##"use trueno_db::prelude::*;
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
"##,
                )
                .with_related(vec!["rag-pipeline"]),
        );
    }

    // =========================================================================
    // Visualization Recipes
    // =========================================================================

    fn register_viz_recipes(&mut self) {
        // Terminal Visualization
        self.recipes.push(
            Recipe::new("viz-terminal", "Terminal Visualization")
                .with_problem("Create charts and plots in the terminal")
                .with_components(vec!["trueno-viz"])
                .with_tags(vec!["visualization", "terminal", "charts", "ascii"])
                .with_code(
                    r##"use trueno_viz::prelude::*;

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
"##,
                )
                .with_related(vec!["viz-png", "training-autograd"]),
        );

        // PNG Export
        self.recipes.push(
            Recipe::new("viz-png", "PNG Chart Export")
                .with_problem("Export publication-quality charts as PNG images")
                .with_components(vec!["trueno-viz"])
                .with_tags(vec!["visualization", "png", "export", "charts"])
                .with_code(
                    r##"use trueno_viz::prelude::*;

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
"##,
                )
                .with_related(vec!["viz-terminal"]),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cookbook_standard() {
        let cookbook = Cookbook::standard();
        assert!(!cookbook.recipes().is_empty());
    }

    #[test]
    fn test_find_by_tag() {
        let cookbook = Cookbook::standard();
        let wasm_recipes = cookbook.find_by_tag("wasm");
        assert!(!wasm_recipes.is_empty());
    }

    #[test]
    fn test_find_by_component() {
        let cookbook = Cookbook::standard();
        let simular_recipes = cookbook.find_by_component("simular");
        assert!(!simular_recipes.is_empty());
    }

    #[test]
    fn test_get_by_id() {
        let cookbook = Cookbook::standard();
        let recipe = cookbook.get("wasm-zero-js");
        assert!(recipe.is_some());
        assert_eq!(recipe.unwrap().title, "Zero-JS WASM Application");
    }

    #[test]
    fn test_search() {
        let cookbook = Cookbook::standard();
        let results = cookbook.search("random forest");
        assert!(!results.is_empty());
    }
}
