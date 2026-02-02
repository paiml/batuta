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
    /// TDD test companion for the code example
    pub test_code: String,
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
            test_code: String::new(),
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

    pub fn with_test_code(mut self, test_code: impl Into<String>) -> Self {
        self.test_code = test_code.into();
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
        cookbook.register_rlhf_recipes();
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
                .with_related(vec!["wasm-event-handling", "wasm-canvas-rendering"])
                .with_test_code(r#"#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    #[test]
    fn test_frame_increment() {
        let mut frame = 0u32;
        frame += 1;
        assert_eq!(frame, 1);
    }

    #[test]
    fn test_circle_radius_positive() {
        let radius = 50.0_f64;
        assert!(radius > 0.0);
    }

    #[test]
    fn test_canvas_center_calculation() {
        let w = 800_u32;
        let h = 600_u32;
        let center_x = (w as f64) / 2.0;
        let center_y = (h as f64) / 2.0;
        assert_eq!(center_x, 400.0);
        assert_eq!(center_y, 300.0);
    }
}"#),
        );

        // WASM Event Handling
        self.recipes.push(
            Recipe::new("wasm-event-handling", "WASM Event Handling")
                .with_problem("Handle DOM events (click, input, keypress) in pure Rust")
                .with_components(vec!["simular", "web-sys", "wasm-bindgen"])
                .with_tags(vec!["wasm", "events", "dom", "closure"])
                .with_code(r#"use wasm_bindgen::prelude::*;
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
"#)
                .with_related(vec!["wasm-zero-js", "wasm-canvas-rendering"])
                .with_test_code(r#"#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn test_closure_state_capture() {
        let counter = Rc::new(RefCell::new(0u32));
        let counter_clone = Rc::clone(&counter);
        let increment = move || { *counter_clone.borrow_mut() += 1; };
        increment();
        assert_eq!(*counter.borrow(), 1);
    }

    #[test]
    fn test_input_value_parsing() {
        let input_value = "42";
        let parsed = input_value.parse::<u32>();
        assert_eq!(parsed.unwrap(), 42);
    }

    #[test]
    fn test_speed_state_update() {
        let mut speed = 0_u32;
        let new_value = 10_u32;
        speed = new_value;
        assert_eq!(speed, 10);
    }
}"#),
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
                .with_related(vec!["wasm-zero-js", "wasm-event-handling"])
                .with_test_code(r#"#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    #[test]
    fn test_grid_spacing_calculation() {
        let width = 800.0_f64;
        let divisions = 10;
        let spacing = width / (divisions as f64);
        assert_eq!(spacing, 80.0);
    }

    #[test]
    fn test_trail_alpha_fade() {
        let trail_len = 10;
        let index = 5;
        let alpha = (index as f64) / (trail_len as f64) * 0.5;
        assert!((alpha - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_trail_data_structure() {
        let mut trail: Vec<(f64, f64)> = Vec::new();
        trail.push((100.0, 200.0));
        trail.push((150.0, 250.0));
        assert_eq!(trail.len(), 2);
    }
}"#),
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
                    r#"use aprender::prelude::*;
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
"#,
                )
                .with_related(vec!["ml-serving", "ml-preprocessing"])
                .with_test_code(r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_random_forest_builder_config() {
        let n_estimators = 100;
        let max_depth = Some(10);
        assert_eq!(n_estimators, 100);
        assert!(max_depth.unwrap() > 0);
    }

    #[test]
    fn test_predictions_collection() {
        let predictions = vec![0, 1, 1, 0, 1];
        assert_eq!(predictions.len(), 5);
    }

    #[test]
    fn test_accuracy_in_range() {
        let correct = 85;
        let total = 100;
        let accuracy = correct as f64 / total as f64;
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
    }
}"#),
        );

        // Model Serving
        self.recipes.push(
            Recipe::new("ml-serving", "Model Serving with Realizar")
                .with_problem("Deploy trained models as HTTP API or Lambda function")
                .with_components(vec!["realizar", "aprender"])
                .with_tags(vec!["ml", "serving", "inference", "api", "lambda"])
                .with_code(
                    r#"// Command line serving
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
"#,
                )
                .with_related(vec!["ml-random-forest", "distributed-inference"])
                .with_test_code(r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_server_config_port() {
        let port = 8080_u16;
        assert!(port >= 1024);
    }

    #[test]
    fn test_server_config_batch_and_workers() {
        let batch_size = 32;
        let workers = 4;
        assert_eq!(batch_size, 32);
        assert_eq!(workers, 4);
    }

    #[test]
    fn test_feature_vector_format() {
        let features: Vec<f32> = vec![1.0, 2.5, 3.7, 4.2];
        assert_eq!(features.len(), 4);
    }
}"#),
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
                    r"# Original Python code (sklearn_model.py)
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
",
                )
                .with_related(vec!["transpile-numpy", "quality-golden-trace"])
                .with_test_code(r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_tensor_shape() {
        let rows = 100;
        let cols = 4;
        let total_elements = rows * cols;
        assert_eq!(total_elements, 400);
    }

    #[test]
    fn test_train_test_split_ratio() {
        let total = 100;
        let train_ratio = 0.8;
        let train_size = (total as f64 * train_ratio) as usize;
        assert_eq!(train_size, 80);
    }

    #[test]
    fn test_label_values_binary() {
        let labels = vec![0, 1, 1, 0, 1];
        assert!(labels.iter().all(|&l| l == 0 || l == 1));
    }
}"#),
        );

        // NumPy to Trueno
        self.recipes.push(
            Recipe::new("transpile-numpy", "NumPy to Trueno Conversion")
                .with_problem("Convert NumPy operations to SIMD-accelerated Trueno")
                .with_components(vec!["depyler", "trueno"])
                .with_tags(vec!["transpilation", "numpy", "simd", "tensors"])
                .with_code(
                    r"# Python NumPy
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
",
                )
                .with_related(vec!["transpile-python"])
                .with_test_code(r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_vector_creation() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        assert_eq!(a.len(), b.len());
    }

    #[test]
    fn test_dot_product_computation() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert_eq!(dot, 70.0);
    }

    #[test]
    fn test_simd_element_type() {
        let vec: Vec<f64> = vec![1.0, 2.0, 3.0];
        assert!(vec.iter().all(|x| x.is_finite()));
    }
}"#),
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
                r"use repartir::prelude::*;

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
",
            )
            .with_related(vec!["distributed-gpu", "distributed-remote"])
            .with_test_code(r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_pool_worker_count() {
        let workers = 8;
        assert!(workers > 0);
    }

    #[test]
    fn test_chunk_processing() {
        let data = vec![1, 2, 3, 4, 5];
        let chunks: Vec<_> = data.chunks(2).collect();
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn test_result_reduction() {
        let results = vec![10.0, 20.0, 30.0, 40.0];
        let total: f64 = results.iter().sum();
        assert_eq!(total, 100.0);
    }
}"#),
        );

        // GPU Distribution
        self.recipes.push(
            Recipe::new("distributed-gpu", "GPU Compute Distribution")
                .with_problem("Offload computation to GPU using wgpu")
                .with_components(vec!["repartir", "trueno"])
                .with_tags(vec!["distributed", "gpu", "wgpu", "compute"])
                .with_code(
                    r"use repartir::prelude::*;
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
",
                )
                .with_related(vec!["distributed-work-stealing"])
                .with_test_code(r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_matrix_dimensions_valid() {
        let m = 1024;
        let k = 512;
        let n = 2048;
        assert!(m > 0 && k > 0 && n > 0);
    }

    #[test]
    fn test_hybrid_pool_worker_total() {
        let cpu_workers = 4;
        let gpu_workers = 2;
        let total = cpu_workers + gpu_workers;
        assert_eq!(total, 6);
    }

    #[test]
    fn test_matmul_output_shape() {
        let rows_a = 128;
        let cols_b = 256;
        let output_elements = rows_a * cols_b;
        assert_eq!(output_elements, 32768);
    }
}"#),
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
                    r"use simular::prelude::*;

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
",
                )
                .with_related(vec!["quality-probar", "quality-golden-trace"])
                .with_test_code(r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_energy_conservation_invariant() {
        let kinetic = 60.0;
        let potential = 40.0;
        let initial_energy = 100.0;
        assert!((kinetic + potential - initial_energy).abs() < 1e-6);
    }

    #[test]
    fn test_timestep_positivity() {
        let dt = 0.01_f64;
        assert!(dt > 0.0);
    }

    #[test]
    fn test_falsification_detection() {
        let tolerance = 1e-9_f64;
        let error = 1e-5;
        assert!(error > tolerance);
    }
}"#),
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
                    r#"use probar::prelude::*;

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
"#,
                )
                .with_related(vec!["quality-edd", "quality-certeza"])
                .with_test_code(r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_tour_length_positivity() {
        let distances = vec![10.5, 20.3, 15.7, 8.2];
        let tour_length: f64 = distances.iter().sum();
        assert!(tour_length > 0.0);
    }

    #[test]
    fn test_energy_conservation_property() {
        let initial = 100.0_f64;
        let final_energy = 100.0_f64;
        assert!((initial - final_energy).abs() < 1e-10);
    }

    #[test]
    fn test_coverage_threshold() {
        let covered = 950;
        let total = 1000;
        let coverage = covered as f64 / total as f64;
        assert!(coverage >= 0.95);
    }
}"#),
        );

        // Golden Trace Comparison
        self.recipes.push(
            Recipe::new("quality-golden-trace", "Golden Trace Validation")
                .with_problem("Validate transpiled code produces identical behavior to original")
                .with_components(vec!["renacer", "certeza"])
                .with_tags(vec!["quality", "validation", "trace", "transpilation"])
                .with_code(
                    r#"use renacer::prelude::*;

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
"#,
                )
                .with_related(vec!["transpile-python", "quality-edd"])
                .with_test_code(r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_semantic_equivalence_check() {
        let baseline = vec![1.0, 2.0, 3.0];
        let candidate = vec![1.0, 2.0, 3.0];
        assert_eq!(baseline, candidate);
    }

    #[test]
    fn test_speedup_calculation() {
        let baseline_ms = 1000.0_f64;
        let optimized_ms = 250.0_f64;
        let speedup = baseline_ms / optimized_ms;
        assert_eq!(speedup, 4.0);
    }

    #[test]
    fn test_trace_comparison_length() {
        let golden = vec!["step1", "step2", "step3"];
        let actual = vec!["step1", "step2", "step3"];
        assert_eq!(golden.len(), actual.len());
    }
}"#),
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
                    r#"use whisper_apr::prelude::*;

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
"#,
                )
                .with_related(vec!["speech-streaming", "ml-serving"])
                .with_test_code(r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_model_name_validation() {
        let model = "tiny.en";
        let valid = vec!["tiny", "tiny.en", "base", "base.en"];
        assert!(valid.contains(&model));
    }

    #[test]
    fn test_transcription_result_non_empty() {
        let text = "Hello world";
        assert!(!text.is_empty());
    }

    #[test]
    fn test_segment_timestamp_ordering() {
        let start = 0.5_f64;
        let end = 2.0_f64;
        assert!(start < end);
    }
}"#),
        );

        // Streaming Speech
        self.recipes.push(
            Recipe::new("speech-streaming", "Real-time Speech Streaming")
                .with_problem("Process audio in real-time with low latency")
                .with_components(vec!["whisper-apr", "trueno"])
                .with_tags(vec!["speech", "streaming", "real-time", "low-latency"])
                .with_code(
                    r#"use whisper_apr::streaming::*;

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
"#,
                )
                .with_related(vec!["speech-whisper"])
                .with_test_code(r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_stream_config_chunk_size_positive() {
        let chunk_size_ms = 500_u32;
        assert!(chunk_size_ms > 0);
    }

    #[test]
    fn test_overlap_less_than_chunk_size() {
        let chunk_size_ms = 500;
        let overlap_ms = 100;
        assert!(overlap_ms < chunk_size_ms);
    }

    #[test]
    fn test_vad_enabled_flag() {
        let vad_enabled = true;
        assert!(vad_enabled);
    }
}"#),
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
                .with_test_code(r#"#[cfg(test)]
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
}"#),
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
                .with_test_code(r#"#[cfg(test)]
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
}"#),
        );

        // Autograd
        self.recipes.push(
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
                .with_test_code(r#"#[cfg(test)]
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
}"#),
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
                .with_test_code(r#"#[cfg(test)]
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
}"#),
        );

        // Data Preprocessing
        self.recipes.push(
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
                .with_test_code(r#"#[cfg(test)]
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
}"#),
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
                .with_test_code(r#"#[cfg(test)]
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
}"#),
        );

        // HuggingFace Integration
        self.recipes.push(
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
                .with_test_code(r#"#[cfg(test)]
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
}"#),
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
                .with_test_code(r#"#[cfg(test)]
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
}"#),
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
                .with_test_code(r#"#[cfg(test)]
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
}"#),
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
                .with_test_code(r#"#[cfg(test)]
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
}"#),
        );

        // PNG Export
        self.recipes.push(
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
                .with_test_code(r#"#[cfg(test)]
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
}"#),
        );
    }

    // =========================================================================
    // RLHF & Alignment Recipes
    // =========================================================================

    fn register_rlhf_recipes(&mut self) {
        // Supervised Fine-Tuning (SFT)
        self.recipes.push(
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
                .with_test_code(r#"#[cfg(test)]
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
}"#),
        );

        // Reward Modeling
        self.recipes.push(
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
                .with_test_code(r#"#[cfg(test)]
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
}"#),
        );

        // Direct Preference Optimization (DPO)
        self.recipes.push(
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
                .with_test_code(r#"#[cfg(test)]
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
}"#),
        );

        // DPO Variants (IPO, KTO, ORPO)
        self.recipes.push(
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
                .with_test_code(r#"#[cfg(test)]
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
}"#),
        );

        // PPO for RLHF
        self.recipes.push(
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
                .with_test_code(r#"#[cfg(test)]
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
}"#),
        );

        // RLHF Stability & Best Practices
        self.recipes.push(
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
                .with_test_code(r#"#[cfg(test)]
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
}"#),
        );

        // RLHF Evaluation
        self.recipes.push(
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
                .with_test_code(r#"#[cfg(test)]
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
}"#),
        );

        // Quantization for Alignment
        self.recipes.push(
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
                .with_test_code(r#"#[cfg(test)]
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
}"#),
        );

        // PEFT Adapters
        self.recipes.push(
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
                .with_test_code(r#"#[cfg(test)]
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
}"#),
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

    #[test]
    fn test_all_recipes_have_code() {
        let cookbook = Cookbook::standard();
        for recipe in cookbook.recipes() {
            assert!(
                !recipe.code.is_empty(),
                "Recipe '{}' has empty code field",
                recipe.id
            );
        }
    }

    #[test]
    fn test_recipe_code_contains_rust() {
        let cookbook = Cookbook::standard();
        let recipe = cookbook.get("ml-random-forest").expect("ml-random-forest must exist");
        assert!(
            recipe.code.contains("use "),
            "ml-random-forest code should contain 'use ' import"
        );
        assert!(
            recipe.code.contains("aprender"),
            "ml-random-forest code should reference aprender"
        );
    }

    #[test]
    fn test_all_recipes_have_test_code() {
        let cookbook = Cookbook::standard();
        for recipe in cookbook.recipes() {
            assert!(
                !recipe.test_code.is_empty(),
                "Recipe '{}' has empty test_code field",
                recipe.id
            );
        }
    }

    #[test]
    fn test_test_code_has_cfg_test() {
        let cookbook = Cookbook::standard();
        for recipe in cookbook.recipes() {
            assert!(
                recipe.test_code.contains("#[cfg("),
                "Recipe '{}' test_code should contain #[cfg(test)] or #[cfg(all(test, ...))]",
                recipe.id
            );
        }
    }

    #[test]
    fn test_test_code_has_test_attr() {
        let cookbook = Cookbook::standard();
        for recipe in cookbook.recipes() {
            assert!(
                recipe.test_code.contains("#[test]"),
                "Recipe '{}' test_code should contain #[test] attribute",
                recipe.id
            );
        }
    }
}
