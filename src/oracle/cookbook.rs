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
