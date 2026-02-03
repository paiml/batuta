//! SVG Generation Demo
//!
//! Demonstrates the Material Design 3 compliant SVG generation system.
//!
//! Run with: cargo run --example svg_generation_demo

use batuta::oracle::svg::{
    documentation_diagram, sovereign_stack_diagram, ShapeHeavyRenderer, TextHeavyRenderer,
};
use batuta::oracle::svg::shapes::Point;

fn main() {
    println!("=== SVG Generation Demo ===\n");

    // 1. Generate Sovereign Stack architecture diagram
    println!("1. Sovereign Stack Architecture Diagram");
    println!("   Using ShapeHeavyRenderer for component visualization");
    let stack_svg = sovereign_stack_diagram();
    println!("   Generated SVG: {} bytes", stack_svg.len());
    println!("   Contains layers: Compute, ML, Orchestration\n");

    // 2. Custom shape-heavy diagram
    println!("2. Custom Architecture Diagram");
    let custom_svg = ShapeHeavyRenderer::new()
        .title("Data Pipeline Architecture")
        .layer("ingestion", 50.0, 100.0, 800.0, 150.0, "Data Ingestion")
        .horizontal_stack(
            &[
                ("kafka", "Kafka"),
                ("spark", "Spark"),
                ("trueno", "Trueno"),
            ],
            Point::new(100.0, 130.0),
        )
        .layer("processing", 50.0, 300.0, 800.0, 150.0, "Processing")
        .horizontal_stack(
            &[
                ("aprender", "Aprender"),
                ("realizar", "Realizar"),
            ],
            Point::new(100.0, 330.0),
        )
        .build();
    println!("   Generated SVG: {} bytes\n", custom_svg.len());

    // 3. Text-heavy documentation diagram
    println!("3. Documentation Diagram");
    let doc_svg = documentation_diagram(
        "API Reference",
        &[
            ("Authentication", "All endpoints require Bearer token authentication."),
            ("Rate Limiting", "100 requests per minute per API key."),
            ("Versioning", "API version specified via Accept header."),
        ],
    );
    println!("   Generated SVG: {} bytes\n", doc_svg.len());

    // 4. Dark mode diagram
    println!("4. Dark Mode Diagram");
    let dark_svg = ShapeHeavyRenderer::new()
        .dark_mode()
        .title("Dark Mode Components")
        .component("comp1", 100.0, 200.0, "Component A", "trueno")
        .component("comp2", 300.0, 200.0, "Component B", "aprender")
        .connect(Point::new(260.0, 240.0), Point::new(300.0, 240.0))
        .build();
    println!("   Generated SVG: {} bytes\n", dark_svg.len());

    // 5. Text-heavy with code-like content
    println!("5. Code Documentation Diagram");
    let code_doc = TextHeavyRenderer::new()
        .title("Function Reference")
        .heading("train_model()")
        .paragraph("Trains a machine learning model on the provided dataset.")
        .heading("Parameters")
        .paragraph("• data: &Dataset - Training data")
        .paragraph("• epochs: usize - Number of training epochs")
        .paragraph("• lr: f32 - Learning rate")
        .heading("Returns")
        .paragraph("Result<Model, TrainError>")
        .build();
    println!("   Generated SVG: {} bytes\n", code_doc.len());

    println!("=== Demo Complete ===");
    println!("All diagrams use Material Design 3 colors and 8px grid alignment.");
}
