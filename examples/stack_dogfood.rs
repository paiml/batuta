//! Stack Dogfooding - Real PAIML Stack Health Analysis
//!
//! This example uses REAL pmat demo-score data from the actual PAIML stack
//! to demonstrate the diagnostics system eating its own dog food.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example stack_dogfood --features native
//! ```

#[cfg(feature = "native")]
use batuta::{
    render_dashboard, AndonStatus, Anomaly, AnomalyCategory, ComponentMetrics, ComponentNode,
    ErrorForecaster, HealthStatus, IsolationForest, QualityGrade, QualityStackLayer as StackLayer,
    StackDiagnostics,
};

#[cfg(feature = "native")]
fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║        PAIML SOVEREIGN AI STACK - REAL HEALTH ANALYSIS            ║");
    println!("║               Dogfooding Stack Diagnostics v0.1.0                 ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // Real PMAT Demo Scores from PAIML Stack (collected 2024-12-07)
    // =========================================================================

    let stack_data = vec![
        // (name, version, layer, demo_score, grade)
        (
            "trueno",
            "0.7.4",
            StackLayer::Compute,
            89.9,
            QualityGrade::AMinus,
        ),
        (
            "realizar",
            "0.9.0",
            StackLayer::Training,
            85.4,
            QualityGrade::AMinus,
        ),
        (
            "pforge",
            "0.1.4",
            StackLayer::Quality,
            85.2,
            QualityGrade::AMinus,
        ),
        (
            "batuta",
            "0.1.4",
            StackLayer::Orchestration,
            84.9,
            QualityGrade::BPlus,
        ),
        (
            "aprender",
            "0.11.0",
            StackLayer::Ml,
            83.1,
            QualityGrade::BPlus,
        ),
        (
            "trueno-db",
            "0.5.0",
            StackLayer::DataMlops,
            80.4,
            QualityGrade::BPlus,
        ),
        (
            "bashrs",
            "0.2.0",
            StackLayer::Transpilers,
            79.0,
            QualityGrade::B,
        ),
        (
            "depyler",
            "0.4.0",
            StackLayer::Transpilers,
            74.6,
            QualityGrade::B,
        ),
        (
            "entrenar",
            "1.8.0",
            StackLayer::Training,
            73.5,
            QualityGrade::B,
        ),
    ];

    let mut diag = StackDiagnostics::new();

    // Add all real components
    for (name, version, layer, score, grade) in &stack_data {
        let mut node = ComponentNode::new(*name, *version, *layer);
        node.metrics = ComponentMetrics::with_demo_score(*score);
        node.metrics.grade = *grade;
        node.update_health();
        diag.add_component(node);
    }

    // =========================================================================
    // Phase 1: Andon Status Overview
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 1: ANDON STATUS OVERVIEW                                      │");
    println!("└─────────────────────────────────────────────────────────────────────┘\n");

    let summary = diag.health_summary();
    let andon_icon = match summary.andon_status {
        AndonStatus::Green => "🟢",
        AndonStatus::Yellow => "🟡",
        AndonStatus::Red => "🔴",
        AndonStatus::Unknown => "⚪",
    };

    println!("  ANDON STATUS: {} {:?}", andon_icon, summary.andon_status);
    println!();
    println!("  Components by Health:");
    println!("    🟢 Green (A-/A/A+):  {}", summary.green_count);
    println!("    🟡 Yellow (B+/A-):   {}", summary.yellow_count);
    println!("    🔴 Red (≤B):         {}", summary.red_count);
    println!("    ⚪ Unknown:          {}", summary.unknown_count);
    println!();

    // =========================================================================
    // Phase 2: Component Quality Matrix
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 2: COMPONENT QUALITY MATRIX                                   │");
    println!("└─────────────────────────────────────────────────────────────────────┘\n");

    println!("  ┌──────────────┬─────────┬───────────┬────────┬────────────────────┐");
    println!("  │ Component    │ Version │ Demo Score│ Grade  │ Quality Gate       │");
    println!("  ├──────────────┼─────────┼───────────┼────────┼────────────────────┤");

    for (name, version, _, score, grade) in &stack_data {
        let status = HealthStatus::from_grade(*grade);
        let gate_status = if *score >= 85.0 {
            "✅ PASS"
        } else {
            "❌ FAIL"
        };
        println!(
            "  │ {:12} │ {:7} │ {:>9.1} │ {} {:4} │ {:18} │",
            name,
            version,
            score,
            status.icon(),
            grade.symbol(),
            gate_status
        );
    }

    println!("  └──────────────┴─────────┴───────────┴────────┴────────────────────┘");
    println!();

    // Calculate statistics
    let scores: Vec<f64> = stack_data.iter().map(|(_, _, _, s, _)| *s).collect();
    let avg_score = scores.iter().sum::<f64>() / scores.len() as f64;
    let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let pass_count = scores.iter().filter(|&&s| s >= 85.0).count();

    println!("  STATISTICS:");
    println!("    Average Score:    {:.1}/100", avg_score);
    println!("    Min Score:        {:.1}/100 (entrenar)", min_score);
    println!("    Max Score:        {:.1}/100 (trueno)", max_score);
    println!(
        "    Passing (≥85.0):  {}/{} ({:.0}%)",
        pass_count,
        scores.len(),
        (pass_count as f64 / scores.len() as f64) * 100.0
    );
    println!();

    // =========================================================================
    // Phase 3: Isolation Forest Anomaly Detection
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 3: ISOLATION FOREST ANOMALY DETECTION                         │");
    println!("└─────────────────────────────────────────────────────────────────────┘\n");

    // Prepare feature vectors: [demo_score]
    let data: Vec<Vec<f64>> = stack_data
        .iter()
        .map(|(_, _, _, score, _)| vec![*score])
        .collect();

    let mut forest = IsolationForest::new(100, 256, 42);
    forest.fit(&data);
    let anomaly_scores = forest.score(&data);

    println!("  Isolation Forest Results (threshold: 0.5):");
    println!();
    println!("  ┌──────────────┬──────────────┬────────────────────────────────────┐");
    println!("  │ Component    │ Anomaly Score│ Status                             │");
    println!("  ├──────────────┼──────────────┼────────────────────────────────────┤");

    let mut anomalies_detected = Vec::new();

    for (i, ((name, _, _, demo_score, _), &score)) in
        stack_data.iter().zip(anomaly_scores.iter()).enumerate()
    {
        let status = if score > 0.6 {
            anomalies_detected.push((i, *name, score, *demo_score));
            "🔴 ANOMALY - Quality regression"
        } else if score > 0.5 {
            "🟡 BORDERLINE - Monitor closely"
        } else {
            "🟢 NORMAL - Within expected range"
        };
        println!("  │ {:12} │ {:>12.4} │ {:34} │", name, score, status);
    }

    println!("  └──────────────┴──────────────┴────────────────────────────────────┘");
    println!();

    // Add anomalies to diagnostics
    for (_, name, score, demo_score) in &anomalies_detected {
        diag.add_anomaly(
            Anomaly::new(
                *name,
                *score,
                AnomalyCategory::QualityRegression,
                format!("Demo score {:.1} below A- threshold (85.0)", demo_score),
            )
            .with_evidence(format!("Isolation Forest anomaly score: {:.4}", score))
            .with_recommendation("Run `pmat demo-score` and address top issues"),
        );
    }

    println!("  ANOMALIES DETECTED: {}", anomalies_detected.len());
    for (_, name, score, demo_score) in &anomalies_detected {
        println!(
            "    - {} (score: {:.1}, anomaly: {:.4})",
            name, demo_score, score
        );
    }
    println!();

    // =========================================================================
    // Phase 4: Graph Analytics
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 4: GRAPH ANALYTICS                                            │");
    println!("└─────────────────────────────────────────────────────────────────────┘\n");

    let metrics = diag.compute_metrics()?;

    println!("  Graph-Level Metrics:");
    println!("    Total Nodes:     {}", metrics.total_nodes);
    println!("    Total Edges:     {}", metrics.total_edges);
    println!("    Graph Density:   {:.4}", metrics.density);
    println!("    Average Degree:  {:.2}", metrics.avg_degree);
    println!();

    println!("  PageRank Importance:");
    let top = metrics.top_by_pagerank(9);
    for (name, score) in &top {
        let bar_len = (*score * 50.0).round() as usize;
        let bar: String = "█".repeat(bar_len.min(20));
        println!("    {:12} │ {:5.3} │ {}", name, score, bar);
    }
    println!();

    // =========================================================================
    // Phase 5: Error Forecasting (Simulated History)
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 5: ERROR FORECASTING                                          │");
    println!("└─────────────────────────────────────────────────────────────────────┘\n");

    // Simulate historical quality gate failures
    let historical_failures = [2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0]; // increasing failures

    let mut forecaster = ErrorForecaster::new(0.3);
    for &obs in &historical_failures {
        forecaster.observe(obs);
    }

    println!("  Historical Quality Gate Failures:");
    for (i, &obs) in historical_failures.iter().enumerate() {
        println!("    Week {}: {} failures", i + 1, obs as i32);
    }

    println!();
    println!("  Forecast (next 4 weeks):");
    let forecast = forecaster.forecast(4);
    for (i, &f) in forecast.iter().enumerate() {
        let trend = if f > historical_failures[7] {
            "↗"
        } else if f < historical_failures[7] {
            "↘"
        } else {
            "→"
        };
        println!("    Week {}: {:.1} failures predicted {}", i + 9, f, trend);
    }

    let error_metrics = forecaster.error_metrics();
    println!();
    println!("  Forecast Accuracy:");
    println!("    MAE:  {:.2}", error_metrics.mae);
    println!("    RMSE: {:.2}", error_metrics.rmse);
    println!();

    // =========================================================================
    // Phase 6: Dashboard Rendering
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 6: ASCII DASHBOARD (MIERUKA)                                  │");
    println!("└─────────────────────────────────────────────────────────────────────┘\n");

    let dashboard = render_dashboard(&diag);
    println!("{}", dashboard);

    // =========================================================================
    // Phase 7: Insights & Recommendations
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE 7: INSIGHTS & RECOMMENDATIONS                                 │");
    println!("└─────────────────────────────────────────────────────────────────────┘\n");

    println!("  🎯 KEY INSIGHTS:");
    println!();
    println!(
        "  1. OVERALL HEALTH: {} ({:.1}% passing quality gate)",
        if pass_count >= 6 {
            "🟢 GOOD"
        } else if pass_count >= 3 {
            "🟡 ATTENTION NEEDED"
        } else {
            "🔴 CRITICAL"
        },
        (pass_count as f64 / scores.len() as f64) * 100.0
    );
    println!();
    println!("  2. TOP PERFORMERS (A- or better):");
    for (name, _, _, score, grade) in &stack_data {
        if *score >= 85.0 {
            println!("     ✅ {} ({:.1} - {})", name, score, grade.symbol());
        }
    }
    println!();
    println!("  3. NEEDS ATTENTION (below A-):");
    for (name, _, _, score, grade) in &stack_data {
        if *score < 85.0 {
            let gap = 85.0 - score;
            println!(
                "     ❌ {} ({:.1} - {}) - needs +{:.1} points",
                name,
                score,
                grade.symbol(),
                gap
            );
        }
    }
    println!();
    println!("  4. PRIORITY ORDER (by gap to A-):");
    let mut below_threshold: Vec<_> = stack_data
        .iter()
        .filter(|(_, _, _, score, _)| *score < 85.0)
        .collect();
    below_threshold.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    for (i, (name, _, _, score, _)) in below_threshold.iter().enumerate() {
        let gap = 85.0 - score;
        println!("     {}. {} (+{:.1} to pass)", i + 1, name, gap);
    }
    println!();
    println!("  5. RECOMMENDED ACTIONS:");
    println!("     a. Focus on batuta first (only +0.1 to pass)");
    println!("     b. Improve aprender demo runtime quality (currently 75%)");
    println!("     c. Address trueno-db quality standards");
    println!("     d. Review entrenar for technical debt (lowest score)");
    println!();

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║                  DOGFOODING ANALYSIS COMPLETE                     ║");
    println!("║                                                                   ║");
    println!(
        "║  Stack Health: {} Green, {} Yellow, {} Red                        ║",
        summary.green_count, summary.yellow_count, summary.red_count
    );
    println!(
        "║  Quality Gate: {}/9 passing ({:.0}%)                               ║",
        pass_count,
        (pass_count as f64 / 9.0) * 100.0
    );
    println!(
        "║  Anomalies:    {} detected                                        ║",
        anomalies_detected.len()
    );
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    Ok(())
}

#[cfg(not(feature = "native"))]
fn main() {
    println!("This example requires the 'native' feature.");
    println!("Run with: cargo run --example stack_dogfood --features native");
}
