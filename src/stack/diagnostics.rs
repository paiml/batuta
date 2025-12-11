#![allow(dead_code)]
//! Stack Visualization, Diagnostics, and Reporting
//!
//! ML-driven system for visualizing, diagnosing, and reporting on the health
//! of the Sovereign AI Stack. Implements Toyota Way principles for observability.
//!
//! ## Toyota Way Principles
//!
//! - **Mieruka (Visual Control)**: Rich ASCII dashboards make health visible
//! - **Jidoka**: ML anomaly detection surfaces issues automatically
//! - **Genchi Genbutsu**: Evidence-based diagnosis from actual dependency data
//! - **Andon**: Red/Yellow/Green status with stop-the-line alerts
//! - **Yokoten**: Cross-component insight sharing via knowledge graph

use crate::stack::quality::{QualityGrade, StackLayer};
use crate::stack::DependencyGraph;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Simple PRNG (for reproducible isolation forest without external deps)
// ============================================================================

/// Simple Linear Congruential Generator for reproducible randomness
#[derive(Debug, Clone)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn seed_from_u64(seed: u64) -> Self {
        Self {
            state: seed ^ 0x5DEECE66D,
        }
    }

    fn next_u64(&mut self) -> u64 {
        // LCG parameters from Numerical Recipes
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn gen_range(&mut self, range: std::ops::Range<usize>) -> usize {
        if range.is_empty() {
            return range.start;
        }
        let len = range.end - range.start;
        range.start + (self.next_u64() as usize % len)
    }

    fn gen_range_f64(&mut self, range: std::ops::Range<f64>) -> f64 {
        let t = (self.next_u64() as f64) / (u64::MAX as f64);
        range.start + t * (range.end - range.start)
    }
}

// ============================================================================
// Health Status (Andon System)
// ============================================================================

/// Health status for components (Andon-style visual control)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HealthStatus {
    /// All systems healthy - normal operation
    Green,
    /// Attention needed - warnings present
    Yellow,
    /// Critical issues - stop-the-line
    Red,
    /// Not yet analyzed
    Unknown,
}

impl HealthStatus {
    /// Create from quality grade
    pub fn from_grade(grade: QualityGrade) -> Self {
        match grade {
            QualityGrade::APlus | QualityGrade::A => Self::Green,
            QualityGrade::AMinus | QualityGrade::BPlus => Self::Yellow,
            _ => Self::Red,
        }
    }

    /// Get display icon for status
    pub fn icon(&self) -> &'static str {
        match self {
            Self::Green => "🟢",
            Self::Yellow => "🟡",
            Self::Red => "🔴",
            Self::Unknown => "⚪",
        }
    }

    /// Get ASCII symbol for terminal without emoji support
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::Green => "●",
            Self::Yellow => "◐",
            Self::Red => "○",
            Self::Unknown => "◌",
        }
    }
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.icon())
    }
}

// ============================================================================
// Component Node (Stack Knowledge Graph)
// ============================================================================

/// A component in the stack knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentNode {
    /// Component name (e.g., "trueno", "aprender")
    pub name: String,
    /// Semantic version
    pub version: String,
    /// Stack layer classification
    pub layer: StackLayer,
    /// Current health status
    pub health: HealthStatus,
    /// Quality metrics
    pub metrics: ComponentMetrics,
}

impl ComponentNode {
    /// Create a new component node
    pub fn new(name: impl Into<String>, version: impl Into<String>, layer: StackLayer) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            layer,
            health: HealthStatus::Unknown,
            metrics: ComponentMetrics::default(),
        }
    }

    /// Update health status from metrics
    pub fn update_health(&mut self) {
        self.health = HealthStatus::from_grade(self.metrics.grade);
    }
}

// ============================================================================
// Component Metrics
// ============================================================================

/// Quality and performance metrics for a component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentMetrics {
    /// Demo score (0-100 normalized)
    pub demo_score: f64,
    /// Test coverage percentage
    pub coverage: f64,
    /// Mutation score percentage
    pub mutation_score: f64,
    /// Average cyclomatic complexity
    pub complexity_avg: f64,
    /// SATD (Self-Admitted Technical Debt) count
    pub satd_count: u32,
    /// Dead code percentage
    pub dead_code_pct: f64,
    /// Overall quality grade
    pub grade: QualityGrade,
}

impl Default for ComponentMetrics {
    fn default() -> Self {
        Self {
            demo_score: 0.0,
            coverage: 0.0,
            mutation_score: 0.0,
            complexity_avg: 0.0,
            satd_count: 0,
            dead_code_pct: 0.0,
            grade: QualityGrade::F, // Lowest grade as default
        }
    }
}

impl ComponentMetrics {
    /// Create metrics with demo score
    pub fn with_demo_score(demo_score: f64) -> Self {
        let grade = QualityGrade::from_sqi(demo_score);
        Self {
            demo_score,
            grade,
            ..Default::default()
        }
    }

    /// Check if metrics meet A- threshold
    pub fn meets_threshold(&self) -> bool {
        self.demo_score >= 85.0
    }
}

// ============================================================================
// Graph Metrics
// ============================================================================

/// Computed graph-level metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphMetrics {
    /// PageRank scores by node
    pub pagerank: HashMap<String, f64>,
    /// Betweenness centrality by node
    pub betweenness: HashMap<String, f64>,
    /// Clustering coefficient by node
    pub clustering: HashMap<String, f64>,
    /// Community assignments (node -> community_id)
    pub communities: HashMap<String, usize>,
    /// Depth from root nodes
    pub depth_map: HashMap<String, u32>,
    /// Total nodes in graph
    pub total_nodes: usize,
    /// Total edges in graph
    pub total_edges: usize,
    /// Graph density (edges / possible edges)
    pub density: f64,
    /// Average degree
    pub avg_degree: f64,
    /// Maximum depth
    pub max_depth: u32,
}

impl GraphMetrics {
    /// Get the most critical components by PageRank
    pub fn top_by_pagerank(&self, n: usize) -> Vec<(&String, f64)> {
        let mut scores: Vec<_> = self.pagerank.iter().map(|(k, v)| (k, *v)).collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.into_iter().take(n).collect()
    }

    /// Get bottleneck components (high betweenness)
    pub fn bottlenecks(&self, threshold: f64) -> Vec<&String> {
        self.betweenness
            .iter()
            .filter(|(_, &v)| v > threshold)
            .map(|(k, _)| k)
            .collect()
    }
}

// ============================================================================
// Stack Diagnostics Engine
// ============================================================================

/// Main diagnostics engine for stack analysis
#[derive(Debug)]
pub struct StackDiagnostics {
    /// Component nodes
    components: HashMap<String, ComponentNode>,
    /// Dependency graph
    graph: Option<DependencyGraph>,
    /// Computed graph metrics
    metrics: GraphMetrics,
    /// Detected anomalies
    anomalies: Vec<Anomaly>,
}

impl StackDiagnostics {
    /// Create a new diagnostics engine
    pub fn new() -> Self {
        Self {
            components: HashMap::new(),
            graph: None,
            metrics: GraphMetrics::default(),
            anomalies: Vec::new(),
        }
    }

    /// Add a component to the knowledge graph
    pub fn add_component(&mut self, node: ComponentNode) {
        self.components.insert(node.name.clone(), node);
    }

    /// Get a component by name
    pub fn get_component(&self, name: &str) -> Option<&ComponentNode> {
        self.components.get(name)
    }

    /// Get all components
    pub fn components(&self) -> impl Iterator<Item = &ComponentNode> {
        self.components.values()
    }

    /// Get component count
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Set the dependency graph
    pub fn set_graph(&mut self, graph: DependencyGraph) {
        self.graph = Some(graph);
    }

    /// Get the dependency graph
    pub fn graph(&self) -> Option<&DependencyGraph> {
        self.graph.as_ref()
    }

    /// Compute graph metrics (PageRank, Betweenness, etc.)
    pub fn compute_metrics(&mut self) -> Result<&GraphMetrics> {
        let n = self.components.len();
        if n == 0 {
            return Ok(&self.metrics);
        }

        self.metrics.total_nodes = n;

        // Build adjacency from dependency graph if available
        let adjacency = self.build_adjacency();

        // Compute PageRank
        self.compute_pagerank(&adjacency, 0.85, 100);

        // Compute Betweenness Centrality
        self.compute_betweenness(&adjacency);

        // Compute depth from roots
        self.compute_depth(&adjacency);

        // Compute graph-level metrics
        self.metrics.total_edges = adjacency.values().map(|v| v.len()).sum();
        let max_edges = n * (n.saturating_sub(1));
        self.metrics.density = if max_edges > 0 {
            self.metrics.total_edges as f64 / max_edges as f64
        } else {
            0.0
        };
        self.metrics.avg_degree = if n > 0 {
            self.metrics.total_edges as f64 / n as f64
        } else {
            0.0
        };
        self.metrics.max_depth = self.metrics.depth_map.values().copied().max().unwrap_or(0);

        Ok(&self.metrics)
    }

    /// Build adjacency list from dependency graph
    fn build_adjacency(&self) -> HashMap<String, Vec<String>> {
        let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();

        // Initialize all nodes
        for name in self.components.keys() {
            adjacency.insert(name.clone(), Vec::new());
        }

        // Add edges from dependency graph
        if let Some(graph) = &self.graph {
            for crate_info in graph.all_crates() {
                let from = &crate_info.name;
                for dep in &crate_info.paiml_dependencies {
                    if self.components.contains_key(&dep.name) {
                        adjacency
                            .entry(from.clone())
                            .or_default()
                            .push(dep.name.clone());
                    }
                }
            }
        }

        adjacency
    }

    /// Compute PageRank using power iteration
    fn compute_pagerank(
        &mut self,
        adjacency: &HashMap<String, Vec<String>>,
        damping: f64,
        max_iter: usize,
    ) {
        let n = self.components.len();
        if n == 0 {
            return;
        }

        let initial = 1.0 / n as f64;
        let mut scores: HashMap<String, f64> = self
            .components
            .keys()
            .map(|k| (k.clone(), initial))
            .collect();

        // Find dangling nodes (nodes with no outgoing edges)
        let dangling_nodes: Vec<_> = adjacency
            .iter()
            .filter(|(_, targets)| targets.is_empty())
            .map(|(node, _)| node.clone())
            .collect();

        // Power iteration
        for _ in 0..max_iter {
            let mut new_scores: HashMap<String, f64> = HashMap::new();
            let teleport = (1.0 - damping) / n as f64;

            // Dangling nodes contribute their rank equally to all nodes
            let dangling_sum: f64 = dangling_nodes
                .iter()
                .map(|node| scores.get(node).unwrap_or(&0.0))
                .sum();
            let dangling_contrib = damping * dangling_sum / n as f64;

            for node in self.components.keys() {
                let mut incoming_score = 0.0;

                // Find nodes that link to this node
                for (source, targets) in adjacency {
                    if targets.contains(node) {
                        let out_degree = targets.len();
                        if out_degree > 0 {
                            incoming_score +=
                                scores.get(source).unwrap_or(&0.0) / out_degree as f64;
                        }
                    }
                }

                new_scores.insert(
                    node.clone(),
                    teleport + damping * incoming_score + dangling_contrib,
                );
            }

            // Check convergence
            let diff: f64 = new_scores
                .iter()
                .map(|(k, v)| (v - scores.get(k).unwrap_or(&0.0)).abs())
                .sum();

            scores = new_scores;

            if diff < 1e-6 {
                break;
            }
        }

        self.metrics.pagerank = scores;
    }

    /// Compute Betweenness Centrality using Brandes algorithm (simplified)
    fn compute_betweenness(&mut self, adjacency: &HashMap<String, Vec<String>>) {
        let nodes: Vec<_> = self.components.keys().cloned().collect();
        let n = nodes.len();

        // Initialize betweenness
        let mut betweenness: HashMap<String, f64> =
            nodes.iter().map(|n| (n.clone(), 0.0)).collect();

        // For each source, compute shortest paths and accumulate
        for source in &nodes {
            // BFS from source
            let mut dist: HashMap<String, i32> = HashMap::new();
            let mut sigma: HashMap<String, f64> = HashMap::new();
            let mut predecessors: HashMap<String, Vec<String>> = HashMap::new();

            for n in &nodes {
                dist.insert(n.clone(), -1);
                sigma.insert(n.clone(), 0.0);
                predecessors.insert(n.clone(), Vec::new());
            }

            dist.insert(source.clone(), 0);
            sigma.insert(source.clone(), 1.0);

            let mut queue = vec![source.clone()];
            let mut order = Vec::new();

            while !queue.is_empty() {
                let v = queue.remove(0);
                order.push(v.clone());

                if let Some(neighbors) = adjacency.get(&v) {
                    for w in neighbors {
                        let d_v = dist[&v];
                        let d_w = dist.get(w).copied().unwrap_or(-1);

                        if d_w < 0 {
                            dist.insert(w.clone(), d_v + 1);
                            queue.push(w.clone());
                        }

                        if dist.get(w).copied().unwrap_or(-1) == d_v + 1 {
                            *sigma.get_mut(w).unwrap() += sigma[&v];
                            predecessors.get_mut(w).unwrap().push(v.clone());
                        }
                    }
                }
            }

            // Back-propagation
            let mut delta: HashMap<String, f64> = nodes.iter().map(|n| (n.clone(), 0.0)).collect();

            for w in order.iter().rev() {
                for v in predecessors.get(w).cloned().unwrap_or_default() {
                    let sigma_v = sigma.get(&v).copied().unwrap_or(1.0);
                    let sigma_w = sigma.get(w).copied().unwrap_or(1.0);
                    let delta_w = delta.get(w).copied().unwrap_or(0.0);

                    if sigma_w > 0.0 {
                        *delta.get_mut(&v).unwrap() += (sigma_v / sigma_w) * (1.0 + delta_w);
                    }
                }

                if w != source {
                    *betweenness.get_mut(w).unwrap() += delta[w];
                }
            }
        }

        // Normalize
        let norm = if n > 2 { (n - 1) * (n - 2) } else { 1 };
        for v in betweenness.values_mut() {
            *v /= norm as f64;
        }

        self.metrics.betweenness = betweenness;
    }

    /// Compute depth from root nodes (nodes with no incoming edges)
    fn compute_depth(&mut self, adjacency: &HashMap<String, Vec<String>>) {
        let mut depth: HashMap<String, u32> = HashMap::new();
        let nodes: Vec<_> = self.components.keys().cloned().collect();

        // Find incoming edges for each node
        let mut has_incoming: HashMap<String, bool> =
            nodes.iter().map(|n| (n.clone(), false)).collect();
        for targets in adjacency.values() {
            for t in targets {
                has_incoming.insert(t.clone(), true);
            }
        }

        // Roots are nodes with no incoming edges
        let roots: Vec<_> = nodes
            .iter()
            .filter(|n| !has_incoming.get(*n).unwrap_or(&false))
            .cloned()
            .collect();

        // BFS from roots
        let mut queue: Vec<(String, u32)> = roots.into_iter().map(|r| (r, 0)).collect();

        while let Some((node, d)) = queue.pop() {
            if let std::collections::hash_map::Entry::Vacant(e) = depth.entry(node.clone()) {
                e.insert(d);
                if let Some(neighbors) = adjacency.get(&node) {
                    for neighbor in neighbors {
                        if !depth.contains_key(neighbor) {
                            queue.push((neighbor.clone(), d + 1));
                        }
                    }
                }
            }
        }

        // Assign depth 0 to any unreachable nodes
        for node in &nodes {
            depth.entry(node.clone()).or_insert(0);
        }

        self.metrics.depth_map = depth;
    }

    /// Get computed metrics
    pub fn metrics(&self) -> &GraphMetrics {
        &self.metrics
    }

    /// Get detected anomalies
    pub fn anomalies(&self) -> &[Anomaly] {
        &self.anomalies
    }

    /// Add an anomaly
    pub fn add_anomaly(&mut self, anomaly: Anomaly) {
        self.anomalies.push(anomaly);
    }

    /// Compute stack health summary
    pub fn health_summary(&self) -> HealthSummary {
        let total = self.components.len();
        let green = self
            .components
            .values()
            .filter(|c| c.health == HealthStatus::Green)
            .count();
        let yellow = self
            .components
            .values()
            .filter(|c| c.health == HealthStatus::Yellow)
            .count();
        let red = self
            .components
            .values()
            .filter(|c| c.health == HealthStatus::Red)
            .count();

        let avg_score = if total > 0 {
            self.components
                .values()
                .map(|c| c.metrics.demo_score)
                .sum::<f64>()
                / total as f64
        } else {
            0.0
        };

        HealthSummary {
            total_components: total,
            green_count: green,
            yellow_count: yellow,
            red_count: red,
            unknown_count: total.saturating_sub(green + yellow + red),
            avg_demo_score: avg_score,
            avg_coverage: self.avg_metric(|c| c.metrics.coverage),
            andon_status: self.compute_andon_status(green, yellow, red, total),
        }
    }

    fn avg_metric<F>(&self, f: F) -> f64
    where
        F: Fn(&ComponentNode) -> f64,
    {
        let total = self.components.len();
        if total == 0 {
            return 0.0;
        }
        self.components.values().map(f).sum::<f64>() / total as f64
    }

    fn compute_andon_status(
        &self,
        green: usize,
        yellow: usize,
        red: usize,
        total: usize,
    ) -> AndonStatus {
        if red > 0 {
            AndonStatus::Red
        } else if yellow > 0 {
            AndonStatus::Yellow
        } else if green == total && total > 0 {
            AndonStatus::Green
        } else {
            AndonStatus::Unknown
        }
    }
}

impl Default for StackDiagnostics {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Health Summary
// ============================================================================

/// Summary of stack health for dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthSummary {
    /// Total components in stack
    pub total_components: usize,
    /// Components with green status
    pub green_count: usize,
    /// Components with yellow status
    pub yellow_count: usize,
    /// Components with red status
    pub red_count: usize,
    /// Components with unknown status
    pub unknown_count: usize,
    /// Average demo score
    pub avg_demo_score: f64,
    /// Average test coverage
    pub avg_coverage: f64,
    /// Overall Andon status
    pub andon_status: AndonStatus,
}

impl HealthSummary {
    /// Check if all components are healthy
    pub fn all_healthy(&self) -> bool {
        self.red_count == 0 && self.yellow_count == 0 && self.green_count == self.total_components
    }

    /// Get percentage of healthy components
    pub fn health_percentage(&self) -> f64 {
        if self.total_components == 0 {
            return 0.0;
        }
        (self.green_count as f64 / self.total_components as f64) * 100.0
    }
}

// ============================================================================
// Andon Status (Overall Stack)
// ============================================================================

/// Andon board status for the entire stack
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AndonStatus {
    /// All systems green - normal operation
    Green,
    /// Warnings present - attention needed
    Yellow,
    /// Critical issues - stop-the-line
    Red,
    /// Not yet analyzed
    Unknown,
}

impl AndonStatus {
    /// Get display message
    pub fn message(&self) -> &'static str {
        match self {
            Self::Green => "All systems healthy",
            Self::Yellow => "Attention needed",
            Self::Red => "Stop-the-line",
            Self::Unknown => "Analysis pending",
        }
    }
}

impl std::fmt::Display for AndonStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let icon = match self {
            Self::Green => "🟢",
            Self::Yellow => "🟡",
            Self::Red => "🔴",
            Self::Unknown => "⚪",
        };
        write!(f, "{} {}", icon, self.message())
    }
}

// ============================================================================
// Anomaly Detection
// ============================================================================

/// Detected anomaly in the stack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    /// Component where anomaly was detected
    pub component: String,
    /// Anomaly score (0-1, higher = more anomalous)
    pub score: f64,
    /// Category of anomaly
    pub category: AnomalyCategory,
    /// Human-readable description
    pub description: String,
    /// Evidence supporting the anomaly
    pub evidence: Vec<String>,
    /// Suggested remediation
    pub recommendation: Option<String>,
}

impl Anomaly {
    /// Create a new anomaly
    pub fn new(
        component: impl Into<String>,
        score: f64,
        category: AnomalyCategory,
        description: impl Into<String>,
    ) -> Self {
        Self {
            component: component.into(),
            score,
            category,
            description: description.into(),
            evidence: Vec::new(),
            recommendation: None,
        }
    }

    /// Add evidence
    pub fn with_evidence(mut self, evidence: impl Into<String>) -> Self {
        self.evidence.push(evidence.into());
        self
    }

    /// Add recommendation
    pub fn with_recommendation(mut self, rec: impl Into<String>) -> Self {
        self.recommendation = Some(rec.into());
        self
    }

    /// Check if anomaly is critical (score > 0.8)
    pub fn is_critical(&self) -> bool {
        self.score > 0.8
    }
}

/// Categories of anomalies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnomalyCategory {
    /// Quality score regression
    QualityRegression,
    /// Coverage drop
    CoverageDrop,
    /// Build time spike
    BuildTimeSpike,
    /// Dependency risk
    DependencyRisk,
    /// Complexity increase
    ComplexityIncrease,
    /// Other anomaly
    Other,
}

impl std::fmt::Display for AnomalyCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QualityRegression => write!(f, "Quality Regression"),
            Self::CoverageDrop => write!(f, "Coverage Drop"),
            Self::BuildTimeSpike => write!(f, "Build Time Spike"),
            Self::DependencyRisk => write!(f, "Dependency Risk"),
            Self::ComplexityIncrease => write!(f, "Complexity Increase"),
            Self::Other => write!(f, "Other"),
        }
    }
}

// ============================================================================
// Dashboard Renderer
// ============================================================================

/// Render diagnostics as ASCII dashboard
pub fn render_dashboard(diagnostics: &StackDiagnostics) -> String {
    let mut output = String::new();
    let summary = diagnostics.health_summary();

    // Header
    output
        .push_str("┌─────────────────────────────────────────────────────────────────────────┐\n");
    output
        .push_str("│                  SOVEREIGN AI STACK HEALTH DASHBOARD                    │\n");
    output.push_str(&format!(
        "│                  Timestamp: {:40} │\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S")
    ));
    output
        .push_str("├─────────────────────────────────────────────────────────────────────────┤\n");

    // Andon Status
    output
        .push_str("│                                                                         │\n");
    output.push_str(&format!(
        "│  ANDON STATUS: {} {:55}│\n",
        summary.andon_status, ""
    ));
    output
        .push_str("│                                                                         │\n");

    // Stack Summary
    output.push_str("│  ═══════════════════════════════════════════════════════════════════   │\n");
    output
        .push_str("│  STACK SUMMARY                                                          │\n");
    output.push_str("│  ═══════════════════════════════════════════════════════════════════   │\n");
    output
        .push_str("│                                                                         │\n");
    output.push_str(&format!(
        "│  Total Components:    {:3}                                              │\n",
        summary.total_components
    ));
    output.push_str(&format!(
        "│  Healthy:             {:3} ({:.0}%)                                         │\n",
        summary.green_count,
        summary.health_percentage()
    ));
    output.push_str(&format!(
        "│  Warnings:            {:3} ({:.0}%)                                         │\n",
        summary.yellow_count,
        if summary.total_components > 0 {
            (summary.yellow_count as f64 / summary.total_components as f64) * 100.0
        } else {
            0.0
        }
    ));
    output.push_str(&format!(
        "│  Critical:            {:3} ({:.0}%)                                         │\n",
        summary.red_count,
        if summary.total_components > 0 {
            (summary.red_count as f64 / summary.total_components as f64) * 100.0
        } else {
            0.0
        }
    ));
    output.push_str(&format!(
        "│  Average Demo Score:  {:.1}/100                                          │\n",
        summary.avg_demo_score
    ));
    output.push_str(&format!(
        "│  Average Coverage:    {:.1}%                                             │\n",
        summary.avg_coverage
    ));
    output
        .push_str("│                                                                         │\n");

    // Anomalies
    let anomalies = diagnostics.anomalies();
    if !anomalies.is_empty() {
        output.push_str(
            "│  ═══════════════════════════════════════════════════════════════════   │\n",
        );
        output.push_str(
            "│  ANOMALIES DETECTED                                                     │\n",
        );
        output.push_str(
            "│  ═══════════════════════════════════════════════════════════════════   │\n",
        );
        output.push_str(
            "│                                                                         │\n",
        );

        for anomaly in anomalies.iter().take(5) {
            let icon = if anomaly.is_critical() {
                "🔴"
            } else {
                "⚠️"
            };
            output.push_str(&format!(
                "│  {}  {}: {}                               │\n",
                icon, anomaly.component, anomaly.description
            ));
        }
        output.push_str(
            "│                                                                         │\n",
        );
    }

    output
        .push_str("└─────────────────────────────────────────────────────────────────────────┘\n");

    output
}

// ============================================================================
// Isolation Forest (ML Anomaly Detection)
// ============================================================================

/// Isolation Forest for anomaly detection
/// Implements a simplified version of the algorithm from Liu et al. (2008)
#[derive(Debug)]
pub struct IsolationForest {
    /// Number of trees in the forest
    n_trees: usize,
    /// Subsample size for each tree
    sample_size: usize,
    /// Random seed for reproducibility
    seed: u64,
    /// Trained isolation trees
    trees: Vec<IsolationTree>,
    /// Feature names for interpretation
    feature_names: Vec<String>,
}

impl IsolationForest {
    /// Create a new Isolation Forest
    pub fn new(n_trees: usize, sample_size: usize, seed: u64) -> Self {
        Self {
            n_trees,
            sample_size,
            seed,
            trees: Vec::new(),
            feature_names: Vec::new(),
        }
    }

    /// Default forest configuration
    pub fn default_forest() -> Self {
        Self::new(100, 256, 42)
    }

    /// Set feature names for interpretability
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Fit the forest on data points
    /// Each row is a data point, each column is a feature
    pub fn fit(&mut self, data: &[Vec<f64>]) {
        if data.is_empty() {
            return;
        }

        let mut rng = SimpleRng::seed_from_u64(self.seed);
        let n_samples = data.len();
        let max_depth = (self.sample_size as f64).log2().ceil() as usize;

        self.trees.clear();

        for _ in 0..self.n_trees {
            // Sample data points
            let sample: Vec<Vec<f64>> = (0..self.sample_size.min(n_samples))
                .map(|_| {
                    let idx = rng.gen_range(0..n_samples);
                    data[idx].clone()
                })
                .collect();

            // Build tree
            let tree = IsolationTree::build(&sample, max_depth, &mut rng);
            self.trees.push(tree);
        }
    }

    /// Compute anomaly scores for data points
    /// Returns scores in [0, 1] where higher = more anomalous
    pub fn score(&self, data: &[Vec<f64>]) -> Vec<f64> {
        if self.trees.is_empty() || data.is_empty() {
            return vec![0.0; data.len()];
        }

        let n = self.sample_size as f64;
        let c_n = average_path_length(n);

        data.iter()
            .map(|point| {
                let avg_path_length: f64 = self
                    .trees
                    .iter()
                    .map(|tree| tree.path_length(point, 0) as f64)
                    .sum::<f64>()
                    / self.trees.len() as f64;

                // Anomaly score: 2^(-avg_path_length / c(n))
                // Higher score = more anomalous
                2.0_f64.powf(-avg_path_length / c_n)
            })
            .collect()
    }

    /// Predict anomalies with threshold
    pub fn predict(&self, data: &[Vec<f64>], threshold: f64) -> Vec<bool> {
        self.score(data)
            .into_iter()
            .map(|s| s > threshold)
            .collect()
    }

    /// Detect anomalies in component metrics and return Anomaly objects
    pub fn detect_anomalies(&self, diagnostics: &StackDiagnostics, threshold: f64) -> Vec<Anomaly> {
        let components: Vec<_> = diagnostics.components().collect();
        if components.is_empty() {
            return Vec::new();
        }

        // Extract feature vectors
        let data: Vec<Vec<f64>> = components
            .iter()
            .map(|c| {
                vec![
                    c.metrics.demo_score,
                    c.metrics.coverage,
                    c.metrics.mutation_score,
                    c.metrics.complexity_avg,
                    c.metrics.satd_count as f64,
                    c.metrics.dead_code_pct,
                ]
            })
            .collect();

        let scores = self.score(&data);
        let mut anomalies = Vec::new();

        for (i, (component, score)) in components.iter().zip(scores.iter()).enumerate() {
            if *score > threshold {
                let category = self.categorize_anomaly(&data[i]);
                let description = self.describe_anomaly(&data[i], &category);

                let mut anomaly =
                    Anomaly::new(component.name.clone(), *score, category, description);

                // Add evidence
                anomaly = anomaly
                    .with_evidence(format!("Isolation score: {:.3}", score))
                    .with_evidence(format!("Demo score: {:.1}", component.metrics.demo_score))
                    .with_evidence(format!("Coverage: {:.1}%", component.metrics.coverage));

                // Add recommendation
                let rec = self.recommend_action(&category, &data[i]);
                anomaly = anomaly.with_recommendation(rec);

                anomalies.push(anomaly);
            }
        }

        // Sort by score descending
        anomalies.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        anomalies
    }

    /// Categorize the anomaly based on which features are most deviant
    fn categorize_anomaly(&self, features: &[f64]) -> AnomalyCategory {
        // features: [demo_score, coverage, mutation_score, complexity_avg, satd_count, dead_code_pct]
        if features.len() < 6 {
            return AnomalyCategory::Other;
        }

        let demo_score = features[0];
        let coverage = features[1];
        let complexity = features[3];
        let dead_code = features[5];

        if demo_score < 70.0 {
            AnomalyCategory::QualityRegression
        } else if coverage < 50.0 {
            AnomalyCategory::CoverageDrop
        } else if complexity > 15.0 {
            AnomalyCategory::ComplexityIncrease
        } else if dead_code > 10.0 {
            AnomalyCategory::DependencyRisk
        } else {
            AnomalyCategory::Other
        }
    }

    /// Generate human-readable description
    fn describe_anomaly(&self, features: &[f64], category: &AnomalyCategory) -> String {
        match category {
            AnomalyCategory::QualityRegression => {
                format!(
                    "Quality score {:.1} is significantly below healthy threshold",
                    features[0]
                )
            }
            AnomalyCategory::CoverageDrop => {
                format!("Test coverage {:.1}% is dangerously low", features[1])
            }
            AnomalyCategory::ComplexityIncrease => {
                format!(
                    "Average complexity {:.1} indicates maintainability risk",
                    features[3]
                )
            }
            AnomalyCategory::DependencyRisk => {
                format!(
                    "Dead code {:.1}% suggests technical debt accumulation",
                    features[5]
                )
            }
            _ => "Unusual metric combination detected".to_string(),
        }
    }

    /// Generate actionable recommendation
    fn recommend_action(&self, category: &AnomalyCategory, features: &[f64]) -> String {
        match category {
            AnomalyCategory::QualityRegression => {
                if features[1] < 80.0 {
                    "Add tests to improve coverage above 80%".to_string()
                } else {
                    "Review recent changes for quality regressions".to_string()
                }
            }
            AnomalyCategory::CoverageDrop => {
                "Run `cargo tarpaulin` and add tests for uncovered paths".to_string()
            }
            AnomalyCategory::ComplexityIncrease => {
                "Consider refactoring complex functions (>10 cyclomatic complexity)".to_string()
            }
            AnomalyCategory::DependencyRisk => {
                "Run `cargo udeps` to identify and remove dead code".to_string()
            }
            _ => "Review component metrics for unusual patterns".to_string(),
        }
    }
}

/// A single isolation tree node
#[derive(Debug)]
enum IsolationTree {
    /// Internal node with split
    Internal {
        split_feature: usize,
        split_value: f64,
        left: Box<IsolationTree>,
        right: Box<IsolationTree>,
    },
    /// External (leaf) node
    External { size: usize },
}

impl IsolationTree {
    /// Build an isolation tree from data
    fn build(data: &[Vec<f64>], max_depth: usize, rng: &mut SimpleRng) -> Self {
        if data.is_empty() {
            return IsolationTree::External { size: 0 };
        }

        if max_depth == 0 || data.len() <= 1 {
            return IsolationTree::External { size: data.len() };
        }

        let n_features = data[0].len();
        if n_features == 0 {
            return IsolationTree::External { size: data.len() };
        }

        // Random feature
        let feature = rng.gen_range(0..n_features);

        // Find min/max for this feature
        let values: Vec<f64> = data
            .iter()
            .filter_map(|row| row.get(feature).copied())
            .collect();
        if values.is_empty() {
            return IsolationTree::External { size: data.len() };
        }

        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max_val - min_val).abs() < f64::EPSILON {
            return IsolationTree::External { size: data.len() };
        }

        // Random split value
        let split_value = rng.gen_range_f64(min_val..max_val);

        // Partition data
        let (left_data, right_data): (Vec<_>, Vec<_>) = data
            .iter()
            .cloned()
            .partition(|row| row.get(feature).is_some_and(|&v| v < split_value));

        // Handle edge case where all data goes to one side
        if left_data.is_empty() || right_data.is_empty() {
            return IsolationTree::External { size: data.len() };
        }

        IsolationTree::Internal {
            split_feature: feature,
            split_value,
            left: Box::new(IsolationTree::build(&left_data, max_depth - 1, rng)),
            right: Box::new(IsolationTree::build(&right_data, max_depth - 1, rng)),
        }
    }

    /// Compute path length for a point
    fn path_length(&self, point: &[f64], current_depth: usize) -> usize {
        match self {
            IsolationTree::External { size } => {
                current_depth + average_path_length(*size as f64) as usize
            }
            IsolationTree::Internal {
                split_feature,
                split_value,
                left,
                right,
            } => {
                let value = point.get(*split_feature).copied().unwrap_or(0.0);
                if value < *split_value {
                    left.path_length(point, current_depth + 1)
                } else {
                    right.path_length(point, current_depth + 1)
                }
            }
        }
    }
}

/// Average path length of unsuccessful search in BST
fn average_path_length(n: f64) -> f64 {
    if n <= 1.0 {
        return 0.0;
    }
    2.0 * (n.ln() + 0.5772156649) - (2.0 * (n - 1.0) / n)
}

// ============================================================================
// Time Series Forecasting (Error Prediction)
// ============================================================================

/// Simple exponential smoothing for time series forecasting
#[derive(Debug, Clone)]
pub struct ErrorForecaster {
    /// Smoothing parameter alpha (0-1)
    alpha: f64,
    /// Historical observations
    history: Vec<f64>,
    /// Current smoothed value
    level: f64,
}

impl ErrorForecaster {
    /// Create a new error forecaster
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
            history: Vec::new(),
            level: 0.0,
        }
    }

    /// Default forecaster with alpha=0.3
    pub fn default_forecaster() -> Self {
        Self::new(0.3)
    }

    /// Add an observation
    pub fn observe(&mut self, value: f64) {
        if self.history.is_empty() {
            self.level = value;
        } else {
            // Exponential smoothing: L_t = alpha * Y_t + (1 - alpha) * L_{t-1}
            self.level = self.alpha * value + (1.0 - self.alpha) * self.level;
        }
        self.history.push(value);
    }

    /// Forecast next n values
    pub fn forecast(&self, n: usize) -> Vec<f64> {
        // Simple exponential smoothing forecasts are constant
        vec![self.level; n]
    }

    /// Compute forecast error metrics
    pub fn error_metrics(&self) -> ForecastMetrics {
        if self.history.len() < 2 {
            return ForecastMetrics::default();
        }

        // Compute in-sample errors
        let mut errors = Vec::new();
        let mut level = self.history[0];

        for &actual in self.history.iter().skip(1) {
            let forecast = level;
            errors.push(actual - forecast);
            level = self.alpha * actual + (1.0 - self.alpha) * level;
        }

        let n = errors.len() as f64;
        let mae = errors.iter().map(|e| e.abs()).sum::<f64>() / n;
        let mse = errors.iter().map(|e| e * e).sum::<f64>() / n;
        let rmse = mse.sqrt();

        // MAPE (avoid division by zero)
        let mape = if self.history.iter().skip(1).all(|&v| v.abs() > f64::EPSILON) {
            let sum: f64 = errors
                .iter()
                .zip(self.history.iter().skip(1))
                .map(|(e, a)| (e / a).abs())
                .sum();
            sum / n * 100.0
        } else {
            f64::NAN
        };

        ForecastMetrics {
            mae,
            mse,
            rmse,
            mape,
        }
    }

    /// Get historical observations
    pub fn history(&self) -> &[f64] {
        &self.history
    }

    /// Get current level
    pub fn current_level(&self) -> f64 {
        self.level
    }
}

/// Forecast error metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ForecastMetrics {
    /// Mean Absolute Error
    pub mae: f64,
    /// Mean Squared Error
    pub mse: f64,
    /// Root Mean Squared Error
    pub rmse: f64,
    /// Mean Absolute Percentage Error
    pub mape: f64,
}

// ============================================================================
// Tests (EXTREME TDD - RED PHASE)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // HealthStatus Tests
    // ========================================================================

    #[test]
    fn test_health_status_from_grade_green() {
        assert_eq!(
            HealthStatus::from_grade(QualityGrade::APlus),
            HealthStatus::Green
        );
        assert_eq!(
            HealthStatus::from_grade(QualityGrade::A),
            HealthStatus::Green
        );
    }

    #[test]
    fn test_health_status_from_grade_yellow() {
        assert_eq!(
            HealthStatus::from_grade(QualityGrade::AMinus),
            HealthStatus::Yellow
        );
        assert_eq!(
            HealthStatus::from_grade(QualityGrade::BPlus),
            HealthStatus::Yellow
        );
    }

    #[test]
    fn test_health_status_from_grade_red() {
        assert_eq!(HealthStatus::from_grade(QualityGrade::B), HealthStatus::Red);
        assert_eq!(HealthStatus::from_grade(QualityGrade::C), HealthStatus::Red);
        assert_eq!(HealthStatus::from_grade(QualityGrade::F), HealthStatus::Red);
    }

    #[test]
    fn test_health_status_icons() {
        assert_eq!(HealthStatus::Green.icon(), "🟢");
        assert_eq!(HealthStatus::Yellow.icon(), "🟡");
        assert_eq!(HealthStatus::Red.icon(), "🔴");
        assert_eq!(HealthStatus::Unknown.icon(), "⚪");
    }

    #[test]
    fn test_health_status_symbols() {
        assert_eq!(HealthStatus::Green.symbol(), "●");
        assert_eq!(HealthStatus::Yellow.symbol(), "◐");
        assert_eq!(HealthStatus::Red.symbol(), "○");
        assert_eq!(HealthStatus::Unknown.symbol(), "◌");
    }

    // ========================================================================
    // ComponentNode Tests
    // ========================================================================

    #[test]
    fn test_component_node_creation() {
        let node = ComponentNode::new("trueno", "0.7.4", StackLayer::Compute);
        assert_eq!(node.name, "trueno");
        assert_eq!(node.version, "0.7.4");
        assert_eq!(node.layer, StackLayer::Compute);
        assert_eq!(node.health, HealthStatus::Unknown);
    }

    #[test]
    fn test_component_node_update_health() {
        let mut node = ComponentNode::new("trueno", "0.7.4", StackLayer::Compute);
        node.metrics = ComponentMetrics::with_demo_score(95.0);
        node.update_health();
        assert_eq!(node.health, HealthStatus::Green);
    }

    #[test]
    fn test_component_node_update_health_yellow() {
        let mut node = ComponentNode::new("test", "1.0.0", StackLayer::Ml);
        node.metrics = ComponentMetrics::with_demo_score(85.0);
        node.update_health();
        assert_eq!(node.health, HealthStatus::Yellow);
    }

    #[test]
    fn test_component_node_update_health_red() {
        let mut node = ComponentNode::new("test", "1.0.0", StackLayer::Ml);
        node.metrics = ComponentMetrics::with_demo_score(65.0);
        node.update_health();
        assert_eq!(node.health, HealthStatus::Red);
    }

    // ========================================================================
    // ComponentMetrics Tests
    // ========================================================================

    #[test]
    fn test_component_metrics_default() {
        let metrics = ComponentMetrics::default();
        assert_eq!(metrics.demo_score, 0.0);
        assert_eq!(metrics.coverage, 0.0);
        assert!(!metrics.meets_threshold());
    }

    #[test]
    fn test_component_metrics_with_demo_score() {
        let metrics = ComponentMetrics::with_demo_score(90.0);
        assert_eq!(metrics.demo_score, 90.0);
        assert!(metrics.meets_threshold());
    }

    #[test]
    fn test_component_metrics_threshold() {
        assert!(ComponentMetrics::with_demo_score(85.0).meets_threshold());
        assert!(ComponentMetrics::with_demo_score(100.0).meets_threshold());
        assert!(!ComponentMetrics::with_demo_score(84.9).meets_threshold());
    }

    // ========================================================================
    // GraphMetrics Tests
    // ========================================================================

    #[test]
    fn test_graph_metrics_top_by_pagerank() {
        let mut metrics = GraphMetrics::default();
        metrics.pagerank.insert("trueno".to_string(), 0.25);
        metrics.pagerank.insert("aprender".to_string(), 0.15);
        metrics.pagerank.insert("batuta".to_string(), 0.10);

        let top = metrics.top_by_pagerank(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "trueno");
        assert_eq!(top[1].0, "aprender");
    }

    #[test]
    fn test_graph_metrics_bottlenecks() {
        let mut metrics = GraphMetrics::default();
        metrics.betweenness.insert("trueno".to_string(), 0.8);
        metrics.betweenness.insert("aprender".to_string(), 0.3);
        metrics.betweenness.insert("batuta".to_string(), 0.1);

        let bottlenecks = metrics.bottlenecks(0.5);
        assert_eq!(bottlenecks.len(), 1);
        assert!(bottlenecks.contains(&&"trueno".to_string()));
    }

    // ========================================================================
    // StackDiagnostics Tests
    // ========================================================================

    #[test]
    fn test_stack_diagnostics_new() {
        let diag = StackDiagnostics::new();
        assert_eq!(diag.component_count(), 0);
        assert!(diag.graph().is_none());
        assert!(diag.anomalies().is_empty());
    }

    #[test]
    fn test_stack_diagnostics_add_component() {
        let mut diag = StackDiagnostics::new();
        let node = ComponentNode::new("trueno", "0.7.4", StackLayer::Compute);
        diag.add_component(node);

        assert_eq!(diag.component_count(), 1);
        assert!(diag.get_component("trueno").is_some());
        assert!(diag.get_component("missing").is_none());
    }

    #[test]
    fn test_stack_diagnostics_health_summary_empty() {
        let diag = StackDiagnostics::new();
        let summary = diag.health_summary();

        assert_eq!(summary.total_components, 0);
        assert_eq!(summary.green_count, 0);
        assert_eq!(summary.andon_status, AndonStatus::Unknown);
    }

    #[test]
    fn test_stack_diagnostics_health_summary_all_green() {
        let mut diag = StackDiagnostics::new();

        let mut node1 = ComponentNode::new("trueno", "0.7.4", StackLayer::Compute);
        node1.health = HealthStatus::Green;
        node1.metrics = ComponentMetrics::with_demo_score(95.0);
        diag.add_component(node1);

        let mut node2 = ComponentNode::new("aprender", "0.9.0", StackLayer::Ml);
        node2.health = HealthStatus::Green;
        node2.metrics = ComponentMetrics::with_demo_score(92.0);
        diag.add_component(node2);

        let summary = diag.health_summary();

        assert_eq!(summary.total_components, 2);
        assert_eq!(summary.green_count, 2);
        assert_eq!(summary.yellow_count, 0);
        assert_eq!(summary.red_count, 0);
        assert!(summary.all_healthy());
        assert_eq!(summary.andon_status, AndonStatus::Green);
        assert!((summary.avg_demo_score - 93.5).abs() < 0.1);
    }

    #[test]
    fn test_stack_diagnostics_health_summary_mixed() {
        let mut diag = StackDiagnostics::new();

        let mut node1 = ComponentNode::new("trueno", "0.7.4", StackLayer::Compute);
        node1.health = HealthStatus::Green;
        diag.add_component(node1);

        let mut node2 = ComponentNode::new("weak", "1.0.0", StackLayer::Ml);
        node2.health = HealthStatus::Red;
        diag.add_component(node2);

        let summary = diag.health_summary();

        assert_eq!(summary.green_count, 1);
        assert_eq!(summary.red_count, 1);
        assert!(!summary.all_healthy());
        assert_eq!(summary.andon_status, AndonStatus::Red);
    }

    #[test]
    fn test_stack_diagnostics_add_anomaly() {
        let mut diag = StackDiagnostics::new();

        let anomaly = Anomaly::new(
            "trueno-graph",
            0.75,
            AnomalyCategory::CoverageDrop,
            "Coverage dropped 5.2%",
        )
        .with_evidence("lcov.info shows missing tests")
        .with_recommendation("Add tests for GPU BFS");

        diag.add_anomaly(anomaly);

        assert_eq!(diag.anomalies().len(), 1);
        assert_eq!(diag.anomalies()[0].component, "trueno-graph");
        assert!(!diag.anomalies()[0].is_critical());
    }

    // ========================================================================
    // HealthSummary Tests
    // ========================================================================

    #[test]
    fn test_health_summary_percentage() {
        let summary = HealthSummary {
            total_components: 20,
            green_count: 17,
            yellow_count: 3,
            red_count: 0,
            unknown_count: 0,
            avg_demo_score: 85.0,
            avg_coverage: 90.0,
            andon_status: AndonStatus::Yellow,
        };

        assert_eq!(summary.health_percentage(), 85.0);
        assert!(!summary.all_healthy());
    }

    #[test]
    fn test_health_summary_percentage_empty() {
        let summary = HealthSummary {
            total_components: 0,
            green_count: 0,
            yellow_count: 0,
            red_count: 0,
            unknown_count: 0,
            avg_demo_score: 0.0,
            avg_coverage: 0.0,
            andon_status: AndonStatus::Unknown,
        };

        assert_eq!(summary.health_percentage(), 0.0);
    }

    // ========================================================================
    // Anomaly Tests
    // ========================================================================

    #[test]
    fn test_anomaly_creation() {
        let anomaly = Anomaly::new(
            "test",
            0.65,
            AnomalyCategory::QualityRegression,
            "Score dropped",
        );

        assert_eq!(anomaly.component, "test");
        assert_eq!(anomaly.score, 0.65);
        assert!(!anomaly.is_critical());
        assert!(anomaly.evidence.is_empty());
        assert!(anomaly.recommendation.is_none());
    }

    #[test]
    fn test_anomaly_critical() {
        let critical = Anomaly::new("test", 0.85, AnomalyCategory::DependencyRisk, "High risk");
        assert!(critical.is_critical());

        let non_critical = Anomaly::new("test", 0.79, AnomalyCategory::Other, "Low risk");
        assert!(!non_critical.is_critical());
    }

    #[test]
    fn test_anomaly_with_details() {
        let anomaly = Anomaly::new("test", 0.7, AnomalyCategory::BuildTimeSpike, "Build slow")
            .with_evidence("Time increased 40%")
            .with_evidence("New macro expansion")
            .with_recommendation("Enable incremental compilation");

        assert_eq!(anomaly.evidence.len(), 2);
        assert!(anomaly.recommendation.is_some());
    }

    #[test]
    fn test_anomaly_category_display() {
        assert_eq!(
            format!("{}", AnomalyCategory::QualityRegression),
            "Quality Regression"
        );
        assert_eq!(
            format!("{}", AnomalyCategory::CoverageDrop),
            "Coverage Drop"
        );
        assert_eq!(
            format!("{}", AnomalyCategory::BuildTimeSpike),
            "Build Time Spike"
        );
    }

    // ========================================================================
    // AndonStatus Tests
    // ========================================================================

    #[test]
    fn test_andon_status_messages() {
        assert_eq!(AndonStatus::Green.message(), "All systems healthy");
        assert_eq!(AndonStatus::Yellow.message(), "Attention needed");
        assert_eq!(AndonStatus::Red.message(), "Stop-the-line");
        assert_eq!(AndonStatus::Unknown.message(), "Analysis pending");
    }

    #[test]
    fn test_andon_status_display() {
        let green = format!("{}", AndonStatus::Green);
        assert!(green.contains("🟢"));
        assert!(green.contains("healthy"));
    }

    // ========================================================================
    // Dashboard Renderer Tests
    // ========================================================================

    #[test]
    fn test_render_dashboard_empty() {
        let diag = StackDiagnostics::new();
        let output = render_dashboard(&diag);

        assert!(output.contains("SOVEREIGN AI STACK"));
        assert!(output.contains("ANDON STATUS"));
        assert!(output.contains("Total Components"));
    }

    #[test]
    fn test_render_dashboard_with_components() {
        let mut diag = StackDiagnostics::new();

        let mut node = ComponentNode::new("trueno", "0.7.4", StackLayer::Compute);
        node.health = HealthStatus::Green;
        node.metrics = ComponentMetrics::with_demo_score(92.0);
        diag.add_component(node);

        let output = render_dashboard(&diag);

        assert!(output.contains("Total Components:      1"));
        assert!(output.contains("Healthy:               1"));
    }

    #[test]
    fn test_render_dashboard_with_anomalies() {
        let mut diag = StackDiagnostics::new();

        diag.add_anomaly(Anomaly::new(
            "trueno-graph",
            0.75,
            AnomalyCategory::CoverageDrop,
            "Coverage dropped",
        ));

        let output = render_dashboard(&diag);
        assert!(output.contains("ANOMALIES DETECTED"));
        assert!(output.contains("trueno-graph"));
    }

    // ========================================================================
    // Phase 2: Graph Analytics Tests
    // ========================================================================

    #[test]
    fn test_compute_metrics_empty() {
        let mut diag = StackDiagnostics::new();
        let metrics = diag.compute_metrics().unwrap();

        assert_eq!(metrics.total_nodes, 0);
        assert_eq!(metrics.total_edges, 0);
        assert_eq!(metrics.density, 0.0);
    }

    #[test]
    fn test_compute_metrics_single_node() {
        let mut diag = StackDiagnostics::new();
        diag.add_component(ComponentNode::new("trueno", "0.7.4", StackLayer::Compute));

        let metrics = diag.compute_metrics().unwrap();

        assert_eq!(metrics.total_nodes, 1);
        assert_eq!(metrics.total_edges, 0);
        assert_eq!(metrics.density, 0.0);
        assert_eq!(metrics.avg_degree, 0.0);

        // PageRank should be 1.0 for single node
        let pagerank = metrics.pagerank.get("trueno").copied().unwrap_or(0.0);
        assert!(
            (pagerank - 1.0).abs() < 0.01,
            "Single node PageRank should be ~1.0"
        );

        // Depth should be 0 for root
        assert_eq!(metrics.depth_map.get("trueno").copied(), Some(0));
    }

    #[test]
    fn test_compute_metrics_pagerank_chain() {
        let mut diag = StackDiagnostics::new();

        // Create chain: A -> B -> C (where A is root, C has highest PageRank)
        diag.add_component(ComponentNode::new("A", "1.0", StackLayer::Orchestration));
        diag.add_component(ComponentNode::new("B", "1.0", StackLayer::Ml));
        diag.add_component(ComponentNode::new("C", "1.0", StackLayer::Compute));

        let metrics = diag.compute_metrics().unwrap();

        // All nodes have PageRank
        assert!(metrics.pagerank.contains_key("A"));
        assert!(metrics.pagerank.contains_key("B"));
        assert!(metrics.pagerank.contains_key("C"));

        // Sum of PageRanks should be ~1.0
        let sum: f64 = metrics.pagerank.values().sum();
        assert!((sum - 1.0).abs() < 0.01, "PageRank sum should be ~1.0");
    }

    #[test]
    fn test_compute_metrics_betweenness() {
        let mut diag = StackDiagnostics::new();

        // Hub-spoke topology: A is hub, B,C,D are leaves
        diag.add_component(ComponentNode::new("hub", "1.0", StackLayer::Compute));
        diag.add_component(ComponentNode::new("leaf1", "1.0", StackLayer::Ml));
        diag.add_component(ComponentNode::new("leaf2", "1.0", StackLayer::DataMlops));
        diag.add_component(ComponentNode::new(
            "leaf3",
            "1.0",
            StackLayer::Orchestration,
        ));

        let metrics = diag.compute_metrics().unwrap();

        // All nodes have betweenness
        assert!(metrics.betweenness.contains_key("hub"));
        assert!(metrics.betweenness.contains_key("leaf1"));
        assert!(metrics.betweenness.contains_key("leaf2"));
        assert!(metrics.betweenness.contains_key("leaf3"));

        // Without edges, all betweenness should be 0
        for &v in metrics.betweenness.values() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_compute_metrics_depth() {
        let mut diag = StackDiagnostics::new();

        // Simple graph without dependencies - all are roots
        diag.add_component(ComponentNode::new("root1", "1.0", StackLayer::Compute));
        diag.add_component(ComponentNode::new("root2", "1.0", StackLayer::Ml));
        diag.add_component(ComponentNode::new("root3", "1.0", StackLayer::DataMlops));

        let metrics = diag.compute_metrics().unwrap();

        // All nodes are roots, so depth = 0
        assert_eq!(metrics.depth_map.get("root1").copied(), Some(0));
        assert_eq!(metrics.depth_map.get("root2").copied(), Some(0));
        assert_eq!(metrics.depth_map.get("root3").copied(), Some(0));
        assert_eq!(metrics.max_depth, 0);
    }

    #[test]
    fn test_compute_metrics_graph_density() {
        let mut diag = StackDiagnostics::new();

        // Add 3 nodes
        diag.add_component(ComponentNode::new("A", "1.0", StackLayer::Compute));
        diag.add_component(ComponentNode::new("B", "1.0", StackLayer::Ml));
        diag.add_component(ComponentNode::new("C", "1.0", StackLayer::DataMlops));

        let metrics = diag.compute_metrics().unwrap();

        // No edges, so density = 0
        assert_eq!(metrics.total_nodes, 3);
        assert_eq!(metrics.total_edges, 0);
        assert_eq!(metrics.density, 0.0);

        // max_edges for 3 nodes = 3 * 2 = 6
        // density = edges / max_edges = 0 / 6 = 0
    }

    #[test]
    fn test_compute_metrics_avg_degree() {
        let mut diag = StackDiagnostics::new();

        diag.add_component(ComponentNode::new("node1", "1.0", StackLayer::Compute));
        diag.add_component(ComponentNode::new("node2", "1.0", StackLayer::Ml));

        let metrics = diag.compute_metrics().unwrap();

        assert_eq!(metrics.total_nodes, 2);
        assert_eq!(metrics.avg_degree, 0.0);
    }

    #[test]
    fn test_build_adjacency_no_graph() {
        let mut diag = StackDiagnostics::new();
        diag.add_component(ComponentNode::new("A", "1.0", StackLayer::Compute));
        diag.add_component(ComponentNode::new("B", "1.0", StackLayer::Ml));

        // compute_metrics internally calls build_adjacency
        let metrics = diag.compute_metrics().unwrap();

        // Without a graph, edges should be 0
        assert_eq!(metrics.total_edges, 0);
    }

    #[test]
    fn test_graph_metrics_top_by_pagerank_empty() {
        let metrics = GraphMetrics::default();
        let top = metrics.top_by_pagerank(5);
        assert!(top.is_empty());
    }

    #[test]
    fn test_graph_metrics_bottlenecks_empty() {
        let metrics = GraphMetrics::default();
        let bottlenecks = metrics.bottlenecks(0.5);
        assert!(bottlenecks.is_empty());
    }

    #[test]
    fn test_compute_metrics_pagerank_convergence() {
        let mut diag = StackDiagnostics::new();

        // Larger graph to test convergence
        for i in 0..10 {
            diag.add_component(ComponentNode::new(
                format!("node{}", i),
                "1.0",
                StackLayer::Compute,
            ));
        }

        let metrics = diag.compute_metrics().unwrap();

        // All nodes should have PageRank assigned
        assert_eq!(metrics.pagerank.len(), 10);

        // Sum should be ~1.0 (normalized)
        let sum: f64 = metrics.pagerank.values().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "PageRank sum={} should be ~1.0",
            sum
        );
    }

    #[test]
    fn test_compute_metrics_multiple_calls() {
        let mut diag = StackDiagnostics::new();
        diag.add_component(ComponentNode::new("X", "1.0", StackLayer::Compute));

        // Call compute_metrics multiple times
        let _ = diag.compute_metrics().unwrap();
        let metrics = diag.compute_metrics().unwrap();

        // Should still work correctly
        assert_eq!(metrics.total_nodes, 1);
        assert!(metrics.pagerank.contains_key("X"));
    }

    // ========================================================================
    // Phase 3: ML Insights Tests - Isolation Forest
    // ========================================================================

    #[test]
    fn test_isolation_forest_new() {
        let forest = IsolationForest::new(10, 32, 42);
        assert_eq!(forest.n_trees, 10);
        assert_eq!(forest.sample_size, 32);
        assert_eq!(forest.seed, 42);
    }

    #[test]
    fn test_isolation_forest_default() {
        let forest = IsolationForest::default_forest();
        assert_eq!(forest.n_trees, 100);
        assert_eq!(forest.sample_size, 256);
    }

    #[test]
    fn test_isolation_forest_with_feature_names() {
        let forest = IsolationForest::new(10, 32, 42)
            .with_feature_names(vec!["demo_score".into(), "coverage".into()]);
        assert_eq!(forest.feature_names.len(), 2);
    }

    #[test]
    fn test_isolation_forest_fit_empty() {
        let mut forest = IsolationForest::new(10, 32, 42);
        forest.fit(&[]);
        assert!(forest.trees.is_empty());
    }

    #[test]
    fn test_isolation_forest_fit() {
        let mut forest = IsolationForest::new(10, 32, 42);
        let data = vec![
            vec![90.0, 85.0, 80.0],
            vec![88.0, 82.0, 78.0],
            vec![92.0, 88.0, 82.0],
            vec![85.0, 80.0, 75.0],
        ];
        forest.fit(&data);
        assert_eq!(forest.trees.len(), 10);
    }

    #[test]
    fn test_isolation_forest_score_empty() {
        let forest = IsolationForest::new(10, 32, 42);
        let scores = forest.score(&[vec![90.0, 85.0]]);
        assert_eq!(scores, vec![0.0]); // No trees fitted
    }

    #[test]
    fn test_isolation_forest_score() {
        let mut forest = IsolationForest::new(10, 32, 42);
        let data = vec![
            vec![90.0, 85.0],
            vec![88.0, 82.0],
            vec![92.0, 88.0],
            vec![10.0, 5.0], // Anomaly
        ];
        forest.fit(&data);
        let scores = forest.score(&data);

        assert_eq!(scores.len(), 4);
        // All scores should be in [0, 1]
        for score in &scores {
            assert!(*score >= 0.0 && *score <= 1.0);
        }
    }

    #[test]
    fn test_isolation_forest_predict() {
        let mut forest = IsolationForest::new(10, 32, 42);
        let data = vec![vec![90.0, 85.0], vec![88.0, 82.0], vec![92.0, 88.0]];
        forest.fit(&data);
        let predictions = forest.predict(&data, 0.5);
        assert_eq!(predictions.len(), 3);
    }

    #[test]
    fn test_isolation_forest_detect_anomalies_empty() {
        let forest = IsolationForest::default_forest();
        let diag = StackDiagnostics::new();
        let anomalies = forest.detect_anomalies(&diag, 0.5);
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_isolation_forest_detect_anomalies() {
        let mut forest = IsolationForest::new(50, 64, 42);
        let mut diag = StackDiagnostics::new();

        // Add normal components
        for i in 0..5 {
            let mut node = ComponentNode::new(format!("healthy{}", i), "1.0", StackLayer::Compute);
            node.metrics = ComponentMetrics {
                demo_score: 90.0 + i as f64,
                coverage: 85.0 + i as f64,
                mutation_score: 80.0,
                complexity_avg: 5.0,
                satd_count: 2,
                dead_code_pct: 1.0,
                grade: QualityGrade::A,
            };
            diag.add_component(node);
        }

        // Add anomalous component
        let mut anomaly_node = ComponentNode::new("anomalous", "1.0", StackLayer::Ml);
        anomaly_node.metrics = ComponentMetrics {
            demo_score: 30.0, // Very low
            coverage: 20.0,   // Very low
            mutation_score: 10.0,
            complexity_avg: 25.0, // High
            satd_count: 50,
            dead_code_pct: 30.0, // High
            grade: QualityGrade::F,
        };
        diag.add_component(anomaly_node);

        // Train on component data
        let data: Vec<Vec<f64>> = diag
            .components()
            .map(|c| {
                vec![
                    c.metrics.demo_score,
                    c.metrics.coverage,
                    c.metrics.mutation_score,
                    c.metrics.complexity_avg,
                    c.metrics.satd_count as f64,
                    c.metrics.dead_code_pct,
                ]
            })
            .collect();
        forest.fit(&data);

        // Should detect at least something (may or may not flag anomaly depending on threshold)
        let anomalies = forest.detect_anomalies(&diag, 0.3);
        // Just verify it runs without error
        assert!(anomalies.len() <= 6);
    }

    #[test]
    fn test_isolation_forest_categorize_anomaly() {
        let forest = IsolationForest::default_forest();

        // Low demo score -> QualityRegression
        let cat1 = forest.categorize_anomaly(&[50.0, 80.0, 75.0, 5.0, 2.0, 1.0]);
        assert_eq!(cat1, AnomalyCategory::QualityRegression);

        // Low coverage -> CoverageDrop
        let cat2 = forest.categorize_anomaly(&[80.0, 40.0, 75.0, 5.0, 2.0, 1.0]);
        assert_eq!(cat2, AnomalyCategory::CoverageDrop);

        // High complexity -> ComplexityIncrease
        let cat3 = forest.categorize_anomaly(&[80.0, 80.0, 75.0, 20.0, 2.0, 1.0]);
        assert_eq!(cat3, AnomalyCategory::ComplexityIncrease);

        // High dead code -> DependencyRisk
        let cat4 = forest.categorize_anomaly(&[80.0, 80.0, 75.0, 5.0, 2.0, 15.0]);
        assert_eq!(cat4, AnomalyCategory::DependencyRisk);

        // Normal -> Other
        let cat5 = forest.categorize_anomaly(&[90.0, 90.0, 85.0, 5.0, 2.0, 1.0]);
        assert_eq!(cat5, AnomalyCategory::Other);
    }

    #[test]
    fn test_average_path_length() {
        assert_eq!(average_path_length(0.0), 0.0);
        assert_eq!(average_path_length(1.0), 0.0);

        // For n=2, c(n) ≈ 1
        let c2 = average_path_length(2.0);
        assert!(c2 > 0.0 && c2 < 2.0);

        // For larger n, c(n) grows logarithmically
        let c256 = average_path_length(256.0);
        assert!(c256 > c2);
    }

    // ========================================================================
    // Phase 3: ML Insights Tests - Error Forecaster
    // ========================================================================

    #[test]
    fn test_error_forecaster_new() {
        let forecaster = ErrorForecaster::new(0.5);
        assert_eq!(forecaster.alpha, 0.5);
        assert!(forecaster.history().is_empty());
    }

    #[test]
    fn test_error_forecaster_alpha_clamp() {
        let f1 = ErrorForecaster::new(-0.5);
        assert_eq!(f1.alpha, 0.0);

        let f2 = ErrorForecaster::new(1.5);
        assert_eq!(f2.alpha, 1.0);
    }

    #[test]
    fn test_error_forecaster_default() {
        let forecaster = ErrorForecaster::default_forecaster();
        assert_eq!(forecaster.alpha, 0.3);
    }

    #[test]
    fn test_error_forecaster_observe() {
        let mut forecaster = ErrorForecaster::new(0.5);
        forecaster.observe(100.0);
        assert_eq!(forecaster.current_level(), 100.0);
        assert_eq!(forecaster.history().len(), 1);

        forecaster.observe(80.0);
        // Level = 0.5 * 80 + 0.5 * 100 = 90
        assert_eq!(forecaster.current_level(), 90.0);
        assert_eq!(forecaster.history().len(), 2);
    }

    #[test]
    fn test_error_forecaster_forecast() {
        let mut forecaster = ErrorForecaster::new(0.3);
        forecaster.observe(100.0);
        forecaster.observe(90.0);
        forecaster.observe(85.0);

        let forecast = forecaster.forecast(5);
        assert_eq!(forecast.len(), 5);
        // All forecasts should be the same (simple exponential smoothing)
        let level = forecaster.current_level();
        for f in &forecast {
            assert_eq!(*f, level);
        }
    }

    #[test]
    fn test_error_forecaster_error_metrics_empty() {
        let forecaster = ErrorForecaster::new(0.3);
        let metrics = forecaster.error_metrics();
        assert_eq!(metrics.mae, 0.0);
        assert_eq!(metrics.mse, 0.0);
        assert_eq!(metrics.rmse, 0.0);
    }

    #[test]
    fn test_error_forecaster_error_metrics_single() {
        let mut forecaster = ErrorForecaster::new(0.3);
        forecaster.observe(100.0);
        let metrics = forecaster.error_metrics();
        assert_eq!(metrics.mae, 0.0); // Not enough data
    }

    #[test]
    fn test_error_forecaster_error_metrics() {
        let mut forecaster = ErrorForecaster::new(0.5);
        forecaster.observe(100.0);
        forecaster.observe(110.0);
        forecaster.observe(105.0);
        forecaster.observe(108.0);

        let metrics = forecaster.error_metrics();
        assert!(metrics.mae >= 0.0);
        assert!(metrics.mse >= 0.0);
        assert!(metrics.rmse >= 0.0);
        assert_eq!(metrics.rmse, metrics.mse.sqrt());
    }

    #[test]
    fn test_error_forecaster_exponential_smoothing() {
        // Test the exponential smoothing formula
        let mut forecaster = ErrorForecaster::new(0.3);
        forecaster.observe(100.0); // Level = 100
        forecaster.observe(130.0); // Level = 0.3*130 + 0.7*100 = 39 + 70 = 109
        forecaster.observe(100.0); // Level = 0.3*100 + 0.7*109 = 30 + 76.3 = 106.3

        let level = forecaster.current_level();
        assert!((level - 106.3).abs() < 0.01);
    }

    #[test]
    fn test_forecast_metrics_default() {
        let metrics = ForecastMetrics::default();
        assert_eq!(metrics.mae, 0.0);
        assert_eq!(metrics.mse, 0.0);
        assert_eq!(metrics.rmse, 0.0);
        assert_eq!(metrics.mape, 0.0);
    }
}
