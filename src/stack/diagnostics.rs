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

// Re-export ML components from diagnostics_ml module
pub use super::diagnostics_ml::{ErrorForecaster, ForecastMetrics, IsolationForest};

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

}
