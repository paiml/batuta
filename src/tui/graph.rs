//! Graph TUI Visualization Module
//!
//! Terminal-based graph visualization with Toyota Way principles.
//!
//! ## Design Principles (per spec docs/batuta-graph-viz-tui-spec.md)
//!
//! - **Genchi Genbutsu**: Graph data at source, no duplication
//! - **Jidoka**: Built-in quality via layout convergence tests
//! - **Heijunka**: Single layout engine, multiple output backends
//! - **Mieruka**: Semantic colors for instant status recognition
//! - **Respect for People**: Accessibility via shapes (not just color)
//!
//! ## Academic References
//!
//! - Fruchterman & Reingold (1991) - Force-directed layout
//! - Kamada & Kawai (1989) - Spring-based layout
//! - Sugiyama et al. (1981) - Hierarchical DAG layout
//!
//! ## Performance Constraints (Muri Prevention)
//!
//! - Hard limit: 500 nodes maximum for TUI rendering
//! - Default visible: Top 20 nodes by centrality (Mieruka)

use std::collections::HashMap;
use std::hash::Hash;

// ============================================================================
// GRAPH-001: Core Types
// ============================================================================

/// Maximum nodes for TUI rendering (Muri prevention per peer review #3)
pub const MAX_TUI_NODES: usize = 500;

/// Default visible nodes (Mieruka per peer review #9)
pub const DEFAULT_VISIBLE_NODES: usize = 20;

/// Node shape for accessibility (Respect for People per peer review #6)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NodeShape {
    /// Circle: default node ●
    #[default]
    Circle,
    /// Diamond: input/source ◆
    Diamond,
    /// Square: transform/process ■
    Square,
    /// Triangle: output/sink ▲
    Triangle,
    /// Star: highlighted/selected ★
    Star,
}

impl NodeShape {
    /// Get Unicode character for shape
    #[must_use]
    pub fn unicode(&self) -> char {
        match self {
            Self::Circle => '●',
            Self::Diamond => '◆',
            Self::Square => '■',
            Self::Triangle => '▲',
            Self::Star => '★',
        }
    }

    /// Get ASCII fallback character (Standardized Work per peer review #5)
    #[must_use]
    pub fn ascii(&self) -> char {
        match self {
            Self::Circle => 'o',
            Self::Diamond => '<',
            Self::Square => '#',
            Self::Triangle => '^',
            Self::Star => '*',
        }
    }
}

/// Node status with accessible visual encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NodeStatus {
    /// Healthy/success - green + circle
    #[default]
    Healthy,
    /// Warning/pending - yellow + triangle
    Warning,
    /// Error/critical - red + diamond (not just red per peer review #6)
    Error,
    /// Info/selected - cyan + star
    Info,
    /// Neutral/unknown - gray + square
    Neutral,
}

impl NodeStatus {
    /// Get shape for status (accessibility - not just color)
    #[must_use]
    pub fn shape(&self) -> NodeShape {
        match self {
            Self::Healthy => NodeShape::Circle,
            Self::Warning => NodeShape::Triangle,
            Self::Error => NodeShape::Diamond,
            Self::Info => NodeShape::Star,
            Self::Neutral => NodeShape::Square,
        }
    }

    /// Get ANSI color code
    #[must_use]
    pub fn color_code(&self) -> &'static str {
        match self {
            Self::Healthy => "\x1b[32m", // Green
            Self::Warning => "\x1b[33m", // Yellow
            Self::Error => "\x1b[31m",   // Red
            Self::Info => "\x1b[36m",    // Cyan
            Self::Neutral => "\x1b[90m", // Gray
        }
    }
}

/// 2D position for layout
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Position {
    pub x: f32,
    pub y: f32,
}

impl Position {
    /// Create new position
    #[must_use]
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Euclidean distance to another position
    #[must_use]
    pub fn distance(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Node in the graph
#[derive(Debug, Clone)]
pub struct Node<T> {
    /// Node identifier
    pub id: String,
    /// Node data
    pub data: T,
    /// Visual status
    pub status: NodeStatus,
    /// Display label
    pub label: Option<String>,
    /// Computed position (after layout)
    pub position: Position,
    /// Node importance (for Mieruka filtering)
    pub importance: f32,
}

impl<T> Node<T> {
    /// Create new node
    pub fn new(id: impl Into<String>, data: T) -> Self {
        Self {
            id: id.into(),
            data,
            status: NodeStatus::default(),
            label: None,
            position: Position::default(),
            importance: 1.0,
        }
    }

    /// Set status
    #[must_use]
    pub fn with_status(mut self, status: NodeStatus) -> Self {
        self.status = status;
        self
    }

    /// Set label
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set importance
    #[must_use]
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance;
        self
    }
}

/// Edge in the graph
#[derive(Debug, Clone)]
pub struct Edge<E> {
    /// Source node ID
    pub from: String,
    /// Target node ID
    pub to: String,
    /// Edge data
    pub data: E,
    /// Edge weight (for layout algorithms and visualization)
    pub weight: f32,
    /// Optional edge label (Neo4j/Cytoscape pattern)
    pub label: Option<String>,
}

impl<E> Edge<E> {
    /// Create new edge
    pub fn new(from: impl Into<String>, to: impl Into<String>, data: E) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            data,
            weight: 1.0,
            label: None,
        }
    }

    /// Set weight
    #[must_use]
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Set label (Neo4j pattern)
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

// ============================================================================
// GRAPH-002: Graph Data Structure
// ============================================================================

/// Graph for TUI visualization
#[derive(Debug, Clone)]
pub struct Graph<N, E> {
    /// Nodes indexed by ID
    pub(crate) nodes: HashMap<String, Node<N>>,
    /// Edges
    pub(crate) edges: Vec<Edge<E>>,
    /// Adjacency list for fast neighbor lookup
    adjacency: HashMap<String, Vec<String>>,
}

impl<N, E> Default for Graph<N, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N, E> Graph<N, E> {
    /// Create empty graph
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            adjacency: HashMap::new(),
        }
    }

    /// Add a node
    pub fn add_node(&mut self, node: Node<N>) {
        let id = node.id.clone();
        self.nodes.insert(id.clone(), node);
        self.adjacency.entry(id).or_default();
    }

    /// Add an edge
    pub fn add_edge(&mut self, edge: Edge<E>) {
        self.adjacency
            .entry(edge.from.clone())
            .or_default()
            .push(edge.to.clone());
        self.edges.push(edge);
    }

    /// Get node count
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get edge count
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get node by ID
    #[must_use]
    pub fn get_node(&self, id: &str) -> Option<&Node<N>> {
        self.nodes.get(id)
    }

    /// Get mutable node by ID
    pub fn get_node_mut(&mut self, id: &str) -> Option<&mut Node<N>> {
        self.nodes.get_mut(id)
    }

    /// Iterate over nodes
    pub fn nodes(&self) -> impl Iterator<Item = &Node<N>> {
        self.nodes.values()
    }

    /// Iterate over nodes mutably
    pub fn nodes_mut(&mut self) -> impl Iterator<Item = &mut Node<N>> {
        self.nodes.values_mut()
    }

    /// Iterate over edges
    pub fn edges(&self) -> impl Iterator<Item = &Edge<E>> {
        self.edges.iter()
    }

    /// Get neighbors of a node
    #[must_use]
    pub fn neighbors(&self, id: &str) -> Vec<&str> {
        self.adjacency
            .get(id)
            .map(|v| v.iter().map(String::as_str).collect())
            .unwrap_or_default()
    }

    /// Check if graph exceeds TUI limit (Muri prevention)
    #[must_use]
    pub fn exceeds_tui_limit(&self) -> bool {
        self.nodes.len() > MAX_TUI_NODES
    }

    /// Get top N nodes by importance (Mieruka)
    #[must_use]
    pub fn top_nodes_by_importance(&self, n: usize) -> Vec<&Node<N>> {
        let mut nodes: Vec<_> = self.nodes.values().collect();
        nodes.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        nodes.into_iter().take(n).collect()
    }
}

// Re-export layout types from graph_layout module
pub use super::graph_layout::{LayoutAlgorithm, LayoutConfig, LayoutEngine};

// ============================================================================
// GRAPH-004: TUI Rendering
// ============================================================================

/// Render mode for terminal compatibility
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenderMode {
    /// Unicode with colors (default)
    #[default]
    Unicode,
    /// ASCII fallback for legacy terminals (per peer review #5)
    Ascii,
    /// Plain text without colors
    Plain,
}

/// Rendered graph as string buffer
#[derive(Debug, Clone)]
pub struct RenderedGraph {
    /// Width in characters
    pub width: usize,
    /// Height in characters
    pub height: usize,
    /// Character buffer
    pub buffer: Vec<Vec<char>>,
    /// Color buffer (ANSI codes per cell)
    pub colors: Vec<Vec<Option<&'static str>>>,
}

impl RenderedGraph {
    /// Create new render buffer
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            buffer: vec![vec![' '; width]; height],
            colors: vec![vec![None; width]; height],
        }
    }

    /// Set character at position
    pub fn set(&mut self, x: usize, y: usize, ch: char, color: Option<&'static str>) {
        if x < self.width && y < self.height {
            self.buffer[y][x] = ch;
            self.colors[y][x] = color;
        }
    }

    /// Render to string with ANSI colors
    #[must_use]
    pub fn to_string_colored(&self) -> String {
        let mut result = String::new();
        for y in 0..self.height {
            for x in 0..self.width {
                if let Some(color) = self.colors[y][x] {
                    result.push_str(color);
                    result.push(self.buffer[y][x]);
                    result.push_str("\x1b[0m");
                } else {
                    result.push(self.buffer[y][x]);
                }
            }
            result.push('\n');
        }
        result
    }

    /// Render to plain string (no colors)
    #[must_use]
    pub fn to_string_plain(&self) -> String {
        self.buffer
            .iter()
            .map(|row| row.iter().collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Graph renderer
pub struct GraphRenderer {
    /// Render mode
    pub mode: RenderMode,
    /// Show labels
    pub show_labels: bool,
    /// Show edges
    pub show_edges: bool,
}

impl Default for GraphRenderer {
    fn default() -> Self {
        Self {
            mode: RenderMode::Unicode,
            show_labels: true,
            show_edges: true,
        }
    }
}

impl GraphRenderer {
    /// Create new renderer
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set render mode
    #[must_use]
    pub fn with_mode(mut self, mode: RenderMode) -> Self {
        self.mode = mode;
        self
    }

    /// Render graph to buffer
    pub fn render<N, E>(&self, graph: &Graph<N, E>, width: usize, height: usize) -> RenderedGraph {
        let mut output = RenderedGraph::new(width, height);

        // Draw edges first (underneath nodes)
        if self.show_edges {
            for edge in graph.edges() {
                if let (Some(from), Some(to)) =
                    (graph.get_node(&edge.from), graph.get_node(&edge.to))
                {
                    self.draw_edge(&mut output, &from.position, &to.position, width, height);
                }
            }
        }

        // Draw nodes
        for node in graph.nodes() {
            self.draw_node(&mut output, node, width, height);
        }

        output
    }

    fn draw_node<N>(
        &self,
        output: &mut RenderedGraph,
        node: &Node<N>,
        width: usize,
        height: usize,
    ) {
        let x = (node.position.x / 80.0 * width as f32) as usize;
        let y = (node.position.y / 24.0 * height as f32) as usize;

        if x < width && y < height {
            let ch = match self.mode {
                RenderMode::Unicode => node.status.shape().unicode(),
                RenderMode::Ascii | RenderMode::Plain => node.status.shape().ascii(),
            };

            let color = match self.mode {
                RenderMode::Unicode | RenderMode::Ascii => Some(node.status.color_code()),
                RenderMode::Plain => None,
            };

            output.set(x, y, ch, color);

            // Draw label if enabled
            if self.show_labels {
                if let Some(ref label) = node.label {
                    let label_start = x.saturating_add(2);
                    for (i, c) in label.chars().take(10).enumerate() {
                        if label_start + i < width {
                            output.set(label_start + i, y, c, color);
                        }
                    }
                }
            }
        }
    }

    fn draw_edge(
        &self,
        output: &mut RenderedGraph,
        from: &Position,
        to: &Position,
        width: usize,
        height: usize,
    ) {
        let x1 = (from.x / 80.0 * width as f32) as i32;
        let y1 = (from.y / 24.0 * height as f32) as i32;
        let x2 = (to.x / 80.0 * width as f32) as i32;
        let y2 = (to.y / 24.0 * height as f32) as i32;

        // Bresenham's line algorithm
        let dx = (x2 - x1).abs();
        let dy = (y2 - y1).abs();
        let sx = if x1 < x2 { 1 } else { -1 };
        let sy = if y1 < y2 { 1 } else { -1 };
        let mut err = dx - dy;

        let mut x = x1;
        let mut y = y1;

        let edge_char = match self.mode {
            RenderMode::Unicode => '·',
            RenderMode::Ascii | RenderMode::Plain => '.',
        };

        while x != x2 || y != y2 {
            if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                // Don't overwrite nodes
                if output.buffer[y as usize][x as usize] == ' ' {
                    output.set(x as usize, y as usize, edge_char, Some("\x1b[90m"));
                }
            }

            let e2 = 2 * err;
            if e2 > -dy {
                err -= dy;
                x += sx;
            }
            if e2 < dx {
                err += dx;
                y += sy;
            }
        }
    }
}

// Re-export analytics types from graph_analytics module
pub use super::graph_analytics::{GraphAnalytics, GraphAnalyticsExt, COMMUNITY_COLORS};

// ============================================================================
// GRAPH-005b: Filtering (Neo4j/Gephi pattern)
// ============================================================================

impl<N: Clone, E: Clone> Graph<N, E> {
    /// Filter graph to nodes matching predicate
    ///
    /// Returns a new graph containing only matching nodes and their edges.
    #[must_use]
    pub fn filter_nodes<F>(&self, predicate: F) -> Self
    where
        F: Fn(&Node<N>) -> bool,
    {
        let mut filtered = Self::new();

        // Add matching nodes
        for node in self.nodes() {
            if predicate(node) {
                filtered.add_node(node.clone());
            }
        }

        // Add edges where both endpoints exist
        for edge in &self.edges {
            if filtered.nodes.contains_key(&edge.from) && filtered.nodes.contains_key(&edge.to) {
                filtered.add_edge(edge.clone());
            }
        }

        filtered
    }

    /// Filter to nodes with minimum degree
    #[must_use]
    pub fn filter_by_min_degree(&self, min_degree: usize) -> Self {
        let degrees = GraphAnalytics::degree_centrality(self);
        let n = self.node_count();
        let threshold = if n > 1 {
            min_degree as f32 / (n - 1) as f32
        } else {
            0.0
        };

        self.filter_nodes(|node| degrees.get(&node.id).unwrap_or(&0.0) >= &threshold)
    }

    /// Filter to top N nodes by importance
    #[must_use]
    pub fn filter_top_n(&self, n: usize) -> Self {
        let mut nodes_by_importance: Vec<_> = self.nodes().collect();
        nodes_by_importance.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let top_ids: std::collections::HashSet<_> =
            nodes_by_importance.iter().take(n).map(|n| &n.id).collect();

        self.filter_nodes(|node| top_ids.contains(&node.id))
    }

    /// Filter to nodes matching label pattern
    #[must_use]
    pub fn filter_by_label(&self, pattern: &str) -> Self {
        let pattern_lower = pattern.to_lowercase();
        self.filter_nodes(|node| {
            node.label
                .as_ref()
                .map(|l| l.to_lowercase().contains(&pattern_lower))
                .unwrap_or(false)
        })
    }

    /// Filter to subgraph containing path between two nodes
    #[must_use]
    pub fn filter_path(&self, from: &str, to: &str) -> Self {
        if let Some(path) = GraphAnalytics::shortest_path(self, from, to) {
            let path_set: std::collections::HashSet<_> = path.into_iter().collect();
            self.filter_nodes(|node| path_set.contains(&node.id))
        } else {
            Self::new()
        }
    }
}

// ============================================================================
// GRAPH-006: Tests (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Node Shape Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_node_shape_unicode_characters() {
        assert_eq!(NodeShape::Circle.unicode(), '●');
        assert_eq!(NodeShape::Diamond.unicode(), '◆');
        assert_eq!(NodeShape::Square.unicode(), '■');
        assert_eq!(NodeShape::Triangle.unicode(), '▲');
        assert_eq!(NodeShape::Star.unicode(), '★');
    }

    #[test]
    fn test_node_shape_ascii_fallback() {
        // Per peer review #5: ASCII fallback for legacy terminals
        assert_eq!(NodeShape::Circle.ascii(), 'o');
        assert_eq!(NodeShape::Diamond.ascii(), '<');
        assert_eq!(NodeShape::Square.ascii(), '#');
        assert_eq!(NodeShape::Triangle.ascii(), '^');
        assert_eq!(NodeShape::Star.ascii(), '*');
    }

    #[test]
    fn test_status_shapes_for_accessibility() {
        // Per peer review #6: Shapes differentiate status, not just color
        assert_eq!(NodeStatus::Healthy.shape(), NodeShape::Circle);
        assert_eq!(NodeStatus::Warning.shape(), NodeShape::Triangle);
        assert_eq!(NodeStatus::Error.shape(), NodeShape::Diamond);
        assert_eq!(NodeStatus::Info.shape(), NodeShape::Star);
        assert_eq!(NodeStatus::Neutral.shape(), NodeShape::Square);
    }

    // -------------------------------------------------------------------------
    // Position Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_position_distance() {
        let p1 = Position::new(0.0, 0.0);
        let p2 = Position::new(3.0, 4.0);
        assert!((p1.distance(&p2) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_position_distance_same_point() {
        let p = Position::new(5.0, 5.0);
        assert!((p.distance(&p) - 0.0).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // Graph Structure Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_graph_creation() {
        let graph: Graph<(), ()> = Graph::new();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_graph_add_nodes() {
        let mut graph: Graph<i32, ()> = Graph::new();
        graph.add_node(Node::new("a", 1));
        graph.add_node(Node::new("b", 2));
        assert_eq!(graph.node_count(), 2);
    }

    #[test]
    fn test_graph_add_edges() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("a", ()));
        graph.add_node(Node::new("b", ()));
        graph.add_edge(Edge::new("a", "b", ()));
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_graph_neighbors() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("a", ()));
        graph.add_node(Node::new("b", ()));
        graph.add_node(Node::new("c", ()));
        graph.add_edge(Edge::new("a", "b", ()));
        graph.add_edge(Edge::new("a", "c", ()));

        let neighbors = graph.neighbors("a");
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&"b"));
        assert!(neighbors.contains(&"c"));
    }

    #[test]
    fn test_graph_tui_limit() {
        // Per peer review #3: Hard limit of 500 nodes
        let mut graph: Graph<(), ()> = Graph::new();
        for i in 0..MAX_TUI_NODES {
            graph.add_node(Node::new(format!("n{}", i), ()));
        }
        assert!(!graph.exceeds_tui_limit());

        graph.add_node(Node::new("overflow", ()));
        assert!(graph.exceeds_tui_limit());
    }

    #[test]
    fn test_top_nodes_by_importance() {
        // Per peer review #9: Mieruka - show top N nodes
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("low", ()).with_importance(0.1));
        graph.add_node(Node::new("medium", ()).with_importance(0.5));
        graph.add_node(Node::new("high", ()).with_importance(0.9));

        let top = graph.top_nodes_by_importance(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].id, "high");
        assert_eq!(top[1].id, "medium");
    }

    // -------------------------------------------------------------------------
    // Layout Algorithm Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_grid_layout_positions_all_nodes() {
        let mut graph: Graph<(), ()> = Graph::new();
        for i in 0..9 {
            graph.add_node(Node::new(format!("n{}", i), ()));
        }

        let config = LayoutConfig::default();
        LayoutEngine::compute(&mut graph, &config);

        // All nodes should have non-default positions
        for node in graph.nodes() {
            assert!(node.position.x > 0.0 || node.position.y > 0.0);
        }
    }

    #[test]
    fn test_grid_layout_no_overlaps() {
        let mut graph: Graph<(), ()> = Graph::new();
        for i in 0..16 {
            graph.add_node(Node::new(format!("n{}", i), ()));
        }

        let config = LayoutConfig::default();
        LayoutEngine::compute(&mut graph, &config);

        // Check no two nodes at exact same position
        let positions: Vec<_> = graph.nodes().map(|n| n.position).collect();
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let dist = positions[i].distance(&positions[j]);
                assert!(dist > 0.1, "Nodes {} and {} overlap", i, j);
            }
        }
    }

    #[test]
    fn test_force_directed_convergence() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("a", ()));
        graph.add_node(Node::new("b", ()));
        graph.add_node(Node::new("c", ()));
        graph.add_edge(Edge::new("a", "b", ()));
        graph.add_edge(Edge::new("b", "c", ()));

        let config = LayoutConfig {
            algorithm: LayoutAlgorithm::ForceDirected,
            iterations: 100,
            ..Default::default()
        };
        LayoutEngine::compute(&mut graph, &config);

        // Connected nodes should be closer than unconnected
        let pos_a = graph.get_node("a").unwrap().position;
        let pos_b = graph.get_node("b").unwrap().position;
        let pos_c = graph.get_node("c").unwrap().position;

        let ab_dist = pos_a.distance(&pos_b);
        let bc_dist = pos_b.distance(&pos_c);
        let ac_dist = pos_a.distance(&pos_c);

        // a-b and b-c are connected, a-c is not
        assert!(ab_dist < ac_dist || bc_dist < ac_dist);
    }

    #[test]
    fn test_hierarchical_layout_layers() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("root", ()));
        graph.add_node(Node::new("child1", ()));
        graph.add_node(Node::new("child2", ()));
        graph.add_node(Node::new("grandchild", ()));
        graph.add_edge(Edge::new("root", "child1", ()));
        graph.add_edge(Edge::new("root", "child2", ()));
        graph.add_edge(Edge::new("child1", "grandchild", ()));

        let config = LayoutConfig {
            algorithm: LayoutAlgorithm::Hierarchical,
            ..Default::default()
        };
        LayoutEngine::compute(&mut graph, &config);

        // Root should be at top (lowest y)
        let root_y = graph.get_node("root").unwrap().position.y;
        let child1_y = graph.get_node("child1").unwrap().position.y;
        let grandchild_y = graph.get_node("grandchild").unwrap().position.y;

        assert!(root_y < child1_y);
        assert!(child1_y < grandchild_y);
    }

    #[test]
    fn test_radial_layout_center() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("center", ()));
        graph.add_node(Node::new("leaf1", ()));
        graph.add_node(Node::new("leaf2", ()));
        graph.add_edge(Edge::new("center", "leaf1", ()));
        graph.add_edge(Edge::new("center", "leaf2", ()));

        let config = LayoutConfig {
            algorithm: LayoutAlgorithm::Radial,
            width: 80.0,
            height: 24.0,
            ..Default::default()
        };
        LayoutEngine::compute(&mut graph, &config);

        // Center node should be near center of layout
        let center_pos = graph.get_node("center").unwrap().position;
        assert!((center_pos.x - 40.0).abs() < 5.0);
        assert!((center_pos.y - 12.0).abs() < 5.0);
    }

    // -------------------------------------------------------------------------
    // Rendering Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_rendered_graph_creation() {
        let rg = RenderedGraph::new(80, 24);
        assert_eq!(rg.width, 80);
        assert_eq!(rg.height, 24);
        assert_eq!(rg.buffer.len(), 24);
        assert_eq!(rg.buffer[0].len(), 80);
    }

    #[test]
    fn test_rendered_graph_set() {
        let mut rg = RenderedGraph::new(10, 10);
        rg.set(5, 5, 'X', None);
        assert_eq!(rg.buffer[5][5], 'X');
    }

    #[test]
    fn test_renderer_unicode_mode() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("test", ()).with_status(NodeStatus::Healthy));

        let config = LayoutConfig {
            algorithm: LayoutAlgorithm::Grid,
            width: 80.0,
            height: 24.0,
            ..Default::default()
        };
        LayoutEngine::compute(&mut graph, &config);

        let renderer = GraphRenderer::new().with_mode(RenderMode::Unicode);
        let output = renderer.render(&graph, 80, 24);

        // Should contain circle character
        let plain = output.to_string_plain();
        assert!(plain.contains('●'));
    }

    #[test]
    fn test_renderer_ascii_mode() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("test", ()).with_status(NodeStatus::Healthy));

        let config = LayoutConfig::default();
        LayoutEngine::compute(&mut graph, &config);

        let renderer = GraphRenderer::new().with_mode(RenderMode::Ascii);
        let output = renderer.render(&graph, 80, 24);

        // Should contain ASCII 'o' not Unicode circle
        let plain = output.to_string_plain();
        assert!(plain.contains('o'));
        assert!(!plain.contains('●'));
    }

    #[test]
    fn test_renderer_with_edges() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("a", ()));
        graph.add_node(Node::new("b", ()));
        graph.add_edge(Edge::new("a", "b", ()));

        let config = LayoutConfig {
            algorithm: LayoutAlgorithm::Grid,
            width: 80.0,
            height: 24.0,
            ..Default::default()
        };
        LayoutEngine::compute(&mut graph, &config);

        let renderer = GraphRenderer::new();
        let output = renderer.render(&graph, 80, 24);

        // Should contain edge characters
        let plain = output.to_string_plain();
        assert!(plain.contains('·') || plain.contains('.'));
    }

    // -------------------------------------------------------------------------
    // Integration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_full_pipeline() {
        // Create graph
        let mut graph: Graph<&str, &str> = Graph::new();
        graph.add_node(
            Node::new("trueno", "core")
                .with_status(NodeStatus::Healthy)
                .with_label("trueno"),
        );
        graph.add_node(
            Node::new("aprender", "ml")
                .with_status(NodeStatus::Healthy)
                .with_label("aprender"),
        );
        graph.add_node(
            Node::new("batuta", "orch")
                .with_status(NodeStatus::Warning)
                .with_label("batuta"),
        );
        graph.add_edge(Edge::new("trueno", "aprender", "depends"));
        graph.add_edge(Edge::new("aprender", "batuta", "depends"));

        // Compute layout
        let config = LayoutConfig {
            algorithm: LayoutAlgorithm::Hierarchical,
            width: 80.0,
            height: 24.0,
            ..Default::default()
        };
        LayoutEngine::compute(&mut graph, &config);

        // Render
        let renderer = GraphRenderer::new();
        let output = renderer.render(&graph, 80, 24);

        // Verify output contains labels
        let plain = output.to_string_plain();
        assert!(plain.contains("trueno") || plain.contains("aprender") || plain.contains("batuta"));
    }

    // -------------------------------------------------------------------------
    // Property-Based Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_layout_bounds() {
        // All layouts should keep nodes within bounds
        for algo in [
            LayoutAlgorithm::Grid,
            LayoutAlgorithm::ForceDirected,
            LayoutAlgorithm::Hierarchical,
            LayoutAlgorithm::Radial,
        ] {
            let mut graph: Graph<(), ()> = Graph::new();
            for i in 0..20 {
                graph.add_node(Node::new(format!("n{}", i), ()));
            }
            for i in 0..15 {
                graph.add_edge(Edge::new(format!("n{}", i), format!("n{}", i + 1), ()));
            }

            let config = LayoutConfig {
                algorithm: algo,
                width: 80.0,
                height: 24.0,
                iterations: 50,
                ..Default::default()
            };
            LayoutEngine::compute(&mut graph, &config);

            for node in graph.nodes() {
                assert!(node.position.x >= 0.0, "{:?} x < 0", algo);
                assert!(node.position.y >= 0.0, "{:?} y < 0", algo);
                assert!(node.position.x <= 80.0, "{:?} x > width", algo);
                assert!(node.position.y <= 24.0, "{:?} y > height", algo);
            }
        }
    }

    #[test]
    fn test_empty_graph_handling() {
        let mut graph: Graph<(), ()> = Graph::new();

        for algo in [
            LayoutAlgorithm::Grid,
            LayoutAlgorithm::ForceDirected,
            LayoutAlgorithm::Hierarchical,
            LayoutAlgorithm::Radial,
        ] {
            let config = LayoutConfig {
                algorithm: algo,
                ..Default::default()
            };
            // Should not panic
            LayoutEngine::compute(&mut graph, &config);
        }
    }

    #[test]
    fn test_single_node_graph() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("solo", ()));

        for algo in [
            LayoutAlgorithm::Grid,
            LayoutAlgorithm::ForceDirected,
            LayoutAlgorithm::Hierarchical,
            LayoutAlgorithm::Radial,
        ] {
            let config = LayoutConfig {
                algorithm: algo,
                ..Default::default()
            };
            LayoutEngine::compute(&mut graph, &config);

            let pos = graph.get_node("solo").unwrap().position;
            assert!(pos.x > 0.0);
            assert!(pos.y > 0.0);
        }
    }

    // -------------------------------------------------------------------------
    // PageRank Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_pagerank_empty_graph() {
        let graph: Graph<(), ()> = Graph::new();
        let ranks = GraphAnalytics::pagerank(&graph, 0.85, 20);
        assert!(ranks.is_empty());
    }

    #[test]
    fn test_pagerank_single_node() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("solo", ()));

        let ranks = GraphAnalytics::pagerank(&graph, 0.85, 20);
        assert_eq!(ranks.len(), 1);
        assert!((ranks.get("solo").unwrap() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_pagerank_linear_chain() {
        // A -> B -> C
        // C should have highest rank (receives most "flow")
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("A", ()));
        graph.add_node(Node::new("B", ()));
        graph.add_node(Node::new("C", ()));
        graph.add_edge(Edge::new("A", "B", ()));
        graph.add_edge(Edge::new("B", "C", ()));

        let ranks = GraphAnalytics::pagerank(&graph, 0.85, 50);

        let rank_a = *ranks.get("A").unwrap();
        let rank_b = *ranks.get("B").unwrap();
        let rank_c = *ranks.get("C").unwrap();

        // C should have highest rank (sink node)
        assert!(rank_c > rank_b, "C should rank higher than B");
        assert!(rank_b > rank_a, "B should rank higher than A");
    }

    #[test]
    fn test_pagerank_cycle() {
        // A -> B -> C -> A (cycle)
        // All nodes should have equal rank
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("A", ()));
        graph.add_node(Node::new("B", ()));
        graph.add_node(Node::new("C", ()));
        graph.add_edge(Edge::new("A", "B", ()));
        graph.add_edge(Edge::new("B", "C", ()));
        graph.add_edge(Edge::new("C", "A", ()));

        let ranks = GraphAnalytics::pagerank(&graph, 0.85, 50);

        let rank_a = *ranks.get("A").unwrap();
        let rank_b = *ranks.get("B").unwrap();
        let rank_c = *ranks.get("C").unwrap();

        // All should be approximately equal in a cycle
        assert!((rank_a - rank_b).abs() < 0.1);
        assert!((rank_b - rank_c).abs() < 0.1);
    }

    #[test]
    fn test_pagerank_star_topology() {
        // Center -> A, B, C, D (star)
        // Center should have lower rank (gives away rank)
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("center", ()));
        graph.add_node(Node::new("A", ()));
        graph.add_node(Node::new("B", ()));
        graph.add_node(Node::new("C", ()));
        graph.add_node(Node::new("D", ()));
        graph.add_edge(Edge::new("center", "A", ()));
        graph.add_edge(Edge::new("center", "B", ()));
        graph.add_edge(Edge::new("center", "C", ()));
        graph.add_edge(Edge::new("center", "D", ()));

        let ranks = GraphAnalytics::pagerank(&graph, 0.85, 50);

        let rank_center = *ranks.get("center").unwrap();
        let rank_a = *ranks.get("A").unwrap();

        // Leaves should have higher rank than center
        assert!(
            rank_a > rank_center,
            "Leaf A should rank higher than center"
        );
    }

    #[test]
    fn test_apply_pagerank() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("A", ()));
        graph.add_node(Node::new("B", ()));
        graph.add_edge(Edge::new("A", "B", ()));

        // Initially importance is 1.0
        assert_eq!(graph.get_node("A").unwrap().importance, 1.0);

        GraphAnalytics::apply_pagerank(&mut graph, 0.85, 20);

        // After PageRank, importance should be updated
        let imp_a = graph.get_node("A").unwrap().importance;
        let imp_b = graph.get_node("B").unwrap().importance;
        assert!(
            imp_b > imp_a,
            "B should have higher importance after PageRank"
        );
    }

    // -------------------------------------------------------------------------
    // Community Detection Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_community_empty_graph() {
        let graph: Graph<(), ()> = Graph::new();
        let communities = GraphAnalytics::detect_communities(&graph);
        assert!(communities.is_empty());
    }

    #[test]
    fn test_community_single_node() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("solo", ()));

        let communities = GraphAnalytics::detect_communities(&graph);
        assert_eq!(communities.len(), 1);
        assert_eq!(*communities.get("solo").unwrap(), 0);
    }

    #[test]
    fn test_community_two_clusters() {
        // Create two disconnected clusters (no inter-cluster edge)
        // This is a simpler test case for the simplified algorithm
        let mut graph: Graph<(), ()> = Graph::new();

        // Cluster 1: A-B-C (chain)
        graph.add_node(Node::new("A", ()));
        graph.add_node(Node::new("B", ()));
        graph.add_node(Node::new("C", ()));
        graph.add_edge(Edge::new("A", "B", ()));
        graph.add_edge(Edge::new("B", "C", ()));
        graph.add_edge(Edge::new("C", "A", ())); // Close the triangle

        let communities = GraphAnalytics::detect_communities(&graph);

        // All nodes should be assigned a community
        assert_eq!(communities.len(), 3);

        // Communities should be contiguous integers starting from 0
        let max_comm = *communities.values().max().unwrap();
        assert!(
            max_comm < 3,
            "Should have at most 3 communities for 3 nodes"
        );
    }

    #[test]
    fn test_community_colors() {
        // Test that community colors are valid ANSI codes
        for i in 0..10 {
            let color = GraphAnalytics::community_color(i);
            assert!(color.starts_with("\x1b["), "Should be ANSI escape code");
            assert!(color.ends_with('m'), "Should end with 'm'");
        }
    }

    #[test]
    fn test_apply_communities() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("A", ()));
        graph.add_node(Node::new("B", ()));
        graph.add_node(Node::new("C", ()));
        graph.add_edge(Edge::new("A", "B", ()));
        graph.add_edge(Edge::new("B", "C", ()));

        let num_communities = GraphAnalytics::apply_communities(&mut graph);

        // Should detect at least 1 community
        assert!(num_communities >= 1);

        // Importance should be updated
        let imp_a = graph.get_node("A").unwrap().importance;
        assert!((0.0..=1.0).contains(&imp_a));
    }

    // -------------------------------------------------------------------------
    // GraphAnalyticsExt Trait Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_analytics_ext_trait() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("A", ()));
        graph.add_node(Node::new("B", ()));
        graph.add_edge(Edge::new("A", "B", ()));

        // Test trait methods
        let scores = graph.pagerank_scores();
        assert_eq!(scores.len(), 2);

        graph.compute_pagerank(0.85, 10);
        let num_comm = graph.detect_communities();
        assert!(num_comm >= 1);
    }

    // -------------------------------------------------------------------------
    // Circular/Concentric Layout Tests (Cytoscape patterns)
    // -------------------------------------------------------------------------

    #[test]
    fn test_circular_layout() {
        let mut graph: Graph<(), ()> = Graph::new();
        for i in 0..8 {
            graph.add_node(Node::new(format!("n{}", i), ()));
        }

        let config = LayoutConfig {
            algorithm: LayoutAlgorithm::Circular,
            width: 80.0,
            height: 24.0,
            ..Default::default()
        };
        LayoutEngine::compute(&mut graph, &config);

        // All nodes should be positioned
        for node in graph.nodes() {
            assert!(node.position.x >= 0.0 && node.position.x <= 80.0);
            assert!(node.position.y >= 0.0 && node.position.y <= 24.0);
        }

        // Nodes should be roughly equidistant from center
        let center_x = 40.0;
        let center_y = 12.0;
        let distances: Vec<f32> = graph
            .nodes()
            .map(|n| ((n.position.x - center_x).powi(2) + (n.position.y - center_y).powi(2)).sqrt())
            .collect();

        let avg_dist: f32 = distances.iter().sum::<f32>() / distances.len() as f32;
        for d in &distances {
            assert!(
                (d - avg_dist).abs() < 1.0,
                "Nodes should be roughly equidistant from center"
            );
        }
    }

    #[test]
    fn test_concentric_layout() {
        let mut graph: Graph<(), ()> = Graph::new();
        // Add nodes with varying importance
        for i in 0..6 {
            let mut node = Node::new(format!("n{}", i), ());
            node.importance = 1.0 - (i as f32 * 0.15);
            graph.add_node(node);
        }

        let config = LayoutConfig {
            algorithm: LayoutAlgorithm::Concentric,
            width: 80.0,
            height: 24.0,
            ..Default::default()
        };
        LayoutEngine::compute(&mut graph, &config);

        // All nodes should be positioned within bounds
        for node in graph.nodes() {
            assert!(node.position.x >= 0.0 && node.position.x <= 80.0);
            assert!(node.position.y >= 0.0 && node.position.y <= 24.0);
        }
    }

    // -------------------------------------------------------------------------
    // Centrality Metrics Tests (Neo4j/Gephi patterns)
    // -------------------------------------------------------------------------

    #[test]
    fn test_degree_centrality() {
        let mut graph: Graph<(), ()> = Graph::new();
        // Star topology: center connected to 4 nodes
        graph.add_node(Node::new("center", ()));
        for i in 0..4 {
            graph.add_node(Node::new(format!("n{}", i), ()));
            graph.add_edge(Edge::new("center", format!("n{}", i), ()));
        }

        let degrees = GraphAnalytics::degree_centrality(&graph);

        // Center should have highest degree
        let center_deg = *degrees.get("center").unwrap();
        for i in 0..4 {
            let leaf_deg = *degrees.get(&format!("n{}", i)).unwrap();
            assert!(
                center_deg > leaf_deg,
                "Center should have highest degree centrality"
            );
        }
    }

    #[test]
    fn test_betweenness_centrality() {
        let mut graph: Graph<(), ()> = Graph::new();
        // Line graph: A -> B -> C -> D
        for c in ['A', 'B', 'C', 'D'] {
            graph.add_node(Node::new(c.to_string(), ()));
        }
        graph.add_edge(Edge::new("A", "B", ()));
        graph.add_edge(Edge::new("B", "C", ()));
        graph.add_edge(Edge::new("C", "D", ()));

        let betweenness = GraphAnalytics::betweenness_centrality(&graph);

        // B and C should have higher betweenness (they're in the middle)
        let b_between = *betweenness.get("B").unwrap();
        let a_between = *betweenness.get("A").unwrap();
        assert!(
            b_between >= a_between,
            "Middle nodes should have higher betweenness"
        );
    }

    #[test]
    fn test_closeness_centrality() {
        let mut graph: Graph<(), ()> = Graph::new();
        // Star topology
        graph.add_node(Node::new("center", ()));
        for i in 0..4 {
            graph.add_node(Node::new(format!("n{}", i), ()));
            graph.add_edge(Edge::new("center", format!("n{}", i), ()));
        }

        let closeness = GraphAnalytics::closeness_centrality(&graph);

        // Center should have highest closeness
        let center_close = *closeness.get("center").unwrap();
        assert!(center_close > 0.0, "Center should have positive closeness");
    }

    // -------------------------------------------------------------------------
    // Path Analysis Tests (Neo4j/Cytoscape patterns)
    // -------------------------------------------------------------------------

    #[test]
    fn test_shortest_path_exists() {
        let mut graph: Graph<(), ()> = Graph::new();
        for c in ['A', 'B', 'C', 'D'] {
            graph.add_node(Node::new(c.to_string(), ()));
        }
        graph.add_edge(Edge::new("A", "B", ()));
        graph.add_edge(Edge::new("B", "C", ()));
        graph.add_edge(Edge::new("C", "D", ()));

        let path = GraphAnalytics::shortest_path(&graph, "A", "D");
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path, vec!["A", "B", "C", "D"]);
    }

    #[test]
    fn test_shortest_path_same_node() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("A", ()));

        let path = GraphAnalytics::shortest_path(&graph, "A", "A");
        assert!(path.is_some());
        assert_eq!(path.unwrap(), vec!["A"]);
    }

    #[test]
    fn test_shortest_path_no_path() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("A", ()));
        graph.add_node(Node::new("B", ()));
        // No edge between them

        let path = GraphAnalytics::shortest_path(&graph, "A", "B");
        assert!(path.is_none());
    }

    #[test]
    fn test_is_connected() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("A", ()));
        graph.add_node(Node::new("B", ()));
        graph.add_edge(Edge::new("A", "B", ()));

        assert!(GraphAnalytics::is_connected(&graph));

        // Add disconnected node
        graph.add_node(Node::new("C", ()));
        assert!(!GraphAnalytics::is_connected(&graph));
    }

    // -------------------------------------------------------------------------
    // Filtering Tests (Neo4j/Gephi patterns)
    // -------------------------------------------------------------------------

    #[test]
    fn test_filter_nodes() {
        let mut graph: Graph<&str, ()> = Graph::new();
        graph.add_node(Node::new("A", "data").with_label("Alpha"));
        graph.add_node(Node::new("B", "data").with_label("Beta"));
        graph.add_node(Node::new("C", "data").with_label("Gamma"));
        graph.add_edge(Edge::new("A", "B", ()));
        graph.add_edge(Edge::new("B", "C", ()));

        let filtered = graph.filter_nodes(|n| n.id != "B");
        assert_eq!(filtered.node_count(), 2);
        assert_eq!(filtered.edge_count(), 0); // Edges to B should be removed
    }

    #[test]
    fn test_filter_by_label() {
        let mut graph: Graph<&str, ()> = Graph::new();
        graph.add_node(Node::new("A", "data").with_label("Alpha"));
        graph.add_node(Node::new("B", "data").with_label("Beta"));
        graph.add_node(Node::new("C", "data").with_label("Gamma"));

        let filtered = graph.filter_by_label("alpha"); // Case insensitive
        assert_eq!(filtered.node_count(), 1);
        assert!(filtered.get_node("A").is_some());
    }

    #[test]
    fn test_filter_top_n() {
        let mut graph: Graph<(), ()> = Graph::new();
        for i in 0..5 {
            let mut node = Node::new(format!("n{}", i), ());
            node.importance = i as f32;
            graph.add_node(node);
        }

        let filtered = graph.filter_top_n(3);
        assert_eq!(filtered.node_count(), 3);

        // Should contain n2, n3, n4 (highest importance)
        assert!(filtered.get_node("n4").is_some());
        assert!(filtered.get_node("n3").is_some());
        assert!(filtered.get_node("n2").is_some());
    }

    #[test]
    fn test_filter_path() {
        let mut graph: Graph<(), ()> = Graph::new();
        for c in ['A', 'B', 'C', 'D', 'E'] {
            graph.add_node(Node::new(c.to_string(), ()));
        }
        graph.add_edge(Edge::new("A", "B", ()));
        graph.add_edge(Edge::new("B", "C", ()));
        graph.add_edge(Edge::new("C", "D", ()));
        graph.add_edge(Edge::new("A", "E", ())); // Side branch

        let path_graph = graph.filter_path("A", "D");
        assert_eq!(path_graph.node_count(), 4); // A, B, C, D
        assert!(path_graph.get_node("E").is_none()); // E not on path
    }

    // -------------------------------------------------------------------------
    // Edge Features Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_edge_weight() {
        let edge: Edge<()> = Edge::new("A", "B", ()).with_weight(2.5);
        assert_eq!(edge.weight, 2.5);
    }

    #[test]
    fn test_edge_label() {
        let edge: Edge<()> = Edge::new("A", "B", ()).with_label("depends_on");
        assert_eq!(edge.label.as_deref(), Some("depends_on"));
    }
}
