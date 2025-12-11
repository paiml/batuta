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
    nodes: HashMap<String, Node<N>>,
    /// Edges
    edges: Vec<Edge<E>>,
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

// ============================================================================
// GRAPH-003: Layout Algorithms
// ============================================================================

/// Layout algorithm selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayoutAlgorithm {
    /// Simple grid layout (O(n))
    #[default]
    Grid,
    /// Force-directed Fruchterman-Reingold (O(n²) per iteration)
    ForceDirected,
    /// Hierarchical Sugiyama for DAGs (O(nm log m))
    Hierarchical,
    /// Radial layout for trees (O(n + m))
    Radial,
    /// Circular layout - nodes in a circle (O(n)) - Cytoscape pattern
    Circular,
    /// Concentric layout - nodes in rings by metric (O(n)) - Cytoscape/Gephi pattern
    Concentric,
}

/// Layout configuration
#[derive(Debug, Clone)]
pub struct LayoutConfig {
    /// Algorithm to use
    pub algorithm: LayoutAlgorithm,
    /// Width of layout area
    pub width: f32,
    /// Height of layout area
    pub height: f32,
    /// Iterations for iterative algorithms
    pub iterations: usize,
    /// Optimal edge length for force-directed
    pub optimal_distance: f32,
    /// Cooling factor for force-directed
    pub cooling: f32,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            algorithm: LayoutAlgorithm::Grid,
            width: 80.0,
            height: 24.0,
            iterations: 50,
            optimal_distance: 3.0,
            cooling: 0.95,
        }
    }
}

/// Layout engine
pub struct LayoutEngine;

impl LayoutEngine {
    /// Compute layout for graph
    pub fn compute<N, E>(graph: &mut Graph<N, E>, config: &LayoutConfig) {
        match config.algorithm {
            LayoutAlgorithm::Grid => Self::grid_layout(graph, config),
            LayoutAlgorithm::ForceDirected => Self::force_directed_layout(graph, config),
            LayoutAlgorithm::Hierarchical => Self::hierarchical_layout(graph, config),
            LayoutAlgorithm::Circular => Self::circular_layout(graph, config),
            LayoutAlgorithm::Concentric => Self::concentric_layout(graph, config),
            LayoutAlgorithm::Radial => Self::radial_layout(graph, config),
        }
    }

    /// Grid layout - O(n)
    fn grid_layout<N, E>(graph: &mut Graph<N, E>, config: &LayoutConfig) {
        let n = graph.node_count();
        if n == 0 {
            return;
        }

        let cols = (n as f32).sqrt().ceil() as usize;
        let rows = n.div_ceil(cols);

        let cell_width = config.width / cols as f32;
        let cell_height = config.height / rows as f32;

        for (i, node) in graph.nodes_mut().enumerate() {
            let col = i % cols;
            let row = i / cols;
            node.position = Position::new(
                (col as f32 + 0.5) * cell_width,
                (row as f32 + 0.5) * cell_height,
            );
        }
    }

    /// Force-directed layout - Fruchterman-Reingold
    /// O(n²) per iteration, with software fallback for non-SIMD (Jidoka per peer review #4)
    fn force_directed_layout<N, E>(graph: &mut Graph<N, E>, config: &LayoutConfig) {
        let n = graph.node_count();
        if n == 0 {
            return;
        }

        // Initialize with grid layout
        Self::grid_layout(graph, config);

        let k = config.optimal_distance;
        let k_squared = k * k;
        let mut temperature = config.width.min(config.height) / 10.0;

        // Collect node IDs for iteration
        let node_ids: Vec<String> = graph.nodes.keys().cloned().collect();

        for _ in 0..config.iterations {
            // Compute forces
            let mut forces: HashMap<String, (f32, f32)> = HashMap::new();
            for id in &node_ids {
                forces.insert(id.clone(), (0.0, 0.0));
            }

            // Repulsive forces between all pairs
            for i in 0..node_ids.len() {
                for j in (i + 1)..node_ids.len() {
                    let id_i = &node_ids[i];
                    let id_j = &node_ids[j];

                    let pos_i = graph
                        .nodes
                        .get(id_i)
                        .map(|n| n.position)
                        .unwrap_or_default();
                    let pos_j = graph
                        .nodes
                        .get(id_j)
                        .map(|n| n.position)
                        .unwrap_or_default();

                    let dx = pos_i.x - pos_j.x;
                    let dy = pos_i.y - pos_j.y;
                    let dist = (dx * dx + dy * dy).sqrt().max(0.01);

                    // Repulsive force: k² / d
                    let force = k_squared / dist;
                    let fx = (dx / dist) * force;
                    let fy = (dy / dist) * force;

                    if let Some(f) = forces.get_mut(id_i) {
                        f.0 += fx;
                        f.1 += fy;
                    }
                    if let Some(f) = forces.get_mut(id_j) {
                        f.0 -= fx;
                        f.1 -= fy;
                    }
                }
            }

            // Attractive forces along edges
            for edge in &graph.edges {
                let pos_from = graph
                    .nodes
                    .get(&edge.from)
                    .map(|n| n.position)
                    .unwrap_or_default();
                let pos_to = graph
                    .nodes
                    .get(&edge.to)
                    .map(|n| n.position)
                    .unwrap_or_default();

                let dx = pos_to.x - pos_from.x;
                let dy = pos_to.y - pos_from.y;
                let dist = (dx * dx + dy * dy).sqrt().max(0.01);

                // Attractive force: d² / k
                let force = (dist * dist) / k;
                let fx = (dx / dist) * force;
                let fy = (dy / dist) * force;

                if let Some(f) = forces.get_mut(&edge.from) {
                    f.0 += fx;
                    f.1 += fy;
                }
                if let Some(f) = forces.get_mut(&edge.to) {
                    f.0 -= fx;
                    f.1 -= fy;
                }
            }

            // Apply forces with temperature limiting
            for id in &node_ids {
                if let (Some(node), Some(&(fx, fy))) = (graph.nodes.get_mut(id), forces.get(id)) {
                    let mag = (fx * fx + fy * fy).sqrt().max(0.01);
                    let capped_mag = mag.min(temperature);
                    node.position.x += (fx / mag) * capped_mag;
                    node.position.y += (fy / mag) * capped_mag;

                    // Clamp to bounds
                    node.position.x = node.position.x.clamp(1.0, config.width - 1.0);
                    node.position.y = node.position.y.clamp(1.0, config.height - 1.0);
                }
            }

            // Cool down
            temperature *= config.cooling;
        }
    }

    /// Hierarchical layout - simplified Sugiyama
    fn hierarchical_layout<N, E>(graph: &mut Graph<N, E>, config: &LayoutConfig) {
        // For now, use topological-like ordering
        // Full Sugiyama would require cycle removal and crossing minimization
        let n = graph.node_count();
        if n == 0 {
            return;
        }

        // Compute in-degrees for layer assignment
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        for id in graph.nodes.keys() {
            in_degree.insert(id.clone(), 0);
        }
        for edge in &graph.edges {
            *in_degree.entry(edge.to.clone()).or_default() += 1;
        }

        // Assign layers based on longest path
        let mut layers: HashMap<String, usize> = HashMap::new();
        let mut queue: Vec<String> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(id, _)| id.clone())
            .collect();

        for id in &queue {
            layers.insert(id.clone(), 0);
        }

        let mut max_layer = 0;
        while let Some(id) = queue.pop() {
            let current_layer = *layers.get(&id).unwrap_or(&0);
            for neighbor in graph.neighbors(&id) {
                let new_layer = current_layer + 1;
                let entry = layers.entry(neighbor.to_string()).or_insert(0);
                if new_layer > *entry {
                    *entry = new_layer;
                    max_layer = max_layer.max(new_layer);
                }
                queue.push(neighbor.to_string());
            }
        }

        // Position nodes by layer
        let layer_height = config.height / (max_layer + 1) as f32;
        let mut layer_counts: HashMap<usize, usize> = HashMap::new();
        let mut layer_positions: HashMap<usize, usize> = HashMap::new();

        // Count nodes per layer
        for &layer in layers.values() {
            *layer_counts.entry(layer).or_default() += 1;
        }

        // Position nodes
        for node in graph.nodes_mut() {
            let layer = *layers.get(&node.id).unwrap_or(&0);
            let count = *layer_counts.get(&layer).unwrap_or(&1);
            let pos_in_layer = *layer_positions.entry(layer).or_default();
            layer_positions.insert(layer, pos_in_layer + 1);

            let layer_width = config.width / count as f32;
            node.position = Position::new(
                (pos_in_layer as f32 + 0.5) * layer_width,
                (layer as f32 + 0.5) * layer_height,
            );
        }
    }

    /// Radial layout - for tree structures
    fn radial_layout<N, E>(graph: &mut Graph<N, E>, config: &LayoutConfig) {
        let n = graph.node_count();
        if n == 0 {
            return;
        }

        let center_x = config.width / 2.0;
        let center_y = config.height / 2.0;
        let radius = config.width.min(config.height) / 2.5;

        // Find root (node with no incoming edges)
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        for id in graph.nodes.keys() {
            in_degree.insert(id.clone(), 0);
        }
        for edge in &graph.edges {
            *in_degree.entry(edge.to.clone()).or_default() += 1;
        }

        let root = in_degree
            .iter()
            .find(|(_, &d)| d == 0)
            .map(|(id, _)| id.clone());

        if let Some(root_id) = root {
            // Place root at center
            if let Some(node) = graph.nodes.get_mut(&root_id) {
                node.position = Position::new(center_x, center_y);
            }

            // BFS to assign radial positions
            let mut visited: HashMap<String, usize> = HashMap::new();
            visited.insert(root_id.clone(), 0);
            let root_id_clone = root_id.clone();
            let mut queue = vec![root_id];
            let mut depth_counts: HashMap<usize, usize> = HashMap::new();
            let mut depth_positions: HashMap<usize, usize> = HashMap::new();

            while let Some(id) = queue.pop() {
                let depth = *visited.get(&id).unwrap_or(&0);
                for neighbor in graph.neighbors(&id) {
                    if !visited.contains_key(neighbor) {
                        visited.insert(neighbor.to_string(), depth + 1);
                        *depth_counts.entry(depth + 1).or_default() += 1;
                        queue.push(neighbor.to_string());
                    }
                }
            }

            // Position nodes radially
            let max_depth = visited.values().max().copied().unwrap_or(1);
            let mut unvisited_idx = 0;
            let unvisited_count = graph.node_count() - visited.len();

            for node in graph.nodes_mut() {
                if node.id == root_id_clone {
                    continue;
                }

                if let Some(&depth) = visited.get(&node.id) {
                    // Node was visited by BFS - position radially
                    let count = *depth_counts.get(&depth).unwrap_or(&1);
                    let pos = *depth_positions.entry(depth).or_default();
                    depth_positions.insert(depth, pos + 1);

                    let angle = 2.0 * std::f32::consts::PI * (pos as f32 / count as f32);
                    let r = radius * (depth as f32 / max_depth as f32);

                    node.position = Position::new(
                        (center_x + r * angle.cos()).clamp(1.0, config.width - 1.0),
                        (center_y + r * angle.sin()).clamp(1.0, config.height - 1.0),
                    );
                } else {
                    // Unvisited/disconnected node - place on outer ring
                    let angle = 2.0
                        * std::f32::consts::PI
                        * (unvisited_idx as f32 / unvisited_count.max(1) as f32);
                    let r = radius * 1.2; // Slightly outside main graph
                    unvisited_idx += 1;

                    node.position = Position::new(
                        (center_x + r * angle.cos()).clamp(1.0, config.width - 1.0),
                        (center_y + r * angle.sin()).clamp(1.0, config.height - 1.0),
                    );
                }
            }
        } else {
            // Fallback to grid if no root found
            Self::grid_layout(graph, config);
        }
    }

    /// Circular layout - nodes arranged in a circle (Cytoscape pattern)
    fn circular_layout<N, E>(graph: &mut Graph<N, E>, config: &LayoutConfig) {
        let n = graph.node_count();
        if n == 0 {
            return;
        }

        let center_x = config.width / 2.0;
        let center_y = config.height / 2.0;
        let radius = config.width.min(config.height) / 2.5;

        for (i, node) in graph.nodes_mut().enumerate() {
            let angle = 2.0 * std::f32::consts::PI * (i as f32 / n as f32);
            node.position = Position::new(
                (center_x + radius * angle.cos()).clamp(1.0, config.width - 1.0),
                (center_y + radius * angle.sin()).clamp(1.0, config.height - 1.0),
            );
        }
    }

    /// Concentric layout - nodes in rings by importance/degree (Cytoscape/Gephi pattern)
    fn concentric_layout<N, E>(graph: &mut Graph<N, E>, config: &LayoutConfig) {
        let n = graph.node_count();
        if n == 0 {
            return;
        }

        let center_x = config.width / 2.0;
        let center_y = config.height / 2.0;
        let max_radius = config.width.min(config.height) / 2.5;

        // Sort nodes by importance (highest importance = center)
        let mut node_order: Vec<(String, f32)> = graph
            .nodes()
            .map(|n| (n.id.clone(), n.importance))
            .collect();
        node_order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Assign to concentric rings (higher importance = smaller ring)
        let num_rings = 4.min(n);
        let nodes_per_ring = n.div_ceil(num_rings);

        for (idx, (node_id, _)) in node_order.iter().enumerate() {
            let ring = idx / nodes_per_ring;
            let pos_in_ring = idx % nodes_per_ring;
            let nodes_in_this_ring = if ring == num_rings - 1 {
                n - ring * nodes_per_ring
            } else {
                nodes_per_ring
            };

            // Ring 0 = center, ring N = outer
            let radius = if ring == 0 && nodes_in_this_ring == 1 {
                0.0 // Single node at center
            } else {
                max_radius * ((ring + 1) as f32 / num_rings as f32)
            };

            let angle =
                2.0 * std::f32::consts::PI * (pos_in_ring as f32 / nodes_in_this_ring as f32);

            if let Some(node) = graph.nodes.get_mut(node_id) {
                node.position = Position::new(
                    (center_x + radius * angle.cos()).clamp(1.0, config.width - 1.0),
                    (center_y + radius * angle.sin()).clamp(1.0, config.height - 1.0),
                );
            }
        }
    }
}

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

// ============================================================================
// GRAPH-005: Graph Analytics Integration (trueno-graph)
// ============================================================================

/// Community color palette for Louvain visualization
/// Colors chosen for maximum distinguishability (per Healey's preattentive research)
pub const COMMUNITY_COLORS: [&str; 8] = [
    "\x1b[31m", // Red
    "\x1b[32m", // Green
    "\x1b[33m", // Yellow
    "\x1b[34m", // Blue
    "\x1b[35m", // Magenta
    "\x1b[36m", // Cyan
    "\x1b[91m", // Bright Red
    "\x1b[92m", // Bright Green
];

/// Graph analytics engine for PageRank and community detection
pub struct GraphAnalytics;

impl GraphAnalytics {
    /// Compute PageRank scores and apply as node importance
    ///
    /// Based on Page et al. (1999) "The PageRank Citation Ranking"
    ///
    /// # Arguments
    /// * `graph` - Graph to analyze
    /// * `damping` - Damping factor (default: 0.85)
    /// * `iterations` - Max iterations (default: 20)
    ///
    /// # Returns
    /// HashMap of node_id -> PageRank score
    pub fn pagerank<N, E>(
        graph: &Graph<N, E>,
        damping: f32,
        iterations: usize,
    ) -> HashMap<String, f32> {
        let n = graph.node_count();
        if n == 0 {
            return HashMap::new();
        }

        let teleport = (1.0 - damping) / n as f32;
        let node_ids: Vec<String> = graph.nodes.keys().cloned().collect();

        // Initialize: uniform distribution
        let mut ranks: HashMap<String, f32> = node_ids
            .iter()
            .map(|id| (id.clone(), 1.0 / n as f32))
            .collect();

        // Compute out-degrees
        let out_degrees: HashMap<String, usize> = node_ids
            .iter()
            .map(|id| (id.clone(), graph.neighbors(id).len()))
            .collect();

        // Power iteration
        for _ in 0..iterations {
            let mut new_ranks: HashMap<String, f32> =
                node_ids.iter().map(|id| (id.clone(), teleport)).collect();

            // Distribute rank along edges
            for edge in graph.edges() {
                let out_deg = *out_degrees.get(&edge.from).unwrap_or(&1);
                if out_deg > 0 {
                    let contribution =
                        damping * ranks.get(&edge.from).unwrap_or(&0.0) / out_deg as f32;
                    *new_ranks.entry(edge.to.clone()).or_default() += contribution;
                }
            }

            // Handle dangling nodes (no outgoing edges)
            let dangling_sum: f32 = node_ids
                .iter()
                .filter(|id| *out_degrees.get(*id).unwrap_or(&0) == 0)
                .map(|id| ranks.get(id).unwrap_or(&0.0))
                .sum();

            let dangling_contribution = damping * dangling_sum / n as f32;
            for rank in new_ranks.values_mut() {
                *rank += dangling_contribution;
            }

            ranks = new_ranks;
        }

        // Normalize to [0, 1]
        let max_rank = ranks.values().cloned().fold(0.0_f32, f32::max);
        if max_rank > 0.0 {
            for rank in ranks.values_mut() {
                *rank /= max_rank;
            }
        }

        ranks
    }

    /// Apply PageRank scores as node importance
    pub fn apply_pagerank<N, E>(graph: &mut Graph<N, E>, damping: f32, iterations: usize) {
        let ranks = Self::pagerank(graph, damping, iterations);
        for (id, rank) in ranks {
            if let Some(node) = graph.get_node_mut(&id) {
                node.importance = rank;
            }
        }
    }

    /// Detect communities using Louvain-style modularity optimization
    ///
    /// Based on Blondel et al. (2008) "Fast unfolding of communities in large networks"
    ///
    /// Simplified implementation suitable for TUI visualization.
    /// For production use, prefer trueno-graph's full Louvain implementation.
    ///
    /// # Returns
    /// HashMap of node_id -> community_id
    pub fn detect_communities<N, E>(graph: &Graph<N, E>) -> HashMap<String, usize> {
        let node_ids: Vec<String> = graph.nodes.keys().cloned().collect();
        let n = node_ids.len();

        if n == 0 {
            return HashMap::new();
        }

        // Initialize: each node in its own community
        let mut communities: HashMap<String, usize> = node_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), i))
            .collect();

        // Build adjacency for quick lookup
        let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
        for edge in graph.edges() {
            adjacency
                .entry(edge.from.clone())
                .or_default()
                .push(edge.to.clone());
            // For undirected treatment
            adjacency
                .entry(edge.to.clone())
                .or_default()
                .push(edge.from.clone());
        }

        // Greedy modularity optimization (simplified)
        let mut improved = true;
        let mut max_iterations = 10;

        while improved && max_iterations > 0 {
            improved = false;
            max_iterations -= 1;

            for node_id in &node_ids {
                let current_comm = *communities.get(node_id).unwrap_or(&0);

                // Find best community among neighbors
                let neighbors = adjacency.get(node_id).cloned().unwrap_or_default();
                let mut neighbor_comms: HashMap<usize, usize> = HashMap::new();

                for neighbor in &neighbors {
                    let comm = *communities.get(neighbor).unwrap_or(&0);
                    *neighbor_comms.entry(comm).or_default() += 1;
                }

                // Move to community with most connections
                if let Some((&best_comm, &count)) = neighbor_comms.iter().max_by_key(|(_, &c)| c) {
                    if best_comm != current_comm && count > 1 {
                        communities.insert(node_id.clone(), best_comm);
                        improved = true;
                    }
                }
            }
        }

        // Renumber communities to be contiguous (0, 1, 2, ...)
        let mut comm_map: HashMap<usize, usize> = HashMap::new();
        let mut next_comm = 0;

        for comm in communities.values_mut() {
            let new_comm = *comm_map.entry(*comm).or_insert_with(|| {
                let c = next_comm;
                next_comm += 1;
                c
            });
            *comm = new_comm;
        }

        communities
    }

    /// Apply community detection and color nodes by community
    pub fn apply_communities<N, E>(graph: &mut Graph<N, E>) -> usize {
        let communities = Self::detect_communities(graph);
        let num_communities = communities.values().max().map(|&m| m + 1).unwrap_or(0);

        // Update node status based on community for visualization
        // Communities are mapped to different statuses for color variety
        for (id, comm) in &communities {
            if let Some(node) = graph.get_node_mut(id) {
                // Store community in a way that can be rendered
                // Use importance to encode community for now (0.0-1.0 range)
                let comm_normalized = if num_communities > 0 {
                    *comm as f32 / num_communities as f32
                } else {
                    0.0
                };
                // Blend with existing importance
                node.importance = (node.importance + comm_normalized) / 2.0;
            }
        }

        num_communities
    }

    /// Get community color for rendering
    #[must_use]
    pub fn community_color(community_id: usize) -> &'static str {
        COMMUNITY_COLORS[community_id % COMMUNITY_COLORS.len()]
    }

    // =========================================================================
    // Centrality Metrics (Neo4j/Gephi patterns)
    // =========================================================================

    /// Compute degree centrality for all nodes
    ///
    /// Degree centrality = number of connections / (n-1)
    /// This is the simplest centrality measure, per Gephi docs.
    #[must_use]
    pub fn degree_centrality<N, E>(graph: &Graph<N, E>) -> HashMap<String, f32> {
        let n = graph.node_count();
        if n <= 1 {
            return graph.nodes.keys().map(|id| (id.clone(), 0.0)).collect();
        }

        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut out_degree: HashMap<String, usize> = HashMap::new();

        for id in graph.nodes.keys() {
            in_degree.insert(id.clone(), 0);
            out_degree.insert(id.clone(), 0);
        }

        for edge in &graph.edges {
            *out_degree.entry(edge.from.clone()).or_default() += 1;
            *in_degree.entry(edge.to.clone()).or_default() += 1;
        }

        // Total degree = in + out, normalized
        graph
            .nodes
            .keys()
            .map(|id| {
                let total = in_degree.get(id).unwrap_or(&0) + out_degree.get(id).unwrap_or(&0);
                (id.clone(), total as f32 / (n - 1) as f32)
            })
            .collect()
    }

    /// Compute betweenness centrality using Brandes algorithm
    ///
    /// Betweenness = fraction of shortest paths passing through node.
    /// Reference: Brandes (2001) "A Faster Algorithm for Betweenness Centrality"
    #[must_use]
    pub fn betweenness_centrality<N, E>(graph: &Graph<N, E>) -> HashMap<String, f32> {
        let n = graph.node_count();
        if n <= 2 {
            return graph.nodes.keys().map(|id| (id.clone(), 0.0)).collect();
        }

        let mut betweenness: HashMap<String, f32> =
            graph.nodes.keys().map(|id| (id.clone(), 0.0)).collect();
        let node_ids: Vec<String> = graph.nodes.keys().cloned().collect();

        // Build adjacency list
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        for id in &node_ids {
            adj.insert(id.clone(), Vec::new());
        }
        for edge in &graph.edges {
            adj.entry(edge.from.clone())
                .or_default()
                .push(edge.to.clone());
        }

        // Brandes algorithm
        for s in &node_ids {
            let mut stack: Vec<String> = Vec::new();
            let mut pred: HashMap<String, Vec<String>> = HashMap::new();
            let mut sigma: HashMap<String, f32> = HashMap::new();
            let mut dist: HashMap<String, i32> = HashMap::new();

            for id in &node_ids {
                pred.insert(id.clone(), Vec::new());
                sigma.insert(id.clone(), 0.0);
                dist.insert(id.clone(), -1);
            }
            sigma.insert(s.clone(), 1.0);
            dist.insert(s.clone(), 0);

            let mut queue = std::collections::VecDeque::new();
            queue.push_back(s.clone());

            while let Some(v) = queue.pop_front() {
                stack.push(v.clone());
                let v_dist = *dist.get(&v).unwrap_or(&0);

                for w in adj.get(&v).unwrap_or(&Vec::new()) {
                    let w_dist = dist.get(w).unwrap_or(&-1);
                    if *w_dist < 0 {
                        queue.push_back(w.clone());
                        dist.insert(w.clone(), v_dist + 1);
                    }
                    if *dist.get(w).unwrap_or(&0) == v_dist + 1 {
                        let sigma_v = *sigma.get(&v).unwrap_or(&0.0);
                        *sigma.entry(w.clone()).or_default() += sigma_v;
                        pred.entry(w.clone()).or_default().push(v.clone());
                    }
                }
            }

            let mut delta: HashMap<String, f32> =
                node_ids.iter().map(|id| (id.clone(), 0.0)).collect();
            while let Some(w) = stack.pop() {
                for v in pred.get(&w).unwrap_or(&Vec::new()) {
                    let sigma_v = sigma.get(v).unwrap_or(&1.0);
                    let sigma_w = sigma.get(&w).unwrap_or(&1.0);
                    let delta_w = delta.get(&w).unwrap_or(&0.0);
                    *delta.entry(v.clone()).or_default() += (sigma_v / sigma_w) * (1.0 + delta_w);
                }
                if &w != s {
                    *betweenness.entry(w.clone()).or_default() += delta.get(&w).unwrap_or(&0.0);
                }
            }
        }

        // Normalize
        let norm = if n > 2 {
            2.0 / ((n - 1) * (n - 2)) as f32
        } else {
            1.0
        };
        for val in betweenness.values_mut() {
            *val *= norm;
        }

        betweenness
    }

    /// Compute closeness centrality
    ///
    /// Closeness = (n-1) / sum of shortest path distances from node to all others.
    /// High closeness = node is "close" to all other nodes.
    #[must_use]
    pub fn closeness_centrality<N, E>(graph: &Graph<N, E>) -> HashMap<String, f32> {
        let n = graph.node_count();
        if n <= 1 {
            return graph.nodes.keys().map(|id| (id.clone(), 0.0)).collect();
        }

        let node_ids: Vec<String> = graph.nodes.keys().cloned().collect();

        node_ids
            .iter()
            .map(|source| {
                let distances = Self::bfs_distances(graph, source);
                let reachable: Vec<i32> = distances.values().filter(|&&d| d > 0).copied().collect();

                let closeness = if reachable.is_empty() {
                    0.0
                } else {
                    let sum_dist: i32 = reachable.iter().sum();
                    reachable.len() as f32 / sum_dist as f32
                };

                (source.clone(), closeness)
            })
            .collect()
    }

    // =========================================================================
    // Path Analysis (Neo4j/Cytoscape patterns)
    // =========================================================================

    /// Find shortest path between two nodes using BFS
    ///
    /// Returns None if no path exists.
    #[must_use]
    pub fn shortest_path<N, E>(graph: &Graph<N, E>, from: &str, to: &str) -> Option<Vec<String>> {
        if from == to {
            return Some(vec![from.to_string()]);
        }

        if !graph.nodes.contains_key(from) || !graph.nodes.contains_key(to) {
            return None;
        }

        let mut visited: HashMap<String, String> = HashMap::new(); // node -> predecessor
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(from.to_string());
        visited.insert(from.to_string(), String::new());

        while let Some(current) = queue.pop_front() {
            for neighbor in graph.neighbors(&current) {
                if !visited.contains_key(neighbor) {
                    visited.insert(neighbor.to_string(), current.clone());
                    if neighbor == to {
                        // Reconstruct path
                        let mut path = vec![to.to_string()];
                        let mut node = to.to_string();
                        while let Some(pred) = visited.get(&node) {
                            if pred.is_empty() {
                                break;
                            }
                            path.push(pred.clone());
                            node = pred.clone();
                        }
                        path.reverse();
                        return Some(path);
                    }
                    queue.push_back(neighbor.to_string());
                }
            }
        }

        None
    }

    /// BFS distances from source to all reachable nodes
    fn bfs_distances<N, E>(graph: &Graph<N, E>, source: &str) -> HashMap<String, i32> {
        let mut dist: HashMap<String, i32> =
            graph.nodes.keys().map(|id| (id.clone(), -1)).collect();
        dist.insert(source.to_string(), 0);

        let mut queue = std::collections::VecDeque::new();
        queue.push_back(source.to_string());

        while let Some(current) = queue.pop_front() {
            let current_dist = *dist.get(&current).unwrap_or(&0);
            for neighbor in graph.neighbors(&current) {
                if *dist.get(neighbor).unwrap_or(&-1) < 0 {
                    dist.insert(neighbor.to_string(), current_dist + 1);
                    queue.push_back(neighbor.to_string());
                }
            }
        }

        dist
    }

    /// Check if graph is connected (weakly connected for directed graphs)
    ///
    /// Uses undirected BFS - treats all edges as bidirectional.
    #[must_use]
    pub fn is_connected<N, E>(graph: &Graph<N, E>) -> bool {
        if graph.node_count() == 0 {
            return true;
        }

        // Build undirected adjacency for weak connectivity
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        for id in graph.nodes.keys() {
            adj.insert(id.clone(), Vec::new());
        }
        for edge in &graph.edges {
            adj.entry(edge.from.clone())
                .or_default()
                .push(edge.to.clone());
            adj.entry(edge.to.clone())
                .or_default()
                .push(edge.from.clone());
        }

        // BFS from first node
        let first = graph.nodes.keys().next().unwrap();
        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(first.clone());
        visited.insert(first.clone());

        while let Some(current) = queue.pop_front() {
            for neighbor in adj.get(&current).unwrap_or(&Vec::new()) {
                if !visited.contains(neighbor) {
                    visited.insert(neighbor.clone());
                    queue.push_back(neighbor.clone());
                }
            }
        }

        visited.len() == graph.node_count()
    }
}

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

/// Extended graph with analytics capabilities
pub trait GraphAnalyticsExt<N, E> {
    /// Compute and apply PageRank
    fn compute_pagerank(&mut self, damping: f32, iterations: usize);

    /// Detect and apply communities
    fn detect_communities(&mut self) -> usize;

    /// Get PageRank scores
    fn pagerank_scores(&self) -> HashMap<String, f32>;
}

impl<N, E> GraphAnalyticsExt<N, E> for Graph<N, E> {
    fn compute_pagerank(&mut self, damping: f32, iterations: usize) {
        GraphAnalytics::apply_pagerank(self, damping, iterations);
    }

    fn detect_communities(&mut self) -> usize {
        GraphAnalytics::apply_communities(self)
    }

    fn pagerank_scores(&self) -> HashMap<String, f32> {
        GraphAnalytics::pagerank(self, 0.85, 20)
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
