//! Core Graph data structure
//!
//! Contains the `Graph` struct and its core methods for node/edge management.

use std::collections::HashMap;

use super::types::{Edge, Node, MAX_TUI_NODES};

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
