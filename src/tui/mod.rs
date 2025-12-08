//! TUI (Terminal User Interface) Module
//!
//! Provides terminal-based visualization components for the batuta stack.
//!
//! ## Modules
//!
//! - `graph`: Graph visualization with force-directed and hierarchical layouts
//!
//! ## Design Principles
//!
//! Follows Toyota Way principles:
//! - **Mieruka**: Visual management for instant status recognition
//! - **Jidoka**: Built-in quality via comprehensive tests
//! - **Respect for People**: Accessibility via shapes (not just colors)

pub mod graph;

pub use graph::{
    Edge, Graph, GraphAnalytics, GraphAnalyticsExt, GraphRenderer, LayoutAlgorithm, LayoutConfig,
    LayoutEngine, Node, NodeShape, NodeStatus, Position, RenderMode, RenderedGraph,
    COMMUNITY_COLORS, DEFAULT_VISIBLE_NODES, MAX_TUI_NODES,
};
