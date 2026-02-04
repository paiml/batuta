//! Core types for graph visualization
//!
//! Contains `NodeShape`, `NodeStatus`, `Position`, `Node`, and `Edge` types.

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
    /// Circle: default node
    #[default]
    Circle,
    /// Diamond: input/source
    Diamond,
    /// Square: transform/process
    Square,
    /// Triangle: output/sink
    Triangle,
    /// Star: highlighted/selected
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
