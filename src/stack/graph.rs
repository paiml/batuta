#![allow(dead_code)]
//! Dependency Graph Analysis
//!
//! Uses petgraph to build and analyze the dependency graph
//! for topological sorting and conflict detection.

use crate::stack::is_paiml_crate;
use crate::stack::types::*;
use anyhow::{anyhow, Result};
use petgraph::algo::{is_cyclic_directed, toposort};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use std::collections::HashMap;
use std::path::Path;

/// Dependency graph for the PAIML stack
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    /// The underlying directed graph
    graph: DiGraph<String, DependencyEdge>,

    /// Map from crate name to node index
    node_indices: HashMap<String, NodeIndex>,

    /// Crate information for each node
    crate_info: HashMap<String, CrateInfo>,
}

/// Edge data in the dependency graph
#[derive(Debug, Clone)]
pub struct DependencyEdge {
    /// Version requirement
    pub version_req: String,

    /// Whether this is a path dependency
    pub is_path: bool,

    /// Dependency kind
    pub kind: DependencyKind,
}

impl DependencyGraph {
    /// Create a new empty dependency graph
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_indices: HashMap::new(),
            crate_info: HashMap::new(),
        }
    }

    /// Build graph from cargo metadata
    #[cfg(feature = "native")]
    pub fn from_workspace(workspace_path: &Path) -> Result<Self> {
        use cargo_metadata::MetadataCommand;

        let metadata = MetadataCommand::new()
            .manifest_path(workspace_path.join("Cargo.toml"))
            .exec()
            .map_err(|e| anyhow!("Failed to read cargo metadata: {}", e))?;

        let mut graph = Self::new();

        // First pass: add all PAIML crates as nodes
        for package in &metadata.packages {
            if is_paiml_crate(&package.name) {
                let version = package.version.clone();
                let semver_version =
                    semver::Version::new(version.major, version.minor, version.patch);

                let crate_info = CrateInfo::new(
                    &package.name,
                    semver_version,
                    package.manifest_path.clone().into_std_path_buf(),
                );

                graph.add_crate(crate_info);
            }
        }

        // Second pass: add edges for dependencies
        for package in &metadata.packages {
            if !is_paiml_crate(&package.name) {
                continue;
            }

            for dep in &package.dependencies {
                if is_paiml_crate(&dep.name) {
                    let is_path = dep.path.is_some();
                    let version_req = dep.req.to_string();

                    let kind = match dep.kind {
                        cargo_metadata::DependencyKind::Normal => DependencyKind::Normal,
                        cargo_metadata::DependencyKind::Development => DependencyKind::Dev,
                        cargo_metadata::DependencyKind::Build => DependencyKind::Build,
                        _ => DependencyKind::Normal,
                    };

                    let edge = DependencyEdge {
                        version_req,
                        is_path,
                        kind,
                    };

                    // Add edge: package depends on dep
                    graph.add_dependency(&package.name, &dep.name, edge);

                    // Update crate info with dependency
                    if let Some(info) = graph.crate_info.get_mut(&package.name) {
                        let dep_info = if is_path {
                            let path = dep
                                .path
                                .as_ref()
                                .map(|p| p.clone().into_std_path_buf())
                                .unwrap_or_default();
                            DependencyInfo::path(&dep.name, path)
                        } else {
                            DependencyInfo::new(&dep.name, dep.req.to_string())
                        };
                        info.paiml_dependencies.push(dep_info);
                    }
                }
            }
        }

        Ok(graph)
    }

    /// Add a crate to the graph
    pub fn add_crate(&mut self, info: CrateInfo) {
        let name = info.name.clone();
        if !self.node_indices.contains_key(&name) {
            let idx = self.graph.add_node(name.clone());
            self.node_indices.insert(name.clone(), idx);
        }
        self.crate_info.insert(name, info);
    }

    /// Add a dependency edge between crates
    pub fn add_dependency(&mut self, from: &str, to: &str, edge: DependencyEdge) {
        // Ensure both nodes exist
        if !self.node_indices.contains_key(from) {
            let idx = self.graph.add_node(from.to_string());
            self.node_indices.insert(from.to_string(), idx);
        }
        if !self.node_indices.contains_key(to) {
            let idx = self.graph.add_node(to.to_string());
            self.node_indices.insert(to.to_string(), idx);
        }

        let from_idx = self.node_indices[from];
        let to_idx = self.node_indices[to];

        self.graph.add_edge(from_idx, to_idx, edge);
    }

    /// Check if the graph has cycles (excluding dev-dependencies)
    ///
    /// Dev-dependencies are excluded because they don't create real dependency
    /// cycles for release ordering. A crate can have a dev-dependency on another
    /// crate that depends on it (common for integration testing).
    pub fn has_cycles(&self) -> bool {
        let release_graph = self.build_release_graph();
        is_cyclic_directed(&release_graph)
    }

    /// Build a graph that only includes release-relevant dependencies
    /// (excludes dev-dependencies, includes normal and build dependencies)
    fn build_release_graph(&self) -> DiGraph<String, ()> {
        let mut release_graph: DiGraph<String, ()> = DiGraph::new();
        let mut node_map: HashMap<NodeIndex, NodeIndex> = HashMap::new();

        // Add all nodes
        for idx in self.graph.node_indices() {
            let new_idx = release_graph.add_node(self.graph[idx].clone());
            node_map.insert(idx, new_idx);
        }

        // Add only non-dev edges
        for edge in self.graph.edge_references() {
            // Skip dev dependencies - they don't affect release ordering
            if edge.weight().kind != DependencyKind::Dev {
                let from = node_map[&edge.source()];
                let to = node_map[&edge.target()];
                release_graph.add_edge(from, to, ());
            }
        }

        release_graph
    }

    /// Get topological order for releases (dependencies first)
    ///
    /// Uses the release graph which excludes dev-dependencies to avoid
    /// false cycle detection (fixes issue #13).
    pub fn topological_order(&self) -> Result<Vec<String>> {
        let release_graph = self.build_release_graph();

        if is_cyclic_directed(&release_graph) {
            return Err(anyhow!("Circular dependency detected in the graph"));
        }

        let sorted = toposort(&release_graph, None)
            .map_err(|_| anyhow!("Failed to compute topological order"))?;

        // Reverse because toposort gives dependents first, we want dependencies first
        Ok(sorted
            .into_iter()
            .rev()
            .map(|idx| release_graph[idx].clone())
            .collect())
    }

    /// Get release order for a specific crate and its dependencies
    pub fn release_order_for(&self, crate_name: &str) -> Result<Vec<String>> {
        let full_order = self.topological_order()?;
        let deps = self.all_dependencies(crate_name);

        // Filter to only include the target crate and its dependencies
        let mut order: Vec<String> = full_order
            .into_iter()
            .filter(|name| name == crate_name || deps.contains(name))
            .collect();

        // Ensure the target crate is at the end
        if let Some(pos) = order.iter().position(|n| n == crate_name) {
            order.remove(pos);
            order.push(crate_name.to_string());
        }

        Ok(order)
    }

    /// Get all dependencies (transitive) for a crate
    pub fn all_dependencies(&self, crate_name: &str) -> Vec<String> {
        let mut deps = Vec::new();
        let mut visited = std::collections::HashSet::new();

        if let Some(&idx) = self.node_indices.get(crate_name) {
            self.collect_dependencies(idx, &mut deps, &mut visited);
        }

        deps
    }

    fn collect_dependencies(
        &self,
        idx: NodeIndex,
        deps: &mut Vec<String>,
        visited: &mut std::collections::HashSet<NodeIndex>,
    ) {
        if visited.contains(&idx) {
            return;
        }
        visited.insert(idx);

        for neighbor in self.graph.neighbors_directed(idx, Direction::Outgoing) {
            let name = self.graph[neighbor].clone();
            if !deps.contains(&name) {
                deps.push(name);
            }
            self.collect_dependencies(neighbor, deps, visited);
        }
    }

    /// Get crates that depend on this crate (reverse dependencies)
    pub fn dependents(&self, crate_name: &str) -> Vec<String> {
        let mut dependents = Vec::new();

        if let Some(&idx) = self.node_indices.get(crate_name) {
            for neighbor in self.graph.neighbors_directed(idx, Direction::Incoming) {
                dependents.push(self.graph[neighbor].clone());
            }
        }

        dependents
    }

    /// Get all path dependencies in the graph
    pub fn find_path_dependencies(&self) -> Vec<PathDependencyIssue> {
        let mut issues = Vec::new();

        for edge_ref in self.graph.edge_references() {
            if edge_ref.weight().is_path {
                let from = &self.graph[edge_ref.source()];
                let to = &self.graph[edge_ref.target()];

                issues.push(PathDependencyIssue {
                    crate_name: from.clone(),
                    dependency: to.clone(),
                    current: format!("path = \"../{}\"", to),
                    recommended: None, // Will be filled in by checker
                });
            }
        }

        issues
    }

    /// Detect version conflicts for external dependencies
    pub fn detect_conflicts(&self) -> Vec<VersionConflict> {
        let mut dep_versions: HashMap<String, Vec<ConflictUsage>> = HashMap::new();

        // Collect all external dependency versions
        for (crate_name, info) in &self.crate_info {
            for dep in &info.external_dependencies {
                dep_versions
                    .entry(dep.name.clone())
                    .or_default()
                    .push(ConflictUsage {
                        crate_name: crate_name.clone(),
                        version_req: dep.version_req.clone(),
                    });
            }
        }

        // Find conflicts (different versions of same dependency)
        let mut conflicts = Vec::new();
        for (dep_name, usages) in dep_versions {
            if usages.len() > 1 {
                let versions: std::collections::HashSet<_> =
                    usages.iter().map(|u| &u.version_req).collect();

                if versions.len() > 1 {
                    conflicts.push(VersionConflict {
                        dependency: dep_name,
                        usages,
                        recommendation: None,
                    });
                }
            }
        }

        conflicts
    }

    /// Get crate info by name
    pub fn get_crate(&self, name: &str) -> Option<&CrateInfo> {
        self.crate_info.get(name)
    }

    /// Get mutable crate info by name
    pub fn get_crate_mut(&mut self, name: &str) -> Option<&mut CrateInfo> {
        self.crate_info.get_mut(name)
    }

    /// Get all crates in the graph
    pub fn all_crates(&self) -> impl Iterator<Item = &CrateInfo> {
        self.crate_info.values()
    }

    /// Get number of crates in the graph
    pub fn crate_count(&self) -> usize {
        self.crate_info.len()
    }
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Path dependency issue for reporting
#[derive(Debug, Clone)]
pub struct PathDependencyIssue {
    /// Crate that has the path dependency
    pub crate_name: String,

    /// Dependency that is path-based
    pub dependency: String,

    /// Current specification
    pub current: String,

    /// Recommended crates.io version
    pub recommended: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> DependencyGraph {
        let mut graph = DependencyGraph::new();

        // Create crates
        graph.add_crate(CrateInfo::new(
            "trueno",
            semver::Version::new(1, 2, 0),
            std::path::PathBuf::from("trueno/Cargo.toml"),
        ));
        graph.add_crate(CrateInfo::new(
            "aprender",
            semver::Version::new(0, 8, 0),
            std::path::PathBuf::from("aprender/Cargo.toml"),
        ));
        graph.add_crate(CrateInfo::new(
            "entrenar",
            semver::Version::new(0, 2, 0),
            std::path::PathBuf::from("entrenar/Cargo.toml"),
        ));
        graph.add_crate(CrateInfo::new(
            "alimentar",
            semver::Version::new(0, 3, 0),
            std::path::PathBuf::from("alimentar/Cargo.toml"),
        ));

        // Add dependencies
        // aprender -> trueno
        graph.add_dependency(
            "aprender",
            "trueno",
            DependencyEdge {
                version_req: "^1.0".to_string(),
                is_path: false,
                kind: DependencyKind::Normal,
            },
        );

        // entrenar -> aprender
        graph.add_dependency(
            "entrenar",
            "aprender",
            DependencyEdge {
                version_req: "^0.8".to_string(),
                is_path: false,
                kind: DependencyKind::Normal,
            },
        );

        // entrenar -> alimentar (PATH DEPENDENCY - the bug!)
        graph.add_dependency(
            "entrenar",
            "alimentar",
            DependencyEdge {
                version_req: String::new(),
                is_path: true,
                kind: DependencyKind::Normal,
            },
        );

        // alimentar -> trueno
        graph.add_dependency(
            "alimentar",
            "trueno",
            DependencyEdge {
                version_req: "^1.0".to_string(),
                is_path: false,
                kind: DependencyKind::Normal,
            },
        );

        graph
    }

    #[test]
    fn test_graph_creation() {
        let graph = DependencyGraph::new();
        assert_eq!(graph.crate_count(), 0);
        assert!(!graph.has_cycles());
    }

    #[test]
    fn test_add_crate() {
        let mut graph = DependencyGraph::new();
        graph.add_crate(CrateInfo::new(
            "trueno",
            semver::Version::new(1, 0, 0),
            std::path::PathBuf::new(),
        ));

        assert_eq!(graph.crate_count(), 1);
        assert!(graph.get_crate("trueno").is_some());
        assert!(graph.get_crate("serde").is_none());
    }

    #[test]
    fn test_dependency_edges() {
        let graph = create_test_graph();

        // Check dependencies
        let aprender_deps = graph.all_dependencies("aprender");
        assert!(aprender_deps.contains(&"trueno".to_string()));

        let entrenar_deps = graph.all_dependencies("entrenar");
        assert!(entrenar_deps.contains(&"aprender".to_string()));
        assert!(entrenar_deps.contains(&"alimentar".to_string()));
        assert!(entrenar_deps.contains(&"trueno".to_string())); // transitive
    }

    #[test]
    fn test_no_cycles() {
        let graph = create_test_graph();
        assert!(!graph.has_cycles());
    }

    #[test]
    fn test_cycle_detection() {
        let mut graph = DependencyGraph::new();

        graph.add_crate(CrateInfo::new(
            "a",
            semver::Version::new(1, 0, 0),
            std::path::PathBuf::new(),
        ));
        graph.add_crate(CrateInfo::new(
            "b",
            semver::Version::new(1, 0, 0),
            std::path::PathBuf::new(),
        ));
        graph.add_crate(CrateInfo::new(
            "c",
            semver::Version::new(1, 0, 0),
            std::path::PathBuf::new(),
        ));

        // Create cycle: a -> b -> c -> a
        graph.add_dependency(
            "a",
            "b",
            DependencyEdge {
                version_req: "1.0".to_string(),
                is_path: false,
                kind: DependencyKind::Normal,
            },
        );
        graph.add_dependency(
            "b",
            "c",
            DependencyEdge {
                version_req: "1.0".to_string(),
                is_path: false,
                kind: DependencyKind::Normal,
            },
        );
        graph.add_dependency(
            "c",
            "a",
            DependencyEdge {
                version_req: "1.0".to_string(),
                is_path: false,
                kind: DependencyKind::Normal,
            },
        );

        assert!(graph.has_cycles());
        assert!(graph.topological_order().is_err());
    }

    #[test]
    fn test_topological_order() {
        let graph = create_test_graph();
        let order = graph.topological_order().unwrap();

        // trueno should come before aprender
        let trueno_pos = order.iter().position(|n| n == "trueno").unwrap();
        let aprender_pos = order.iter().position(|n| n == "aprender").unwrap();
        assert!(trueno_pos < aprender_pos);

        // aprender should come before entrenar
        let entrenar_pos = order.iter().position(|n| n == "entrenar").unwrap();
        assert!(aprender_pos < entrenar_pos);

        // alimentar should come before entrenar
        let alimentar_pos = order.iter().position(|n| n == "alimentar").unwrap();
        assert!(alimentar_pos < entrenar_pos);
    }

    #[test]
    fn test_release_order_for_crate() {
        let graph = create_test_graph();
        let order = graph.release_order_for("entrenar").unwrap();

        // Should include entrenar and its dependencies
        assert!(order.contains(&"trueno".to_string()));
        assert!(order.contains(&"aprender".to_string()));
        assert!(order.contains(&"alimentar".to_string()));
        assert!(order.contains(&"entrenar".to_string()));

        // entrenar should be last
        assert_eq!(order.last().unwrap(), "entrenar");

        // trueno should be first (base dependency)
        assert_eq!(order.first().unwrap(), "trueno");
    }

    #[test]
    fn test_find_path_dependencies() {
        let graph = create_test_graph();
        let path_deps = graph.find_path_dependencies();

        // Should find the entrenar -> alimentar path dependency
        assert_eq!(path_deps.len(), 1);
        assert_eq!(path_deps[0].crate_name, "entrenar");
        assert_eq!(path_deps[0].dependency, "alimentar");
    }

    #[test]
    fn test_dependents() {
        let graph = create_test_graph();

        // trueno has dependents: aprender, alimentar
        let trueno_dependents = graph.dependents("trueno");
        assert!(trueno_dependents.contains(&"aprender".to_string()));
        assert!(trueno_dependents.contains(&"alimentar".to_string()));

        // aprender has dependent: entrenar
        let aprender_dependents = graph.dependents("aprender");
        assert!(aprender_dependents.contains(&"entrenar".to_string()));
    }

    #[test]
    fn test_version_conflict_detection() {
        let mut graph = DependencyGraph::new();

        // Create crates with conflicting arrow versions
        let mut renacer = CrateInfo::new(
            "renacer",
            semver::Version::new(0, 6, 0),
            std::path::PathBuf::new(),
        );
        renacer
            .external_dependencies
            .push(DependencyInfo::new("arrow", "54.0"));

        let mut trueno_graph = CrateInfo::new(
            "trueno-graph",
            semver::Version::new(0, 2, 0),
            std::path::PathBuf::new(),
        );
        trueno_graph
            .external_dependencies
            .push(DependencyInfo::new("arrow", "53.0"));

        graph.add_crate(renacer);
        graph.add_crate(trueno_graph);

        let conflicts = graph.detect_conflicts();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].dependency, "arrow");
        assert_eq!(conflicts[0].usages.len(), 2);
    }

    #[test]
    fn test_no_conflict_same_version() {
        let mut graph = DependencyGraph::new();

        // Create crates with same arrow version
        let mut crate_a = CrateInfo::new(
            "a",
            semver::Version::new(1, 0, 0),
            std::path::PathBuf::new(),
        );
        crate_a
            .external_dependencies
            .push(DependencyInfo::new("arrow", "54.0"));

        let mut crate_b = CrateInfo::new(
            "b",
            semver::Version::new(1, 0, 0),
            std::path::PathBuf::new(),
        );
        crate_b
            .external_dependencies
            .push(DependencyInfo::new("arrow", "54.0"));

        graph.add_crate(crate_a);
        graph.add_crate(crate_b);

        let conflicts = graph.detect_conflicts();
        assert!(conflicts.is_empty());
    }

    // ============================================================================
    // ISSUE-13: False circular dependency detection
    // https://github.com/paiml/batuta/issues/13
    // ============================================================================

    /// RED PHASE: Dev-dependencies should NOT create cycles for release ordering
    /// This reproduces issue #13: presentar workspace reports false cycle
    #[test]
    fn test_ISSUE_13_dev_dependency_not_cycle() {
        // ARRANGE: Create graph where:
        // - presentar depends on trueno (normal)
        // - trueno has dev-dependency on presentar (for testing)
        // This is NOT a real cycle for release purposes
        let mut graph = DependencyGraph::new();

        graph.add_crate(CrateInfo::new(
            "trueno",
            semver::Version::new(1, 0, 0),
            std::path::PathBuf::new(),
        ));
        graph.add_crate(CrateInfo::new(
            "presentar",
            semver::Version::new(0, 1, 0),
            std::path::PathBuf::new(),
        ));

        // presentar -> trueno (normal dependency)
        graph.add_dependency(
            "presentar",
            "trueno",
            DependencyEdge {
                version_req: "^1.0".to_string(),
                is_path: false,
                kind: DependencyKind::Normal,
            },
        );

        // trueno -> presentar (DEV dependency - for testing only)
        graph.add_dependency(
            "trueno",
            "presentar",
            DependencyEdge {
                version_req: "^0.1".to_string(),
                is_path: false,
                kind: DependencyKind::Dev,
            },
        );

        // ACT & ASSERT: Should NOT have cycles when excluding dev deps
        assert!(
            !graph.has_cycles(),
            "Dev dependencies should not create cycles"
        );

        // Topological order should work
        let order = graph.topological_order();
        assert!(
            order.is_ok(),
            "Should compute topological order: {:?}",
            order.err()
        );

        // trueno should come before presentar
        let order = order.unwrap();
        let trueno_pos = order.iter().position(|n| n == "trueno").unwrap();
        let presentar_pos = order.iter().position(|n| n == "presentar").unwrap();
        assert!(
            trueno_pos < presentar_pos,
            "trueno should be released before presentar"
        );
    }

    /// RED PHASE: Real cycles (normal deps) should still be detected
    #[test]
    fn test_ISSUE_13_real_cycle_still_detected() {
        // ARRANGE: Create actual cycle with normal dependencies
        let mut graph = DependencyGraph::new();

        graph.add_crate(CrateInfo::new(
            "a",
            semver::Version::new(1, 0, 0),
            std::path::PathBuf::new(),
        ));
        graph.add_crate(CrateInfo::new(
            "b",
            semver::Version::new(1, 0, 0),
            std::path::PathBuf::new(),
        ));

        // a -> b (normal)
        graph.add_dependency(
            "a",
            "b",
            DependencyEdge {
                version_req: "1.0".to_string(),
                is_path: false,
                kind: DependencyKind::Normal,
            },
        );

        // b -> a (normal) - REAL CYCLE!
        graph.add_dependency(
            "b",
            "a",
            DependencyEdge {
                version_req: "1.0".to_string(),
                is_path: false,
                kind: DependencyKind::Normal,
            },
        );

        // ACT & ASSERT: Should detect this real cycle
        assert!(graph.has_cycles(), "Real cycles should still be detected");
        assert!(graph.topological_order().is_err());
    }

    /// RED PHASE: Build dependencies should also be considered for cycles
    #[test]
    fn test_ISSUE_13_build_dep_creates_cycle() {
        // Build deps are needed at compile time, so they create real cycles
        let mut graph = DependencyGraph::new();

        graph.add_crate(CrateInfo::new(
            "a",
            semver::Version::new(1, 0, 0),
            std::path::PathBuf::new(),
        ));
        graph.add_crate(CrateInfo::new(
            "b",
            semver::Version::new(1, 0, 0),
            std::path::PathBuf::new(),
        ));

        // a -> b (normal)
        graph.add_dependency(
            "a",
            "b",
            DependencyEdge {
                version_req: "1.0".to_string(),
                is_path: false,
                kind: DependencyKind::Normal,
            },
        );

        // b -> a (build) - Build deps are needed at compile time
        graph.add_dependency(
            "b",
            "a",
            DependencyEdge {
                version_req: "1.0".to_string(),
                is_path: false,
                kind: DependencyKind::Build,
            },
        );

        // ACT & ASSERT: Build dependency cycle should be detected
        assert!(
            graph.has_cycles(),
            "Build dependency cycles should be detected"
        );
    }
}
