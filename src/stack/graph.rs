#![allow(dead_code)]
//! Dependency Graph Analysis
//!
//! Uses trueno-graph for cycle detection and topological sorting.
//! Edge metadata stored separately since trueno-graph only stores f32 weights.

use crate::stack::is_paiml_crate;
use crate::stack::types::*;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::Path;
use trueno_graph::{is_cyclic, toposort, CsrGraph, NodeId};

/// Dependency graph for the PAIML stack
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    /// The underlying trueno-graph CSR graph
    graph: CsrGraph,

    /// Map from crate name to node ID
    name_to_id: HashMap<String, NodeId>,

    /// Map from node ID to crate name
    id_to_name: HashMap<NodeId, String>,

    /// Edge metadata (not stored in trueno-graph which only has f32 weights)
    edge_data: HashMap<(NodeId, NodeId), DependencyEdge>,

    /// Crate information for each node
    crate_info: HashMap<String, CrateInfo>,

    /// Next node ID to assign
    next_id: u32,
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
            graph: CsrGraph::new(),
            name_to_id: HashMap::new(),
            id_to_name: HashMap::new(),
            edge_data: HashMap::new(),
            crate_info: HashMap::new(),
            next_id: 0,
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
        if !self.name_to_id.contains_key(&name) {
            let id = NodeId(self.next_id);
            self.next_id += 1;
            self.name_to_id.insert(name.clone(), id);
            self.id_to_name.insert(id, name.clone());
            self.graph.set_node_name(id, name.clone());
        }
        self.crate_info.insert(name, info);
    }

    /// Ensure a node exists for a crate name
    fn ensure_node(&mut self, name: &str) -> NodeId {
        if let Some(&id) = self.name_to_id.get(name) {
            id
        } else {
            let id = NodeId(self.next_id);
            self.next_id += 1;
            self.name_to_id.insert(name.to_string(), id);
            self.id_to_name.insert(id, name.to_string());
            self.graph.set_node_name(id, name.to_string());
            id
        }
    }

    /// Add a dependency edge between crates
    pub fn add_dependency(&mut self, from: &str, to: &str, edge: DependencyEdge) {
        let from_id = self.ensure_node(from);
        let to_id = self.ensure_node(to);

        // Store edge metadata
        self.edge_data.insert((from_id, to_id), edge);

        // Add edge to trueno-graph (weight = 1.0)
        let _ = self.graph.add_edge(from_id, to_id, 1.0);
    }

    /// Build a release graph (excludes dev-dependencies)
    fn build_release_graph(&self) -> CsrGraph {
        let mut edges: Vec<(NodeId, NodeId, f32)> = Vec::new();

        // Collect non-dev edges
        for ((from_id, to_id), edge_data) in &self.edge_data {
            if edge_data.kind != DependencyKind::Dev {
                edges.push((*from_id, *to_id, 1.0));
            }
        }

        CsrGraph::from_edge_list(&edges).unwrap_or_else(|_| CsrGraph::new())
    }

    /// Check if the graph has cycles (excluding dev-dependencies)
    ///
    /// Dev-dependencies are excluded because they don't create real dependency
    /// cycles for release ordering. A crate can have a dev-dependency on another
    /// crate that depends on it (common for integration testing).
    pub fn has_cycles(&self) -> bool {
        let release_graph = self.build_release_graph();
        is_cyclic(&release_graph)
    }

    /// Get topological order for releases (dependencies first)
    ///
    /// Uses the release graph which excludes dev-dependencies to avoid
    /// false cycle detection (fixes issue #13).
    pub fn topological_order(&self) -> Result<Vec<String>> {
        let release_graph = self.build_release_graph();

        if is_cyclic(&release_graph) {
            return Err(anyhow!("Circular dependency detected in the graph"));
        }

        let sorted =
            toposort(&release_graph).map_err(|_| anyhow!("Failed to compute topological order"))?;

        // Map NodeIds back to names and reverse
        // Reverse because toposort gives dependents first, we want dependencies first
        let names: Vec<String> = sorted
            .iter()
            .rev() // Dependencies should come before dependents
            .filter_map(|id| self.id_to_name.get(id).cloned())
            .collect();

        Ok(names)
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

        if let Some(&id) = self.name_to_id.get(crate_name) {
            self.collect_dependencies(id, &mut deps, &mut visited);
        }

        deps
    }

    fn collect_dependencies(
        &self,
        id: NodeId,
        deps: &mut Vec<String>,
        visited: &mut std::collections::HashSet<NodeId>,
    ) {
        if visited.contains(&id) {
            return;
        }
        visited.insert(id);

        // Get outgoing neighbors from trueno-graph
        if let Ok(neighbors) = self.graph.outgoing_neighbors(id) {
            for &neighbor_id in neighbors {
                let neighbor = NodeId(neighbor_id);
                if let Some(name) = self.id_to_name.get(&neighbor) {
                    if !deps.contains(name) {
                        deps.push(name.clone());
                    }
                    self.collect_dependencies(neighbor, deps, visited);
                }
            }
        }
    }

    /// Get crates that depend on this crate (reverse dependencies)
    pub fn dependents(&self, crate_name: &str) -> Vec<String> {
        let mut dependents = Vec::new();

        if let Some(&id) = self.name_to_id.get(crate_name) {
            if let Ok(neighbors) = self.graph.incoming_neighbors(id) {
                for &neighbor_id in neighbors {
                    if let Some(name) = self.id_to_name.get(&NodeId(neighbor_id)) {
                        dependents.push(name.clone());
                    }
                }
            }
        }

        dependents
    }

    /// Get all path dependencies in the graph
    pub fn find_path_dependencies(&self) -> Vec<PathDependencyIssue> {
        let mut issues = Vec::new();

        for ((from_id, to_id), edge_data) in &self.edge_data {
            if edge_data.is_path {
                if let (Some(from), Some(to)) =
                    (self.id_to_name.get(from_id), self.id_to_name.get(to_id))
                {
                    issues.push(PathDependencyIssue {
                        crate_name: from.clone(),
                        dependency: to.clone(),
                        current: format!("path = \"../{}\"", to),
                        recommended: None, // Will be filled in by checker
                    });
                }
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

    /// Check if a node exists (for compatibility with tests)
    #[allow(dead_code)]
    pub(crate) fn node_indices_contains(&self, name: &str) -> bool {
        self.name_to_id.contains_key(name)
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

    // =========================================================================
    // Test Helpers: Reduce structural repetition (PMAT entropy)
    // =========================================================================

    /// Add a test crate with the given name and version to the graph.
    /// Uses an empty PathBuf since test crates don't need real paths.
    fn add_test_crate(graph: &mut DependencyGraph, name: &str, major: u64, minor: u64, patch: u64) {
        graph.add_crate(CrateInfo::new(
            name,
            semver::Version::new(major, minor, patch),
            std::path::PathBuf::new(),
        ));
    }

    /// Add a normal (non-path) dependency edge between two crates.
    fn add_normal_dep(graph: &mut DependencyGraph, from: &str, to: &str, version: &str) {
        graph.add_dependency(
            from,
            to,
            DependencyEdge {
                version_req: version.to_string(),
                is_path: false,
                kind: DependencyKind::Normal,
            },
        );
    }

    /// Add a dev dependency edge between two crates.
    fn add_dev_dep(graph: &mut DependencyGraph, from: &str, to: &str, version: &str) {
        graph.add_dependency(
            from,
            to,
            DependencyEdge {
                version_req: version.to_string(),
                is_path: false,
                kind: DependencyKind::Dev,
            },
        );
    }

    /// Add a path dependency edge between two crates.
    fn add_path_dep(graph: &mut DependencyGraph, from: &str, to: &str) {
        graph.add_dependency(
            from,
            to,
            DependencyEdge {
                version_req: String::new(),
                is_path: true,
                kind: DependencyKind::Normal,
            },
        );
    }

    /// Assert that a list of crate names contains the given name.
    fn assert_contains_crate(list: &[String], name: &str) {
        assert!(
            list.contains(&name.to_string()),
            "Expected list to contain '{}', but it was not found in: {:?}",
            name,
            list
        );
    }

    fn create_test_graph() -> DependencyGraph {
        let mut graph = DependencyGraph::new();

        // Create crates (uses PathBuf::from for create_test_graph to preserve original paths)
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
        add_normal_dep(&mut graph, "aprender", "trueno", "^1.0");
        add_normal_dep(&mut graph, "entrenar", "aprender", "^0.8");
        add_path_dep(&mut graph, "entrenar", "alimentar");
        add_normal_dep(&mut graph, "alimentar", "trueno", "^1.0");

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
        add_test_crate(&mut graph, "trueno", 1, 0, 0);

        assert_eq!(graph.crate_count(), 1);
        assert!(graph.get_crate("trueno").is_some());
        assert!(graph.get_crate("serde").is_none());
    }

    #[test]
    fn test_dependency_edges() {
        let graph = create_test_graph();

        // Check dependencies
        let aprender_deps = graph.all_dependencies("aprender");
        assert_contains_crate(&aprender_deps, "trueno");

        let entrenar_deps = graph.all_dependencies("entrenar");
        assert_contains_crate(&entrenar_deps, "aprender");
        assert_contains_crate(&entrenar_deps, "alimentar");
        assert_contains_crate(&entrenar_deps, "trueno"); // transitive
    }

    #[test]
    fn test_no_cycles() {
        let graph = create_test_graph();
        assert!(!graph.has_cycles());
    }

    #[test]
    fn test_cycle_detection() {
        let mut graph = DependencyGraph::new();

        add_test_crate(&mut graph, "a", 1, 0, 0);
        add_test_crate(&mut graph, "b", 1, 0, 0);
        add_test_crate(&mut graph, "c", 1, 0, 0);

        // Create cycle: a -> b -> c -> a
        add_normal_dep(&mut graph, "a", "b", "1.0");
        add_normal_dep(&mut graph, "b", "c", "1.0");
        add_normal_dep(&mut graph, "c", "a", "1.0");

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
        assert_contains_crate(&order, "trueno");
        assert_contains_crate(&order, "aprender");
        assert_contains_crate(&order, "alimentar");
        assert_contains_crate(&order, "entrenar");

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
        assert_contains_crate(&trueno_dependents, "aprender");
        assert_contains_crate(&trueno_dependents, "alimentar");

        // aprender has dependent: entrenar
        let aprender_dependents = graph.dependents("aprender");
        assert_contains_crate(&aprender_dependents, "entrenar");
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
    fn test_issue_13_dev_dependency_not_cycle() {
        // ARRANGE: Create graph where:
        // - presentar depends on trueno (normal)
        // - trueno has dev-dependency on presentar (for testing)
        // This is NOT a real cycle for release purposes
        let mut graph = DependencyGraph::new();

        add_test_crate(&mut graph, "trueno", 1, 0, 0);
        add_test_crate(&mut graph, "presentar", 0, 1, 0);

        // presentar -> trueno (normal dependency)
        add_normal_dep(&mut graph, "presentar", "trueno", "^1.0");

        // trueno -> presentar (DEV dependency - for testing only)
        add_dev_dep(&mut graph, "trueno", "presentar", "^0.1");

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

    /// RED PHASE: Multiple dev dependencies should not create cycle
    #[test]
    fn test_multiple_dev_deps_no_cycle() {
        let mut graph = DependencyGraph::new();

        add_test_crate(&mut graph, "a", 1, 0, 0);
        add_test_crate(&mut graph, "b", 1, 0, 0);
        add_test_crate(&mut graph, "c", 1, 0, 0);

        // a -> b (normal)
        add_normal_dep(&mut graph, "a", "b", "^1.0");

        // b -> a (dev)
        add_dev_dep(&mut graph, "b", "a", "^1.0");

        // c -> a (dev)
        add_dev_dep(&mut graph, "c", "a", "^1.0");

        assert!(!graph.has_cycles());
        let order = graph.topological_order();
        assert!(order.is_ok());
    }

    /// Test graph with build dependencies
    #[test]
    fn test_build_dependencies() {
        let mut graph = DependencyGraph::new();

        add_test_crate(&mut graph, "main", 1, 0, 0);
        add_test_crate(&mut graph, "build-dep", 1, 0, 0);

        graph.add_dependency(
            "main",
            "build-dep",
            DependencyEdge {
                version_req: "^1.0".to_string(),
                is_path: false,
                kind: DependencyKind::Build,
            },
        );

        // Build deps should be in the graph
        let deps = graph.all_dependencies("main");
        assert!(deps.iter().any(|d| d == "build-dep"));
    }

    /// Test graph removal of crate
    #[test]
    fn test_graph_get_crate() {
        let mut graph = DependencyGraph::new();
        add_test_crate(&mut graph, "test", 1, 0, 0);

        let crate_info = graph.get_crate("test");
        assert!(crate_info.is_some());
        assert_eq!(crate_info.unwrap().name, "test");

        let missing = graph.get_crate("nonexistent");
        assert!(missing.is_none());
    }

    /// Test graph contains
    #[test]
    fn test_graph_contains() {
        let mut graph = DependencyGraph::new();
        add_test_crate(&mut graph, "exists", 1, 0, 0);

        assert!(graph.get_crate("exists").is_some());
        assert!(graph.get_crate("not-exists").is_none());
    }

    /// Test graph all_crates iteration
    #[test]
    fn test_graph_all_crates() {
        let mut graph = DependencyGraph::new();
        add_test_crate(&mut graph, "a", 1, 0, 0);
        add_test_crate(&mut graph, "b", 2, 0, 0);

        let crates: Vec<_> = graph.all_crates().collect();
        assert_eq!(crates.len(), 2);
    }

    /// Test crate count
    #[test]
    fn test_graph_crate_count() {
        let mut graph = DependencyGraph::new();
        assert_eq!(graph.crate_count(), 0);

        add_test_crate(&mut graph, "test", 1, 0, 0);
        assert_eq!(graph.crate_count(), 1);
    }

    /// Test graph with no edges
    #[test]
    fn test_graph_no_dependencies() {
        let mut graph = DependencyGraph::new();
        add_test_crate(&mut graph, "lone", 1, 0, 0);

        let deps = graph.all_dependencies("lone");
        assert!(deps.is_empty());

        let dependents = graph.dependents("lone");
        assert!(dependents.is_empty());
    }

    /// Test get_crate_mut
    #[test]
    fn test_graph_get_crate_mut() {
        let mut graph = DependencyGraph::new();
        add_test_crate(&mut graph, "mutable", 1, 0, 0);

        if let Some(crate_info) = graph.get_crate_mut("mutable") {
            crate_info.status = CrateStatus::Healthy;
        }

        assert_eq!(
            graph.get_crate("mutable").unwrap().status,
            CrateStatus::Healthy
        );
    }

    /// RED PHASE: Real cycles (normal deps) should still be detected
    #[test]
    fn test_issue_13_real_cycle_still_detected() {
        // ARRANGE: Create actual cycle with normal dependencies
        let mut graph = DependencyGraph::new();

        add_test_crate(&mut graph, "a", 1, 0, 0);
        add_test_crate(&mut graph, "b", 1, 0, 0);

        // a -> b (normal)
        add_normal_dep(&mut graph, "a", "b", "1.0");

        // b -> a (normal) - REAL CYCLE!
        add_normal_dep(&mut graph, "b", "a", "1.0");

        // ACT & ASSERT: Should detect this real cycle
        assert!(graph.has_cycles(), "Real cycles should still be detected");
        assert!(graph.topological_order().is_err());
    }

    /// RED PHASE: Build dependencies should also be considered for cycles
    #[test]
    fn test_issue_13_build_dep_creates_cycle() {
        // Build deps are needed at compile time, so they create real cycles
        let mut graph = DependencyGraph::new();

        add_test_crate(&mut graph, "a", 1, 0, 0);
        add_test_crate(&mut graph, "b", 1, 0, 0);

        // a -> b (normal)
        add_normal_dep(&mut graph, "a", "b", "1.0");

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

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_graph_cov_001_all_deps_missing_crate() {
        let graph = DependencyGraph::new();
        let deps = graph.all_dependencies("nonexistent");
        assert!(deps.is_empty());
    }

    #[test]
    fn test_graph_cov_002_dependents_missing_crate() {
        let graph = DependencyGraph::new();
        let deps = graph.dependents("nonexistent");
        assert!(deps.is_empty());
    }

    #[test]
    fn test_graph_cov_003_release_order_missing_crate() {
        let graph = DependencyGraph::new();
        // Should still work with empty graph
        let order = graph.topological_order();
        assert!(order.is_ok());
        assert!(order.unwrap().is_empty());
    }

    #[test]
    fn test_graph_cov_004_dependency_edge_debug() {
        let edge = DependencyEdge {
            version_req: "^1.0".to_string(),
            is_path: true,
            kind: DependencyKind::Normal,
        };
        let debug = format!("{:?}", edge);
        assert!(debug.contains("DependencyEdge"));
        assert!(debug.contains("is_path: true"));
    }

    #[test]
    fn test_graph_cov_005_dependency_edge_clone() {
        let edge = DependencyEdge {
            version_req: "^1.0".to_string(),
            is_path: false,
            kind: DependencyKind::Dev,
        };
        let cloned = edge.clone();
        assert_eq!(cloned.version_req, edge.version_req);
        assert_eq!(cloned.is_path, edge.is_path);
    }

    #[test]
    fn test_graph_cov_006_path_dep_issue_debug() {
        let issue = PathDependencyIssue {
            crate_name: "test".to_string(),
            dependency: "dep".to_string(),
            current: "path = \"../dep\"".to_string(),
            recommended: Some("1.0.0".to_string()),
        };
        let debug = format!("{:?}", issue);
        assert!(debug.contains("PathDependencyIssue"));
        assert!(debug.contains("test"));
    }

    #[test]
    fn test_graph_cov_007_path_dep_issue_clone() {
        let issue = PathDependencyIssue {
            crate_name: "test".to_string(),
            dependency: "dep".to_string(),
            current: "path = \"../dep\"".to_string(),
            recommended: None,
        };
        let cloned = issue.clone();
        assert_eq!(cloned.crate_name, issue.crate_name);
        assert!(cloned.recommended.is_none());
    }

    #[test]
    fn test_graph_cov_008_graph_debug() {
        let graph = DependencyGraph::new();
        let debug = format!("{:?}", graph);
        assert!(debug.contains("DependencyGraph"));
    }

    #[test]
    fn test_graph_cov_009_graph_clone() {
        let mut graph = DependencyGraph::new();
        add_test_crate(&mut graph, "test", 1, 0, 0);
        let cloned = graph.clone();
        assert_eq!(cloned.crate_count(), 1);
    }

    #[test]
    fn test_graph_cov_010_default() {
        let graph = DependencyGraph::default();
        assert_eq!(graph.crate_count(), 0);
    }

    #[test]
    fn test_graph_cov_011_add_dep_creates_nodes() {
        let mut graph = DependencyGraph::new();
        // Add dependency without first adding crates
        add_normal_dep(&mut graph, "new_from", "new_to", "1.0");

        // Both nodes should be created
        assert!(graph.node_indices_contains("new_from"));
        assert!(graph.node_indices_contains("new_to"));
    }

    #[test]
    fn test_graph_cov_012_add_crate_duplicate() {
        let mut graph = DependencyGraph::new();
        add_test_crate(&mut graph, "dup", 1, 0, 0);
        // Add again - should update, not duplicate
        add_test_crate(&mut graph, "dup", 2, 0, 0);
        assert_eq!(graph.crate_count(), 1);
        assert_eq!(
            graph.get_crate("dup").unwrap().local_version,
            semver::Version::new(2, 0, 0)
        );
    }

    #[test]
    fn test_graph_cov_013_release_order_for_leaf() {
        let graph = create_test_graph();
        // trueno is a leaf node (no dependencies)
        let order = graph.release_order_for("trueno").unwrap();
        assert_eq!(order.len(), 1);
        assert_eq!(order[0], "trueno");
    }

    #[test]
    fn test_graph_cov_014_no_path_deps() {
        let mut graph = DependencyGraph::new();
        add_test_crate(&mut graph, "a", 1, 0, 0);
        add_test_crate(&mut graph, "b", 1, 0, 0);
        add_normal_dep(&mut graph, "a", "b", "1.0");

        let path_deps = graph.find_path_dependencies();
        assert!(path_deps.is_empty());
    }

    #[test]
    fn test_graph_cov_015_detect_conflicts_no_deps() {
        let mut graph = DependencyGraph::new();
        add_test_crate(&mut graph, "empty", 1, 0, 0);

        let conflicts = graph.detect_conflicts();
        assert!(conflicts.is_empty());
    }

    #[test]
    fn test_graph_cov_016_single_usage_no_conflict() {
        let mut graph = DependencyGraph::new();
        let mut crate_a = CrateInfo::new(
            "a",
            semver::Version::new(1, 0, 0),
            std::path::PathBuf::new(),
        );
        crate_a
            .external_dependencies
            .push(DependencyInfo::new("serde", "1.0"));
        graph.add_crate(crate_a);

        let conflicts = graph.detect_conflicts();
        assert!(conflicts.is_empty()); // Single usage = no conflict
    }

    #[test]
    fn test_graph_cov_017_dependency_kind_variants() {
        // Test all DependencyKind variants
        let normal = DependencyKind::Normal;
        let dev = DependencyKind::Dev;
        let build = DependencyKind::Build;

        assert!(matches!(normal, DependencyKind::Normal));
        assert!(matches!(dev, DependencyKind::Dev));
        assert!(matches!(build, DependencyKind::Build));
    }
}
