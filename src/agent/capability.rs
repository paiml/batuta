//! Capability-based access control for agent tools.
//!
//! Implements the Poka-Yoke (mistake-proofing) pattern from Toyota
//! Production System. Each tool declares its required capability;
//! the agent manifest grants capabilities. Mismatch → denied.
//!
//! See: arXiv:2406.09187 (GuardAgent), arXiv:2509.22256 (access control).

use serde::{Deserialize, Serialize};

/// Capability grants for tools (Poka-Yoke pattern).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Capability {
    /// Access RAG pipeline for document retrieval.
    Rag,
    /// Read/write agent memory.
    Memory,
    /// Execute shell commands (sandboxed).
    Shell {
        /// Allowed command prefixes. `["*"]` allows all.
        allowed_commands: Vec<String>,
    },
    /// Launch headless browser via jugar-probar.
    Browser,
    /// Invoke sub-inference on a different model.
    Inference,
    /// Submit work to repartir compute pool.
    Compute,
    /// Network egress (blocked in Sovereign tier).
    Network {
        /// Allowed hostnames. `["*"]` allows all.
        allowed_hosts: Vec<String>,
    },
    /// MCP tool from external server (agents-mcp feature).
    Mcp {
        /// MCP server name.
        server: String,
        /// Tool name on that server. `"*"` matches all.
        tool: String,
    },
}

/// Check if granted capabilities satisfy a required capability.
///
/// Returns `true` if at least one granted capability matches the
/// required capability. Wildcard (`"*"`) matching is supported
/// for Shell commands, Network hosts, and MCP tools.
pub fn capability_matches(granted: &[Capability], required: &Capability) -> bool {
    granted.iter().any(|g| single_match(g, required))
}

fn single_match(granted: &Capability, required: &Capability) -> bool {
    match (granted, required) {
        (Capability::Rag, Capability::Rag)
        | (Capability::Memory, Capability::Memory)
        | (Capability::Browser, Capability::Browser)
        | (Capability::Inference, Capability::Inference)
        | (Capability::Compute, Capability::Compute) => true,

        (
            Capability::Shell {
                allowed_commands: g,
            },
            Capability::Shell {
                allowed_commands: r,
            },
        ) => r
            .iter()
            .all(|cmd| g.contains(cmd) || g.iter().any(|p| p == "*")),

        (
            Capability::Network { allowed_hosts: g },
            Capability::Network { allowed_hosts: r },
        ) => r
            .iter()
            .all(|h| g.contains(h) || g.iter().any(|p| p == "*")),

        (
            Capability::Mcp {
                server: gs,
                tool: gt,
            },
            Capability::Mcp {
                server: rs,
                tool: rt,
            },
        ) => (gs == rs || gs == "*") && (gt == rt || gt == "*"),

        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match_simple() {
        assert!(capability_matches(&[Capability::Rag], &Capability::Rag));
        assert!(capability_matches(
            &[Capability::Memory],
            &Capability::Memory
        ));
        assert!(capability_matches(
            &[Capability::Browser],
            &Capability::Browser
        ));
        assert!(capability_matches(
            &[Capability::Inference],
            &Capability::Inference
        ));
        assert!(capability_matches(
            &[Capability::Compute],
            &Capability::Compute
        ));
    }

    #[test]
    fn test_mismatch_denied() {
        assert!(!capability_matches(&[Capability::Rag], &Capability::Memory));
        assert!(!capability_matches(
            &[Capability::Browser],
            &Capability::Compute
        ));
        assert!(!capability_matches(&[], &Capability::Rag));
    }

    #[test]
    fn test_shell_wildcard() {
        let granted = Capability::Shell {
            allowed_commands: vec!["*".into()],
        };
        let required = Capability::Shell {
            allowed_commands: vec!["ls".into(), "cat".into()],
        };
        assert!(capability_matches(&[granted], &required));
    }

    #[test]
    fn test_shell_specific() {
        let granted = Capability::Shell {
            allowed_commands: vec!["ls".into()],
        };
        let required = Capability::Shell {
            allowed_commands: vec!["ls".into()],
        };
        assert!(capability_matches(&[granted.clone()], &required));

        let denied = Capability::Shell {
            allowed_commands: vec!["rm".into()],
        };
        assert!(!capability_matches(&[granted], &denied));
    }

    #[test]
    fn test_network_wildcard() {
        let granted = Capability::Network {
            allowed_hosts: vec!["*".into()],
        };
        let required = Capability::Network {
            allowed_hosts: vec!["api.example.com".into()],
        };
        assert!(capability_matches(&[granted], &required));
    }

    #[test]
    fn test_network_specific() {
        let granted = Capability::Network {
            allowed_hosts: vec!["localhost".into()],
        };
        let required = Capability::Network {
            allowed_hosts: vec!["localhost".into()],
        };
        assert!(capability_matches(&[granted.clone()], &required));

        let denied = Capability::Network {
            allowed_hosts: vec!["evil.com".into()],
        };
        assert!(!capability_matches(&[granted], &denied));
    }

    #[test]
    fn test_mcp_exact() {
        let granted = Capability::Mcp {
            server: "fs".into(),
            tool: "read".into(),
        };
        let required = Capability::Mcp {
            server: "fs".into(),
            tool: "read".into(),
        };
        assert!(capability_matches(&[granted], &required));
    }

    #[test]
    fn test_mcp_tool_wildcard() {
        let granted = Capability::Mcp {
            server: "fs".into(),
            tool: "*".into(),
        };
        let required = Capability::Mcp {
            server: "fs".into(),
            tool: "read".into(),
        };
        assert!(capability_matches(&[granted], &required));
    }

    #[test]
    fn test_mcp_server_mismatch() {
        let granted = Capability::Mcp {
            server: "fs".into(),
            tool: "*".into(),
        };
        let required = Capability::Mcp {
            server: "db".into(),
            tool: "query".into(),
        };
        assert!(!capability_matches(&[granted], &required));
    }

    #[test]
    fn test_multiple_granted_any_match() {
        let granted = vec![Capability::Rag, Capability::Memory, Capability::Browser];
        assert!(capability_matches(&granted, &Capability::Memory));
        assert!(!capability_matches(&granted, &Capability::Compute));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let caps = vec![
            Capability::Rag,
            Capability::Shell {
                allowed_commands: vec!["ls".into()],
            },
            Capability::Mcp {
                server: "s".into(),
                tool: "t".into(),
            },
        ];
        for cap in &caps {
            let json = serde_json::to_string(cap).expect("serialize failed");
            let back: Capability = serde_json::from_str(&json).expect("deserialize failed");
            assert_eq!(*cap, back);
        }
    }
}
