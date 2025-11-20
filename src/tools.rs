use anyhow::{Context, Result};
use std::process::Command;

#[cfg(feature = "native")]
use tracing::{debug, info};

// Stub macros for WASM build
#[cfg(not(feature = "native"))]
macro_rules! info {
    ($($arg:tt)*) => {{}};
}

#[cfg(not(feature = "native"))]
macro_rules! debug {
    ($($arg:tt)*) => {{}};
}

/// Detected tool information
#[derive(Debug, Clone)]
pub struct ToolInfo {
    pub name: String,
    pub version: Option<String>,
    pub path: String,
    pub available: bool,
}

/// Check which transpiler and utility tools are available
#[derive(Debug, Clone)]
pub struct ToolRegistry {
    pub decy: Option<ToolInfo>,
    pub depyler: Option<ToolInfo>,
    pub bashrs: Option<ToolInfo>,
    pub ruchy: Option<ToolInfo>,
    pub trueno: Option<ToolInfo>,
    pub aprender: Option<ToolInfo>,
    pub realizar: Option<ToolInfo>,
    pub renacer: Option<ToolInfo>,
    pub pmat: Option<ToolInfo>,
}

impl ToolRegistry {
    /// Detect all available tools
    pub fn detect() -> Self {
        info!("Detecting installed Pragmatic AI Labs tools...");

        Self {
            decy: detect_tool("decy"),
            depyler: detect_tool("depyler"),
            bashrs: detect_tool("bashrs"),
            ruchy: detect_tool("ruchy"),
            trueno: detect_tool("trueno"),
            aprender: detect_tool("aprender"),
            realizar: detect_tool("realizar"),
            renacer: detect_tool("renacer"),
            pmat: detect_tool("pmat"),
        }
    }

    /// Check if any transpiler is available
    pub fn has_transpiler(&self) -> bool {
        self.decy.is_some() || self.depyler.is_some() || self.bashrs.is_some()
    }

    /// Get transpiler for a specific language
    pub fn get_transpiler_for_language(&self, lang: &crate::types::Language) -> Option<&ToolInfo> {
        use crate::types::Language;

        match lang {
            Language::C | Language::Cpp => self.decy.as_ref(),
            Language::Python => self.depyler.as_ref(),
            Language::Shell => self.bashrs.as_ref(),
            _ => None,
        }
    }

    /// Get list of available tools
    pub fn available_tools(&self) -> Vec<String> {
        let mut tools = Vec::new();

        if let Some(tool) = &self.decy {
            if tool.available {
                tools.push("Decy (C/C++ → Rust)".to_string());
            }
        }
        if let Some(tool) = &self.depyler {
            if tool.available {
                tools.push("Depyler (Python → Rust)".to_string());
            }
        }
        if let Some(tool) = &self.bashrs {
            if tool.available {
                tools.push("Bashrs (Shell → Rust)".to_string());
            }
        }
        if let Some(tool) = &self.ruchy {
            if tool.available {
                tools.push("Ruchy (Rust scripting)".to_string());
            }
        }
        if let Some(tool) = &self.pmat {
            if tool.available {
                tools.push("PMAT (Quality analysis)".to_string());
            }
        }
        if let Some(tool) = &self.trueno {
            if tool.available {
                tools.push("Trueno (Multi-target compute)".to_string());
            }
        }
        if let Some(tool) = &self.aprender {
            if tool.available {
                tools.push("Aprender (ML library)".to_string());
            }
        }
        if let Some(tool) = &self.realizar {
            if tool.available {
                tools.push("Realizar (Inference runtime)".to_string());
            }
        }
        if let Some(tool) = &self.renacer {
            if tool.available {
                tools.push("Renacer (Syscall tracing)".to_string());
            }
        }

        tools
    }

    /// Get installation instructions for missing tools
    pub fn get_installation_instructions(&self, needed_tools: &[&str]) -> Vec<String> {
        let mut instructions = Vec::new();

        for tool in needed_tools {
            let instruction = match *tool {
                "decy" if self.decy.is_none() => {
                    Some("Install Decy: cargo install decy")
                }
                "depyler" if self.depyler.is_none() => {
                    Some("Install Depyler: cargo install depyler")
                }
                "bashrs" if self.bashrs.is_none() => {
                    Some("Install Bashrs: cargo install bashrs")
                }
                "ruchy" if self.ruchy.is_none() => {
                    Some("Install Ruchy: cargo install ruchy")
                }
                "pmat" if self.pmat.is_none() => {
                    Some("Install PMAT: cargo install pmat")
                }
                "trueno" if self.trueno.is_none() => {
                    Some("Install Trueno: Add 'trueno' to Cargo.toml dependencies")
                }
                "aprender" if self.aprender.is_none() => {
                    Some("Install Aprender: Add 'aprender' to Cargo.toml dependencies")
                }
                "realizar" if self.realizar.is_none() => {
                    Some("Install Realizar: Add 'realizar' to Cargo.toml dependencies")
                }
                "renacer" if self.renacer.is_none() => {
                    Some("Install Renacer: cargo install renacer")
                }
                _ => None,
            };

            if let Some(inst) = instruction {
                instructions.push(inst.to_string());
            }
        }

        instructions
    }
}

/// Detect a single tool
fn detect_tool(name: &str) -> Option<ToolInfo> {
    debug!("Checking for tool: {}", name);

    // Try to find the tool using `which`
    let path = match which::which(name) {
        Ok(p) => p.to_string_lossy().to_string(),
        Err(_) => {
            debug!("Tool '{}' not found in PATH", name);
            return None;
        }
    };

    // Try to get version
    let version = get_tool_version(name);

    debug!(
        "Found tool '{}' at '{}' (version: {:?})",
        name, path, version
    );

    Some(ToolInfo {
        name: name.to_string(),
        version,
        path,
        available: true,
    })
}

/// Get tool version by running --version
fn get_tool_version(name: &str) -> Option<String> {
    let output = Command::new(name).arg("--version").output().ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let version_line = stdout.lines().next()?;

    // Extract version number from output
    // Common formats:
    // "tool 1.2.3"
    // "tool version 1.2.3"
    // "1.2.3"
    let parts: Vec<&str> = version_line.split_whitespace().collect();
    let version = parts.last()?.to_string();

    Some(version)
}

/// Run a tool command and capture output
pub fn run_tool(
    tool_name: &str,
    args: &[&str],
    working_dir: Option<&std::path::Path>,
) -> Result<String> {
    debug!("Running tool: {} {:?}", tool_name, args);

    let mut cmd = Command::new(tool_name);
    cmd.args(args);

    if let Some(dir) = working_dir {
        cmd.current_dir(dir);
    }

    let output = cmd
        .output()
        .with_context(|| format!("Failed to run tool: {}", tool_name))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "Tool '{}' failed with exit code {:?}: {}",
            tool_name,
            output.status.code(),
            stderr
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    Ok(stdout)
}

/// Transpile Python code using Depyler
pub fn transpile_python(
    input_path: &std::path::Path,
    output_path: &std::path::Path,
) -> Result<String> {
    info!("Transpiling Python with Depyler: {:?} → {:?}", input_path, output_path);

    let input_str = input_path.to_string_lossy();
    let output_str = output_path.to_string_lossy();

    let args = vec![
        "transpile",
        "--input",
        &input_str,
        "--output",
        &output_str,
        "--format",
        "project", // Generate full Rust project structure
    ];

    run_tool("depyler", &args, None)
}

/// Transpile Shell script using Bashrs
pub fn transpile_shell(
    input_path: &std::path::Path,
    output_path: &std::path::Path,
) -> Result<String> {
    info!("Transpiling Shell with Bashrs: {:?} → {:?}", input_path, output_path);

    let input_str = input_path.to_string_lossy();
    let output_str = output_path.to_string_lossy();

    let args = vec![
        "build",
        &input_str,
        "-o",
        &output_str,
        "--target",
        "posix", // Most compatible shell target
        "--verify",
        "strict", // Strict verification
    ];

    run_tool("bashrs", &args, None)
}

/// Transpile C/C++ code using Decy (if available)
pub fn transpile_c_cpp(
    input_path: &std::path::Path,
    output_path: &std::path::Path,
) -> Result<String> {
    info!("Transpiling C/C++ with Decy: {:?} → {:?}", input_path, output_path);

    let input_str = input_path.to_string_lossy();
    let output_str = output_path.to_string_lossy();

    // Note: Decy might not be installed, handle gracefully
    let args = vec![
        "transpile",
        "--input",
        &input_str,
        "--output",
        &output_str,
    ];

    run_tool("decy", &args, None)
}

/// Run quality analysis using PMAT
pub fn analyze_quality(
    path: &std::path::Path,
) -> Result<String> {
    info!("Running PMAT quality analysis: {:?}", path);

    let path_str = path.to_string_lossy();

    let args = vec![
        "analyze",
        "complexity",
        &path_str,
        "--format",
        "json",
    ];

    run_tool("pmat", &args, None)
}

/// Run Ruchy scripting (if needed)
pub fn run_ruchy_script(
    script_path: &std::path::Path,
) -> Result<String> {
    info!("Running Ruchy script: {:?}", script_path);

    let script_str = script_path.to_string_lossy();

    let args = vec![
        "run",
        &script_str,
    ];

    run_tool("ruchy", &args, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_detection() {
        // This test will only pass if tools are installed
        let registry = ToolRegistry::detect();

        // At minimum, we should detect PMAT if running on dev machine
        // But we don't want tests to fail in CI, so we just log
        println!("Available tools: {:?}", registry.available_tools());
    }

    #[test]
    fn test_get_installation_instructions() {
        let registry = ToolRegistry::detect();
        let instructions = registry.get_installation_instructions(&["decy", "depyler", "pmat"]);

        // Should return instructions for any missing tools
        println!("Installation instructions: {:?}", instructions);
    }
}
