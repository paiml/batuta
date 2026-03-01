// CLI binary is only available for native targets (not WASM)
#![cfg(feature = "native")]
#![allow(dead_code)]
#![allow(unused_imports)]

mod analyzer;
mod ansi_colors;
mod backend;
mod bug_hunter;
mod cli;
mod comply;
mod config;
mod content;
mod data;
mod experiment;
mod hf;
mod numpy_converter;
mod oracle;
mod pacha;
mod parf;
mod pipeline;
mod pipeline_analysis;
mod playbook;
mod pytorch_converter;
mod report;
mod sklearn_converter;
mod stack;
mod timing;
mod tools;
mod types;
mod viz;

// Split modules for QA-002 compliance (<=500 lines per file)
#[path = "main_cli.rs"]
mod main_cli;
#[path = "main_dispatch.rs"]
mod main_dispatch;
#[path = "main_drift.rs"]
mod main_drift;
#[path = "main_oracle_args.rs"]
mod main_oracle_args;
#[path = "main_oracle_dispatch.rs"]
mod main_oracle_dispatch;

use clap::Parser;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn main() -> anyhow::Result<()> {
    let cli = main_cli::Cli::parse();

    // Initialize tracing
    let filter_layer = if cli.debug {
        tracing_subscriber::EnvFilter::new("debug")
    } else if cli.verbose {
        tracing_subscriber::EnvFilter::new("info")
    } else {
        tracing_subscriber::EnvFilter::new("warn")
    };

    tracing_subscriber::registry()
        .with(filter_layer)
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Batuta v{}", env!("CARGO_PKG_VERSION"));

    // --allow-drift or --unsafe-skip-drift-check skip the check entirely
    if !cli.unsafe_skip_drift_check && !cli.allow_drift {
        main_drift::enforce_drift_check(cli.strict)?;
    }

    main_dispatch::dispatch_command(cli.command)
}
