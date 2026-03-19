//! Banco TCP server binding.

use super::router::create_banco_router;
use super::state::BancoState;
use tokio::net::TcpListener;

/// Start the Banco HTTP server.
pub async fn start_server(host: &str, port: u16, state: BancoState) -> anyhow::Result<()> {
    let addr = format!("{host}:{port}");
    let listener = TcpListener::bind(&addr).await?;

    eprintln!("┌─────────────────────────────────────────────┐");
    eprintln!("│  Banco – Local AI Workbench                 │");
    eprintln!("├─────────────────────────────────────────────┤");
    eprintln!("│  GET  /health                               │");
    eprintln!("│  GET  /api/v1/models                        │");
    eprintln!("│  POST /api/v1/chat/completions              │");
    eprintln!("│  GET  /api/v1/system                        │");
    eprintln!("├─────────────────────────────────────────────┤");
    eprintln!("│  Privacy: {:?}", state.privacy_tier);
    eprintln!("│  Listening: {addr}");
    eprintln!("└─────────────────────────────────────────────┘");

    let app = create_banco_router(state);
    axum::serve(listener, app).await?;
    Ok(())
}
