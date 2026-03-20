//! Banco TCP server binding.

use super::router::create_banco_router;
use super::state::BancoState;
use tokio::net::TcpListener;

/// Start the Banco HTTP server.
pub async fn start_server(host: &str, port: u16, state: BancoState) -> anyhow::Result<()> {
    let addr = format!("{host}:{port}");
    let listener = TcpListener::bind(&addr).await?;

    let model_status = if let Some(info) = state.model.info() {
        format!("{} ({})", info.model_id, format!("{:?}", info.format).to_lowercase())
    } else {
        "none (echo mode)".to_string()
    };

    eprintln!("┌──────────────────────────────────────────────────┐");
    eprintln!("│  Banco – Local AI Workbench                      │");
    eprintln!("├──────────────────────────────────────────────────┤");
    eprintln!("│  Listening:  {addr:<36}│");
    eprintln!("│  Privacy:    {:?}", state.privacy_tier);
    eprintln!("│  Model:      {model_status}");
    eprintln!("│  Telemetry:  disabled");
    eprintln!("├──────────────────────────────────────────────────┤");
    eprintln!("│  Core:       /health /api/v1/models /api/v1/system");
    eprintln!("│  Chat:       /api/v1/chat/completions (SSE)");
    eprintln!("│  Data:       /api/v1/tokenize /detokenize /embeddings");
    eprintln!("│  Models:     /api/v1/models/load|unload|status");
    eprintln!("│  Chat cfg:   /api/v1/chat/parameters");
    eprintln!("│  Convos:     /api/v1/conversations + /export /import");
    eprintln!("│  Presets:    /api/v1/prompts");
    eprintln!("│  Files:      /api/v1/data/upload /files");
    eprintln!("│  Recipes:    /api/v1/data/recipes /datasets");
    eprintln!("│  RAG:        /api/v1/rag/index /status");
    eprintln!("│  OpenAI:     /v1/models /v1/chat/completions /v1/embeddings");
    eprintln!("│  Ollama:     /api/generate /api/chat /api/tags /api/show");
    eprintln!("└──────────────────────────────────────────────────┘");

    let app = create_banco_router(state);
    axum::serve(listener, app).await?;
    Ok(())
}
