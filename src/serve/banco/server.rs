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
        "none — load via POST /api/v1/models/load or --model flag".to_string()
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
    eprintln!("│  Convos:     /api/v1/conversations + /search /export /import");
    eprintln!("│  Presets:    /api/v1/prompts");
    eprintln!("│  Files:      /api/v1/data/upload /files");
    eprintln!("│  Recipes:    /api/v1/data/recipes /datasets");
    eprintln!("│  RAG:        /api/v1/rag/index /status /search");
    eprintln!("│  Eval:       /api/v1/eval/perplexity /runs");
    eprintln!("│  Training:   /api/v1/train/start /runs /presets /metrics /export");
    eprintln!("│  Merge:      /api/v1/models/merge /strategies (TIES/DARE/SLERP)");
    eprintln!("│  Experiments:/api/v1/experiments /compare");
    eprintln!("│  Batch:      /api/v1/batch");
    eprintln!("│  Registry:   /api/v1/models/pull /registry (pacha)");
    eprintln!("│  WebSocket:  /api/v1/ws (real-time events)");
    eprintln!("│  OpenAI:     /v1/models /v1/chat/completions /v1/embeddings");
    eprintln!("│  Ollama:     /api/generate /api/chat /api/tags /api/show");
    eprintln!("├──────────────────────────────────────────────────┤");
    let sys = state.system_info();
    eprintln!("│  Data:       {} files, {} conversations", sys.files, sys.conversations);
    eprintln!(
        "│  RAG:        {}",
        if sys.rag_indexed {
            format!("{} chunks indexed", sys.rag_chunks)
        } else {
            "empty".to_string()
        }
    );
    eprintln!("│  Storage:    ~/.banco/");
    eprintln!("└──────────────────────────────────────────────────┘");

    let app = create_banco_router(state);
    axum::serve(listener, app).await?;
    Ok(())
}
