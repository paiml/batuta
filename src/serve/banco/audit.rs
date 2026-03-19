//! Request audit logging for Banco.
//!
//! Logs every API request as a JSON line to an in-memory buffer.
//! Phase 1: in-memory ring buffer. Phase 2+: append to `~/.banco/audit.jsonl`.

use axum::{
    body::Body,
    http::{Method, Request, Response, StatusCode},
    middleware::Next,
};
use serde::Serialize;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// A single audit log entry.
#[derive(Debug, Clone, Serialize)]
pub struct AuditEntry {
    pub ts: String,
    pub method: String,
    pub path: String,
    pub status: u16,
    pub latency_ms: u64,
}

/// In-memory audit log (ring buffer, max 10,000 entries).
#[derive(Debug, Clone)]
pub struct AuditLog {
    entries: Arc<Mutex<Vec<AuditEntry>>>,
}

const MAX_ENTRIES: usize = 10_000;

impl AuditLog {
    #[must_use]
    pub fn new() -> Self {
        Self { entries: Arc::new(Mutex::new(Vec::with_capacity(256))) }
    }

    pub fn push(&self, entry: AuditEntry) {
        if let Ok(mut entries) = self.entries.lock() {
            if entries.len() >= MAX_ENTRIES {
                entries.remove(0);
            }
            entries.push(entry);
        }
    }

    #[must_use]
    pub fn recent(&self, limit: usize) -> Vec<AuditEntry> {
        self.entries
            .lock()
            .map(|e| e.iter().rev().take(limit).cloned().collect())
            .unwrap_or_default()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.lock().map(|e| e.len()).unwrap_or(0)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new()
    }
}

/// Axum middleware that logs every request to the audit log.
pub async fn audit_layer(
    audit_log: AuditLog,
    request: Request<Body>,
    next: Next,
) -> Response<Body> {
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let start = Instant::now();

    let response = next.run(request).await;

    let entry = AuditEntry {
        ts: iso_now(),
        method: method.to_string(),
        path,
        status: response.status().as_u16(),
        latency_ms: start.elapsed().as_millis() as u64,
    };
    audit_log.push(entry);

    response
}

fn iso_now() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    // Simple ISO-ish format without chrono dependency
    format!("{secs}")
}
