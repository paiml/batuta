//! Session persistence for `apr code`.
//!
//! Serializes conversation history to `~/.apr/sessions/{id}/messages.jsonl`
//! for crash recovery and `/resume`. Each message is one JSON line.
//!
//! See: apr-code.md §6

use std::fs;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::driver::Message;

/// Session manifest stored alongside messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionManifest {
    /// Unique session ID.
    pub id: String,
    /// Agent name (e.g., "apr-code").
    pub agent: String,
    /// Working directory at session start.
    pub cwd: String,
    /// Timestamp of session creation (ISO 8601).
    pub created: String,
    /// Number of turns completed.
    pub turns: u32,
}

/// Persistent session storage backed by JSONL files.
pub struct SessionStore {
    /// Root directory for this session.
    pub dir: PathBuf,
    /// Session manifest.
    pub manifest: SessionManifest,
}

impl SessionStore {
    /// Create a new session in `~/.apr/sessions/{id}/`.
    pub fn create(agent_name: &str) -> anyhow::Result<Self> {
        let id = generate_session_id();
        let sessions_dir = sessions_root()?;
        let dir = sessions_dir.join(&id);
        fs::create_dir_all(&dir)?;

        let cwd =
            std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_else(|_| ".".into());

        let manifest = SessionManifest {
            id,
            agent: agent_name.to_string(),
            cwd,
            created: chrono_now(),
            turns: 0,
        };

        // Write manifest
        let manifest_path = dir.join("manifest.json");
        let json = serde_json::to_string_pretty(&manifest)?;
        fs::write(&manifest_path, json)?;

        Ok(Self { dir, manifest })
    }

    /// Resume an existing session by ID.
    pub fn resume(session_id: &str) -> anyhow::Result<Self> {
        let dir = sessions_root()?.join(session_id);
        if !dir.is_dir() {
            anyhow::bail!("session not found: {session_id}");
        }

        let manifest_path = dir.join("manifest.json");
        let json = fs::read_to_string(&manifest_path)?;
        let manifest: SessionManifest = serde_json::from_str(&json)?;

        Ok(Self { dir, manifest })
    }

    /// Find the most recent session for the current working directory.
    pub fn find_recent_for_cwd() -> Option<SessionManifest> {
        let sessions_dir = sessions_root().ok()?;
        if !sessions_dir.is_dir() {
            return None;
        }

        let cwd = std::env::current_dir().ok()?.display().to_string();

        let mut best: Option<(SessionManifest, std::time::SystemTime)> = None;
        for entry in fs::read_dir(&sessions_dir).ok()?.flatten() {
            let manifest_path = entry.path().join("manifest.json");
            if !manifest_path.is_file() {
                continue;
            }
            if let Ok(json) = fs::read_to_string(&manifest_path) {
                if let Ok(m) = serde_json::from_str::<SessionManifest>(&json) {
                    if m.cwd == cwd && m.turns > 0 {
                        let mtime = entry.metadata().ok()?.modified().ok()?;
                        if best.as_ref().is_none_or(|(_, t)| mtime > *t) {
                            best = Some((m, mtime));
                        }
                    }
                }
            }
        }

        best.map(|(m, _)| m)
    }

    /// Session ID.
    pub fn id(&self) -> &str {
        &self.manifest.id
    }

    /// Append a message to the JSONL log.
    pub fn append_message(&self, msg: &Message) -> anyhow::Result<()> {
        let path = self.dir.join("messages.jsonl");
        let mut file = fs::OpenOptions::new().create(true).append(true).open(&path)?;
        let json = serde_json::to_string(msg)?;
        writeln!(file, "{json}")?;
        Ok(())
    }

    /// Append multiple messages (e.g., after a turn completes).
    pub fn append_messages(&self, msgs: &[Message]) -> anyhow::Result<()> {
        let path = self.dir.join("messages.jsonl");
        let mut file = fs::OpenOptions::new().create(true).append(true).open(&path)?;
        for msg in msgs {
            let json = serde_json::to_string(msg)?;
            writeln!(file, "{json}")?;
        }
        Ok(())
    }

    /// Load all messages from the JSONL log.
    pub fn load_messages(&self) -> anyhow::Result<Vec<Message>> {
        let path = self.dir.join("messages.jsonl");
        if !path.is_file() {
            return Ok(Vec::new());
        }

        let file = fs::File::open(&path)?;
        let reader = std::io::BufReader::new(file);
        let mut messages = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let msg: Message = serde_json::from_str(&line)?;
            messages.push(msg);
        }

        Ok(messages)
    }

    /// Update the turn count in the manifest.
    pub fn record_turn(&mut self) -> anyhow::Result<()> {
        self.manifest.turns += 1;
        let manifest_path = self.dir.join("manifest.json");
        let json = serde_json::to_string_pretty(&self.manifest)?;
        fs::write(&manifest_path, json)?;
        Ok(())
    }
}

/// Root directory for all sessions.
fn sessions_root() -> anyhow::Result<PathBuf> {
    let home =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("cannot determine home directory"))?;
    Ok(home.join(".apr").join("sessions"))
}

/// Generate a short session ID from timestamp + nanos for uniqueness.
fn generate_session_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let dur = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let ts = dur.as_secs();
    let nanos = dur.subsec_nanos();
    format!("{ts:x}-{nanos:08x}")
}

/// Current UTC time as ISO 8601 string (no chrono dependency).
fn chrono_now() -> String {
    // Simple UTC timestamp without external dependency
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    // Approximate: days since epoch, then h/m/s
    let days = secs / 86400;
    let rem = secs % 86400;
    let h = rem / 3600;
    let m = (rem % 3600) / 60;
    let s = rem % 60;
    // Year calculation (good enough for display)
    let years = 1970 + days / 365;
    let day_of_year = days % 365;
    let month = day_of_year / 30 + 1;
    let day = day_of_year % 30 + 1;
    format!("{years:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a session store in a temp dir (isolated from ~/.apr/sessions/).
    fn create_test_store() -> SessionStore {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let id = generate_session_id();
        let tmp_path = tmp.path().to_path_buf();
        // Prevent cleanup so test can use the dir
        std::mem::forget(tmp);
        let dir = tmp_path.join(&id);
        fs::create_dir_all(&dir).expect("mkdir");

        let manifest = SessionManifest {
            id,
            agent: "test-agent".into(),
            cwd: ".".into(),
            created: chrono_now(),
            turns: 0,
        };
        let json = serde_json::to_string_pretty(&manifest).expect("json");
        fs::write(dir.join("manifest.json"), json).expect("write");
        SessionStore { dir, manifest }
    }

    #[test]
    fn test_session_create_and_persist() {
        let store = create_test_store();
        assert!(!store.id().is_empty());
        assert!(store.dir.is_dir());

        store.append_message(&Message::User("hello".into())).expect("append");
        store.append_message(&Message::Assistant("hi".into())).expect("append");

        let msgs = store.load_messages().expect("load");
        assert_eq!(msgs.len(), 2);
        assert!(matches!(&msgs[0], Message::User(s) if s == "hello"));
        assert!(matches!(&msgs[1], Message::Assistant(s) if s == "hi"));

        let _ = fs::remove_dir_all(&store.dir);
    }

    #[test]
    fn test_session_resume_by_path() {
        let store = create_test_store();
        store.append_message(&Message::User("test".into())).expect("append");

        // Resume by reading from same dir
        let manifest_json = fs::read_to_string(store.dir.join("manifest.json")).expect("read");
        let manifest: SessionManifest = serde_json::from_str(&manifest_json).expect("parse");
        let resumed = SessionStore { dir: store.dir.clone(), manifest };
        let msgs = resumed.load_messages().expect("load");
        assert_eq!(msgs.len(), 1);

        let _ = fs::remove_dir_all(&store.dir);
    }

    #[test]
    fn test_session_resume_nonexistent() {
        let result = SessionStore::resume("nonexistent-id-12345");
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_session_id_unique() {
        let id1 = generate_session_id();
        // Small sleep to ensure different nanos
        std::thread::sleep(std::time::Duration::from_millis(1));
        let id2 = generate_session_id();
        assert_ne!(id1, id2, "IDs should be unique");
        assert!(id1.contains('-'));
    }

    #[test]
    fn test_append_and_load_empty() {
        let store = create_test_store();
        let msgs = store.load_messages().expect("load");
        assert!(msgs.is_empty());
        let _ = fs::remove_dir_all(&store.dir);
    }

    #[test]
    fn test_record_turn() {
        let mut store = create_test_store();
        assert_eq!(store.manifest.turns, 0);
        store.record_turn().expect("record");
        assert_eq!(store.manifest.turns, 1);

        // Reload manifest from disk
        let json = fs::read_to_string(store.dir.join("manifest.json")).expect("read");
        let reloaded: SessionManifest = serde_json::from_str(&json).expect("parse");
        assert_eq!(reloaded.turns, 1);

        let _ = fs::remove_dir_all(&store.dir);
    }
}
