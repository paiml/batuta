//! Embedded browser UI — minimal SPA served from the binary.
//!
//! The HTML/CSS/JS is compiled into the binary via `include_str!`.
//! No separate build step, no external files needed.

/// Minimal chat UI — single-page app that uses the Banco API.
pub const INDEX_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Banco — Local AI Workbench</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, system-ui, sans-serif; background: #1a1a2e; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }
header { background: #16213e; padding: 12px 20px; display: flex; align-items: center; gap: 12px; border-bottom: 1px solid #0f3460; }
header h1 { font-size: 16px; font-weight: 600; color: #e94560; }
header .status { font-size: 12px; color: #888; margin-left: auto; }
header .status.connected { color: #4caf50; }
#messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
.msg { max-width: 80%; padding: 10px 14px; border-radius: 12px; font-size: 14px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
.msg.user { align-self: flex-end; background: #0f3460; color: #fff; }
.msg.assistant { align-self: flex-start; background: #16213e; border: 1px solid #0f3460; }
.msg.system { align-self: center; color: #888; font-size: 12px; }
#input-area { background: #16213e; padding: 12px 20px; border-top: 1px solid #0f3460; display: flex; gap: 8px; }
#input { flex: 1; padding: 10px 14px; border: 1px solid #0f3460; border-radius: 8px; background: #1a1a2e; color: #e0e0e0; font-size: 14px; outline: none; }
#input:focus { border-color: #e94560; }
#send { padding: 10px 20px; background: #e94560; color: #fff; border: none; border-radius: 8px; cursor: pointer; font-size: 14px; }
#send:hover { background: #c73e54; }
#send:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
</head>
<body>
<header>
  <h1>Banco</h1>
  <span>Local AI Workbench</span>
  <span class="status" id="ws-status">disconnected</span>
</header>
<div id="messages">
  <div class="msg system">Welcome to Banco. Type a message to chat with your local model.</div>
</div>
<div id="input-area">
  <input id="input" type="text" placeholder="Type a message..." autocomplete="off">
  <button id="send">Send</button>
</div>
<script>
const msgs = document.getElementById('messages');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const wsStatus = document.getElementById('ws-status');

function addMsg(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

// WebSocket for real-time events
let ws;
function connectWs() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + location.host + '/api/v1/ws');
  ws.onopen = () => { wsStatus.textContent = 'connected'; wsStatus.className = 'status connected'; };
  ws.onclose = () => { wsStatus.textContent = 'disconnected'; wsStatus.className = 'status'; setTimeout(connectWs, 3000); };
  ws.onmessage = (e) => {
    try {
      const evt = JSON.parse(e.data);
      if (evt.type === 'training_metric') {
        /* suppress metric spam */
      } else if (evt.type !== 'connected') {
        addMsg('system', '[event] ' + evt.type);
      }
    } catch(_) {}
  };
}
connectWs();

async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  addMsg('user', text);
  sendBtn.disabled = true;

  try {
    const res = await fetch('/api/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'user', content: text }] })
    });
    const data = await res.json();
    const reply = data.choices?.[0]?.message?.content || JSON.stringify(data);
    addMsg('assistant', reply);
  } catch (err) {
    addMsg('system', 'Error: ' + err.message);
  }
  sendBtn.disabled = false;
  input.focus();
}

sendBtn.onclick = sendMessage;
input.onkeydown = (e) => { if (e.key === 'Enter') sendMessage(); };
input.focus();

// Load system info
fetch('/api/v1/system').then(r => r.json()).then(info => {
  addMsg('system', `${info.version} | ${info.privacy_tier} | ${info.endpoints} endpoints | model: ${info.model_id || 'none'}`);
}).catch(() => {});
</script>
</body>
</html>"#;
