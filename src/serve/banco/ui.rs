//! Embedded browser UI — single-page chat app served from the binary.
//!
//! Features: chat, model status, settings, file upload, RAG toggle, conversations.
//! All in one HTML file — no build step, no external dependencies.

/// Full chat UI with sidebar, settings, and file upload.
pub const INDEX_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Banco — Local AI Workbench</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#1a1a2e;--surface:#16213e;--border:#0f3460;--accent:#e94560;--text:#e0e0e0;--dim:#888}
body{font-family:-apple-system,system-ui,sans-serif;background:var(--bg);color:var(--text);height:100vh;display:flex}
.sidebar{width:260px;background:var(--surface);border-right:1px solid var(--border);display:flex;flex-direction:column;flex-shrink:0}
.sidebar h2{padding:12px 16px;font-size:14px;color:var(--accent);border-bottom:1px solid var(--border)}
.sidebar .model-info{padding:8px 16px;font-size:11px;color:var(--dim);border-bottom:1px solid var(--border)}
.sidebar .model-info .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px}
.sidebar .model-info .dot.on{background:#4caf50}.sidebar .model-info .dot.off{background:#f44336}
.convos{flex:1;overflow-y:auto;padding:8px}
.convos .conv{padding:8px 12px;border-radius:6px;cursor:pointer;font-size:13px;margin-bottom:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.convos .conv:hover{background:var(--border)}
.sidebar .actions{padding:8px 16px;border-top:1px solid var(--border);display:flex;gap:6px}
.sidebar .actions button{flex:1;padding:6px;font-size:11px;border:1px solid var(--border);border-radius:4px;background:var(--bg);color:var(--text);cursor:pointer}
.sidebar .actions button:hover{border-color:var(--accent)}
.main{flex:1;display:flex;flex-direction:column}
header{background:var(--surface);padding:10px 20px;display:flex;align-items:center;gap:12px;border-bottom:1px solid var(--border)}
header h1{font-size:15px;font-weight:600;color:var(--accent)}
header .controls{margin-left:auto;display:flex;gap:8px;align-items:center}
header .controls label{font-size:11px;color:var(--dim)}
header .controls input[type=range]{width:60px;accent-color:var(--accent)}
header .controls .toggle{padding:3px 8px;font-size:11px;border:1px solid var(--border);border-radius:4px;background:var(--bg);color:var(--text);cursor:pointer}
header .controls .toggle.active{background:var(--accent);border-color:var(--accent);color:#fff}
.ws-status{font-size:10px;padding:2px 6px;border-radius:3px;background:var(--bg)}
.ws-status.connected{color:#4caf50}.ws-status.disconnected{color:#f44336}
#messages{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:10px}
.msg{max-width:80%;padding:10px 14px;border-radius:12px;font-size:14px;line-height:1.5;white-space:pre-wrap;word-break:break-word}
.msg.user{align-self:flex-end;background:var(--border);color:#fff}
.msg.assistant{align-self:flex-start;background:var(--surface);border:1px solid var(--border)}
.msg.system{align-self:center;color:var(--dim);font-size:11px}
#input-area{background:var(--surface);padding:10px 20px;border-top:1px solid var(--border);display:flex;gap:8px}
#input{flex:1;padding:10px 14px;border:1px solid var(--border);border-radius:8px;background:var(--bg);color:var(--text);font-size:14px;outline:none}
#input:focus{border-color:var(--accent)}
#send{padding:10px 20px;background:var(--accent);color:#fff;border:none;border-radius:8px;cursor:pointer;font-size:14px}
#send:hover{background:#c73e54}#send:disabled{opacity:.5;cursor:not-allowed}
.upload-btn{padding:10px;background:var(--bg);border:1px solid var(--border);border-radius:8px;cursor:pointer;font-size:14px;color:var(--dim)}
.upload-btn:hover{border-color:var(--accent)}
</style>
</head>
<body>
<div class="sidebar">
  <h2>Banco</h2>
  <div class="model-info" id="model-info"><span class="dot off"></span>No model</div>
  <div class="convos" id="convos"></div>
  <div class="actions">
    <button onclick="newConvo()">+ New Chat</button>
    <button onclick="loadConvos()">Refresh</button>
  </div>
</div>
<div class="main">
<header>
  <h1>Chat</h1>
  <div class="controls">
    <label>Temp <input type="range" id="temp" min="0" max="2" step="0.1" value="0.7"></label>
    <label>Tokens <input type="range" id="maxtok" min="32" max="2048" step="32" value="256"></label>
    <button class="toggle" id="rag-toggle" onclick="toggleRag()">RAG</button>
    <span class="ws-status disconnected" id="ws-status">ws</span>
  </div>
</header>
<div id="messages"><div class="msg system">Welcome to Banco. Type a message to chat.</div></div>
<div id="input-area">
  <button class="upload-btn" onclick="uploadFile()" title="Upload file for RAG">📎</button>
  <input id="input" type="text" placeholder="Type a message..." autocomplete="off">
  <button id="send">Send</button>
</div>
</div>
<script>
const M=document.getElementById('messages'),I=document.getElementById('input'),S=document.getElementById('send');
let ragOn=false,convId=null;
function addMsg(r,t){const d=document.createElement('div');d.className='msg '+r;d.textContent=t;M.appendChild(d);M.scrollTop=M.scrollHeight}
// WebSocket
let ws;function connectWs(){const p=location.protocol==='https:'?'wss:':'ws:';ws=new WebSocket(p+'//'+location.host+'/api/v1/ws');
ws.onopen=()=>{document.getElementById('ws-status').textContent='ws';document.getElementById('ws-status').className='ws-status connected'};
ws.onclose=()=>{document.getElementById('ws-status').className='ws-status disconnected';setTimeout(connectWs,3000)};
ws.onmessage=e=>{try{const ev=JSON.parse(e.data);if(ev.type==='model_loaded')loadModel();else if(ev.type!=='connected'&&ev.type!=='training_metric')addMsg('system','['+ev.type+']')}catch(_){}}}
connectWs();
// Model info
function loadModel(){fetch('/api/v1/system').then(r=>r.json()).then(d=>{const el=document.getElementById('model-info');
if(d.model_loaded){el.innerHTML='<span class="dot on"></span>'+d.model_id}else{el.innerHTML='<span class="dot off"></span>No model'}}).catch(()=>{})}
loadModel();
// Conversations
function loadConvos(){fetch('/api/v1/conversations').then(r=>r.json()).then(d=>{const el=document.getElementById('convos');el.innerHTML='';
(d.conversations||[]).forEach(c=>{const div=document.createElement('div');div.className='conv';div.textContent=c.title||c.id;
div.onclick=()=>{convId=c.id;loadConvo(c.id)};el.appendChild(div)})}).catch(()=>{})}
function loadConvo(id){fetch('/api/v1/conversations/'+id).then(r=>r.json()).then(d=>{M.innerHTML='';
(d.messages||[]).forEach(m=>addMsg(m.role,m.content))}).catch(()=>{})}
function newConvo(){convId=null;M.innerHTML='<div class="msg system">New conversation started.</div>'}
loadConvos();
// Chat
async function sendMessage(){const t=I.value.trim();if(!t)return;I.value='';addMsg('user',t);S.disabled=true;
try{const body={messages:[{role:'user',content:t}],temperature:parseFloat(document.getElementById('temp').value),
max_tokens:parseInt(document.getElementById('maxtok').value),rag:ragOn};
if(convId)body.conversation_id=convId;
const r=await fetch('/api/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
const d=await r.json();const reply=d.choices?.[0]?.message?.content||JSON.stringify(d);addMsg('assistant',reply);
if(!convId&&d.conversation_id)convId=d.conversation_id;loadConvos()}catch(e){addMsg('system','Error: '+e.message)}S.disabled=false;I.focus()}
S.onclick=sendMessage;I.onkeydown=e=>{if(e.key==='Enter')sendMessage()};I.focus();
// RAG toggle
function toggleRag(){ragOn=!ragOn;const b=document.getElementById('rag-toggle');b.classList.toggle('active',ragOn)}
// File upload
function uploadFile(){const input=document.createElement('input');input.type='file';input.onchange=async()=>{
const f=input.files[0];if(!f)return;const text=await f.text();
await fetch('/api/v1/data/upload/json',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({name:f.name,content:text})});addMsg('system','Uploaded: '+f.name+' (auto-indexed for RAG)')};input.click()}
</script>
</body>
</html>"##;
