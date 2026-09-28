// Calls the running Codex B app's own task tools. Never touches A's profile.
import fs from 'node:fs';
import net from 'node:net';
import crypto from 'node:crypto';

const [toolName, argsPath, outputPath] = process.argv.slice(2);
if (!toolName || !outputPath) throw new Error('Usage: node codex_b_rpc.mjs TOOL ARGS_JSON OUTPUT_JSON');
const targets = await (await fetch('http://127.0.0.1:9339/json/list')).json();
if (targets.length !== 1) throw new Error('Ambiguous B main process');
const metadata = await new Promise((resolve, reject) => {
  const ws = new WebSocket(targets[0].webSocketDebuggerUrl);
  const timer = setTimeout(() => { ws.close(); reject(new Error('B inspection timed out')); }, 5000);
  ws.onopen = () => ws.send(JSON.stringify({id: 1, method: 'Runtime.evaluate', params: {
    expression: 'JSON.stringify({exe:process.execPath,pid:process.pid,pipe:process.env.CODEX_APP_TOOLS_PIPE_PATH})', returnByValue: true
  }}));
  ws.onmessage = e => { const m = JSON.parse(e.data); if (m.id === 1) {
    clearTimeout(timer); ws.close();
    try { resolve(JSON.parse(m.result.result.value)); } catch (err) { reject(err); }
  }};
  ws.onerror = e => { clearTimeout(timer); reject(e); };
});
if (!metadata.exe.toLowerCase().includes('\\codex-b\\app\\') || !metadata.pipe) throw new Error('Not verified Codex B');
const args = JSON.parse(fs.readFileSync(argsPath, 'utf8').replace(/^\uFEFF/, ''));
const request = toolName === '__list__'
  ? {jsonrpc: '2.0', id: 1, method: 'tools/list', params: {threadStartKind: 'all'}}
  : {jsonrpc: '2.0', id: 1, method: 'tools/call', params: {
      namespace: 'codex_app', tool: toolName, arguments: args,
      threadId: '01a0cbf5-033f-74e2-88ea-43876a15c047', callId: crypto.randomUUID(), turnId: crypto.randomUUID()
    }};
const message = await new Promise((resolve, reject) => {
  const socket = net.createConnection(metadata.pipe);
  let data = Buffer.alloc(0);
  const timer = setTimeout(() => { socket.destroy(); reject(new Error('Timeout: inspect state before repeating a mutation')); }, 55000);
  socket.on('connect', () => {
    const body = Buffer.from(JSON.stringify(request)), head = Buffer.alloc(4);
    head.writeUInt32LE(body.length); socket.write(Buffer.concat([head, body]));
  });
  socket.on('data', chunk => {
    data = Buffer.concat([data, chunk]);
    while (data.length >= 4) {
      const n = data.readUInt32LE(0);
      if (n > 16 * 1024 * 1024) { socket.destroy(); clearTimeout(timer); reject(new Error('Unexpected response size')); return; }
      if (data.length < n + 4) return;
      const m = JSON.parse(data.subarray(4, n + 4).toString('utf8')); data = data.subarray(n + 4);
      if (m.id === 1) { clearTimeout(timer); socket.end(); resolve(m); return; }
    }
  });
  socket.on('error', e => { clearTimeout(timer); reject(e); });
});
fs.writeFileSync(outputPath, JSON.stringify({bProcess: metadata.pid, tool: toolName, response: message}, null, 2));
if (message.error) throw new Error(JSON.stringify(message.error));
if (toolName === '__list__') {
  console.log(JSON.stringify(message.result.tools.filter(t => /create_thread|list_projects|wait_threads|list_threads/.test(t.name))));
} else {
  console.log(JSON.stringify(message.result));
}
