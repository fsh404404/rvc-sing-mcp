// 冒烟测试：以 MCP 客户端身份启动本 server，调用 sing_probe 打印环境自检结果。
//
// 用法：
//   node scripts/probe.mjs                # 每一项 ok 都必须是 true（最严格）
//   node scripts/probe.mjs --mode=ai      # 只要求「能翻唱」（看 aiReady）
//   node scripts/probe.mjs --mode=original # 只要求「能出原唱片段」（看 originalReady）
//
// 退出码：0 = 达标，1 = 有缺失项。
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const modeArg = process.argv.find((a) => a.startsWith('--mode='));
const mode = modeArg ? modeArg.split('=')[1] : 'all';

const transport = new StdioClientTransport({
  command: process.execPath,
  args: [path.join(root, 'src', 'mcp-sing.js')],
  // MCP 客户端默认只透传少量环境变量；这里把当前环境全传下去，
  // 才能用 SING_DIR / SING_FFMPEG 之类的变量做临时覆盖测试。
  env: { ...process.env },
  stderr: 'inherit',
});

const client = new Client({ name: 'probe', version: '1.0.0' });
await client.connect(transport);

const tools = await client.listTools();
console.log('已注册工具:', tools.tools.map((t) => t.name).join(', '));

const result = await client.callTool({ name: 'sing_probe', arguments: {} });
const text = result.content?.find((c) => c.type === 'text')?.text ?? '(空)';
console.log(text);

await client.close();

let checks = null;
try {
  checks = JSON.parse(text);
} catch { /* 解析失败：退回「有 ok:false 就算失败」 */ }

let ok;
if (checks && mode === 'ai') {
  ok = checks.aiReady === true;
  console.log(`\n[mode=ai] aiReady=${checks.aiReady} · 缺: ${(checks.aiMissing ?? []).join('、') || '无'}`);
} else if (checks && mode === 'original') {
  ok = checks.originalReady === true;
  console.log(`\n[mode=original] originalReady=${checks.originalReady}`);
} else if (checks) {
  ok = Object.values(checks).every((v) => !v || typeof v !== 'object' || v.ok !== false);
} else {
  ok = !/"(ok)":\s*false/.test(text);
  if (!checks) console.log('\n[提示] 自检结果没解析成 JSON，按字符串兜底判断');
}

console.log(ok ? '结果: 达标' : '结果: 未达标（上面 ok:false 的就是缺什么）');
process.exit(ok ? 0 : 1);
