// 冒烟测试：以 MCP 客户端身份启动本 server，调用 sing_probe 打印环境自检结果。
//
// 用法： node scripts/probe.mjs
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

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

// 有任一项 ok=false 就以非零退出，方便 CI 或脚本判断
let ok = true;
try {
  const checks = JSON.parse(text);
  ok = Object.values(checks).every((v) => !v || typeof v !== 'object' || v.ok !== false);
} catch { /* 解析失败不影响退出码判断的默认值 */ }
process.exit(ok ? 0 : 1);
