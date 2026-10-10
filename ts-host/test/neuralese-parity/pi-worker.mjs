// pi's browser entry (applications/pi/browser.ts, bundled by scripts/build-application-browser.mjs) in a dedicated
// worker, where pi-durable's session lives in an OPFS SQLite database. The agent model is scripted by the driver's HTTP
// server; the executor (pi's natural-language tool functions) by the eval code below. Phase `first` runs a coding task;
// phase `again`, after a page reload, reopens the session and continues the same conversation.
import * as natlang from '../../dist/browser/natlang.js';

/** The eval code a small model writes for pi's write tool (applications/pi/test/browser-bundle.test.mjs). */
const WRITE = `
    const p = await env.toolPath(args.path); if (p.error) throw new Error(p.error.message);
    const held = await env.lock(p.value); if (held.error) throw new Error(held.error.message);
    const w = await env.writeFile(p.value, args.content);
    env.unlock(held.value);
    if (w.error) throw new Error(w.error.message);
    return { content: [{ type: 'text', text: 'Successfully wrote to ' + args.path }] };`;

const executor = async ({ messages }) => {
  if (natlang.modelTurnsSoFar(messages) === 0) {
    return String(messages[1].content).includes('Write args.content to the file args.path') ? { calls: [['eval', { code: WRITE }]] } :
      { calls: [['return_result', { status: 'failed', reason: 'The scripted executor has no answer for this task.' }]] };
  }
  const last = messages.at(-1);
  if (last.role === 'tool' && /^(?:rejected|error)|\nerror/.test(String(last.content)))
    return { calls: [['return_result', { status: 'failed', reason: `Scripted eval failed: ${String(last.content).slice(0, 300)}` }]] };
  return { text: 'done' };
};

self.onmessage = async ({ data: { phase, agent, entry } }) => {
  const reports = [];
  try {
    const pi = await import(entry);
    const folder = new pi.Folder({ 'README.md': '# demo\n' });
    // pi's browser modules as applications/pi/test/browser-bundle.test.mjs composes them: pi-durable's harness with pi's
    // coding tools (their natural-language functions run by the scripted executor) and the agent reached through
    // natlang's transport. openBrowserPi itself also makes pi's task kinds natural-language functions (generation.nl,
    // tool.nl), which need a real executor model; the session is the same openBrowserSession.
    const natlangRuntime = natlang.createNatlangRuntime({ model: executor, calls: false });
    const { storage, kind } = await pi.openBrowserSession('pi-parity-session', error => reports.push(String(error?.message ?? error)));
    const { models, ref } = pi.agentModels({ endpoint: agent, modelId: 'scripted', transport: 'natlang' });
    const envs = new pi.ExecutionEnvs(pi.WORKSPACE, cwd => new pi.FolderExecutionEnv({ folder, cwd }));
    const registry = pi.codingRegistry(natlangRuntime, { prompt: { cwd: pi.WORKSPACE, packageDir: '/pi' }, subagent: false });
    const context = pi.BACKGROUND_CONTEXT;
    const harness = await pi.Harness.open(storage, { models, registry, env: envs.env, onReport: error => reports.push(String(error?.stack ?? error)) }, context);
    const opened = { session: kind, harness, run: task => pi.runPiTask(harness, { model: ref, cwd: pi.WORKSPACE }, task, context),
      close: async () => { await harness.close(context); await envs.cleanup(context); } };
    // The root conversation is opened with its agent, as runPiTask opens it (a root opened without one has no model).
    const before = await harness.root(context, { agent: { model: ref, thinkingLevel: 'off', cwd: pi.WORKSPACE } }).then(root => root.entries({}, 1000, undefined, context)).then(page => page.items.length);
    const task = phase === 'first' ? 'Write hello.txt saying hi.' : 'What did hello.txt say?';
    const result = await opened.run(task);
    const files = phase === 'first' ? { 'hello.txt': await folder.readText('hello.txt').catch(() => null) } : {};
    const entries = await opened.harness.root(context).then(root => root.entries({}, 1000, undefined, context)).then(page => page.items);
    await opened.close();
    // What the session holds, for diagnosis when a task does not finish.
    const transcript = result.status === 'done' ? undefined : entries.map(entry => JSON.stringify(entry).slice(0, 1500));
    self.postMessage({ ok: true, phase, session: opened.session, entries_before: before, entries_after: entries.length, result, files, reports,
      ...(transcript ? { transcript } : {}) });
  } catch (error) {
    self.postMessage({ ok: false, phase, error: String(error?.stack ?? error), reports });
  }
};
