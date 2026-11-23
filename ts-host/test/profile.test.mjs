/**
 * WS2 R5: `natlang profile [DIR] [--json]`, `nextOptimization` pluggable policy, and
 * `natlang.json` `"rewrites": { "measurements": PATH }` reaching `RewriteGate`.
 */
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { CallStore, buildProject, compileProject } from '../dist/index.js';
import { CALLS_VERSION } from '../dist/calls/types.js';
import { hexDigest } from '../dist/native/hash.js';
import { profileStore, renderProfileReport } from '../dist/cli/profile.js';
import { loadRewriteGate } from '../dist/compiler/project.js';
import { crispNextOptimization, nextOptimization, nextOptimizationPolicy } from '../dist/optimization/policies.js';

function freshStore() {
  const root = mkdtempSync(join(tmpdir(), 'natlang-profile-store-'));
  const store = CallStore.open(root);
  store.writeSettings({ minFreeBytes: 0, specialization: 'off' });
  return {
    root,
    store,
    dispose() {
      try { store.close(); } finally { rmSync(root, { recursive: true, force: true }); }
    },
  };
}

function recordSyntheticCall(store, opts) {
  const instrText = JSON.stringify(opts.instructions ?? 'Do the task.');
  const instrHash = hexDigest(instrText);
  const blobs = new Map([[instrHash, instrText]]);
  const record = {
    version: CALLS_VERSION,
    call_id: opts.callId,
    parent_call_id: opts.parentCallId ?? null,
    parent_action_index: null,
    task_id: opts.taskId ?? 'task-1',
    program_id: opts.programId ?? 'demo-app',
    build_hash: null,
    program_root: opts.programRoot ?? null,
    definition: {
      id: opts.definitionId ?? opts.name,
      name: opts.name,
      source: opts.source ?? `${opts.name}.nl`,
      key: opts.key ?? `key-${opts.name}`,
      interface: 'iface-1',
      site: opts.site ?? 'named',
      subtype: 'function',
      params: [{ name: 'input', type: 'string' }],
      returns: opts.returns ?? 'string',
      instructions: { complete: true, hash: instrHash, bytes: instrText.length },
      types: opts.types ?? {},
      ...(opts.readout ? { readout: opts.readout } : {}),
    },
    executor: {
      kind: opts.executor ?? 'agent',
      model_id: opts.executor === 'crisp' ? null : 'test-model',
      model_revision: null,
    },
    inputs: {},
    captures: {},
    capture_writes: [],
    output: null,
    outcome: opts.outcome ?? 'done',
    detail: opts.detail ?? '',
    started_at: opts.startedAt ?? '2026-11-23T10:00:00.000Z',
    ended_at: opts.endedAt ?? '2026-11-23T10:00:01.000Z',
    effects: [],
    folder: null,
    approach: { evals: [], hash: null },
    cost: {
      model_requests: opts.executor === 'crisp' ? 0 : 1,
      tokens_in: opts.tokensIn ?? (opts.executor === 'crisp' ? 0 : 400),
      tokens_out: opts.tokensOut ?? (opts.executor === 'crisp' ? 0 : 100),
      wall_ms: opts.wallMs ?? 250,
      turns: 1,
      evals: 0,
    },
    features: {},
    audit_of: null,
    events: null,
  };
  const eventLines = opts.events?.length ? opts.events.map(e => JSON.stringify(e)).join('\n') + '\n' : undefined;
  store.record(record, blobs, eventLines);
}

test('profileStore aggregates metrics from CallStore and invokes nextOptimization per function', async () => {
  const { store, dispose } = freshStore();
  const appDir = mkdtempSync(join(tmpdir(), 'natlang-profile-app-'));
  const otherDir = mkdtempSync(join(tmpdir(), 'natlang-profile-other-'));
  try {
    // 1. Finite return without readout: decision -> recommend adding readout: decision
    for (let i = 0; i < 4; i++) {
      recordSyntheticCall(store, {
        callId: `finite-${i}`,
        name: 'classify',
        source: 'classify.nl',
        programRoot: appDir,
        returns: 'Label',
        types: { Label: "'billing' | 'technical' | 'spam'" },
        wallMs: 200 + i * 50,
        tokensIn: 300,
        tokensOut: 50,
      });
    }

    // 2. High failure rate -> recommend natlang improve
    for (let i = 0; i < 6; i++) {
      recordSyntheticCall(store, {
        callId: `fail-${i}`,
        name: 'summarize',
        source: 'summarize.nl',
        programRoot: appDir,
        returns: 'string',
        outcome: i < 2 ? 'quiesced' : 'done',
        wallMs: 300,
      });
    }

    // 3. High call volume + low compiledShare -> recommend natlang specialize
    for (let i = 0; i < 24; i++) {
      recordSyntheticCall(store, {
        callId: `hot-${i}`,
        name: 'extract',
        source: 'extract.nl',
        programRoot: appDir,
        returns: 'string',
        executor: i < 4 ? 'crisp' : 'agent',
        wallMs: i < 4 ? 5 : 400,
        tokensIn: i < 4 ? 0 : 600,
        tokensOut: i < 4 ? 0 : 100,
      });
    }

    // 4. Decision readout with low confidence + decision_escalate events -> recommend natlang tune
    for (let i = 0; i < 10; i++) {
      const confidence = 0.55 + i * 0.03; // p10=0.55, p50=0.67
      const events = [
        { kind: 'decision_readout', phase: 'scored', options: [true, false], chosen: 0, probabilities: [confidence, 1 - confidence] },
        ...(i < 3 ? [{ kind: 'decision_escalate', to: 'frontier', reason: 'confidence', confidence }] : []),
      ];
      recordSyntheticCall(store, {
        callId: `urg-${i}`,
        name: 'is_urgent',
        source: 'is_urgent.nl',
        programRoot: appDir,
        returns: 'boolean',
        readout: 'decision',
        wallMs: 80,
        events,
      });
    }

    // 5. Healthy compiled function -> none
    for (let i = 0; i < 25; i++) {
      recordSyntheticCall(store, {
        callId: `fast-${i}`,
        name: 'normalize',
        source: 'normalize.nl',
        programRoot: appDir,
        returns: 'string',
        executor: i < 20 ? 'crisp' : 'agent',
        wallMs: 10,
      });
    }

    // Unrelated program call must be filtered out when `dir: appDir` is passed.
    recordSyntheticCall(store, {
      callId: 'other-1',
      name: 'otherFn',
      source: 'otherFn.nl',
      programRoot: otherDir,
      returns: 'string',
    });

    const report = await profileStore(store, { dir: appDir });
    assert.equal(report.program, appDir);
    assert.equal(report.functions.find(f => f.name === 'otherFn'), undefined);

    const byName = Object.fromEntries(report.functions.map(f => [f.name, f]));

    assert.equal(byName.classify.finiteReturn, true);
    assert.equal(byName.classify.hasDecisionReadout, false);
    assert.equal(byName.classify.recommendation.action, 'readout-decision');
    assert.match(byName.classify.recommendation.command ?? '', /readout: decision/);

    assert.equal(byName.summarize.failureClasses.quiesced, 2);
    assert.ok(byName.summarize.failureRate > 0.3);
    assert.equal(byName.summarize.recommendation.action, 'improve');
    assert.match(byName.summarize.recommendation.command ?? '', /natlang improve/);

    assert.equal(byName.extract.calls, 24);
    assert.equal(byName.extract.compiledCalls, 4);
    assert.equal(byName.extract.recommendation.action, 'specialize');
    assert.match(byName.extract.recommendation.command ?? '', /natlang specialize/);

    assert.equal(byName.is_urgent.hasDecisionReadout, true);
    assert.ok(byName.is_urgent.decisionP10 !== null && byName.is_urgent.decisionP10 <= 0.6);
    assert.ok(byName.is_urgent.decisionP50 !== null && byName.is_urgent.decisionP50 < 0.8);
    assert.equal(byName.is_urgent.escalations, 3);
    assert.equal(byName.is_urgent.escalationRate, 0.3);
    assert.equal(byName.is_urgent.recommendation.action, 'tune');
    assert.match(byName.is_urgent.recommendation.command ?? '', /natlang tune/);

    assert.equal(byName.normalize.recommendation.action, 'none');

    const rendered = renderProfileReport(report);
    assert.match(rendered, /classify\.nl/);
    assert.match(rendered, /Recommendations:/);
    assert.match(rendered, /natlang specialize/);
  } finally {
    dispose();
    rmSync(appDir, { recursive: true, force: true });
    rmSync(otherDir, { recursive: true, force: true });
  }
});

test('nextOptimization pluggable policy supports crisp default, nl override, and shadow mode', async () => {
  const sample = {
    name: 'route',
    calls: 30,
    p50Ms: 250,
    avgTokens: 500,
    compiledShare: 0.1,
    decisionP10: null,
    decisionP50: null,
    escalationRate: 0,
    failureRate: 0,
    hasDecisionReadout: false,
    finiteReturn: false,
  };
  const crisp = crispNextOptimization(sample);
  assert.equal(crisp.action, 'specialize');

  const seen = [];
  const customNl = async profile => {
    seen.push(profile.name);
    return { action: 'tune', command: 'natlang tune .', reason: 'NL policy prefers tuning first.' };
  };

  const defaultResult = await nextOptimization(sample, { nl: customNl });
  assert.equal(defaultResult.action, 'specialize');
  assert.deepEqual(seen, [], 'crisp default never calls nl');

  const nlResult = await nextOptimizationPolicy('nl', customNl)(sample);
  assert.equal(nlResult.action, 'tune');
  assert.deepEqual(seen, ['route']);

  const shadowResult = await nextOptimizationPolicy('shadow', customNl)(sample);
  assert.equal(shadowResult.action, 'specialize', 'shadow serves crisp');
  assert.deepEqual(seen, ['route', 'route'], 'shadow invokes nl alongside crisp');
});

test('natlang.json rewrites.measurements reaches RewriteGate and keeps rules off when absent or uncertified', () => {
  const dir = mkdtempSync(join(tmpdir(), 'natlang-rewrites-proj-'));
  try {
    writeFileSync(join(dir, 'package.json'), JSON.stringify({ name: 'rw-test', type: 'module' }));
    writeFileSync(join(dir, 'tsconfig.json'), JSON.stringify({ compilerOptions: { target: 'ES2022', module: 'NodeNext', moduleResolution: 'NodeNext' } }));
    writeFileSync(join(dir, 'main.ts'), `import { map } from 'natlang:neuralese';\nexport async function pipeline(v: Neuralese<string>) {\n  const step1 = (x: string) => Promise.resolve(x.trim());\n  const step2 = (x: string) => Promise.resolve(x.toUpperCase());\n  return map(await map(v, step1), step2);\n}\n`);

    // 1. Absent rewrites key -> rules stay off.
    writeFileSync(join(dir, 'natlang.json'), JSON.stringify({ schema: 'natlang.package/v2', name: 'rw-test', version: '0.1.0' }));
    const withoutKey = buildProject({ project: dir, write: false, writeDeclarations: false });
    assert.equal(withoutKey.ok, true, JSON.stringify(withoutKey.diagnostics));
    assert.deepEqual(withoutKey.rewrites, {});

    // 2. Uncertified measurement (verdict: 'harm') -> rules stay off.
    writeFileSync(join(dir, 'measurements.json'), JSON.stringify({
      model: 'lfm3-1.7b',
      dialect: 'lfm3-v1',
      measurements: [{ rule: 'map-fusion', model: 'lfm3-1.7b', dialect: 'lfm3-v1', comparison: 'cmp-harm', verdict: 'harm' }],
    }));
    writeFileSync(join(dir, 'natlang.json'), JSON.stringify({
      schema: 'natlang.package/v2',
      name: 'rw-test',
      version: '0.1.0',
      rewrites: { measurements: 'measurements.json' },
    }));
    const uncertified = buildProject({ project: dir, write: false, writeDeclarations: false });
    assert.equal(uncertified.ok, true, JSON.stringify(uncertified.diagnostics));
    assert.deepEqual(uncertified.rewrites, {});

    // 3. Certified measurement (verdict: 'no-harm') -> map-fusion applies through RewriteGate.
    writeFileSync(join(dir, 'measurements.json'), JSON.stringify({
      model: 'lfm3-1.7b',
      dialect: 'lfm3-v1',
      measurements: [{ rule: 'map-fusion', model: 'lfm3-1.7b', dialect: 'lfm3-v1', comparison: 'cmp-ok-1', verdict: 'no-harm' }],
    }));
    const certified = buildProject({ project: dir, write: false, writeDeclarations: false });
    assert.equal(certified.ok, true, JSON.stringify(certified.diagnostics));
    const applied = Object.values(certified.rewrites ?? {}).flat();
    assert.equal(applied.length, 1);
    assert.equal(applied[0].rule, 'map-fusion');
    assert.equal(applied[0].enabled_by, 'cmp-ok-1');
    const emittedMain = Object.entries(certified.outputs).find(([path]) => path.endsWith('main.js'))?.[1] ?? '';
    assert.match(emittedMain, /compose\(step2,\s*step1\)/);

    // 4. Verify loadRewriteGate directly for a mismatched model override in natlang.json.
    writeFileSync(join(dir, 'natlang.json'), JSON.stringify({
      schema: 'natlang.package/v2',
      name: 'rw-test',
      version: '0.1.0',
      rewrites: { measurements: 'measurements.json', model: 'other-model', dialect: 'lfm3-v1' },
    }));
    const mismatched = buildProject({ project: dir, write: false, writeDeclarations: false });
    assert.equal(mismatched.ok, true);
    assert.deepEqual(mismatched.rewrites, {});
    assert.equal(typeof loadRewriteGate, 'function');
    assert.equal(typeof compileProject, 'function');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('the shipped nextOptimization.nl runs as a natlang call and its answer is checked', async () => {
  const { createNatlangRuntime } = await import('../dist/index.js');
  const { scriptedModel } = await import('./support/natlang.mjs');
  const sample = { name: 'route', calls: 30, p50Ms: 250, avgTokens: 500, compiledShare: 0.1, decisionP10: null, decisionP50: null,
    escalationRate: 0, failureRate: 0, hasDecisionReadout: false, finiteReturn: false };
  const model = scriptedModel(opening => {
    assert.match(opening, /Choose the next optimization action|single best next optimization action/);
    return 'return { action: "specialize", command: "natlang specialize --definition route", reason: "Many calls, few compiled." };';
  });
  const result = await createNatlangRuntime({ model: model.driver, calls: false }).run(() => nextOptimizationPolicy('nl')(sample));
  assert.equal(result.action, 'specialize');
  assert.equal(model.openings.length, 1);
});
