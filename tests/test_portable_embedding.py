import json
import subprocess
from pathlib import Path

from natlang.invocation import SeedPolicy
from natlang.runtime import Runtime
from natlang.trace import TraceReader
from natlang.values import load_program

ROOT = Path(__file__).resolve().parents[1]


def node(script):
    result = subprocess.run(["node", "--input-type=module", "-e", script], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_browser_js_seed_vectors_and_finite_combinators_match_python():
    js = node("""
      import { deriveSeed, checkedGraph, runGraph } from './web/natlang_lite.mjs';
      const graph = checkedGraph({
        double: { returns: 'Num', code: 'return args.item*2;', engine: 'browser-js' },
        map: { kind: 'map', over: 'items', fn: 'double', returns: 'Num[]' },
        add: { returns: 'Num', code: 'return args.acc+args.item;', engine: 'browser-js' },
        fold: { kind: 'fold', over: 'items', step: 'add', init: 0, returns: 'Num' }
      }, 'map');
      const mapped = await runGraph(graph, { items: [1,2,3] });
      const folded = await runGraph(checkedGraph(graph.definitions, 'fold'), { items: [1,2,3] });
      console.log(JSON.stringify({ seed: await deriveSeed(43, '', 1, 'model-turn', 0),
                                   mapped: mapped.value, folded: folded.value,
                                   mapOutcome: mapped.outcome, foldOutcome: folded.outcome }));
    """)
    py_map = load_program({"$map": {"type": "Map<Num, Num>", "over": [1, 2, 3],
        "fn": {"$lambda": {"type": "Lambda<{ item: Num }, Num>", "code": "return args.item*2;"}}}})
    py_fold = load_program({"$fold": {"type": "Fold<Num, Num>", "over": [1, 2, 3], "init": 0,
        "step": {"$lambda": {"type": "Lambda<{ acc: Num, item: Num }, Num>",
                              "code": "return args.acc+args.item;"}}}})
    map_out, map_value = Runtime(None).run_root(py_map)
    fold_out, fold_value = Runtime(None).run_root(py_fold)
    assert js == {"seed": SeedPolicy("derived", 43).seed("", 1, "model-turn", 0),
                  "mapped": map_value, "folded": fold_value,
                  "mapOutcome": map_out.kind, "foldOutcome": fold_out.kind}


def test_browser_reader_opens_python_trace_without_execution(tmp_path):
    root = load_program({"$lambda": {"type": "Lambda<{}, Num>", "code": "return 7;"}})
    path = tmp_path / "trace.jsonl"
    Runtime(None, trace_path=path).run_root(root)
    inspected = node(f"""
      import fs from 'node:fs';
      import {{ readTrace }} from './web/natlang_lite.mjs';
      const events = fs.readFileSync({json.dumps(str(path))}, 'utf8').trim().split('\\n').map(JSON.parse);
      const result = readTrace(events);
      console.log(JSON.stringify({{ final: result.final, outcome: result.outcome,
                                     version: result.manifest.version }}));
    """)
    assert inspected == {"final": 7, "outcome": "done", "version": "reduction-trace/1"}
    assert TraceReader.open(path).final_state() == 7


def test_browser_recorded_decisions_and_source_snapshot():
    result = node("""
      import { checkedGraph, runGraph } from './web/natlang_lite.mjs';
      const source = { cell: { returns: 'Bool', instructions: 'Write true.' } };
      const graph = checkedGraph(source, 'cell');
      source.cell.instructions = 'Write false.';
      const run = await runGraph(graph, {}, { decisions: { '': [
        { name: 'write', arguments: { path: 'return', value: true } }
      ] } });
      let unsupported = false;
      try {
        await runGraph(checkedGraph({ cell: { returns: 'Num', code: 'return 1;',
                                               engine: 'quickjs-isolated' } }, 'cell'), {});
      } catch (error) { unsupported = error.message.includes('unsupported engine'); }
      console.log(JSON.stringify({ value: run.value, outcome: run.outcome,
                                   source: graph.definitions.cell.instructions,
                                   actions: run.events.filter(e => e.kind === 'action').length,
                                   unsupported }));
    """)
    assert result == {"value": True, "outcome": "done", "source": "Write true.",
                      "actions": 1, "unsupported": True}
