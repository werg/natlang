"""The SGLang front as serve_native_checkpoint.py builds it (load -> export manifest -> release -> bind -> store -> start
-> serve), bound to an already running (mock) SGLang URL instead of launching one."""
import json, os, sys, tempfile, time
from pathlib import Path
repo = Path(os.environ['NATLANG_REPO'])
sys.path.insert(0, str(repo / 'scripts/neuralese_data'))
from serve_native_checkpoint import bound_host_heap
from natlang_neuralese.serve import load_native_engine
from natlang_neuralese.serve.sglang_actor import SglangEngine
from natlang_neuralese.serve.http import serve
from natlang_neuralese.serve.store import TensorStore
ckpt, export, url, port = sys.argv[1], Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
bound_host_heap()
t0 = time.time()
base = os.environ.get('BENCH_BASE') or None
engine = load_native_engine(ckpt, base=base, device='cpu', engine_class=SglangEngine, training_state=False)
manifest = json.loads((export / 'natlang-export.json').read_text())
engine.release_native({'checkpoint': ckpt, 'base': base, 'device': 'cpu', 'participant': None})
engine.bind_sglang(url, export_manifest=manifest,
                   fork={'repository': 'bench', 'branch': 'bench', 'commit': 'bench', 'declared_commit': 'bench', 'dirty': False},
                   adapter_slots=4, adapter_rank=32, projection='vendor', kv_cache_dtype='fp8_e4m3', kv_layout='hybrid-swa',
                   weight_dtype='fp8', max_running=160, track_input_end=True, tool_grammar=True, export_dir=export)
engine._tool_grammar_info()   # as serve_native_checkpoint does before serving: the bench measures steady state
engine.store = TensorStore(tempfile.mkdtemp(prefix='bench-front-store-'))
engine.start()
server = serve(engine, '127.0.0.1', port)
print(json.dumps({'event': 'bench_front_serving', 'port': port, 'load_s': round(time.time() - t0, 1)}), flush=True)
server.serve_forever()
