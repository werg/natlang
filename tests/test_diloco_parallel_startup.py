"""Startup overlaps real HTTP transfers without retaining an entire model."""
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Barrier, Thread
from types import SimpleNamespace

import torch
from safetensors.torch import save

from natlang_neuralese.train.diloco.protocol import TensorSurface, pack_fragment, plan_fragments
from natlang_neuralese.train.diloco.worker import DilocoLearner, Transport


def test_http_startup_is_parallel_bounded_ordered_and_preserves_local_residual(tmp_path):
    model = torch.nn.Linear(1536, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(4)
    surface = TensorSurface.from_module(model)
    fragments = plan_fragments(surface.masters, target_bytes=1024)
    assert len(fragments) == 6
    installed = []
    barriers = [Barrier(2) for _ in range(3)]
    violations = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            index = int(self.path.split('?')[0].rsplit('/', 1)[1])
            # A serial fetch cannot pass this barrier. Later waves may not
            # start until the previous decoded tensors have been installed.
            cutoff = 2 * (index // 2)
            if installed[:cutoff] != list(range(cutoff)):
                violations.append(index)
            barriers[index // 2].wait(timeout=5)
            payload = save({'theta': torch.full((fragments[index].length,), 10. + index)})
            self.send_response(200)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('X-Fragment-Version', '1')
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        learner = object.__new__(DilocoLearner)
        learner.transport = Transport(f'http://127.0.0.1:{server.server_port}', 'x' * 24, 10, allow_http=True)
        learner.outbox = tmp_path
        learner.settings = SimpleNamespace(io_workers=2)
        learner.surface, learner.fragments = surface, fragments
        learner.versions = [0] * len(fragments)
        learner.baselines = [torch.full((part.length,), 3.) for part in fragments]
        learner.awaiting_global = set(range(len(fragments)))
        learner.installs = 0
        install = learner._install

        def observed_install(index, update):
            result = install(index, update)
            installed.append(index)
            return result

        learner._install = observed_install
        with ThreadPoolExecutor(max_workers=2) as learner.io:
            learner._fetch_all()
        assert not violations
        assert installed == list(range(len(fragments)))
        assert learner.versions == [1] * len(fragments)
        assert not learner.awaiting_global
        for index, fragment in enumerate(fragments):
            assert torch.equal(pack_fragment(surface.masters, fragment), torch.full((fragment.length,), 11. + index))
        assert not list(tmp_path.iterdir())
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
