from concurrent.futures import Future
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import torch
from safetensors.torch import save
from natlang_neuralese.train.diloco.worker import DilocoLearner, Transport


def test_truncated_and_invalid_fragment_retry_until_valid(tmp_path):
    good=save({'theta':torch.tensor([2.0])})
    class Handler(BaseHTTPRequestHandler):
        requests=0
        def do_GET(self):
            type(self).requests+=1
            data=good[:10] if self.requests==1 else b'corrupt' if self.requests==2 else good
            self.send_response(200);self.send_header('Content-Length',str(len(good) if self.requests==1 else len(data)));self.send_header('X-Fragment-Version','1');self.end_headers();self.wfile.write(data);self.close_connection=True
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        learner=object.__new__(DilocoLearner);learner.transport=Transport(f'http://127.0.0.1:{server.server_port}','x'*24,5,allow_http=True);learner.outbox=tmp_path;learner.fragments=[SimpleNamespace(length=1)]
        learner.downloads={};learner.remote_versions=None;learner.offline_steps=0;learner.last_error=None
        for _ in range(2):
            f=Future()
            try:learner._fetch(0,0)
            except OSError as exc:f.set_exception(exc)
            else:raise AssertionError('invalid fragment accepted')
            learner.downloads={0:f};changed,errors=learner._collect_downloads();assert not changed and errors==1 and learner.downloads=={} and not list(tmp_path.iterdir())
        version,tensor=learner._fetch(0,0);assert version==1 and torch.equal(tensor,torch.tensor([2.]));assert not list(tmp_path.iterdir())
    finally:server.shutdown();server.server_close();thread.join()
