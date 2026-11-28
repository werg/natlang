"""Mode-P T3 Compiled-Site Serving Bridge (`plans/NATLANG_COMPILER.md` §5–§6, T3).

Unblocks T3 immediately on Mode-P (`soft:instructions` + `xs` adapter), which already passes 32/32
real-weights parity on SGLang and llama-server (`check_compiled_site_parity.py`). Loads any saved T2
`learner-state.pt` (`CompiledSiteHead` + optional `LearnedBankBasis` + optional Live-KB writer/store),
compiles call sites on demand via `PerCallNeuralHKBCompiler` with exact provenance caching, and serves
`POST /v1/natlang/compile-site` matching `ts-host`'s `compiler.jit.compile` (`createHttpJitCompiler`) contract.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import torch

from ..common.hashing import sha256_file_hex
from .compiled_site import (
    CompiledSiteConfig,
    CompiledSiteHead,
    create_compiler_adapter,
    create_learned_compiler_basis,
)
from .per_call_hkb import PerCallCompileRequest, PerCallNeuralHKBCompiler


class CompiledSiteService:
    """Thread-safe Mode-P compiled-site service backed by `PerCallNeuralHKBCompiler`."""

    def __init__(self, compiler: PerCallNeuralHKBCompiler, *, compiler_sha: str, policy_sha: str):
        self.compiler = compiler
        self.compiler_sha = str(compiler_sha)
        self.policy_sha = str(policy_sha)
        self._lock = threading.Lock()

    @classmethod
    def from_t2_state(
        cls,
        *,
        state_path: str | Path,
        checkpoint: str | Path,
        base: str | Path,
        release_root: str | Path,
        actor_store_dir: str | Path | None = None,
        device: str = 'cuda',
        dtype: str = 'bfloat16',
        writer=None,
        scoped_kb=None,
        native_query_projector=None,
    ) -> 'CompiledSiteService':
        from ..serve import load_native_engine
        from ..serve.store import TensorStore
        state_path = Path(state_path)
        release_root = Path(release_root)
        release_root.mkdir(parents=True, exist_ok=True)
        saved = torch.load(state_path, map_location='cpu', weights_only=False)
        config = CompiledSiteConfig.from_dict(saved['config'])
        engine = load_native_engine(
            str(checkpoint), base=str(base), device=device,
            dtype=getattr(torch, dtype), training_state=False,
        )
        # Share the serving actor's TensorStore when co-hosted so compiled .nzs + .nnt + basis artifacts are visible immediately.
        engine.store = TensorStore(Path(actor_store_dir) if actor_store_dir else (release_root / 'store'))
        basis_module = None
        if saved.get('learn_basis') and saved.get('basis') is not None:
            basis_module = create_learned_compiler_basis(engine, config.targets, config.rank)
            basis_module.load_state_dict(saved['basis'])
            basis_id, adapter_id = basis_module.save_artifact(engine)
            basis_module.basis_id, basis_module.adapter_id = basis_id, adapter_id
            basis_module = basis_module.to(device)
        else:
            adapter_id = create_compiler_adapter(engine, config.targets, config.rank)
        head = CompiledSiteHead(config).to(device)
        head.load_state_dict(saved['head'])
        head.eval()
        compiler = PerCallNeuralHKBCompiler(
            engine, head, adapter_id=adapter_id, release_root=release_root,
            writer=writer, scoped_kb=scoped_kb, basis_module=basis_module,
            native_query_projector=native_query_projector,
        )
        return cls(
            compiler,
            compiler_sha=sha256_file_hex(state_path),
            policy_sha=sha256_file_hex(checkpoint),
        )

    def compile_site(
        self,
        *,
        site_id: str,
        header: str,
        messages: list[dict] | None = None,
        tools: list[dict] | None = None,
        partial_args_json: str | None = None,
        scopes: tuple[str, ...] = ('natlang:global',),
        exclude_tasks: tuple[str, ...] = (),
        scope_headers: tuple[str, ...] = (),
        query_text: str | None = None,
        max_kb_slots: int = 8,
    ) -> dict:
        effective_header = (
            f'{header}\n\nBound arguments: {partial_args_json}'
            if partial_args_json and partial_args_json not in header else header
        )
        msgs = list(messages) if messages else [
            {'role': 'system', 'content': 'You are executing a Natlang function call.'},
            {'role': 'user', 'content': effective_header},
        ]
        req = PerCallCompileRequest(
            site_id=site_id,
            header=effective_header,
            messages=msgs,
            tools=list(tools or ()),
            compiler_sha=self.compiler_sha,
            policy_sha=self.policy_sha,
            scopes=tuple(scopes),
            exclude_tasks=tuple(exclude_tasks),
            scope_headers=tuple(scope_headers),
            query_text=query_text,
            max_kb_slots=int(max_kb_slots),
        )
        with self._lock:
            compiled = self.compiler.compile_call(req)
        artifacts = compiled.get('artifacts') or []
        adapter_id = artifacts[0]['artifact'] if artifacts else None
        soft = (compiled.get('softStates') or {}).get('soft:instructions') or {}
        native_state_id = soft.get('state')
        return {
            'adapterId': adapter_id,
            'nativeStateId': native_state_id,
            'cacheKey': compiled.get('cacheKey'),
            'kbHits': compiled.get('kbHits', 0),
            'releaseDir': compiled.get('releaseDir'),
            'mode': 'mode-p',
        }


def make_http_server(service: CompiledSiteService, host: str = '127.0.0.1', port: int = 8775) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _send_json(self, code: int, body: dict):
            raw = json.dumps(body).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == '/health':
                self._send_json(200, {
                    'ok': True, 'mode': 'mode-p',
                    'compilerSha': service.compiler_sha, 'policySha': service.policy_sha,
                })
                return
            self._send_json(404, {'error': 'not found'})

        def do_POST(self):
            if self.path != '/v1/natlang/compile-site':
                self._send_json(404, {'error': 'not found'})
                return
            length = int(self.headers.get('Content-Length') or 0)
            try:
                body = json.loads(self.rfile.read(length).decode('utf-8') if length else '{}')
                if not isinstance(body, dict) or not body.get('site_id') or not body.get('header'):
                    self._send_json(400, {'error': 'site_id and header are required'})
                    return
                res = service.compile_site(
                    site_id=str(body['site_id']),
                    header=str(body['header']),
                    messages=body.get('messages'),
                    tools=body.get('tools'),
                    partial_args_json=body.get('partial_args_json'),
                    scopes=tuple(body.get('scopes') or ('natlang:global',)),
                    exclude_tasks=tuple(body.get('exclude_tasks') or ()),
                    scope_headers=tuple(body.get('scope_headers') or ()),
                    query_text=body.get('query_text'),
                    max_kb_slots=int(body.get('max_kb_slots') or 8),
                )
                self._send_json(200, res)
            except Exception as exc:
                self._send_json(500, {'error': str(exc)})

    return ThreadingHTTPServer((host, int(port)), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Mode-P T3 compiled-site HTTP server')
    parser.add_argument('--state', type=Path, required=True, help='T2 learner-state.pt checkpoint')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Base policy native checkpoint')
    parser.add_argument('--base', type=Path, required=True, help='Base HF model dir')
    parser.add_argument('--release-root', type=Path, required=True, help='Cache directory for compiled releases')
    parser.add_argument('--actor-store-dir', type=Path, default=None, help='Shared serving actor TensorStore directory')
    parser.add_argument('--scoped-kb', type=Path, default=None, help='Optional ScopedExperienceKB SQLite database')
    parser.add_argument('--kb-admission-rows', type=Path, default=None, help='Pool rows JSONL for HeldAdmission check')
    parser.add_argument('--native-query-projector', type=Path, default=None, help='Distilled native_query_projector.pt')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8775)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='bfloat16')
    args = parser.parse_args(argv)
    scoped_kb = None
    nqp = None
    if args.scoped_kb is not None:
        from .hkb_bridge import HeldAdmission, ScopedExperienceKB
        if args.kb_admission_rows is None:
            raise SystemExit('--scoped-kb requires --kb-admission-rows for fail-closed HeldAdmission')
        admission = HeldAdmission.from_rows(args.kb_admission_rows)
        scoped_kb = ScopedExperienceKB(args.scoped_kb, admission=admission, readonly=True)
    if args.native_query_projector is not None:
        from .native_query import NativeQueryProjector
        ckpt = torch.load(args.native_query_projector, map_location='cpu', weights_only=True)
        nqp = NativeQueryProjector(
            tap_dims=tuple(ckpt['tap_dims']),
            key_dim=int(ckpt['key_dim']),
            hidden_dim=int(ckpt['hidden_dim']),
            tap_source=str(ckpt.get('tap_source', 'state')),
            tap_names=tuple(ckpt.get('tap_names') or ()),
        ).to(args.device)
        nqp.load_state_dict(ckpt['state_dict'], strict=True)
        nqp.eval()
    service = CompiledSiteService.from_t2_state(
        state_path=args.state, checkpoint=args.checkpoint, base=args.base,
        release_root=args.release_root, actor_store_dir=args.actor_store_dir,
        device=args.device, dtype=args.dtype, scoped_kb=scoped_kb, native_query_projector=nqp,
    )
    server = make_http_server(service, host=args.host, port=args.port)
    print(json.dumps({'serving': f'http://{args.host}:{args.port}', 'mode': 'mode-p',
                      'compilerSha': service.compiler_sha, 'policySha': service.policy_sha}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
