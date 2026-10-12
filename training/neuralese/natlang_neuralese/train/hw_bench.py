"""Hardware benchmark matrix for a new training GPU (plans/COREWEAVE_HARDWARE_OPTIMIZATION.md, "Benchmark harness").

One command measures, on the GPU it runs on:

1. **probe**: device facts, library versions and which kernels work here, with their rates: BF16 GEMM (cuBLAS),
   FP8 ``torch._scaled_mm`` (rowwise), MXFP8 and NVFP4 block-scaled ``_scaled_mm``, ``torch._grouped_mm`` against a
   per-expert loop (sm_120 has no CUTLASS fast path for it), ``torch._scaled_grouped_mm`` availability, our Triton FP8
   expert GEMM (``maple/expert_gemm.py``), FlexAttention sliding-window forward+backward, SDPA backends and device
   memory bandwidth;
2. **arms**: ``train/speed_bench.py`` (one packed direct_sft update with the recipe's QAT, MuonSR stepping in backward,
   the fused readout) for Mellum and LFM under each candidate setting: checkpointing on, off and selective, FP8 on/off,
   tokens per update, ``NATLANG_MAPLE_COMPILE``, plus one profiled arm per backbone (GPU idle and per-scope time).
   Each arm runs in its own process, so an out-of-memory arm is recorded as such and the matrix goes on.

Everything lands in ``<out>/hw-bench.json`` (schema ``natlang.neuralese-hw-bench/1``): per arm the median seconds per
update, peak GPU memory, host peak, forward and supervised tokens/s and the fixed-batch losses (update 1 runs before
any optimizer step, so arms that claim equal numerics must agree on it). ``derived`` names the fastest arm per backbone
among those numerically equal to that backbone's reference arm, and the fastest overall; the plan and
``training/neuralese/hardware-profiles.json`` take their values from it.

  python -m natlang_neuralese.train.hw_bench --out runs/coreweave-hw-bench/<date> [--matrix full|smoke] [--only RE]
                                             [--dry-run]

Runs as the CoreWeave job ``scripts/coreweave/jobs/hw-bench-train.json``; on DGX run it under the memory ledger.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = 'natlang.neuralese-hw-bench/1'
MELLUM_RECIPE = 'recipes/neuralese-direct-mellum-v4.json'
LFM_RECIPE = 'recipes/neuralese-direct-lfm-v3.json'
# Arms whose first-update loss must equal their backbone's reference (same math, different execution).
LOSS_TOLERANCE = 2e-3


def arms(matrix: str) -> list[dict]:
    """The arm list: name, backbone, speed_bench flags, environment, and the reference arm it must match."""
    if matrix == 'smoke':  # a minute on any GPU: every code path once
        m = ['--backbone', 'maple', '--recipe', MELLUM_RECIPE, '--layers', '2', '--rows', '2', '--row-tokens', '1024',
             '--span', '960', '--updates', '3', '--warmup', '1', '--qat', 'q4:1.0', '--moe-kernel', 'grouped']
        lf = ['--backbone', 'lfm', '--recipe', LFM_RECIPE, '--rows', '2', '--row-tokens', '1024', '--span', '960',
              '--updates', '3', '--warmup', '1', '--qat', 'q4:1.0']
        return [
            {'name': 'm-ref', 'args': m + ['--fp8-training', 'triton-experts'], 'reference': 'm-ref'},
            {'name': 'm-unckpt1', 'args': m + ['--fp8-training', 'triton-experts', '--uncheckpointed-layers', '1'],
             'reference': 'm-ref'},
            {'name': 'm-nockpt', 'args': m + ['--fp8-training', 'triton-experts', '--checkpoint', 'off'],
             'reference': 'm-ref'},
            {'name': 'l-ref', 'args': lf, 'reference': 'l-ref'},
            {'name': 'l-nockpt', 'args': lf + ['--checkpoint', 'off'], 'reference': 'l-ref'},
        ]
    # full: Mellum at the v4 recipe's geometry (32k forward tokens per update, whole-trajectory supervision), all 28
    # layers; LFM at its recipe's 16k tokens per update and above.
    m = ['--backbone', 'maple', '--recipe', MELLUM_RECIPE, '--layers', '28', '--updates', '7', '--warmup', '2',
         '--qat', 'q4:1.0', '--moe-kernel', 'grouped']
    m32 = ['--rows', '4', '--row-tokens', '8192', '--span', '8000']
    fp8 = ['--fp8-training', 'triton-experts']
    lf = ['--backbone', 'lfm', '--recipe', LFM_RECIPE, '--updates', '8', '--warmup', '2', '--qat', 'q4:1.0']
    l16 = ['--rows', '4', '--row-tokens', '4096', '--span', '4000']
    out = [
        # Mellum: the v4 configuration as on DGX (checkpointed, FP8 experts) is the reference.
        {'name': 'm-ref', 'args': m + m32 + fp8, 'reference': 'm-ref'},
        {'name': 'm-bf16', 'args': m + m32, 'reference': None, 'note': 'numerics differ (BF16 experts)'},
        {'name': 'm-unckpt4', 'args': m + m32 + fp8 + ['--uncheckpointed-layers', '4'], 'reference': 'm-ref'},
        {'name': 'm-unckpt8', 'args': m + m32 + fp8 + ['--uncheckpointed-layers', '8'], 'reference': 'm-ref'},
        {'name': 'm-unckpt14', 'args': m + m32 + fp8 + ['--uncheckpointed-layers', '14'], 'reference': 'm-ref'},
        {'name': 'm-compile', 'args': m + m32 + fp8, 'env': {'NATLANG_MAPLE_COMPILE': '1'}, 'reference': 'm-ref'},
        {'name': 'm-16k', 'args': m + fp8 + ['--rows', '2', '--row-tokens', '8192', '--span', '8000'],
         'reference': None, 'note': 'tokens per update is a recipe decision (batch_tokens), measured for the plan'},
        {'name': 'm-64k', 'args': m + fp8 + ['--rows', '8', '--row-tokens', '8192', '--span', '8000'],
         'reference': None, 'note': 'tokens per update is a recipe decision (batch_tokens), measured for the plan'},
        {'name': 'm-profile', 'args': m + m32 + fp8 + ['--profile'], 'reference': 'm-ref',
         'note': 'profiler overhead; read its profile-*.json for GPU idle and scopes'},
        # LFM: the recipe's 16k tokens per update, checkpointed, is the reference.
        {'name': 'l-ref', 'args': lf + l16, 'reference': 'l-ref'},
        {'name': 'l-nockpt', 'args': lf + l16 + ['--checkpoint', 'off'], 'reference': 'l-ref'},
        {'name': 'l-32k-nockpt', 'args': lf + ['--rows', '8', '--row-tokens', '4096', '--span', '4000',
                                              '--checkpoint', 'off'], 'reference': None,
         'note': 'tokens per update is a recipe decision'},
        {'name': 'l-long-nockpt', 'args': lf + ['--rows', '1', '--row-tokens', '32768', '--span', '32000',
                                               '--checkpoint', 'off'], 'reference': None,
         'note': 'one 32k window (the recipe window capacity)'},
        {'name': 'l-te-fp8', 'args': lf + l16 + ['--checkpoint', 'off', '--fp8-training', 'te-fp8'],
         'reference': None, 'note': 'TE FP8 replaces QAT (direct_sft refuses both); informational'},
        {'name': 'l-profile', 'args': lf + l16 + ['--checkpoint', 'off', '--profile'], 'reference': 'l-nockpt'},
    ]
    return out


def probe() -> dict:
    """Device facts, versions, kernel availability and rates on this GPU. Each test records its own failure."""
    import torch
    result: dict = {}
    props = torch.cuda.get_device_properties(0)
    result['device'] = {'name': props.name, 'capability': [props.major, props.minor],
                        'memory_gb': round(props.total_memory / 2 ** 30, 1), 'sms': props.multi_processor_count,
                        'integrated': bool(props.is_integrated),
                        'l2_mb': getattr(props, 'L2_cache_size', 0) / 2 ** 20}
    versions = {'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda,
                'cudnn': torch.backends.cudnn.version(), 'arch_list': torch.cuda.get_arch_list()}
    for module in ('triton', 'triton_kernels', 'transformer_engine', 'torchao', 'flash_attn', 'transformers', 'vllm',
                   'flashinfer'):
        try:
            versions[module] = getattr(__import__(module), '__version__', 'present')
        except Exception as exc:  # absent or broken: record which
            versions[module] = None if isinstance(exc, ImportError) else f'error: {type(exc).__name__}'
    result['versions'] = versions
    # Driver against toolkit (CUDA minor-version compatibility): CoreWeave's driver 595.58 is CUDA 13.2 while NGC
    # 26.09 is built with 13.4. SASS code runs from driver >= 580; device code that only exists as PTX does not.
    try:
        driver = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:
        driver = f'error: {type(exc).__name__}'
    result['driver'] = {'nvidia_smi_driver': driver,
                        'cuda_driver_api': getattr(torch._C, '_cuda_getDriverVersion', lambda: None)(),
                        'cuda_runtime': torch.version.cuda,
                        'compat_libs_loaded': [line.split()[-1] for line in open('/proc/self/maps')
                                               if 'compat/libcuda' in line][:1]}

    def timed(fn, flops=None, nbytes=None, repeat=20):
        fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        torch.cuda.synchronize()
        seconds = start.elapsed_time(end) / 1e3 / repeat
        row = {'ms': seconds * 1e3}
        if flops:
            row['tflops'] = flops / seconds / 1e12
        if nbytes:
            row['gb_per_s'] = nbytes / seconds / 1e9
        return row

    def attempt(name, fn):
        try:
            result.setdefault('kernels', {})[name] = fn()
        except Exception as exc:
            result.setdefault('kernels', {})[name] = {'error': f'{type(exc).__name__}: {str(exc)[:300]}'}

    dev = 'cuda'

    def triton_kernel():  # a raw Triton JIT kernel: PTX built and loaded through the bundled ptxas on this driver
        import triton
        import triton.language as tl

        @triton.jit
        def add(x, y, out, n, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            tl.store(out + offs, tl.load(x + offs, mask=offs < n) + tl.load(y + offs, mask=offs < n), mask=offs < n)
        x = torch.randn(1 << 20, device=dev)
        out = torch.empty_like(x)
        add[(x.numel() // 1024,)](x, x, out, x.numel(), BLOCK=1024)
        return {'ok': bool(torch.allclose(out, 2 * x))}
    attempt('triton_jit', triton_kernel)
    attempt('jiterator_nvrtc', lambda: {'ok': bool(torch.allclose(
        torch.special.zeta(torch.full((1024,), 2.0, device=dev), torch.ones(1024, device=dev)).cpu(),
        torch.full((1024,), 1.6449341)))})
    attempt('inductor_compile', lambda: {'ok': bool(torch.allclose(
        torch.compile(lambda t: torch.nn.functional.gelu(t) * 2)(torch.ones(4096, device=dev)).cpu(),
        torch.nn.functional.gelu(torch.ones(4096)) * 2, atol=1e-5))})

    def te_fp8_linear():
        import transformer_engine.pytorch as te
        layer = te.Linear(2048, 2048, params_dtype=torch.bfloat16, device=dev)
        x = torch.randn(4096, 2048, device=dev, dtype=torch.bfloat16, requires_grad=True)
        with te.fp8_autocast(enabled=True):
            layer(x).sum().backward()
        return {'ok': bool(torch.isfinite(x.grad).all())}
    attempt('te_fp8_linear', te_fp8_linear)
    M, N, K = 16384, 2304, 1792  # the GB10 measurement shape (TRAINING_SPEED_20261012.md)
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    b = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    flops = 2 * M * N * K
    attempt('bf16_mm', lambda: timed(lambda: a @ b.t(), flops))
    big = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
    attempt('bf16_mm_8192', lambda: timed(lambda: big @ big, 2 * 8192 ** 3))

    def fp8_rowwise():
        a8, b8 = a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn)
        sa = torch.ones(M, 1, device=dev, dtype=torch.float32)
        sb = torch.ones(1, N, device=dev, dtype=torch.float32)
        return timed(lambda: torch._scaled_mm(a8, b8.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16), flops)
    attempt('fp8_scaled_mm_rowwise', fp8_rowwise)

    def mxfp8():
        a8, b8 = a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn)
        sa = torch.full((M, K // 32), 127, device=dev, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        sb = torch.full((N, K // 32), 127, device=dev, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        return timed(lambda: torch._scaled_mm(a8, b8.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16), flops)
    attempt('mxfp8_scaled_mm', mxfp8)

    def nvfp4():
        a4 = torch.randint(0, 255, (M, K // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        b4 = torch.randint(0, 255, (N, K // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        sa = torch.ones(M, K // 16, device=dev, dtype=torch.float8_e4m3fn)
        sb = torch.ones(N, K // 16, device=dev, dtype=torch.float8_e4m3fn)
        return timed(lambda: torch._scaled_mm(a4, b4.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16), flops)
    attempt('nvfp4_scaled_mm', nvfp4)

    # MoE: 64 experts, 32k routed rows (Mellum-like), grouped against a per-expert loop.
    E, rows = 64, 32768
    x = torch.randn(rows, K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(E, N, K, device=dev, dtype=torch.bfloat16)
    offsets = torch.arange(1, E + 1, device=dev, dtype=torch.int32) * (rows // E)
    moe_flops = 2 * rows * N * K
    attempt('grouped_mm_bf16', lambda: timed(lambda: torch._grouped_mm(x, w.transpose(-2, -1), offs=offsets), moe_flops))
    chunks = rows // E
    attempt('loop_mm_bf16', lambda: timed(lambda: [x[e * chunks:(e + 1) * chunks] @ w[e].t() for e in range(E)],
                                          moe_flops))

    def scaled_grouped():
        x8, w8 = x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)
        sx = torch.ones(rows, device=dev, dtype=torch.float32)
        sw = torch.ones(E, N, device=dev, dtype=torch.float32)
        return timed(lambda: torch._scaled_grouped_mm(x8, w8.transpose(-2, -1), sx, sw, offs=offsets,
                                                      out_dtype=torch.bfloat16), moe_flops)
    attempt('scaled_grouped_mm_fp8', scaled_grouped)

    def triton_fp8_experts():
        from ..maple import expert_gemm
        names = [n for n in dir(expert_gemm) if not n.startswith('_')]
        return {'module': 'maple.expert_gemm', 'exports': names[:40],
                'note': 'exercised end to end by the m-ref arm (fp8_training triton-experts)'}
    attempt('triton_fp8_experts_import', triton_fp8_experts)

    def flex_window():
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        T, H, D, window = 8192, 16, 128, 1024
        q, k, v = (torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
        mask = create_block_mask(lambda b, h, qi, ki: (qi >= ki) & (qi - ki < window), 1, 1, T, T, device=dev)
        compiled = torch.compile(flex_attention, dynamic=False)

        def step():
            compiled(q, k, v, block_mask=mask).sum().backward()
        return timed(step, repeat=10)
    attempt('flex_window_fwd_bwd_8k', flex_window)

    def sdpa(backend):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        T, H, D = 8192, 16, 128
        q, k, v = (torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16, requires_grad=True) for _ in range(3))

        def step():
            with sdpa_kernel([getattr(SDPBackend, backend)]):
                torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True).sum().backward()
        return timed(step, flops=3.5 * 2 * 2 * H * T * T * D / 2, repeat=10)
    for backend in ('FLASH_ATTENTION', 'CUDNN_ATTENTION', 'EFFICIENT_ATTENTION'):
        attempt('sdpa_causal_fwd_bwd_8k_' + backend.lower(), lambda backend=backend: sdpa(backend))

    src = torch.empty(2 ** 30, device=dev, dtype=torch.uint8)
    dst = torch.empty_like(src)
    attempt('device_copy_1gb', lambda: timed(lambda: dst.copy_(src), nbytes=2 * src.numel()))
    host = torch.empty(2 ** 30, dtype=torch.uint8).pin_memory()
    attempt('h2d_pinned_1gb', lambda: timed(lambda: src.copy_(host, non_blocking=True), nbytes=src.numel(), repeat=5))
    return result


def run_arm(arm: dict, out: Path, timeout: float) -> dict:
    """One speed_bench process; its bench.json, or the failure (out of memory named as such)."""
    directory = out / 'arms' / arm['name']
    directory.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **arm.get('env', {}))
    command = [sys.executable, '-m', 'natlang_neuralese.train.speed_bench', '--out', str(directory), *arm['args']]
    record = {'name': arm['name'], 'command': command, 'env': arm.get('env', {}), 'reference': arm.get('reference'),
              'note': arm.get('note')}
    tick = time.time()
    with open(directory / 'log.txt', 'w') as log:
        try:
            status = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
                                    cwd=Path(__file__).resolve().parents[2]).returncode
        except subprocess.TimeoutExpired:
            status = 'timeout'
    record['wall_seconds'] = time.time() - tick
    bench = directory / 'bench.json'
    if status == 0 and bench.is_file():
        summary = json.loads(bench.read_text())
        record.update({k: summary[k] for k in ('median_seconds', 'mean_seconds', 'tokens_per_update',
                                               'tokens_per_second', 'supervised_per_update',
                                               'supervised_tokens_per_second', 'peak_gb', 'host_peak_gb', 'losses',
                                               'device') if k in summary})
        record['status'] = 'ok'
    else:
        tail = (directory / 'log.txt').read_text(errors='replace')[-3000:]
        record['status'] = 'oom' if 'OutOfMemoryError' in tail or 'out of memory' in tail else (
            'timeout' if status == 'timeout' else 'failed')
        record['returncode'] = status
        record['log_tail'] = tail[-1200:]
    return record


def derive(records: list[dict]) -> dict:
    """Fastest numerically equal arm per reference; first-update loss agreement per arm."""
    by_name = {r['name']: r for r in records}
    derived = {'numerics': {}, 'fastest_equal': {}, 'fastest_any': {}}
    for record in records:
        reference = by_name.get(record.get('reference') or '')
        if record.get('status') != 'ok' or reference is None or reference.get('status') != 'ok':
            continue
        delta = abs(record['losses'][0] - reference['losses'][0])
        derived['numerics'][record['name']] = {'reference': reference['name'], 'first_loss_delta': delta,
                                               'equal': delta <= LOSS_TOLERANCE}
    for family in ('m-', 'l-'):
        ok = [r for r in records if r['name'].startswith(family) and r.get('status') == 'ok'
              and 'profile' not in r['name']]
        equal = [r for r in ok if derived['numerics'].get(r['name'], {}).get('equal')]
        if equal:
            best = max(equal, key=lambda r: r['supervised_tokens_per_second'])
            ref = by_name[best['reference']]
            derived['fastest_equal'][family + 'ref'] = {
                'arm': best['name'], 'supervised_tokens_per_second': best['supervised_tokens_per_second'],
                'speedup_over_reference': ref['median_seconds'] / best['median_seconds'] * (
                    best['supervised_per_update'] / ref['supervised_per_update'])}
        if ok:
            best = max(ok, key=lambda r: r['supervised_tokens_per_second'])
            derived['fastest_any'][family.rstrip('-')] = {'arm': best['name'],
                                                          'supervised_tokens_per_second':
                                                              best['supervised_tokens_per_second']}
    return derived


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--matrix', choices=('full', 'smoke'), default='full')
    p.add_argument('--only', help='regular expression over arm names (and "probe")')
    p.add_argument('--mellum', help='Mellum model directory (default: speed_bench\'s)')
    p.add_argument('--lfm', help='LFM2.5 base directory (default: the pinned hub snapshot)')
    p.add_argument('--arm-timeout', type=float, default=3600)
    p.add_argument('--dry-run', action='store_true', help='print the arms and exit')
    a = p.parse_args(argv)
    a.out = a.out.resolve()  # the arms run with training/neuralese as their working directory
    selected = [arm for arm in arms(a.matrix) if not a.only or re.search(a.only, arm['name'])]
    for arm in selected:
        model = a.mellum if arm['name'].startswith('m-') else a.lfm
        if model:
            arm['args'] = arm['args'] + ['--model', model]
    if a.dry_run:
        print(json.dumps([{'name': x['name'], 'args': x['args'], 'env': x.get('env', {})} for x in selected], indent=1))
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    document = {'schema': SCHEMA, 'matrix': a.matrix, 'started': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'host': {'cpus': os.cpu_count(), 'sandbox': os.environ.get('CW_SANDBOX_ID')},
                'git_sha': os.environ.get('NATLANG_CW_GIT_SHA'), 'arms': []}
    path = a.out / 'hw-bench.json'

    def save():
        document['derived'] = derive(document['arms'])
        pending = path.with_suffix('.pending')
        pending.write_text(json.dumps(document, indent=1) + '\n')
        pending.replace(path)

    if not a.only or re.search(a.only, 'probe'):
        # In its own process, so a crashing kernel cannot take the matrix down.
        probe_out = a.out / 'probe.json'
        code = ('import json,sys; from natlang_neuralese.train.hw_bench import probe; '
                f'open({str(probe_out)!r},"w").write(json.dumps(probe(), indent=1))')
        subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[2], timeout=1800)
        document['probe'] = json.loads(probe_out.read_text()) if probe_out.is_file() else {'error': 'probe failed'}
        save()
    for arm in selected:
        print(json.dumps({'event': 'arm_start', 'arm': arm['name']}), flush=True)
        record = run_arm(arm, a.out, a.arm_timeout)
        document['arms'].append(record)
        print(json.dumps({k: record.get(k) for k in ('name', 'status', 'median_seconds',
                                                      'supervised_tokens_per_second', 'peak_gb')}), flush=True)
        save()
    document['finished'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    save()
    print(json.dumps({'event': 'hw_bench_done', 'path': str(path), 'derived': document['derived']}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
