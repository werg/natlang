"""CoreWeave sandbox job declarations (pure: no network, no SDK import).

A job is a JSON object of schema ``natlang.coreweave-job/1``. It says what runs (command over a repo snapshot), on
what (image, resources, lifetime, scratch disk), with which inputs (a bundle manifest in object storage, see
bundle.py) and where its outputs go (an object-storage prefix). Placement (which job goes to CoreWeave instead of DGX
or Pop) is decided by the placement program (plans/COREWEAVE_SANDBOXES.md, "Placement policy"); the job records that
decision in ``placement.reason`` and this module only checks and translates the declaration.

``sandbox_request(job, bootstrap)`` is the exact keyword set passed to ``cwsandbox.Sandbox.run`` (Python client
>= 1.14.2; documented at https://docs.coreweave.com/products/sandboxes/client/ref/core/sandbox.md). Nothing here
holds or prints credentials: the API key stays in the controller's environment, and in-sandbox object storage uses
the sandbox's own Workload Identity Federation token (object_storage_access), never a copied key.
"""
from __future__ import annotations

import re
from typing import Any

SCHEMA = 'natlang.coreweave-job/1'
BOOTSTRAP_PATH = '/opt/natlang/bootstrap.sh'
# Documented platform bounds (docs.coreweave.com/products/sandboxes/reference/limits-and-quotas.md).
MAX_LIFETIME_SECONDS = 30 * 24 * 3600
# Serverless GPU sandboxes get exactly one GPU (docs .../gpu-sandboxes.md); more needs a CKS runner.
SERVERLESS_MAX_GPUS = 1
# Sandbox price card (coreweave.com/forge-pricing/inference-agents, read 2026-10-12): CPU and memory are billed on the
# request, by the millisecond, from ready to terminated. The GPU rate is private preview and unpublished.
PRICE_CPU_CORE_HOUR = 0.128
PRICE_MEMORY_GIB_HOUR = 0.0212

KINDS = ('smoke', 'eval', 'render', 'train', 'probe')
_ID = re.compile(r'^[a-z0-9][a-z0-9-]{0,62}$')
_QUANTITY = re.compile(r'^(\d+(?:\.\d+)?)(m|Ki|Mi|Gi|Ti)?$')
_BUCKET = re.compile(r'^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$')
_SECRETISH = re.compile(r'(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)', re.I)


class JobError(ValueError):
    pass


def _quantity(value: str, what: str) -> float:
    match = _QUANTITY.match(str(value))
    if not match:
        raise JobError(f'{what} must be a Kubernetes quantity such as "16" or "128Gi", got {value!r}')
    number, unit = float(match.group(1)), match.group(2)
    return number * {None: 1, 'm': 1e-3, 'Ki': 2**-20, 'Mi': 2**-10, 'Gi': 1, 'Ti': 1024}[unit]


def cpu_cores(value: str) -> float:
    if str(value).endswith(('Ki', 'Mi', 'Gi', 'Ti')):
        raise JobError(f'cpu takes cores or millicores, got {value!r}')
    return _quantity(value, 'cpu')


def memory_gib(value: str) -> float:
    if not str(value).endswith(('Ki', 'Mi', 'Gi', 'Ti')):
        raise JobError(f'memory needs a binary unit such as "128Gi", got {value!r}')
    return _quantity(value, 'memory')


def validate(job: dict[str, Any]) -> dict[str, Any]:
    """Return the job with defaults filled in, or raise JobError naming the first problem and its fix."""
    if not isinstance(job, dict) or job.get('schema') != SCHEMA:
        raise JobError(f'a job is a JSON object with "schema": "{SCHEMA}"')
    job = dict(job)
    if not isinstance(job.get('id'), str) or not _ID.match(job['id']):
        raise JobError('id: lower-case letters, digits and dashes, at most 63 characters (it becomes a sandbox tag)')
    if job.get('kind') not in KINDS:
        raise JobError(f'kind is one of {", ".join(KINDS)}')
    placement = job.get('placement')
    if not isinstance(placement, dict) or not str(placement.get('reason', '')).strip():
        raise JobError('placement.reason: record why this job runs on CoreWeave (the placement decision)')
    if not isinstance(job.get('image'), str) or not job['image']:
        raise JobError('image: a public image reference, e.g. nvcr.io/nvidia/pytorch:26.09-py3 (x86_64 variant)')
    resources = job.get('resources')
    if not isinstance(resources, dict) or 'cpu' not in resources or 'memory' not in resources:
        raise JobError('resources needs cpu and memory: a GPU request reserves only the GPU (docs gpu-sandboxes.md)')
    cpu_cores(resources['cpu'])
    memory_gib(resources['memory'])
    gpus = resources.get('gpu', 0)
    if not isinstance(gpus, int) or gpus < 0:
        raise JobError('resources.gpu is a GPU count (0 or 1 on serverless)')
    if job.get('placement_mode', 'serverless') == 'serverless' and gpus > SERVERLESS_MAX_GPUS:
        raise JobError('serverless sandboxes get exactly 1 GPU; run independent arms as separate sandboxes, or '
                       'use placement_mode "cks" with a runner that has multi-GPU nodes')
    if job.get('kind') == 'train' and 'lifetime_hours' not in job:
        # Training: the platform maximum (owner 2026-10-12). The bootstrap exits when the job ends, so the sandbox
        # completes then; the lifetime only bounds a run that never ends, and expiry kills without grace.
        job['lifetime_hours'] = MAX_LIFETIME_SECONDS / 3600
    hours = job.get('lifetime_hours')
    if not isinstance(hours, (int, float)) or hours <= 0 or hours * 3600 > MAX_LIFETIME_SECONDS:
        raise JobError('lifetime_hours: positive and at most 720 (30 days); the platform default is only 10 minutes '
                       'and a lifetime cannot be extended after start')
    command = job.get('command')
    if not isinstance(command, list) or not command or not all(isinstance(part, str) for part in command):
        raise JobError('command: a non-empty list of strings, run from the repo snapshot after bootstrap')
    bucket = job.get('bucket')
    if bucket is not None and (not isinstance(bucket, str) or not _BUCKET.match(bucket)):
        raise JobError('bucket: an AI Object Storage bucket name (lower case, 3-63 characters)')
    if job.get('resume') and (bucket is None or not job.get('outputs')):
        raise JobError('resume fetches the published outputs from the bucket: it needs bucket and outputs')
    if job.get('bundle') is not None and bucket is None:
        raise JobError('a bundle lives in a bucket: set bucket')
    env = job.get('env', {})
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise JobError('env maps names to strings')
    leaked = sorted(name for name in env if _SECRETISH.search(name))
    if leaked:
        raise JobError(f'env must not carry credentials ({", ".join(leaked)}): object storage uses the sandbox '
                       'identity (object_storage_access); other secrets go through a secret store reference')
    scratch = job.get('scratch', {'path': '/work', 'size': '200Gi'})
    if not isinstance(scratch, dict) or not str(scratch.get('path', '')).startswith('/'):
        raise JobError('scratch: {"path": absolute mount path, "size": e.g. "600Gi"}')
    memory_gib(scratch.get('size', '200Gi'))
    storage = scratch.get('storage', 'volume')
    if storage not in ('volume', 'root'):
        raise JobError('scratch.storage is "volume" or "root" (ephemeral root filesystem, no reserved size)')
    job['scratch'] = {'path': scratch['path'], 'size': scratch.get('size', '200Gi'), 'storage': storage}
    job['env'] = env
    job.setdefault('transport', 'bucket' if bucket else 'api')
    if job['transport'] not in ('bucket', 'api') or (job['transport'] == 'bucket' and not bucket):
        raise JobError('transport is "bucket" (needs bucket) or "api" (cwctl pushes inputs and pulls outputs '
                       'through the sandbox file API)')
    if job.get('resume') and job['transport'] != 'bucket':
        raise JobError('resume fetches published outputs from the bucket: use transport "bucket"')
    job.setdefault('placement_mode', 'serverless')
    job.setdefault('tags', [])
    return job


def tags(job: dict[str, Any]) -> list[str]:
    """Sandbox tags: list/cleanup select natlang sandboxes and one job's sandboxes by these."""
    return ['natlang', f'job-{job["id"]}', f'kind-{job["kind"]}', *job['tags']]


def bootstrap_env(job: dict[str, Any]) -> dict[str, str]:
    """Non-secret environment the bootstrap reads (bootstrap.sh documents each name)."""
    env = {
        'NATLANG_CW_JOB': job['id'],
        'NATLANG_CW_WORK': job['scratch']['path'],
        'NATLANG_CW_TRANSPORT': job['transport'],
        'PYTHONUNBUFFERED': '1',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    }
    if job.get('bucket'):
        env['NATLANG_CW_BUCKET'] = job['bucket']
        env['NATLANG_CW_OUTPUT_PREFIX'] = job.get('output_prefix', f'outputs/{job["id"]}/')
    if job.get('bundle'):
        env['NATLANG_CW_BUNDLE'] = job['bundle']
    if job.get('region'):
        env['AWS_REGION'] = job['region']
    if job.get('workdir'):
        env['NATLANG_CW_WORKDIR'] = job['workdir']
    if job.get('outputs'):
        env['NATLANG_CW_OUTPUTS'] = ' '.join(job['outputs'])
    if job.get('resume'):
        # Relaunch of the same job id: fetch <output prefix><dir>/latest.json of every output before running.
        env['NATLANG_CW_RESUME'] = '1'
    env.update(job['env'])
    return env


def sandbox_request(job: dict[str, Any], bootstrap: str) -> dict[str, Any]:
    """Keyword arguments for cwsandbox.Sandbox.run (dict forms, which the client converts itself)."""
    job = validate(job)
    resources: dict[str, Any] = {'cpu': str(job['resources']['cpu']), 'memory': str(job['resources']['memory'])}
    if job['resources'].get('gpu'):
        # Serverless picks the GPU type; a `type` filter that matches nothing fails (docs gpu-sandboxes.md).
        resources['gpu'] = {'count': job['resources']['gpu']}
    request: dict[str, Any] = {
        'container_image': job['image'],
        'command': '/bin/bash',
        'args': [BOOTSTRAP_PATH, *job['command']],
        'resources': resources,
        'max_lifetime_seconds': int(job['lifetime_hours'] * 3600),
        # No max_timeout_seconds: the docs still list it, but cwsandbox 1.19.0 raises TypeError ("removed in 1.x").
        'tags': tags(job),
        'placement_mode': job['placement_mode'],
        # Mapping form {path: content}: the docs show [{"path", "content"}] but cwsandbox 1.19.0 reads list entries as
        # {"mount_path", "file_content"}; the mapping form is the one both accept unambiguously.
        'mounted_files': {BOOTSTRAP_PATH: bootstrap},
        'environment_variables': bootstrap_env(job),
        # Measured 2026-10-12 under W&B auth: the default (auto, which prefers a direct mTLS data plane) fails every
        # exec/log/file call with "Connect to sandbox data plane failed: project not found"; the gateway works.
        'data_plane_mode': job.get('data_plane_mode', 'gateway'),
    }
    if job['scratch']['storage'] == 'volume':
        request['volumes'] = [{'name': 'work', 'mount_path': job['scratch']['path'], 'size': job['scratch']['size']}]
    if job.get('bucket'):
        request['object_storage_access'] = {'buckets': [job['bucket']], 'permission': 'read_write'}
    if job.get('runner_ids'):
        request['runner_ids'] = list(job['runner_ids'])
    return request


def cost_estimate(job: dict[str, Any], gpu_hour_price: float | None = None) -> dict[str, Any]:
    """Upper bound in USD for the declared lifetime (billing follows the request, not the use)."""
    job = validate(job)
    hours = float(job['lifetime_hours'])
    cpu = cpu_cores(job['resources']['cpu']) * PRICE_CPU_CORE_HOUR
    memory = memory_gib(job['resources']['memory']) * PRICE_MEMORY_GIB_HOUR
    gpus = job['resources'].get('gpu', 0)
    gpu = None if gpus and gpu_hour_price is None else gpus * (gpu_hour_price or 0.0)
    per_hour = None if gpu is None else round(cpu + memory + gpu, 4)
    return {'cpu_memory_per_hour': round(cpu + memory, 4), 'gpu_per_hour': gpu, 'per_hour': per_hour,
            'lifetime_hours': hours, 'max_total': None if per_hour is None else round(per_hour * hours, 2),
            'gpu_rate_known': gpu is not None}
