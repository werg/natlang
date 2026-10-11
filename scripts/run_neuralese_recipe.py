#!/usr/bin/env python3
"""Run one declared direct-stage recipe in its exact, locally available container image.

This is a launcher for the shared ``natlang_neuralese.train.recipe`` CLI, not a
second trainer. It verifies every declared input and shared-package file before
Docker starts, then records launch intent and the final process result outside
the recipe's output directory.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
NEURALESE = REPO / "training" / "neuralese"

sys.path.insert(0, str(NEURALESE))
from natlang_neuralese.common.hashing import sha256_file_hex as sha256  # noqa: E402


def under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def parse_bind(value: str) -> tuple[Path, Path, str]:
    # Host paths with ':' are intentionally unsupported: Docker's short bind
    # syntax cannot represent them portably across Pop and DGX.
    pieces = value.rsplit(":", 2)
    if len(pieces) != 3 or pieces[2] not in ("ro", "rw"):
        raise argparse.ArgumentTypeError("bind must be HOST:CONTAINER:ro|rw")
    host, target, mode = pieces
    if not host or not target or not Path(target).is_absolute():
        raise argparse.ArgumentTypeError("bind host and absolute container target are required")
    host_path = Path(host).expanduser().resolve()
    target_path = Path(target).resolve()
    if host_path != target_path:
        raise argparse.ArgumentTypeError("recipe paths are absolute; additional bind source and target must be the same path")
    return host_path, target_path, mode


def map_path(path: Path, mounts: list[tuple[Path, Path, str]]) -> Path:
    matches = [(host, target) for host, target, _ in mounts if under(path, host)]
    if not matches:
        raise ValueError(f"path is outside declared container mounts: {path}")
    host, target = max(matches, key=lambda pair: len(pair[0].parts))
    return target / path.relative_to(host)


def mount_for(path: Path, mounts: list[tuple[Path, Path, str]]) -> tuple[Path, Path, str]:
    matches = [mount for mount in mounts if under(path, mount[0])]
    if not matches:
        raise ValueError(f"path is outside declared container mounts: {path}")
    return max(matches, key=lambda mount: len(mount[0].parts))


def write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


class AttachedProcess:
    """Wait for one owned docker CLI and forward stop signals to its process group."""
    def __init__(self):
        self.child = None
        self.signal_received = None
        self.previous = {}

    def _forward(self, signum, _frame):
        self.signal_received = signum
        if self.child is not None and self.child.poll() is None:
            try:
                os.killpg(self.child.pid, signum)
            except ProcessLookupError:
                pass

    def run(self, argv, *, stdout, stderr):
        self.previous = {sig: signal.signal(sig, self._forward)
                         for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            self.child = subprocess.Popen(argv, stdout=stdout, stderr=stderr, start_new_session=True)
            if self.signal_received is not None:
                self._forward(self.signal_received, None)
            return self.child.wait()
        finally:
            for sig, handler in self.previous.items():
                signal.signal(sig, handler)
            self.previous.clear()


def read_direct_recipe(path: Path) -> dict:
    recipe = json.loads(path.read_text())
    if recipe.get("schema") != "natlang.neuralese-declared-direct-stage/1":
        raise ValueError("only declared direct-stage recipes can use this container launcher")
    if recipe.get("execution") != "direct_shared_stage":
        raise ValueError("recipe must declare execution=direct_shared_stage")
    runtime = recipe.get("runtime")
    image = runtime.get("image") if isinstance(runtime, dict) else None
    if not isinstance(image, str) or not image.startswith("sha256:") or len(image) != 71:
        raise ValueError("recipe must pin a sha256 container image")
    if not isinstance(runtime.get("package_code"), dict) or not runtime["package_code"]:
        raise ValueError("recipe must pin the shared package_code inventory")
    if not isinstance(recipe.get("inputs"), dict) or not recipe["inputs"]:
        raise ValueError("recipe must declare pinned input files")
    if not isinstance(recipe.get("lineage"), dict) or not isinstance(recipe.get("lineage", {}).get("evidence", []), list):
        raise ValueError("recipe lineage/evidence declaration is invalid")
    outputs = recipe.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(outputs.get("directory"), str):
        raise ValueError("recipe must declare outputs.directory")
    stage = recipe.get("stage")
    if not isinstance(stage, dict) or not isinstance(stage.get("id"), str) or not isinstance(stage.get("kind"), str):
        raise ValueError("recipe must declare stage.id and stage.kind")
    return recipe


def check_file_binding(role: str, binding: dict) -> dict:
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError(f"invalid pinned file binding: {role}")
    path = Path(binding["path"]).expanduser().resolve()
    expected = binding["sha256"]
    if not path.is_file():
        raise ValueError(f"missing declared input: {role} ({path})")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"declared input hash mismatch: {role}")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("/srv/storage"),
                        help="host artifact mount root (default: /srv/storage)")
    parser.add_argument("--hf-hub-cache", type=Path,
                        help="host Hugging Face hub directory; defaults to HF_HUB_CACHE, HF_HOME/hub, repo models/hf/hub, then ~/.cache/huggingface/hub")
    parser.add_argument("--bind", action="append", type=parse_bind, default=[], metavar="HOST:CONTAINER:ro|rw",
                        help="additional same-path bind for an input outside the default mounts")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--launch-metadata", type=Path,
                        help="external intent/result receipt; default is beside outputs.directory")
    parser.add_argument("--log", type=Path, help="external combined container log; default beside receipt")
    args = parser.parse_args(argv)

    recipe_path = args.recipe.expanduser().resolve()
    recipe = read_direct_recipe(recipe_path)
    artifact_root = args.artifact_root.expanduser().resolve()
    configured_hub = (args.hf_hub_cache or
                      (Path(os.environ["HF_HUB_CACHE"]) if os.environ.get("HF_HUB_CACHE") else None) or
                      (Path(os.environ["HF_HOME"]) / "hub" if os.environ.get("HF_HOME") else None) or
                      (REPO / "models" / "hf" / "hub" if (REPO / "models" / "hf" / "hub").is_dir() else None) or
                      (Path.home() / ".cache" / "huggingface" / "hub"))
    hf_hub_cache = configured_hub.expanduser().resolve()
    output = Path(recipe["outputs"]["directory"]).expanduser().resolve()
    if not artifact_root.is_dir():
        raise ValueError(f"artifact root must already exist: {artifact_root}")
    if not under(output, artifact_root) or output == artifact_root:
        raise ValueError("outputs.directory must be a child of the explicit artifact root")
    if output.exists():
        raise ValueError(f"fresh output directory required: {output}")
    metadata = args.launch_metadata.expanduser().resolve() if args.launch_metadata else output.parent / (output.name + ".container-launch.json")
    if metadata.exists() or under(metadata, output):
        raise ValueError("launch metadata must be a fresh path outside the child output")
    log_path = args.log.expanduser().resolve() if args.log else metadata.with_suffix(".log")
    if log_path.exists() or under(log_path, output):
        raise ValueError("container log must be a fresh path outside the child output")
    if log_path == metadata:
        raise ValueError("container log and launch receipt must use different paths")
    child_metadata = output.parent / (output.name + ".recipe-launch.json")
    if child_metadata.exists() or under(child_metadata, output) or child_metadata == metadata or child_metadata == log_path:
        raise ValueError("fresh external child recipe metadata path required")
    if not hf_hub_cache.is_dir():
        raise ValueError(f"HF hub cache must exist and be a directory: {hf_hub_cache}")

    input_bindings = {role: check_file_binding(role, binding) for role, binding in recipe["inputs"].items()}
    lineage = []
    for item in recipe["lineage"].get("evidence", []):
        checked = check_file_binding("lineage:" + item.get("role", ""),
                                     {"path": item.get("path"), "sha256": item.get("sha256")})
        lineage.append({"role": item["role"], **checked})
    package = NEURALESE / "natlang_neuralese"
    package_code = {str(path.relative_to(package)): sha256(path)
                    for path in sorted(package.rglob("*.py")) if path.is_file()}
    if package_code != recipe["runtime"]["package_code"]:
        raise ValueError("shared Python package differs from pinned runtime.package_code inventory")

    hf_home = hf_hub_cache.parent
    mounts = [(REPO, REPO, "ro"), (artifact_root, artifact_root, "rw"), (hf_home, hf_home, "rw")]
    for bind in args.bind:
        host, target, mode = bind
        if not host.exists():
            raise ValueError(f"additional bind source does not exist: {host}")
        if any(host == existing_host or target == existing_target for existing_host, existing_target, _ in mounts):
            raise ValueError("additional bind duplicates an existing host or container mount")
        mounts.append((host, target, mode))
    for binding in recipe["inputs"].values():
        map_path(Path(binding["path"]).expanduser().resolve(), mounts)
    for item in recipe["lineage"].get("evidence", []):
        map_path(Path(item["path"]).expanduser().resolve(), mounts)
    container_recipe = map_path(recipe_path, mounts)
    container_output = map_path(output, mounts)
    container_metadata = map_path(child_metadata, mounts)
    container_hub_cache = map_path(hf_hub_cache, mounts)
    container_hf_home = container_hub_cache.parent

    docker = shutil.which(args.docker)
    if docker is None:
        raise ValueError(f"Docker CLI not found: {args.docker}")
    image = recipe["runtime"]["image"]
    inspected = subprocess.run([docker, "image", "inspect", image, "--format", "{{.Id}}"],
                               text=True, capture_output=True, check=False)
    if inspected.returncode or inspected.stdout.strip() != image:
        raise ValueError("the exact declared image is not available locally; launcher will not pull or substitute it")

    # Create only the external metadata/log parents, then atomically reserve the
    # receipt path before Docker is invoked. The recipe CLI owns its output tree.
    metadata.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    # JIT kernels must not write to the image's read-only home directory.
    # Keep their caches in the same declared artifact mount as this run.
    kernel_cache = output.parent / "kernel-cache"
    kernel_cache.mkdir(exist_ok=True)
    container_kernel_cache = map_path(kernel_cache, mounts)
    if output.exists():
        raise ValueError("output appeared during preflight; refusing to launch")
    child = ["-m", "natlang_neuralese.train.recipe", "--recipe", str(container_recipe),
             "--out", str(container_output), "--launch-metadata", str(container_metadata),
             "--device", args.device]
    common_docker = [docker, "run", "--rm", "--pull=never", "--network", "none", "--read-only",
                     "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g",
                     "--user", f"{os.getuid()}:{os.getgid()}", "-w", str(map_path(REPO, mounts))]
    for host, target, mode in mounts:
        common_docker += ["-v", f"{host}:{target}:{mode}"]
    common_docker += ["-e", f"HF_HOME={container_hf_home}", "-e", f"HF_HUB_CACHE={container_hub_cache}",
                      "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
                      "-e", "PYTHONDONTWRITEBYTECODE=1",
                      "-e", f"TRITON_CACHE_DIR={container_kernel_cache / 'triton'}",
                      "-e", f"TORCHINDUCTOR_CACHE_DIR={container_kernel_cache / 'inductor'}",
                      "-e", f"TORCH_EXTENSIONS_DIR={container_kernel_cache / 'extensions'}",
                      "-e", f"XDG_CACHE_HOME={container_kernel_cache / 'xdg'}",
                      "-e", f"PYTHONPATH={map_path(NEURALESE, mounts)}"]
    preflight_code = (
        "import importlib,json,os,sys; "
        "from natlang_neuralese.train.recipe import HANDLERS; "
        "module=HANDLERS[sys.argv[1]]['module']; importlib.import_module(module); "
        "hub=sys.argv[2]; assert os.path.isdir(hub) and os.access(hub,os.R_OK), 'HF hub cache is not readable'; "
        "files=sys.argv[3:]; missing=[p for p in files if not os.path.isfile(p)]; "
        "assert not missing, 'declared recipe inputs are not visible in their same-path mounts: '+','.join(missing); "
        "models=sorted(n for n in os.listdir(hub) if n.startswith('models--')); "
        "print(json.dumps({'stage_module':module,'hf_hub_cache':hub,'cached_model_repositories':models}))"
    )
    preflight_paths = [item["path"] for item in input_bindings.values()] + [item["path"] for item in lineage]
    preflight = [*common_docker, "--gpus", "all", "-e", "CUDA_VISIBLE_DEVICES=", "--entrypoint", "/usr/bin/python3",
                 image, "-c", preflight_code, recipe["stage"]["kind"], str(container_hub_cache), *preflight_paths]
    command = [*common_docker]
    command += ["--gpus", "all"]
    if args.device == "cpu":
        command += ["-e", "CUDA_VISIBLE_DEVICES="]
    command += ["--entrypoint", "/usr/bin/python3", image, *child]
    output_mount = mount_for(output, mounts)
    if output_mount[2] != "rw":
        raise ValueError("the most-specific bind covering outputs.directory must be read-write")
    intent = {"schema": "natlang.neuralese-container-launch/1", "status": "launch_intent",
              "created_unix": time.time(), "recipe_id": recipe["id"], "recipe_path": str(recipe_path),
              "recipe_sha256": sha256(recipe_path), "runtime_image": image, "stage": recipe["stage"],
              "inputs": input_bindings, "lineage_evidence": lineage, "package_code_count": len(package_code),
              "artifact_root": str(artifact_root), "output": str(output), "launch_metadata": str(metadata),
              "hf_hub_cache": str(hf_hub_cache),
              "log": str(log_path), "device": args.device, "mounts": [
                  {"host": str(host), "container": str(target), "mode": mode} for host, target, mode in mounts],
              "child_recipe_metadata": str(child_metadata), "python_executable": "/usr/bin/python3",
              "preflight_argv": preflight, "preflight_scope": "stage_import_and_input_cache_visibility; model_not_loaded",
              "docker_argv": command, "network": "none", "offline": True}
    write_exclusive(metadata, intent)
    try:
        with log_path.open("xb") as log:
            attached = AttachedProcess()
            cpu_preflight = attached.run(preflight, stdout=log, stderr=subprocess.STDOUT)
            preflight_signal = attached.signal_received
            if preflight_signal is not None:
                completed_code = 128 + preflight_signal
                phase = "cpu_preflight_interrupted"
            elif cpu_preflight:
                completed_code = cpu_preflight
                phase = "cpu_preflight"
            else:
                if output.exists():
                    raise ValueError("output appeared during CPU preflight; refusing to start the stage")
                attached = AttachedProcess()
                completed_code = attached.run(command, stdout=log, stderr=subprocess.STDOUT)
                if attached.signal_received is not None:
                    preflight_signal = attached.signal_received
                    completed_code = 128 + attached.signal_received
                    phase = "training_recipe_cli_interrupted"
                else:
                    phase = "training_recipe_cli"
        child_result_path = output / "direct-stage-result.json"
        child_result = json.loads(child_result_path.read_text()) if child_result_path.is_file() else None
        child_metadata_sha = sha256(child_metadata) if child_metadata.is_file() else None
        result = {**intent, "status": "process_succeeded" if completed_code == 0 else "process_failed",
                  "finished_unix": time.time(), "exit_code": completed_code,
                  "signal_received": signal.Signals(preflight_signal).name if preflight_signal else None,
                  "failure_phase": None if completed_code == 0 else phase,
                  "log_sha256": sha256(log_path), "log_bytes": log_path.stat().st_size,
                  "child_result": str(child_result_path) if child_result is not None else None,
                  "child_result_sha256": sha256(child_result_path) if child_result is not None else None,
                  "child_stage_completed": child_result.get("completed") if child_result else None,
                  "child_stage_qualified": child_result.get("qualified") if child_result else None,
                  "child_recipe_metadata_sha256": child_metadata_sha}
        temp = metadata.with_suffix(metadata.suffix + ".pending")
        temp.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        os.replace(temp, metadata)
        return completed_code
    except BaseException as error:
        result = {**intent, "status": "launcher_error", "finished_unix": time.time(),
                  "error_type": type(error).__name__, "error": str(error)}
        temp = metadata.with_suffix(metadata.suffix + ".pending")
        temp.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        os.replace(temp, metadata)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
