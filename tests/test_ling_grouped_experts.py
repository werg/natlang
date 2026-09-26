"""The stacked-experts rewrite of Ling's MoE block computes what the released block computes."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "models/candidates/ling3-tiny-hf"
pytestmark = pytest.mark.skipif(not (SOURCE / "modeling_bailing_moe_v3.py").exists(), reason="Ling source absent")


def _load(directory, package, modeling_source):
    import transformers.utils.import_utils as import_utils
    if not hasattr(import_utils, "is_torch_fx_available"):  # the released code imports it; Transformers 5 lacks it
        import_utils.is_torch_fx_available = lambda: False
    sys.modules[package] = types.ModuleType(package)
    sys.modules[package].__path__ = [str(directory)]
    modules = {}
    for name, source in (("configuration_bailing_moe_v3", None), ("modeling_bailing_moe_v3", modeling_source)):
        spec = importlib.util.spec_from_file_location(f"{package}.{name}", directory / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        if source is None:
            spec.loader.exec_module(module)
        else:
            exec(compile(source, str(directory / f"{name}.py"), "exec"), module.__dict__)
        modules[name] = module
    return modules["configuration_bailing_moe_v3"], modules["modeling_bailing_moe_v3"]


def test_stacked_experts_match_the_released_block_in_training_and_inference():
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from ling_grouped_experts import patch_modeling, stack_experts

    source = (SOURCE / "modeling_bailing_moe_v3.py").read_text()
    config_values = json.loads((SOURCE / "config.json").read_text())
    config_values.update(hidden_size=64, moe_intermediate_size=32, moe_shared_expert_intermediate_size=32, num_experts=16,
                         n_group=4, topk_group=2, num_experts_per_tok=4)
    config_module, released = _load(SOURCE, "ling_released", source)
    _, grouped = _load(SOURCE, "ling_grouped", patch_modeling(source))
    config = config_module.BailingMoeV3Config(**config_values)
    torch.manual_seed(0)
    original = released.BailingMoeV3SparseMoeBlock(config).float()
    for parameter in original.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    rewritten = grouped.BailingMoeV3SparseMoeBlock(config).float()
    state = {k: v for k, v in original.state_dict().items() if ".experts." not in f".{k}"}
    parts = {(i, name): getattr(expert, name).weight for i, expert in enumerate(original.experts)
             for name in ("gate_proj", "up_proj", "down_proj")}
    state["experts.gate_up_proj"], state["experts.down_proj"] = stack_experts(parts, config.num_experts)
    rewritten.load_state_dict(state, strict=True)
    x = torch.randn(2, 19, config.hidden_size)
    for mode in ("eval", "train"):
        getattr(original, mode)(); getattr(rewritten, mode)()
        with torch.no_grad():
            torch.testing.assert_close(rewritten(x)[0], original(x)[0], atol=1e-5, rtol=1e-5)


def test_chunked_loss_equals_the_transformers_causal_lm_loss_and_its_gradient():
    import torch
    from transformers.loss.loss_utils import ForCausalLMLoss
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from ling_grouped_experts import patch_modeling

    _, grouped = _load(SOURCE, "ling_loss", patch_modeling((SOURCE / "modeling_bailing_moe_v3.py").read_text()))
    torch.manual_seed(0)
    head = torch.nn.Linear(16, 50, bias=False)
    hidden = torch.randn(1, 1100, 16, requires_grad=True)
    labels = torch.randint(0, 50, (1, 1100))
    labels[:, :300] = -100
    chunked = grouped._chunked_causal_lm_loss(head, hidden, labels, block=256)
    (chunked_grad,) = torch.autograd.grad(chunked, hidden)
    reference = ForCausalLMLoss(head(hidden), labels, 50)
    (reference_grad,) = torch.autograd.grad(reference, hidden)
    torch.testing.assert_close(chunked, reference)
    torch.testing.assert_close(chunked_grad, reference_grad)
