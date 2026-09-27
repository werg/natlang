"""The trainable Spark copy computes what the released Spark computes, with SDPA, a chunked loss and per-layer
checkpointing."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "models/candidates/spark-x25-4b-hf"
pytestmark = pytest.mark.skipif(not (SOURCE / "modeling_spark.py").exists(), reason="Spark source absent")


def _load(package, modeling_source):
    sys.modules[package] = types.ModuleType(package)
    sys.modules[package].__path__ = [str(SOURCE)]
    modules = {}
    for name, source in (("configuration_spark", None), ("modeling_spark", modeling_source)):
        spec = importlib.util.spec_from_file_location(f"{package}.{name}", SOURCE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        if source is None:
            spec.loader.exec_module(module)
        else:
            exec(compile(source, str(SOURCE / f"{name}.py"), "exec"), module.__dict__)
        modules[name] = module
    return modules["configuration_spark"], modules["modeling_spark"]


def _models(attn):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from spark_training_model import compat_modeling, patch_modeling

    source = (SOURCE / "modeling_spark.py").read_text()
    values = json.loads((SOURCE / "config.json").read_text())
    # Two layer groups (three sliding, one full) and a window shorter than the sequence, so both masks matter.
    values.update(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=32, vocab_size=300, layer_types=values["layer_types"][:4],
                  sliding_window=6)
    for key in ("architectures", "auto_map", "transformers_version", "dtype"):
        values.pop(key, None)
    released_config, released = _load("spark_released", compat_modeling(source))
    patched_config, patched = _load("spark_patched", patch_modeling(source))
    torch.manual_seed(0)
    before = released.Spark2_5ForCausalLM(released_config.Spark2_5Config(**values, attn_implementation="eager"))
    after = patched.Spark2_5ForCausalLM(patched_config.Spark2_5Config(**values, attn_implementation=attn))
    after.load_state_dict(before.state_dict())
    # Transformers 5 normalizes rope_parameters; each layer type must keep its own theta and rotary share.
    assert after.config.get_rope_theta("full_attention") == 5000000
    assert after.config.get_partial_rotary_factor("full_attention") == 0.25
    assert after.config.get_rope_theta("sliding_attention") == 10000
    return before, after, patched


def test_sdpa_inference_matches_the_released_eager_model():
    import torch
    before, after, _ = _models("sdpa")
    before.eval(), after.eval()
    ids = torch.randint(0, 300, (2, 20))
    with torch.no_grad():
        torch.testing.assert_close(after(input_ids=ids).logits, before(input_ids=ids).logits, rtol=1e-4, atol=1e-4)


def test_chunked_training_loss_and_gradients_match_the_released_model_with_checkpointed_layers():
    import torch
    before, after, patched = _models("sdpa")
    before.train(), after.train()
    after.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert sum(isinstance(m, patched.GradientCheckpointingLayer) and m.gradient_checkpointing
               for m in after.modules()) == 4
    ids = torch.randint(0, 300, (2, 20))
    labels = ids.clone()
    labels[:, :5] = -100
    loss_before = before(input_ids=ids, labels=labels).loss
    loss_after = after(input_ids=ids, labels=labels).loss
    torch.testing.assert_close(loss_after, loss_before, rtol=1e-5, atol=1e-5)
    loss_before.backward(), loss_after.backward()
    grads = dict(before.named_parameters())
    for name, parameter in after.named_parameters():
        torch.testing.assert_close(parameter.grad, grads[name].grad, rtol=1e-3, atol=1e-5, msg=name)
