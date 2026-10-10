"""Conversion v3 (plans/mellum-port.md): token-id records teacher, prompt-weighted KL to the BF16 model, and the
generation gate (greedy through the port's KV cache, chat-structure checks)."""
import json

import pytest
import torch

from test_mellum_port import tiny_mellum

from natlang_neuralese.maple.model import load_maple

EOS = 7


def test_gate_greedy_through_the_cache_matches_full_recompute(tmp_path):
    from natlang_neuralese.maple.generation_gate import deployed_cache, greedy
    from natlang_neuralese.maple.qat_convert import install_full_latent_qat
    from natlang_neuralese.maple.ternary import QUANT_MIX

    tiny_mellum(tmp_path)
    model = load_maple(tmp_path, device="cpu", dtype=torch.float32, ternary_attention=False).eval()
    install_full_latent_qat(model)
    prompt = torch.randint(0, 128, (11,)).tolist()  # prompt + continuation crosses the sliding window (8)
    try:
        for mix in (0.0, 1.0):
            QUANT_MIX["value"] = mix
            with deployed_cache():
                cached = greedy(model, prompt, 9, eos=-1)
            naive = list(prompt)
            with torch.no_grad():
                for _ in range(9):
                    naive.append(int(model(torch.tensor([naive])).logits[0, -1].argmax()))
            assert cached == naive[len(prompt):]
    finally:
        QUANT_MIX["value"] = 1.0
        QUANT_MIX["cache"] = None


def test_gate_checks_chat_structure_tool_calls_loops_and_answers():
    from natlang_neuralese.maple.generation_gate import check

    tools = [{"type": "function", "function": {"name": "weather", "parameters": {"required": ["city"]}}}]
    thinking = {"enable_thinking": True, "answer": "42"}
    assert all(check(thinking, "<think>\nsum</think>\n\n42<|im_end|>", [1, 2, EOS], EOS).values())
    assert not check(thinking, "42<|im_end|>", [1, EOS], EOS)["think"]
    assert not check(thinking, "<think>41</think> 41", [1, 2], EOS)["end"]
    assert not check({"answer": "42"}, "<think>x</think>42", [1, EOS], EOS)["think"]  # thinking off: no new block
    probe = {"kind": "tool", "tools": tools, "expected_tool": "weather"}
    good = '<tool_call>\n{"name": "weather", "arguments": {"city": "Berlin"}}\n</tool_call>'
    assert check(probe, good, [1, EOS], EOS)["tool"]
    assert not check(probe, good.replace('"city"', '"town"'), [1, EOS], EOS)["tool"]
    assert not check(probe, good.replace("}}", "}"), [1, EOS], EOS)["tool"]
    assert not check({}, "", [3, 4, 5, 6, 7, 8, 9, 10] * 4, EOS)["no_loop"]


def test_prompt_weighted_kl_weights_positions_and_vanishes_against_itself(tmp_path):
    from natlang_neuralese.maple.qat_convert import weighted_topk_kl

    tiny_mellum(tmp_path)
    model = load_maple(tmp_path, device="cpu", dtype=torch.float32, ternary_attention=False).eval()
    ids = torch.randint(0, 128, (1, 20))
    with torch.no_grad():
        logp = torch.log_softmax(model(ids).logits[0, :-1].float(), -1)
    values, index = logp.topk(128, -1)
    own = weighted_topk_kl(model, ids, index, values, prompt_len=6, prompt_weight=0.25, chunk=7)
    assert float(own["loss_kl"]) < 1e-5 and float(own["response_agreement"]) == 1.0
    assert torch.isclose(own["response_ce"], -logp[5:].gather(-1, ids[0, 6:, None]).mean(), atol=1e-5)
    teacher = torch.log_softmax(torch.randn(19, 128), -1)
    values, index = teacher.topk(16, -1)
    scores = weighted_topk_kl(model, ids, index, values, prompt_len=6, prompt_weight=0.25, chunk=7)
    t = values - torch.logsumexp(values, -1, keepdim=True)
    kl = (t.exp() * (t - logp.gather(-1, index))).sum(-1)
    weight = torch.tensor([0.25] * 5 + [1.0] * 14)
    assert torch.isclose(scores["loss_kl"], (weight * kl).sum() / weight.sum(), atol=1e-5)
    assert torch.isclose(scores["prompt_kl"], kl[:5].mean(), atol=1e-5)
    assert torch.isclose(scores["response_kl"], kl[5:].mean(), atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the conversion trainer (fused Lion) is CUDA-only")
def test_records_teacher_and_kl_only_training_run_and_resume(tmp_path):
    from natlang_neuralese.maple.qat_convert import main

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tiny_mellum(model_dir)
    generator = torch.Generator().manual_seed(1)
    with open(tmp_path / "records.jsonl", "w") as handle:
        for i in range(9):
            prompt = torch.randint(0, 128, (5 + i,), generator=generator).tolist()
            response = torch.randint(0, 128, (4 + i % 3,), generator=generator).tolist()
            handle.write(json.dumps({"id": f"r{i}", "source": "test", "enable_thinking": i % 2 == 0,
                                     "prompt_ids": prompt, "response_ids": response}) + "\n")
    teacher = tmp_path / "teacher"
    main(["teacher", "--model", str(model_dir), "--records", str(tmp_path / "records.jsonl"), "--out", str(teacher),
          "--k", "128", "--held", "2", "--shard", "3", "--tokens", "64"])
    meta = json.loads((teacher / "teacher.json").read_text())
    assert meta["format"] == "records" and (meta["train"], meta["test"]) == (7, 2)
    assert meta["shards"]["train"] == ["train-0000.pt", "train-0001.pt", "train-0002.pt"]
    shard = torch.load(teacher / "train-0000.pt")
    assert shard["top_ids"][0].shape == (len(shard["ids"][0]) - 1, 128)
    out = tmp_path / "train"
    args = ["train", "--model", str(model_dir), "--teacher", str(teacher), "--out", str(out), "--ce-weight", "0",
            "--ramp-steps", "2", "--eval-every", "2", "--held-eval", "2", "--checkpoint-every", "2"]
    main(args + ["--steps", "2"])
    main(args + ["--steps", "4"])  # resumes from the rolling slot
    rows = [json.loads(line) for line in open(out / "train.jsonl")]
    assert [r["step"] for r in rows] == [0, 1, 2, 3, 4]
    assert "held_kl_bf16" in rows[0] and rows[0]["held_kl_bf16"] < 1e-2  # mix 0 is the teacher (full-vocabulary top-k)
    assert all("held_kl" in rows[i] for i in (0, 2, 4)) and [r["mix"] for r in rows[1:]] == [0.0, 0.5, 1.0, 1.0]
