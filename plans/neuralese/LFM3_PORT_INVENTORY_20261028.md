# LFM3 port inventory (2026-10-28, read-only sweep of main @ 96a2c8029..04d6a5c89)

Companion to `LFM3_PORT_20261028.md`. Line numbers are as of the sweep; re-check before editing.

## Model: forward and state
- Extend `model/hf_port.py:64` `QwenPortBackbone` (Mellum `maple/maple_port.py:15` only overrides `_apply_rope` with per-layer-type tables from `maple/model.py:176` `LayerTypeRotary`; Qwen3.5 precedent `docs/QWEN35_NATIVE_SUPPORT.md`). It already has full/sliding attention (`window()` :138, banded/FlexAttention :348-416), a sigmoid output gate (:242-278, but Qwen3.5's per-element in-q_proj form), `_apply_rope` hook (:151). Missing: conv layers (`conv_window=0`), LFM names, sandwich post-norms, per-head gate from normed input, `_run_checkpointed` (:300-341) only knows Attention/DeltaState.
- `model/lfm2_port.py`: DEFAULT_BASE LFM2.5 + reserved_7/8 markers (:27-34, :245-294); causal-conv1d kernel builds (:297-423) support widths 2-4 only (LFM3 width 7); `feed_forward_residual` (:426-435) lacks post_ffn_norm (used by shared_state :91/:99, isolated_sequence); `PortBackbone` (:439-717) LFM2-only (q_layernorm/out_proj, attention == full only, one rotary, LFM2 apply_rotary_pos_emb import :621).
- `model/shared_state.py`: `_rotary_tables`/`_rotate_pair` (:45-57) single table + LFM2 fallback; `_layer_inputs`/`_finish` (:60-100) gate only for qwen3_5, no post norms; native_step/import validation (:997-1092, :1126-1184, :1264); role-token provisioning (:548, :925-990) adds 4 tokens (fits unused rows 125033+).
- `model/isolated_sequence.py:4, 106-182`: chooses LFM2 vs Qwen names by `hasattr(_apply_rope)`; LFM3 matches neither; no window on LFM path.
- `model/native_decode_batch.py:11, 61-93`: rotary per start position only; per-element gate only.
- `model/native_model.py:174-199` dispatch `{'lfm2','lfm2_moe'}`; `model/native_backbone_restore.py:11, 28-44` trust_remote_code False, no lfm3.
- Norm/conv type sets: `model/native_parameter_precision.py:9`, `model/shared_execution.py:8`, `model/native_weight_carrier.py:21, 123-140, 250-254` (Lfm2RMSNorm, Lfm2ShortConv, causal_conv1d_fn).
- `serve/batch.py:17-86` TRAP: LFM2 fast path chosen by `hasattr(layer,'operator_norm')` (LFM3 has it) then uses q_layernorm/out_proj; no gate/window/sandwich.

## Training
- `train/adapters.py:15-43` ATTENTION_TARGETS `out_proj` (LFM3 `o_proj` + `self_attn.gate_proj`); `train/fp8_backend.py:18-26, 73-111` model_type lfm2 only; `train/memory_estimator.py:25-55` misses q_norm/post norms; `train/native_state_sft.py:1618-1674` model_id = architecture:inventory sha; `train/native_resident.py:436, 670-933` 'lfm25' keys/bridge layout; `train/native_capacity_continuation.py:145-175` pinned 'lfm' code-review compositions; `train/native_deployment.py:228-319` role lfm -> 'lfm25', port 8097; `train/quantization.py:34-36,107,478` output_convention lfm = Pythonic; `train/line_parity.py:10`.

## Serving, format, data
- `serve/chat.py`: Pythonic `_CALLS` (:35); `_JSON_CALLS` (:37) matches `<tool_call>` but `parse_json_call` (:675) expects JSON so XML fails; `BACKBONE_HISTORY_REASONING` (:307) lacks lfm3; `_literal` (:617); `call_reply`/`write_reply` (:500-543) require quoted string args -> every Neuralese write site raises neuralese-template under XML.
- `serve/guidance.py:36, 144` `<|tool_call_start|>` forced prefix and quoted `eval(code='...')` decoder; `serve/http.py:480-481` stream markers.
- `data/render.py:27-49, 79, 123-125, 178-187` TRAP: detects LFM2 by `<|tool_call_start|>` in vocab (LFM3 passes); BOS as attention sink.
- `data/text_corpus.py:469, 1882-1895` last_turn_only history projection (LFM3 injects `<think></think>` into earlier turns: re-prove).
- `scripts/neuralese_data/records.py:37-41` markup list; `text_tokenizer_twin.py:78, 144-234` parses LFM2 Pythonic calls as structure; twin shell scripts hard-code the LFM2.5 snapshot; `--tokenizer lfm25=...` in render_native_turn_corpus, build_native_turn_union, target_set_*, swe_bulk_*, register_swe_bulk.
- Hard-coded LFM2.5 identities (fingerprints e52de346/be949ec4, `lfm2:` model ids, 'lfm25' keys): scripts/coreweave/{teacher_internal_conversion,teacher_internal_admission_worker,teacher_tranche_admission,publish_teacher_internal_tranche,admit_completed_search_shards,prepare_source_correction_continuation,assemble_native_graph_union,build_verified_teacher_trajectory_graphs,build_static_repository_graphs,prepare_private_teacher_append,prepare_recovery_teacher_queue,verify_completed_search_parent,run_history_reasoning_live_window,prepare_local_cycle_shared_semantic,cursor_checkpoint_export}.py.
- `training/neuralese_corpora.json`: 1383/2766 entries mention LFM (renders.<view>.lfm25, native_graph_views.lfm25, tokenizer_twin_of).

## Export / llama.cpp
- `export/native_gguf.py:31, 61` arch whitelist; vendored converter snapshot `vendor/llama.cpp` -> `.development-source-snapshots/llama.cpp-972d2313` has no lfm3. `tune/soft_export.py:4,33-36`, `tune/targets.py:18-33` TABLES['lfm2'], `tune/stack_export.py:8`, `tune/export.py:125`.
- Fork `/home/werg/llama.cpp-neuralese` (branch neuralese @ d71b8c85b, base 1537a0a8b): no lfm3; layer-range execution already removed (413e981db). Hooks in `src/models/lfm2.cpp` (:48-54 producer adapter, :204-209 capture/state span on pre-RoPE k and v, :255-258 conv input B*x, :309-310 embedding capture/producer, :356). LFM2 gates `src/llama-context.cpp:2762, 2812`; `src/llama-graph.cpp:2769-2828`, `src/llama-state-span.h`, `common/native-state.cpp`, `tools/server/server-native-state.cpp:161-347`, `tools/neuralese/state-span-dump.cpp`, `tools/server/server-guidance.cpp:94-212` (Pythonic eval). Upstream with LFM3: ggml-org/llama.cpp d05706b (clone at /home/werg/data/lfm3-eval/llama.cpp). Partial RoPE layout risk: HF zero-tail pairs i/i+64; check the converter's k_proj permutation before exchanging pre-RoPE raw_k.

## Recipes, eval, harness, tests
- `recipes/native-state-production-base-20261015-v1.json` is LFM by default (lfm25-qad base, float32, lfm25 inputs); `line-parity.json:5-70`; `hardware-profiles.json:19-20` (350M sizing); DECISIONS #5, #32, #35-36, QAD base section, task-eval budgets.
- `eval/generation_checks.py:47-80` output_convention lfm; `common/memetic_teacher_ferry.py:35,91,96-131`, `rl/native_rounds.py:65`, `eval/native_async.py:188`, `eval/native_async_service.py:378`, `scripts/task_eval_dispatch.py:59` architecture sets.
- ts-host: `src/model-default.ts:3-20` (LFM2.5 GGUF default, logit_bias 64402), `src/tune/settings.ts:9-12,150,186`, `applications/pi/host/agent-models.ts:44-47`; `src/native/tool-markup.ts:2` already covers XML.
- Tests: `tests/neuralese/conftest.py:58-66` real LFM2.5 fixture; tiny Lfm2Config tests; ~68 test files reference lfm2; `scripts/check-main.sh:117` Lfm2Config probe.
