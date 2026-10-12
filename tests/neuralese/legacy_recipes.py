"""Inline legacy-lineage recipes for the recipe-mechanism tests.

The token-foundation and raw-recurrence recipe files were deleted with the discarded designs (owner decision
2026-10-12, plans/REVIEW_20261012.md rec. 9). The loader still validates such lineages (identity and embedding gates,
mapped warm-up, AR fixup, adapted runtime, recurrence), so the tests keep exercising those mechanisms on these inline
copies (foundation-v1 and raw-recurrence-v1 as of 03339381, renamed)."""
import copy

_FOUNDATION = {
 "schema": "natlang.neuralese-training-recipe/1",
 "id": "legacy-foundation",
 "description": "Inline fixture: a legacy token-foundation lineage (identity, embedding distillation, runtime qualification).",
 "stages": [
  {
   "id": "token_identity",
   "kind": "token_identity",
   "requires": [],
   "parameters": {
    "limit": 32,
    "max_tokens": 512
   }
  },
  {
   "id": "embedding_distillation",
   "kind": "causal_embedding_distillation",
   "requires": [
    "token_identity"
   ],
   "parameters": {
    "cutoff": "full",
    "steps": 8192,
    "batch": 512,
    "lr": 0.0003,
    "tokens": 1024,
    "contexts": 128,
    "context_tokens": 1024,
    "eval_every": 128,
    "checkpoint_every": 128,
    "agreement_gate": 0.9,
    "kl_gate": 0.25,
    "source_fraction": 0.25,
    "argmax_weight": 0.1,
    "stop_on_gate": True,
    "seed": 0
   }
  },
  {
   "id": "runtime_qualification",
   "kind": "raw_runtime_qualification",
   "requires": [
    "embedding_distillation"
   ],
   "parameters": {
    "limit": 8,
    "max_length": 128
   }
  }
 ]
}

_RAW_RECURRENCE = {
 "schema": "natlang.neuralese-training-recipe/1",
 "id": "legacy-raw-recurrence",
 "description": "Inline fixture: a legacy foundation + mapped text warm-up + AR fixup + adapted runtime + recurrence lineage.",
 "stages": [
  {
   "id": "token_identity",
   "kind": "token_identity",
   "requires": [],
   "parameters": {
    "limit": 32,
    "max_tokens": 512
   }
  },
  {
   "id": "embedding_distillation",
   "kind": "causal_embedding_distillation",
   "requires": [
    "token_identity"
   ],
   "parameters": {
    "cutoff": "full",
    "steps": 8192,
    "batch": 512,
    "lr": 0.0003,
    "tokens": 1024,
    "contexts": 128,
    "context_tokens": 1024,
    "eval_every": 128,
    "checkpoint_every": 128,
    "agreement_gate": 0.9,
    "kl_gate": 0.25,
    "source_fraction": 0.25,
    "argmax_weight": 0.1,
    "stop_on_gate": True,
    "seed": 0
   }
  },
  {
   "id": "runtime_qualification",
   "kind": "raw_runtime_qualification",
   "requires": [
    "embedding_distillation"
   ],
   "parameters": {
    "limit": 8,
    "max_length": 512
   }
  },
  {
   "id": "core_text_warmup",
   "kind": "core_text_warmup",
   "requires": [
    "runtime_qualification"
   ],
   "parameters": {
    "steps": 4096
   }
  },
  {
   "id": "autoregressive_text_fixup",
   "kind": "core_text_warmup",
   "requires": [
    "core_text_warmup"
   ],
   "parameters": {
    "steps": 1024,
    "ar_feedback_fixup": True,
    "channel_consistency_weight": 1.0,
    "tokens": 16384,
    "target_tokens": 256
   }
  },
  {
   "id": "adapted_runtime",
   "kind": "text_warmup_runtime",
   "requires": [
    "autoregressive_text_fixup"
   ],
   "parameters": {}
  },
  {
   "id": "recurrence_warmup",
   "kind": "raw_recurrence_training",
   "requires": [
    "autoregressive_text_fixup",
    "adapted_runtime"
   ],
   "parameters": {
    "steps": 2048,
    "batch": 1,
    "lr": 0.0001,
    "backbone_training": "auto",
    "backbone_lr": 3e-05,
    "rank": 16,
    "lora_lr": 2e-05,
    "max_tokens": 65536,
    "train": 2000,
    "eval": 28,
    "handover": "written",
    "write_curriculum": "joint",
    "max_writes": 0,
    "write_depth": 5,
    "tokens_per_vector": 1,
    "max_write_vectors": 512,
    "heads_lr": 3e-05,
    "distill": 0.25,
    "crisp_weight": 1,
    "backward_policy": "auto",
    "graph_headroom_gb": 0.25,
    "checkpoint_layers": True,
    "ffn_chunk_tokens": 1024,
    "optimizer": "muon",
    "checkpoint_every": 25,
    "eval_every": 128,
    "seed": 0,
    "writer_text_weight": 1.0,
    "content_transport": "top-state",
    "writer_length_policy": "native-value",
    "writer_supervision": "native-value",
    "stop_supervision": "gold-native-boundary",
    "sketch_gradient": "local_stage",
    "sketch_target_weight": 0.1,
    "sketch_target_backbone_scale": 0.05,
    "train_control_rows": True,
    "local_stage_batch_size": 16
   }
  }
 ],
 "stage_parameter_defaults": {
  "core_text_warmup": {
   "tokens": 1024,
   "prefix_tokens": 32,
   "cutoff": 4,
   "backbone_training": "auto",
   "optimizer": "muon",
   "eval_every": 128,
   "checkpoint_every": 128,
   "lr": 3e-05,
   "sketch_lr": 0.0003,
   "embedding_weight": 1.0,
   "sketch_weight": 1.0,
   "text_weight": 0.25,
   "max_ce_delta": 0.1,
   "max_relative_mse": 0.25,
   "min_agreement": 0.9,
   "consecutive_gates": 2,
   "batch": 2,
   "projection_patience": 3,
   "projection_min_evals": 2,
   "projection_min_improvement": 0.01,
   "backbone_ramp_evals": 4,
   "pass_ramp_evals": 2,
   "input_map_kernel": 4,
   "input_map_rank": 64
  }
 }
}


def foundation():
    return copy.deepcopy(_FOUNDATION)


def raw_recurrence():
    return copy.deepcopy(_RAW_RECURRENCE)
