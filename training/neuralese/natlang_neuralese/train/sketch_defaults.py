"""Canonical consumer-training defaults, frozen with the shared Python runtime.

These are training policy defaults, not a foundation certificate or admission.
Explicit CLI settings override them; checkpoint identity rejects silent changes.
Machine owners choose replay group size and CUDA envelopes independently.
"""

SKETCH_TRAINING_DEFAULTS = {
    "sketch_gradient": "local_stage",
    "sketch_target_weight": 0.1,
    "sketch_target_backbone_scale": 0.05,
    "train_control_rows": True,
    "content_transport": "top-state",
    "handover": "written",
    "write_curriculum": "joint",
    "write_depth": 5,
    "tokens_per_vector": 1.0,
    "writer_supervision": "native-value",
    "writer_length_policy": "native-value",
    "stop_supervision": "gold-native-boundary",
    "writer_text_weight": 1.0,
    "crisp_weight": 1.0,
    "distill": 0.25,
    "optimizer": "muon",
    "rank": 16,
    "lr": 1e-4,
    "lora_lr": 2e-5,
    "heads_lr": 3e-5,
    "max_tokens": 65536,
    "steps": 2200,
    "batch": 1,
    "backward_policy": "auto",
    "checkpoint_layers": True,
    "ffn_chunk_tokens": 1024,
    "token_cache_mib": 32,
    "eval_every": 64,
}


def apply_sketch_defaults(parser):
    actions = {action.dest: action for action in parser._actions}
    if not SKETCH_TRAINING_DEFAULTS.keys() <= actions.keys():
        raise ValueError("canonical sketch defaults contain unknown trainer options")
    for name, value in SKETCH_TRAINING_DEFAULTS.items():
        choices = actions[name].choices
        if choices is not None and value not in choices:
            raise ValueError("invalid canonical sketch default: " + name)
    parser.set_defaults(**SKETCH_TRAINING_DEFAULTS)
