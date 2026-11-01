"""Execution policy for shared-graph program adaptation gradients."""


def configure_adaptation_checkpointing(engine, requested=None):
    """Choose the saved boundary grouping; one retains per-layer checkpoints."""
    port=engine.shared_model.port
    value=getattr(port,'checkpoint_layer_group_size',4) if requested is None else requested
    if type(value) is not int or value < 1:
        raise ValueError('adaptation checkpoint layer group size must be a positive integer')
    port.checkpoint_layer_group_size=value
    return value
