"""Fail closed unless provider preparation executes from its frozen code tree."""
from pathlib import Path
import sys

def require_frozen_provider_runtime(root):
    root=Path(root).resolve(strict=True)
    from natlang_neuralese.common.runtime_inventory import immutable_native_runtime_tree
    identity=immutable_native_runtime_tree(root)
    for name,module in tuple(sys.modules.items()):
        if not (name.startswith('natlang_neuralese.') or name.startswith('scripts.neuralese_data.')):continue
        source=getattr(module,'__file__',None)
        if source is None:continue
        try:Path(source).resolve(strict=True).relative_to(root)
        except ValueError as error:raise ValueError(f'provider execution imported {name} outside its frozen runtime') from error
    return identity
