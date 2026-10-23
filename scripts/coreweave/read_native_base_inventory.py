"""Read only immutable model identity metadata from a bound actor view/checkpoint."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    kind, source = sys.argv[1], Path(sys.argv[2])
    if kind == 'actor-view':
        document = json.loads(source.read_bytes())
        model_config = document.get('model_config')
        result = {'kind': kind, 'source': {'path': str(source), 'bytes': source.stat().st_size},
                  'source_model': document.get('source'), 'step': document.get('step'),
                  'model_config': model_config}
    elif kind == 'native-checkpoint':
        import torch
        state = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
        model_config = state.get('model_config')
        result = {'kind': kind, 'source': {'path': str(source), 'bytes': source.stat().st_size},
                  'model_config': model_config,
                  'tensor_payload_loaded': False, 'load_mode': 'mmap-metadata-only'}
        del state
    else:
        raise ValueError('kind must be actor-view or native-checkpoint')
    if not isinstance(model_config, dict) or not isinstance(model_config.get('base_inventory'), dict):
        raise ValueError('source lacks the exact native model base inventory')
    print(json.dumps(result, sort_keys=True, separators=(',', ':')))


if __name__ == '__main__':
    main()
