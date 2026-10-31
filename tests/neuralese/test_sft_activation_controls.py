import pytest
from scripts.neuralese_data.sft_activation_controls import configure, validate


def test_device_and_file_modes_replace_stale_controls():
    base = {'checkpoint_input_offload': True, 'checkpoint_input_offload_directory': '/old',
            'memory_qualification': {'path': '/proof'}, 'batch_graphs': 64, 'batch_tokens': 32768}
    device = configure(base, use_device=True, scratch='/new')
    assert device['checkpoint_input_offload'] is False
    assert 'checkpoint_input_offload_directory' not in device
    file = configure(device, use_device=False, scratch='/new')
    assert file['checkpoint_input_offload'] is True
    assert file['checkpoint_input_offload_directory'] == '/new'
    for changed in ({'checkpoint_input_offload_directory': '/stale'}, {'checkpoint_layers': False},
                    {'device': 'cpu'}, {'memory_qualification': None}, {'batch_tokens': 1024}):
        with pytest.raises(ValueError):
            validate({**device, **changed})
