"""Consistent device/file activation controls for a capacity-gated SFT launch."""


def configure(params, *, use_device, scratch):
    result = dict(params)
    result['checkpoint_input_offload'] = not use_device
    if use_device:
        result.pop('checkpoint_input_offload_directory', None)
    else:
        if not scratch:
            raise ValueError('file activation storage needs a scratch directory')
        result['checkpoint_input_offload_directory'] = scratch
    validate(result)
    return result


def validate(params):
    offload = params.get('checkpoint_input_offload')
    directory = params.get('checkpoint_input_offload_directory')
    if type(offload) is not bool or bool(directory) != offload:
        raise ValueError('activation offload and file directory must agree')
    if params.get('checkpoint_layers', True) is not True:
        raise ValueError('capacity qualification requires checkpointed layers')
    if params.get('device', 'cuda') != 'cuda':
        raise ValueError('qualified SFT device must be CUDA')
    if not params.get('memory_qualification'):
        raise ValueError('trainer GPU capacity qualification is required')
    if params.get('batch_graphs') != 64 or params.get('batch_tokens') != 32768:
        raise ValueError('batch settings differ from the qualified packed geometry')
