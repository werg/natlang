"""Exact model-origin native marker failure observations; no repair policy."""
from __future__ import annotations

FAILURE_SCHEMA='natlang.native-marker-failure/1'
FAILURE_CODES=frozenset({'neuralese-native-state-open','neuralese-native-state-close',
    'neuralese-native-state-consume-marker','neuralese-native-state-nested'})


def capture_enabled(seq):
    return seq.request.rl_capture or getattr(seq,'native_trace_enabled',False)


def failure_descriptor(engine,seq,error):
    """Bind the actual final draw/input boundary, never synthesize a state block."""
    samples=seq.rl_samples
    sample_index=(None if error.code=='neuralese-native-state-open' and
                  seq.generated_positions>=seq.request.max_tokens else len(samples)-1)
    if sample_index is not None and sample_index<0:
        raise ValueError('native marker failure lost its actual sampled draw')
    fed={value['sample_index'] for value in seq.native_rl_inputs if value.get('sample_index') is not None}
    return {'schema':FAILURE_SCHEMA,'code':error.code,'message':str(error),
        'trigger':'length' if sample_index is None else 'sample',
        'sample_index':sample_index,'fed':sample_index in fed if sample_index is not None else False,
        'own_input_count':seq.native_own_position,
        'generated_input_count':len(seq.native_rl_inputs),'max_tokens':seq.request.max_tokens,
        'stop_token_ids':sorted(engine.stop_ids)}


def generated_tokens(seq):
    """All actual forced/fed tokens plus an actual final unfed draw, in order."""
    result=[value['token_id'] for value in seq.native_rl_inputs]
    fed={value['sample_index'] for value in seq.native_rl_inputs if value.get('sample_index') is not None}
    missing=[sample for sample in seq.rl_samples if sample['index'] not in fed]
    if len(missing)>1 or missing and missing[0]['index']!=len(seq.rl_samples)-1:
        raise ValueError('native marker failure lost its sampled input ordering')
    return result+[sample['token_id'] for sample in missing]


FAILURE_SAMPLING_SCHEMA='natlang.native-rl-sampling/4'
FAILURE_STRUCTURAL_SCHEMA='natlang.native-structural-capture/2'


def validate_failure(capture,prompt_length):
    """Validate the versioned failure boundary before accepting any replay action."""
    failure=capture.get('protocol_failure')
    versioned=capture.get('schema') in (FAILURE_SAMPLING_SCHEMA,FAILURE_STRUCTURAL_SCHEMA)
    if not versioned:
        if failure is not None:raise ValueError('native protocol failure needs its explicit capture schema')
        return None
    keys={'schema','code','message','trigger','sample_index','fed','own_input_count',
          'generated_input_count','max_tokens','stop_token_ids'}
    if (not isinstance(failure,dict) or set(failure)!=keys or failure.get('schema')!=FAILURE_SCHEMA
            or failure.get('code') not in FAILURE_CODES
            or not isinstance(failure.get('message'),str)
            or not failure['message'].startswith(failure['code']+': ')
            or capture.get('finish_reason')!='error'
            or type(failure.get('fed')) is not bool
            or any(type(failure.get(key)) is not int or failure[key]<0
                   for key in ('own_input_count','generated_input_count','max_tokens'))
            or failure['max_tokens']<1):
        raise ValueError('native protocol failure boundary is malformed')
    stop=failure['stop_token_ids']
    if (not isinstance(stop,list) or not stop or any(type(token) is not int or token<0 for token in stop)
            or stop!=sorted(set(stop))):
        raise ValueError('native protocol failure needs exact distinct stop-token identities')
    samples=capture.get('samples') or [];inputs=capture.get('generated_inputs') or []
    if (len(inputs)!=failure['generated_input_count']
            or prompt_length+len(inputs)!=failure['own_input_count']):
        raise ValueError('native protocol failure changed its actual fed-input positions')
    index=failure['sample_index']
    if failure['trigger']=='sample':
        fed=any(value.get('sample_index')==index for value in inputs)
        if type(index) is not int or index!=len(samples)-1 or index<0 or fed!=failure['fed']:
            raise ValueError('native protocol failure changed its exact offending draw or fed disposition')
    elif failure['trigger']=='length':
        if (index is not None or failure['fed'] or failure['code']!='neuralese-native-state-open'
                or len(inputs)<failure['max_tokens']):
            raise ValueError('native protocol failure changed its actual length boundary')
    else:raise ValueError('native protocol failure trigger is unsupported')
    if any(type(sample.get('terminal')) is not bool or sample['terminal']!=(sample.get('token_id') in stop)
           for sample in samples):
        raise ValueError('native protocol failure changed sampled stop-token semantics')
    return failure


def validate_failure_draw(failure,sample,opened,roles):
    """Authenticate exactly the state-machine rejection the actual decoder made."""
    token=sample['token_id'];code=failure['code'];fed=failure['fed']
    valid={
        'neuralese-native-state-close':token==roles.produce_close and opened is None and not fed,
        'neuralese-native-state-consume-marker':token in (roles.consume_open,roles.consume_close) and not fed,
        'neuralese-native-state-nested':token==roles.produce_open and opened is not None and fed,
        'neuralese-native-state-open':sample['terminal'] and opened is not None and not fed,
    }
    if not valid[code]:raise ValueError('native protocol failure does not match its actual control draw and prefix')
    return 'produce' if opened is not None and token not in (roles.produce_open,roles.produce_close) else 'ordinary'
