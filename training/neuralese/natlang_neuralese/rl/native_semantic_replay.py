"""Authenticate the original compact wire and actual decoded observation.

These are crisp archive/transport checks, not a policy for choosing a dialect.
The dependency executor recomputes captured producers and the real codec; this
module never promotes an observed decoded tensor to an in-episode producer.
"""
from ..model.semantic_message import SemanticStateMessage,retained_source_offsets
from ..model.semantic_vae import alignment_weights


def authenticate_semantic_input(segment,observed,*,identity,channel,fetch_block):
    if channel is None:
        raise ValueError('native semantic replay needs the exact differentiable semantic channel')
    source_id=segment.get('source_state_id')
    if not isinstance(source_id,str) or source_id==segment.get('id'):
        raise ValueError('native semantic replay needs original compact and observed decoded artifact identities')
    message=fetch_block(source_id)
    if not isinstance(message,SemanticStateMessage):
        raise ValueError('native semantic replay source must be the actual compact message')
    message.validate();observed.require_native(**identity)
    source=channel.model_name(message)
    targets=[name for name,binding in channel.bindings.items() if binding==identity]
    if len(targets)!=1:
        raise ValueError('semantic replay target must be one exactly trained participant')
    target=targets[0]
    source_offsets,policy=retained_source_offsets(channel.tokenizers[source],message.text,message.token_ids.tolist())
    tokens=channel.tokenizers[target](message.text,add_special_tokens=False,return_offsets_mapping=True)
    _,matching=alignment_weights(source_offsets,tokens['offset_mapping'],allow_nearest=True)
    matching['best_effort']=matching['best_effort'] or policy!='exact-tokenizer-offsets'
    expected={'source_model':source,'adapter_sha256':channel.adapter_sha256,'posterior':'mean',
        'source_native_identity':channel.bindings[source],'source_offset_policy':policy,
        'target_model':target,'alignment':matching,'mode':'learned-semantic-state-deltas',
        'target_native_identity':identity}
    if (message.adapter_sha256!=channel.adapter_sha256 or message.source_model!=source
            or message.offsets!=source_offsets or message.offset_policy!=policy
            or segment.get('semantic_adaptation')!=expected
            or observed.text!=message.text or observed.token_ids.tolist()!=tokens['input_ids']
            or segment.get('native_identity')!=identity):
        raise ValueError('semantic replay changed its exact adapter, tokenizer, source or actual decode receipt')
    return {'source_state_id':source_id,'source_model':source,'target_model':target,
        'text':message.text,'adapter_sha256':channel.adapter_sha256,
        'source_token_ids':tuple(message.token_ids.tolist()),'target_token_ids':tuple(observed.token_ids.tolist())}
