"""Exact sampled-token replay through one flattened native invocation graph.

Native state is a deterministic function of sampled producer tokens and current
weights. Replay the producers in the graph, rather than treating their recorded
KV as a separate stochastic action or recursively running a writer per read.
Uncaptured external inputs retain their exact recorded native state.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from ..model.shared_state import Invocation, ImportSpan, compile_graph


@dataclass
class NativeReplay:
    plan: object
    actions: tuple[dict, ...]
    external_blocks: dict
    identity: dict
    adapter_binding: dict | None = None
    external_receipts: dict | None = None
    invocation_actors: dict | None = None
    external_invocations: dict | None = None


def _tokens(ids):
    if not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError('native RL capture token IDs must be nonnegative integers')
    return tuple(ids)


def _input_stream(capture, prompt_length):
    """Normalize receipts to actual inputs plus sampled-only terminal events.

    Forced inputs are explicitly marked, never assigned a behavior logp or
    treated as sampled actions. /1 remains readable for earlier diagnostics.
    """
    if capture['schema'] == 'natlang.native-rl-sampling/1':
        return [{**sample, 'forced': False, 'source_sample': sample['index']}
                for sample in capture.get('samples') or []], capture.get('writes') or []
    samples = capture.get('samples') or []
    if any(sample.get('index') != index for index, sample in enumerate(samples)):
        raise ValueError('native RL sample indices must retain their actual order')
    stream, consumed = [], set()
    for value in capture.get('generated_inputs') or []:
        token = _tokens([value.get('token_id')])[0]
        sample_index = value.get('sample_index')
        if value.get('input_role') not in {'ordinary', 'produce'}:
            raise ValueError('native RL forced input needs its actual embedding role')
        if sample_index is None:
            event = {'token_id': token, 'input_role': value['input_role'], 'terminal': False,
                     'forced': True, 'source_sample': None}
        else:
            if type(sample_index) is not int or not 0 <= sample_index < len(samples) or sample_index in consumed:
                raise ValueError('native RL input has a missing or repeated sampled-token binding')
            sample = samples[sample_index]
            if (sample.get('query_own_position') != prompt_length+len(stream)-1
                    or sample.get('token_id') != token or sample.get('input_role') != value['input_role']
                    or sample.get('terminal')):
                raise ValueError('native RL sampled input/query coordinate differs from actual model inputs')
            event = {**sample, 'forced': False, 'source_sample': sample_index}
            consumed.add(sample_index)
        stream.append({**event, 'index': len(stream)})
    for sample_index, sample in enumerate(samples):
        if sample_index in consumed:
            continue
        if (not sample.get('terminal') or sample_index != len(samples)-1
                or sample.get('query_own_position') != prompt_length+len(stream)-1):
            raise ValueError('native RL unconsumed action is not its actual final terminal sample')
        stream.append({**sample, 'index': len(stream), 'forced': False, 'source_sample': sample_index})
    order = [event['source_sample'] for event in stream if not event['forced']]
    if order != list(range(len(samples))):
        raise ValueError('native RL model inputs reorder the actual sampled action stream')
    writes = []
    for write in capture.get('writes') or []:
        keys = ('own_begin', 'own_end', 'payload_own_begin', 'payload_own_end')
        if any(type(write.get(key)) is not int for key in keys):
            raise ValueError('native RL /2 writes require exact own-input coordinates')
        writes.append({**write, 'begin_sample': write['own_begin']-prompt_length,
            'end_sample': write['own_end']-prompt_length,
            'payload_begin': write['payload_own_begin']-prompt_length,
            'payload_end': write['payload_own_end']-prompt_length})
    return stream, writes


def capture_invocation_actors(turns,base):
    """Exact host-capture actor inventory; graph/state proof is compile_replay's."""
    import copy
    from ..model.native_actor_graph import validate_invocation_actor_receipt
    actors={}
    for number,turn in enumerate(turns):
        name=turn.get('graph_turn') or f'turn-{number}'
        if name in actors:raise ValueError('native RL graph turn identities must be unique')
        capture=turn.get('sampling') or {}
        actors[name]={'native_identity':copy.deepcopy(capture.get('policy')),
                     'adapter_binding':copy.deepcopy(capture.get('adapter_binding'))}
    return validate_invocation_actor_receipt({'schema':'natlang.native-invocation-actors/1',
        'base_native_identity':dict(base),'invocations':actors},base)


def compile_replay(turns, *, identity, roles, fetch_block, invocation_actors=None, structural=False):
    """Compile real turn captures, preserving exact producer occurrence bindings.

    ``turns`` use graph_replay.producer_turns' graph_turn/written_nodes/read_bindings
    metadata. A repeated content ID needs its actual execution-graph occurrence.
    ``fetch_block`` loads verified NativeStateBlock artifacts from the durable
    rollout bundle, never a regenerated text approximation. ``structural``
    compiles the identical causal inputs/edges without policy actions; greedy
    receipts have no behavior density and are never accepted as RL samples.
    """
    if invocation_actors is not None:
        from ..model.native_actor_graph import validate_invocation_actor_receipt
        validate_invocation_actor_receipt(invocation_actors,identity)
        if invocation_actors!=capture_invocation_actors(turns,identity):
            raise ValueError('native RL declared invocation actors differ from actual captured behavior')
    prepared, writers = {}, {}
    adapter_binding = None
    binding_seen = False
    for number, turn in enumerate(turns):
        name = turn.get('graph_turn') or f'turn-{number}'
        if name in prepared:
            raise ValueError('native RL graph turn identities must be unique')
        capture = turn.get('sampling') or {}
        actor_identity=(invocation_actors['invocations'][name]['native_identity'] if invocation_actors is not None else identity)
        greedy = capture.get('schema') == 'natlang.native-structural-capture/1'
        decision = capture.get('schema') == 'natlang.native-decision-capture/1'
        schemas = {'natlang.native-rl-sampling/1', 'natlang.native-rl-sampling/2', 'natlang.native-rl-sampling/3'}
        if structural:
            schemas.add('natlang.native-structural-capture/1')
            schemas.add('natlang.native-decision-capture/1')
        if (capture.get('schema') not in schemas
                or capture.get('policy') != actor_identity
                or not decision and capture.get('sample_range_convention') != 'half-open'):
            raise ValueError('native RL capture has a different policy or unsupported sampling schema')
        binding = capture.get('adapter_binding')
        if capture['schema'] == 'natlang.native-rl-sampling/3' or (greedy or decision) and binding is not None:
            from types import SimpleNamespace
            from ..model.native_adapters import validate_state_binding
            validate_state_binding(binding, SimpleNamespace(**actor_identity))
        elif binding is not None:
            raise ValueError('adapted native RL capture requires its explicit adapter-bound sampling schema')
        if invocation_actors is None and binding_seen and binding != adapter_binding:
            raise ValueError('native RL graph requires one exact adapter binding; heterogeneous actors need per-invocation executor support')
        adapter_binding, binding_seen = binding, True
        law = capture.get('sampling_law') or {}
        temperature = law.get('temperature')
        if decision:
            if any(capture.get(key) for key in ('samples','generated_inputs','writes','sampling_law')):
                raise ValueError('native decision capture cannot declare sampled actions or a sampling law')
        elif greedy:
            if (type(temperature) not in (int, float) or temperature != 0
                    or law.get('probabilities') != 'greedy-argmax'):
                raise ValueError('native structural receipt needs its actual greedy decoding law')
        elif (type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0
                or law.get('probabilities') != 'post-processing-normalized'):
            raise ValueError('native RL needs its exact positive-temperature sampling law')
        if set(law) - {'temperature', 'seed', 'guidance', 'probabilities'}:
            raise ValueError('native RL sampling processor needs explicit replay support')
        own, mask, reads, read_stops = [], [], [], {}
        for segment in capture.get('prompt') or []:
            if segment.get('kind') == 'tokens':
                ids = _tokens(segment.get('token_ids'))
                flags = segment.get('producing')
                if not isinstance(flags, list) or len(flags) != len(ids) or any(type(x) is not bool for x in flags):
                    raise ValueError('native RL prompt requires exact per-token embedding roles')
                own.extend(ids); mask.extend(flags)
            elif segment.get('kind') == 'native_state':
                observed=fetch_block(segment['id'])
                if segment.get('state_derivation')!=observed.derivation:
                    raise ValueError('native RL input changed its actual state derivation receipt')
                receipt = segment.get('adapter_state_consumption')
                if receipt is not None:
                    from ..model.native_adapters import validate_adapter_state_receipt
                    block = observed
                    validate_adapter_state_receipt(block, receipt, target_identity=actor_identity, target_binding=binding)
                    if segment.get('native_identity') != receipt['source_native_identity']:
                        raise ValueError('native RL input source identity differs from its consumption receipt')
                    logical_position = len(own) + sum(len(fetch_block(ident).token_ids) for _, ident, _ in reads)
                    if receipt['target_position_start'] != logical_position:
                        raise ValueError('native RL consumption receipt changed actual consumer position')
                elif segment.get('native_identity') != actor_identity:
                    raise ValueError('foreign native RL state needs an explicit adaptation/reencoding capture')
                if 'stop_gradient' in segment and segment['stop_gradient'] is not True:
                    raise ValueError('native input stop_gradient must be explicitly true or absent')
                read_stops[len(own)] = segment.get('stop_gradient', False)
                reads.append((len(own), segment['id'], receipt))
            else:
                raise ValueError('unknown native RL prompt segment')
        if not own or any(at == len(own) for at, _, _ in reads):
            raise ValueError('native RL prompt must end in its own consumer query token')
        prompt_length = len(own)
        samples, writes = _input_stream(capture, prompt_length)
        events, actual_writes, opened = [], [], None
        for index, sample in enumerate(samples):
            token = _tokens([sample.get('token_id')])[0]
            if (sample.get('index') != index or (not sample['forced'] and (
                    (sample.get('logp') is not None if greedy else
                     type(sample.get('logp')) not in (int, float)
                     or not math.isfinite(sample['logp']) or sample.get('logp') > 1e-6)))
                    or sample.get('input_role') not in {'ordinary', 'produce'}
                    or type(sample.get('terminal')) is not bool
                    or (sample['terminal'] and index != len(samples)-1)):
                raise ValueError('native RL sampled-token stream is malformed')
            if token == roles.produce_open:
                if opened is not None:
                    raise ValueError('native RL produce spans cannot nest')
                opened = index
                expected_role = 'ordinary'
            elif token == roles.produce_close:
                if opened is None:
                    raise ValueError('native RL close marker lacks its producer opening marker')
                actual_writes.append((opened, index+1, opened+1, index))
                opened = None
                expected_role = 'ordinary'
            else:
                expected_role = 'produce' if opened is not None else 'ordinary'
            if token in (roles.consume_open, roles.consume_close) or sample['input_role'] != expected_role:
                raise ValueError('native RL sampled control marker/embedding role disagrees with its write span')
            if sample['terminal'] and opened is not None:
                raise ValueError('native RL terminal token occurs inside a producer payload')
            allowed = sample.get('allowed_token_ids')
            banned = sample.get('banned_token_ids')
            if allowed is not None and banned is not None:
                raise ValueError('native RL mask must use one exact allowed or banned representation')
            if not sample['forced'] and law.get('guidance') and allowed is None and banned is None:
                raise ValueError('guided native RL requires actual normalized-mask replay')
            if allowed is not None:
                allowed = _tokens(allowed)
                if token not in allowed or len(set(allowed)) != len(allowed):
                    raise ValueError('native RL guide mask must contain its sampled token exactly once')
            if banned is not None:
                banned = _tokens(banned)
                if token in banned or len(set(banned)) != len(banned):
                    raise ValueError('native RL banned mask excludes its sampled token or repeats IDs')
            if not sample['forced'] and not structural:
                events.append({'local_query': prompt_length + index - 1, 'token_id': token,
                    'behavior_logp': sample['logp'], 'temperature': float(temperature),
                    'allowed_token_ids': allowed, 'banned_token_ids': banned, 'turn': name, 'sample': sample['source_sample']})
            own.append(token); mask.append(sample['input_role'] == 'produce')
        if opened is not None:
            raise ValueError('native RL capture ended inside a producer payload')
        declared_writes = [(write.get('begin_sample'), write.get('end_sample'),
                            write.get('payload_begin'), write.get('payload_end'))
                           for write in writes]
        if actual_writes != declared_writes:
            raise ValueError('native RL captured write ranges do not cover actual sampled produce spans')
        spans = []
        for write in writes:
            begin, end = write.get('begin_sample'), write.get('end_sample')
            lo, hi = write.get('payload_begin'), write.get('payload_end')
            if (any(type(value) is not int for value in (begin, end, lo, hi))
                    or not 0 <= begin < lo <= hi < end <= len(samples)
                    or lo != begin+1 or hi != end-1
                    or samples[begin]['token_id'] != roles.produce_open
                    or samples[end-1]['token_id'] != roles.produce_close
                    or write.get('native_identity') != actor_identity
                    or not all(mask[prompt_length+i] for i in range(lo, hi))
                    or mask[prompt_length+begin] or mask[prompt_length+end-1]):
                raise ValueError('native RL write boundaries/embedding roles differ from the captured payload')
            block_id = write['state_id']
            block = fetch_block(write.get('raw_state_id', block_id))
            block.require_native(**actor_identity)
            if getattr(block, 'adapter_binding', None) != binding:
                raise ValueError('native RL producer state adapter binding differs from its actor')
            if write.get('raw_state_id', block_id) != block_id:
                from ..model.semantic_message import SemanticStateMessage
                message = fetch_block(block_id)
                if not isinstance(message, SemanticStateMessage):
                    raise ValueError('native RL semantic write must reference its actual compact message')
                message.validate()
                if (any(getattr(message, key) != value for key, value in actor_identity.items())
                        or message.text != block.text
                        or not torch.equal(message.token_ids, block.token_ids)
                        or not torch.equal(message.embeddings, block.embeddings)
                        or not torch.equal(message.positions, block.positions)):
                    raise ValueError('native RL semantic message differs from its archived producer state')
            if block.token_ids.tolist() != [sample['token_id'] for sample in samples[lo:hi]]:
                raise ValueError('native RL archived write tokens differ from its sampled payload')
            span = (prompt_length+lo, prompt_length+hi)
            spans.append(span)
            occurrence = (turn.get('written_nodes') or {}).get(block_id)
            writers.setdefault(block_id, []).append((name, *span, occurrence))
        prepared[name] = {'turn': turn, 'own': tuple(own), 'mask': mask, 'reads': reads, 'read_stops': read_stops,
                          'events': events, 'spans': spans, 'actor_identity':actor_identity,'adapter_binding':binding}
    if not prepared:
        raise ValueError('native RL graph has no captured turns')
    externals, nodes, ordered, external_receipts, external_invocations = {}, [], [], {}, {}
    pending = dict(prepared)
    while pending:
        progress = False
        for name, item in list(pending.items()):
            actor_identity,binding=item['actor_identity'],item['adapter_binding']
            imports, dependencies = [], set()
            for at, block_id, receipt in item['reads']:
                choices = writers.get(block_id, [])
                occurrence_binding = (item['turn'].get('read_bindings') or {}).get(block_id)
                outside = (item['turn'].get('external_read_bindings') or {}).get(block_id)
                if outside is not None:
                    if (not isinstance(outside,str) or not outside
                            or occurrence_binding is not None
                            or any(choice[3]==outside for choice in choices)):
                        raise ValueError('native external origin must remain outside actual captured writer occurrences')
                    choices = []
                elif occurrence_binding is not None:
                    choices = [choice for choice in choices if choice[3] == occurrence_binding]
                    if len(choices) != 1:
                        raise ValueError('native RL read has no unique captured writer occurrence')
                elif len(choices) > 1:
                    raise ValueError('native RL repeated state ID needs an exact writer occurrence binding')
                if choices:
                    source, start, stop, _ = choices[0]
                    dependencies.add(source)
                else:
                    source = 'external-' + block_id
                    if source not in externals:
                        externals[source] = fetch_block(block_id)
                    block = externals[source]
                    external_invocations.setdefault(source, []).append({'invocation':name,'at':at,'receipt':receipt})
                    if receipt is not None:
                        external_receipts.setdefault(source, []).append(receipt)
                    else:
                        block.require_native(**actor_identity)
                        if getattr(block, 'adapter_binding', None) != binding:
                            raise ValueError('native RL external state adapter binding differs from its actor')
                    start, stop = 0, len(block.token_ids)
                imports.append(ImportSpan(at, source, start, stop, stop_gradient=item['read_stops'].get(at, False)))
            if any(dependency in pending for dependency in dependencies):
                continue
            # All exact roles, including STOP and delimiter inputs, come from
            # the receipt. Only payload ranges export native state to readers.
            spans, start = [], None
            for index, producing in enumerate(item['mask'] + [False]):
                if producing and start is None:
                    start = index
                if not producing and start is not None:
                    spans.append((start, index)); start = None
            nodes.append(Invocation(name, item['own'], imports=tuple(imports), producing=tuple(spans)))
            ordered.append(item)
            del pending[name]; progress = True
        if not progress:
            raise ValueError('native RL captured writer graph contains a cycle')
    plan = compile_graph(nodes, external_sizes={name: len(block.token_ids) for name, block in externals.items()})
    actions = tuple({**event, 'query': offset + event['local_query']}
        for offset, item in zip(plan.offsets, ordered) for event in item['events'])
    return NativeReplay(plan, actions, externals, dict(identity), adapter_binding if invocation_actors is None else None, external_receipts, invocation_actors, external_invocations)


def replay_logps(model, replay, *, checkpoint_layers=True, chunk_tokens=128):
    """Score every actual sample, including markers and terminal tokens, once."""
    from ..model.native_adapters import require_model_execution_actor
    if replay.invocation_actors is None:
        require_model_execution_actor(model, replay.identity, replay.adapter_binding)
    else:
        from ..model.native_actor_graph import invocation_actor_bank
        bank=invocation_actor_bank(model.port)
        if bank is None or bank.receipt()!=replay.invocation_actors:
            raise ValueError('native heterogeneous replay needs its exact per-invocation actor scope')
    model.port.checkpoint_layers = checkpoint_layers
    result = model(replay.plan, external_blocks=replay.external_blocks,
                   external_identity={name: {key: getattr(block, key) for key in replay.identity}
                       for name, block in replay.external_blocks.items()}, retain_states=False)
    return action_logps(model, result['h_final'][0], replay.actions, chunk_tokens=chunk_tokens)


def action_logps(model, hidden, actions, *, chunk_tokens=128):
    """Shared chunked readout for standalone qualification and trainer callbacks."""
    logps = token_action_logps(model, hidden, actions, chunk_tokens=chunk_tokens)
    return logps, torch.tensor([action['behavior_logp'] for action in actions],
                              device=hidden.device, dtype=torch.float32)


def token_action_logps(model, hidden, actions, *, chunk_tokens=128, readout=None):
    """Exact masked conditional scores; no invented behavior-density field."""
    if chunk_tokens < 1:
        raise ValueError('native RL loss chunks must be positive')
    from torch.utils.checkpoint import checkpoint
    groups = {}
    for index, action in enumerate(actions):
        groups.setdefault((action['temperature'],tuple(action['allowed_token_ids']) if action['allowed_token_ids'] is not None else None,
                           tuple(action['banned_token_ids']) if action.get('banned_token_ids') is not None else None), []).append(index)
    logps = [None] * len(actions)
    def score(states,labels,temperature,allowed,banned):
        from ..common.sampling import masked_log_distribution
        log_probs=masked_log_distribution((readout or model.logits)(states),temperature,allowed=allowed,banned=banned)
        return log_probs.gather(1,labels[:,None]).squeeze(1)
    from ..common.native_higher_order import capture_reference_execution,active_reference_arithmetic
    if active_reference_arithmetic() is not None:
        from ..model.native_adapters import capture_native_execution
        score=capture_reference_execution(capture_native_execution(score))
    for (temperature,allowed,banned),indices in groups.items():
        for start in range(0,len(indices),chunk_tokens):
            selected=indices[start:start+chunk_tokens]
            queries=torch.tensor([actions[i]['query'] for i in selected],device=hidden.device)
            labels=torch.tensor([actions[i]['token_id'] for i in selected],device=hidden.device)
            values=checkpoint(score,hidden[queries],labels,temperature,allowed,banned,use_reentrant=False)
            for index,value in zip(selected,values.unbind()):logps[index]=value
    return torch.stack(logps)


def _validate_adapter_transition(engine,before_identity,before_binding,after_identity,after_binding,bundle,archive_owner):
    """One exact current/behavior adapter contract for uniform and call graphs."""
    from types import SimpleNamespace
    from ..model.native_adapters import validate_state_binding
    if before_binding is None:
        if after_binding is not None or before_identity!=engine.native_identity or after_identity!=before_identity:
            raise ValueError('native-policy objective is bound to another exact actor checkpoint')
        return
    validate_state_binding(before_binding,SimpleNamespace(**before_identity))
    if (before_binding['base_native_identity']!=engine.native_identity or after_binding is None
            or len(after_binding['adapters'])!=len(before_binding['adapters'])):
        raise ValueError('native-policy evaluation changed the exact behavior base or adapter scope')
    validate_state_binding(after_binding,SimpleNamespace(**after_identity))
    if after_binding['base_native_identity']!=engine.native_identity:
        raise ValueError('native-policy evaluation changed the exact behavior base or adapter scope')
    if bundle is None:
        raise ValueError('adapted native policy evaluation requires its original behavior artifact bundle')
    from .graph_replay import verified_bundle_bytes
    from ..model.native_artifact import decode_native_tensor_artifact
    checked={}
    def archived(ident):
        if ident not in checked:
            raw=verified_bundle_bytes(bundle[0],ident,policy_id=archive_owner)
            if raw is None:raise ValueError('native-policy archive is missing exact behavior adapter dependencies')
            artifact,actual=decode_native_tensor_artifact(raw)
            if actual!=ident or artifact.native_identity!=engine.native_identity:
                raise ValueError('native-policy behavior adapter archive changed identity')
            checked[ident]=artifact
        return checked[ident]
    for before,after in zip(before_binding['adapters'],after_binding['adapters']):
        if before['scale_hex']!=after['scale_hex']:
            raise ValueError('native-policy evaluation changed an ordered behavior adapter scale')
        behavior=archived(before['id'])
        record=engine.store.get_native_tensor_artifact(after['id'])
        if record is None:raise ValueError('native-policy current adapter coefficients are unavailable')
        current=record.artifact.validate()
        if (behavior.kind!='adapter-coefficients' or current.kind!='adapter-coefficients'
                or current.native_identity!=engine.native_identity or behavior.source_artifact_id!=current.source_artifact_id
                or (behavior.provenance or {}).get('adapter_spec')!=(current.provenance or {}).get('adapter_spec')):
            raise ValueError('native-policy evaluation changed the exact frozen basis or matrix specification')
        basis=archived(behavior.source_artifact_id)
        if basis.kind!='adapter-basis' or (basis.provenance or {}).get('adapter_spec')!=(behavior.provenance or {}).get('adapter_spec'):
            raise ValueError('native-policy behavior archive does not bind its actual frozen basis')


class NativePolicyObjective:
    """Native GRPO/RFT loss plugged into the shared native trainer's ONE forward."""
    def __init__(self, spec_path):
        import json
        from pathlib import Path
        self.path = Path(spec_path).resolve()
        self.spec = json.loads(self.path.read_text())
        if (self.spec.get('schema') not in {'natlang.native-policy-objective/1','natlang.native-policy-objective/2'}
                or self.spec.get('kind') != 'native-policy' or self.spec.get('mode') not in {'grpo', 'rft'}):
            raise ValueError('native policy objective schema/kind/mode is unsupported')
        self.rows = self.spec['rows']
        if not self.rows:
            raise ValueError('native policy objective has no admitted episodes')
        for row in self.rows.values():
            names = row.get('episode_turns')
            if (not isinstance(names, list) or not names
                    or any(not isinstance(name, str) or not name for name in names)
                    or len(names) != len(set(names))):
                raise ValueError('native policy objective needs its complete admitted episode-turn inventory')
        self.behavior_policy=self.spec.get('behavior_policy') if self.spec['schema']=='natlang.native-policy-objective/2' else None
        if self.spec['schema']=='natlang.native-policy-objective/2':
            self.identity=self.spec['base_native_identity']
            validate_invocation_behavior_policy(self.behavior_policy,self.identity)
            from ..common.hashing import canonical_json_sha256_hex
            if ('policy' in self.spec or self.spec.get('adapter_binding') is not None
                    or self.spec.get('behavior_policy_sha256')!=canonical_json_sha256_hex(self.behavior_policy)):
                raise ValueError('native invocation objective must retain its exact behavior descriptor, not one effective model')
            for key,row in self.rows.items():
                if key not in self.behavior_policy['episodes'] or row.get('invocation_actors')!=self.behavior_policy['episodes'][key]:
                    raise ValueError('native invocation objective row changed its captured behavior actor inventory')
            self.archive_owner=self.spec['behavior_policy_sha256']
        else:
            self.identity=self.spec['policy']
            self.archive_owner=self.identity['checkpoint_sha256']
        self.bound_example=None
        self.evaluation_invocation_actors=None
        self.adapter_binding = self.spec.get('adapter_binding')
        if self.adapter_binding is not None:
            from types import SimpleNamespace
            from ..model.native_adapters import validate_state_binding
            validate_state_binding(self.adapter_binding, SimpleNamespace(**self.identity))
        self.evaluation_identity = self.identity
        self.evaluation_adapter_binding = self.adapter_binding
        self.bundle = None
        self.native_bundle_store = None
        self.external_state_overrides = {}
        self.engine = None
        self.native_state_modes = {}
        self.evaluation_external_contracts = {}
        if self.spec.get('block_bundle'):
            bundle_path = Path(self.spec['block_bundle']).resolve()
            self.bundle = (bundle_path, json.loads(bundle_path.read_text()))
            if (self.bundle[1].get('schema') != 'natlang.rl-block-bundle/1'
                    or self.bundle[1].get('policy_id') != self.archive_owner):
                raise ValueError('native objective archive belongs to a different actor policy')
            from .graph_replay import NativeBundleStore
            self.native_bundle_store=NativeBundleStore(bundle_path,policy_id=self.archive_owner,manifest=self.bundle[1])
        if self.behavior_policy is not None and self.bundle is None:
            raise ValueError('native invocation policy requires its immutable actor bundle')

    def bind_actor(self, engine, *, example=None):
        """Separate immutable sampled behavior from current same-basis coefficients."""
        from ..model.native_adapters import execution_identity, execution_binding, SCHEMA
        self.engine = engine
        if self.behavior_policy is not None:
            if example is None:raise ValueError('native invocation policy needs its exact source graph during actor binding')
            from ..model.native_actor_graph import invocation_actor_bank
            from ..common.hashing import canonical_json_sha256_hex
            bank=invocation_actor_bank(engine.shared_model.port)
            row=self.rows.get(example.id)
            if (bank is None or bank.engine is not engine or row is None
                    or canonical_json_sha256_hex(graph_identity(example))!=row['graph_sha256']):
                raise ValueError('native invocation policy needs its exact source graph and actual actor bank')
            before=row['invocation_actors']
            names=getattr(bank,'source_names',{name:name for name in bank.actors})
            if set(names)!=set(before['invocations']) or set(names.values())!=set(bank.actors):
                raise ValueError('native invocation current actor inventory differs from its behavior graph')
            current_receipt=bank.receipt()
            for name,actor in before['invocations'].items():
                after=current_receipt['invocations'][names[name]]
                _validate_adapter_transition(engine,actor['native_identity'],actor['adapter_binding'],
                    after['native_identity'],after['adapter_binding'],self.bundle,self.archive_owner)
            self.bound_example=example.id
            self.evaluation_invocation_actors=current_receipt
            self.invocation_source_names=dict(names)
            return
        from ..model.native_actor_graph import invocation_actor_bank
        if invocation_actor_bank(engine.shared_model.port) is not None:
            raise ValueError('per-invocation policy execution requires its versioned actual behavior descriptor')
        current = execution_identity(engine)
        active = execution_binding(engine)
        if self.adapter_binding is None:
            if current != self.identity or active:
                raise ValueError('native-policy objective is bound to another exact actor checkpoint')
            self.evaluation_identity, self.evaluation_adapter_binding = current, None
            return
        current_binding={'schema':SCHEMA,'base_native_identity':dict(engine.native_identity),'adapters':active} if active else None
        _validate_adapter_transition(engine,self.identity,self.adapter_binding,current,current_binding,self.bundle,self.archive_owner)
        self.evaluation_identity,self.evaluation_adapter_binding=current,current_binding

    def _invocation_external_contracts(self, example, name, binding, block):
        """Verify each actual imported occurrence against its own behavior/current actor."""
        from ..model.native_actor_graph import invocation_actor_bank
        from ..model.native_adapters import adapter_state_input_receipt, validate_adapter_state_receipt, SAME_BASE_STATE_MODE
        bank=invocation_actor_bank(self.engine.shared_model.port)
        if bank is None or bank.receipt()!=self.evaluation_invocation_actors:
            raise ValueError('native invocation external state requires the exact current actor bank')
        captured=self.rows[example.id].get('external_invocations',{}).get(name)
        actual=[(node,span) for node in example.plan.invocations for span in node.imports if span.source==name]
        if (not isinstance(captured,list) or len(captured)!=len(actual) or not captured):
            raise ValueError('native invocation external observation lost its exact consumer occurrence inventory')
        mode=self.native_state_modes.get(binding['id'])
        before_actors=self.rows[example.id]['invocation_actors']['invocations']
        for occurrence,(node,span) in zip(captured,actual):
            if (set(occurrence)!={'invocation','at','receipt'} or occurrence['invocation']!=node.name
                    or occurrence['at']!=span.at or span.start!=0 or span.stop!=len(block.token_ids)):
                raise ValueError('native invocation external observation changed its exact consumer or imported span')
            before=before_actors[node.name]
            packed_name=self.invocation_source_names[node.name]
            after=self.evaluation_invocation_actors['invocations'][packed_name]
            receipt=occurrence['receipt']
            # Import order, including earlier spans at the same own-token
            # position, is the causal logical position used by the Engine.
            position=span.at
            for prior in node.imports:
                if prior is span:break
                position+=prior.stop-prior.start
            if receipt is not None:
                validate_adapter_state_receipt(block,receipt,target_identity=before['native_identity'],target_binding=before['adapter_binding'])
                if receipt['target_position_start']!=position:
                    raise ValueError('native invocation external observation changed its actual logical position')
            else:
                block.require_native(**before['native_identity'])
                if block.adapter_binding!=before['adapter_binding']:
                    raise ValueError('native invocation strict behavior observation changed its adapter binding')
            if mode is not None:
                if mode!=SAME_BASE_STATE_MODE or receipt is None:
                    raise ValueError('same-base adapter-state evaluation requires its actual behavior consumption receipt')
                current=bank.execute(packed_name,lambda:adapter_state_input_receipt(self.engine,block,binding['id'],target_position_start=position))
                self.evaluation_external_contracts[f'{example.id}:{name}:{len(self.evaluation_external_contracts)}']={
                    'invocation':node.name,'input_name':name,**current}
            else:
                if before!=after:
                    raise ValueError('changed adapter policy with external behavior KV requires a fresh rollout under strict transport; same-base adapter-state transport must be explicitly supported and requested')
                block.require_native(**after['native_identity'])
                if block.adapter_binding!=after['adapter_binding']:
                    raise ValueError('native invocation strict current observation changed its adapter binding')

    def forward_inputs(self, context):
        blocks, identities = {}, {}
        self.evaluation_external_contracts = {}
        for example in context['examples']:
            if example.id not in self.rows:
                raise ValueError('native objective has no captured policy row for this graph')
            for name, binding in example.external_states.items():
                if self.bundle is None:
                    raise ValueError('native RL external observations require their durable state bundle')
                block=self.native_bundle_store.state(binding['id'])
                block_id=binding['id']
                receipts = self.rows[example.id].get('external_consumption_receipts', {}).get(name)
                mode = self.native_state_modes.get(binding['id'])
                if self.behavior_policy is not None:
                    self._invocation_external_contracts(example,name,binding,block)
                elif mode is not None:
                    from ..model.native_adapters import adapter_state_input_receipt, validate_adapter_state_receipt, SAME_BASE_STATE_MODE
                    if mode != SAME_BASE_STATE_MODE or not isinstance(receipts, list) or not receipts or self.engine is None:
                        raise ValueError('same-base adapter-state evaluation requires its actual behavior consumption receipt')
                    for receipt in receipts:
                        validate_adapter_state_receipt(block, receipt, target_identity=self.identity, target_binding=self.adapter_binding)
                        adapter_state_input_receipt(self.engine, block, binding['id'], target_position_start=receipt['target_position_start'])
                else:
                    if self.evaluation_identity != self.identity:
                        raise ValueError('changed adapter policy with external behavior KV requires a fresh rollout under strict transport; same-base adapter-state transport must be explicitly supported and requested')
                    block.require_native(**self.identity)
                source_identity = {key: getattr(block, key) for key in self.identity}
                if block_id != binding['id'] or len(block.token_ids) != binding['token_count']:
                    raise ValueError('native RL archived state differs from graph input binding')
                override = self.external_state_overrides.get(binding['id'])
                if override is not None:
                    override.require_native(**source_identity)
                    if len(override.token_ids) != binding['token_count']:
                        raise ValueError('native RL state-gradient override differs from admitted graph binding')
                    block = override
                packed_name = context['external_names'][example.id][name]
                blocks[packed_name] = block
                identities[packed_name] = source_identity
        return {'external_blocks': blocks, 'external_identity': identities} if blocks else {}

    def __call__(self, model, result, context):
        from .full_policy import clipped_token_loss
        from ..model.native_adapters import require_model_execution_actor
        if self.behavior_policy is None:
            require_model_execution_actor(model, self.evaluation_identity, self.evaluation_adapter_binding)
        else:
            from ..model.native_actor_graph import invocation_actor_bank
            bank=invocation_actor_bank(model.port)
            if (self.bound_example is None or len(context['examples'])!=1 or context['examples'][0].id!=self.bound_example
                    or bank is None or bank.receipt()!=self.evaluation_invocation_actors
                    or result.get('invocation_actors')!=self.evaluation_invocation_actors):
                raise ValueError('native invocation objective requires its actual bound current actor graph')
            example=context['examples'][0]
            bounds=context['row_offsets'][example.id]
            actual=context['plan'].invocations[bounds['node_start']:bounds['node_stop']]
            names={before.name:after.name for before,after in zip(example.plan.invocations,actual)}
            if names!=self.invocation_source_names:
                raise ValueError('native invocation objective actor names differ from exact packed ownership')
        hidden = result['h_final'][0]
        losses, count = [], 0
        for example in context['examples']:
            row = self.rows[example.id]
            if set(row['episode_turns']) != {node.name for node in example.plan.invocations}:
                raise ValueError('native policy graph omits or adds an admitted episode turn')
            # Bind objective coordinates to the exact source graph and masks,
            # rather than accepting same-shaped replacement training rows.
            from ..common.hashing import canonical_json_sha256_hex
            if canonical_json_sha256_hex(graph_identity(example)) != row['graph_sha256']:
                raise ValueError('native policy graph/masks differ from admitted objective coordinates')
            offset = context['row_offsets'][example.id]['physical_start']
            actions = [{**action, 'query': action['query'] + offset,
                        'allowed_token_ids': tuple(action['allowed_token_ids'])
                            if action['allowed_token_ids'] is not None else None}
                       for action in row['actions']]
            logps, behavior = action_logps(model, hidden, actions,
                                          chunk_tokens=self.spec.get('chunk_tokens', 128))
            terms = (-logps if self.spec['mode'] == 'rft' else clipped_token_loss(logps, behavior,
                row['advantage'], clip_low=self.spec.get('clip_low', .2), clip_high=self.spec.get('clip_high', .28)))
            losses.append(terms.sum()); count += len(actions)
        if count == 0:
            raise ValueError('native objective batch has no actual sampled actions')
        return torch.stack(losses).sum() / count, {'sampled_tokens': count,
            'episodes': len(losses), 'objective': self.spec['mode']}


def graph_identity(example):
    """Content identity excludes derived padding/compiler layouts, includes every semantic mask and edge."""
    from ..model.native_graph_plan import invocation_dict
    return {'id': example.id, 'split': example.split, 'source_groups': list(example.source_groups),
            'tokenizer_fingerprint': example.tokenizer_fingerprint,
            'nodes': [invocation_dict(node) for node in example.plan.invocations],
            'target_weights': list(example.target_weights), 'external_states': example.external_states}


from ..model.native_actor_graph import validate_invocation_behavior_policy


def _require_scored_training_episode(rollout, identity):
    if (rollout.get('split') != 'train' or rollout.get('training_admission') not in (None, 'grpo')
            or not isinstance(rollout.get('reward'), dict) or rollout['reward'].get('quality') not in (0, 1)
            or (rollout.get('policy') or {}).get('checkpoint_sha256') != identity['checkpoint_sha256']):
        raise ValueError('native RL requires exact scored train-only actor episodes')
    if not rollout.get('task_id') or not rollout.get('turns') or any(
            not (turn.get('replay') or {}).get('raw_calls_retained') for turn in rollout['turns']):
        raise ValueError('native RL episode lacks task identity or exact raw-call replay')
    from .episode_graph import validate_rollout_graph
    validate_rollout_graph(rollout)


def invocation_behavior_policy(rollouts,*,identity,roles,fetch_block):
    """Bind actual scored train calls; a collector base header is not behavior."""
    from .episode_graph import whole_episode_turns
    from ..common.hashing import canonical_json_sha256_hex
    episodes={}
    for rollout in rollouts:
        _require_scored_training_episode(rollout,identity)
        turns=whole_episode_turns(rollout)
        actors=capture_invocation_actors(turns,identity)
        compile_replay(turns,identity=identity,roles=roles,fetch_block=fetch_block,invocation_actors=actors)
        key=canonical_json_sha256_hex(rollout)
        if key in episodes:raise ValueError('native invocation behavior duplicated an original episode')
        episodes[key]=actors
    return validate_invocation_behavior_policy({'schema':'natlang.native-invocation-policy/1',
        'base_native_identity':dict(identity),'collector_policy_role':'base-runtime-metadata','episodes':episodes},identity)


def prepare_objective(rollouts, *, identity, roles, fetch_block, block_bundle=None, mode='grpo', behavior_policy=None):
    """Admit whole training episodes and retain real graph/action provenance."""
    from ..model.native_graph_plan import invocation_dict
    from ..data.native_graph import SCHEMA, parse_graph
    from ..common.hashing import canonical_json_sha256_hex
    from .episode_graph import whole_episode_turns
    from .grpo import group_advantages
    from ..serve.store import native_state_id
    if behavior_policy is not None:
        validate_invocation_behavior_policy(behavior_policy,identity)
        if block_bundle is None:raise ValueError('native invocation behavior requires its complete immutable actor bundle')
        import json
        from pathlib import Path
        bundle_manifest=json.loads(Path(block_bundle).read_text())
        if (bundle_manifest.get('schema')!='natlang.rl-block-bundle/1' or
                bundle_manifest.get('policy_id')!=canonical_json_sha256_hex(behavior_policy)):
            raise ValueError('native invocation behavior archive differs from its actual policy descriptor')
    if mode not in {'grpo', 'rft'}:
        raise ValueError('native replay objective must be grpo or rft')
    grouped = {}
    for rollout in rollouts:
        if rollout.get('reward') is None:
            continue
        _require_scored_training_episode(rollout,identity)
        grouped.setdefault((rollout['task_id'], rollout.get('group')), []).append(rollout)
    graphs, objective_rows = [], {}
    adapter_binding, binding_seen = None, False
    for members in grouped.values():
        advantages, stats = group_advantages([row['reward']['quality'] for row in members])
        if mode == 'grpo' and not stats['spread']:
            continue
        for rollout, advantage in zip(members, advantages):
            if mode == 'rft' and rollout['reward']['quality'] != 1:
                continue
            episode_key=canonical_json_sha256_hex(rollout)
            actors=behavior_policy['episodes'].get(episode_key) if behavior_policy is not None else None
            if behavior_policy is not None and actors is None:
                raise ValueError('native invocation behavior descriptor omits an admitted source episode')
            replay = compile_replay(whole_episode_turns(rollout), identity=identity, roles=roles, fetch_block=fetch_block,
                                    invocation_actors=actors)
            from .episode_graph import rollout_turn_names
            episode_turns = rollout_turn_names(rollout['turns'])
            if set(episode_turns) != {node.name for node in replay.plan.invocations}:
                raise ValueError('native replay lost an actual whole-episode model turn')
            if binding_seen and adapter_binding != replay.adapter_binding:
                raise ValueError('native RL objective cannot mix different adapter-bound actors')
            adapter_binding, binding_seen = replay.adapter_binding, True
            if replay.adapter_binding and block_bundle is None:
                raise ValueError('adapted native RL requires its durable coefficient/basis actor bundle')
            identifier = canonical_json_sha256_hex(rollout)
            if identifier in objective_rows:
                raise ValueError('native RL episodes duplicated without distinct sampling provenance')
            weights = [0.] * replay.plan.token_count
            for action in replay.actions:
                weights[action['query']+1] = 1.
            nodes = []
            for node, offset in zip(replay.plan.invocations, replay.plan.offsets):
                value = invocation_dict(node)
                value['token_ids'] = list(node.token_ids)
                value['target_weights'] = weights[offset:offset+len(node.token_ids)]
                nodes.append(value)
            external = {name: {'id': native_state_id(block),
                               'token_count': len(block.token_ids)} for name, block in replay.external_blocks.items()}
            graph = {'schema': SCHEMA, 'id': identifier, 'split': 'train',
                'source_groups': ['s7-task:' + rollout['task_id']],
                'tokenizer_fingerprint': identity['tokenizer_fingerprint'], 'nodes': nodes,
                'external_states': external}
            example = parse_graph(graph, tokenizer_fingerprint=identity['tokenizer_fingerprint'])
            objective_rows[identifier] = {'graph_sha256': canonical_json_sha256_hex(graph_identity(example)),
                'episode_turns': episode_turns,
                'actions': list(replay.actions), 'advantage': float(advantage or 0.),
                'external_consumption_receipts': replay.external_receipts or {},
                **({'external_invocations':replay.external_invocations or {}} if replay.invocation_actors is not None else {}),
                'task_id': rollout['task_id'], 'sample': rollout.get('sample'),
                **({'invocation_actors':replay.invocation_actors} if replay.invocation_actors is not None else {})}
            graphs.append(graph)
    return graphs, {'schema': 'natlang.native-policy-objective/2' if behavior_policy is not None else 'natlang.native-policy-objective/1', 'kind': 'native-policy',
        'mode': mode, 'rows': objective_rows,
        **({'base_native_identity':dict(identity),'behavior_policy':behavior_policy,'behavior_policy_sha256':canonical_json_sha256_hex(behavior_policy)}
           if behavior_policy is not None else {'policy':dict(identity)}),
        **({'adapter_binding': adapter_binding} if adapter_binding else {}),
        'block_bundle': str(block_bundle) if block_bundle else None}
