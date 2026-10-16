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


def compile_replay(turns, *, identity, roles, fetch_block):
    """Compile real turn captures, preserving exact producer occurrence bindings.

    ``turns`` use graph_replay.producer_turns' graph_turn/written_nodes/read_bindings
    metadata. A repeated content ID needs its actual execution-graph occurrence.
    ``fetch_block`` loads verified NativeStateBlock artifacts from the durable
    rollout bundle, never a regenerated text approximation.
    """
    prepared, writers = {}, {}
    adapter_binding = None
    binding_seen = False
    for number, turn in enumerate(turns):
        name = turn.get('graph_turn') or f'turn-{number}'
        if name in prepared:
            raise ValueError('native RL graph turn identities must be unique')
        capture = turn.get('sampling') or {}
        if (capture.get('schema') not in {'natlang.native-rl-sampling/1', 'natlang.native-rl-sampling/2', 'natlang.native-rl-sampling/3'}
                or capture.get('policy') != identity
                or capture.get('sample_range_convention') != 'half-open'):
            raise ValueError('native RL capture has a different policy or unsupported sampling schema')
        binding = capture.get('adapter_binding')
        if capture['schema'] == 'natlang.native-rl-sampling/3':
            from types import SimpleNamespace
            from ..model.native_adapters import validate_state_binding
            validate_state_binding(binding, SimpleNamespace(**identity))
        elif binding is not None:
            raise ValueError('adapted native RL capture requires its explicit adapter-bound sampling schema')
        if binding_seen and binding != adapter_binding:
            raise ValueError('native RL graph requires one exact adapter binding; heterogeneous actors need per-invocation executor support')
        adapter_binding, binding_seen = binding, True
        law = capture.get('sampling_law') or {}
        temperature = law.get('temperature')
        if (type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0
                or law.get('probabilities') != 'post-processing-normalized'):
            raise ValueError('native RL needs its exact positive-temperature sampling law')
        if set(law) - {'temperature', 'seed', 'guidance', 'probabilities'}:
            raise ValueError('native RL sampling processor needs explicit replay support')
        own, mask, reads = [], [], []
        for segment in capture.get('prompt') or []:
            if segment.get('kind') == 'tokens':
                ids = _tokens(segment.get('token_ids'))
                flags = segment.get('producing')
                if not isinstance(flags, list) or len(flags) != len(ids) or any(type(x) is not bool for x in flags):
                    raise ValueError('native RL prompt requires exact per-token embedding roles')
                own.extend(ids); mask.extend(flags)
            elif segment.get('kind') == 'native_state':
                if segment.get('native_identity') != identity:
                    raise ValueError('foreign native RL state needs an explicit adaptation/reencoding capture')
                reads.append((len(own), segment['id']))
            else:
                raise ValueError('unknown native RL prompt segment')
        if not own or any(at == len(own) for at, _ in reads):
            raise ValueError('native RL prompt must end in its own consumer query token')
        prompt_length = len(own)
        samples, writes = _input_stream(capture, prompt_length)
        events, actual_writes, opened = [], [], None
        for index, sample in enumerate(samples):
            token = _tokens([sample.get('token_id')])[0]
            if (sample.get('index') != index or (not sample['forced'] and (
                    type(sample.get('logp')) not in (int, float)
                    or not math.isfinite(sample['logp']) or sample.get('logp') > 1e-6))
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
            if not sample['forced']:
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
                    or write.get('native_identity') != identity
                    or not all(mask[prompt_length+i] for i in range(lo, hi))
                    or mask[prompt_length+begin] or mask[prompt_length+end-1]):
                raise ValueError('native RL write boundaries/embedding roles differ from the captured payload')
            block_id = write['state_id']
            block = fetch_block(write.get('raw_state_id', block_id))
            block.require_native(**identity)
            if getattr(block, 'adapter_binding', None) != adapter_binding:
                raise ValueError('native RL producer state adapter binding differs from its actor')
            if write.get('raw_state_id', block_id) != block_id:
                from ..model.semantic_message import SemanticStateMessage
                message = fetch_block(block_id)
                if not isinstance(message, SemanticStateMessage):
                    raise ValueError('native RL semantic write must reference its actual compact message')
                message.validate()
                if (any(getattr(message, key) != value for key, value in identity.items())
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
        prepared[name] = {'turn': turn, 'own': tuple(own), 'mask': mask, 'reads': reads,
                          'events': events, 'spans': spans}
    if not prepared:
        raise ValueError('native RL graph has no captured turns')
    externals, nodes, ordered = {}, [], []
    pending = dict(prepared)
    while pending:
        progress = False
        for name, item in list(pending.items()):
            imports, dependencies = [], set()
            for at, block_id in item['reads']:
                choices = writers.get(block_id, [])
                binding = (item['turn'].get('read_bindings') or {}).get(block_id)
                if binding is not None:
                    choices = [choice for choice in choices if choice[3] == binding]
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
                    block.require_native(**identity)
                    if getattr(block, 'adapter_binding', None) != adapter_binding:
                        raise ValueError('native RL external state adapter binding differs from its actor')
                    start, stop = 0, len(block.token_ids)
                imports.append(ImportSpan(at, source, start, stop))
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
    return NativeReplay(plan, actions, externals, dict(identity), adapter_binding)


def replay_logps(model, replay, *, checkpoint_layers=True, chunk_tokens=128):
    """Score every actual sample, including markers and terminal tokens, once."""
    from ..model.native_adapters import require_model_execution_actor
    require_model_execution_actor(model, replay.identity, replay.adapter_binding)
    model.port.checkpoint_layers = checkpoint_layers
    result = model(replay.plan, external_blocks=replay.external_blocks,
                   external_identity=replay.identity, retain_states=False)
    return action_logps(model, result['h_final'][0], replay.actions, chunk_tokens=chunk_tokens)


def action_logps(model, hidden, actions, *, chunk_tokens=128):
    """Shared chunked readout for standalone qualification and trainer callbacks."""
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
        log_probs=masked_log_distribution(model.logits(states),temperature,allowed=allowed,banned=banned)
        return log_probs.gather(1,labels[:,None]).squeeze(1)
    for (temperature,allowed,banned),indices in groups.items():
        for start in range(0,len(indices),chunk_tokens):
            selected=indices[start:start+chunk_tokens]
            queries=torch.tensor([actions[i]['query'] for i in selected],device=hidden.device)
            labels=torch.tensor([actions[i]['token_id'] for i in selected],device=hidden.device)
            values=checkpoint(score,hidden[queries],labels,temperature,allowed,banned,use_reentrant=False)
            for index,value in zip(selected,values.unbind()):logps[index]=value
    return torch.stack(logps), torch.tensor([action['behavior_logp'] for action in actions],
                                           device=hidden.device, dtype=torch.float32)


class NativePolicyObjective:
    """Native GRPO/RFT loss plugged into the shared native trainer's ONE forward."""
    def __init__(self, spec_path):
        import json
        from pathlib import Path
        self.path = Path(spec_path).resolve()
        self.spec = json.loads(self.path.read_text())
        if (self.spec.get('schema') != 'natlang.native-policy-objective/1'
                or self.spec.get('kind') != 'native-policy' or self.spec.get('mode') not in {'grpo', 'rft'}):
            raise ValueError('native policy objective schema/kind/mode is unsupported')
        self.rows = self.spec['rows']
        if not self.rows:
            raise ValueError('native policy objective has no admitted episodes')
        self.identity = self.spec['policy']
        self.adapter_binding = self.spec.get('adapter_binding')
        if self.adapter_binding is not None:
            from types import SimpleNamespace
            from ..model.native_adapters import validate_state_binding
            validate_state_binding(self.adapter_binding, SimpleNamespace(**self.identity))
        self.bundle = None
        self.external_state_overrides = {}
        if self.spec.get('block_bundle'):
            bundle_path = Path(self.spec['block_bundle']).resolve()
            self.bundle = (bundle_path, json.loads(bundle_path.read_text()))
            if (self.bundle[1].get('schema') != 'natlang.rl-block-bundle/1'
                    or self.bundle[1].get('policy_id') != self.identity['checkpoint_sha256']):
                raise ValueError('native objective archive belongs to a different actor policy')

    def forward_inputs(self, context):
        from pathlib import Path
        from ..common.hashing import sha256_file_hex_cached
        from ..serve.store import decode_native_state
        blocks = {}
        for example in context['examples']:
            if example.id not in self.rows:
                raise ValueError('native objective has no captured policy row for this graph')
            for name, binding in example.external_states.items():
                if self.bundle is None:
                    raise ValueError('native RL external observations require their durable state bundle')
                path, manifest = self.bundle
                entry = manifest['blocks'][binding['id']]
                artifact = (path.parent / entry['path']).resolve()
                if artifact.parent != path.parent or artifact.stat().st_size != entry['bytes']:
                    raise ValueError('native RL external artifact escapes archive or changed size')
                if sha256_file_hex_cached(artifact) != entry['sha256']:
                    raise ValueError('native RL external artifact bytes changed')
                block, block_id = decode_native_state(artifact.read_bytes())
                block.require_native(**self.identity)
                if block_id != binding['id'] or len(block.token_ids) != binding['token_count']:
                    raise ValueError('native RL archived state differs from graph input binding')
                override = self.external_state_overrides.get(binding['id'])
                if override is not None:
                    override.require_native(**self.identity)
                    if len(override.token_ids) != binding['token_count']:
                        raise ValueError('native RL state-gradient override differs from admitted graph binding')
                    block = override
                packed_name = context['external_names'][example.id][name]
                blocks[packed_name] = block
        return {'external_blocks': blocks, 'external_identity': self.identity} if blocks else {}

    def __call__(self, model, result, context):
        from .full_policy import clipped_token_loss
        from ..model.native_adapters import require_model_execution_actor
        require_model_execution_actor(model, self.identity, self.adapter_binding)
        hidden = result['h_final'][0]
        losses, count = [], 0
        for example in context['examples']:
            row = self.rows[example.id]
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


def prepare_objective(rollouts, *, identity, roles, fetch_block, block_bundle=None, mode='grpo'):
    """Admit whole training episodes and retain real graph/action provenance."""
    from ..model.native_graph_plan import invocation_dict
    from ..data.native_graph import SCHEMA, parse_graph
    from ..common.hashing import canonical_json_sha256_hex
    from .graph_replay import producer_turns
    from .grpo import group_advantages
    from ..serve.store import native_state_id
    if mode not in {'grpo', 'rft'}:
        raise ValueError('native replay objective must be grpo or rft')
    grouped = {}
    for rollout in rollouts:
        if rollout.get('reward') is None:
            continue
        if (rollout.get('split') != 'train' or rollout.get('training_admission') not in (None, 'grpo')
                or rollout['reward'].get('quality') not in (0, 1)
                or (rollout.get('policy') or {}).get('checkpoint_sha256') != identity['checkpoint_sha256']):
            raise ValueError('native RL requires exact scored train-only actor episodes')
        if not rollout.get('task_id') or not rollout.get('turns') or any(
                not (turn.get('replay') or {}).get('raw_calls_retained') for turn in rollout['turns']):
            raise ValueError('native RL episode lacks task identity or exact raw-call replay')
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
            replay = compile_replay(producer_turns(rollout), identity=identity, roles=roles, fetch_block=fetch_block)
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
                'actions': list(replay.actions), 'advantage': float(advantage or 0.),
                'task_id': rollout['task_id'], 'sample': rollout.get('sample')}
            graphs.append(graph)
    return graphs, {'schema': 'natlang.native-policy-objective/1', 'kind': 'native-policy',
        'mode': mode, 'policy': dict(identity), 'rows': objective_rows,
        **({'adapter_binding': adapter_binding} if adapter_binding else {}),
        'block_bundle': str(block_bundle) if block_bundle else None}
