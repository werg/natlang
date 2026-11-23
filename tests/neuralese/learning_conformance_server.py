"""A real Python native engine over a tiny LFM backbone, for the `natlang:learning` conformance cases.

`ts-host/scripts/native-conformance.mjs` starts it, reads one JSON line (`endpoint`, the native identity and a
projection parameter name for adapter targets) from stdout, runs the cases against it and closes stdin to stop it.
Commands on stdin, one per line, answer one JSON line each:

- `weights`: SHA-256 of the live model parameters, so a case can check that learning never changed them.
- `calls`: how many generation requests the engine has served since start.
- `policy default|alt`: the token policy below; `alt` changes every choice, so a replay that sampled again would differ.

Token selection is a deterministic fixture policy, not a learned behaviour: a call whose opening line declares a
`Neuralese<...>` result writes a native span, every other call ends at once (see `fixture_token`). The driver on the
Node side turns each reply into a typed `return_result`; the forwards, captures, splices, replays and gradients are
the engine's own. Same construction as `test_native_typed_harness.py` (mode `recorded`).
"""
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import WhitespaceSplit  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from test_shared_state import tiny  # noqa: E402
from natlang_neuralese.model.shared_state import SharedStateModel, RoleTokens, ROLE_MARKERS  # noqa: E402
from natlang_neuralese.model.native_model import native_identity, model_state  # noqa: E402
from natlang_neuralese.data.text_corpus import tokenizer_fingerprint  # noqa: E402
from natlang_neuralese.common.hashing import sha256_file_hex  # noqa: E402
from natlang_neuralese.serve.engine import Engine  # noqa: E402
from natlang_neuralese.serve.store import TensorStore  # noqa: E402
from natlang_neuralese.serve.http import serve  # noqa: E402

PRODUCER = re.compile(r'^You are inside this call: [^\n]*\): Neuralese<')


def _opening(messages):
    for message in messages:
        if message.get('role') == 'user':
            content = message.get('content')
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return ''.join(part.get('text', '') for part in content if isinstance(part, dict))
            return ''
    return ''


def build(device):
    port = tiny('lfm')
    if device != 'cpu':
        port.hf.to(device=device, dtype=torch.float32)
    model = SharedStateModel(port, (60, 61, 62, 63), rank=4).eval()
    torch.manual_seed(7)
    torch.nn.init.normal_(model.adapter.up.weight, std=.02)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    vocabulary = {f't{i}': i for i in range(port.embedding_weight.shape[0])}
    for marker, token in zip(ROLE_MARKERS, model.marker_ids):
        del vocabulary[f't{token}']
        vocabulary[marker] = token
    for spelling, index in [('<|im_start|>', 4), ('<|im_end|>', 10)]:
        del vocabulary[f't{index}']
        vocabulary[spelling] = index
    raw = Tokenizer(WordLevel(vocabulary, unk_token='t0'))
    raw.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='t0', bos_token='t1', eos_token='t2',
        pad_token='t3', additional_special_tokens=list(ROLE_MARKERS) + ['<|im_start|>', '<|im_end|>'])
    tokenizer.chat_template = ("{% for message in messages %}<|im_start|>{{ message['role'] }} "
        "{% if message['role'] == 'assistant' and message.get('reasoning_content') and loop.last %}"
        "{{ message['reasoning_content'] }} {% endif %}{{ message['content'] }}"
        "{% for tool in message.get('tool_calls', []) %}"
        "{{ tool['function']['name'] }}({{ tool['function']['arguments'] | tojson }})"
        "{% endfor %}<|im_end|>{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant {% endif %}")
    config = {'architecture': port.config.model_type, 'tokenizer_fingerprint': tokenizer_fingerprint(tokenizer)}
    weights = Path(tempfile.mkdtemp(prefix='learning-conformance-')) / 'weights.pt'
    torch.save(model_state(model), weights)
    identity = native_identity(model, config, checkpoint_sha256=sha256_file_hex(weights))
    engine = Engine(port, tokenizer, TensorStore(), f"native-state/{config['architecture']}",
        shared_model=model, role_tokens=RoleTokens(*model.marker_ids), model_config=config,
        checkpoint_sha256=identity['checkpoint_sha256'], embedding_weight_sha256=identity['embedding_weight_sha256'],
        device=device)
    engine.native_identity = identity
    engine.context = 8192
    roles = engine.role_tokens

    # `default`: a producer writes one payload token, any other call stops at once. `alt` writes two payload tokens
    # and puts one text token before every stop, so a replay that sampled again could not match a recorded turn.
    policy = {'name': 'default'}

    def fixture_token(sequence):
        alt = policy['name'] == 'alt'
        if sequence.native_capture_open:
            span = (5, 6) if alt else (5,)
            written = len(sequence.native_capture_ids)
            return (span[written] if written < len(span) else roles.produce_close), None
        if sequence.generated_positions == 0 and PRODUCER.match(_opening(sequence.request.messages)):
            return roles.produce_open, None
        if alt and not sequence.items:
            return 7, None
        return 2, None

    engine._native_token = fixture_token
    target = next(name for name, _ in model.named_parameters() if name.endswith('q_proj.weight'))
    return engine, model, identity, target, policy


def weights_sha(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().to('cpu', torch.float32).contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    # The protocol owns stdout; anything a library prints goes to stderr.
    protocol, sys.stdout = sys.stdout, sys.stderr

    def say(value):
        protocol.write(json.dumps(value) + '\n')
        protocol.flush()

    torch.set_num_threads(int(os.environ.get('NATLANG_CONFORMANCE_THREADS', '1')))
    device = os.environ.get('NATLANG_CONFORMANCE_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')
    engine, model, identity, target, policy = build(device)
    served = {'calls': 0}
    original = engine.submit if hasattr(engine, 'submit') else None
    if original is not None:
        def counted(*args, **kwargs):
            served['calls'] += 1
            return original(*args, **kwargs)
        engine.submit = counted
    engine.start()
    server = serve(engine, '127.0.0.1', 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    say({'endpoint': f'http://127.0.0.1:{server.server_address[1]}', 'identity': identity,
        'target': target, 'device': device})
    for line in sys.stdin:
        command = line.strip()
        if command == 'weights':
            say({'weights': weights_sha(model)})
        elif command.startswith('policy '):
            name = command.split(' ', 1)[1]
            if name not in ('default', 'alt'):
                say({'error': f'unknown policy {name!r}'})
                continue
            policy['name'] = name
            say({'policy': name})
        elif command == 'calls':
            say({'calls': served['calls']})
        elif command:
            say({'error': f'unknown command {command!r}'})
    server.shutdown()


if __name__ == '__main__':
    main()
