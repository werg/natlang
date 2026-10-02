import json

import pytest

from scripts import ling_training_compat as compat


FORWARD_SOURCE = '''class BailingMoeV3ForCausalLM:
    def forward(
        self,
        input_ids=None,
        labels=None,
        output_attentions=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        outputs = self.model(input_ids=input_ids, **kwargs)
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config.vocab_size, **kwargs)
        return SimpleNamespace(loss=loss, logits=logits)
'''


def test_unknown_upstream_source_is_rejected():
    with pytest.raises(ValueError, match='Unreviewed Ling source'):
        compat.patch_source(FORWARD_SOURCE)


def test_patch_requires_unambiguous_anchors():
    with pytest.raises(ValueError, match='missing or ambiguous'):
        compat.replace_once('one one', 'one', 'two')


@pytest.mark.parametrize('completion_length', [1, 7])
def test_completion_projection_matches_full_loss_and_gradients(completion_length):
    torch = pytest.importorskip('torch')
    from types import SimpleNamespace
    patched = compat.patch_completion_forward(FORWARD_SOURCE)
    namespace = {'SimpleNamespace': SimpleNamespace}
    exec(patched, namespace)
    forward = namespace['BailingMoeV3ForCausalLM'].forward
    torch.manual_seed(137)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(32, 16)
            self.lm_head = torch.nn.Linear(16, 32, bias=False)
            self.config = SimpleNamespace(output_attentions=False, vocab_size=32)
            self.num_nextn_predict_layers = 0
            self.forward = forward.__get__(self)

        def model(self, input_ids, **kwargs):
            return (self.embedding(input_ids),)

        def loss_function(self, logits, labels, vocab_size, **kwargs):
            return torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, vocab_size),
                labels[:, 1:].reshape(-1), ignore_index=-100)

    model = Model()
    tokens = torch.randint(0, 32, (1, 9 + completion_length))
    labels = tokens.clone()
    labels[:, :9] = -100
    full = model(input_ids=tokens, labels=labels).loss
    full.backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    tail = completion_length + 1
    trimmed = model(input_ids=tokens, labels=labels[:, -tail:], logits_to_keep=tail)
    assert trimmed.logits.shape == (1, tail, 32)
    trimmed.loss.backward()
    torch.testing.assert_close(trimmed.loss, full)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, gradients[name])
    with pytest.raises(TypeError, match='integer'):
        model(input_ids=tokens, logits_to_keep=True)
    with pytest.raises(ValueError, match='nonnegative'):
        model(input_ids=tokens, logits_to_keep=-1)
    with pytest.raises(ValueError, match='exceeds'):
        model(input_ids=tokens, logits_to_keep=tokens.shape[1] + 1)
    model.num_nextn_predict_layers = 1
    with pytest.raises(ValueError, match='MTP'):
        model(input_ids=tokens, logits_to_keep=1)


def fixture_snapshot(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    (source / compat.MODEL_FILE).write_text(FORWARD_SOURCE)
    config = {'architectures': ['BailingMoeV3ForCausalLM'], 'num_nextn_predict_layers': 0}
    (source / 'config.json').write_text(json.dumps(config))
    (source / 'configuration_bailing_moe_v3.py').write_text('# configuration\n')
    monkeypatch.setattr(compat, 'patch_source', lambda text: text + '\n# patched\n')
    monkeypatch.setattr(compat, 'SOURCE_SHA256', compat.sha256_file(source / compat.MODEL_FILE))
    monkeypatch.setattr(compat, 'CONFIG_SHA256', compat.sha256_file(source / 'config.json'))
    monkeypatch.setattr(compat, 'CONFIGURATION_SHA256', compat.sha256_file(source / 'configuration_bailing_moe_v3.py'))
    return source


def test_prepare_preserves_source_links_weights_and_records_patch(tmp_path, monkeypatch):
    source = fixture_snapshot(tmp_path, monkeypatch)
    weight = tmp_path / 'weight-blob'
    weight.write_bytes(b'test weights')
    (source / 'model.safetensors').symlink_to(weight)
    original = (source / compat.MODEL_FILE).read_bytes()
    output = tmp_path / 'prepared'
    receipt = compat.prepare(source, output)
    assert (source / compat.MODEL_FILE).read_bytes() == original
    assert (output / 'model.safetensors').is_symlink()
    assert (output / 'model.safetensors').resolve() == weight
    assert not (output / compat.MODEL_FILE).is_symlink()
    assert receipt['patched_source_sha256'] == compat.sha256_file(output / compat.MODEL_FILE)
    assert receipt['not_enabled'] == ['sorted_moe_dispatch', 'padding_mask_fix']
    with pytest.raises(FileExistsError):
        compat.prepare(source, output)
    with pytest.raises(ValueError, match='outside'):
        compat.prepare(source, source / 'nested')


def test_failed_prepare_cleans_only_new_directory(tmp_path, monkeypatch):
    source = fixture_snapshot(tmp_path, monkeypatch)
    (source / 'model.safetensors.index.json').write_text(json.dumps(
        {'weight_map': {'weight': 'missing.safetensors'}}))
    output = tmp_path / 'prepared'
    with pytest.raises(ValueError, match='weight index'):
        compat.prepare(source, output)
    assert source.exists()
    assert not output.exists()


def test_mutated_configuration_rejected_before_output(tmp_path, monkeypatch):
    source = fixture_snapshot(tmp_path, monkeypatch)
    (source / 'configuration_bailing_moe_v3.py').write_text('# changed\n')
    output = tmp_path / 'prepared'
    with pytest.raises(ValueError, match='Unreviewed Ling configuration'):
        compat.prepare(source, output)
    assert not output.exists()


def test_model_source_hash_checks_original_bytes(tmp_path, monkeypatch):
    source = fixture_snapshot(tmp_path, monkeypatch)
    path = source / compat.MODEL_FILE
    # read_text would normalize CRLF and incorrectly accept the original hash.
    path.write_bytes(path.read_bytes().replace(b'\n', b'\r\n'))
    with pytest.raises(ValueError, match='source bytes'):
        compat.prepare(source, tmp_path / 'prepared')
