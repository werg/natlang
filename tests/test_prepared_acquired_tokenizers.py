"""Actual two-source fresh/reused tokenizer byte parity and mutation protection."""
import copy
import json
from pathlib import Path
import shutil

import pytest
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import write_jsonl_text
from natlang_neuralese.common.paths import resolve
from natlang_neuralese.model import native_model
from scripts.neuralese_data import build_acquired_full_session_graphs as builder

BASES=[resolve('data_nvme','dgxops/qad-drift-probe/base'),resolve('models','mellum21-12b-a2.5b-thinking')]


def normalized(identifier):
    return {'schema':'natlang.normalized-agent-trajectory/1','whole_trajectory':True,'windowed':False,
        'positive_candidate':True,'source_row':identifier+':0','source_file':{'path':identifier,'sha256':'a'*64},
        'source_revision':'pinned','trajectory':{'id':identifier,'split':'train','split_groups':[identifier],
        'outcome':{'label':'teacher'},'lineage':{'teacher':'existing source'},'license':{'name':'source'},
        'tools':None,'messages':[{'role':'user','content':'Complete original task '+identifier},
            {'role':'assistant','reasoning':'Original early reasoning','content':'Inspect source'},
            {'role':'tool','content':'Exact original observation'},
            {'role':'assistant','reasoning':'Original final reasoning','content':'Original answer '+identifier}]}}


def export(path,identifier):
    path.mkdir();row=normalized(identifier);source=path/'full-trajectories.jsonl.zst'
    with write_jsonl_text(source) as stream:stream.write(json.dumps(row)+'\n')
    receipt=path/'export.json';receipt.write_text(json.dumps({'schema':'natlang.normalized-agent-source-file/1',
        'contract':{'source':row['source_file'],'revision':row['source_revision']},
        'outputs':{source.name:sha256_file_hex(source)},'complete_sessions':1}))
    return receipt


@pytest.mark.parametrize('base',BASES,ids=['lfm','mellum'])
def test_actual_fresh_and_reused_two_source_outputs_and_receipts_are_byte_identical(base,tmp_path,monkeypatch):
    if not (base/'tokenizer.json').is_file():pytest.skip('actual pinned tokenizer unavailable')
    prepared=builder.prepare_tokenizer(base)
    monkeypatch.setattr(builder,'current_protection',lambda path:{'held':set(),'quality':set(),'pending':set(),'proof':{}})
    holds=sorted((Path(builder.__file__).resolve().parents[2]/'training/decision-holds').glob('*.jsonl'))
    for index in range(2):
        receipt=export(tmp_path/('source-'+str(index)),'original-source-'+str(index));output=tmp_path/('graphs-'+str(index))
        argv=['--export-receipt',str(receipt),'--base-tokenizer',str(base),'--protection','unit-fixture','--output',str(output)]
        for hold in holds:argv+=['--decision-hold',str(hold)]
        builder.main(argv)
        expected={str(p.relative_to(output)):sha256_file_hex(p) for p in output.rglob('*') if p.is_file() and p.name!='.conversion.lock'}
        shutil.rmtree(output)
        builder.main(argv,prepared=prepared)
        observed={str(p.relative_to(output)):sha256_file_hex(p) for p in output.rglob('*') if p.is_file() and p.name!='.conversion.lock'}
        assert observed==expected,'graphs, all immutable receipts and resume-state must match fresh loading byte for byte'
        assert builder.acquire_output_lease(output).close() is None
        prepared.validate(base)


@pytest.mark.parametrize('base',BASES,ids=['lfm','mellum'])
def test_actual_reusable_context_rejects_backend_render_and_binding_mutation(base):
    if not (base/'tokenizer.json').is_file():pytest.skip('actual pinned tokenizer unavailable')
    prepared=builder.prepare_tokenizer(base);tokenizer=prepared.tokenizer
    old=tokenizer.chat_template;tokenizer.chat_template=old+' changed'
    with pytest.raises(ValueError,match='vocabulary/backend/template'):prepared.validate(base)
    tokenizer.chat_template=old;prepared.validate(base)
    old=tokenizer.padding_side;tokenizer.padding_side='left' if old=='right' else 'right'
    with pytest.raises(ValueError,match='rendering configuration'):prepared.validate(base)
    tokenizer.padding_side=old;prepared.validate(base)
    saved=copy.deepcopy(prepared.rendering_configuration);prepared.rendering_configuration['history_reasoning']['policy']='changed'
    with pytest.raises(ValueError,match='binding was mutated'):prepared.validate(base)
    prepared.rendering_configuration.clear();prepared.rendering_configuration.update(saved);prepared.validate(base)
    old=tokenizer.apply_chat_template;tokenizer.apply_chat_template=lambda *args,**kw:'different live renderer'
    with pytest.raises(ValueError,match='rendering configuration'):prepared.validate(base)
    tokenizer.apply_chat_template=old;prepared.validate(base)


def test_shared_inventory_preserves_base_result_without_tokenizer_weight_reads(tmp_path,monkeypatch):
    for name in ('config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','model.safetensors','irrelevant.txt'):
        (tmp_path/name).write_text(name)
    expected={name:sha256_file_hex(tmp_path/name) for name in
        ('chat_template.jinja','config.json','model.safetensors','tokenizer.json','tokenizer_config.json')}
    assert native_model.base_inventory(tmp_path)==expected
    original=native_model.sha256_file_hex_cached
    def metadata_only(path):
        assert Path(path).suffix!='.safetensors','tokenizer preparation must not read backbone shards'
        return original(path)
    monkeypatch.setattr(native_model,'sha256_file_hex_cached',metadata_only)
    assert native_model.tokenizer_inventory(tmp_path)=={k:v for k,v in expected.items() if k!='model.safetensors'}


def test_reusable_context_rejects_changed_base_inventory(tmp_path,monkeypatch):
    # This exact verifier is independent of GPU/model size and the renderer.
    files={'config.json':'a'}
    monkeypatch.setattr(builder,'tokenizer_inventory',lambda path:{'config.json':'changed'})
    binding={'schema':'natlang.prepared-native-tokenizer/1','base_files':files,'source_fingerprint':'s',
        'expanded_fingerprint':'e','rendering_configuration':{}}
    context=builder.PreparedTokenizer(object(),tmp_path,'s','e',files,{},None,builder.canonical_json_sha256_hex(binding))
    with pytest.raises(ValueError,match='base/configuration inventory changed'):context.validate(tmp_path)
