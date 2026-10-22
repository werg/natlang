"""Checkpoint snapshots keep mutable ownership and Python's copy protocol."""
import copy
from collections import defaultdict
from pathlib import Path

import pytest
import torch

from natlang_neuralese.common.copying import deepcopy_metadata


def test_atomic_tables_are_independent_and_keep_atom_identity():
    source={'rows':{'one':'binding1','two':'binding2'},'counts':{'one':1,'two':0},
            'order':['one','two'], 'empty':{}, 'none':None,'bytes':b'payload'}
    actual=deepcopy_metadata(source)
    assert actual==copy.deepcopy(source)
    for key in ['rows','counts','order','empty']:
        assert actual[key]is not source[key]
    assert actual['rows']['one']is source['rows']['one']
    actual['counts']['one']=9;actual['order'].append('new')
    assert source['counts']['one']==1 and source['order']==['one','two']


def test_mixed_cycles_aliases_and_tuple_fallback_match_deepcopy():
    row={'binding':'hash','visits':1};source={'rows':[row,row]}
    source['self']=source;source['tuple']=(source['rows'],row)
    actual=deepcopy_metadata(source)
    assert actual is actual['self']
    assert actual['rows'][0]is actual['rows'][1]is actual['tuple'][1]
    assert actual['tuple'][0]is actual['rows']
    assert actual['rows'][0]is not row
    actual['rows'][0]['visits']=2
    assert row['visits']==1


class CustomAtom(str):
    def __deepcopy__(self,memo):
        return CustomAtom('copied:'+self)


class CustomReference:
    def __init__(self,target):self.target=target
    def __deepcopy__(self,memo):
        result=type(self)(None);memo[id(self)]=result
        result.target=copy.deepcopy(self.target,memo)
        return result


def test_subclasses_and_custom_hooks_share_the_normal_copy_memo():
    row=defaultdict(list,{'binding':['original']})
    source={'row':row,'again':row,'reference':CustomReference(row),
            'atom':CustomAtom('original'),'path':Path('/exact/source')}
    actual=deepcopy_metadata(source);reference=copy.deepcopy(source)
    assert type(actual['row'])is defaultdict
    assert actual['row']==reference['row']
    assert actual['row']is actual['again']is actual['reference'].target
    assert actual['row']is not row and actual['row']['binding']is not row['binding']
    assert type(actual['atom'])is CustomAtom and actual['atom']=='copied:original'
    assert actual['path']==reference['path']


def test_tensor_fallback_retains_storage_aliases_and_independent_values():
    tensor=torch.arange(6,dtype=torch.float32)
    source={'master':tensor,'same':tensor,'view':tensor.view(2,3)}
    actual=deepcopy_metadata(source)
    assert actual['master']is actual['same']
    assert actual['master'].untyped_storage().data_ptr()==actual['view'].untyped_storage().data_ptr()
    assert actual['master'].untyped_storage().data_ptr()!=tensor.untyped_storage().data_ptr()
    actual['master'][0]=99
    assert actual['view'][0,0].item()==99 and tensor[0].item()==0


def test_full_progress_matches_original_capture_and_remains_mutable_independently(monkeypatch,tmp_path):
    from natlang_neuralese.train import native_continuation as continuation
    binding={str(i):str(i*17)for i in range(100)}
    prior={'source_row_bindings':binding,'source_row_order':list(binding),
           'exposure_counts':{key:1 for key in binding},
           'options':{'epochs':1},'completed_foundations':[{'earned':True}]}
    training={'schema':continuation.TRAINING_SCHEMA,'identity':{'objective':None},
              'source_row_bindings':binding,'source_row_order':list(binding),
              'exposure_counts':{key:2 for key in binding},
              'options':{'stage_id':'active','epochs':1},
              'sft_progress':{'prior':prior,'alias':prior,'active':prior}}
    actual=continuation.collect_sft_progress(training)
    monkeypatch.setattr(continuation,'deepcopy_metadata',copy.deepcopy)
    reference=continuation.collect_sft_progress(training)
    assert actual==reference
    assert actual['prior']is actual['alias']
    assert actual['active']['source_row_bindings']is not binding
    actual['active']['exposure_counts']['1']=99
    actual['prior']['completed_foundations'][0]['earned']=False
    assert training['exposure_counts']['1']==2 and prior['completed_foundations'][0]['earned']
    saved=tmp_path/'progress.pt';torch.save(actual,saved)
    restored=torch.load(saved,weights_only=True)
    assert restored==actual and restored['prior']is restored['alias']
