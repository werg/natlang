"""Actual resident metadata selection survives a serialized DPO→ordinary switch."""
import ast,copy,inspect,json
from pathlib import Path
from types import SimpleNamespace
import pytest,torch
from test_native_resident_predecessors import adopted
from natlang_neuralese.train import native_state_sft as trainer
from natlang_neuralese.train.native_continuation import collect_sft_progress,restore_sft_progress
from natlang_neuralese.train import native_resident as resident


def execute_actual_resident_selection(scope):
    tree=ast.parse(inspect.getsource(trainer.main))
    expected=ast.parse("resident_state and resume and resident_state['payload']['stage_id'] == args.stage_id",mode='eval').body
    blocks=[n for n in ast.walk(tree)if isinstance(n,ast.If)and ast.dump(n.test,include_attributes=False)==ast.dump(expected,include_attributes=False)]
    assert len(blocks)==1
    exec(compile(ast.fix_missing_locations(ast.Module(body=blocks,type_ignores=[])),trainer.__file__,'exec'),scope)


@pytest.mark.parametrize('ordinary,parent,have_frame',[(True,True,True),(False,True,True),(True,False,True),(True,True,False)])
def test_serialized_resident_frame_selects_ordinary_source_without_touching_live_state(tmp_path,monkeypatch,ordinary,parent,have_frame):
    paths,recipe,frame=adopted(tmp_path,monkeypatch)
    directory=tmp_path/'run/resident-data';current=resident.current(directory)
    frame.update(schema=trainer.TRAINING_SCHEMA,step=7,global_step=7,exposure_counts={'a':1,'b':1,'c':0},source_row_order=['a','b','c'])
    frame['options'].update(stage_id='sft',epochs=1)
    preference=tmp_path/'preference.jsonl';preference.write_text('a distinct objective source')
    held=paths['held'];resume={'schema':trainer.TRAINING_SCHEMA,'options':{'stage_id':'preference','epochs':1},'identity':{'objective':{'kind':'preference'},'objective_kind':'native-offline-pairwise'},
        'source_receipts':[{'path':str(preference),'sha256':resident.sha(preference)},frame['source_receipts'][1]],
        'sft_progress':{'sft':frame}if have_frame else{},'source_row_order':[],
        'optimizer':{'actual_current_moment':torch.tensor([3.,4.])},'rng':{'cpu':torch.get_rng_state()},'global_step':17}
    serialized=tmp_path/'training.pt';torch.save(resume,serialized);loaded=torch.load(serialized,weights_only=False);before=copy.deepcopy(loaded)
    carried=collect_sft_progress(loaded)
    args=SimpleNamespace(stage_id='sft',objective_spec=None if ordinary else preference,
        continued_native_graphs=paths['old'],continued_held_native_graphs=held)
    scope={'resident_state':current,'resume':loaded,'args':args,'parent_continuation':parent,
        'carried_progress':carried,'checkpoint_options':{},'Path':Path}
    execute_actual_resident_selection(scope)
    if ordinary and parent and have_frame:
        assert args.continued_native_graphs==paths['old']
        selected=restore_sft_progress(loaded,'sft',collected_progress=carried)
        result=trainer.continuation_predecessors(selected,{'continued_native_graphs':args.continued_native_graphs,'continued_held_native_graphs':args.continued_held_native_graphs},stage_id='sft')
        assert result['continued_native_graphs']==resident.sha(paths['old'])
        assert selected['optimizer']is loaded['optimizer']and selected['rng']is loaded['rng']
        assert selected['exposure_counts']==frame['exposure_counts']and selected['global_step']==17
        with pytest.raises(ValueError,match='predecessor differs'):
            trainer.continuation_predecessors(selected,{'continued_native_graphs':preference},stage_id='sft')
    else:
        assert args.continued_native_graphs==preference
    assert torch.equal(loaded['optimizer']['actual_current_moment'],before['optimizer']['actual_current_moment'])
    assert torch.equal(loaded['rng']['cpu'],before['rng']['cpu'])
    assert loaded['sft_progress']==before['sft_progress']
