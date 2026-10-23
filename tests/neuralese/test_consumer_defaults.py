"""Current training recipes bind positional native graphs, not projected sketches."""


def test_native_state_stage_requires_train_and_held_graphs_without_flat_projection_inputs():
    from natlang_neuralese.train.recipe import HANDLERS

    stage = HANDLERS['native_state_sft']
    assert stage['required_inputs'] == {'native_graphs', 'held_native_graphs'}
    roles = stage['required_inputs'] | stage['optional_inputs']
    assert not roles & {'records', 'pieces', 'text_data', 'direct_artifact', 'heads'}
    assert not stage['parameters'] & {
        'sketch_gradient', 'sketch_target_weight', 'sketch_target_backbone_scale', 'content_transport',
        'handover', 'write_curriculum', 'write_depth', 'writer_supervision', 'projection_anchor_weight',
    }
