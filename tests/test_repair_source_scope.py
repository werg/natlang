from scripts.prepare_online_repair_batch import source_scope_reason

def test_reviewed_sources_outside_batch_are_deferred_not_missing():
    assert source_scope_reason('selected',{'selected'},{'selected','later'}) is None
    assert source_scope_reason('later',{'selected'},{'selected','later'})=='not-selected-for-current-batch'
    assert source_scope_reason('unknown',{'selected'},{'selected','later'})=='reviewed-repair-source-missing'
