from scripts.neuralese_data.review_acquired_supervised_samples import assess_call


def test_complete_tool_markup_is_checked_without_claiming_execution():
    assert assess_call('<function=browser.search><parameter=query>x</parameter></function>') == {
        'well_formed_markup': True, 'function':'browser.search'}
    assert assess_call('<function=browser.search><parameter=query>x</function>')['well_formed_markup'] is False
    assert assess_call('<function=bad name></function>')['well_formed_markup'] is False
    assert assess_call('<function=browser.search>unexpected</function>')['well_formed_markup'] is False
