import pytest
from scripts.neuralese_data.prepare_static_acquired_successor import append_family_plan


def test_static_native_append_preserves_remaining_quota_objective_and_counts_seven_visits():
    mass={'general-agent':100., 'native-communication':700.}
    parent={'quota_basis':{'communication_min_visits':7},'family_balance':{'unscaled_remaining_mass':mass}}
    balance={'scales':{'general-agent':2.,'native-communication':6/7},'max_scale':8.}
    # The raw single-pass census deliberately has a different family mix.
    plan=append_family_plan(parent,{'families':{'general-agent':{'unscaled_mass':100.},
                                             'native-communication':{'unscaled_mass':100.}}},balance,10.)
    assert plan['appended_required_visits']==7
    assert plan['unscaled_remaining_mass']=={'general-agent':170.,'native-communication':700.}
    assert plan['effective_remaining_shares']==pytest.approx({'general-agent':.25,'native-communication':.75})
    assert plan['total_scaled_remaining_mass']==pytest.approx(plan['total_unscaled_remaining_mass'])
    assert max(plan['scales'].values())<=8
    assert mass=={'general-agent':100.,'native-communication':700.}


def test_static_append_rejects_unknown_or_invalid_actual_quota():
    parent={'quota_basis':{'communication_min_visits':True},'family_balance':{'unscaled_remaining_mass':{'general-agent':10.}}}
    with pytest.raises(ValueError,match='actual positive communication quotas'):
        append_family_plan(parent,{}, {'scales':{'general-agent':1.},'max_scale':8.},1.)
