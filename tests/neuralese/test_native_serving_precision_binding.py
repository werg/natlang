from natlang_neuralese.common.hashing import canonical_json_sha256_hex


def test_native_serving_binds_master_precision(tmp_path):
    from test_native_serving_adapters import engine_fixture
    engine=engine_fixture(tmp_path,'lfm')
    try:
        precision=engine.serving_config['precision']
        assert precision['weights']=='checkpoint-master'
        assert precision['inline_qat_point']=='inactive'
        assert engine.rl_policy_binding['serving_precision']==precision
        assert engine.rl_policy_binding['serving_config_sha256']==canonical_json_sha256_hex(
            {key:value for key,value in engine.serving_config.items() if key!='identity_sha256'})
    finally:
        engine.store.close()
