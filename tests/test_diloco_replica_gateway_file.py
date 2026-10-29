import pytest
from natlang_neuralese.train.diloco.replica_cli import gateway_credential

def test_gateway_credential_is_read_from_bound_file(tmp_path):
    path=tmp_path/'gateway';path.write_text('secret-token\n')
    assert gateway_credential({'gateway_file':str(path)})=='secret-token'
    assert gateway_credential({}) is None
    with pytest.raises(ValueError,match='one credential'):gateway_credential({'gateway_file':str(path),'gateway':'other'})
    path.write_text(' ')
    with pytest.raises(ValueError,match='empty'):gateway_credential({'gateway_file':str(path)})
