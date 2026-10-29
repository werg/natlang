import json,tarfile
from pathlib import Path
import pytest
from scripts.coreweave.native_control_archive import digest,seal_archive


@pytest.mark.parametrize("field",["files","assets"])
def test_archives_committed_proof_and_exact_restore_map_and_detects_tampering(tmp_path,field):
    out=tmp_path/'out';out.mkdir();control_dir=out/'control';control_dir.mkdir()
    proof=tmp_path/'proof.json';proof.write_text('{"schema":"proof"}')
    proof_ref={'path':str(proof),'sha256':digest(proof)}
    producer=tmp_path/'producer.py';provider=tmp_path/'provider.py';helper=tmp_path/'helper.py';selection=tmp_path/'selection.json'
    for path in (producer,provider,helper,selection):path.write_text(path.name)
    control=control_dir/'control.json';control.write_text(json.dumps({'dependencies':{field:[proof_ref]}}))
    ref={'path':str(control),'sha256':digest(control),'bytes':control.stat().st_size}
    result=seal_archive(out,'a'*64,ref,producer,provider,selection,helper,[proof_ref]);archive=Path(result['archive']['path'])
    with tarfile.open(archive) as tar:
        restored=json.load(tar.extractfile('restore-map.json'))['entries'];assert restored[str(proof)]['sha256']==proof_ref['sha256'];assert tar.extractfile(restored[str(proof)]['member']).read()==proof.read_bytes()
    assert seal_archive(out,'a'*64,ref,producer,provider,selection,helper,[proof_ref])==result
    proof.write_text('tampered')
    with pytest.raises(ValueError,match='differs'):seal_archive(out,'a'*64,ref,producer,provider,selection,helper,[proof_ref])


def test_rejects_proof_not_in_actual_collector_closure(tmp_path):
    out=tmp_path/'out';out.mkdir();control=out/'control.json';control.write_text(json.dumps({'dependencies':{'files':[]}}))
    proof=tmp_path/'proof';proof.write_text('proof');extras=[tmp_path/f'file{i}' for i in range(4)]
    for item in extras:item.write_text('x')
    with pytest.raises(ValueError,match='missing from provider closure'):
        seal_archive(out,'a'*64,{'path':str(control),'sha256':digest(control)},*extras,[{'path':str(proof),'sha256':digest(proof)}])
