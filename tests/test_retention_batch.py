import pytest
from pathlib import Path
from scripts.coreweave.retention_batch import prepare_reference_evidence,use_reference_evidence


def test_shared_scan_baseline_precedes_scan_and_refreshes_new_consumer(tmp_path):
    digest='a'*64;target=tmp_path/'consumer.json';target.write_text('{}');calls=[]
    def scan(ds,roots):
        calls.append(tuple(map(str,roots)))
        if len(calls)==1:target.write_text(digest);return {d:set() for d in ds}
        return {d:{str(p) for p in map(Path,roots) if p.is_file() and d in p.read_text()} for d in ds}
    evidence=prepare_reference_evidence([tmp_path],[digest],scan)
    assert evidence.references[digest]=={str(target)} and len(calls)==2
    assert use_reference_evidence(evidence,[tmp_path],[digest])[1][digest]=={str(target)}
    with pytest.raises(ValueError):use_reference_evidence(evidence,[tmp_path],['b'*64])
    with pytest.raises(ValueError):use_reference_evidence(evidence,[tmp_path/'other'],[digest])
