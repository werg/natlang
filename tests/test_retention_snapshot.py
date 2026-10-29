import pytest
from scripts.coreweave.retention_snapshot import snapshot,revalidate


def test_new_removed_and_changed_json_invalidate_snapshot(tmp_path):
    p=tmp_path/'a.json';p.write_text('{}');before=snapshot([tmp_path]);revalidate([tmp_path],before)
    p.write_text('{"x":1}')
    with pytest.raises(RuntimeError):revalidate([tmp_path],before)
    before=snapshot([tmp_path]);(tmp_path/'b.json').write_text('{}')
    with pytest.raises(RuntimeError):revalidate([tmp_path],before)
    before=snapshot([tmp_path]);p.unlink()
    with pytest.raises(RuntimeError):revalidate([tmp_path],before)


def test_delta_refresh_includes_new_references_and_forgets_deleted(tmp_path):
    from scripts.coreweave.retention_snapshot import refresh_references
    p=tmp_path/'a.json';p.write_text('safe');old=snapshot([tmp_path]);d='a'*64
    p.write_text(d);q=tmp_path/'b.json';q.write_text(d)
    def scan(ds,paths):return {x:{str(p) for p in map(__import__('pathlib').Path,paths) if x in p.read_text()} for x in ds}
    current,refs=refresh_references([tmp_path],old,{d:set()},[d],scan)
    assert refs[d]=={str(p),str(q)}
    q.unlink();_,refs=refresh_references([tmp_path],current,refs,[d],scan)
    assert refs[d]=={str(p)}
