import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.coreweave.teacher_capacity_coverage import flatten_teacher_fit_coverage


def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return bind(p)
def leaf(root,cid,ident):
    root.mkdir();row={'model':'fixture','corpus':cid,'id':ident,'split':'train','disposition':'fit','source_graph_sha256':hashlib.sha256(ident.encode()).hexdigest(),'geometry':{'forward_tokens':7,'kv_slots':7},'retained_source_evidence':{'exact':cid}}
    coverage=root/'coverage.jsonl';coverage.write_text(json.dumps(row)+'\n')
    receipt={'schema':'natlang.admitted-ordinary-source-capacity-preparation/1','passed':True,'source_eligibility_unchanged':True,'exact_original_row_bytes':True,'source_corpora':[cid],'counts':{'fixture':{cid:{'fit':1,'pending':0,'held':0,'duplicate':0}}},'pending_rows':{'fixture':0},'files':[bind(coverage)]}
    return write(root/'receipt.json',receipt),row

def merge(root,children):
    root.mkdir();values=[json.loads(Path(c['path']).read_text())for c in children];counts={}
    for v in values:counts.update(v['counts']['fixture'])
    component=write(root/'component-coverage.json',{'new_closures':children})
    receipt={**values[0],'source_corpora':list(counts),'counts':{'fixture':counts},'files':[component],'component_receipts':children}
    return write(root/'receipt.json',receipt)

def fixture(tmp):
    a,ra=leaf(tmp/'a','A','a');b,rb=leaf(tmp/'b','B','b');inner=merge(tmp/'inner',[a]);root=merge(tmp/'root',[inner,b]);manifest=tmp/'fit-sources.json';manifest.write_text('{}')
    entries=[{'id':r['id'],'binding':hashlib.sha256(('normalized-'+r['id']).encode()).hexdigest(),'forward_tokens':7,'communication':False}for r in(rb,ra)]
    source=SimpleNamespace(entries=entries,path=manifest,file_sha256=bind(manifest)['sha256'])
    import copy
    source.origin=SimpleNamespace(entries=copy.deepcopy(entries),path=manifest,file_sha256=source.file_sha256)
    value=json.loads(Path(root['path']).read_text());value['fit_sources']={'fixture':bind(manifest)};root=write(Path(root['path']),value)
    return root,{'fixture':source},[a,b],(ra,rb)


def test_recursive_coverage_uses_actual_source_order_and_preserves_original_rows(tmp_path):
    root,sources,_,(ra,rb)=fixture(tmp_path);out=tmp_path/'flat.jsonl';proof=flatten_teacher_fit_coverage(root,sources,out,origin_sources={'fixture':sources['fixture'].origin})
    assert [json.loads(l)for l in out.read_text().splitlines()]==[rb,ra]
    assert proof['rows']=={'fixture':2}and len(proof['component_receipts'])==4
    assert proof['source_graphs_reparsed']is False and proof['training_admission']is False
    assert flatten_teacher_fit_coverage(root,sources,out,origin_sources={'fixture':sources['fixture'].origin})['coverage']==proof['coverage']

@pytest.mark.parametrize('bad',['source-binding','source-count','wrong-geometry','held','foreign-corpus','unbound-coverage','omission','duplicate','wrong-child-binding','lost-component'])
def test_rejects_changed_or_incomplete_coverage_before_publication(tmp_path,bad):
    root,sources,children,_=fixture(tmp_path);p=Path(children[0]['path']);receipt=json.loads(p.read_text());coverage=Path(receipt['files'][0]['path']);row=json.loads(coverage.read_text())
    if bad=='source-binding':sources['fixture'].entries[1]['binding']='foreign'
    elif bad=='source-count':sources['fixture'].entries.pop()
    elif bad=='wrong-child-binding':root={**root,'sha256':'0'*64}
    elif bad=='lost-component':
        value=json.loads(Path(root['path']).read_text());value['counts']['fixture'].pop('B');root=write(Path(root['path']),value)
    else:
        if bad=='wrong-geometry':row['geometry']['kv_slots']=6
        elif bad=='held':row['split']='test'
        elif bad=='foreign-corpus':row['corpus']='foreign'
        elif bad=='unbound-coverage':row['retained_source_evidence']['exact']='changed'
        lines=[]if bad=='omission'else[row,row]if bad=='duplicate'else[row]
        coverage.write_text(''.join(json.dumps(r)+'\n'for r in lines))
        if bad!='unbound-coverage':
            receipt['files']=[bind(coverage)];new=write(p,receipt)
            inner=Path(json.loads(Path(root['path']).read_text())['component_receipts'][0]['path']);iv=json.loads(inner.read_text());iv['component_receipts']=[new];cp=Path(iv['files'][0]['path']);iv['files']=[write(cp,{'new_closures':[new]})];new_inner=write(inner,iv)
            rv=json.loads(Path(root['path']).read_text());rv['component_receipts'][0]=new_inner;cp=Path(rv['files'][0]['path']);rv['files']=[write(cp,{'new_closures':rv['component_receipts']})];root=write(Path(root['path']),rv)
    out=tmp_path/'not-published.jsonl'
    with pytest.raises((ValueError,KeyError)):flatten_teacher_fit_coverage(root,sources,out,origin_sources={'fixture':sources['fixture'].origin})
    assert not out.exists()


def test_existing_output_is_never_overwritten(tmp_path):
    root,sources,_,_=fixture(tmp_path);out=tmp_path/'flat.jsonl';out.write_text('tampered\n')
    with pytest.raises(ValueError,match='existing'):flatten_teacher_fit_coverage(root,sources,out,origin_sources={'fixture':sources['fixture'].origin})
    assert out.read_text()=='tampered\n'
