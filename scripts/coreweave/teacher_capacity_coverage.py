"""Flatten authenticated completed teacher coverage without reading graph payloads."""
from pathlib import Path
import hashlib,json,os,tempfile


def _state(path):
    value=path.stat()
    return value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns


def _read(reference):
    path=Path(reference['path']).resolve(strict=True);before=_state(path);raw=path.read_bytes()
    if len(raw)!=reference['bytes'] or hashlib.sha256(raw).hexdigest()!=reference['sha256'] or _state(path)!=before:
        raise ValueError('completed teacher coverage binding changed: '+str(path))
    return path,raw,before


def _member(receipt,root,name):
    matches=[]
    for ref in receipt['files']:
        path=Path(ref['path'])
        if '..'in path.parts:raise ValueError('teacher coverage member traverses its receipt root')
        path=path if path.is_absolute()else root/path
        try:relative=path.relative_to(root)
        except ValueError:raise ValueError('teacher coverage member escaped its receipt root')from None
        if str(relative)==name:matches.append({**ref,'path':str(path)})
    if len(matches)!=1:raise ValueError('completed teacher coverage member missing or repeated: '+name)
    return matches[0]


def flatten_teacher_fit_coverage(receipt_reference,sources,output,*,origin_sources):
    """Preserve each original line; order it against the exact warm source entries."""
    receipts=[];coverage=[];guards={};rows={};active=set()
    def read(ref):
        path,raw,before=_read(ref);old=guards.setdefault(str(path),(ref,before))
        if old!=(ref,before):raise ValueError('teacher coverage repeats a changed binding')
        return path,raw
    def visit(ref):
        path,raw=read(ref)
        if str(path)in active:raise ValueError('teacher coverage component cycle')
        active.add(str(path));receipt=json.loads(raw);receipts.append(ref)
        if(receipt.get('schema')!='natlang.admitted-ordinary-source-capacity-preparation/1'
           or receipt.get('passed')is not True or receipt.get('source_eligibility_unchanged')is not True
           or receipt.get('exact_original_row_bytes')is not True or set(receipt['counts'])!=set(sources)
           or set(receipt['source_corpora'])!={cid for model in receipt['counts'].values()for cid in model}
           or any(n!=0 for n in receipt['pending_rows'].values())
           or any(any(c[k]!=0 for k in('held','pending','duplicate'))for model in receipt['counts'].values()for c in model.values())):
            raise ValueError('teacher coverage needs a complete admitted FIT-only partition')
        if any(Path(f['path']).name=='coverage.jsonl'for f in receipt['files']):
            bound=_member(receipt,path.parent,'coverage.jsonl');_,data=read(bound);coverage.append(bound);counts={m:{cid:0 for cid in receipt['source_corpora']}for m in sources}
            for raw_line in data.splitlines(keepends=True):
                if not raw_line.endswith(b'\n'):raise ValueError('teacher coverage line is incomplete')
                row=json.loads(raw_line);model=row.get('model');cid=row.get('corpus');key=(model,row.get('id'))
                if(model not in sources or cid not in counts[model]or row.get('split')!='train'or row.get('disposition')!='fit'or key in rows):
                    raise ValueError('teacher coverage repeats or includes a foreign/non-FIT row')
                rows[key]=(row,raw_line);counts[model][cid]+=1
            if any(counts[m][cid]!=receipt['counts'][m][cid]['fit']for m in sources for cid in receipt['source_corpora']):
                raise ValueError('teacher component coverage count differs')
        else:
            _,data=read(_member(receipt,path.parent,'component-coverage.json'));component=json.loads(data)
            declared={r['path']:r for r in receipt['component_receipts']};children=component['new_closures'];seen=set();merged={m:{}for m in sources}
            if not children:raise ValueError('teacher merged coverage has no new components')
            for child in children:
                if declared.get(child['path'])!=child:raise ValueError('teacher coverage component lacks exact receipt authority')
                value=visit(child);cids=set(value['source_corpora'])
                if seen&cids:raise ValueError('teacher coverage component repeats a source corpus')
                seen.update(cids)
                for model in sources:merged[model].update(value['counts'][model])
            if seen!=set(receipt['source_corpora'])or merged!=receipt['counts']:
                raise ValueError('teacher merged coverage lost or added component obligations')
        active.remove(str(path));return receipt
    root=visit(receipt_reference);emitted=[];order={};source_bindings={}
    for model,source in sources.items():
        source_ref={'path':str(source.path),'bytes':source.path.stat().st_size,'sha256':source.file_sha256};read(source_ref);source_bindings[model]=source_ref
        original=origin_sources[model];declared=root['fit_sources'][model];read(declared)
        if(str(original.path.resolve())!=str(Path(declared['path']).resolve())or original.file_sha256!=declared['sha256']
           or source.entries!=original.entries):
            raise ValueError('teacher merged cached source binding/order differs from original fit authority')
        expected=sum(row['fit']for row in root['counts'][model].values())
        if len(source.entries)!=expected:raise ValueError('teacher merged source count differs from coverage')
        order[model]=[]
        for entry in source.entries:
            key=(model,entry['id']);value=rows.pop(key,None)
            if value is None:raise ValueError('teacher merged coverage omitted a source row')
            row,raw=value;geometry=row.get('geometry')or{}
            if(not isinstance(row.get('source_graph_sha256'),str)or len(row['source_graph_sha256'])!=64 or entry.get('communication')
               or set(geometry)!={'forward_tokens','kv_slots'}
               or any(type(v)is not int or v<1 for v in geometry.values())
               or geometry['forward_tokens']!=entry['forward_tokens']or geometry['kv_slots']!=entry['forward_tokens']):
                raise ValueError('teacher merged coverage graph binding or geometry differs')
            emitted.append(raw);order[model].append({'id':entry['id'],'binding':entry['binding']})
    if rows:raise ValueError('teacher merged coverage contains unconsumed fitted rows')
    for ref,before in guards.values():
        if _read(ref)[2]!=before:raise ValueError('teacher coverage inputs changed while flattening')
    data=b''.join(emitted);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        if output.read_bytes()!=data:raise ValueError('existing flat teacher coverage differs')
    else:
        fd,name=tempfile.mkstemp(prefix='.coverage-',dir=output.parent)
        try:
            with os.fdopen(fd,'wb')as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
            try:os.link(name,output)
            except FileExistsError:
                if output.read_bytes()!=data:raise ValueError('concurrent flat teacher coverage differs')
        finally:Path(name).unlink(missing_ok=True)
    fd=os.open(output.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return {'schema':'natlang.merged-teacher-flat-fit-coverage/1','source_receipt':receipt_reference,
            'component_receipts':receipts,'component_coverage':coverage,'source_manifests':source_bindings,'rows':{m:len(s.entries)for m,s in sources.items()},
            'coverage':{'path':str(output),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()},
            'ordered_source_rows':order,'original_indexed_sources':root['fit_sources'],
            'binding_laws':'source_graph_sha256 retains original raw graph hash; indexed binding retains normalized native graph identity; exact old/new indexed entries and row IDs are compared, never equated across hash laws',
            'source_graphs_reparsed':False,'training_admission':False}
