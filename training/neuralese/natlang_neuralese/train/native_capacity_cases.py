"""Exact capacity case identity checks against real cached source-tree order."""

def resolve_capacity_case_requests(requests, train, *, relocate=False, preparation=None):
    import copy
    from pathlib import Path
    from natlang_neuralese.train.native_resident import bound
    from natlang_neuralese.train.native_source_preparation import NativeSourcePreparation
    if type(relocate) is not bool:
        raise ValueError('capacity case relocation mode must be an explicit boolean')
    if preparation is None:
        preparation = NativeSourcePreparation(train.tokenizer_fingerprint, train.role_ids, cache_only=True)
    if not preparation.cache_only:
        raise ValueError('capacity case metadata requires the existing cache-only source loader')
    preparation.retain(train)
    train_key = (str(Path(train.path).resolve()), train.file_sha256, train.required_split, train.pair_model)
    sources = {train_key: train}
    wanted = {train_key: set()}
    selected = []
    for request in requests:
        if request.get('schema') != 'natlang.native-memory-qualification-request/1' or not request.get('cases'):
            raise ValueError('capacity case metadata needs complete declared requests')
        ref = request['source']
        path = bound(ref)
        if 'bytes' in ref and (type(ref['bytes']) is not int or path.stat().st_size != ref['bytes']):
            raise ValueError('capacity case source byte binding changed')
        split, pair_model = request.get('split', 'train'), request.get('pair_model')
        key = (str(path), ref['sha256'], split, pair_model)
        if key not in sources:
            sources[key] = preparation.load(ref, split=split, pair_model=pair_model)
        wanted.setdefault(key, set())
        for case in request['cases']:
            wanted[key].add((case['id'], case['binding']))
        reference = request['reference']
        wanted[train_key].add((reference['id'], reference['binding']))
        selected.append(key)
    found = {}
    for key, source in sources.items():
        matches = {}
        for index, entry in enumerate(source.entries):
            identity = (entry['id'], entry['binding'])
            if identity not in wanted[key]:
                continue
            if identity in matches:
                raise ValueError('capacity case identity is ambiguous in the actual source: ' + repr(identity))
            matches[identity] = (index, entry)
        missing = wanted[key] - matches.keys()
        if missing:
            raise ValueError('capacity case identity is absent from the actual source: ' + repr(sorted(missing)))
        found[key] = matches
    audit, outputs = [], []
    for request_index, (original, key) in enumerate(zip(requests, selected, strict=True)):
        output = copy.deepcopy(original)
        selections = [('case', item, key, sources[key]) for item in output['cases']]
        selections.append(('reference', output['reference'], train_key, train))
        for kind, item, source_key, source in selections:
            before = item['index']
            if type(before) is not int or before < 0:
                raise ValueError('capacity case index must be a nonnegative integer')
            after, entry = found[source_key][(item['id'], item['binding'])]
            for metric in ('forward_tokens', 'kv_slots'):
                if metric in item and metric in entry and item[metric] != entry[metric]:
                    raise ValueError('capacity case declared geometry differs from its unchanged source row')
            actual = source.entries[before] if before < len(source.entries) else None
            observed = ({name: actual[name] for name in ('id', 'binding')} if actual is not None else None)
            audit.append(dict(request=request_index, kind=kind, source=str(source.path),
                id=item['id'], binding=item['binding'], before=before, after=after,
                actual_at_declared_index=observed))
            if not relocate and before != after:
                raise ValueError('capacity case index differs from the actual source: ' + repr(audit[-1]))
            if relocate:
                item['index'] = after
        indices = [item['index'] for item in output['cases']]
        if len(set(indices)) != len(indices):
            raise ValueError('capacity request repeats a selected case index')
        outputs.append(output)
    return dict(requests=outputs, checks=audit, source_groups=len(sources),
        case_checks=sum(len(value['cases']) for value in requests), reference_checks=len(requests),
        changed_indices=sum(check['before'] != check['after'] for check in audit),
        full_source_raw_graph_parses=0, numerical_measurements=0)
