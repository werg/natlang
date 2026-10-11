import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
prot = json.load(open(sys.argv[2]))
keys = sorted({k for r in rows for f in ('source_ids_original', 'source_groups_original', 'source_ids', 'source_groups') for k in (r.get(f) or [])})
keys = sorted(set(keys) | {k.split(':', 1)[-1] for k in keys})
hits = {k: prot['ids'][k] for k in keys if k in prot['ids']}
json.dump({'schema': 'natlang.raw-key-protected-check/1', 'rows': len(rows), 'keys_checked': len(keys), 'protected_id_hits': hits}, open(sys.argv[3], 'w'), indent=1)
print('protected raw-key check', len(keys), 'keys', len(hits), 'hits')
