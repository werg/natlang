"""Recorded generation turns of a collection run, for replaying through a SGLang front against a mock SGLang."""
import glob, json, sys, hashlib
KEY = 64
run, out, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
episodes = []
for path in sorted(glob.glob(f'{run}/groups/*/row-*/sample-*/results.jsonl')):
    for line in open(path):
        r = json.loads(line)
        turns = []
        for t in r.get('trajectory') or []:
            mr = t.get('model_response') or {}
            s = mr.get('rl_sampling') or {}
            prompt = s.get('prompt') or []
            if not s.get('samples') or mr.get('native_state_imports') or s.get('writes') or len(prompt) != 1:
                continue
            ids = prompt[0].get('token_ids') or []
            if len(ids) < KEY:
                continue
            turns.append({'messages': t['context'], 'tools': t.get('tools_offered'),
                          'key': ids[-KEY:],
                          'tokens': [x['token_id'] for x in s['samples']],
                          'logps': [x['logp'] for x in s['samples']]})
        if turns:
            episodes.append(turns)
        if len(episodes) >= limit:
            break
    if len(episodes) >= limit:
        break
json.dump(episodes, open(out, 'w'))
print(len(episodes), 'episodes', sum(map(len, episodes)), 'turns', sum(len(t['tokens']) for e in episodes for t in e), 'tokens')
