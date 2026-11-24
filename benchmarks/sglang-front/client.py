"""Streams recorded chat turns through a front from P processes x T threads for D seconds; prints tokens/s."""
import json, sys, time, threading, http.client, multiprocessing as mp, random
def worker(args):
    episodes, port, threads, seconds, offset, q = args
    stop = time.monotonic() + seconds
    out = {'turns': 0, 'tokens': 0, 'errors': 0, 'latency': [], 'err': []}
    lock = threading.Lock()
    def run(i):
        rng = random.Random(offset + i)
        while time.monotonic() < stop:
            for turn in episodes[(offset + i + rng.randrange(len(episodes))) % len(episodes)]:
                if time.monotonic() >= stop:
                    return
                body = {'messages': turn['messages'], 'tools': turn['tools'], 'temperature': 1.0, 'top_p': 0.95,
                        'stream': True, 'max_tokens': 16384, 'seed': rng.randrange(1 << 30),
                        'x_natlang_rl_capture': True, 'x_natlang_rl_top_k': 0,
                        'x_natlang_native_protocol_recovery': True}
                t0 = time.monotonic()
                try:
                    c = http.client.HTTPConnection('127.0.0.1', port, timeout=max(1.0, stop + 30 - time.monotonic()))
                    c.request('POST', '/v1/chat/completions', json.dumps(body), {'content-type': 'application/json'})
                    r = c.getresponse()
                    tokens, err = None, None
                    for line in r:
                        if line.startswith(b'data: {'):
                            v = json.loads(line[6:])
                            if 'error' in v:
                                err = v['error']
                            elif v.get('usage'):
                                tokens = v['usage'].get('completion_tokens')
                    c.close()
                except Exception as e:
                    err = repr(e)
                with lock:
                    if tokens is None or err:
                        out['errors'] += 1; out['err'].append(str(err)[:200])
                    else:
                        out['turns'] += 1; out['tokens'] += tokens; out['latency'].append(time.monotonic() - t0)
    ts = [threading.Thread(target=run, args=(i,)) for i in range(threads)]
    for t in ts: t.start()
    for t in ts: t.join()
    q.put(out)
if __name__ == '__main__':
    episodes = json.load(open(sys.argv[1])); port = int(sys.argv[2]); procs, threads, seconds = map(int, sys.argv[3:6])
    q = mp.Queue()
    ps = [mp.Process(target=worker, args=((episodes, port, threads, seconds, p * 1000, q),)) for p in range(procs)]
    t0 = time.monotonic()
    for p in ps: p.start()
    res = [q.get() for _ in ps]
    for p in ps: p.join()
    dt = time.monotonic() - t0
    turns = sum(r['turns'] for r in res); tokens = sum(r['tokens'] for r in res); errors = sum(r['errors'] for r in res)
    lat = sorted(x for r in res for x in r['latency'])
    print(json.dumps({'streams': procs * threads, 'seconds': round(dt, 1), 'turns': turns, 'tokens': tokens,
                      'tok_s': round(tokens / dt, 1), 'turns_h': round(turns / dt * 3600), 'errors': errors,
                      'err_sample': [e for r in res for e in r['err']][:3],
                      'latency_p50': lat[len(lat) // 2] if lat else None}))
