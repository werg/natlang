import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('skill_queue', Path(__file__).resolve().parents[1] / 'scripts/run_skill_authoring_queue.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)

FAKE_NODE = '''#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
if '--version' in sys.argv: print('v22.22.0');sys.exit(0)
if '--input-type=module' in sys.argv: sys.exit(0)
a=dict(zip(sys.argv[2::2],sys.argv[3::2]));row=json.loads(Path(a['--episodes']).read_text());out=Path(a['--out'])
if row['family']=='crash':sys.exit(4)
p=out/hashlib.sha256(row['id'].encode()).hexdigest()[:20];p.mkdir()
transient=row['family']=='transient' and out.name=='attempt-001'
result={'episode':row['id'],'disposition':'failed' if transient or row['family']=='semantic' else 'evaluated', 'positive':row['family']=='positive'}
if transient:result['error']='HTTP 429 rate limit'
if row['family']=='semantic':result['error']='invalid skill description'
(p/'result.json').write_text(json.dumps(result))
'''

class QueueTests(unittest.TestCase):
    def fixture(self, families):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        runtime = root/'runtime';runtime.mkdir()
        (runtime/'frozen-runtime.json').write_text(json.dumps({'schema':'natlang.skill-authoring-runtime/1'}))
        node = root/'node';node.write_text(FAKE_NODE);node.chmod(0o755)
        packet = root/'episodes.jsonl'
        packet.write_text(''.join(json.dumps({'id':str(n),'family':family,'split':'train'})+'\n' for n,family in enumerate(families)))
        return argparse.Namespace(runtime=runtime,node=node,episodes=packet,out=root/'out',endpoint='fixture',model='fixture',workers=2,experiments=2,ablations=0,max_attempts=3,backoff_seconds=.001)

    def test_transient_retry_and_finite_negative_accounting(self):
        args=self.fixture(['positive','transient','semantic','crash'])
        summary=queue.run_queue(args)
        self.assertEqual(summary['terminal'],4)
        self.assertEqual(summary['positive_candidates'],1)
        states=[json.loads(p.read_text()) for p in args.out.glob('tasks/*/state.json')]
        by_id={s['episode']:s for s in states}
        self.assertEqual(len(by_id['1']['attempts']),2)
        self.assertEqual(len(by_id['2']['attempts']),1)
        self.assertEqual(by_id['3']['disposition'],'collector_failed_without_artifact')
        before={str(p):p.read_bytes() for p in args.out.glob('tasks/*/attempt-*/*/result.json')}
        self.assertEqual(queue.run_queue(args)['terminal'],4)
        self.assertEqual(before,{str(p):p.read_bytes() for p in args.out.glob('tasks/*/attempt-*/*/result.json')})

    def test_resume_refuses_identity_change(self):
        args=self.fixture(['positive']);queue.run_queue(args);args.model='changed'
        with self.assertRaisesRegex(ValueError,'identity changed'):queue.run_queue(args)

    def test_retry_is_transport_only(self):
        self.assertFalse(queue.retryable({'disposition':'evaluated','error':'429'}))
        self.assertFalse(queue.retryable({'disposition':'failed','error':'compile error'}))
        self.assertTrue(queue.retryable({'disposition':'failed','error':'fetch failed'}))

    def test_missing_artifacts_distinguish_stops_signals_and_crashes(self):
        self.assertEqual(queue.missing_result_disposition(1, True), 'interrupted_attempt_requires_review')
        self.assertEqual(queue.missing_result_disposition(-15, True), 'interrupted_attempt_requires_review')
        self.assertEqual(queue.missing_result_disposition(-9, False), 'collector_terminated_by_signal')
        self.assertEqual(queue.missing_result_disposition(4, False), 'collector_failed_without_artifact')

if __name__=='__main__':unittest.main()
