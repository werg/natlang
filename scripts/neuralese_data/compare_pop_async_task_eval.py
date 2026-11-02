#!/usr/bin/env python3
"""Compare two completed, exact-runtime 16-family protected development evaluations."""
import argparse,json
from pathlib import Path
from natlang_neuralese.eval.native_async import compare_transported_development,file_binding
p=argparse.ArgumentParser(description=__doc__)
for name in ('parent-score','parent-score-sha256','parent-request','parent-request-sha256','parent-source-sha256',
             'candidate-score','candidate-score-sha256','candidate-request','candidate-request-sha256','candidate-source-sha256','out'):
    p.add_argument('--'+name,required=True)
a=p.parse_args()
def checked(path,sha):
    ref=file_binding(path)
    if ref['sha256']!=sha:raise ValueError('comparison input differs from its declared SHA256')
    return ref
result=compare_transported_development(checked(a.parent_score,a.parent_score_sha256),checked(a.parent_request,a.parent_request_sha256),
    checked(a.candidate_score,a.candidate_score_sha256),checked(a.candidate_request,a.candidate_request_sha256),
    parent_source_sha256=a.parent_source_sha256,candidate_source_sha256=a.candidate_source_sha256)
from natlang_neuralese.common.jsonio import write_json_durable
out=Path(a.out)
if out.exists():raise ValueError('refusing to replace comparison evidence')
write_json_durable(out,result)
print(json.dumps({'result':result,'receipt':file_binding(out)}),flush=True)
