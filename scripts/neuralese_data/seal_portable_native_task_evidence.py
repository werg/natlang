#!/usr/bin/env python3
"""Seal actual evaluator output on its original host before copying it."""
import argparse, json
from natlang_neuralese.eval.portable_task_evidence import seal, digest
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation-dir',required=True);p.add_argument('--spec',required=True)
p.add_argument('--packet',required=True);p.add_argument('--out',required=True)
p.add_argument('--source-root',required=True);p.add_argument('--container-root',required=True)
p.add_argument('--transport-receipt',required=True);p.add_argument('--view-receipt',required=True)
p.add_argument('--runtime-closure',required=True)
a=p.parse_args();path=seal(a.evaluation_dir,a.spec,a.packet,a.out,source_root=a.source_root,container_root=a.container_root,
    transport_path=a.transport_receipt,view_receipt_path=a.view_receipt,runtime_closure_path=a.runtime_closure)
print(json.dumps({'path':str(path),'sha256':digest(path.read_bytes())}))
