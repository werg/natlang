"""Mandatory text-warm-up handoff; availability never grants admission."""
import hashlib,json
from pathlib import Path
import torch
from .output_embedding_projection import sha


def weights_digest(backbone,heads):
    digest=hashlib.sha256()
    for section,values in [('backbone',backbone),('heads',heads)]:
        for name,value in sorted(values.items()):
            tensor=value.detach().cpu().contiguous()
            digest.update(json.dumps([section,name,list(tensor.shape),str(tensor.dtype)]).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def require_text_warmup(path,runtime_report=None):
    path=Path(path)
    state=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    warmup=state.get('warmup') or {}
    if warmup.get('alignment_qualified') is not True:
        raise ValueError('mandatory projection-first/full-stack text warm-up has not qualified')
    report_path=path.parent/'report.json'
    if not report_path.is_file():report_path=Path(warmup.get('report_path',''))
    if not report_path.is_file() or sha(report_path)!=warmup.get('report_sha256'):
        raise ValueError('warm-up qualification report missing or changed')
    report=json.loads(report_path.read_text())
    digest=weights_digest(state.get('backbone_trainables',{}),state['heads'])
    if report.get('qualified') is not True or report.get('weights_digest')!=digest or not all(report.get('updates',{}).get(k) for k in ('backbone','sketch')):
        raise ValueError('warm-up report does not qualify these exact adapted weights')
    if runtime_report is not None:
        runtime=json.loads(Path(runtime_report).read_text())
        if (runtime.get('parent_sha256')!=sha(path) or runtime.get('runtime_qualified') is not True
            or runtime.get('output_reference_qualified') is not True):
            raise ValueError('adapted warm-up weights lack exact output/transport requalification')
    return report
