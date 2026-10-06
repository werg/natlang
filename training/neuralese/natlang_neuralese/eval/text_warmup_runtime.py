"""Requalify the exact adapted text-warm-up output and execution channel."""
import argparse,json
from pathlib import Path
from . import latent_sketch
from ..train.warmup_admission import require_text_warmup


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--heads',type=Path,required=True);p.add_argument('--records',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args(argv)
    if a.device!='cuda':raise ValueError('actual-weight runtime qualification currently requires CUDA')
    require_text_warmup(a.heads)
    latent_sketch.main(['--checkpoint',str(a.heads),'--out',str(a.out),
                        '--retain-trained-heads','--lengths','8','32'])
    path=a.out/'report.json';report=json.loads(path.read_text())
    report['output_reference_qualified']=all(r['reference_next_token_delta']==0 for r in report['rows'])
    path.write_text(json.dumps(report,indent=2)+'\n')
    if not report['output_reference_qualified']:raise SystemExit('adapted full-depth output reference failed')

if __name__=='__main__':main()
