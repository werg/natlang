"""Merge reviewed train turns without losing admission metadata or token fidelity.

If any proposed chain fails exact token-history/length checks, retain its original
rows and report why. Protected held-out rows are passed through unchanged.
"""
from scripts.merge_sft_chains import merge
from scripts.audit_training_corpus import assess


def prepare_chains(rows, tokenizer, end_token, max_len):
    by_id={r['id']:r for r in rows}
    if len(by_id)!=len(rows):raise ValueError('duplicate input row IDs')
    held=[r for r in rows if r.get('split')!='train']
    train=[r for r in rows if r.get('split')=='train']
    output=list(held);ledger=[]
    for chain in merge(train):
        # Adjacent context pieces are one served prompt, not separate BPE prefixes.
        segments=[]
        for text,trained in chain['segments']:
            if segments and not trained and not segments[-1][1]:segments[-1][0]+=text
            else:segments.append([text,trained])
        chain['segments']=segments
        result=assess(chain,tokenizer,end_token,max_len)
        if result['record'] is None:
            output.extend(by_id[id] for id in chain['turns'])
            ledger.append({'turns':chain['turns'],'disposition':'original_turns',
                           'reason':result['summary']['reason']})
        else:
            output.append(result['record'])
            ledger.append({'turns':chain['turns'],'disposition':'merged_chain',
                           'actions':len(chain['turns']),'token_counts':result['record']['token_counts']})
    return output,ledger
