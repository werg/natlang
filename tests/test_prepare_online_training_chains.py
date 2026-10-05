from scripts.prepare_online_training_chains import prepare_chains


class Tokenizer:
    def __call__(self,text,**kwargs):return {'input_ids':list(text.encode())}


def row(id,prompt,split='train'):
    return {'id':id,'teacher_trajectory_id':'a','program_id':'p','prompt':prompt,'completion':'ok!',
            'split':split,'training_admission':{'approved':True}}


def test_protected_rows_unchanged_and_train_targets_accounted_for():
    held=row('held','held','test');a=row('a','hi');b=row('b','hiok!next')
    output,ledger=prepare_chains([held,a,b],Tokenizer(),'!',100)
    assert output[0] is held and len(output)==2
    assert output[1]['turns']==['a','b'] and ledger[0]['actions']==2


def test_overlength_merge_retains_original_samples():
    a=row('a','hi');b=row('b','hiok!next')
    output,ledger=prepare_chains([a,b],Tokenizer(),'!',10)
    assert output==[a,b] and ledger[0]['reason']=='over_token_budget'
