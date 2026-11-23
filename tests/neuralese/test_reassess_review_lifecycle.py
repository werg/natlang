"""A lifecycle correction needs an independent content audit bound to exact evidence."""
import copy
import hashlib
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/neuralese_data'))
import reassess_review_lifecycle as correction

def write(root,name,value):
 p=root/name;p.write_text(json.dumps(value));return p

def fixture(root):
 docs=[{'path':'guide.md','text':'Retain no names.'}]
 expected={'goal':'Prepare a private guide draft','facts':[{'id':'privacy','text':'Retain no names.'}],'documentRoles':['Privacy guide']}
 before={'committed':{'kind':'event','eventId':'baseline'},'workingHead':{'kind':'event','eventId':'baseline'},'clean':True,'mainEventId':'main'}
 current={**before,'clean':False,'workingHead':{'kind':'application','applicationId':'draft'}}
 review_input={'actual':{'documents':docs,'committedEvent':before['committed'],'state':current['workingHead']},'expected':expected,'rubric':'Check exact supplied work'}
 packet={'test':{'name':'draft'},'result':{'passed':True},'execution':{'diagnostics':{'natlangTeacher':{'rootOutcome':'done','report':{'final':{'outcome':'draft'}}},'openGoalVerification':{'goal':expected['goal'],'checks':{'native_state':True},'preparedContentProject':{'before':before,'actual':current,'expectedOutcome':'draft','journal':[],'artifact':{'passed':True,'semanticReviewInput':review_input}}}}}}
 review={'schema':'natlang.live-workflow-review/1','input':review_input,'verdict':{'accepted':False,'verdict':'The content covers the guide. Baseline handle mistaken for a new commit.'},'exchange':{'request':{},'reply':{}}}
 packet_path=write(root,'packet.json',packet);review_path=write(root,'review.json',review)
 audit={'schema':'natlang.source-bound-content-audit/1','auditor':'Independent content audit','accepted':True,'rationale':'Actual privacy guide preserves the supplied condition.',
  'original_packet_sha256':correction.digest(packet_path),'original_review_sha256':correction.digest(review_path),'document_sha256':hashlib.sha256(json.dumps(docs,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
  'original_content_positive_excerpt':'The content covers the guide.',
  'fact_coverage':[{'fact_id':'privacy','source_fact':'Retain no names.','path':'guide.md','quote':'Retain no names.','rationale':'Exact supported restriction.'}],
  'role_coverage':[{'role':'Privacy guide','path':'guide.md','rationale':'Supplied privacy instruction is directly usable.'}]}
 audit_path=write(root,'audit.json',audit);catalog=write(root,'catalog.jsonl',{'scenario':'draft','taskId':'task','split':'train'})
 return packet_path,review_path,audit_path,catalog,packet,review,audit

@pytest.mark.parametrize('mutation',[None,'new-commit','accepted-review','unsupported-content','missing-role','stale-audit'])
def test_correction_preserves_original_review_and_requires_exact_native_and_semantic_evidence(tmp_path,mutation):
 packet_path,review_path,audit_path,catalog,packet,review,audit=fixture(tmp_path)
 if mutation=='new-commit':packet['execution']['diagnostics']['openGoalVerification']['preparedContentProject']['actual']['committed']={'kind':'event','eventId':'new'}
 if mutation=='accepted-review':review['verdict']['accepted']=True
 if mutation=='unsupported-content':audit['fact_coverage'][0]['quote']='Invented instruction'
 if mutation=='missing-role':audit['role_coverage']=[]
 if mutation=='stale-audit':audit['document_sha256']='0'*64
 write(tmp_path,'packet.json',packet);write(tmp_path,'review.json',review);write(tmp_path,'audit.json',audit)
 out=tmp_path/'corrected'
 if mutation:
  with pytest.raises(ValueError):correction.assess(packet_path,review_path,audit_path,catalog,out)
 else:
  result=correction.assess(packet_path,review_path,audit_path,catalog,out)
  assert result['schema']=='natlang.live-reassessment/1'
  assert json.loads((out/'original-rejected-review.json').read_text())['verdict']['accepted'] is False
  assert json.loads((out/'completed-outcome.json').read_text())['passed'] is True
