"""Admission custody for CE scales derived from remaining SFT obligations."""
import hashlib,json

SCHEMA='natlang.quota-aware-ce-basis/1'

def frame_digest(frame):
    rows=frame.get('source_row_bindings') or {};order=frame.get('source_row_order') or []
    quotas=frame.get('required_visits') or {};credits=frame.get('exposure_counts') or {}
    if (not isinstance(rows,dict) or not isinstance(order,list) or len(order)!=len(rows) or len(set(order))!=len(order)
            or set(order)!=set(rows) or not set(rows)<=set(quotas) or not set(credits)<=set(quotas)):
        raise ValueError('quota-aware CE requires exact saved row, quota and credit inventories')
    # Rows removed from the active source keep their saved quota and earned credit (historical exposure); they are
    # bound after the active rows so frames without retired rows keep their original digest.
    retired=sorted(set(quotas)-set(rows))
    digest=hashlib.sha256()
    for ident in (*order,*retired):
        quota=quotas[ident];credit=credits.get(ident,0)
        if type(quota) is not int or type(credit) is not int or quota<0 or credit<0 or credit>quota:
            raise ValueError('quota-aware CE encountered invalid saved obligations')
        record=[ident,rows[ident],quota,credit] if ident in rows else ['retired',ident,quota,credit]
        digest.update(json.dumps(record,ensure_ascii=False,separators=(',',':')).encode('utf8'));digest.update(b'\n')
    return digest.hexdigest()

def make_basis(frame,*,stage_id,source_sha256,candidate_sha256,epochs,communication_min_visits):
    receipts=frame.get('source_receipts') or []
    if not receipts or receipts[0].get('sha256')!=source_sha256:raise ValueError('quota-aware CE parent source differs from saved SFT frame')
    return {'schema':SCHEMA,'stage_id':stage_id,'parent_source_sha256':source_sha256,'candidate_source_sha256':candidate_sha256,
            'epochs':epochs,'communication_min_visits':communication_min_visits,'parent_frame_sha256':frame_digest(frame)}

def validate_admission(policy,frame,*,stage_id,candidate_sha256,epochs,communication_min_visits,previous_options,policy_binding):
    basis=(policy or {}).get('quota_aware_basis')
    previous_options=previous_options or {}
    previous_binding=previous_options.get('ce_supervision_policy_binding') or {}
    inherited_visits=(previous_options.get('communication_quota_policy') or {}).get('visits',1)
    effective_visits=communication_min_visits if communication_min_visits is not None else inherited_visits
    if basis is None:
        if (previous_binding and previous_binding.get('sha256')!=policy_binding.get('sha256')
                and type(effective_visits) is int and effective_visits>1):
            raise ValueError('CE continuation with repeated communication quotas requires a measured quota-aware basis')
        return None
    fields={'schema','stage_id','parent_source_sha256','candidate_source_sha256','epochs','communication_min_visits','parent_frame_sha256'}
    if not isinstance(basis,dict) or set(basis)!=fields or basis.get('schema')!=SCHEMA or basis['stage_id']!=stage_id or basis['candidate_source_sha256']!=candidate_sha256 or basis['epochs']!=epochs or basis['communication_min_visits']!=communication_min_visits:
        raise ValueError('quota-aware CE basis does not match requested source or quota settings')
    history=previous_options.get('quota_aware_ce_history') or []
    record={'policy_sha256':policy_binding['sha256'],'basis':basis}
    if previous_binding.get('sha256')==policy_binding['sha256']:
        if record not in history:raise ValueError('quota-aware CE resume lacks recorded initial admission')
    else:
        if not isinstance(frame,dict):raise ValueError('quota-aware CE transition needs its saved SFT parent frame')
        expected=make_basis(frame,stage_id=stage_id,source_sha256=basis['parent_source_sha256'],candidate_sha256=candidate_sha256,epochs=epochs,communication_min_visits=communication_min_visits)
        if expected!=basis:raise ValueError('quota-aware CE parent quotas or exposure changed; recensus before training')
    return record
