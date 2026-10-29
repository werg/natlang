"""Policy replay retains SFT QAT but uses the captured unquantized actor."""
from types import SimpleNamespace
from natlang_neuralese.train.native_state_sft import policy_qat_update,training_precision_pass


class QuantizationFixture:
    def __init__(self):
        self.calibration={'step':7,'batches':2,'source':{'cursor':9}}
        self._window={'start':8};self._observed_batches=1
        self.inputs={'x':SimpleNamespace(static=True,amax=1.)}
        self.points={'nvfp4':{'modules':{'x':{'activations':{'format':'nvfp4'}}}}}
        self.cancelled=0;self.observed=0
    @property
    def calibration_window(self):return self._window
    def state_dict(self):return {'window':{'start':self._window['start'],'batches':self._observed_batches,'observed':{}}}
    def cancel_calibration(self):self.cancelled+=1;self._window=None;self._observed_batches=0
    def calibration_config(self):return {'batches':2,'refresh':100}
    def calibration_due(self,step):return False
    def begin_calibration(self,step):self._window={'start':step};self._observed_batches=0
    def observe_calibration(self,forward,batch):self.observed+=1;self._observed_batches+=1;forward(batch)
    def plan(self,step):return [('nvfp4',1.,1.)]
    def commit_calibration(self,step,source):self.calibration={'step':step,'batches':self._observed_batches,'source':source};self._window=None
    def check_calibrated(self,name):assert self.calibration['step']==20


def test_policy_is_bf16_and_preserves_qat_until_real_policy_update_then_refreshes():
    q=QuantizationFixture();options={'quantization':{'sha256':'a'*64},'quantization_history':[]}
    prior=dict(q.calibration)
    assert training_precision_pass(q,policy_operation=True,checkpoint_options=options,global_step=19,batch=[1],cursor=1,forward=lambda x:None) is None
    assert q.calibration==prior and q.observed==0 and 'policy_qat_refresh_required' not in options
    policy_qat_update(options,q,19);policy_qat_update(options,q,20)
    saved=options['policy_qat_refresh_required'];assert saved['policy_precision']=='bf16' and saved['first_policy_step']==19 and saved['last_policy_step']==20
    precision=training_precision_pass(q,policy_operation=False,checkpoint_options=options,global_step=20,batch=[1],cursor=1,forward=lambda x:None)
    assert precision[1:]==('nvfp4',1.,1.) and q.cancelled==1 and q.observed==1
    assert q.calibration['step']==20 and 'policy_qat_refresh_required' not in options
    assert options['quantization_history'][-1]['obligation']['calibration_before']==prior
    assert options['quantization_history'][-1]['obligation']['retired_partial_calibration']['start']==8


def test_parametric_handoff_requires_exact_drained_checkpoint_and_source(tmp_path):
    import hashlib,json,pytest,torch
    from natlang_neuralese.rl.jolt_handoff import checked_parent
    path=tmp_path/'checkpoint.pt';source='a'*64
    torch.save({'training':{'global_step':14001,'source_receipts':[{'sha256':source}],
        'optimizer':{'state':1},'rng':{'state':1},'sft_progress':{'unfinished':1},'exposure_counts':{}}},path)
    pin={'schema':'natlang.lfm3-final-standalone-drain/1','phase':'pinned','ready':True,'immutable_pin_verified':True,
        'full_state_verified':True,'optimizer_verified':True,'rng_verified':True,'sandbox_id':'peer','boot_id':'boot',
        'train_source_sha256':source,'step':14001,'global_step':14001,'path':str(path),'bytes':path.stat().st_size,
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    receipt=tmp_path/'pin.json';receipt.write_text(json.dumps(pin))
    assert checked_parent(receipt,sandbox_id='peer',boot_id='boot',min_step=14000,source_sha256=source)[0]==path
    for changes in ({'step':13999},{'sandbox_id':'other'},{'rng_verified':False},{'train_source_sha256':'b'*64}):
        bad={**pin,**changes};receipt.write_text(json.dumps(bad))
        with pytest.raises(ValueError):checked_parent(receipt,sandbox_id='peer',boot_id='boot',min_step=14000,source_sha256=source)


def test_fresh_parent_rejects_inherited_updates_or_wrong_bytes(tmp_path):
    import hashlib,pytest,torch
    from natlang_neuralese.rl.jolt_handoff import checked_fresh_parent
    path=tmp_path/'fresh.pt';source='a'*64
    init={'schema':'natlang.fresh-native-initialization/1','parent_checkpoint':None,'optimizer_updates':0,'earned_exposure':0}
    training={'step':0,'global_step':0,'exposure_counts':{'row':0},'rng':{'a':1},'sft_progress':{'sft':1},
        'optimizer':{'state':{},'param_groups':[{}]},'options':{'fresh_initialization':init},
        'source_receipts':[{'sha256':source}]}
    def save():
        torch.save({'training':training},path)
        return hashlib.sha256(path.read_bytes()).hexdigest()
    sha=save();assert checked_fresh_parent(path,expected_sha256=sha,source_sha256=source)==path
    training['optimizer']['state']={'1':{'step':1}};sha=save()
    with pytest.raises(ValueError):checked_fresh_parent(path,expected_sha256=sha,source_sha256=source)
