import os,signal,subprocess,sys


def test_diagnostic_signal_dumps_stack_without_stopping_process():
    code='''import time\nfrom natlang_neuralese.train.native_diagnostics import learner_stack_signal\nwith learner_stack_signal():\n print("ready",flush=True)\n time.sleep(10)\n'''
    proc=subprocess.Popen([sys.executable,'-u','-c',code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        assert proc.stdout.readline().strip()=='ready'
        os.kill(proc.pid,signal.SIGQUIT)
        line=proc.stderr.readline()
        assert 'Current thread' in line
        assert proc.poll() is None
    finally:
        proc.terminate();proc.wait(timeout=5)


def test_compiled_caller_preserves_chunk_boundaries_and_gradients():
    import torch
    from natlang_neuralese.model.native_gradient_partitions import GradientPartitions,gradient_partitions,token_local_chunks
    calls=[]
    def operation(x):
        def callback(part):
            calls.append(part.shape[1])
            return part.square()
        with gradient_partitions(GradientPartitions((3,2))):
            return token_local_chunks(x,2,callback)
    compiled=torch.compile(operation,backend='eager')
    x=torch.arange(5.,requires_grad=True).reshape(1,5,1);x.retain_grad()
    y=compiled(x);y.sum().backward()
    assert torch.equal(y,x.square())
    assert torch.equal(x.grad,2*x)
    assert calls==[2,1,2]
