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
