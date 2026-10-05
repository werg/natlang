import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('watch_training_job', Path(__file__).resolve().parents[1] / 'scripts/watch_training_job.py')
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


def test_exit_wakes_wait_without_waiting_full_cycle():
    jobs = iter([{'id': 'original', 'state': {'Running': True}},
                 {'id': 'original', 'state': {'Running': False, 'Status': 'exited', 'ExitCode': 1}}])
    sleeps = []
    result, code = watch.wait_for_job('job', 3000, 60, inspect=lambda _: next(jobs),
                                     now=lambda: 0, sleep=sleeps.append)
    assert sleeps == [60]
    assert result['event'] == 'job_exited' and code == 2


def test_cycle_due_and_container_replacement_are_distinct():
    job = {'id': 'original', 'state': {'Running': True}}
    times = iter([0, 0, 50])
    sleeps = []
    result, code = watch.wait_for_job('job', 50, 60, inspect=lambda _: job,
                                     now=lambda: next(times), sleep=sleeps.append)
    assert sleeps == [50]
    assert result['event'] == 'monitoring_cycle_due' and code == 0
    jobs = iter([job, {'id': 'replacement', 'state': {'Running': True}}])
    result, code = watch.wait_for_job('job', 3000, 60, inspect=lambda _: next(jobs),
                                     now=lambda: 0, sleep=lambda _: None)
    assert result['event'] == 'job_replaced' and code == 3
