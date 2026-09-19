"""Run independent cases concurrently; each worker must own its decoder and runtime."""
from concurrent.futures import ThreadPoolExecutor, as_completed


def completed_cases(run, jobs, workers):
    if workers < 1:
        raise ValueError('workers must be positive')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, job): i for i, job in enumerate(jobs)}
        for future in as_completed(futures):
            yield futures[future], future.result()


def total_usage(rows):
    return {key: sum(row['usage'][key] for row in rows)
            for key in ('turns', 'completion_tokens', 'seconds')}
