import json,subprocess,sys
import pytest
from scripts.coreweave.native_remote_budget import bounded_remote


def test_budget_wrapper_keeps_future_import_argv_and_bounds_cpu():
    source="from __future__ import annotations\nimport json,os,sys\nprint(json.dumps({'nice':os.nice(0),'omp':os.environ['OMP_NUM_THREADS'],'mkl':os.environ['MKL_NUM_THREADS'],'openblas':os.environ['OPENBLAS_NUM_THREADS'],'argument':sys.argv[1]}))"
    result=json.loads(subprocess.check_output([sys.executable,'-c',bounded_remote(source),'unchanged'],text=True))
    assert result['nice']>=15 and result['omp']==result['mkl']==result['openblas']=='1' and result['argument']=='unchanged'


def test_budget_wrapper_rejects_invalid_source_before_transport():
    with pytest.raises(SyntaxError):bounded_remote('if :')
