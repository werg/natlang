"""Bound CPU priority and numerical-library thread counts for remote probes."""
from __future__ import annotations


def bounded_remote(source):
    """Keep original Python in a fresh compilation unit, including future imports."""
    compile(source, '<native-remote>', 'exec')
    wrapper=("import os\nos.nice(15)\n"
             "os.environ['OMP_NUM_THREADS']='1'\nos.environ['MKL_NUM_THREADS']='1'\n"
             "os.environ['OPENBLAS_NUM_THREADS']='1'\n"
             "exec(compile("+repr(source)+",'<native-remote>','exec'))\n")
    compile(wrapper,'<bounded-native-remote>','exec')
    return wrapper
