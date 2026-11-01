import os,sys
from scripts.neuralese_data.build_local_recovery_plan import ldd_files,mapped_executables


def test_inventories_real_process_and_executable_dependencies():
    assert os.path.realpath(sys.executable) in mapped_executables(os.getpid())
    libraries=ldd_files(sys.executable)
    assert libraries and all(os.path.isabs(path) and os.path.isfile(path) for path in libraries)
