"""Architecture/ABI pinning is independent of live CUDA availability."""
import sys
from types import SimpleNamespace

import pytest
from natlang_neuralese.model import lfm2_port as lfm


@pytest.mark.parametrize('capability', [(8, 9), (9, 0)])
def test_exact_architecture_build_loads_and_reports_provenance(monkeypatch, capability):
    build = lfm.CONV_KERNEL_BUILDS[capability]
    kernel = SimpleNamespace(__version__='1.7.0', causal_conv1d_fn=lambda *args: None)
    monkeypatch.setenv('NATLANG_NEURALESE_KERNELS', '1')
    monkeypatch.setattr(lfm.torch, '__version__', lfm.CONV_KERNEL_TORCH_VERSION)
    monkeypatch.setattr(lfm.torch.version, 'cuda', lfm.CONV_KERNEL_CUDA_VERSION)
    monkeypatch.setattr(lfm.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(lfm.torch.cuda, 'get_device_capability', lambda: capability)
    monkeypatch.setattr(lfm.importlib.metadata, 'version', lambda package: build['package_version'])
    monkeypatch.setitem(sys.modules, 'causal_conv1d', kernel)
    assert lfm.load_conv_kernel() is kernel
    provenance = lfm.conv_kernel_provenance(kernel)
    assert provenance['device_capability'] == list(capability)
    assert provenance['wheel_sha256'] == build['wheel_sha256']
    assert provenance['build_manifest_sha256'] == build['build_manifest_sha256']
    # Another architecture's wheel must never cross the selection guard.
    other = lfm.CONV_KERNEL_BUILDS[(9, 0) if capability == (8, 9) else (8, 9)]
    monkeypatch.setattr(lfm.importlib.metadata, 'version', lambda package: other['package_version'])
    assert lfm.load_conv_kernel() is None
    monkeypatch.setattr(lfm.importlib.metadata, 'version', lambda package: build['package_version'])
    monkeypatch.setattr(lfm.torch.cuda, 'get_device_capability', lambda: (12, 1))
    assert lfm.load_conv_kernel() is None
    monkeypatch.setattr(lfm.torch.cuda, 'get_device_capability', lambda: capability)
    monkeypatch.setattr(lfm.torch.version, 'cuda', '13.3')
    assert lfm.load_conv_kernel() is None
    monkeypatch.setattr(lfm.torch.version, 'cuda', lfm.CONV_KERNEL_CUDA_VERSION)
    monkeypatch.setenv('NATLANG_NEURALESE_KERNELS', '0')
    assert lfm.load_conv_kernel() is None
