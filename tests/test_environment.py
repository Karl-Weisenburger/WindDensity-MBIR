"""Checks that the installed JAX can use the GPU when one is present."""
import os
import shutil
import subprocess

import jax
import pytest


def _nvidia_gpu_present():
    if shutil.which('nvidia-smi') is None:
        return False
    out = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
    return out.returncode == 0 and 'GPU' in out.stdout


@pytest.mark.skipif(os.environ.get('JAX_PLATFORMS', '') == 'cpu', reason='JAX restricted to CPU')
@pytest.mark.skipif(not _nvidia_gpu_present(), reason='no NVIDIA GPU on this machine')
def test_jax_sees_the_gpu():
    platforms = {d.platform for d in jax.devices()}
    assert 'gpu' in platforms, (
        'An NVIDIA GPU is present but JAX is running on CPU. Install the GPU build with '
        'pip install -e ".[cuda12]". On clusters, a loaded CUDA module can shadow the CUDA '
        'libraries installed by pip; try `module unload cuda`.'
    )
