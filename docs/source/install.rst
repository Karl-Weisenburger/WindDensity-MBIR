============
Installation 
============

The ``winddensity_mbir`` package currently is only available to download and install from source through GitHub.


Downloading and installing from source
-----------------------------------------

1. Download the source code:

  In order to download the python code, move to a directory of your choice and run the following two commands.

    | ``git clone https://github.com/Karl-Weisenburger/WindDensity-MBIR.git``
    | ``cd WindDensity-MBIR``


2. Create a Virtual Environment:

  It is recommended that you install to a virtual environment.
  If you have Anaconda installed, you can run the following:

    | ``conda create --name winddensity_mbir python=3.11``
    | ``conda activate winddensity_mbir``

  Install the package and its pinned dependencies using:

    ``pip install .``

  or to edit the source code while using the package, install using

    ``pip install -e .``

  On a machine with a CUDA 12 GPU (required for the paper's data collection scripts), install the GPU build of JAX instead:

    ``pip install -e ".[cuda12]"``

  Now to use the package, this ``winddensity_mbir`` environment needs to be activated.


3. Verify the installation:

  Install the test dependencies and run the test suite from the repository root:

    | ``pip install -e ".[test]"``
    | ``python -m pytest``

  All tests should pass. To check that JAX can see your GPU, run

    ``python -c "import jax; print(jax.devices())"``

  which should list a ``CudaDevice``. See :doc:`testing` for details.

  On clusters with environment modules (e.g. Purdue RCAC), do not load a ``cuda`` module:
  it can shadow the CUDA libraries installed by pip so that JAX silently falls back to the
  CPU. On Gautschi, run ``module unload cuda`` after ``module load modtree/gpu``.

