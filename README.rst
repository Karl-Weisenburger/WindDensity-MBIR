.. docs-include-ref

WindDensity-MBIR
===============================

A Python package for simulating wind tunnel density tomography experiments, processing raw tomography data, and performing tomographic reconstruction using MBIRJAX_.

Approved for public release; distribution is unlimited. Public Affairs release approval # 2025-5579

..
    Include more detailed description here.

Installing
----------
1. *Clone or download the repository:*

    .. code-block::

        git clone git@github.com:Karl-Weisenburger/WindDensity-MBIR

2. Install the conda environment and package

    a. Option 1: Clean install from dev_scripts

        *******You can skip all other steps if you do a clean install.******

        To do a clean install, use the command:

        .. code-block::

            cd dev_scripts
            source clean_install_all.sh

    b. Option 2: Manual install

        1. *Create conda environment:*

            Create a new conda environment named ``winddensity_mbir`` using the following commands:

            .. code-block::

                conda create --name winddensity_mbir python=3.11
                conda activate winddensity_mbir
                pip install -r requirements.txt

            Anytime you want to use this package, this ``winddensity_mbir`` environment should be activated with the following:

            .. code-block::

                conda activate winddensity_mbir


        2. *Install wind_tomo package:*

            Navigate to the main directory ``winddensity_mbir/`` and run the following:

            .. code-block::

                pip install .

            To allow editing of the package source while using the package, use

            .. code-block::

                pip install -e .


Running Demo(s)
---------------

Run the simulation + reconstruction demo from the parent directory
(the ``winddensity_mbir`` folder containing this file):

    .. code-block::

        python demo/demo_simulation_and_tomography.py

The demo generates (or loads, on subsequent runs) a cached atmospheric
phase volume at ``experiments/shared_data/vol_seed17.npy``, simulates
OPD_TT measurements, and runs an MBIR reconstruction. Output figures
are written to ``demo/output/``. The cached volume is the same one
used by the Fig 6 / Fig 12 / Fig 16 scripts, so running the demo first
pre-populates that cache (and vice versa).

There is also an experimental-data processing demo:

    .. code-block::

        python demo/demo_processing_experimental_data.py

Both demos generate the atmospheric volume on the fly using JAX, so a
CUDA-enabled GPU is strongly recommended.

Reproducing the Paper Figures
-----------------------------

The ``experiments/`` directory contains everything needed to regenerate
the figures and tables from the WindDensity-MBIR paper. The workflow is
split into two stages:

1. **Data collection** — runs reconstructions on many ground-truth
   volumes and saves per-figure ``.npz`` files under each experiment's
   ``data/`` subfolder.
2. **Visualization** — reads the cached ``.npz`` files and writes the
   figures to each experiment's ``figures/`` subfolder.

.. warning::

    Data collection is **GPU-only**. End-to-end it takes roughly
    **1–3 days on a modern CUDA GPU** (e.g. A100, 3090). Running on CPU
    is not supported — it would take weeks to months and was never
    validated. Each data collection script checks for a visible CUDA
    device at startup and aborts if none is found. To override at your
    own risk, set the environment variable ``WINDDENSITY_ALLOW_CPU=1``.

Approximate per-script GPU runtimes (measured at ~13.6 s per MBIR
reconstruction — see Table 1 in ``fig7_table1/``):

==================================  =====================
Script                              Approx. GPU runtime
==================================  =====================
``table2_data_collection``          ~1.1 h
``fig7_data_collection``            ~15.1 h
``fig9_data_collection``            ~22.7 h
``fig10_11_data_collection``        ~7.6 h
``fig13_14_15_17_data_collection``  ~7.6 h
**Total**                           **~54 h (~2.3 days)**
==================================  =====================

Step 1 — collect data
~~~~~~~~~~~~~~~~~~~~~

From the repository root, with the ``winddensity_mbir`` conda
environment activated:

    .. code-block::

        bash experiments/run_all_data_collection.sh

This runs all five data collection scripts sequentially. Each script
prints a runtime-estimate banner at startup and pauses briefly before
launching so that you can cancel with Ctrl-C if the GPU is unavailable.
Individual scripts can also be run directly, e.g.:

    .. code-block::

        python experiments/fig7_table1/fig7_data_collection.py

Intermediate results are saved incrementally to the per-experiment
``data/`` folders so that re-running a script resumes from where it
left off.

Step 2 — generate figures
~~~~~~~~~~~~~~~~~~~~~~~~~

Once data collection has completed, regenerate every paper figure with:

    .. code-block::

        bash experiments/run_all_visualizations.sh

This runs all visualization scripts non-interactively (``MPLBACKEND=Agg``)
so no plot windows open. Figures are written as ``.pdf`` and ``.png``
files under each experiment's ``figures/`` subfolder, e.g.:

    .. code-block::

        experiments/fig6_table2/figures/fig6_mbir_vs_fbp.pdf
        experiments/fig7_table1/figures/fig7_nrmse_vs_angle.pdf
        experiments/fig12/figures/fig12_geometry_comparison.pdf
        ...

Tables 1 and 2 are printed to stdout (and captured in the corresponding
logs under ``experiments/logs/``) rather than saved to files:

- Table 1 (reconstruction performance) —
  ``experiments/fig7_table1/table1_visualization.py``
- Table 2 (FBP vs MBIR NRMSE) —
  ``experiments/fig6_table2/table2_visualization.py``

Individual visualization scripts can also be run directly, for example:

    .. code-block::

        python experiments/fig7_table1/fig7_visualization.py

Public Release Approval
-----------------------
Approved for public release; distribution is unlimited. Public Affairs release approval # 2025-5579


.. _MBIRJAX: https://mbirjax.readthedocs.io/en/latest/