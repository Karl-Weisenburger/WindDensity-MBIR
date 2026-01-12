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

Before running any of the demo files, download the data files and pre-trained models from the data depot using the
following:

    .. code-block::

        cd demo
        source get_demo_data_server.sh
        cd ..


After downloading the data files, run the demo script ``demo_simulation_and_tomography.py`` from the parent directory (the winddensity_mbir folder
containing this file) with the following command:

    .. code-block::

        python demo/demo_simulation_and_tomography.py

Public Release Approval
-----------------------
Approved for public release; distribution is unlimited. Public Affairs release approval # 2025-5579


.. _MBIRJAX: https://mbirjax.readthedocs.io/en/latest/