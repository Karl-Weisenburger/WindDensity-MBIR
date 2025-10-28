.. docs-include-ref

aomodel
=======

This project includes a data-driven algorithm that generates synthetic time-series of images (of arbitrary duration)
by estimating statistical parameters from an input time-series of images.

..
    Include more detailed description here.

Installing
----------
1. *Clone or download the repository:*

    .. code-block::

        git clone git@github.com:cabouman/aomodel

2. Install the conda environment and package

    a. Option 1: Clean install from dev_scripts

        *******You can skip all other steps if you do a clean install.******

        To do a clean install, use the command:

        .. code-block::

            cd dev_scripts
            source clean_install_all.sh

    b. Option 2: Manual install

        1. *Create conda environment:*

            Create a new conda environment named ``aomodel`` using the following commands:

            .. code-block::

                conda create --name aomodel python=3.11
                conda activate aomodel
                pip install -r requirements.txt

            Anytime you want to use this package, this ``aomodel`` environment should be activated with the following:

            .. code-block::

                conda activate aomodel


        2. *Install aomodel package:*

            Navigate to the main directory ``aomodel/`` and run the following:

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


After downloading the data files, run the demo script ``demo_ReVAR.py`` from the parent directory (the aomodel folder
containing this file) with the following command:

    .. code-block::

        python demo/demo_ReVAR.py