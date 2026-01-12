============
Installation 
============

The ``winddensity_mbir`` package currently is only available to download and install from source through GitHub.


Downloading and installing from source
-----------------------------------------

1. Download the source code:

  In order to download the python code, move to a directory of your choice and run the following two commands.

    | ``git clone https://github.com/Karl-Weisenburger/WindDensity-MBIR.git``
    | ``cd winddensity_mbir``


2. Create a Virtual Environment:

  It is recommended that you install to a virtual environment.
  If you have Anaconda installed, you can run the following:

    | ``conda create --name winddensity_mbir python=3.11``
    | ``conda activate winddensity_mbir``

  Install the dependencies using:

    ``pip install -r requirements.txt``

  Install the package using:

    ``pip install .``

  or to edit the source code while using the package, install using

    ``pip install -e .``

  Now to use the package, this ``winddensity_mbir`` environment needs to be activated.


3. Install:

You can verify the installation by running ``pip show winddensity_mbir``, which should display a brief summary of the packages installed in the ``winddensity_mbir`` environment.
Now you will be able to use the ``winddensity_mbir`` python commands from any directory by running the python command ``import winddensity_mbir``.

