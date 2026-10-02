===============================
Build documentation with Sphinx
===============================

Build HTML locally
------------------

1. Go to the docs folder

	``cd docs``

2. Install sphinx dependencies if they're not already

	``pip install -r requirements.txt``

3. Build HTML files

	``make clean html``

If the build was successful, the HTML files will be in the build/html folder.
Open index.html to review the documentation.

Build HTML in readthedocs
-------------------------

The build is configured by ``.readthedocs.yaml`` in the repository root, which
installs the package with its ``docs`` extra and builds ``docs/source``.

1. Sign in to https://readthedocs.org with your GitHub account.
2. Click "Add project" and import the ``WindDensity-MBIR`` repository.
3. Read the Docs builds the docs on every push to GitHub and publishes them at
   ``https://<project-slug>.readthedocs.io``.
