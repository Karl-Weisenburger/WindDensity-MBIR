=======
Testing
=======

The test suite lives in ``tests/`` and uses `pytest <https://docs.pytest.org>`_.
Install the test dependencies and run it from the repository root:

    | ``pip install -e ".[test]"``
    | ``python -m pytest``

The default tests run on a CPU in a few minutes and need no data files. They cover:

- **Unit tests** of the package's numerical routines (tip/tilt/piston removal,
  Zernike fitting, optical-path-length sectioning, NRMSE, optical setup,
  atmospheric volume generation and beam field-of-view weights), each checked
  against a known analytic answer.
- **Projector checks** that the MBIRJAX back projector is the adjoint of the
  forward projector, which catches a broken or incompatible MBIRJAX/JAX install.
- **Regression tests** that simulate and reconstruct a small synthetic phantom
  in a scaled-down 7-view, 8-degree geometry and compare the reconstruction
  error to stored reference values.

Full-size checks
----------------

Tests marked ``slow`` regenerate the full-size seed-17 volume used in Fig 6 and
check that it reconstructs with the NRMSE reported for the paper. They need a
CUDA GPU to run in reasonable time and are skipped by ``-m "not slow"``:

    | ``python -m pytest -m slow``      (full-size checks only)
    | ``python -m pytest -m "not slow"`` (everything else)

If a regression or full-size test fails after a dependency change, the
pipeline no longer reproduces the published numbers. The dependency versions
used for the paper are pinned in ``pyproject.toml``.

Continuous integration
----------------------

A GitHub Actions workflow (``.github/workflows/tests.yml``) installs the
package with pip on a fresh Ubuntu machine and runs the non-slow tests on
every push and pull request.
