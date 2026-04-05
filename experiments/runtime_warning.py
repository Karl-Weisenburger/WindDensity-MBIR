"""
Shared runtime warning helper for data collection scripts.

Each data-collection entry point should call ``warn_and_confirm`` early.
It prints a prominent warning about expected runtime and aborts if no
CUDA-capable GPU is visible to JAX (unless the user explicitly overrides
with WINDDENSITY_ALLOW_CPU=1).
"""
from __future__ import annotations

import os
import sys
import time


# Approximate per-script GPU runtimes (hours). Computed from measured
# MBIR reconstruction performance (mean ≈ 13.6 s per reconstruction on a
# modern CUDA GPU, from the Table 1 performance data in fig7_table1) and
# the number of reconstructions each script performs:
#
#   table2         : 100 vols × 3 geometries × 1 MBIR    = 300   recons  →  ~1.1 h
#   fig7_table1    : 100 vols × 8 extents × 5 view counts = 4000 recons  → ~15.1 h
#   fig9           : 3000 vols × 2 geometries            = 6000  recons  → ~22.7 h
#   fig10_11       : 1000 vols × 2 geometries            = 2000  recons  →  ~7.6 h
#   fig13_14_15_17 : 1000 vols × 2 measurement types     = 2000  recons  →  ~7.6 h
#
# Actual wall-clock time varies with GPU model, driver, and disk I/O;
# these values are meant to give the user a realistic sense of scale.
SCRIPT_ETA_HOURS = {
    'table2':         1.1,
    'fig7_table1':   15.1,
    'fig9':          22.7,
    'fig10_11':       7.6,
    'fig13_14_15_17': 7.6,
}
TOTAL_ETA_HOURS = sum(SCRIPT_ETA_HOURS.values())


def _detect_gpu():
    """Return list of JAX devices whose platform is 'gpu' / 'cuda'."""
    try:
        import jax
    except ImportError:
        return []
    gpus = []
    for d in jax.devices():
        plat = getattr(d, 'platform', '').lower()
        if plat in ('gpu', 'cuda'):
            gpus.append(d)
    return gpus


def warn_and_confirm(script_label: str, pause_sec: int = 10) -> None:
    """
    Print the runtime warning banner and verify a GPU is available.

    Parameters
    ----------
    script_label : key into SCRIPT_ETA_HOURS (e.g. 'fig7').
    pause_sec    : seconds to sleep before execution continues, giving
                   the user a chance to cancel with Ctrl-C.
    """
    eta_this = SCRIPT_ETA_HOURS.get(script_label)
    gpus = _detect_gpu()

    bar = '=' * 68
    print('\n' + bar)
    print(f'  WindDensity-MBIR data collection — {script_label}')
    print(bar)
    if eta_this is not None:
        print(f'  Approximate runtime for this script  : ~{eta_this:.0f} h on GPU')
    print(f'  Approximate runtime for ALL scripts  : ~{TOTAL_ETA_HOURS:.0f} h '
          f'(~{TOTAL_ETA_HOURS/24:.1f} days)')
    print('  These estimates assume a modern CUDA GPU (e.g. A100 / 3090).')
    print(bar)

    if gpus:
        names = ', '.join(str(d) for d in gpus)
        print(f'  GPU detected: {names}')
        print(f'  Starting in {pause_sec} s  (Ctrl-C to abort)')
        print(bar + '\n')
        try:
            time.sleep(pause_sec)
        except KeyboardInterrupt:
            print('\nAborted by user.')
            sys.exit(130)
        return

    # No GPU found.
    print('  NO CUDA GPU DETECTED.')
    print('  Running on CPU is not supported — it would take weeks to')
    print('  months to finish and was never tested for this project.')
    print('')
    print('  If you really want to proceed anyway, re-run with the env')
    print('  variable  WINDDENSITY_ALLOW_CPU=1  set. Otherwise, configure')
    print('  JAX to see your GPU (install jax[cuda12] and ensure the CUDA')
    print('  runtime is visible) and try again.')
    print(bar + '\n')

    if os.environ.get('WINDDENSITY_ALLOW_CPU') == '1':
        print('WINDDENSITY_ALLOW_CPU=1 set — continuing on CPU at your own risk.\n')
        return
    sys.exit(1)
