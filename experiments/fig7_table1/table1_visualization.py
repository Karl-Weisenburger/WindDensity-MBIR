"""
Table 1: Reconstruction performance for three paper geometries (3v2, 7v8, 11v16),
evaluated on OPD_TT measurements.

Prints a 3x3 table (geometry x metric) with mean ± std across the volumes
processed by fig7_data_collection.py:

  - Recon Time (s)
  - VCD Iterations
  - Final Pct Chg (%)

The same numbers are also printed at the end of fig7_data_collection.py; this
script simply reloads them from the saved .npz so the table can be regenerated
without rerunning data collection.

Data: data/fig7_geometry_sweep.npz
  perf_recon_time     : (N_VOLS, n_perf)  float  — seconds per reconstruction
  perf_num_iterations : (N_VOLS, n_perf)  int    — VCD iteration count
  perf_final_pct      : (N_VOLS, n_perf)  float  — final stop-threshold pct change
  perf_geo_names      : (n_perf,)         bytes  — geometry labels (e.g. b'3v2')
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np

DATA_FILE = Path(__file__).parent / 'data' / 'fig7_geometry_sweep.npz'


def _decode(arr):
    return [s.decode() if isinstance(s, bytes) else s for s in arr]


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f'Data not found: {DATA_FILE}\n'
            'Run fig7_data_collection.py first.'
        )

    data                = np.load(DATA_FILE, allow_pickle=True)
    perf_recon_time     = data['perf_recon_time']       # (N_VOLS, n_perf)
    perf_num_iterations = data['perf_num_iterations']   # (N_VOLS, n_perf)
    perf_final_pct      = data['perf_final_pct']        # (N_VOLS, n_perf)
    geo_names           = _decode(data['perf_geo_names'])
    n_vols              = perf_recon_time.shape[0]

    col_w     = 14
    geo_col_w = 8
    print('\n')
    print('=' * 60)
    print('Table 1: Reconstruction Performance (OPD_TT measurements)')
    print(f'         Mean ± std across {n_vols} volumes')
    print('=' * 60)
    header = (f"{'Geometry':<{geo_col_w}}"
              f"{'Recon Time (s)':>{col_w}}"
              f"{'VCD Iterations':>{col_w}}"
              f"{'Final Pct Chg (%)':>{col_w + 3}}")
    print(header)
    print('-' * 60)

    for pi, geo_name in enumerate(geo_names):
        mask = ~np.isnan(perf_recon_time[:, pi])
        if not mask.any():
            continue
        t_mean = np.mean(perf_recon_time[mask, pi])
        t_std  = np.std(perf_recon_time[mask, pi], ddof=1)
        i_mean = np.mean(perf_num_iterations[mask, pi])
        i_std  = np.std(perf_num_iterations[mask, pi].astype(float), ddof=1)
        p_mean = np.mean(perf_final_pct[mask, pi])
        p_std  = np.std(perf_final_pct[mask, pi], ddof=1)
        row = (f"{geo_name:<{geo_col_w}}"
               f"{f'{t_mean:.1f} ± {t_std:.1f}':>{col_w}}"
               f"{f'{i_mean:.1f} ± {i_std:.1f}':>{col_w}}"
               f"{f'{p_mean:.3f} ± {p_std:.3f}':>{col_w + 3}}")
        print(row)
    print('=' * 60)


if __name__ == '__main__':
    main()
