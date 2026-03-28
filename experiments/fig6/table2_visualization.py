"""
Table 2: Mean NRMSE (%) comparison of FBP, Scale-Corrected FBP, and WindDensity-MBIR
for three geometries (3v2, 7v8, 11v16), evaluated on 4 OPD_TT planes.

Data: data/table2_fbp_comparison.npz
  nrmse:          (N_VOLS, n_geos, n_methods)
  geometry_names: ['3v2', '7v8', '11v16']
  method_names:   ['FBP', 'FBP_scaled', 'MBIR']
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np

# ---- Paths ----------------------------------------------------------------
DATA_FILE = Path(__file__).parent / 'data' / 'table2_fbp_comparison.npz'


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f'Data not found: {DATA_FILE}\n'
            'Run table2_data_collection.py first.'
        )

    data         = np.load(DATA_FILE, allow_pickle=True)
    nrmse        = data['nrmse']           # (N_VOLS, n_geos, n_methods)
    geo_names    = [s.decode() if isinstance(s, bytes) else s for s in data['geometry_names']]
    method_names = [s.decode() if isinstance(s, bytes) else s for s in data['method_names']]
    n_vols       = nrmse.shape[0]

    col_w = 18
    header = f"{'Geometry':<10}" + ''.join(f'{m:>{col_w}}' for m in method_names)
    print('\nTable 2: Mean NRMSE ± 2σ/√N (%) — 4 OPD_TT planes')
    print('-' * len(header))
    print(header)
    print('-' * len(header))

    for geo_idx, geo_name in enumerate(geo_names):
        row = f'{geo_name:<10}'
        for m_idx in range(len(method_names)):
            vals = nrmse[:, geo_idx, m_idx] * 100
            mean = vals.mean()
            err  = 2 * vals.std() / np.sqrt(n_vols)
            row += f'{f"{mean:.2f} ± {err:.2f}":>{col_w}}'
        print(row)

    print('-' * len(header))
    print(f'N = {n_vols} volumes\n')


if __name__ == '__main__':
    main()
