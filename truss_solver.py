#!/usr/bin/env python3
# truss_solver.py
#
# Usage:  python truss_solver.py  <folder>
#     or  python truss_solver.py  nodes.csv members.csv loads.csv
#     or  python truss_solver.py  bridge.xlsx
#
# Physics, FEA, and I/O live in truss_physics.py (shared with truss_adjuster.py).

import sys, json, pathlib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from truss_physics import *

# ═══════════════════════════════════════════════════════════════════════════════
# Load physical parameters from options.json (overrides defaults in truss_physics)
# ═══════════════════════════════════════════════════════════════════════════════
_opts_path = pathlib.Path('options.json')
if _opts_path.exists():
    apply_options(json.load(open(_opts_path)))

# ═══════════════════════════════════════════════════════════════════════════════
# Read input
# ═══════════════════════════════════════════════════════════════════════════════
if len(sys.argv) < 2:
    print("Usage: python truss_solver.py <folder | excel | csvs>")
    sys.exit(1)

src_folder = None
in_files   = [pathlib.Path(p) for p in sys.argv[1:]]

if in_files[0].suffix.lower() in ('.xlsx', '.xls'):
    nodes, members, loads, _ = read_from_excel(in_files[0])
elif len(in_files) == 1 and in_files[0].is_dir():
    src_folder = in_files[0]
    nodes, members, loads, _ = read_from_folder(src_folder)
else:
    if len(in_files) < 3:
        print("Need at least: nodes.csv  members.csv  loads.csv")
        sys.exit(1)
    nodes, members, loads, _ = read_from_csvs(in_files)

node_dict = {int(r.id): (float(r.x), float(r.y))       for r in nodes.itertuples()}
supports  = {int(r.id): (bool(r.fix_x), bool(r.fix_y)) for r in nodes.itertuples()}
elements  = {int(r.id): (int(r.start), int(r.end))      for r in members.itertuples()}
load_dict = process_loads(loads, nodes)

# ═══════════════════════════════════════════════════════════════════════════════
# Analyse  (geometry is height-scaled; violations are reported but not enforced)
# ═══════════════════════════════════════════════════════════════════════════════
result = compute_pv(node_dict, elements, load_dict, supports, members,
                    target_height_cm=None,
                    enforce_geometry=True)

if result is None:
    print("ERROR: FEA system is singular -- check geometry / supports.")
    sys.exit(1)

nd        = result['node_dict']
forces    = result['forces']
cap_T     = result['cap_T']    # tension limit per member (N)
cap_C     = result['cap_C']    # compression limit per member = min(buckling, bearing cap) (N)
Ls        = result['Ls']
length_cm = result['length_cm']
height_cm = result['height_cm']
pv        = result['pv']

# ═══════════════════════════════════════════════════════════════════════════════
# Geometry validation report
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Geometry Validation ===")
if result['geom_ok']:
    print(f"  OK  Span   = {length_cm:.2f} cm  [{LENGTH_MIN_CM}-{LENGTH_MAX_CM} cm]")
    print(f"  OK  Height = {height_cm:.2f} cm  [{HEIGHT_MIN_CM}-{HEIGHT_MAX_CM} cm]")
for v in result['violations']:
    print(v)

# ═══════════════════════════════════════════════════════════════════════════════
# Member analysis table
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Member Analysis ===")
print(f"  Section      : {SECTION_W_MM} x {SECTION_T_MM} mm  (A = {SECTION_A_MM2:.2f} mm2)")
print(f"  Euler params : E = {BALSA_E_MPA:.0f} MPa,  I = per-member (I_mm4 column in members.csv)")
print(f"               P_cr = pi^2 * E * I / L^2  (pin-pin, K=1)")
print(f"  Cap T        : Max_Tension_N from members.csv")
print(f"  Cap C        : min( Euler buckling P_cr=pi^2*E*I/L^2,  Max_Compression_N from members.csv )")
print()

hdr = (f"{'ID':>3}  {'Nodes':>7}  {'L mm':>6}  "
       f"{'Force N':>9}  {'T/C':>3}  {'MaxT N':>7}  {'MaxC N':>7}  Status")
print(hdr)
print("-" * len(hdr))

any_fail = False
for eid, (i, j) in elements.items():
    f   = forces[eid]
    typ = "T" if f > 0 else ("C" if f < 0 else "-")
    tT, tC = cap_T[eid], cap_C[eid]
    if   f > 0 and abs(f) > tT: status = "FAIL-T !!"; any_fail = True
    elif f < 0 and abs(f) > tC: status = "FAIL-C !!"; any_fail = True
    else:                        status = "OK"

    print(f"{eid:>3}  {i:>3}->{j:<3}  {Ls[eid]*10:>6.1f}  "
          f"{f:>9.2f}  {typ:>3}  {tT:>7.1f}  {tC:>7.1f}  {status}")

# ═══════════════════════════════════════════════════════════════════════════════
# Performance Value summary
# ═══════════════════════════════════════════════════════════════════════════════
pv_gg = pv / 9.81   # convert N/kg -> g/g  (grams-force / gram)

print(f"\n=== Performance Value  (height = {height_cm:.2f} cm) ===")
print(f"  Reference load      : {result['ref_load_N']/9.81*1000:.3f} gf")
print(f"  First-failure mult. : {result['load_multiplier']:.4f}")
print(f"  Max applicable load : {result['max_load_N']/9.81*1000:.2f} gf")
print(f"  Truss mass          : {result['mass_kg']*1000:.3f} g"
      f"  ({NUM_FACES} faces, rho = {BALSA_DENSITY_KG_M3} kg/m3)")
print(f"  PV                  = {pv_gg:.1f} g/g")
if any_fail:
    print("  *** WARNING: member(s) exceed capacity at current load scale ***")
if not result['geom_ok']:
    print("  *** WARNING: geometry constraint violated -- fix span/height first ***")

# ═══════════════════════════════════════════════════════════════════════════════
# Write computed limits back to members.csv
# ═══════════════════════════════════════════════════════════════════════════════
members['Section_A_mm2'] = [round(result['areas'][eid], 4) for eid in range(len(elements))]
members['Max_Tension_N'] = [round(cap_T[eid], 2)           for eid in range(len(elements))]

if src_folder is not None:
    out_path = pathlib.Path(src_folder) / 'members.csv'
    members.to_csv(out_path, index=False)
    print(f"\n  members.csv updated -> {out_path}")
else:
    print("\n  (run with a folder argument to auto-save members.csv)")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot  --  blue = tension, red = compression, magenta = failure
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(15, 6))

for eid, (i, j) in elements.items():
    f  = forces[eid]
    tT, tC = cap_T[eid], cap_C[eid]
    failed = (f > 0 and abs(f) > tT) or (f < 0 and abs(f) > tC)
    col = 'magenta' if failed else ('blue' if f > 0 else ('red' if f < 0 else 'black'))
    x = [nd[i][0], nd[j][0]]; y = [nd[i][1], nd[j][1]]
    ax.plot(x, y, col, lw=3 if failed else 2)
    ax.text((x[0]+x[1])/2, (y[0]+y[1])/2, f"{abs(f):.1f}", fontsize=7,
            ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

for n, (x, y) in nd.items():
    ax.plot(x, y, 'ko', ms=5)
    ax.text(x, y + 0.05 * height_cm, str(n), ha='center', fontsize=8)

max_f  = max((abs(fy) for fx, fy in load_dict.values()), default=1.0)
fscale = min(length_cm, height_cm) / max_f / 4 if max_f else 1.0
for n, (fx, fy) in load_dict.items():
    ax.arrow(nd[n][0], nd[n][1], fx*fscale, fy*fscale,
             head_width=0.08, head_length=0.15, fc='green', ec='green')

ax.set_aspect('equal'); ax.grid(True)
ax.set_xlabel("cm"); ax.set_ylabel("cm")
ax.margins(x=0.2, y=0.4)
plt.title(f"H={height_cm:.2f} cm  L={length_cm:.2f} cm  PV={pv_gg:.1f} g/g"
          f"  |  blue=T  red=C  magenta=FAIL")
plt.tight_layout()
plt.show()
