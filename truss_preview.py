#!/usr/bin/env python3
# truss_solver.py
#
# Usage:  python truss_solver.py  bridge.xlsx          # Excel with sheets
#     or  python truss_solver.py  nodes.csv members.csv loads.csv  [options.json]

import sys, math, json, pathlib
import numpy as np, pandas as pd, matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────────────
# 1.  Read input ─ Excel workbook OR individual CSVs
# ────────────────────────────────────────────────────────────────────────────────
def read_from_excel(fname: pathlib.Path):
    xls = pd.ExcelFile(fname)
    nodes_df   = xls.parse('Nodes')
    members_df = xls.parse('Members')
    loads_df   = xls.parse('Loads')
    opts_df    = xls.parse('Options') if 'Options' in xls.sheet_names else pd.DataFrame()
    return nodes_df, members_df, loads_df, opts_df

def read_from_csvs(paths):
    nodes_df   = pd.read_csv(paths[0])
    members_df = pd.read_csv(paths[1])
    loads_df   = pd.read_csv(paths[2])
    opts_df    = pd.read_json(paths[3], typ='series').to_frame().T if len(paths) > 3 else pd.DataFrame()
    return nodes_df, members_df, loads_df, opts_df

def read_from_folder(folder):
    folder = pathlib.Path(folder)
    nodes_df   = pd.read_csv(folder / 'nodes.csv')
    members_df = pd.read_csv(folder / 'members.csv')
    loads_df   = pd.read_csv(folder / 'loads.csv')
    opts_df    = pd.read_json(folder / 'options.json', typ='series').to_frame().T if (folder / 'options.json').exists() else pd.DataFrame()
    return nodes_df, members_df, loads_df, opts_df

if len(sys.argv) < 2:
    print("Give an Excel file (nodes/members/loads) or 3‑4 CSVs!")
    sys.exit(1)

in_files = [pathlib.Path(p) for p in sys.argv[1:]]
if in_files[0].suffix.lower() in ('.xlsx', '.xls'):
    nodes, members, loads, opts = read_from_excel(in_files[0])
else:
    if len(in_files) == 1 and in_files[0].is_dir():
        nodes, members, loads, opts = read_from_folder(in_files[0])
    else:
        if len(in_files) < 3:
            print("Need at least nodes.csv  members.csv  loads.csv")
            sys.exit(1)
        nodes, members, loads, opts = read_from_csvs(in_files)

# ────────────────────────────────────────────────────────────────────────────────
# 2.  Convert to convenient Python structures
# ────────────────────────────────────────────────────────────────────────────────
node_dict   = {int(r.id): (float(r.x), float(r.y))          for r in nodes.itertuples()}
supports    = {int(r.id): (bool(r.fix_x), bool(r.fix_y))    for r in nodes.itertuples()}
elements    = {int(r.id): (int(r.start), int(r.end))        for r in members.itertuples()}
load_dict   = {int(r.node): (float(r.Fx), float(r.Fy))      for r in loads.itertuples()}

# Defaults / overrides
T_LIMIT      = opts.get('T_limit',  8.0).iloc[0] if not opts.empty else 8.0
C_LIMIT      = opts.get('C_limit',  5.0).iloc[0] if not opts.empty else 5.0
COST_PER_M   = opts.get('cost_per_m', 9.0).iloc[0] if not opts.empty else 9.0
COST_GUSSET  = opts.get('cost_gusset', 5.0).iloc[0] if not opts.empty else 5.0

# ────────────────────────────────────────────────────────────────────────────────
# 6.  Quick plot
# ────────────────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(15,6))
ys = [y for _,y in node_dict.values()]

for eid,(i,j) in elements.items():
    x=[node_dict[i][0], node_dict[j][0]]
    y=[node_dict[i][1], node_dict[j][1]]
    col='black'
    ax1.plot(x,y,col,lw=2)
    xm,ym=(x[0]+x[1])/2,(y[0]+y[1])/2

for n,(x,y) in node_dict.items():
    ax1.plot(x,y,'ko'); ax1.text(x,y+0.25,str(n),ha='center')

# Calculate force_scale so that the largest force arrow is about the same as the truss size
truss_width = max(x for x, _ in node_dict.values()) - min(x for x, _ in node_dict.values())
truss_height = max(y for _, y in node_dict.values()) - min(y for _, y in node_dict.values())
truss_size = min(truss_width, truss_height)
max_force = max(np.hypot(fx, fy) for fx, fy in load_dict.values()) if load_dict else 1.0
force_scale = truss_size / max_force if max_force != 0 else 1.0
force_scale /= 2

for n,(fx,fy) in load_dict.items():
    ax1.arrow(node_dict[n][0], node_dict[n][1], fx * force_scale, fy * force_scale,
              head_width=0.12, head_length=0.25, fc='green', ec='green')
    xm,ym=node_dict[n][0]+fx*force_scale/2,node_dict[n][1]+fy*force_scale/2
    ax1.text(xm,ym,f"{np.hypot(fx, fy)}kN",fontsize=8,ha='center',va='center',bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))

ax1.grid(True)
ax1.set_xlabel("m");  ax1.set_ylabel("m")
plt.margins(x=0.15, y=0.15)
# Expand plot limits for more space around the truss
x_vals = [x for x, _ in node_dict.values()]
y_vals = [y for _, y in node_dict.values()]
plt.title("Truss")
plt.tight_layout()
plt.axis('equal')
plt.show()
