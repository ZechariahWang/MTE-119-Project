#!/usr/bin/env python3
# truss_adjuster.py
#
# Usage: python truss_adjuster.py <folder> <adjustment_strength> [chaos_prob] [return_prob]
#
# Maximises Performance Value (PV = max_load / mass) by hill-climbing node positions.
# Objective: minimise -PV  (infeasible configurations return float('inf')).
# Physics, FEA, and I/O live in truss_physics.py (shared with truss_solver.py).

import sys, json, pathlib, random
import numpy as np
import pandas as pd
from truss_physics import *

# ═══════════════════════════════════════════════════════════════════════════════
# Load physical parameters from options.json
# ═══════════════════════════════════════════════════════════════════════════════
_opts_path = pathlib.Path('options.json')
if _opts_path.exists():
    apply_options(json.load(open(_opts_path)))

# ═══════════════════════════════════════════════════════════════════════════════
# Objective function
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(nodes_df, members_df, loads_df, supports_dict):
    """
    Return -PV for a feasible configuration, or float('inf') if infeasible.
    Infeasible conditions:
      - Singular FEA system
      - Any member length outside its min_dist / max_dist bounds
      - Geometry constraint violated (span or height out of range)
    """
    nd       = {int(r.id): (float(r.x), float(r.y)) for r in nodes_df.itertuples()}
    elems    = {int(r.id): (int(r.start), int(r.end)) for r in members_df.itertuples()}
    ld       = process_loads(loads_df, nodes_df)
    result   = compute_pv(nd, elems, ld, supports_dict, members_df,
                          target_height_cm=None,   # adjuster moves nodes freely
                          enforce_geometry=True)   # reject out-of-spec geometry
    if result is None:
        return float('inf')
    return -result['pv']


# ═══════════════════════════════════════════════════════════════════════════════
# Optimisation loop
# ═══════════════════════════════════════════════════════════════════════════════
def adjust_truss(folder_path, adjustment_strength,
                 chaos_probability=0.0, return_probability=0.0):

    nodes_df, members_df, loads_df, opts_df = read_from_folder(folder_path)
    supports = {int(r.id): (bool(r.fix_x), bool(r.fix_y)) for r in nodes_df.itertuples()}

    # Build list of (dataframe_index, axis) pairs that are free to move
    possible_adjustments = []
    if 'movable_x' in nodes_df.columns and 'movable_y' in nodes_df.columns:
        for idx, row in nodes_df.iterrows():
            if not row.fix_x and row.movable_x: possible_adjustments.append((idx, 'x'))
            if not row.fix_y and row.movable_y: possible_adjustments.append((idx, 'y'))
    else:
        for idx, row in nodes_df.iterrows():
            if not (row.fix_x or row.fix_y):
                possible_adjustments.extend([(idx, 'x'), (idx, 'y')])

    if not possible_adjustments:
        print("No movable nodes found. Exiting.")
        sys.exit(0)

    # Symmetrical node pairs (from options.csv in the bridge folder)
    symmetrical_pairs = {}
    if not opts_df.empty and 'symmetrical_nodes' in opts_df.columns:
        pairs = opts_df['symmetrical_nodes'].dropna().astype(str).str.split('-').tolist()
        for pair in pairs:
            if len(pair) == 2:
                n1, n2 = int(pair[0]), int(pair[1])
                symmetrical_pairs[n1] = n2
                symmetrical_pairs[n2] = n1
        print(f"Symmetry pairs: {pairs}")

    best_nodes_df = nodes_df.copy()
    best_score    = evaluate(nodes_df, members_df, loads_df, supports)
    current_score = best_score

    def score_str(s):
        return f"{-s:.2f} N/kg" if s != float('inf') else "infeasible"

    print(f"Initial PV: {score_str(best_score)}")

    iteration = 0
    while True:
        iteration += 1
        new_nodes_df = nodes_df.copy()

        idx, axis = random.choice(possible_adjustments)
        delta = random.uniform(-adjustment_strength, adjustment_strength)
        dx = dy = 0.0
        if axis == 'x':
            dx = delta
            new_nodes_df.loc[idx, 'x'] += dx
        else:
            dy = delta
            new_nodes_df.loc[idx, 'y'] += dy

        # Enforce symmetry: mirror the movement across the y-axis
        node_id = int(new_nodes_df.loc[idx, 'id'])
        if node_id in symmetrical_pairs:
            partner_id   = symmetrical_pairs[node_id]
            partner_idxs = new_nodes_df[new_nodes_df['id'] == partner_id].index
            if len(partner_idxs) > 0:
                pidx = partner_idxs[0]
                new_nodes_df.loc[pidx, 'x'] -= dx   # mirror x
                new_nodes_df.loc[pidx, 'y'] += dy   # same y shift

        if not is_valid(new_nodes_df):
            continue

        new_score = evaluate(new_nodes_df, members_df, loads_df, supports)

        if new_score < current_score:
            nodes_df      = new_nodes_df
            current_score = new_score
            if new_score < best_score:
                best_score    = new_score
                best_nodes_df = new_nodes_df.copy()
                best_nodes_df.to_csv(pathlib.Path(folder_path) / 'nodes.csv', index=False)
                print(f"Iter {iteration:>7}: New best PV = {score_str(best_score)} -- saved")

        elif random.random() < chaos_probability:
            nodes_df      = new_nodes_df
            current_score = new_score
            print(f"Iter {iteration:>7}: Chaos accepted  PV = {score_str(new_score)}")

        elif random.random() < return_probability:
            nodes_df      = best_nodes_df.copy()
            current_score = best_score
            print(f"Iter {iteration:>7}: Returned to best PV = {score_str(best_score)}")

        if iteration % 100 == 0:
            print(f"Iter {iteration:>7}: current = {score_str(current_score)}"
                  f"  best = {score_str(best_score)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    if len(sys.argv) not in [3, 4, 5]:
        print("Usage: python truss_adjuster.py <folder> <adj_strength>"
              " [chaos_prob] [return_prob]")
        sys.exit(1)

    folder = sys.argv[1]
    if not pathlib.Path(folder).is_dir():
        print(f"Error: folder not found: '{folder}'")
        sys.exit(1)

    adj  = float(sys.argv[2])
    chao = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.0
    ret  = float(sys.argv[4]) if len(sys.argv) >= 5 else 1.0

    print(f"Optimising PV in '{folder}'  adj={adj}  chaos={chao}  return={ret}")
    print(f"Physical params: tensile={BALSA_TENSILE_MPA} MPa  "
          f"bearing={BALSA_BEARING_MPA} MPa  rho={BALSA_DENSITY_KG_M3} kg/m3")
    adjust_truss(folder, adj, chao, ret)
