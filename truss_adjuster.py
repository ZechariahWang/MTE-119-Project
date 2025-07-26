#!/usr/bin/env python3
# truss_adjuster.py
#
# Usage: python truss_adjuster.py <folder_path> <adjustment_strength>

import sys, math, json, pathlib, random, time
import numpy as np, pandas as pd

# ────────────────────────────────────────────────────────────────────────────────
# 1.  Read input ─ folder with nodes.csv, members.csv, loads.csv
# ────────────────────────────────────────────────────────────────────────────────
def read_from_folder(folder):
    folder = pathlib.Path(folder)
    nodes_df   = pd.read_csv(folder / 'nodes.csv')
    members_df = pd.read_csv(folder / 'members.csv')
    loads_df   = pd.read_csv(folder / 'loads.csv')
    opts_df    = pd.read_csv(folder / 'options.csv') if (folder / 'options.csv').exists() else pd.DataFrame()
    return nodes_df, members_df, loads_df, opts_df

# ────────────────────────────────────────────────────────────────────────────────
# 2.  FEA helpers (from truss_solver.py)
# ────────────────────────────────────────────────────────────────────────────────
def dist(p,q): return np.hypot(*(np.subtract(p,q)))

def build_system(nodes, elems, loads, supports, E=1.0):
    ndof = 2*len(nodes);  K = np.zeros((ndof, ndof));  F = np.zeros(ndof)
    Ls, dirs = [], []

    for (i,j) in elems.values():
        xi,yi = nodes[i];  xj,yj = nodes[j]
        L  = dist((xi,yi),(xj,yj));
        if L == 0: return None, None, None # Avoid division by zero
        cx,cy = (xj-xi)/L, (yj-yi)/L
        k = (E/L)*np.array([[ cx*cx, cx*cy, -cx*cx, -cx*cy],
                            [ cx*cy, cy*cy, -cx*cy, -cy*cy],
                            [-cx*cx,-cx*cy,  cx*cx,  cx*cy],
                            [-cx*cy,-cy*cy,  cx*cy,  cy*cy]])
        dof = [2*i, 2*i+1, 2*j, 2*j+1]
        K[np.ix_(dof,dof)] += k
        Ls.append(L); dirs.append((cx,cy))

    for n,(fx,fy) in loads.items():
        F[2*n] += fx; F[2*n+1] += fy

    fixed = [2*n   for n,(fx,_) in supports.items() if fx] + \
            [2*n+1 for n,(_,fy) in supports.items() if fy]
    free = [d for d in range(ndof) if d not in fixed]

    try:
        u = np.zeros(ndof);  u[free] = np.linalg.solve(K[np.ix_(free,free)], F[free])
        return u, Ls, dirs
    except np.linalg.LinAlgError:
        return None, None, None

def member_forces(nodes, elems, u, Ls, dirs, E=1.0):
    N=[]
    for eid,(i,j) in elems.items():
        cx,cy = dirs[eid];  L = Ls[eid];  dof = [2*i,2*i+1,2*j,2*j+1]
        ul = u[dof];  N.append((E/L)*np.dot([-cx,-cy,cx,cy],ul))
    return np.array(N)

def calculate_cost(nodes, members, loads, supports, opts):
    def process_loads(loads_df, nodes_df):
        load_dict = {}
        node_pos_dict = {int(r.id): (r.x, r.y) for r in nodes_df.itertuples()}

        for r in loads_df.itertuples():
            node_val = str(r.node)
            fx = getattr(r, 'Fx', 0.0); fy = getattr(r, 'Fy', 0.0)
            fx = 0.0 if pd.isna(fx) else float(fx)
            fy = 0.0 if pd.isna(fy) else float(fy)

            if '-' in node_val:  # Pressure load
                node_ids = [int(n) for n in node_val.split('-')]
                if len(node_ids) < 2: continue
                if not all(nid in node_pos_dict for nid in node_ids): continue

                x = {nid: node_pos_dict[nid][0] for nid in node_ids}
                y = {nid: node_pos_dict[nid][1] for nid in node_ids}

                for i, nid in enumerate(node_ids):
                    nfx, nfy = 0.0, 0.0
                    if fy != 0:
                        if i == 0: nfy = (x[node_ids[i+1]] - x[nid]) / 2.0 * fy
                        elif i == len(node_ids)-1: nfy = (x[nid] - x[node_ids[i-1]]) / 2.0 * fy
                        else: nfy = (x[node_ids[i+1]] - x[node_ids[i-1]]) / 2.0 * fy
                    if fx != 0:
                        if i == 0: nfx = (y[node_ids[i+1]] - y[nid]) / 2.0 * fx
                        elif i == len(node_ids)-1: nfx = (y[nid] - y[node_ids[i-1]]) / 2.0 * fx
                        else: nfx = (y[node_ids[i+1]] - y[node_ids[i-1]]) / 2.0 * fx
                    
                    if nid in load_dict: load_dict[nid] = (load_dict[nid][0] + nfx, load_dict[nid][1] + nfy)
                    else: load_dict[nid] = (nfx, nfy)
            else:  # Point load
                nid = int(node_val)
                if nid in load_dict: load_dict[nid] = (load_dict[nid][0] + fx, load_dict[nid][1] + fy)
                else: load_dict[nid] = (fx, fy)
        return load_dict

    node_dict   = {int(r.id): (float(r.x), float(r.y)) for r in nodes.itertuples()}
    elements    = {int(r.id): (int(r.start), int(r.end)) for r in members.itertuples()}
    load_dict   = process_loads(loads, nodes)

    T_LIMIT      = float(opts['T_limit'].iloc[0]) if not opts.empty and 'T_limit' in opts.columns and not opts['T_limit'].dropna().empty else 8.0
    C_LIMIT      = float(opts['C_limit'].iloc[0]) if not opts.empty and 'C_limit' in opts.columns and not opts['C_limit'].dropna().empty else 5.0
    COST_PER_M   = float(opts['cost_per_m'].iloc[0]) if not opts.empty and 'cost_per_m' in opts.columns and not opts['cost_per_m'].dropna().empty else 9.0
    COST_GUSSET  = float(opts['cost_gusset'].iloc[0]) if not opts.empty and 'cost_gusset' in opts.columns and not opts['cost_gusset'].dropna().empty else 5.0

    u, Ls, dirs = build_system(node_dict, elements, load_dict, supports)
    if u is None:
        return float('inf')

    forces = member_forces(node_dict, elements, u, Ls, dirs)
    
    # Check for constraint violations
    if np.any(np.iscomplex(forces)) or np.any(np.isnan(forces)) or np.any(np.isinf(forces)):
        return float('inf')

    dup = [max(1, math.ceil(abs(f)/(T_LIMIT if f>0 else C_LIMIT))) for f in forces]

    cost_members = sum(COST_PER_M * Ls[eid] * dup[eid] for eid in range(len(elements)))
    cost_gusset  = COST_GUSSET * len(node_dict)
    total_cost   = cost_members + cost_gusset
    return total_cost

# ────────────────────────────────────────────────────────────────────────────────
# 3.  Main adjustment loop
# ────────────────────────────────────────────────────────────────────────────────
def adjust_truss(folder_path, adjustment_strength, chaos_probability=0.0, return_threshold=1.0):
    nodes_df, members_df, loads_df, opts_df = read_from_folder(folder_path)
    
    supports = {int(r.id): (bool(r.fix_x), bool(r.fix_y)) for r in nodes_df.itertuples()}
    
    possible_adjustments = []
    if 'movable_x' in nodes_df.columns and 'movable_y' in nodes_df.columns:
        for idx, row in nodes_df.iterrows():
            if not row.fix_x and row.movable_x: possible_adjustments.append((idx, 'x'))
            if not row.fix_y and row.movable_y: possible_adjustments.append((idx, 'y'))
    elif 'movable' in nodes_df.columns:
        movable_nodes = [idx for idx, row in nodes_df.iterrows() if row.movable and not (row.fix_x or row.fix_y)]
        for idx in movable_nodes:
            possible_adjustments.append((idx, 'x'))
            possible_adjustments.append((idx, 'y'))
    else:
        movable_nodes = [idx for idx, row in nodes_df.iterrows() if not (row.fix_x or row.fix_y)]
        for idx in movable_nodes:
            possible_adjustments.append((idx, 'x'))
            possible_adjustments.append((idx, 'y'))

    if not possible_adjustments:
        print("No movable nodes found. Exiting.")
        sys.exit(0)

    # Handle symmetrical nodes
    symmetrical_pairs = {}
    if not opts_df.empty and 'symmetrical_nodes' in opts_df.columns:
        symmetrical_nodes = opts_df['symmetrical_nodes'].dropna().astype(str).str.split('-').tolist()
        for pair in symmetrical_nodes:
            if len(pair) == 2:
                node1, node2 = int(pair[0]), int(pair[1])
                symmetrical_pairs[node1] = node2
                symmetrical_pairs[node2] = node1
        print(f"Symmetry enabled for node pairs: {symmetrical_nodes}")

    best_nodes_df = nodes_df.copy()
    best_cost = calculate_cost(nodes_df, members_df, loads_df, supports, opts_df)
    current_cost = best_cost
    print(f"Initial cost: ${best_cost:,.2f}")

    iteration = 0
    while True:
        iteration += 1
        
        # Create a copy to modify from the current state
        new_nodes_df = nodes_df.copy()

        # Pick a random movable node and axis and adjust its position
        node_to_adjust_idx, axis_to_adjust = random.choice(possible_adjustments)
        
        adjustment = random.uniform(-adjustment_strength, adjustment_strength)
        dx, dy = 0, 0

        if axis_to_adjust == 'x':
            dx = adjustment
            new_nodes_df.loc[node_to_adjust_idx, 'x'] += dx
        else: # 'y'
            dy = adjustment
            new_nodes_df.loc[node_to_adjust_idx, 'y'] += dy

        # Enforce symmetry if applicable
        node_id_to_adjust = int(new_nodes_df.loc[node_to_adjust_idx, 'id'])
        if node_id_to_adjust in symmetrical_pairs:
            partner_id = symmetrical_pairs[node_id_to_adjust]
            partner_indices = new_nodes_df[new_nodes_df['id'] == partner_id].index
            if len(partner_indices) > 0:
                partner_idx = partner_indices[0]
                
                # Enforce y-axis symmetry
                new_nodes_df.loc[partner_idx, 'x'] -= dx
                new_nodes_df.loc[partner_idx, 'y'] += dy

        new_cost = calculate_cost(new_nodes_df, members_df, loads_df, supports, opts_df)

        if new_cost < current_cost:
            nodes_df = new_nodes_df
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_nodes_df = new_nodes_df.copy()
                best_nodes_df.to_csv(pathlib.Path(folder_path) / 'nodes.csv', index=False)
                print(f"Iteration {iteration}: New best cost: ${best_cost:,.2f} - Saved to nodes.csv")
        elif random.random() < chaos_probability:
            if new_cost < return_threshold * best_cost:
                nodes_df = new_nodes_df
                current_cost = new_cost
                print(f"Iteration {iteration}: Accepted worse cost due to chaos: ${new_cost:,.2f}")
            else:
                nodes_df = best_nodes_df.copy()
                current_cost = best_cost
                print(f"Iteration {iteration}: Returning to best_cost state due to chaos: ${new_cost:,.2f}")
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Current cost: ${current_cost:,.2f}, Best cost: ${best_cost:,.2f}")

if __name__ == "__main__":
    if len(sys.argv) not in [3, 5]:
        print("Usage: python truss_adjuster.py <folder_path> <adjustment_strength> [chaos_probability] <return_threshold>")
        print("Example: python truss_adjuster.py ./bridge_folder 0.1 0.05 1.2")
        sys.exit(1)
    
    folder = sys.argv[1]
    if not pathlib.Path(folder).is_dir():
        print(f"Error: Folder not found at '{folder}'")
        sys.exit(1)

    adjustment_strength = float(sys.argv[2])
    chaos_probability = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.0
    return_threshold = float(sys.argv[4]) if len(sys.argv) >= 5 else 1.0

    print(f"Starting truss adjustment in folder: {folder} with adjustment strength: {adjustment_strength}, chaos: {chaos_probability}, return threshold: {return_threshold}")
    adjust_truss(folder, adjustment_strength, chaos_probability, return_threshold)
