"""
truss_physics.py  --  shared physics, FEA, and I/O for the MTE 219 truss project.

Imported by both truss_solver.py and truss_adjuster.py.
Do not add plotting or CLI argument parsing here.

Coordinate system
-----------------
All node coordinates in nodes.csv are in centimetres (cm).
Member lengths from the FEA are therefore also in cm.
The buckling formula requires L in mm, so lengths are converted with  L_mm = L_cm * 10.
There is no abstract unit-conversion constant -- the factor of 10 is written explicitly
wherever it appears.
"""

import pathlib
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# Default physical parameters  --  overridden at runtime by apply_options()
# ═══════════════════════════════════════════════════════════════════════════════

# Cross-section: 14.3 mm wide x 3.2 mm thick balsa strip
SECTION_W_MM        = 14.3
SECTION_T_MM        =  3.2
SECTION_A_MM2       = SECTION_W_MM * SECTION_T_MM          # 45.76 mm2

# Second moment of area  *** UPDATE FROM LAB RESULTS / actual cross-section ***
# Minimum I governs Euler buckling (bending about the weak axis).
# For a solid rectangle: I_min = w * t^3 / 12  (t is the smaller dimension).
# Default is computed from SECTION_W_MM and SECTION_T_MM; override in options.json.
SECTION_I_MM4       = SECTION_W_MM * SECTION_T_MM**3 / 12  # ~39.05 mm4  <- UPDATE

# Balsa Young's modulus  *** UPDATE FROM LAB RESULTS ***
BALSA_E_MPA         = 3400.0        # MPa (typical balsa range 2000-5000)  <- UPDATE

# Balsa density  *** UPDATE FROM LAB RESULTS ***
BALSA_DENSITY_KG_M3 = 160.0         # kg/m3 (typical balsa range 100-200)  <- UPDATE

# Force limits  *** SET IN options.json — update from lab results ***
#
# MAX_TENSION_N       -- max axial tensile force any member can carry (N).
#                        Should reflect the governing failure mode from testing
#                        (normal stress rupture or bearing/crushing at the pin,
#                        whichever is lower).
#
# MAX_COMPRESSION_BEARING_N -- uniform compression cap from bearing/crushing at the
#                        pin joint (N).  Buckling (per-member, length-dependent) is
#                        applied on top; effective limit = min(buckling, this value).
#
# Defaults below are computed from typical material properties as a starting point.
# Override them in options.json once you have measured values.
_BALSA_TENSILE_MPA   = 20.0         # used only to compute the default below
_BALSA_BEARING_MPA   = 15.0         # used only to compute the default below
_PIN_DIA_MM          = 25.4 / 8     # 3.175 mm
_BEARING_A_MM2       = _PIN_DIA_MM * SECTION_T_MM           # 10.16 mm2

MAX_TENSION_N             = min(_BALSA_TENSILE_MPA * SECTION_A_MM2,
                                _BALSA_BEARING_MPA * _BEARING_A_MM2)  # 152.4 N
MAX_COMPRESSION_BEARING_N = _BALSA_BEARING_MPA * _BEARING_A_MM2       # 152.4 N

# Height optimisation target -- solver sweeps this; adjuster enforces the range
TRUSS_HEIGHT_TARGET_CM = 10.0

# Physical dimension bounds (cm) -- validated at run-time
HEIGHT_MIN_CM,  HEIGHT_MAX_CM  =  9.5, 10.5
LENGTH_MIN_CM,  LENGTH_MAX_CM  = 39.0, 41.0
WIDTH_MIN_CM,   WIDTH_MAX_CM   =  7.5,  8.5   # out-of-plane; verify physically

# Number of parallel planar faces in the full 3-D truss (used for mass estimation)
NUM_FACES = 2


def apply_options(opts: dict):
    """
    Override module-level defaults from a dict loaded from options.json.
    Call this once at startup in truss_solver.py and truss_adjuster.py.

    Recognised keys
    ---------------
    density_kg_m3    -- balsa density for mass calculation
    E_mpa            -- balsa Young's modulus (MPa) for Euler buckling
    num_faces        -- number of parallel planar faces in 3-D truss
    height_target_cm -- default height target for solver height sweep

    Tension/compression limits and I_mm4 per member come from members.csv
    (Max_Tension_N, Max_Compression_N, and I_mm4 columns), not from options.json.
    """
    global BALSA_DENSITY_KG_M3, BALSA_E_MPA, NUM_FACES, TRUSS_HEIGHT_TARGET_CM

    if 'density_kg_m3'    in opts: BALSA_DENSITY_KG_M3   = float(opts['density_kg_m3'])
    if 'E_mpa'            in opts: BALSA_E_MPA            = float(opts['E_mpa'])
    if 'num_faces'        in opts: NUM_FACES              = int(opts['num_faces'])
    if 'height_target_cm' in opts: TRUSS_HEIGHT_TARGET_CM = float(opts['height_target_cm'])


# ═══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════════════

def read_from_excel(fname):
    xls        = pd.ExcelFile(fname)
    nodes_df   = xls.parse('Nodes')
    members_df = xls.parse('Members')
    loads_df   = xls.parse('Loads')
    opts_df    = xls.parse('Options') if 'Options' in xls.sheet_names else pd.DataFrame()
    return nodes_df, members_df, loads_df, opts_df


def read_from_csvs(paths):
    nodes_df   = pd.read_csv(paths[0])
    members_df = pd.read_csv(paths[1])
    loads_df   = pd.read_csv(paths[2])
    opts_df    = (pd.read_json(paths[3], typ='series').to_frame().T
                  if len(paths) > 3 else pd.DataFrame())
    return nodes_df, members_df, loads_df, opts_df


def read_from_folder(folder):
    """Read all CSVs from a bridge folder.  options.csv is topology-only (symmetry pairs)."""
    folder     = pathlib.Path(folder)
    nodes_df   = pd.read_csv(folder / 'nodes.csv')
    members_df = pd.read_csv(folder / 'members.csv')
    loads_df   = pd.read_csv(folder / 'loads.csv')
    opts_df    = (pd.read_csv(folder / 'options.csv')
                  if (folder / 'options.csv').exists() else pd.DataFrame())
    return nodes_df, members_df, loads_df, opts_df


# ═══════════════════════════════════════════════════════════════════════════════
# FEA core
# ═══════════════════════════════════════════════════════════════════════════════

def dist(p, q):
    return np.hypot(*(np.subtract(p, q)))


def process_loads(loads_df, nodes_df):
    """Convert loads.csv into {node_id: (Fx, Fy)} with tributary-area distribution."""
    load_dict     = {}
    node_pos_dict = {int(r.id): (r.x, r.y) for r in nodes_df.itertuples()}

    for r in loads_df.itertuples():
        node_val = str(r.node)
        fx = getattr(r, 'Fx', 0.0); fy = getattr(r, 'Fy', 0.0)
        fx = 0.0 if pd.isna(fx) else float(fx)
        fy = 0.0 if pd.isna(fy) else float(fy)

        if '-' in node_val:                               # distributed load
            node_ids = [int(n) for n in node_val.split('-')]
            if len(node_ids) < 2:
                continue
            if not all(nid in node_pos_dict for nid in node_ids):
                print(f"Warning: load '{node_val}' references missing nodes -- skipped.")
                continue
            x = {nid: node_pos_dict[nid][0] for nid in node_ids}
            y = {nid: node_pos_dict[nid][1] for nid in node_ids}
            for i, nid in enumerate(node_ids):
                nfx = nfy = 0.0
                if fy != 0:
                    if   i == 0:               nfy = (x[node_ids[i+1]] - x[nid])            / 2.0 * fy
                    elif i == len(node_ids)-1: nfy = (x[nid] - x[node_ids[i-1]])            / 2.0 * fy
                    else:                      nfy = (x[node_ids[i+1]] - x[node_ids[i-1]]) / 2.0 * fy
                if fx != 0:
                    if   i == 0:               nfx = (y[node_ids[i+1]] - y[nid])            / 2.0 * fx
                    elif i == len(node_ids)-1: nfx = (y[nid] - y[node_ids[i-1]])            / 2.0 * fx
                    else:                      nfx = (y[node_ids[i+1]] - y[node_ids[i-1]]) / 2.0 * fx
                if nid in load_dict:
                    load_dict[nid] = (load_dict[nid][0] + nfx, load_dict[nid][1] + nfy)
                else:
                    load_dict[nid] = (nfx, nfy)
        else:                                             # point load
            nid = int(node_val)
            if nid in load_dict:
                load_dict[nid] = (load_dict[nid][0] + fx, load_dict[nid][1] + fy)
            else:
                load_dict[nid] = (fx, fy)
    return load_dict


def build_system(nd, elems, ld, sup, E=1.0):
    """
    Assemble and solve the FEA system.
    Returns (u, Ls_cm, dirs) or (None, None, None) if singular / zero-length member.
    Ls_cm contains member lengths in the same unit as the node coordinates (cm).
    """
    ndof = 2 * len(nd)
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)
    Ls, dirs = [], []

    for (i, j) in elems.values():
        xi, yi = nd[i]; xj, yj = nd[j]
        L = dist((xi, yi), (xj, yj))
        if L == 0:
            return None, None, None
        cx, cy = (xj - xi) / L, (yj - yi) / L
        k = (E / L) * np.array([[ cx*cx,  cx*cy, -cx*cx, -cx*cy],
                                 [ cx*cy,  cy*cy, -cx*cy, -cy*cy],
                                 [-cx*cx, -cx*cy,  cx*cx,  cx*cy],
                                 [-cx*cy, -cy*cy,  cx*cy,  cy*cy]])
        dof = [2*i, 2*i+1, 2*j, 2*j+1]
        K[np.ix_(dof, dof)] += k
        Ls.append(L); dirs.append((cx, cy))

    for n, (fx, fy) in ld.items():
        F[2*n] += fx; F[2*n+1] += fy

    fixed = ([2*n   for n, (fx, _) in sup.items() if fx] +
             [2*n+1 for n, (_, fy) in sup.items() if fy])
    free  = [d for d in range(ndof) if d not in fixed]

    try:
        u = np.zeros(ndof)
        u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
        return u, Ls, dirs
    except np.linalg.LinAlgError:
        return None, None, None


def member_forces(nd, elems, u, Ls, dirs, E=1.0):
    """Extract axial force in each member from displacement vector u."""
    N = []
    for eid, (i, j) in elems.items():
        cx, cy = dirs[eid]; L = Ls[eid]
        dof = [2*i, 2*i+1, 2*j, 2*j+1]
        N.append((E / L) * np.dot([-cx, -cy, cx, cy], u[dof]))
    return np.array(N)


# ═══════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════════

def is_valid(nodes_df, min_dist=0.1):
    """Return False if any two nodes are closer than min_dist cm."""
    pos = nodes_df[['x', 'y']].to_numpy()
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if dist(pos[i], pos[j]) < min_dist:
                return False
    return True


def validate_geometry(nd, sup):
    """
    Check physical project constraints.
    Returns (violations: list[str], length_cm: float, height_cm: float).
    """
    violations = []

    support_xs = [nd[n][0] for n, (fx, fy) in sup.items() if fy]
    if len(support_xs) < 2:
        violations.append("  !! Could not identify two y-fixed support nodes for span check.")
        length_cm = 0.0
    else:
        length_cm = max(support_xs) - min(support_xs)
        if not (LENGTH_MIN_CM <= length_cm <= LENGTH_MAX_CM):
            violations.append(
                f"  !! Span = {length_cm:.2f} cm  "
                f"(required {LENGTH_MIN_CM}-{LENGTH_MAX_CM} cm)")

    all_y     = [y for (x, y) in nd.values()]
    height_cm = max(all_y) - min(all_y)
    if not (HEIGHT_MIN_CM <= height_cm <= HEIGHT_MAX_CM):
        violations.append(
            f"  !! Height = {height_cm:.2f} cm  "
            f"(required {HEIGHT_MIN_CM}-{HEIGHT_MAX_CM} cm)")

    violations.append(
        f"  !  Width ({WIDTH_MIN_CM}-{WIDTH_MAX_CM} cm) cannot be checked in 2-D model"
        f" -- verify physically.")

    return violations, length_cm, height_cm


def scale_to_target_height(nd, target_cm):
    """
    Proportionally rescale all y-coordinates so the truss height equals target_cm.
    x-positions (span, panel spacing) are unchanged.
    Used by truss_solver.py to sweep heights without editing nodes.csv.
    """
    all_y  = [y for (x, y) in nd.values()]
    h_cur  = max(all_y) - min(all_y)
    y_min  = min(all_y)
    if h_cur == 0:
        return nd
    scale = target_cm / h_cur
    return {n: (x, y_min + (y - y_min) * scale) for n, (x, y) in nd.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Physical failure capacities
# ═══════════════════════════════════════════════════════════════════════════════

def member_capacities(L_cm, I_mm4=None):
    """
    Return (max_tension_N, max_compression_N) for a member of length L_cm.

    Tension
    -------
    MAX_TENSION_N from options.json -- a single measured/calculated force limit
    that already accounts for whichever failure mode governs (normal stress rupture
    or bearing/crushing at the pin joint).  Same value for every member.

    Compression
    -----------
    min( Euler buckling load,  MAX_COMPRESSION_BEARING_N )

      Euler buckling (pin-pin, K = 1):
          P_cr = pi^2 * E * I / L^2

      where:
          E  = BALSA_E_MPA         (MPa = N/mm2)  -- set in options.json as "E_mpa"
          I  = I_mm4 parameter     (mm4)           -- per-member value from members.csv
                                                      falls back to SECTION_I_MM4 if None
          L  = member length       (mm)

      bearing = MAX_COMPRESSION_BEARING_N (uniform joint cap)

    The buckling limit is length-dependent; bearing is the same for all members.
    """
    import math
    I = I_mm4 if I_mm4 is not None else SECTION_I_MM4
    L_mm = L_cm * 10           # coordinates are in cm; formula needs mm
    buckling_N        = (math.pi**2 * BALSA_E_MPA * I) / (L_mm**2)
    max_compression_N = min(buckling_N, MAX_COMPRESSION_BEARING_N)
    return MAX_TENSION_N, max_compression_N


# ═══════════════════════════════════════════════════════════════════════════════
# Full PV computation  (called by both solver and adjuster)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pv(node_dict, elements, load_dict, supports, members_df=None,
               target_height_cm=None, enforce_geometry=True):
    """
    Run FEA, compute per-member capacities, and return the Performance Value.

    Failure modes considered
    ------------------------
    cap_T and cap_C are read directly from members.csv (Max_Tension_N and
    Max_Compression_N columns).  Those values should already reflect all relevant
    failure modes (normal stress, bearing, buckling, etc.) as determined from lab
    results or manual calculation.  member_capacities() is used only as a fallback
    when a column is absent or a cell is NaN.

    Parameters
    ----------
    target_height_cm : if set, y-coordinates are scaled before analysis
                       (solver uses this to sweep heights; adjuster passes None)
    enforce_geometry : if True, return None on constraint violation (adjuster mode)

    Returns
    -------
    dict or None
    """
    nd = node_dict
    if target_height_cm is not None:
        nd = scale_to_target_height(nd, target_height_cm)

    u, Ls, dirs = build_system(nd, elements, load_dict, supports)
    if u is None:
        return None

    # Member length bounds (min_dist / max_dist in cm)
    if members_df is not None:
        for eid, row in enumerate(members_df.itertuples()):
            min_d = getattr(row, 'min_dist', float('nan'))
            max_d = getattr(row, 'max_dist', float('nan'))
            if not pd.isna(min_d) and Ls[eid] < float(min_d):
                return None
            if not pd.isna(max_d) and Ls[eid] > float(max_d):
                return None

    forces = member_forces(nd, elements, u, Ls, dirs)
    if np.any(np.isnan(forces)) or np.any(np.isinf(forces)):
        return None

    # Per-member section areas
    areas = []
    for eid in range(len(elements)):
        a = SECTION_A_MM2
        if members_df is not None and 'Section_A_mm2' in members_df.columns:
            val = members_df.iloc[eid]['Section_A_mm2']
            if not pd.isna(val):
                a = float(val)
        areas.append(a)

    # Per-member limits from members.csv columns (Max_Tension_N, Max_Compression_N).
    # Falls back to member_capacities() (computed from material constants) when the
    # column is absent or the value is NaN.
    has_T = members_df is not None and 'Max_Tension_N'     in members_df.columns
    has_C = members_df is not None and 'Max_Compression_N' in members_df.columns
    has_I = members_df is not None and 'I_mm4'             in members_df.columns
    cap_T, cap_C = [], []
    for eid in range(len(elements)):
        I_mm4 = None
        if has_I:
            val = members_df.iloc[eid]['I_mm4']
            if not pd.isna(val):
                I_mm4 = float(val)
        t_default, c_default = member_capacities(Ls[eid], I_mm4)

        if has_T:
            val = members_df.iloc[eid]['Max_Tension_N']
            t = float(val) if not pd.isna(val) else t_default
        else:
            t = t_default

        if has_C:
            val = members_df.iloc[eid]['Max_Compression_N']
            c = float(val) if not pd.isna(val) else c_default
        else:
            c = c_default

        cap_T.append(t); cap_C.append(c)

    violations, length_cm, height_cm = validate_geometry(nd, supports)
    geom_ok = all(not v.startswith('  !!') for v in violations)
    if enforce_geometry and not geom_ok:
        return None

    # Reference load (N)
    ref_load_N = sum(abs(fy) for (fx, fy) in load_dict.values())

    # cap_T and cap_C already incorporate all failure modes; use directly
    multipliers = []
    for eid, f in enumerate(forces):
        if abs(f) < 1e-12:
            continue
        cap = cap_T[eid] if f > 0 else cap_C[eid]
        multipliers.append(cap / abs(f))

    load_multiplier = min(multipliers) if multipliers else 0.0
    max_load_N      = ref_load_N * load_multiplier

    # Mass: A [mm2] x L [mm] x density [kg/m3] x faces,  converted mm3 -> m3 via 1e-9
    total_vol_mm3 = sum(areas[eid] * Ls[eid] * 10 for eid in range(len(elements)))
    mass_kg       = (total_vol_mm3 * 1e-9) * BALSA_DENSITY_KG_M3 * NUM_FACES

    pv = max_load_N / mass_kg if mass_kg > 0 else 0.0

    return {
        'pv':              pv,
        'max_load_N':      max_load_N,
        'mass_kg':         mass_kg,
        'load_multiplier': load_multiplier,
        'ref_load_N':      ref_load_N,
        'forces':          forces,
        'cap_T':           cap_T,     # tension limit per member (N), from members.csv
        'cap_C':           cap_C,     # compression limit per member (N), from members.csv
        'areas':           areas,
        'Ls':              Ls,              # member lengths in cm
        'dirs':            dirs,
        'node_dict':       nd,
        'violations':      violations,
        'length_cm':       length_cm,
        'height_cm':       height_cm,
        'geom_ok':         geom_ok,
    }
