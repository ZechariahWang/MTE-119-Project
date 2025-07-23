import numpy as np
import matplotlib.pyplot as plt
import math

# -----------------------------
# 1.  Geometry (6–panel Pratt)
# -----------------------------
H = 3.5               # m   ⇐ chosen by parametric sweep
panel = 3.0           # m
n_panels = 5          # 6 bottom joints ⇒ 5 panels

# bottom joints 0‑5, top joints 6‑11
nodes = {i: (panel*i, 0.0)             for i in range(6)}
nodes |= {6+i: (panel*i,  H)           for i in range(6)}

elements = []                              # (start, end)
# bottom & top chords
elements += [(i, i+1)        for i in range(5)]
elements += [(6+i, 6+i+1)    for i in range(5)]
# verticals
elements += [(i, 6+i)        for i in range(6)]
# Pratt diagonals (up‑right)
elements += [(i, 6+i+1)      for i in range(5)]

# convenience: dict keyed by ID
elements = {i: e for i, e in enumerate(elements)}

# -----------------------------
# 2.  Loads & supports
# -----------------------------
# Uniform train load:  6 kN / m × 15 m / 2 sides = 45 kN per truss
# Floor beams every 3 m → nodes 1‑4 carry 9 kN each
loads = {0: (0.0, -4.5), 5: (0.0, -4.5)}
loads.update({i: (0.0, -9) for i in range(1, 5)})    # (Fx, Fy) in kN

supports = {0: (True, True),     # A: pin
            5: (False, True)}    # B: roller (vertical only)

# -----------------------------
# 3.  FEA helper
# -----------------------------
def dist(p, q): return np.linalg.norm(np.subtract(p, q))

def build_K_F(nodes, elements, loads, supports, E=1.0):
    ndof = 2*len(nodes)
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)
    lengths, dirs = [], []

    for eid, (i, j) in elements.items():
        xi, yi = nodes[i];  xj, yj = nodes[j]
        L   = dist((xi, yi), (xj, yj))
        cx, cy = (xj-xi)/L, (yj-yi)/L
        k = (E/L) * np.array([[ cx*cx, cx*cy, -cx*cx, -cx*cy],
                              [ cx*cy, cy*cy, -cx*cy, -cy*cy],
                              [-cx*cx,-cx*cy,  cx*cx,  cx*cy],
                              [-cx*cy,-cy*cy,  cx*cy,  cy*cy]])
        mapdof = [2*i, 2*i+1, 2*j, 2*j+1]
        K[np.ix_(mapdof, mapdof)] += k
        lengths.append(L)
        dirs.append((cx, cy))

    for n, (fx, fy) in loads.items():
        F[2*n]   += fx
        F[2*n+1] += fy

    fixed = [2*n for n,(fx,_) in supports.items() if fx] + \
            [2*n+1 for n,(_,fy) in supports.items() if fy]
    free  = [d for d in range(ndof) if d not in fixed]

    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    return u, lengths, dirs

def member_forces(nodes, elements, u, lengths, dirs, E=1.0):
    forces = []
    for eid, (i, j) in elements.items():
        cx, cy = dirs[eid];  L = lengths[eid]
        dof = [2*i, 2*i+1, 2*j, 2*j+1]
        ul = u[dof]
        N  = (E/L) * np.dot([-cx, -cy, cx, cy], ul)
        forces.append(N)
    return np.array(forces)         # kN

# -----------------------------
# 4.  Analysis
# -----------------------------
u, Ls, dirs = build_K_F(nodes, elements, loads, supports)
N  = member_forces(nodes, elements, u, Ls, dirs)

# capacity check (duplicates where needed)
T_LIMIT, C_LIMIT = 8.0, 5.0        # kN
duplicates = [max(1, math.ceil(abs(f)/(T_LIMIT if f>0 else C_LIMIT)))
              for f in N]

# -----------------------------
# 5.  Bill‑of‑Materials & cost
# -----------------------------
cost_members = sum(9 * Ls[eid] * dup for eid, dup in enumerate(duplicates))
print([9 * Ls[eid] * dup for eid, dup in enumerate(duplicates)])
cost_gusset  = 5 * len(nodes)
total_cost   = cost_members + cost_gusset

print("\n=== Optimal 15 m Railway‑Bridge Truss (Height 3.5 m) ===")
print("Member  Force(kN)  Type  Duplicates  Length(m)  Segment‑Cost($)")
for eid, (i,j) in elements.items():
    typ = "T" if N[eid]>0 else "C" if N[eid]<0 else "-"
    seg_cost = 9 * Ls[eid] * duplicates[eid]
    print(f"{eid:3d}:  {N[eid]:8.2f}   {typ}      {duplicates[eid]:2d}        "
          f"{Ls[eid]:5.2f}        {seg_cost:6.0f}")

print(f"\nGusset plates: {len(nodes)} × $5 = ${cost_gusset:4.0f}")
print(f"TOTAL COST  : ${total_cost:,.2f}")

# -----------------------------
# 6.  Quick plot
# -----------------------------
fig, ax1 = plt.subplots(figsize=(11,4))
ax1.set_ylim(-2, H+1)
for eid,(i,j) in elements.items():
    x = [nodes[i][0], nodes[j][0]]
    y = [nodes[i][1], nodes[j][1]]
    col = 'blue' if N[eid]>0 else 'red' if N[eid]<0 else 'black'
    lw  = 2
    ax1.plot(x, y, col, lw=lw)
    xm, ym = (x[0]+x[1])/2, (y[0]+y[1])/2
    ax1.text(xm, ym, f"{abs(N[eid]):.1f}", fontsize=8,
             ha='center', va='center')
for n,(x,y) in nodes.items():
    ax1.plot(x, y, 'ko'); plt.text(x, y+0.3, str(n), ha='center')
ax1.set_xlabel("m")
ax1.set_ylabel("m")
ax1.tick_params(axis='y', labelcolor='blue')
ax2 = ax1.twinx()
for n,(fx, fy) in loads.items():
    ax2.arrow(nodes[n][0], nodes[n][1], fx, fy, head_width=0.1,
              head_length=0.2, fc='green', ec='green')
plt.grid(True)
plt.title("Pratt Truss – tension (blue) / compression (red)")
ax2.set_ylabel("kN")
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(ax1.get_ylim()[0] * 5, ax1.get_ylim()[1] * 5)
plt.tight_layout()
plt.show()
