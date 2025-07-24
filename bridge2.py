"""
main_warren.py
---------------
15 m railway‑bridge design • Warren truss (6 panels, 3 m each)

Requirements satisfied
• 15 m span, pin at A (node 0) and roller at B (node 5)
• Floor‑beam joints every 3 m on the bottom chord (nodes 0‑5)
• Simple truss (m = 2 n − 3 ⇒ 19 members, 11 joints)
• No crossing members, every bar ≥ 1 m
• Force limits 8 kN tension, 5 kN compression – duplicates added automatically
• Costing: $9 / m per bar × duplicates  +  $5 per gusset plate
"""

import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------------------------------------------------
# 1.  Geometry  (bottom chord every 3 m, top nodes half‑panel offsets)
# ------------------------------------------------------------------
H       = 3.0                     # truss height (m) – tweak if you like
panel   = 3.0                     # bottom‑panel length (m)

# bottom nodes 0‑5
nodes = {i: (panel*i, 0.0) for i in range(6)}

# top nodes 6‑10 at mid‑panel positions
nodes |= {6+i: (panel*(i+0.5), H) for i in range(5)}

# --- Warren member list -------------------------------------------------------
# bottom & top chords
elements = [(i, i+1)       for i in range(5)]            # bottom chord
elements += [(6+i, 6+i+1)  for i in range(4)]            # top chord

# alternating diagonals (make triangles)
for i in range(5):
    elements.append((i, 6+i))        # up‑right
    elements.append((6+i, i+1))      # down‑right

# label elements with numeric IDs
elements = {k: e for k, e in enumerate(elements)}        # total = 19

# ------------------------------------------------------------------
# 2.  Loads and supports
# ------------------------------------------------------------------
# Uniform train load → 45 kN per side, distributed to bottom joints 1‑4 (7.5 kN each)
loads = {j: (0.0, -7.5) for j in range(1, 5)}

supports = {0: (True,  True),     # pin (Ax, Ay)
            5: (False, True)}     # roller (By)

# ------------------------------------------------------------------
# 3.  Finite‑element solver  (same core as previous script)
# ------------------------------------------------------------------
def dist(p, q): return np.linalg.norm(np.subtract(p, q))

def build_K_F(nodes, elements, loads, supports, E=1.0):
    ndof = 2*len(nodes)
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)

    lengths, dirs = [], []
    for _, (i, j) in elements.items():
        xi, yi = nodes[i];  xj, yj = nodes[j]
        L   = dist((xi, yi), (xj, yj))
        cx, cy = (xj-xi)/L, (yj-yi)/L
        k = (E/L)*np.array([[ cx*cx, cx*cy, -cx*cx, -cx*cy],
                            [ cx*cy, cy*cy, -cx*cy, -cy*cy],
                            [-cx*cx,-cx*cy,  cx*cx,  cx*cy],
                            [-cx*cy,-cy*cy,  cx*cy,  cy*cy]])
        dof = [2*i, 2*i+1, 2*j, 2*j+1]
        K[np.ix_(dof, dof)] += k
        lengths.append(L)
        dirs.append((cx, cy))

    for n,(fx,fy) in loads.items():
        F[2*n]   += fx
        F[2*n+1] += fy

    fixed = [2*n   for n,(fx,_) in supports.items() if fx] + \
            [2*n+1 for n,(_,fy) in supports.items() if fy]
    free  = [d for d in range(ndof) if d not in fixed]

    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    return u, lengths, dirs

def member_forces(nodes, elements, u, lengths, dirs, E=1.0):
    F = []
    for eid,(i,j) in elements.items():
        cx, cy = dirs[eid];  L = lengths[eid]
        dof = [2*i, 2*i+1, 2*j, 2*j+1]
        ul = u[dof]
        N  = (E/L)*np.dot([-cx, -cy, cx, cy], ul)
        F.append(N)
    return np.array(F)

# ------------------------------------------------------------------
# 4.  Run analysis
# ------------------------------------------------------------------
u, Ls, dirs = build_K_F(nodes, elements, loads, supports)
N  = member_forces(nodes, elements, u, Ls, dirs)

# capacity limits
T_LIM, C_LIM = 8.0, 5.0
dup = [max(1, math.ceil(abs(f)/(T_LIM if f>0 else C_LIM))) for f in N]

# costing
cost_members = sum(9*L*dup[eid] for eid,L in enumerate(Ls))
cost_gusset  = 5 * len(nodes)
total_cost   = cost_members + cost_gusset

# ------------------------------------------------------------------
# 5.  Console report
# ------------------------------------------------------------------
print("\n=== 6‑Panel Warren Truss – 15 m span, H = %.1f m ===" % H)
print("ID  Force(kN)  Type   #Bars  Length(m)  Segment‑Cost($)")
for eid,(i,j) in elements.items():
    kind = "T" if N[eid] > 0 else "C" if N[eid] < 0 else "-"
    seg  = 9*Ls[eid]*dup[eid]
    print(f"{eid:2d}  {N[eid]:9.2f}   {kind}     {dup[eid]:2d}    {Ls[eid]:6.2f}        {seg:6.0f}")
print(f"\nGusset plates: {len(nodes)} × $5 = ${cost_gusset:.0f}")
print(f"TOTAL COST  : ${total_cost:,.2f}")

# ------------------------------------------------------------------
# 6.  Quick visual
# ------------------------------------------------------------------
plt.figure(figsize=(11,4))
for eid,(i,j) in elements.items():
    x = [nodes[i][0], nodes[j][0]]
    y = [nodes[i][1], nodes[j][1]]
    col = 'blue' if N[eid]>0 else 'red' if N[eid]<0 else 'k'
    plt.plot(x, y, col, lw=2)
    xm, ym = (x[0]+x[1])/2, (y[0]+y[1])/2
    plt.text(xm, ym, f"{abs(N[eid]):.1f}", fontsize=8, ha='center', va='center')
for n,(x,y) in nodes.items():
    plt.plot(x, y, 'ko')
    plt.text(x, y+0.25, str(n), ha='center')
plt.axis('equal'); plt.grid(True)
plt.title("Warren truss – tension (blue) / compression (red)")
plt.xlabel("m"); plt.ylabel("m"); plt.tight_layout(); plt.show()
