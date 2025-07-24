"""
main_howe.py
-------------
15 m railway‑bridge • Howe truss (6 panels × 3 m)

• Span 15 m, pin at A (node 0) and roller at B (node 5)
• Floor‑beam joints every 3 m on the bottom chord (nodes 0‑5)
• Simple truss: 21 members, 12 joints  (m = 2 n − 3)
• No crossing members, every bar ≥ 1 m
• Force limits: 8 kN tension, 5 kN compression – duplicates added automatically
• Costing: $9 / m per bar × duplicates  +  $5 per gusset plate
"""

import numpy as np, matplotlib.pyplot as plt, math

# ------------------------------------------------------------------
# 1.  Geometry  (bottom chord every 3 m, top nodes over joints)
# ------------------------------------------------------------------
H       = 3.5                     # truss height (m) – tweak freely
panel   = 3.0

# bottom nodes 0‑5
nodes = {i: (panel*i, 0.0) for i in range(6)}
# top nodes 6‑11 directly above bottoms
nodes |= {6+i: (panel*i, H) for i in range(6)}

# ----- Howe member list -------------------------------------------
elements = []
# bottom & top chords
elements += [(i, i+1)       for i in range(5)]
elements += [(6+i, 6+i+1)   for i in range(5)]
# verticals
elements += [(i, 6+i)       for i in range(6)]
# Howe diagonals (down‑right)
elements += [(6+i, i+1)     for i in range(5)]

elements = {k: e for k, e in enumerate(elements)}        # 21 total

# ------------------------------------------------------------------
# 2.  Loads and supports
# ------------------------------------------------------------------
loads = {j: (0.0, -7.5) for j in range(1, 5)}             # 7.5 kN at nodes 1‑4
supports = {0: (True,  True), 5: (False, True)}           # pin & roller

# ------------------------------------------------------------------
# 3.  Finite‑element solver (unchanged core)
# ------------------------------------------------------------------
def dist(p, q): return np.hypot(*(np.subtract(p, q)))

def build_K_F(nodes, elements, loads, supports, E=1.0):
    ndof = 2*len(nodes);  K = np.zeros((ndof, ndof));  F = np.zeros(ndof)
    Ls, dirs = [], []
    for (i,j) in elements.values():
        xi,yi = nodes[i];  xj,yj = nodes[j]
        L  = dist((xi,yi),(xj,yj));   cx,cy = (xj-xi)/L, (yj-yi)/L
        k = (E/L)*np.array([[ cx*cx, cx*cy, -cx*cx, -cx*cy],
                            [ cx*cy, cy*cy, -cx*cy, -cy*cy],
                            [-cx*cx,-cx*cy,  cx*cx,  cx*cy],
                            [-cx*cy,-cy*cy,  cx*cy,  cy*cy]])
        dof = [2*i,2*i+1,2*j,2*j+1];  K[np.ix_(dof,dof)] += k
        Ls.append(L); dirs.append((cx,cy))
    for n,(fx,fy) in loads.items(): F[2*n]+=fx; F[2*n+1]+=fy
    fixed = [2*n   for n,(fx,_) in supports.items() if fx] + \
            [2*n+1 for n,(_,fy) in supports.items() if fy]
    free = [d for d in range(ndof) if d not in fixed]
    u = np.zeros(ndof); u[free] = np.linalg.solve(K[np.ix_(free,free)], F[free])
    return u,Ls,dirs

def member_forces(nodes,elements,u,Ls,dirs,E=1.0):
    N=[]
    for eid,(i,j) in elements.items():
        cx,cy = dirs[eid]; L=Ls[eid]; dof=[2*i,2*i+1,2*j,2*j+1]
        ul=u[dof]; N.append((E/L)*np.dot([-cx,-cy,cx,cy],ul))
    return np.array(N)

# ------------------------------------------------------------------
# 4.  Run analysis
# ------------------------------------------------------------------
u, Ls, dirs = build_K_F(nodes,elements,loads,supports)
N = member_forces(nodes,elements,u,Ls,dirs)

T_LIM, C_LIM = 8.0, 5.0
dup = [max(1, math.ceil(abs(f)/(T_LIM if f>0 else C_LIM))) for f in N]

# costing
cost_members = sum(9*L*dup[eid] for eid,L in enumerate(Ls))
cost_gusset  = 5*len(nodes)
total_cost   = cost_members + cost_gusset

# ------------------------------------------------------------------
# 5.  Console report
# ------------------------------------------------------------------
print("\n=== 6‑Panel Howe Truss – 15 m span, H = %.1f m ===" % H)
print("ID  Force(kN)  Type   #Bars  Length(m)  Segment‑Cost($)")
for eid,(i,j) in elements.items():
    typ = "T" if N[eid]>0 else "C" if N[eid]<0 else "-"
    sc  = 9*Ls[eid]*dup[eid]
    print(f"{eid:2d}  {N[eid]:9.2f}   {typ}     {dup[eid]:2d}    {Ls[eid]:6.2f}        {sc:6.0f}")

print(f"\nGusset plates: {len(nodes)} × $5 = ${cost_gusset:.0f}")
print(f"TOTAL COST  : ${total_cost:,.2f}")

# ------------------------------------------------------------------
# 6.  Quick visual
# ------------------------------------------------------------------
plt.figure(figsize=(11,4))
for eid,(i,j) in elements.items():
    x=[nodes[i][0],nodes[j][0]]; y=[nodes[i][1],nodes[j][1]]
    col='blue' if N[eid]>0 else 'red' if N[eid]<0 else 'k'
    plt.plot(x,y,col,lw=2)
    xm,ym=(x[0]+x[1])/2,(y[0]+y[1])/2
    plt.text(xm,ym,f"{abs(N[eid]):.1f}",fontsize=8,ha='center',va='center')
for n,(x,y) in nodes.items():
    plt.plot(x,y,'ko'); plt.text(x,y+0.25,str(n),ha='center')
plt.axis('equal'); plt.grid(True)
plt.title("Howe truss – tension (blue) / compression (red)")
plt.xlabel("m"); plt.ylabel("m"); plt.tight_layout(); plt.show()
