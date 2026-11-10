import collections, statistics as st, sys, os
d, scen = sys.argv[1], sys.argv[2]
insp={}
for l in open(os.path.join(d,'inspect.txt')).readlines()[1:]:
    f=l.split(); insp[f[0]]=(int(f[6]),int(f[7]))
rows=collections.defaultdict(list)
for p in (1,2,3):
    for l in open(os.path.join(d,f'bench_{scen}_{p}.txt')):
        f=l.split()
        if len(f)==9 and f[0].startswith('O'): rows[(f[0],f[1],f[2])].append(float(f[4]))
def cls(op,route,var):
    if route!='d': return 'Simd' if route=='c' else 'Scalar'
    sym=f"u64x64_{'o6' if op=='O6c' else op.lower()}_d_{var}"
    v,s=insp[sym]; return 'Simd' if v>s else 'Scalar'
for op in sorted({k[0] for k in rows}):
    ents={k:v for k,v in rows.items() if k[0]==op}
    best=min(ents,key=lambda k:st.median(ents[k])); bc=cls(*best)
    others=[k for k in ents if cls(*k)!=bc]
    a0=ents.get((op,'a0','-'))
    a0s=f"  A0 {st.median(a0):.1f}" if a0 else ""
    if not others: print(f"{op:5} only {bc}: best {best[1]}/{best[2]} {st.median(ents[best]):.1f} ns{a0s}"); continue
    r=min(others,key=lambda k:st.median(ents[k])); W,R=ents[best],ents[r]
    win=max(W)<min(R) and st.median(W)<=0.85*st.median(R)
    tag=f"{bc.upper()}-WINS ({best[1]}/{best[2]})" if win else "NO-MEASURED-ADVANTAGE"
    print(f"{op:5} W={best[1]}/{best[2]} [{bc}] {st.median(W):.1f} [{min(W):.1f}..{max(W):.1f}]  R={r[1]}/{r[2]} [{cls(*r)}] {st.median(R):.1f} [{min(R):.1f}..{max(R):.1f}]  {st.median(W)/st.median(R):.2f}  {tag}{a0s}")
