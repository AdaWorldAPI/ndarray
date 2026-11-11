import collections, statistics as st, sys, os
d, scen = sys.argv[1], sys.argv[2]
insp={}
for l in open(os.path.join(d,'inspect.txt')).readlines()[1:]:
    f=l.split(); insp[f[0]]=(int(f[6]),int(f[7]))
rows=collections.defaultdict(list)
fixed={}
for p in (1,2,3):
    for l in open(os.path.join(d,f'bench_{scen}_{p}.txt')):
        f=l.split()
        if len(f)==9 and f[0].startswith('O'): rows[(f[0],f[1],f[2])].append(float(f[4]))
        elif len(f)==9 and f[3]=='Fixed': fixed.setdefault(f[0],[]).append(float(f[4]))
def cls(op,route,var):
    if route!='d': return {'c':'Simd','b':'Bitslice'}.get(route,'Scalar')
    sym=f"u64x64_{'o6' if op=='O6c' else op.lower()}_d_{var}"
    v,s=insp[sym]; return 'Simd' if v>s else 'Scalar'
for op in sorted({k[0] for k in rows}):
    # A0 is an independent baseline (plan §3), never a candidate or reference.
    ents={k:v for k,v in rows.items() if k[0]==op and k[1]!='a0'}
    best=min(ents,key=lambda k:st.median(ents[k])); bc=cls(*best)
    others=[k for k in ents if cls(*k)!=bc]
    a0=rows.get((op,'a0','-'))
    a0s=f"  A0 {st.median(a0):.1f}" if a0 else ""
    if not others: print(f"{op:5} only {bc}: best {best[1]}/{best[2]} {st.median(ents[best]):.1f} ns{a0s}"); continue
    r=min(others,key=lambda k:st.median(ents[k])); W,R=ents[best],ents[r]
    win=max(W)<min(R) and st.median(W)<=0.85*st.median(R)
    tag=f"{bc.upper()}-WINS ({best[1]}/{best[2]})" if win else "NO-MEASURED-ADVANTAGE"
    print(f"{op:5} W={best[1]}/{best[2]} [{bc}] {st.median(W):.1f} [{min(W):.1f}..{max(W):.1f}]  R={r[1]}/{r[2]} [{cls(*r)}] {st.median(R):.1f} [{min(R):.1f}..{max(R):.1f}]  {st.median(W)/st.median(R):.2f}  {tag}{a0s}")

# Prepared reuse (plan §8): E's amortised cost per query over N reuses
# against the best unprepared entry for the same op; N* is the smallest N
# that passes max(W) < min(R) and median(W) <= 0.85 median(R).
def amort(prep, first, rest, n): return (prep + first + (n - 1) * rest) / n
if 'E_PREP_E2' in fixed:
    P=fixed['E_PREP_E2']
    for name, qrow, hit, op in (('O2 E2', 'E_O2_E2', 'E_O2_E2', 'O2'), ('O6 E2', 'E_O6_E2', 'E_O6_E2', 'O6'), ('O6 E2+memo (identical query)', 'E_O6_E2', 'E_O6_MEMO_HIT', 'O6')):
        ents={k:v for k,v in rows.items() if k[0]==op and k[1]!='a0'}
        r=min(ents,key=lambda k:st.median(ents[k])); R=ents[r]
        out=[]; nstar=None
        for n in (1,8,128,1024):
            W=[amort(P[i],fixed[qrow][i],fixed[hit][i],n) for i in range(len(P))]
            ok=max(W)<min(R) and st.median(W)<=0.85*st.median(R)
            out.append(f"N={n}: {st.median(W):.1f}{' PASS' if ok else ''}")
            if ok and nstar is None: nstar=n
        print(f"{name:30} vs {r[1]}/{r[2]} {st.median(R):.1f}: " + "  ".join(out) + f"  -> {'PREPARED-REUSE-WINS from N=' + str(nstar) if nstar else 'never (N<=1024)'}")
