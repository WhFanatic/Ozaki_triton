"""
Heuristic Autotune for Triton — multi-start coordinate descent
==============================================================
Run 3-point coordinate descent from N random starting points, then duel
all winners to find the global best.  Each start gets max_steps//n_starts
budget.  Mitigates local optima by covering different regions of the space.

Usage:
    @heuristic_autotune(
        param_ranges={'BLOCK_M': [16,32,64,128,256], 'BLOCK_N': [16,32,64,128],
                       'num_warps': [2,4,8], 'num_stages': [1,2,3,4,5]},
        key=['M', 'N', 'K'],
        max_steps=40, n_starts=3,
        out_csv='tune_log.csv',
    )
    @triton.jit
    def my_kernel(...): ...
"""

import os, io, sys, re, inspect, random, triton, csv
from typing import Dict, List

_JIT = {'num_warps', 'num_stages', 'num_ctas'}

def _build(cfg):
    return triton.Config({k: v for k, v in cfg.items() if k not in _JIT},
                         **{k: v for k, v in cfg.items() if k in _JIT})

def _flat(c):
    d = dict(c.kwargs)
    for a in _JIT:
        if hasattr(c, a): d[a] = getattr(c, a)
    return d

class _Tee:
    def __init__(self, o, b): self.o, self.b = o, b
    def write(self, s):  self.o.write(s); self.b.write(s)
    def flush(self):     self.o.flush();  self.b.flush()

def _capture(fn):
    old = os.environ.get('TRITON_PRINT_AUTOTUNING')
    os.environ['TRITON_PRINT_AUTOTUNING'] = '1'
    buf = io.StringIO()
    so, se = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(so, buf), _Tee(se, buf)
    try:     r = fn()
    finally:
        sys.stdout, sys.stderr = so, se
        if old is None: os.environ.pop('TRITON_PRINT_AUTOTUNING', None)
        else:           os.environ['TRITON_PRINT_AUTOTUNING'] = old
    return r, buf.getvalue()

def _parse(text, names):
    m = re.search(r'best config selected:\s*(.+)', text, re.DOTALL)
    if not m: return None
    line = m.group(1)
    d = {}
    for p in names:
        m2 = re.search(rf'{p}\s*[:=]\s*(\d+)', line)
        if m2: d[p] = int(m2.group(1))
    return d if len(d) == len(names) else None

def _from_cache(afn, names):
    try:
        for bc in [getattr(afn, 'best_config', None),
                   *(afn.cache.values() if hasattr(afn, 'cache') else ())]:
            if bc is None: continue
            d = _flat(bc)
            if all(n in d for n in names): return d
    except Exception: pass
    return None

def _key_values(key_names, args, kwargs, fn):
    try:
        sig = inspect.signature(fn.fn if hasattr(fn, 'fn') else fn)
        params = list(sig.parameters.keys())
        bound = {params[i]: v for i, v in enumerate(args) if i < len(params)}
        bound.update(kwargs)
        return tuple(bound.get(k) for k in key_names)
    except Exception:
        return tuple(kwargs.get(k) for k in key_names)

# ── decorator ────────────────────────────────────────────────────────────────

def heuristic_autotune(param_ranges: Dict[str, list], key: List[str],
                       max_steps=60, n_starts=3, reset_to_zero=None,
                       verbose=True, out_csv='tune_log.csv'):
    ranges = {k: sorted(v) for k, v in param_ranges.items()}
    names = list(ranges)

    def decorator(fn):
        result_cache: Dict[tuple, tuple] = {}
        csv_header = key + names

        class Kernel:
            _csv_written = False

            def __getitem__(s, g): s._g = g; return s

            def __call__(s, *args, **kwargs):
                kv = _key_values(key, args, kwargs, fn)
                if kv not in result_cache:
                    best = s._search(args, kwargs, kv)
                    final = triton.autotune(configs=[_build(best)], key=key,
                                            reset_to_zero=reset_to_zero)(fn)
                    result_cache[kv] = (best, final)
                return result_cache[kv][1][s._g](*args, **kwargs)

            def _bench(self, cfgs, args, kw):
                seen, uniq = set(), []
                for c in cfgs:
                    k = frozenset(c.items())
                    if k not in seen: seen.add(k); uniq.append(c)
                afn = triton.autotune(configs=[_build(c) for c in uniq],
                                     key=key, reset_to_zero=reset_to_zero)(fn)
                _, txt = _capture(lambda: afn[self._g](*args, **kw))
                return _parse(txt, names) or _from_cache(afn, names)

            def _rand_start(self):
                return {k: random.choice(v) for k, v in ranges.items()}

            def _coord_descent(self, center, args, kw, budget) -> dict:
                step = 0
                while step < budget:
                    moved = False
                    for p in names:
                        if step >= budget: break
                        vals = ranges[p]
                        ci = vals.index(center[p])
                        idxs = sorted({max(0, min(ci+d, len(vals)-1))
                                       for d in (-1, 0, 1)})
                        trial = [dict(center, **{p: vals[i]}) for i in idxs]

                        if verbose:
                            print(f"    step {step+1}/{budget}  {p}"
                                  f"  [{','.join(str(vals[i]) for i in idxs)}]"
                                  f"  center={center[p]}", file=sys.stderr)

                        w = self._bench(trial, args, kw)
                        step += 1
                        if w is None: continue

                        wi = vals.index(w[p])
                        if wi != ci:
                            center[p] = vals[wi]
                            moved = True
                            if verbose:
                                print(f"      → {w[p]} ← moved", file=sys.stderr)
                        elif verbose:
                            print(f"      → {w[p]} ✓", file=sys.stderr)

                    if not moved:
                        if verbose:
                            print(f"    converged at {center}", file=sys.stderr)
                        break
                return center

            def _search(self, args, kw, kv) -> dict:
                budget = max(max_steps // n_starts, len(names) + 1)
                if verbose:
                    tot = 1
                    for v in ranges.values(): tot *= len(v)
                    print(f"\n{'='*60}\nHeuristic Autotune: {tot} combos, "
                          f"{n_starts} starts × {budget} steps\n{'='*60}",
                          file=sys.stderr)

                # first start from midpoint, rest random
                starts = [{k: v[len(v)//2] for k, v in ranges.items()}]
                starts += [self._rand_start() for _ in range(n_starts - 1)]

                winners = []
                for i, s in enumerate(starts):
                    if verbose:
                        print(f"\n  ── Start {i+1}/{n_starts}: {s}",
                              file=sys.stderr)
                    w = self._coord_descent(dict(s), args, kw, budget)
                    winners.append(w)
                    if verbose:
                        print(f"  ── Winner: {w}", file=sys.stderr)

                # final duel among all winners
                if verbose:
                    print(f"\n  ── Final duel ({len(winners)} candidates)",
                          file=sys.stderr)
                best = self._bench(winners, args, kw)
                if best is None: best = winners[0]
                if verbose:
                    print(f"\n{'='*60}\nBest: {best}\n{'='*60}\n",
                          file=sys.stderr)
                if out_csv:
                    mode = 'w' if not Kernel._csv_written else 'a'
                    with open(out_csv, mode, newline='') as f:
                        w = csv.DictWriter(f, fieldnames=csv_header)
                        if not Kernel._csv_written:
                            w.writeheader()
                            Kernel._csv_written = True
                        row = dict(zip(key, kv))
                        row.update(best)
                        w.writerow(row)
                return best

        return Kernel()
    return decorator