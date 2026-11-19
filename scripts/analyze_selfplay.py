"""Self-play koşusunun nesil nesil analizi (sadece metrics.jsonl okunur)."""
import json
import sys

import numpy as np

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
gens = sorted(set(r["gen"] for r in rows))
for g in gens:
    R = [r for r in rows if r["gen"] == g]
    role = R[0]["learner"]

    def blockstats(rs, which):
        """which='latest' (ilk yarı bloklar) ya da 'pfsp' (ikinci yarı): (ebeleme%, süre%, maç)"""
        B = len(rs[0]["block_counts"])
        sl = slice(0, B // 2) if which == "latest" else slice(B // 2, B)
        c = np.sum([np.array(r["block_counts"])[sl].sum(0) for r in rs], 0)
        tot = c[1:].sum()
        return 100 * c[1] / max(tot, 1), 100 * c[2] / max(tot, 1), int(tot)

    first, last = R[:5], R[-5:]
    lat_f, lat_l = blockstats(first, "latest"), blockstats(last, "latest")
    pf_l = blockstats(last, "pfsp")
    len_f = np.mean([r["ep_len"] for r in first]) / 50
    len_l = np.mean([r["ep_len"] for r in last]) / 50
    ema = R[-1]["ema"]
    print(f"nesil {g:2d} {role:6s} | maç süresi {len_f:4.1f} → {len_l:4.1f} s | en son rakip: ebeleme %{lat_f[0]:5.1f} → %{lat_l[0]:5.1f}"
          f" | PFSP bloklar (son 5 tur): ebeleme %{pf_l[0]:5.1f} | std {R[-1]['std']:.2f} ev {R[-1]['explained_var']:.2f}"
          f" kl {np.mean([r['approx_kl'] for r in last]):.4f} | havuza karşı kazanma: {[round(x, 2) for x in ema]}")
