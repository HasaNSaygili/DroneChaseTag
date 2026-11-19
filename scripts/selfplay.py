"""Self-play eğitimi başlatır (bkz. src/chase_tag/selfplay.py).

Kullanım:
  python scripts/selfplay.py --name sp-empty-s0                          # varsayılan (Colab)
  python scripts/selfplay.py --name test --small                         # yerel CPU duman testi
  python scripts/selfplay.py --name x --set n_gens=10 "n_obst=(4,8)"

Çıktı: results/<name>/ altında config.json, train_log.txt, metrics.jsonl, checkpoint.pkl, <rol>_genNN.pkl
"""
import argparse
import ast
import dataclasses

from chase_tag.selfplay import SelfPlayConfig, train_selfplay

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--out", default="results")
p.add_argument("--small", action="store_true", help="yerel CPU'da hızlı deneme boyutu")
p.add_argument("--set", nargs="*", default=[], help="config alanlarını değiştir: alan=değer")
p.add_argument("--no-resume", action="store_true")
a = p.parse_args()

cfg = SelfPlayConfig()
if a.small:
    b = 64 * 32
    cfg = dataclasses.replace(cfg, n_worlds=64, n_steps=32, n_blocks=4, n_arenas=20, min_steps=3 * b,
                              max_steps=6 * b, n_gens=3, win_window=2, resample_every=2, log_every=1)
for kv in a.set:
    k, v = kv.split("=", 1)
    if not hasattr(cfg, k):
        raise SystemExit(f"bilinmeyen alan: {k}")
    try:
        val = ast.literal_eval(v)
    except (ValueError, SyntaxError):
        val = v
    if isinstance(getattr(cfg, k), int) and isinstance(val, float):
        val = int(val)
    cfg = dataclasses.replace(cfg, **{k: val})

print(cfg)
train_selfplay(cfg, f"{a.out}/{a.name}", resume=not a.no_resume)
