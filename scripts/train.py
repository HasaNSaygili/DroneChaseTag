"""PPO eğitimi başlatır.

Kullanım:
  python scripts/train.py --name ppo-chaser-empty-s0                          # varsayılan config (Colab)
  python scripts/train.py --name test --small                                  # yerel CPU duman testi
  python scripts/train.py --name x --set n_worlds=32768 total_steps=1e9 phase=1 "n_obst=(4,8)"

Çıktı: results/<name>/ altında config.json, metrics.jsonl, checkpoint.pkl, params_*.pkl
"""
import argparse
import ast
import dataclasses

from chase_tag.ppo import PPOConfig, train

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--out", default="results")
p.add_argument("--small", action="store_true", help="yerel CPU'da hızlı deneme boyutu")
p.add_argument("--set", nargs="*", default=[], help="config alanlarını değiştir: alan=değer")
p.add_argument("--no-resume", action="store_true")
a = p.parse_args()

cfg = PPOConfig()
if a.small:
    cfg = dataclasses.replace(cfg, n_worlds=256, n_steps=64, total_steps=256 * 64 * 40, n_arenas=50, log_every=2,
                              ckpt_every=20)
types = {f.name: f.type for f in dataclasses.fields(PPOConfig)}
for kv in a.set:
    k, v = kv.split("=", 1)
    if k not in types:
        raise SystemExit(f"bilinmeyen alan: {k}")
    try:
        val = ast.literal_eval(v)
    except (ValueError, SyntaxError):
        val = v                                   # düz metin (örn. learner=evader)
    if isinstance(getattr(cfg, k), int) and isinstance(val, float):
        val = int(val)                            # total_steps=1e9 gibi yazımlar
    cfg = dataclasses.replace(cfg, **{k: val})

print(cfg)
train(cfg, f"{a.out}/{a.name}", resume=not a.no_resume)
