"""Bir koşunun notebook'unu notebooks/runs/<koşu>.ipynb olarak yazar: şablonun (notebooks/train.ipynb) verilen
sürümünden alınır, RUN o koşuya ayarlanır ve kurulum hücresine koşunun kod commit'i sabitlenir. Böylece her
koşu kendi notebook'undan, kendi kod sürümüyle tekrar çalıştırılabilir. Koşudan ÖNCE de çalıştırılabilir:
kullanıcı koşuyu bu notebook'tan başlatır.

Kullanım (repo kökünden):  python scripts/archive_notebook.py <koşu>:<notebook_commit>:<kod_commit>[:<şablon>] ...
  şablon varsayılan notebooks/train.ipynb; tarama için notebooks/sweep.ipynb (koşu adı = tarama adı).
Sonra notebooks/README.md'deki listeyi günceller.
"""
import json
import os
import re
import subprocess
import sys

COLAB = "https://colab.research.google.com/github/HasaNSaygili/DroneChaseTag/blob/main/notebooks/runs/"

RUNS = [a.split(":") for a in sys.argv[1:]]  # koşu:notebook_sürümü:kod_commit'i[:şablon]
os.makedirs("notebooks/runs", exist_ok=True)
for parts in RUNS:
    run, nb_rev, code_rev = parts[:3]
    template = parts[3] if len(parts) > 3 else "notebooks/train.ipynb"
    raw = subprocess.run(["git", "show", f"{nb_rev}:{template}"], capture_output=True, check=True).stdout
    nb = json.loads(raw.decode("utf-8"))
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "code" and "git clone" in src:
            src = src.replace("!git log --oneline -1", f"!git -C /content/proje checkout -q {code_rev}   # bu koşunun kod sürümü\n!git log --oneline -1")
            c["source"] = src.splitlines(keepends=True)
        if c["cell_type"] == "code" and src.startswith("# Bütün koşular"):
            src = re.sub(r"^RUN = .*$", f'RUN = "{run}"', src, count=1, flags=re.M)
            c["source"] = src.splitlines(keepends=True)
        if c["cell_type"] == "code" and "SWEEP_NAME = " in src:
            src = re.sub(r"^SWEEP_NAME = .*$", f'SWEEP_NAME = "{run}"', src, count=1, flags=re.M)
            c["source"] = src.splitlines(keepends=True)
        if c["cell_type"] == "markdown" and src.startswith("# Drone Chase Tag"):
            src += (f"\n\n**Bu notebook `{run}` koşusuna ait.** Kod `{code_rev}` sürümüne sabitli; "
                    f"sonuçlar için `experiments/LOG.md`.")
            c["source"] = src.splitlines(keepends=True)
    path = f"notebooks/runs/{run}.ipynb"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(path)

# ---- notebooks/README.md: bütün koşu notebook'larının Colab linkleri ----
runs = sorted(os.path.splitext(f)[0] for f in os.listdir("notebooks/runs") if f.endswith(".ipynb")
              and not f.endswith("_ciktili.ipynb"))
lines = ["# Notebook'lar", "",
         "Her koşunun kendi notebook'u `runs/` altında; kod o koşunun sürümüne sabitli. Colab'da açmak için linke tıkla,",
         "*Çalışma zamanı → A100 GPU* seç, hücreleri sırayla çalıştır. Sonuçlar: `../experiments/LOG.md`.", "",
         "| Koşu | Colab |", "|---|---|"]
lines += [f"| `{r}` | [aç]({COLAB}{r}.ipynb) |" for r in runs]
lines += ["", "Diğerleri: `train.ipynb` (tek koşu şablonu), `sweep.ipynb` (tarama şablonu: birkaç koşu arka arkaya + "
          "karşılaştırma), `elo.ipynb` (bir self-play koşusunun Elo turnuvası, GPU; runs/ altında `elo-<koşu>`), "
          "`speedtest.ipynb` (A100 hız testi).", ""]
with open("notebooks/README.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("notebooks/README.md")
