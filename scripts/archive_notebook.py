"""Her koşunun kullandığı notebook sürümünü git geçmişinden çıkarıp notebooks/runs/ altına koyar.
Kurulum hücresine o koşunun commit'i sabitlenir (aynı kodla tekrar üretilebilsin).

Kullanım (repo kökünden):  python scripts/archive_notebook.py <koşu>:<notebook_commit>:<kod_commit> ...
"""
import json
import os
import subprocess
import sys

RUNS = [tuple(a.split(":")) for a in sys.argv[1:]]  # koşu:notebook_sürümü:kod_commit'i
os.makedirs("notebooks/runs", exist_ok=True)
for run, nb_rev, code_rev in RUNS:
    raw = subprocess.run(["git", "show", f"{nb_rev}:notebooks/train.ipynb"], capture_output=True, check=True).stdout
    nb = json.loads(raw.decode("utf-8"))
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "code" and "git clone" in src:
            src = src.replace("!git log --oneline -1", f"!git -C /content/proje checkout -q {code_rev}   # bu koşunun kod sürümü\n!git log --oneline -1")
            c["source"] = src.splitlines(keepends=True)
        if c["cell_type"] == "code" and src.startswith("# Bütün koşular"):
            src = src.replace(src[src.index("RUN = "):src.index("\nSET")], f'RUN = "{run}"')
            c["source"] = src.splitlines(keepends=True)
        if c["cell_type"] == "markdown" and src.startswith("# Drone Chase Tag"):
            src += (f"\n\n**Arşiv kopyası:** `{run}` koşusunda kullanılan notebook. Kod `{code_rev}` sürümüne "
                    f"sabitlendi; sonuçlar için `experiments/LOG.md`.")
            c["source"] = src.splitlines(keepends=True)
    path = f"notebooks/runs/{run}.ipynb"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(path)
