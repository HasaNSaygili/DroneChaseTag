"""Değerlendirme: politika çiftlerini baştan sona maçlarda oynat, Elo (Bradley–Terry) puanı çıkar.

Self-play'de "en son rakibe karşı kazandı" ilerleme demek değildir: taraflar taş-kâğıt-makas gibi bir döngüde de
dönüyor olabilir (kovalayan_5 kaçan_5'i yener ama kaçan_2'ye kaybeder...). Bunu ayırt etmek için BÜTÜN kovalayan
sürümlerini BÜTÜN kaçan sürümleriyle oynatırız (round-robin) ve iki şeye bakarız:
  * kazanma matrisi W[i, j] = kovalayan i'nin kaçan j'ye karşı kazanma oranı (döngü burada görünür),
  * tek bir puan ölçeği: Bradley–Terry modeli P(kovalayan i, kaçan j'yi yener) = σ(r_i − r_j).
    Puanlar bütün sonuçlara birlikte uydurulur (en çok olabilirlik); Elo ölçeğine çevrilir:
    R = 1000 + 400·log10(e)·r, yani 400 puan fark ≈ 10'a 1 kazanma olasılığı, 200 puan ≈ %76.

Neden sıralı Elo güncellemesi değil? Satrançtaki Elo maçlar geldikçe günceller ve sonuç maçların SIRASINA bağlı
olur. Bizde bütün maçlar hazır; hepsine birden uydurmak sırasız ve daha kararlı.

Roller asimetrik (kovalayan ile kaçan farklı oyunlar oynuyor) ama tek ölçek yine anlamlı: sadece kovalayan−kaçan
farkları olasılığa girer. Rol avantajı (örneğin oyun kovalayan lehineyse) iki rolün ortalama puan farkında görünür.
Aynı roldeki iki sürümün puan farkı = "rakip tarafın bütün sürümlerine karşı hangisi daha iyi".
"""
import jax
import jax.numpy as jnp
import numpy as np

from .ppo import actor_dist
from .rules import CHASER, EVADER, MAX_STEPS, OUTCOME_NAMES

ELO_SCALE = 400.0 / np.log(10.0)


def make_pair_runner(env, n_blocks, squash=True, deterministic=False):
    """run(p_ch, p_ev, key) -> (n,) her dünyanın İLK maçının sonucu.

    Dünyalar n_blocks eşit bloğa bölünür; blok b'de kovalayan p_ch[b], kaçan p_ev[b] oynar (parametreler başta
    blok ekseniyle yığılmış). Bütün dünyalar sıfırdan başlar ve tam MAX_STEPS adım oynanır: süre sınırı yüzünden
    her maç en geç son adımda biter, yani her dünya tam bir maç verir (sonrası otomatik yeniden başlar, sayılmaz).
    Böylece kısa ve uzun maçlar eşit sayılır (eğitim maçlarından saymanın senkron başlangıç yanlılığı yok).
    deterministic=False: iki taraf eğitimdeki gibi olasılıklı oynar (politikalar zaten olasılıklı eğitildi)."""
    n = env.n
    assert n % n_blocks == 0, "dünya sayısı blok sayısına bölünmeli"
    bs = n // n_blocks

    def act(params, obs, key):
        mu, std = jax.vmap(lambda p, o: actor_dist(p, o, squash))(params, obs.reshape(n_blocks, bs, -1))
        a = mu if deterministic else mu + std[:, None, :] * jax.random.normal(key, mu.shape)
        return a.reshape(n, 3)

    def run(p_ch, p_ev, key):
        k_reset, k_run = jax.random.split(key)
        s = env.reset(k_reset)

        def body(carry, k):
            s, first = carry
            k_c, k_e, k_env = jax.random.split(k, 3)
            obs = env.observe(s)
            a = jnp.stack([act(p_ch, obs[:, CHASER], k_c), act(p_ev, obs[:, EVADER], k_e)], axis=1)   # (n, 2, 3)
            s, _, _, info = env.step(s, a, k_env)
            first = jnp.where(first == 0, info["outcome"].astype(jnp.int32), first)   # sadece ilk biten maç
            return (s, first), None

        (_, first), _ = jax.lax.scan(body, (s, jnp.zeros(n, jnp.int32)), jax.random.split(k_run, MAX_STEPS))
        return first

    return jax.jit(run)


def stack(params_list):
    """[params, ...] -> blok ekseniyle yığılmış tek pytree."""
    return jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *params_list)


def round_robin(env, chasers, evaders, games, key, squash=True, deterministic=False, log=print):
    """Her kovalayan × her kaçan çiftine `games` maç. Dönüş: outcome sayıları C[i, j, k] (k = sonuç kodu).

    Çiftler env.n // games'lik gruplar halinde aynı anda oynanır (her blok bir çift). Son grup eksikse ilk çift
    tekrarlanarak doldurulur ve o fazlalık atılır (derlenmiş fonksiyon hep aynı boyutta kalsın diye)."""
    n_blocks = env.n // games
    runner = make_pair_runner(env, n_blocks, squash, deterministic)
    pairs = [(i, j) for i in range(len(chasers)) for j in range(len(evaders))]
    C = np.zeros((len(chasers), len(evaders), len(OUTCOME_NAMES)), np.int64)
    for g0 in range(0, len(pairs), n_blocks):
        group = pairs[g0:g0 + n_blocks]
        padded = group + [group[0]] * (n_blocks - len(group))
        key, k = jax.random.split(key)
        first = np.asarray(runner(stack([chasers[i] for i, _ in padded]), stack([evaders[j] for _, j in padded]), k))
        for b, (i, j) in enumerate(group):
            C[i, j] += np.bincount(first[b * games:(b + 1) * games], minlength=len(OUTCOME_NAMES))
        log(f"  {min(g0 + n_blocks, len(pairs))}/{len(pairs)} çift")
    return C


def win_matrix(C, table):
    """Kovalayanın puanı: kazandığı maç 1, kaçanın kazandığı 0, kimsenin kazanmadığı (örn. ikisi çarptı) 0,5.
    table: env.reward_table; bir tarafın ödülü > 0 olan sonuç o tarafın galibiyetidir (WCT kuralında kaçan
    çarparsa kovalayan kazanır; opp_crash_reward=0 iken kimse kazanmaz)."""
    ks = range(1, len(OUTCOME_NAMES))
    ch = [k for k in ks if table[k, CHASER] > 0]
    nobody = [k for k in ks if table[k, CHASER] <= 0 and table[k, EVADER] <= 0]
    wins = C[..., ch].sum(-1) + 0.5 * C[..., nobody].sum(-1)
    return wins, C[..., 1:].sum(-1)


def bradley_terry(wins, games, prior=0.5):
    """Kovalayan ve kaçan puanları (Elo ölçeğinde). wins[i, j]: kovalayan i'nin kaçan j'ye karşı kazandığı maç.

    En çok olabilirlik: P(i, j'yi yener) = σ(r_i − s_j). `prior`: her çifte eklenen sanal yarım galibiyet +
    yarım mağlubiyet. Neden: %100 kazanılan bir çift olursa olabilirlik puanı sonsuza iter; sanal maçlar bunu
    sınırlar (sayı büyükken etkisi yok denecek kadar az). Ölçeğin sıfır noktası serbest: bütün oyuncuların
    ortalaması 1000'e sabitlenir."""
    from scipy.optimize import minimize
    n_c, n_e = wins.shape
    w = wins + prior
    n = games + 2 * prior

    def nll(x):
        r, s = x[:n_c], x[n_c:]
        z = r[:, None] - s[None, :]
        # −log L = Σ n·log(1 + e^z) − w·z   (logaddexp: büyük z'de taşma olmasın)
        f = np.sum(n * np.logaddexp(0.0, z) - w * z)
        g = n * (1.0 / (1.0 + np.exp(-z))) - w          # ∂f/∂z
        return f, np.concatenate([g.sum(1), -g.sum(0)])

    x = minimize(nll, np.zeros(n_c + n_e), jac=True, method="L-BFGS-B").x
    x = x - x.mean()
    return 1000.0 + ELO_SCALE * x[:n_c], 1000.0 + ELO_SCALE * x[n_c:]
