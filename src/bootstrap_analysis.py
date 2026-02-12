import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

# -----------------------
# Parâmetros
# -----------------------
np.random.seed(123)

n_values = range(2, 21)   # dimensões
K = 10000                 # nº sistemas por dimensão

os.makedirs("results", exist_ok=True)

# -----------------------
# Gauss com pivotagem parcial
# -----------------------
def gauss_pivotagem(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)

    for k in range(n - 1):
        maxindex = np.abs(A[k:, k]).argmax() + k
        if A[maxindex, k] == 0:
            raise ValueError("Sistema singular")

        if maxindex != k:
            A[[k, maxindex]] = A[[maxindex, k]]
            b[[k, maxindex]] = b[[maxindex, k]]

        for i in range(k + 1, n):
            fator = A[i, k] / A[k, k]
            A[i, k:] -= fator * A[k, k:]
            b[i] -= fator * b[k]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    return x

# -----------------------
# IC Bootstrap (BCa) da média
# -----------------------
def bootstrap_IC(lista):
    amostras = np.asarray(lista)
    res = bootstrap(
        (amostras,),
        np.mean,
        n_resamples=2000,
        confidence_level=0.95,
        method="BCa",
        random_state=123
    )
    return res

# -----------------------
# Simulação
# -----------------------
resultados = {}

for n in n_values:
    solucoes = []
    aceites = 0

    while aceites < K:
        A = np.random.randn(n, n)
        b = np.random.randn(n)
        try:
            x = gauss_pivotagem(A, b)
            solucoes.append(x)
            aceites += 1
        except Exception:
            continue

    # bootstrap no 1º componente
    boot = bootstrap_IC([row[0] for row in solucoes])

    solucoes = np.array(solucoes)
    media_global = np.mean(solucoes)
    desvio_global = np.std(solucoes)
    desvios_2 = np.std(solucoes[: K // 100])

    resultados[n] = {
        "media_global": media_global,
        "desvio_global": desvio_global,
        "desvios_2": desvios_2,
        "bootstrap": boot
    }

    print(f"n={n} | média={media_global:.6f} | desvio={desvio_global:.6f} | IC={boot.confidence_interval}")

# -----------------------
# Preparar dados para gráficos
# -----------------------
n_vals = list(resultados.keys())
medias_globais = [resultados[n]["media_global"] for n in n_vals]
desvios_globais = [resultados[n]["desvio_global"] for n in n_vals]
desvios_2 = [resultados[n]["desvios_2"] for n in n_vals]

# -----------------------
# Gráfico média
# -----------------------
plt.figure(figsize=(10, 5))
plt.plot(n_vals, medias_globais, marker="o", label="Média Global")
plt.title("Média global das soluções vs dimensão n")
plt.xlabel("Dimensão n do sistema")
plt.ylabel("Média global")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/medias.png", dpi=200)
plt.close()

# -----------------------
# Gráfico desvio padrão
# -----------------------
plt.figure(figsize=(10, 5))
plt.plot(n_vals, desvios_globais, marker="o", label="Desvio Global")
plt.plot(n_vals, desvios_2, marker="o", label="Desvio para (K/100) sistemas")
plt.title("Desvio-padrão das soluções vs dimensão n")
plt.xlabel("Dimensão n do sistema")
plt.ylabel("Desvio-padrão")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/desvio_padrao.png", dpi=200)
plt.close()

# -----------------------
# Gráfico intervalos de confiança (bootstrap)
# -----------------------
# -----------------------
# Gráfico intervalos de confiança (bootstrap)
# -----------------------
fig, ax = plt.subplots(figsize=(10, 5))

def IC_plot(x, media, liminf, limsup):
    largura = 0.2
    ax.plot([x - largura, x + largura], [liminf, liminf])
    ax.plot([x - largura, x + largura], [limsup, limsup])
    ax.plot([x, x], [liminf, limsup])
    ax.plot(x, media, "o")

for n in n_vals:
    dados = resultados[n]["bootstrap"]
    IC_plot(
        n,
        float(np.mean(dados.bootstrap_distribution)),
        float(dados.confidence_interval.low),
        float(dados.confidence_interval.high)
    )

ax.set_title("Intervalos de confiança (BCa) das médias via Bootstrap")
ax.set_xlabel("Dimensão n do sistema")
ax.set_ylabel("Média (1º componente)")
ax.set_xticks(range(2, 21))
ax.grid(True)

fig.tight_layout()
fig.savefig("results/bootstrap.png", dpi=200)
plt.close(fig)
