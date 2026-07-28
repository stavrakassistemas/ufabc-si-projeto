"""
Harness compartilhado da Fase 1 — usado pelos notebooks das três trilhas e pela conclusão.

Garante que A (gradient boosting), B (rede neural) e C (convolucional) sejam avaliadas na
MESMA régua: mesmo split temporal, mesmas features honestas (sem preço/vazamento), mesma
métrica RMSSE, e a mesma seção de resultados em três níveis de serviço (econômico/médio/seguro).
"""
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titleweight"] = "bold"

PALETA = "Set2"
HORIZONTE = 120                 # dias finais reservados para teste
ANOS_TESTE = HORIZONTE / 365
CUSTO_CARREGAMENTO = 0.25       # 25% a.a. (premissa da fase 0b)
NIVEIS = {"Econômico": 0.85, "Médio": 0.95, "Seguro": 0.99}   # parâmetro de ajuste τ
COR_NIVEL = {"Econômico": "#4C72B0", "Médio": "#DD8452", "Seguro": "#55A868"}

CATEGORICAS = ["loja", "categoria", "clima", "estacao"]
CONTEXTO = ["promocao", "epidemia", "desconto"]
CALENDARIO = ["mes", "cal_ano_sin", "cal_ano_cos", "cal_sem_sin", "cal_sem_cos"]
FEATURES = CATEGORICAS + CONTEXTO + CALENDARIO      # conjunto honesto (sem preço, sem vazamento)


def carregar():
    "Carrega o CSV, cria SKU, features de calendário e as colunas honestas. Retorna o DataFrame."
    caminho = next(arq for pasta in [Path("."), Path("projeto"), Path("..")]
                   for arq in sorted(pasta.glob("sales_data*.csv")))
    nomes = {"Date": "data", "Store ID": "loja", "Product ID": "produto", "Category": "categoria",
             "Region": "regiao", "Inventory Level": "estoque", "Units Sold": "vendas",
             "Units Ordered": "pedidos", "Price": "preco", "Discount": "desconto",
             "Weather Condition": "clima", "Promotion": "promocao",
             "Competitor Pricing": "preco_concorrente", "Seasonality": "estacao",
             "Epidemic": "epidemia", "Demand": "demanda"}
    df = (pd.read_csv(caminho, parse_dates=["Date"]).rename(columns=nomes)
          .assign(sku=lambda d: d["loja"] + "-" + d["produto"])
          .sort_values(["sku", "data"]).reset_index(drop=True))
    df["mes"] = df["data"].dt.month
    doy, dow = df["data"].dt.dayofyear, df["data"].dt.dayofweek
    df["cal_ano_sin"], df["cal_ano_cos"] = np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)
    df["cal_sem_sin"], df["cal_sem_cos"] = np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)
    # Preço de referência ESTÁVEL por SKU (mediana): usado nas valorações em R$, para não
    # herdar a volatilidade do preço diário (contaminado pelo desconto atrelado à demanda).
    df["preco_ref"] = df.groupby("sku")["preco"].transform("median")
    for c in CATEGORICAS:
        df[c] = df[c].astype("category")
    return df


def split(df):
    "Split TEMPORAL: os últimos HORIZONTE dias são teste (janela à frente, como na M5)."
    corte = df["data"].max() - pd.Timedelta(days=HORIZONTE)
    return df[df["data"] <= corte].copy(), df[df["data"] > corte].copy()


def rmsse_medio(teste, pred, treino):
    "RMSSE médio entre SKUs (erro escalado pelo naïve de 1 passo no treino)."
    escala = treino.groupby("sku")["demanda"].apply(lambda s: np.mean(np.diff(s.values) ** 2))
    def _um(g):
        e = np.mean((g["demanda"].values - g[pred].values) ** 2)
        return np.sqrt(e / escala[g.name]) if escala[g.name] > 0 else np.nan
    return teste.groupby("sku").apply(_um).mean()


def pinball(y, pred, q):
    "Perda quantílica (pinball)."
    d = np.asarray(y) - np.asarray(pred)
    return np.mean(np.maximum(q * d, (q - 1) * d))


def benchmarks(treino, teste):
    "Adiciona colunas de benchmark ao teste e devolve o placar (naïve, sazonal, média)."
    g = teste.sort_values(["sku", "data"]).groupby("sku")["demanda"]
    teste["pred_naive"] = g.shift(1)
    teste["pred_sazonal"] = g.shift(7)
    teste["pred_media"] = teste["sku"].map(treino.groupby("sku")["demanda"].mean())
    placar = []
    for nome, col in [("naïve (lag 1)", "pred_naive"), ("naïve sazonal (lag 7)", "pred_sazonal"),
                      ("média do SKU", "pred_media")]:
        sub = teste.dropna(subset=[col])
        placar.append({"modelo": nome, "RMSSE": round(rmsse_medio(sub, col, treino), 4),
                       "MAE": round((sub["demanda"] - sub[col]).abs().mean(), 2)})
    return pd.DataFrame(placar)


# ---------------------------------------------------------------- economia / resultados
def _reconstroi_meia_onda(df):
    "Metade do lote de reposição por SKU (Q/2), para o capital ser comparável ao atual."
    ent = df.groupby("sku")["estoque"].shift(-1) - df["estoque"] + df["vendas"]
    Q = ent[ent > 0].groupby(df["sku"]).mean()
    return Q / 2


def baseline_atual(df, teste):
    "Ruptura, venda perdida (R$/ano) e capital (R$) da operação ATUAL, valorados a preço estável."
    rup = (teste["demanda"] > teste["estoque"]).mean()
    perda = ((teste["demanda"] - teste["estoque"]).clip(lower=0) * teste["preco_ref"]).sum() / ANOS_TESTE
    capital = (teste.assign(v=teste["estoque"] * teste["preco_ref"]).groupby("data")["v"].sum().mean())
    return {"ruptura": rup, "perda": perda, "capital": capital}


def calibrador_residual(treino, teste, col_point):
    """
    Devolve uma função pred_q(tau) que produz o PEDIDO no nível de serviço tau, por
    calibração de resíduos por SKU: pedido = previsão de ponto + quantil_tau(resíduos do SKU).
    Método uniforme entre as trilhas (estilo split-conformal).
    """
    res = (treino["demanda"] - treino[col_point]).groupby(treino["sku"])
    quantis = {tau: res.quantile(tau) for tau in sorted(set(list(NIVEIS.values()) +
              [0.70, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99]))}
    sku_teste = teste["sku"].values
    def pred_q(tau):
        add = teste["sku"].map(quantis[tau]).values
        return np.maximum(teste[col_point].values + add, 0)
    return pred_q


def _linha_politica(df, teste, base, pred, tau):
    "Trio (serviço, ruptura, perda, capital) + economia/benefício para um pedido `pred`."
    meia = teste["sku"].map(_reconstroi_meia_onda(df)).values
    preco = teste["preco_ref"].values          # valoração a preço estável (ver carregar)
    ruptura = (teste["demanda"].values > pred).mean()
    perda = (np.clip(teste["demanda"].values - pred, 0, None) * preco).sum() / ANOS_TESTE
    capital = (pd.DataFrame({"data": teste["data"].values, "v": (pred + meia) * preco})
               .groupby("data")["v"].sum().mean())
    return {"τ": tau, "serviço": 1 - ruptura, "ruptura": ruptura, "perda": perda, "capital": capital,
            "economia_perda": base["perda"] - perda,
            "carreg_extra": (capital - base["capital"]) * CUSTO_CARREGAMENTO,
            "beneficio_liq": (base["perda"] - perda) - (capital - base["capital"]) * CUSTO_CARREGAMENTO}


def fronteira(df, teste, base, pred_q, taus=(0.70, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99)):
    "DataFrame com o trade-off completo variando o parâmetro τ."
    return pd.DataFrame([_linha_politica(df, teste, base, pred_q(t), t) for t in taus])


def tres_niveis(df, teste, base, pred_q):
    "DataFrame com os três níveis nomeados (econômico, médio, seguro)."
    linhas = []
    for nome, tau in NIVEIS.items():
        linhas.append({**_linha_politica(df, teste, base, pred_q(tau), tau), "nível": nome})
    return pd.DataFrame(linhas)


def tabela_niveis(base, tri):
    "Formata a tabela de resultados em três níveis para impressão."
    t = pd.DataFrame({"": ["nível de serviço-alvo (τ)", "risco de ruptura", "venda perdida / ano",
                           "→ economia vs. hoje", "capital parado", "→ custo do capital a mais / ano",
                           "BENEFÍCIO LÍQUIDO / ano"]})
    for _, r in tri.iterrows():
        t[f"{r['nível']} (τ={r['τ']:.2f})"] = [
            f"{r['serviço']:.0%}", f"{r['ruptura']:.1%}", f"R$ {r['perda']/1e6:.2f} M",
            f"R$ {r['economia_perda']/1e6:.2f} M", f"R$ {r['capital']/1e6:.2f} M",
            f"R$ {r['carreg_extra']/1e3:+,.0f} mil", f"R$ {r['beneficio_liq']/1e6:.2f} M"]
    return t


def grafico_resultados(base, frente, tri, titulo=""):
    "Os dois gráficos padrão: fronteira (recuperação × capital) e botão de ajuste (ruptura × capital)."
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    a1.plot(frente["carreg_extra"]/1e3, frente["economia_perda"]/1e6, "-o", color="#999999", lw=1.5, ms=5, zorder=1)
    for _, r in tri.iterrows():
        a1.scatter(r["carreg_extra"]/1e3, r["economia_perda"]/1e6, s=180, color=COR_NIVEL[r["nível"]],
                   zorder=3, edgecolor="white", linewidth=1.5)
        a1.annotate(f"  {r['nível']}\n  (τ={r['τ']:.2f})", (r["carreg_extra"]/1e3, r["economia_perda"]/1e6),
                    fontsize=8, va="center")
    a1.set_title("A fronteira: recuperação por real de capital a mais")
    a1.set_xlabel("custo do capital a mais (R$ mil / ano)"); a1.set_ylabel("venda recuperada (R$ mi / ano)")

    a2.plot(frente["capital"]/1e6, frente["ruptura"]*100, "-o", color="#999999", lw=1.5, ms=5, zorder=1)
    a2.axhline(base["ruptura"]*100, color="#C44E52", lw=1.3, ls="--", label="ruptura de hoje")
    a2.axvline(base["capital"]/1e6, color="#888888", lw=1, ls=":", label="capital de hoje")
    for _, r in tri.iterrows():
        a2.scatter(r["capital"]/1e6, r["ruptura"]*100, s=180, color=COR_NIVEL[r["nível"]],
                   zorder=3, edgecolor="white", linewidth=1.5)
        a2.annotate(f" {r['nível']}", (r["capital"]/1e6, r["ruptura"]*100), fontsize=8, va="bottom")
    a2.set_title("O botão de ajuste: mais estoque ↔ menos ruptura")
    a2.set_xlabel("capital parado (R$ milhões)"); a2.set_ylabel("risco de ruptura (%)")
    a2.legend(frameon=False, fontsize=8)
    if titulo:
        fig.suptitle(titulo, y=1.02, fontsize=12, weight="bold")
    sns.despine(); plt.tight_layout(); plt.show()


def salva_resultado(nome_trilha, teste, col_point, rmsse_valor):
    "Persiste as previsões de teste da trilha para a comparação em fase1_conclusoes."
    Path("resultados").mkdir(exist_ok=True)
    out = teste[["data", "sku", "categoria", "demanda", "estoque", "preco", "preco_ref"]].copy()
    out["pred"] = teste[col_point].values
    out.to_csv(f"resultados/{nome_trilha}.csv", index=False)
    pd.Series({"trilha": nome_trilha, "rmsse": rmsse_valor}).to_json(f"resultados/{nome_trilha}_meta.json")
