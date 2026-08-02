"""
Diagramas de rede reutilizáveis para o projeto:
- macro(): arquitetura em blocos (o "mapa" da rede), caprichado.
- plotnet(): topologia nó-a-nó no estilo clássico (azul = peso positivo, laranja = negativo,
             espessura ∝ |peso|), com viés e cabeçalho "n0 -> n1 -> ... | iterações | perda".
- treina_surrogate(): treina uma rede compacta (para o plotnet ficar legível como o clássico).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

AZUL="#3B6FB0"; LARANJA="#E08A3C"; CINZA="#8A8F98"; TINTA="#2b2b2b"
CORB={"in":"#AEB9D6","emb":"#8DA0CB","hid":"#66C2A5","out":"#FC8D62","op":"#F0C36D","seq":"#B7C9E2"}


# ----------------------------------------------------------------- MACRO (blocos)
def _bloco(ax,x,y,w,h,txt,cor,fs=9,txtcor=TINTA):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.08",
                 facecolor=cor,edgecolor="#3a3a3a",lw=1.3,zorder=3,mutation_aspect=1))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=txtcor,zorder=4,weight="bold")

def _seta(ax,x1,y1,x2,y2,cor="#5b5b5b",lw=1.6):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=13,
                 color=cor,lw=lw,zorder=2,shrinkA=1,shrinkB=1))

def macro(ax,tipo="mlp"):
    "Desenha o mapa em blocos da arquitetura `tipo` in {'mlp'}."
    ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off")
    if tipo=="mlp":
        ax.set_title("Macro — MLP com entity embeddings",fontsize=10.5,weight="bold")
        for k,lab in enumerate(["loja","categoria","clima","estação"]):
            _bloco(ax,0.2,8.0-k*1.35,2.0,0.95,f"{lab}\n↳ embedding",CORB["emb"],7.8)
        _bloco(ax,0.2,1.9,2.0,0.95,"promoção · epidemia\ndesconto · calendário",CORB["in"],7.3)
        _bloco(ax,3.0,3.9,1.5,2.6,"concat",("#e9e9e9"),9)
        _bloco(ax,4.9,4.4,1.9,1.7,"Dense 128\n→ Dense 64\n(ReLU)",CORB["hid"],8.2)
        _bloco(ax,7.3,4.8,2.4,0.95,"demanda prevista",CORB["out"],8.5)
        for k in range(4): _seta(ax,2.2,8.5-k*1.35,3.0,5.6)
        _seta(ax,2.2,2.4,3.0,4.3); _seta(ax,4.5,5.2,4.9,5.25); _seta(ax,6.8,5.25,7.3,5.3)


# ----------------------------------------------------------------- plotnet (micro)
def plotnet(ax,sizes,Ws,in_labels,out_labels,title,biases=None,max_lw=4.2):
    """
    Topologia nó-a-nó estilo clássico. `sizes`=[n0,n1,...]; `Ws`[l] shape (sizes[l], sizes[l+1]);
    `biases`[l] shape (sizes[l+1],). Aresta azul se peso>0, laranja se <0, espessura ∝ |peso|.
    """
    ax.axis("off")
    nL=len(sizes); xs=np.linspace(0.08,0.92,nL)
    def ys(n):  # posições verticais centralizadas
        return np.linspace(0.86,0.14,n) if n>1 else np.array([0.5])
    pos=[list(zip([xs[l]]*sizes[l],ys(sizes[l]))) for l in range(nL)]
    ybias=0.965
    wmax=max((np.abs(W).max() for W in Ws if W.size),default=1.0) or 1.0
    # arestas
    for l,W in enumerate(Ws):
        for i in range(sizes[l]):
            for j in range(sizes[l+1]):
                w=W[i,j]; lw=0.25+max_lw*abs(w)/wmax
                ax.plot([pos[l][i][0],pos[l+1][j][0]],[pos[l][i][1],pos[l+1][j][1]],
                        color=AZUL if w>=0 else LARANJA,lw=lw,alpha=0.72,zorder=1,solid_capstyle="round")
        if biases is not None and biases[l] is not None:  # viés
            for j in range(sizes[l+1]):
                w=biases[l][j]; lw=0.25+max_lw*abs(w)/wmax
                ax.plot([xs[l],pos[l+1][j][0]],[ybias,pos[l+1][j][1]],
                        color=AZUL if w>=0 else LARANJA,lw=lw,alpha=0.5,ls=(0,(2,1.5)),zorder=1)
            ax.add_patch(Circle((xs[l],ybias),0.011,facecolor="white",edgecolor=CINZA,lw=1,zorder=3))
            ax.text(xs[l],ybias+0.03,"viés",ha="center",fontsize=6.5,color=CINZA)
    # nós
    r=0.018
    for l in range(nL):
        for k,(x,y) in enumerate(pos[l]):
            fc=CORB["in"] if l==0 else (CORB["out"] if l==nL-1 else "white")
            ax.add_patch(Circle((x,y),r,facecolor=fc,edgecolor="#3a3a3a",lw=1.2,zorder=4))
            if l==0 and k<len(in_labels):
                ax.text(x-0.03,y,in_labels[k],ha="right",va="center",fontsize=7.4,color=TINTA)
            elif l==nL-1 and k<len(out_labels):
                ax.text(x+0.03,y,out_labels[k],ha="left",va="center",fontsize=7.4,color=TINTA)
            elif 0<l<nL-1:
                ax.text(x,y,f"h{k+1}",ha="center",va="center",fontsize=6.2,color="#555")
    ax.set_title(title,fontsize=9.5,weight="bold",pad=8)
    ax.set_xlim(-0.02,1.02); ax.set_ylim(0.05,1.02)


# ----------------------------------------------------------------- surrogate p/ plotnet
def treina_surrogate(X,y,hidden=4,epochs=1200,seed=0):
    "MLP compacto (numpy) 1 camada oculta, p/ o plotnet ficar legível. Retorna pesos, iterações, SSE."
    rng=np.random.default_rng(seed)
    Xs=(X-X.mean(0))/(X.std(0)+1e-9); ys=(y-y.mean())/(y.std()+1e-9)
    n,d=Xs.shape
    W1=rng.normal(0,0.5,(d,hidden)); b1=np.zeros(hidden); W2=rng.normal(0,0.5,(hidden,1)); b2=np.zeros(1)
    lr=0.05
    for it in range(epochs):
        z1=Xs@W1+b1; a1=np.tanh(z1); out=(a1@W2+b2).ravel()
        err=out-ys;
        gW2=a1.T@err[:,None]/n; gb2=err.mean(keepdims=True)
        da1=(err[:,None]@W2.T)*(1-a1**2)
        gW1=Xs.T@da1/n; gb1=da1.mean(0)
        W2-=lr*gW2; b2-=lr*gb2; W1-=lr*gW1; b1-=lr*gb1
    sse=float(((out-ys)**2).sum()); mse=sse/n
    return (W1,b1,W2.ravel(),b2), epochs, mse


def _design(df):
    "Matriz de entrada interpretável (~8 colunas honestas) para os plotnets."
    import pandas as pd
    return pd.DataFrame({
        "epidemia":df["epidemia"],"promoção":df["promocao"],"desconto":df["desconto"],
        "sol":(df["clima"]=="Sunny").astype(int),"chuva/neve":df["clima"].isin(["Rainy","Snowy"]).astype(int),
        "verão":(df["estacao"]=="Summer").astype(int),
        "mantim.":(df["categoria"]=="Groceries").astype(int),"móveis":(df["categoria"]=="Furniture").astype(int)})


def painel(df,tipo="mlp"):
    "Figura 1×2: macro (blocos) + micro (plotnet com os pesos reais). tipo∈{mlp}."
    X=_design(df); labs=list(X.columns); Xv=X.values.astype(float); y=df["demanda"].values.astype(float)
    fig=plt.figure(figsize=(15,4.6))
    macro(fig.add_subplot(1,2,1),tipo)
    ax=fig.add_subplot(1,2,2)
    (W1,b1,W2,b2),it,mse=treina_surrogate(Xv,y,hidden=4)
    plotnet(ax,[8,4,1],[W1,W2.reshape(-1,1)],labs,["demanda"],
            f"Micro — pesos aprendidos   |   8 → 4 → 1   |   {it} iterações   |   erro {mse:.3f}",biases=[b1,b2])
    plt.tight_layout(); return fig
