import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAVY="#1f3a5f"; TEAL="#2f6f6f"; AMBER="#b45309"; GREY="#8a949e"; RED="#c0392b"
plt.rcParams.update({
    "font.family":"serif","font.serif":["Charter","Georgia","DejaVu Serif"],
    "font.size":11,"axes.titlesize":13,"axes.titleweight":"bold",
    "axes.edgecolor":"#333","axes.linewidth":1.0,
    "xtick.color":"#333","ytick.color":"#333",
    "axes.grid":True,"grid.color":"#e6e6e6","grid.linewidth":0.8,
})
def clean(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ARR=r"$\rightarrow$"; LR=r"$\leftrightarrow$"

# CHEM FIG 1: validation accuracy ladder
fig,ax=plt.subplots(figsize=(7.4,3.3))
labels=["swarm Fock build\nvs single-tab",f"GPU{LR}CPU\n(f32 reduction)",
        "HF vs PySCF\n(spherical-d)","CCSD(T) vs FCI\n(STO-3G)"]
errs=[1e-12,1e-10,5e-5,2.5e-4]
tags=["1e-12 Ha","1e-10 Ha","5e-5 Ha (50 µHa)","2.5e-4 Ha (0.25 mHa)"]
ypos=np.arange(len(labels))[::-1]; cols=[TEAL,TEAL,NAVY,NAVY]
ax.barh(ypos,errs,color=cols,height=0.5,zorder=3)
for y,e,t in zip(ypos,errs,tags):
    ax.text(e*1.5,y,t,va="center",ha="left",fontsize=9.5)
ax.axvline(1.594e-3,color=AMBER,ls="--",lw=1.8,zorder=4)
ax.text(1.594e-3*1.3,len(labels)-1.15,"chemical accuracy\n1.594 mHa",color=AMBER,
        va="top",ha="left",fontsize=9.5,fontweight="bold")
ax.set_xscale("log"); ax.set_xlim(3e-13,1.5e-1)
ax.set_yticks(ypos); ax.set_yticklabels(labels,fontsize=9.5)
ax.set_xlabel("absolute energy error vs reference (Ha, log scale)")
ax.set_title("Validation: every error sits below chemical accuracy")
clean(ax); ax.grid(axis="y",visible=False)
fig.tight_layout(); fig.savefig("fig-validation.pdf"); plt.close(fig)

# CHEM FIG 2: single-tab optimization
fig,ax=plt.subplots(figsize=(7.0,3.6))
steps=["parallel\nbaseline","+ reuse JK\nscratch","+ exploit K\nsymmetry"]
wall=[43,20,17]
ax.bar(range(3),wall,color=[GREY,TEAL,NAVY],width=0.55,zorder=3)
for i,w in enumerate(wall):
    ax.text(i,w+0.9,f"{w:.0f} s",ha="center",fontsize=10,fontweight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels(steps,fontsize=9.5)
ax.set_ylabel("naphthalene cc-pVDZ SCF wall (s)"); ax.set_ylim(0,52); ax.set_xlim(-0.6,2.6)
ax.set_title("Single-tab Fock-build optimization")
ax.text(-0.55,48.5,f"two kept optimizations: 43 s {ARR} 17 s on the SCF wall (~2.5×)",
        fontsize=9.5,style="italic",color="#555")
ax.text(0.0,-0.30,"optimized single-tab end-to-end HF ≈28 s; the 4-tab swarm halves it to 14 s (2×, §3.3)",
        transform=ax.transAxes,fontsize=8.5,color="#777")
clean(ax); ax.grid(axis="x",visible=False)
fig.tight_layout(); fig.savefig("fig-optimization.pdf"); plt.close(fig)

# FUSION FIG 1: dispatch-cost collapse
fig,ax=plt.subplots(figsize=(6.8,3.6))
k=np.array([1,4,8]); aeff=np.array([54.5,17.8,15.8])
kk=np.linspace(1,8,200); alpha=16.0; C=13.8
ax.plot(kk,alpha/kk+C,ls="--",color=GREY,lw=1.6,label=r"$\alpha/k + C$ model ($\alpha{=}16,\,C{\approx}14$)")
ax.plot(k,aeff,"o",color=NAVY,ms=9,zorder=5,label="measured (N=8 single-qubit chains)")
for kx,ay in zip(k,aeff):
    ax.annotate(f"{ay} µs",(kx,ay),textcoords="offset points",xytext=(8,6),fontsize=10)
ax.axhline(C,color=TEAL,ls=":",lw=1.4)
ax.text(8,C+1.0,"asymptote C ≈ 14 µs/gate",color=TEAL,ha="right",fontsize=9.5)
ax.set_xlabel("fusion factor k (gates per dispatch)")
ax.set_ylabel(r"effective per-gate cost $\alpha_{\mathrm{eff}}$ (µs)")
ax.set_title("Kernel fusion collapses the per-dispatch cost as 1/k")
ax.set_xticks([1,2,4,6,8]); ax.set_ylim(0,60); ax.legend(fontsize=9.5,frameon=False)
clean(ax)
fig.tight_layout(); fig.savefig("fig-dispatch.pdf"); plt.close(fig)

# FUSION FIG 2: tier ladder vs target
fig,ax=plt.subplots(figsize=(7.2,3.7))
tiers=[f"B\n2-qubit {ARR} 4×4",f"C\n3-qubit {ARR} 8×8",f"D\n4-qubit {ARR} 16×16"]
speed=[3.04,4.22,3.78]; target=[2,3,5]; cols=[TEAL,TEAL,AMBER]
x=np.arange(3)
ax.bar(x,speed,color=cols,width=0.55,zorder=3)
for i,(s,t) in enumerate(zip(speed,target)):
    ax.text(i,s+0.10,f"{s:.2f}×",ha="center",fontsize=11,fontweight="bold")
    ax.plot([i-0.30,i+0.30],[t,t],color=RED,lw=2.2,zorder=5)
    ax.text(i,t+0.12,f"target ≥{t}×",color=RED,ha="center",fontsize=8.6)
ax.set_xticks(x); ax.set_xticklabels(tiers,fontsize=9.5)
ax.set_ylabel("speedup vs unfused dispatch path"); ax.set_ylim(0,6.2)
ax.set_title("Fusion tier ladder: pass through Tier-C, plateau at Tier-D")
ax.annotate("compute-bound plateau\n(honest negative)",(2,3.78),
            textcoords="offset points",xytext=(-86,40),fontsize=8.8,color=AMBER,ha="center",
            arrowprops=dict(arrowstyle="->",color=AMBER,lw=1.2))
clean(ax); ax.grid(axis="x",visible=False)
fig.tight_layout(); fig.savefig("fig-ladder.pdf"); plt.close(fig)
print("OK")
