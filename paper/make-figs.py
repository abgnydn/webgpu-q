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
ax.text(0.0,-0.30,"4-tab swarm then runs the distributed SCF in ~12 s vs ~23 s single-tab parallel (§3.3)",
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

# CHEM FIG 3: swarm scaling (wall vs n), sourced from committed artifacts
fig,ax=plt.subplots(figsize=(7.4,3.9))
# (n, swarm-SCF wall s) from experiments/results/.../swarm/*.json
ccpvdz=[("benzene",120,5.6),("naphthalene",190,12.0)]      # cc-pVDZ
sto3g =[("anthracene",80,2.1),("pentacene",124,40.0),("C60",300,539.0)]  # STO-3G
for name,n,w in ccpvdz:
    ax.plot(n,w,"o",color=TEAL,ms=9,zorder=5)
    ax.annotate((r"$\mathbf{C_{60}}$"+f"\n{w:g} s" if name=="C60" else f"{name}\n{w:g} s"),(n,w),textcoords="offset points",xytext=(8,-4),fontsize=8.5,color=TEAL)
for name,n,w in sto3g:
    c=AMBER if name=="C60" else NAVY
    ax.plot(n,w,"o",color=c,ms=11 if name=="C60" else 9,zorder=5)
    dy=8 if name in ("anthracene",) else -16
    ax.annotate(f"{name}\n{w:g} s",(n,w),textcoords="offset points",xytext=(8,dy),fontsize=8.5,
                color=c,fontweight="bold" if name=="C60" else "normal")
# naphthalene single-tab parallel reference (open marker)
ax.plot(190,22.6,"o",mfc="none",mec=GREY,ms=9,mew=1.5,zorder=4)
ax.annotate("naphthalene\nsingle-tab 22.6 s",(190,22.6),textcoords="offset points",xytext=(-60,2),fontsize=8,color="#888")
ax.set_yscale("log"); ax.set_xlim(60,330); ax.set_ylim(1.2,1200)
ax.set_xlabel("basis functions  n"); ax.set_ylabel("swarm SCF wall (s, log scale)")
ax.set_title("Browser-tab swarm HF: SCF wall vs system size")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],marker="o",ls="",color=TEAL,label="cc-pVDZ"),
                   Line2D([],[],marker="o",ls="",color=NAVY,label="STO-3G"),
                   Line2D([],[],marker="o",ls="",mfc="none",mec=GREY,label="single-tab ref")],
          fontsize=9,frameon=False,loc="upper left")
ax.text(0.99,0.03,"each point from a committed swarm artifact; C60 from a local high-memory run",
        transform=ax.transAxes,ha="right",fontsize=7.8,color="#999")
clean(ax)
fig.tight_layout(); fig.savefig("fig-scaling.pdf"); plt.close(fig)

# CHEM FIG 4: streaming swarm vs the 2 GB single-allocation wall (H2 lattice, cc-pVDZ)
fig,ax=plt.subplots(figsize=(7.4,4.0))
WALL=(2**31-1)/1e9   # 2.147 GB max Float64Array
# (n, full_B_GB) from committed h2lattice artifacts; per-tab = full/4 (4 tabs);
# peakV = one muBlock (8) streamed = 8*n*n_aux*8 bytes.
pts=[(160,0.16,801),(400,2.56,2000),(490,4.70,2450)]   # (n, fullB GB, n_kept~n_aux)
ns=[p[0] for p in pts]; fullB=[p[1] for p in pts]
perTab=[b/4 for b in fullB]
peakV=[8*n*nx*8/1e9 for (n,_,nx) in pts]
ax.plot(ns,fullB,"o-",color=GREY,lw=1.8,ms=8,zorder=4,label="full B tensor (single tab)")
ax.plot(ns,perTab,"o-",color=TEAL,lw=1.8,ms=8,zorder=5,label="per-tab slice (4-tab swarm)")
ax.plot(ns,peakV,"o-",color=NAVY,lw=1.8,ms=7,zorder=5,label="peak resident V (streamed, 1 µ-block)")
ax.axhline(WALL,color=RED,ls="--",lw=1.8,zorder=3)
ax.text(168,WALL*1.07,"2 GB single-Float64Array wall",color=RED,fontsize=9.5,fontweight="bold")
# mark the headline n=400 point (over the wall)
ax.annotate("40 H$_2$: full B 2.56 GB\n(impossible single-tab)\nsplit: 4×640 MB",
            (400,2.56),textcoords="offset points",xytext=(-150,28),fontsize=8.6,color="#333",
            arrowprops=dict(arrowstyle="->",color="#555",lw=1.1))
ax.set_yscale("log"); ax.set_xlim(150,510); ax.set_ylim(0.02,8)
ax.set_xlabel("basis functions  n"); ax.set_ylabel("memory (GB, log scale)")
ax.set_title("Streaming swarm scales past the single-allocation wall")
ax.legend(fontsize=9,frameon=False,loc="lower right")
ax.text(0.99,0.02,"each point a committed, oracle-validated artifact",
        transform=ax.transAxes,ha="right",va="bottom",fontsize=7.6,color="#999")
clean(ax)
fig.tight_layout(); fig.savefig("fig-streaming-scale.pdf"); plt.close(fig)
print("OK")
