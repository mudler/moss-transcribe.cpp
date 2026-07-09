"""Render the CPU benchmark plots for moss-transcribe.cpp vs upstream PyTorch.

Numbers are the warm, isolated, 8-thread, F32 measurements from BENCHMARK.md
(20-core x86 CPU). Run: python3 benchmarks/make_plots.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "benchmarks/media"

lengths = [11, 44, 132]
ours_rtf = [0.59, 0.55, 0.78]
pt_rtf = [0.96, 0.98, 1.23]
speedup = [round(p / o, 2) for p, o in zip(pt_rtf, ours_rtf)]

OURS = "#0d9488"   # teal
PT = "#f97316"     # orange
GRID = "#e2e8f0"

plt.rcParams.update({"font.size": 12, "axes.edgecolor": "#334155",
                     "axes.grid": True, "grid.color": GRID})


def rtf_vs_length():
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axhspan(0, 1.0, color="#ecfdf5", zorder=0)
    ax.axhline(1.0, color="#94a3b8", lw=1, ls="--")
    ax.text(lengths[0], 1.01, "real time", color="#64748b", fontsize=10, va="bottom")
    ax.plot(lengths, pt_rtf, "-o", color=PT, lw=2.5, ms=8, label="PyTorch (torch CPU)")
    ax.plot(lengths, ours_rtf, "-o", color=OURS, lw=2.5, ms=8, label="moss-transcribe.cpp (ggml CPU)")
    for x, o, p in zip(lengths, ours_rtf, pt_rtf):
        ax.annotate(f"{o:.2f}", (x, o), textcoords="offset points", xytext=(0, -16),
                    ha="center", color=OURS, fontsize=10, fontweight="bold")
        ax.annotate(f"{p:.2f}", (x, p), textcoords="offset points", xytext=(0, 8),
                    ha="center", color=PT, fontsize=10, fontweight="bold")
    ax.set_xlabel("audio length (s)")
    ax.set_ylabel("RTF (lower is faster)")
    ax.set_title("CPU inference speed, F32, 8 threads, identical transcript")
    ax.set_xticks(lengths)
    ax.set_ylim(0, 1.4)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/rtf_vs_length.png", dpi=140)
    print(f"wrote {OUT}/rtf_vs_length.png")


def speedup_bars():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar([f"{d}s" for d in lengths], speedup, color=OURS, width=0.55)
    ax.axhline(1.0, color="#94a3b8", lw=1, ls="--")
    ax.text(2.5, 1.02, "PyTorch baseline", color="#64748b", fontsize=10, ha="right")
    for b, s in zip(bars, speedup):
        ax.annotate(f"{s:.2f}x", (b.get_x() + b.get_width() / 2, s),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontweight="bold", color=OURS)
    ax.set_ylabel("speedup vs PyTorch")
    ax.set_xlabel("audio length")
    ax.set_title("moss-transcribe.cpp is 1.6 to 1.8x faster than PyTorch on CPU (F32)")
    ax.set_ylim(0, 2.1)
    fig.tight_layout()
    fig.savefig(f"{OUT}/speedup.png", dpi=140)
    print(f"wrote {OUT}/speedup.png")


if __name__ == "__main__":
    rtf_vs_length()
    speedup_bars()
