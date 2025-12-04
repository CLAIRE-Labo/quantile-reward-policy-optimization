from math import erf
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 7,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.frameon": True,
        "legend.fontsize": 14,
        "lines.linewidth": 1.7,
    }
)


# ────────────────── helper functions ────────────────────────────────
def f_ref(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)


def F_ref(x):
    vec = erf if np.isscalar(x) else np.vectorize(erf)
    return 0.5 * (1 + vec(x / np.sqrt(2)))


def f_opt(xs, beta):
    """Stable optimal-policy PDF."""
    logu = np.log(f_ref(xs)) + F_ref(xs) / beta
    logu -= logu.max()
    u = np.exp(logu)
    return u / np.trapezoid(u, xs)


# ────────────────── grid & base densities ───────────────────────────
xs = np.linspace(-3, 5, 1400)
ref = f_ref(xs)
opt1 = f_opt(xs, 0.1)  # β = 0.1
opt2 = f_opt(xs, 0.01)  # β = 0.01
opt3 = f_opt(xs, 0.001)  # β = 0.001

# intersection (β=0.1 vs reference)
diff = opt1 - ref
idx = np.where(np.diff(np.sign(diff)))[0][0]
x_int = xs[idx] - diff[idx] * (xs[idx + 1] - xs[idx]) / (diff[idx + 1] - diff[idx])
y_int = f_ref(x_int)

# regions for arrows/labels
left_mask_full = xs < x_int
right_mask_full = xs >= x_int
left_mask = (xs >= -2) & left_mask_full
right_mask = (xs <= 3) & right_mask_full
left_inds, right_inds = np.where(left_mask)[0], np.where(right_mask)[0]

# arrow sampling (same step both sides)
desired_arrows = 5
step = max(len(right_inds) // (desired_arrows + 2), 1)

# ────────────────── figure with two subplots ────────────────────────
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 4), sharey=False)

# ── LEFT PANEL ──────────────────────────────────────────────────────
ax_l.plot(xs, ref, "--", color="#6F6F6F", lw=2, label="$\pi_{ref}$")
ax_l.plot(xs, opt1, color="#E6C1CF", lw=2, label=r"$\pi^*$, $\beta=0.1$")

# shading & splitter line
ax_l.fill_between(xs[xs < x_int], ref[xs < x_int], color="red", alpha=0.08)
ax_l.fill_between(xs[xs >= x_int], ref[xs >= x_int], color="green", alpha=0.08)
ax_l.plot([x_int, x_int], [0, y_int], "k:", lw=1)

# arrows (transparent)
for i in left_inds[::step]:
    x = xs[i]
    ax_l.annotate(
        "",
        xy=(x, opt1[i] + 0.01),
        xytext=(x, ref[i] - 0.01),
        arrowprops=dict(arrowstyle="-|>", color="red", alpha=0.45, lw=1),
    )
for i in right_inds[step // 2 :: step]:
    x = xs[i]
    ax_l.annotate(
        "",
        xy=(x, opt1[i] - 0.01),
        xytext=(x, ref[i] + 0.01),
        arrowprops=dict(arrowstyle="-|>", color="green", alpha=0.45, lw=1),
    )

# percentage labels (rounded boxes)
pct_left = np.trapezoid(ref[xs < x_int], xs[xs < x_int]) * 100
pct_right = np.trapezoid(ref[xs >= x_int], xs[xs >= x_int]) * 100
x_left_label = (-1 + x_int) / 2
x_right_label = (x_int + 1.8) / 2
y_right_label = 0.15 * ref[left_inds].max()
y_left_label = 0.5 * ref[left_inds].max()

bbox_kw = dict(
    boxstyle="round,pad=0.2,rounding_size=0.45",
    facecolor="white",
    alpha=0.85,
    edgecolor="none",
)

ax_l.text(
    x_left_label,
    y_left_label,
    f"{pct_left:.0f}%",
    ha="center",
    va="center",
    color="red",
    fontsize=14,
    bbox=bbox_kw,
)
ax_l.text(
    x_right_label,
    y_right_label,
    f"{pct_right:.0f}%",
    ha="center",
    va="center",
    color="green",
    fontsize=14,
    bbox=bbox_kw,
)

# axes cosmetics
ax_l.set_xlim(-3, 4)
ax_l.set_ylim(0, 0.75)
ax_l.set_xlabel("Reward $\mathcal{R}$")
ax_l.set_ylabel("Probability density")
ax_l.legend()
ax_l.grid(alpha=0.3)

# ── RIGHT PANEL ─────────────────────────────────────────────────────
ax_r.plot(xs, ref, "--", color="#6F6F6F", lw=2, label="$\pi_{ref}$")
ax_r.plot(xs, opt1, color="#E6C1CF", lw=2, label=r"$\pi^*, \beta=0.1$")
ax_r.plot(xs, opt2, color="#D5829F", lw=2, label=r"$\pi^*, \beta=0.01$")
ax_r.plot(xs, opt3, color="#9B0E4C", lw=2, label=r"$\pi^*, \beta=0.001$")

ax_r.set_xlim(-3, 5)
ax_r.set_ylim(0, 1.3)
ax_r.set_xlabel("Reward $\mathcal{R}$")
ax_r.legend()
ax_r.grid(alpha=0.3)

plt.tight_layout()
output_dir = Path(__file__).parent / "gradient"
output_dir.mkdir(exist_ok=True)
plt.savefig(
    output_dir / "gradient-analysis.png",
    dpi=300,
)
plt.show()
