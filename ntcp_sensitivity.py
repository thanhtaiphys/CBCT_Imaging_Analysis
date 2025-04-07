# NTCP Visualization Code - Logistic & LKB Models per Organ (Separate Figures with Risk Zones)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.2)

# === Function 1: NTCP logistic model ===
def ntcp_logistic(D, D50, gamma):
    t = gamma * (D - D50)
    return 1 / (1 + np.exp(-t))

# === Function 2: NTCP LKB model ===
def ntcp_lkb(Deff, D50, m):
    t = (Deff - D50) / (m * D50)
    return norm.cdf(t)

# === Function to plot LKB curve with risk zones ===
def plot_lkb_with_risk_zone(D50, m, roi):
    Deff_vals = np.linspace(0, 70, 300)
    ntcp_vals = ntcp_lkb(Deff_vals, D50, m)

    plt.figure(figsize=(9, 5))
    plt.plot(Deff_vals, ntcp_vals, label=f"NTCP (LKB, D50={D50} Gy, m={m})", color="black")
    plt.axvline(D50, linestyle="--", color="gray", label=f"D50")

    plt.fill_between(Deff_vals, 0, ntcp_vals, where=ntcp_vals < 0.01, color="green", alpha=0.2, label="NTCP < 1% (an toàn)")
    plt.fill_between(Deff_vals, 0, ntcp_vals, where=(ntcp_vals >= 0.01) & (ntcp_vals <= 0.1), color="yellow", alpha=0.3, label="1% < NTCP ≤ 10% (cẩn trọng)")
    plt.fill_between(Deff_vals, 0, ntcp_vals, where=ntcp_vals > 0.1, color="red", alpha=0.3, label="NTCP > 10% (nguy cơ cao)")

    plt.title(f"Đường cong NTCP (LKB) và vùng nguy cơ – {roi}")
    plt.xlabel("Deff (Gy)")
    plt.ylabel("NTCP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"NTCP_LKB_{roi}_riskzones.png", dpi=300)
    plt.close()

# === Organ-specific parameters ===
roi_params = {
    "Breast_CNTR": {"model": "logistic", "D50": 30.89, "gamma": [1.0, 1.3, 2.0]},
    "Lung_IPSI":   {"model": "lkb",     "D50": 37.6, "m": 0.35},
    "Lung_CNTR":   {"model": "lkb",     "D50": 37.6, "m": 0.35},
    "Heart":       {"model": "lkb",     "D50": 48.0, "m": 0.1}
}

# === Vẽ từng cơ quan thành figure riêng ===
for roi, params in roi_params.items():
    if params["model"] == "lkb":
        plot_lkb_with_risk_zone(params["D50"], params["m"], roi)

print("\n✅ Đã lưu các biểu đồ NTCP (LKB) có vùng nguy cơ cho từng cơ quan.")