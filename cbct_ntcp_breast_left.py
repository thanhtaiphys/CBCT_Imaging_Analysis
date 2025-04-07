# Chương trình phân tích NTCP cho Breast (Left) sau MV-CBCT 5MU và 10MU
# ============================================

import os
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns
from dicompylercore import dicomparser, dvhcalc
from scipy.stats import norm

# 1. Đường dẫn dữ liệu
root_dir = r"C:/RT_Project/data_cbct/Breast_CBCT/breast_left"

# 2. ROI và tham số NTCP
roi_dict = {
    10: "Breast_CNTR",
    23: "Lung_IPSI",
    22: "Lung_CNTR",
    19: "Heart"
}

roi_params = {
    "Breast_CNTR": {"model": "logistic", "d50": 30.89, "gamma": 1.3},
    "Lung_IPSI":   {"model": "lkb",     "d50": 24.5, "m": 0.35, "n": 0.87},
    "Lung_CNTR":   {"model": "lkb",     "d50": 24.5, "m": 0.35, "n": 0.87},
    "Heart":       {"model": "lkb",     "d50": 48.0,  "m": 0.1,  "n": 0.35}
}

# 3. Hàm tính NTCP

def compute_ntcp_logistic(mean_dose, d50, gamma):
    t = gamma * (mean_dose - d50)
    return 1 / (1 + np.exp(-t))

def compute_ntcp_lkb(dose_bins, counts, d50, m, n):
    total_counts = np.sum(counts)
    if total_counts == 0:
        return 0.0
    v_fraction = np.array(counts) / total_counts
    doses = np.array(dose_bins[:len(v_fraction)])
    valid = v_fraction > 0
    if not np.any(valid):
        return 0.0
    deff = np.power(np.sum(v_fraction[valid] * (doses[valid] ** n)), 1/n)
    t = (deff - d50) / (m * d50)
    return float(norm.cdf(t))

# 4. Lịt qua và tính NTCP
records = []
for patient_id in os.listdir(root_dir):
    pat_path = os.path.join(root_dir, patient_id)
    if not os.path.isdir(pat_path): continue
    files = os.listdir(pat_path)
    try:
        struct_file = [f for f in files if f.startswith("RS")][0]
        dose_files = {
            "5MU": [f for f in files if "CBCT5" in f][0],
            "10MU": [f for f in files if "CBCT10" in f][0]
        }
    except:
        continue

    rtstruct = dicomparser.DicomParser(os.path.join(pat_path, struct_file))
    for plan_type, dose_file in dose_files.items():
        rtdose = dicomparser.DicomParser(os.path.join(pat_path, dose_file))
        for roi_id, roi_name in roi_dict.items():
            try:
                dvh = dvhcalc.get_dvh(rtstruct.ds, rtdose.ds, roi_id)
                if dvh is None or len(dvh.bins) == 0:
                    continue
                mean_dose = dvh.mean
                min_dose = dvh.min
                max_dose = dvh.max
                param = roi_params[roi_name]
                ntcp = compute_ntcp_logistic(mean_dose, param["d50"], param["gamma"]) if param["model"] == "logistic" \
                       else compute_ntcp_lkb(dvh.bins, dvh.counts, param["d50"], param["m"], param["n"])
                records.append({
                    "PatientID": patient_id,
                    "Plan": plan_type,
                    "ROI": roi_name,
                    "Min Dose": round(min_dose, 2),
                    "Mean Dose": round(mean_dose, 2),
                    "Max Dose": round(max_dose, 2),
                    "NTCP": round(ntcp, 4)
                })
            except Exception as e:
                print(f"Error in {patient_id} - {roi_name}: {e}")

# 5. Lưu file CSV
summary_file = "CBCT_Breast_Left_Summary_WithStats.csv"
df = pd.DataFrame(records)
df.to_csv(summary_file, index=False)
print(f"✅ Saved summary to {summary_file}")

# 6. Phân tích thống kê
summary_stats = []
for roi in df["ROI"].unique():
    g = df[df["ROI"] == roi]
    d5 = g[g["Plan"] == "5MU"].set_index("PatientID")
    d10 = g[g["Plan"] == "10MU"].set_index("PatientID")
    ids = d5.index.intersection(d10.index)
    if len(ids) < 2:
        continue
    d5_dose, d10_dose = d5.loc[ids]["Mean Dose"], d10.loc[ids]["Mean Dose"]
    d5_ntcp, d10_ntcp = d5.loc[ids]["NTCP"], d10.loc[ids]["NTCP"]
    dose_t, dose_p = ttest_rel(d5_dose, d10_dose)
    ntcp_t, ntcp_p = ttest_rel(d5_ntcp, d10_ntcp)
    summary_stats.append({
        "ROI": roi,
        "Dose_5MU_Mean": round(d5_dose.mean(), 2),
        "Dose_5MU_Std": round(d5_dose.std(), 2),
        "Dose_10MU_Mean": round(d10_dose.mean(), 2),
        "Dose_10MU_Std": round(d10_dose.std(), 2),
        "Dose_p-value": round(dose_p, 4),
        "NTCP_5MU_Mean": round(d5_ntcp.mean(), 4),
        "NTCP_5MU_Std": round(d5_ntcp.std(), 4),
        "NTCP_10MU_Mean": round(d10_ntcp.mean(), 4),
        "NTCP_10MU_Std": round(d10_ntcp.std(), 4),
        "NTCP_p-value": round(ntcp_p, 4)
    })

stats_file = "CBCT_Breast_Left_Statistics.csv"
pd.DataFrame(summary_stats).to_csv(stats_file, index=False)
print(f"📊 Saved statistics to {stats_file}")

# 7. Vẽ biểu đồ boxplot
palette = {"5MU": "#2166AC", "10MU": "#B2182B"}
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="ROI", y="NTCP", hue="Plan", palette=palette)
plt.title("Comparison of NTCP Between 5 MU and 10 MU Imaging Plans for Breast (Left)", fontsize=12, fontweight='bold')
plt.xlabel("Organ at Risk (OAR)")
plt.ylabel("NTCP (%)")
plt.legend(title="CBCT Protocol", loc="upper right", frameon=True)
plt.tight_layout()
plt.savefig("CBCT_NTCP_Boxplot_WithStats_for_Breast_Left.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="ROI", y="Mean Dose", hue="Plan", palette=palette)
plt.title("Comparison of Mean CBCT Dose Delivered by 5 MU and 10 MU Imaging Plans for Breast (Left)", fontsize=12, fontweight='bold')
plt.xlabel("Organ at Risk (OAR)")
plt.ylabel("Mean Dose (Gy)")
plt.legend(title="CBCT Protocol", loc="upper right", frameon=True)
plt.tight_layout()
plt.savefig("CBCT_Dose_Boxplot_WithStats_for_Breast_Left.png", dpi=300)
plt.close()

print(f"✅ Done at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
