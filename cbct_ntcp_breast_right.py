# """
# Chương trình phân tích kế hoạch xạ trị CBCT ung thư vú
# =======================================================

# Tác giả: thanhtaiphys
# Ngày tạo: 2025-04-07 15:31:03 UTC

# Mô tả:
# Chương trình này phân tích dữ liệu CBCT (Cone Beam Computed Tomography) cho bệnh nhân 
# ung thư vú, tính toán và so sánh chỉ số NTCP (Normal Tissue Complication Probability) 
# giữa hai chế độ chụp CBCTCBCT 5MU và 10MU.

# Chức năng chính:
# 1. Đọc và xử lý dữ liệu DICOM từ kế hoạch xạ trị
# 2. Tính toán NTCP cho các cơ quan nguy cấp (OARs)
# 3. So sánh thống kê giữa chế độ 5MU và 10MU
# 4. Tạo báo cáo và biểu đồ kết quả

# Thư viện yêu cầu:
# - dicompylercore: Xử lý dữ liệu DICOM
# - numpy: Tính toán số học
# - pandas: Xử lý dữ liệu dạng bảng
# - scipy: Phân tích thống kê
# - matplotlib: Vẽ biểu đồ
# - seaborn: Tạo biểu đồ nâng cao
# """


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
root_dir = r"C:/RT_Project/data_cbct/Breast_CBCT/breast_right"

# 2. Danh sách ROI cần tính và ID tương ứng
roi_dict = {
    10: "Breast_CNTR",
    23: "Lung_IPSI",
    22: "Lung_CNTR",
    19: "Heart"
}

# 3. Tham số sinh học NTCP: #Cao et al. (2024) Lung_IPSI 24.5 
roi_params = {
    "Breast_CNTR": {"model": "logistic", "d50": 30.89, "gamma": 1.3},
    "Lung_IPSI":   {"model": "lkb",     "d50": 24.5, "m": 0.35, "n": 0.87},
    "Lung_CNTR":   {"model": "lkb",     "d50": 24.5, "m": 0.35, "n": 0.87},
    "Heart":       {"model": "lkb",     "d50": 48.0, "m": 0.1,  "n": 0.35}
}




# 4. Các hàm tính NTCP
def compute_ntcp_logistic(mean_dose, d50, gamma):
    t = gamma * (mean_dose - d50)
    return 1 / (1 + np.exp(-t))

# def compute_ntcp_lkb(dose_bins, counts, d50, m, n):
#     v_fraction = np.array(counts) / np.sum(counts)
#     doses = np.array(dose_bins[:len(v_fraction)])
#     deff = np.power(np.sum(v_fraction * (doses ** n)), 1 / n)
#     t = (deff - d50) / (m * d50)
#     ntcp = norm.cdf(t)
#     return float(ntcp)


# 1. Sửa hàm compute_ntcp_lkb để tránh chia cho 0
def compute_ntcp_lkb(dose_bins, counts, d50, m, n):
    total_counts = np.sum(counts)
    if total_counts == 0:
        return 0.0
    
    v_fraction = np.array(counts) / total_counts
    doses = np.array(dose_bins[:len(v_fraction)])
    
    # Tránh tính toán với giá trị 0
    valid_indices = v_fraction > 0
    if not np.any(valid_indices):
        return 0.0
        
    deff = np.power(np.sum(v_fraction[valid_indices] * (doses[valid_indices] ** n)), 1/n)
    t = (deff - d50) / (m * d50)
    ntcp = norm.cdf(t)
    return float(ntcp)

# 2. Sửa phần phân tích thống kê
def perform_statistical_analysis(group_5mu, group_10mu):
    """
    Thực hiện phân tích thống kê giữa hai nhóm
    
    Parameters:
    -----------
    group_5mu, group_10mu : pandas.Series
        Dữ liệu của nhóm 5MU và 10MU
    
    Returns:
    --------
    tuple : (mean_5mu, std_5mu, mean_10mu, std_10mu, p_value)
    """
    if len(group_5mu) < 2 or len(group_10mu) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan
        
    try:
        # Kiểm tra dữ liệu có trùng khớp không
        common_indices = group_5mu.index.intersection(group_10mu.index)
        if len(common_indices) < 2:
            return np.nan, np.nan, np.nan, np.nan, np.nan
            
        data_5mu = group_5mu.loc[common_indices]
        data_10mu = group_10mu.loc[common_indices]
        
        # Tính các thống kê cơ bản
        mean_5mu = np.mean(data_5mu)
        std_5mu = np.std(data_5mu)
        mean_10mu = np.mean(data_10mu)
        std_10mu = np.std(data_10mu)
        
        # Thực hiện paired t-test
        _, p_value = ttest_rel(data_5mu, data_10mu)
        
        return mean_5mu, std_5mu, mean_10mu, std_10mu, p_value
    except Exception as e:
        print(f"Lỗi trong phân tích thống kê: {e}")
        return np.nan, np.nan, np.nan, np.nan, np.nan

def analyze_and_print_statistics(df):
    # Phân tích thống kê chính
    summary_stats = []
    rois = df["ROI"].unique()

    for roi in rois:
        group = df[df["ROI"] == roi]
        data_5mu = group[group["Plan"] == "5MU"].set_index("PatientID")
        data_10mu = group[group["Plan"] == "10MU"].set_index("PatientID")
        
        # Phân tích Mean Dose
        dose_stats = perform_statistical_analysis(
            data_5mu["Mean Dose"],
            data_10mu["Mean Dose"]
        )
        
        # Phân tích NTCP
        ntcp_stats = perform_statistical_analysis(
            data_5mu["NTCP"],
            data_10mu["NTCP"]
        )
        
        summary_stats.append({
            "ROI": roi,
            "Dose_5MU_Mean": round(dose_stats[0], 2),
            "Dose_5MU_Std": round(dose_stats[1], 2),
            "Dose_10MU_Mean": round(dose_stats[2], 2),
            "Dose_10MU_Std": round(dose_stats[3], 2),
            "Dose_p-value": round(dose_stats[4], 4),
            "NTCP_5MU_Mean": round(ntcp_stats[0], 4),
            "NTCP_5MU_Std": round(ntcp_stats[1], 4),
            "NTCP_10MU_Mean": round(ntcp_stats[2], 4),
            "NTCP_10MU_Std": round(ntcp_stats[3], 4),
            "NTCP_p-value": round(ntcp_stats[4], 4)
        })

    df_stats = pd.DataFrame(summary_stats)

    # Thêm thông tin về số lượng mẫu
    print("\nThông tin thống kê:")
    for roi in rois:
        group = df[df["ROI"] == roi]
        n_5mu = len(group[group["Plan"] == "5MU"])
        n_10mu = len(group[group["Plan"] == "10MU"])
        print(f"{roi}: n(5MU)={n_5mu}, n(10MU)={n_10mu}")
    
    return df_stats







# 5. Tổng hợp kết quả cho từng bệnh nhân
records = []

for patient_id in os.listdir(root_dir):
    pat_path = os.path.join(root_dir, patient_id)
    if not os.path.isdir(pat_path):
        continue

    files = os.listdir(pat_path)
    struct_file = [f for f in files if f.startswith("RS")][0]
    dose_files = { "5MU": [f for f in files if "CBCT5" in f][0],
                   "10MU": [f for f in files if "CBCT10" in f][0] }

    rtstruct = dicomparser.DicomParser(os.path.join(pat_path, struct_file))

    for plan_type, dose_file in dose_files.items():
        rtdose = dicomparser.DicomParser(os.path.join(pat_path, dose_file))

        for roi_id, roi_name in roi_dict.items():
            try:
                dvh = dvhcalc.get_dvh(rtstruct.ds, rtdose.ds, roi_id)
                if dvh is None or not hasattr(dvh, "bins") or len(dvh.bins) == 0:
                    continue

                mean_dose = dvh.mean
                max_dose = dvh.max
                min_dose = dvh.min
                ntcp = "N/A"

                p = roi_params[roi_name]
                if p["model"] == "logistic":
                    ntcp = compute_ntcp_logistic(mean_dose, p["d50"], p["gamma"])
                elif p["model"] == "lkb":
                    ntcp = compute_ntcp_lkb(dvh.bins, dvh.counts, p["d50"], p["m"], p["n"])

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
                print(f"Error processing ROI {roi_name} for patient {patient_id}: {str(e)}")
                continue
# 6. Tạo DataFrame và lưu kết quả
df = pd.DataFrame(records)
out_path = "CBCT_Breast_Right_Summary_WithStats.csv"
df.to_csv(out_path, index=False)
print(df.head())
print(f"✅ Đã lưu kết quả vào: {out_path}")

# Phân tích thống kê
df_stats = analyze_and_print_statistics(df)
df.to_csv(out_path, index=False)
print(df.head())
print(f"✅ Đã lưu kết quả vào: {out_path}")

# 7. Phân tích thống kê so sánh giữa 5MU và 10MU
summary_stats = []
rois = df["ROI"].unique()

def perform_statistical_analysis(group_5mu, group_10mu):
    """
    Thực hiện phân tích thống kê giữa hai nhóm
    """
    if len(group_5mu) < 2 or len(group_10mu) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan
        
    try:
        common_indices = group_5mu.index.intersection(group_10mu.index)
        if len(common_indices) < 2:
            return np.nan, np.nan, np.nan, np.nan, np.nan
            
        data_5mu = group_5mu.loc[common_indices]
        data_10mu = group_10mu.loc[common_indices]
        
        mean_5mu = np.mean(data_5mu)
        std_5mu = np.std(data_5mu)
        mean_10mu = np.mean(data_10mu)
        std_10mu = np.std(data_10mu)
        
        _, p_value = ttest_rel(data_5mu, data_10mu)
        
        return mean_5mu, std_5mu, mean_10mu, std_10mu, p_value
    except Exception as e:
        print(f"Lỗi trong phân tích thống kê: {e}")
        return np.nan, np.nan, np.nan, np.nan, np.nan

# Thực hiện phân tích thống kê
for roi in rois:
    group = df[df["ROI"] == roi]
    data_5mu = group[group["Plan"] == "5MU"].set_index("PatientID")
    data_10mu = group[group["Plan"] == "10MU"].set_index("PatientID")
    
    dose_stats = perform_statistical_analysis(
        data_5mu["Mean Dose"],
        data_10mu["Mean Dose"]
    )
    
    ntcp_stats = perform_statistical_analysis(
        data_5mu["NTCP"],
        data_10mu["NTCP"]
    )
    
    summary_stats.append({
        "ROI": roi,
        "Dose_5MU_Mean": round(dose_stats[0], 2),
        "Dose_5MU_Std": round(dose_stats[1], 2),
        "Dose_10MU_Mean": round(dose_stats[2], 2),
        "Dose_10MU_Std": round(dose_stats[3], 2),
        "Dose_p-value": round(dose_stats[4], 4),
        "NTCP_5MU_Mean": round(ntcp_stats[0], 4),
        "NTCP_5MU_Std": round(ntcp_stats[1], 4),
        "NTCP_10MU_Mean": round(ntcp_stats[2], 4),
        "NTCP_10MU_Std": round(ntcp_stats[3], 4),
        "NTCP_p-value": round(ntcp_stats[4], 4)
    })

df_stats = pd.DataFrame(summary_stats)
print(df_stats.head())
df_stats.to_csv("CBCT_Breast_Right_Statistics.csv", index=False)
print("📊 Đã lưu thống kê so sánh giữa 5MU và 10MU.")


# 8. Vẽ biểu đồ boxplot NTCP với style khoa học
plt.figure(figsize=(8, 6))

# Cài đặt style cơ bản
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.2

# Màu sắc chuyên nghiệp cho bài báo
custom_palette = {"5MU": "#2166AC", "10MU": "#B2182B"}

# Vẽ boxplot NTCP và lưu axes để điều chỉnh
ax1 = sns.boxplot(data=df, 
                 x="ROI", 
                 y="NTCP", 
                 hue="Plan",  # Sửa từ "CBCT Protocol" thành "Plan"
                 palette=custom_palette, 
                 linewidth=1.5)

# Thêm lưới cho trục y
ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
ax1.xaxis.grid(False)

# Chỉnh sửa tiêu đề và nhãn
plt.title("Comparison of NTCP Between 5 MU and 10 MU Imaging Plans for Breast (Right)", 
         fontsize=12, fontweight='bold')
plt.xlabel("Organ at Risk (OAR)", fontsize=11, fontweight='bold')
plt.ylabel("NTCP (%)", fontsize=11, fontweight='bold')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Điều chỉnh legend

legend = plt.legend(title="CBCT Protocol",
                    fontsize=10,
                    title_fontsize=11,
                    loc='upper right',       # Đưa vào bên trong
                    frameon=True)
legend.get_frame().set_linewidth(1)
legend.get_frame().set_edgecolor('black')




# Lưu biểu đồ
plt.tight_layout()
# plt.show()
plt.savefig("CBCT_NTCP_Boxplot_WithStats_for_Breast_Right.png", 
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1)
plt.close()

# 9. Vẽ biểu đồ boxplot Mean Dose với style khoa học
plt.figure(figsize=(8, 6))

# Vẽ boxplot Mean Dose và lưu axes
ax2 = sns.boxplot(data=df, 
                 x="ROI", 
                 y="Mean Dose", 
                 hue="Plan",  # Sửa từ "CBCT Protocol" thành "Plan"
                 palette=custom_palette, 
                 linewidth=1.5)

# Thêm lưới cho trục y
ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
ax2.xaxis.grid(False)

# Chỉnh sửa tiêu đề và nhãn
plt.title("Comparison of Mean CBCT Dose Delivered by 5 MU and 10 MU Imaging Plans for Breast (Right)", 
         fontsize=12, fontweight='bold')
plt.xlabel("Organ at Risk (OAR)", fontsize=11, fontweight='bold')
plt.ylabel("Mean Dose (Gy)", fontsize=11, fontweight='bold')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Điều chỉnh legend

legend = plt.legend(title="CBCT Protocol",
                    fontsize=10,
                    title_fontsize=11,
                    loc='upper right',       # Đưa vào bên trong
                    frameon=True)
legend.get_frame().set_linewidth(1)
legend.get_frame().set_edgecolor('black')




# Lưu biểu đồ
plt.tight_layout()
# plt.show()
plt.savefig("CBCT_Dose_Boxplot_WithStats_for_Breast_Right.png", 
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1)
plt.close()

print(f"✅ Đã lưu biểu đồ NTCP và Mean Dose với style khoa học ({datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC)")