#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NTCP Analysis Program for Right Breast after MV-CBCT (5MU and 10MU)

This script analyzes Normal Tissue Complication Probability (NTCP) for various organs
at risk (OARs) after MV-CBCT imaging with 5MU and 10MU protocols in right breast
radiotherapy treatments.

Author: thanhtaiphys
Last Updated: 2025-04-11
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, norm
import matplotlib.pyplot as plt
import seaborn as sns
from dicompylercore import dicomparser, dvhcalc

# Global constants and configuration
ROOT_DIR = r"C:/RT_Project/data_cbct/Breast_CBCT/breast_right"

# Define ROIs and their NTCP model parameters
ROI_NAMES = ["Breast_CNTR", "Lung_IPSI", "Lung_CNTR", "Heart"]

ROI_PARAMS = {
    "Breast_CNTR": {"model": "logistic", "d50": 30.89, "gamma": 1.3},
    "Lung_IPSI":   {"model": "lkb",     "d50": 24.5, "m": 0.35, "n": 0.87},
    "Lung_CNTR":   {"model": "lkb",     "d50": 24.5, "m": 0.35, "n": 0.87},
    "Heart":       {"model": "lkb",     "d50": 48.0,  "m": 0.1,  "n": 0.35}
}

# Figure styling constants
FIGURE_SIZE = (10, 7)
PLOT_PALETTE = {"5MU": "#2166AC", "10MU": "#B2182B"}
DPI = 300

def compute_ntcp_logistic(mean_dose: float, d50: float, gamma: float) -> float:
    """
    Calculate NTCP using the Logistic model.

    Args:
        mean_dose (float): Mean dose to the organ (Gy)
        d50 (float): Dose for 50% complication probability (Gy)
        gamma (float): Slope parameter of the dose-response curve

    Returns:
        float: NTCP value between 0 and 1
    """
    t = gamma * (mean_dose - d50)
    return 1 / (1 + np.exp(-t))

def compute_ntcp_lkb(dose_bins: np.ndarray, counts: np.ndarray, 
                    d50: float, m: float, n: float) -> float:
    """
    Calculate NTCP using the Lyman-Kutcher-Burman (LKB) model.

    Args:
        dose_bins (np.ndarray): Array of dose values
        counts (np.ndarray): Array of volume counts for each dose bin
        d50 (float): Dose for 50% complication probability (Gy)
        m (float): Slope parameter
        n (float): Volume effect parameter

    Returns:
        float: NTCP value between 0 and 1
    """
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

def create_boxplot(data: pd.DataFrame, y_variable: str, title: str, 
                  ylabel: str, output_filename: str) -> None:
    """
    Create and save a boxplot comparing 5MU and 10MU protocols.

    Args:
        data (pd.DataFrame): DataFrame containing the plot data
        y_variable (str): Column name for y-axis values
        title (str): Plot title
        ylabel (str): Y-axis label
        output_filename (str): Name of output file to save the plot
    """
    plt.figure(figsize=FIGURE_SIZE)
    sns.boxplot(data=data, x="ROI", y=y_variable, hue="Plan", palette=PLOT_PALETTE)
    
    plt.title(title, fontsize=12, fontweight='bold', pad=20)
    plt.xlabel("Organ at Risk (OAR)")
    plt.ylabel(ylabel)
    plt.legend(title="CBCT Protocol", loc="upper right", frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=DPI, bbox_inches='tight')
    plt.close()

# Main execution block
if __name__ == "__main__":
    # Process DICOM data and calculate NTCP
    records = []
    for patient_id in os.listdir(ROOT_DIR):
        pat_path = os.path.join(ROOT_DIR, patient_id)
        if not os.path.isdir(pat_path):
            continue

        # Find required DICOM files
        files = os.listdir(pat_path)
        try:
            struct_file = [f for f in files if f.startswith("RS")][0]
            dose_files = {
                "5MU": [f for f in files if "CBCT5" in f][0],
                "10MU": [f for f in files if "CBCT10" in f][0]
            }
        except IndexError:
            print(f"Missing required DICOM files for patient {patient_id}")
            continue

        # Process structure and dose data
        rtstruct = dicomparser.DicomParser(os.path.join(pat_path, struct_file))
        structure_dict = rtstruct.GetStructures()
        roi_name_to_id = {info['name']: rid for rid, info in structure_dict.items()}

        for plan_type, dose_file in dose_files.items():
            rtdose = dicomparser.DicomParser(os.path.join(pat_path, dose_file))
            
            for roi_name in ROI_NAMES:
                try:
                    if roi_name not in roi_name_to_id:
                        print(f"⚠️ Missing ROI '{roi_name}' in {patient_id}")
                        continue
                        
                    roi_id = roi_name_to_id[roi_name]
                    dvh = dvhcalc.get_dvh(rtstruct.ds, rtdose.ds, roi_id)
                    
                    if dvh is None or len(dvh.bins) == 0:
                        continue
                        
                    # Calculate dose statistics and NTCP
                    param = ROI_PARAMS[roi_name]
                    ntcp = (compute_ntcp_logistic(dvh.mean, param["d50"], param["gamma"]) 
                           if param["model"] == "logistic"
                           else compute_ntcp_lkb(dvh.bins, dvh.counts, 
                                               param["d50"], param["m"], param["n"]))
                    
                    records.append({
                        "PatientID": patient_id,
                        "Plan": plan_type,
                        "ROI": roi_name,
                        "Min Dose": round(dvh.min, 2),
                        "Mean Dose": round(dvh.mean, 2),
                        "Max Dose": round(dvh.max, 2),
                        "NTCP": round(ntcp, 4)
                    })
                    
                except Exception as e:
                    print(f"Error processing {patient_id} - {roi_name}: {e}")

    # Create summary DataFrame and save to CSV
    df = pd.DataFrame(records)
    df.to_csv("CBCT_Breast_Right_Summary_WithStats.csv", index=False)
    print("\nSummary Statistics:")
    print(df.head())

    # Perform statistical analysis
    summary_stats = []
    for roi in df["ROI"].unique():
        g = df[df["ROI"] == roi]
        d5 = g[g["Plan"] == "5MU"].set_index("PatientID")
        d10 = g[g["Plan"] == "10MU"].set_index("PatientID")
        ids = d5.index.intersection(d10.index)
        
        if len(ids) < 2:
            continue
            
        # Calculate paired t-test statistics
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

    # Save statistical analysis results
    stats_df = pd.DataFrame(summary_stats)
    stats_df.to_csv("CBCT_Breast_Right_Statistics.csv", index=False)
    print("\nStatistical Analysis:")
    print(stats_df)

    # Create visualization plots
    create_boxplot(
        data=df,
        y_variable="NTCP",
        title="Comparison of NTCP Between 5 MU and 10 MU Imaging Plans for Breast (Right)",
        ylabel="NTCP (%)",
        output_filename="CBCT_NTCP_Boxplot_WithStats_for_Breast_Right.png"
    )

    create_boxplot(
        data=df,
        y_variable="Mean Dose",
        title="Comparison of Mean CBCT Dose Delivered by 5 MU and 10 MU Imaging Plans for Breast (Right)",
        ylabel="Mean Dose (Gy)",
        output_filename="CBCT_Dose_Boxplot_WithStats_for_Breast_Right.png"
    )

    print(f"\n✅ Analysis completed at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")