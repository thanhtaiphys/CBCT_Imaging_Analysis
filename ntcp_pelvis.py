#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NTCP Analysis Program for Pelvis Patients after MV-CBCT

This program analyzes CBCT (Cone Beam Computed Tomography) data for pelvis cancer patients,
calculating and comparing NTCP (Normal Tissue Complication Probability) between
5MU and 10MU CBCT imaging protocols.

Key Features:
1. Reads and processes DICOM data from radiation treatment plans
2. Calculates NTCP for multiple Organs at Risk (OARs)
3. Performs statistical comparison between 5MU and 10MU protocols
4. Generates reports and visualization plots

Required Libraries:
- dicompylercore: DICOM data processing
- numpy: Numerical computations
- pandas: Tabular data processing
- scipy: Statistical analysis
- matplotlib: Basic plotting
- seaborn: Advanced visualization

Author: thanhtaiphys
Created: 2025-04-17 11:21:50 UTC
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, norm
import matplotlib.pyplot as plt
import seaborn as sns
from dicompylercore import dicomparser, dvhcalc

# Configuration constants
ROOT_DIR = r"C:/RT_Project/data_cbct/Pelvis_CBCT"

# ROI definitions and mappings
ROI_NAMES = ["Bladder", "Bowel_Bag", "Femur_L", "Femur_R", "Rectum"]

ROI_MAPPING = {
    "Rectum": ["Rectum", "Mesorectal"]
}

# NTCP model parameters for each ROI
ROI_PARAMS = {
    "Bladder":    {"model": "lkb", "d50": 80.0, "m": 0.15, "n": 0.5},
    "Bowel_Bag":  {"model": "lkb", "d50": 55.0, "m": 0.18, "n": 0.15},
    "Femur_L":    {"model": "lkb", "d50": 50.0, "m": 0.1,  "n": 0.1},
    "Femur_R":    {"model": "lkb", "d50": 50.0, "m": 0.1,  "n": 0.1},
    "Rectum":     {"model": "lkb", "d50": 70.0, "m": 0.12, "n": 0.25}
}

# Visualization settings
PLOT_CONFIG = {
    "figsize": (10, 7),
    "dpi": 300,
    "palette": {"5MU": "#2166AC", "10MU": "#B2182B"},
    "title_fontsize": 12,
    "title_pad": 20
}

def standardize_roi_name(roi_name: str, mapping: dict) -> str:
    """
    Standardize ROI names based on defined mappings.

    Args:
        roi_name (str): Original ROI name
        mapping (dict): Dictionary mapping standard names to lists of aliases

    Returns:
        str: Standardized ROI name
    """
    for std_name, aliases in mapping.items():
        if roi_name in aliases:
            return std_name
    return roi_name

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
    plt.figure(figsize=PLOT_CONFIG["figsize"])
    sns.boxplot(data=data, x="ROI", y=y_variable, hue="Plan", 
                palette=PLOT_CONFIG["palette"])
    
    plt.title(title, fontsize=PLOT_CONFIG["title_fontsize"], 
              fontweight='bold', pad=PLOT_CONFIG["title_pad"])
    plt.xlabel("Organ at Risk (OAR)")
    plt.ylabel(ylabel)
    plt.legend(title="CBCT Protocol", loc="upper right", frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    plt.close()

def process_patient_data(patient_id: str, patient_path: str) -> list:
    """
    Process DICOM data and calculate NTCP for a single patient.

    Args:
        patient_id (str): Patient identifier
        patient_path (str): Path to patient's DICOM files

    Returns:
        list: List of dictionaries containing processed patient data
    """
    patient_records = []
    files = os.listdir(patient_path)
    
    try:
        struct_file = [f for f in files if f.startswith("RS")][0]
        dose_files = {
            "5MU": [f for f in files if "CBCT5" in f][0],
            "10MU": [f for f in files if "CBCT10" in f][0]
        }
    except IndexError:
        print(f"Missing required DICOM files for patient {patient_id}")
        return patient_records

    rtstruct = dicomparser.DicomParser(os.path.join(patient_path, struct_file))
    structure_dict = rtstruct.GetStructures()
    roi_name_to_id = {info['name']: rid for rid, info in structure_dict.items()}

    for plan_type, dose_file in dose_files.items():
        rtdose = dicomparser.DicomParser(os.path.join(patient_path, dose_file))
        
        for roi in ROI_NAMES:
            roi_id = None
            for actual_name in roi_name_to_id:
                if actual_name in ROI_MAPPING.get(roi, [roi]):
                    roi_id = roi_name_to_id[actual_name]
                    break
                    
            if roi_id is None:
                print(f"⚠️ Missing ROI '{roi}' in {patient_id}")
                continue
                
            try:
                dvh = dvhcalc.get_dvh(rtstruct.ds, rtdose.ds, roi_id)
                if dvh is None or len(dvh.bins) == 0:
                    continue
                    
                param = ROI_PARAMS[roi]
                ntcp = (compute_ntcp_logistic(dvh.mean, param["d50"], param["gamma"]) 
                       if param["model"] == "logistic"
                       else compute_ntcp_lkb(dvh.bins, dvh.counts, 
                                           param["d50"], param["m"], param["n"]))
                
                patient_records.append({
                    "PatientID": patient_id,
                    "Plan": plan_type,
                    "ROI": roi,
                    "Min Dose": round(dvh.min, 2),
                    "Mean Dose": round(dvh.mean, 2),
                    "Max Dose": round(dvh.max, 2),
                    "NTCP": round(ntcp, 4)
                })
                
            except Exception as e:
                print(f"Error processing {patient_id} - {roi}: {e}")
                
    return patient_records

def calculate_statistics(df: pd.DataFrame) -> list:
    """
    Calculate statistical comparisons between 5MU and 10MU protocols.

    Args:
        df (pd.DataFrame): DataFrame containing processed patient data

    Returns:
        list: List of dictionaries containing statistical analysis results
    """
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
    
    return summary_stats

def main():
    """Main execution function for the NTCP analysis program."""
    # Process patient data
    records = []
    for patient_id in os.listdir(ROOT_DIR):
        pat_path = os.path.join(ROOT_DIR, patient_id)
        if not os.path.isdir(pat_path):
            continue
        records.extend(process_patient_data(patient_id, pat_path))

    # Create summary DataFrame
    df = pd.DataFrame(records)
    df.to_csv("CBCT_Pelvis_Summary_WithStats.csv", index=False)
    print("\nSummary Statistics:")
    print(df.head())

    # Perform statistical analysis
    summary_stats = calculate_statistics(df)
    stats_df = pd.DataFrame(summary_stats)
    stats_df.to_csv("CBCT_Pelvis_Statistics.csv", index=False)
    print("\nStatistical Analysis:")
    print(stats_df)

    # Create visualization plots
    create_boxplot(
        data=df,
        y_variable="NTCP",
        title="Comparison of NTCP Between 5 MU and 10 MU Imaging Plans for Pelvis",
        ylabel="NTCP (%)",
        output_filename="CBCT_NTCP_Boxplot_WithStats_for_Pelvis.png"
    )

    create_boxplot(
        data=df,
        y_variable="Mean Dose",
        title="Comparison of Mean CBCT Dose Delivered by 5 MU and 10 MU Imaging Plans for Pelvis",
        ylabel="Mean Dose (Gy)",
        output_filename="CBCT_Dose_Boxplot_WithStats_for_Pelvis.png"
    )

    print(f"\n✅ Analysis completed at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

if __name__ == "__main__":
    main()