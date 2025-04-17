#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Patient Information Extractor from DICOM Files

This script extracts patient demographic and treatment information from DICOM files
in a structured radiotherapy dataset. It processes RS (RT Structure Set) and RP 
(RT Plan) files to gather patient age, sex, treatment details, and CBCT protocols.

Author: Dr. Thanh Tai Duong
Created: 2025-04-17
"""

import os
import pydicom
from datetime import datetime
import pandas as pd
from typing import List, Dict, Tuple, Optional


def parse_dicom_dates(birth_date_str: str, study_date_str: str) -> Optional[int]:
    """
    Calculate patient age from DICOM format dates.
    
    Args:
        birth_date_str: Patient's birth date in DICOM format (YYYYMMDD)
        study_date_str: Study date in DICOM format (YYYYMMDD)
        
    Returns:
        int: Age in years, or None if dates are invalid
    """
    try:
        if birth_date_str and study_date_str:
            birth_date = datetime.strptime(birth_date_str, "%Y%m%d")
            study_date = datetime.strptime(study_date_str, "%Y%m%d")
            return (study_date - birth_date).days // 365
    except ValueError as e:
        print(f"⚠️ Error parsing dates: {e}")
    return None


def extract_rt_plan_info(plan_path: str) -> Tuple[Optional[float], Optional[int], Optional[float]]:
    """
    Extract dose and fractionation information from RT Plan file.
    
    Args:
        plan_path: Path to RTPLAN DICOM file
        
    Returns:
        tuple: (total_dose, number_of_fractions, dose_per_fraction)
    """
    try:
        plan = pydicom.dcmread(plan_path)
        total_dose = None
        num_frac = None
        
        if hasattr(plan, "FractionGroupSequence"):
            num_frac = plan.FractionGroupSequence[0].NumberOfFractionsPlanned
            
        if hasattr(plan, "DoseReferenceSequence"):
            total_dose = plan.DoseReferenceSequence[0].TargetPrescriptionDose
            
        frac_dose = round(float(total_dose) / int(num_frac), 2) if total_dose and num_frac else None
        return total_dose, num_frac, frac_dose
        
    except Exception as e:
        print(f"⚠️ Error reading RT Plan: {e}")
        return None, None, None


def get_cancer_subfolders(root_dir: str) -> List[Tuple[str, str]]:
    """
    Generate list of cancer type folders to process.
    
    Args:
        root_dir: Root directory containing cancer type folders
        
    Returns:
        list: Tuples of (cancer_type, folder_path)
    """
    subfolders = []
    for cancer_type in os.listdir(root_dir):
        cancer_path = os.path.join(root_dir, cancer_type)
        
        if cancer_type == "Breast_CBCT":
            for sub in ["breast_left", "breast_right"]:
                subfolders.append((sub, os.path.join(cancer_path, sub)))
        else:
            subfolders.append((cancer_type, cancer_path))
    return subfolders


def process_patient_folder(patient_path: str, patient_id: str, cancer_type: str) -> Optional[Dict]:
    """
    Process individual patient folder to extract required information.
    
    Args:
        patient_path: Path to patient folder
        patient_id: Patient identifier
        cancer_type: Type of cancer being treated
        
    Returns:
        dict: Patient information dictionary or None if error occurs
    """
    try:
        # Find structure and plan files
        struct_file = next((f for f in os.listdir(patient_path) if f.startswith("RS")), None)
        plan_file = next((f for f in os.listdir(patient_path) if f.startswith("RP")), None)
        
        if not struct_file:
            return None
            
        # Read structure set
        ds = pydicom.dcmread(os.path.join(patient_path, struct_file))
        birth_date_str = ds.get("PatientBirthDate", "")
        study_date_str = ds.get("StudyDate", "")
        sex = ds.get("PatientSex", "")
        
        # Calculate age
        age = parse_dicom_dates(birth_date_str, study_date_str)
        
        # Check for CBCT protocols
        files = os.listdir(patient_path)
        has_5mu = any("CBCT5" in f for f in files)
        has_10mu = any("CBCT10" in f for f in files)
        
        # Get RT Plan information
        total_dose, num_frac, frac_dose = (None, None, None)
        if plan_file:
            total_dose, num_frac, frac_dose = extract_rt_plan_info(
                os.path.join(patient_path, plan_file)
            )
            
        return {
            "Cancer_Type": cancer_type,
            "Patient_ID": patient_id,
            "Sex": sex,
            "Age_at_Treatment": age,
            "BirthDate": birth_date_str,
            "StudyDate": study_date_str,
            "Has_CBCT_5MU": has_5mu,
            "Has_CBCT_10MU": has_10mu,
            "Total_Dose_cGy": total_dose,
            "Num_Fractions": num_frac,
            "Fraction_Dose_cGy": frac_dose
        }
        
    except Exception as e:
        print(f"⚠️ Error processing patient {patient_id}: {e}")
        return None


def main():
    """Main execution function."""
    # Root directory containing RT data
    root_dir = r"C:\RT_Project\data_cbct"
    
    # Process all cancer types
    patient_info = []
    for cancer_type, path in get_cancer_subfolders(root_dir):
        if not os.path.isdir(path):
            continue
            
        for patient_id in os.listdir(path):
            patient_path = os.path.join(path, patient_id)
            if not os.path.isdir(patient_path):
                continue
                
            patient_data = process_patient_folder(patient_path, patient_id, cancer_type)
            if patient_data:
                patient_info.append(patient_data)
    
    # Create and save DataFrame
    df = pd.DataFrame(patient_info)
    df.to_csv("Patient_Info.csv", index=False)
    print("✅ Saved: Patient_Info.csv")
    print("\nFirst few records:")
    print(df.head())


if __name__ == "__main__":
    main()