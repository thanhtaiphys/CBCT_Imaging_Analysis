import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
from dicompylercore import dicomparser, dvhcalc
from scipy.stats import norm

# NTCP models
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

class NTCPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NTCP Calculation GUI")

        self.folder_path = tk.StringVar()
        self.roi_list = []
        self.selected_roi_id = tk.IntVar()
        self.model_type = tk.StringVar(value="logistic")

        # Top: Logo
        logo_frame = tk.Frame(root)
        logo_frame.pack(pady=5)
        img = Image.open("C:/RT_Project/VMPC.png")
        img = img.resize((90, 90), Image.Resampling.LANCZOS)
        self.logo = ImageTk.PhotoImage(img)
        tk.Label(logo_frame, image=self.logo).pack()

        # Step 1: Folder selection
        tk.Label(root, text="Step 1: Select folder with RTDOSE and RTSTRUCTURE").pack()
        path_frame = tk.Frame(root)
        path_frame.pack(pady=2)
        tk.Entry(path_frame, textvariable=self.folder_path, width=50).pack(side=tk.LEFT)
        tk.Button(path_frame, text="Browse", command=self.browse_folder).pack(side=tk.LEFT, padx=5)

        # Step 2: ROI selection
        tk.Label(root, text="Step 2: Select ROI").pack(pady=(10, 0))
        self.roi_dropdown = ttk.Combobox(root, state="readonly")
        self.roi_dropdown.pack()

        # Step 3: Enter model parameters
        tk.Label(root, text="Step 3: Enter NTCP model parameters").pack(pady=(10, 0))
        param_frame = tk.Frame(root)
        param_frame.pack()

        tk.Label(param_frame, text="Model").grid(row=0, column=0)
        tk.Radiobutton(param_frame, text="Logistic", variable=self.model_type, value="logistic").grid(row=0, column=1)
        tk.Radiobutton(param_frame, text="LKB", variable=self.model_type, value="lkb").grid(row=0, column=2)

        self.entry_d50 = self.create_param_input(param_frame, "D50", 1)
        self.entry_gamma = self.create_param_input(param_frame, "Gamma (for Logistic)", 2)
        self.entry_m = self.create_param_input(param_frame, "m (for LKB)", 3)
        self.entry_n = self.create_param_input(param_frame, "n (for LKB)", 4)

        # Step 4: Calculate button
        tk.Button(root, text="Calculate NTCP", command=self.calculate_ntcp).pack(pady=10)

        # Output
        self.output_text = tk.Text(root, height=6, width=60)
        self.output_text.pack()

    def create_param_input(self, frame, label, row):
        tk.Label(frame, text=label).grid(row=row, column=0, sticky='w')
        entry = tk.Entry(frame, width=10)
        entry.grid(row=row, column=1, columnspan=2, sticky='w')
        return entry

    def browse_folder(self):
        path = filedialog.askdirectory()
        self.folder_path.set(path)
        self.update_roi_list()

    def update_roi_list(self):
        try:
            path = self.folder_path.get()
            struct_file = [f for f in os.listdir(path) if f.startswith("RS")][0]
            dcm_path = os.path.join(path, struct_file)
            parser = dicomparser.DicomParser(dcm_path)
            structures = parser.GetStructures()
            self.roi_list = [(k, v['name']) for k, v in structures.items()]
            self.roi_dropdown['values'] = [f"{k}: {v}" for k, v in self.roi_list]
            if self.roi_list:
                self.selected_roi_id.set(self.roi_list[0][0])
                self.roi_dropdown.current(0)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load structures: {e}")

    def calculate_ntcp(self):
        try:
            folder = self.folder_path.get()
            struct_file = [f for f in os.listdir(folder) if f.startswith("RS")][0]
            dose_file = [f for f in os.listdir(folder) if f.startswith("RD")][0]

            rtstruct = dicomparser.DicomParser(os.path.join(folder, struct_file))
            rtdose = dicomparser.DicomParser(os.path.join(folder, dose_file))

            roi_index = int(self.roi_dropdown.get().split(':')[0])
            dvh = dvhcalc.get_dvh(rtstruct.ds, rtdose.ds, roi_index)
            mean_dose = dvh.mean

            model = self.model_type.get()
            result = ""

            if model == "logistic":
                d50 = float(self.entry_d50.get())
                gamma = float(self.entry_gamma.get())
                ntcp = compute_ntcp_logistic(mean_dose, d50, gamma)
                result += f"NTCP (Logistic) = {ntcp:.4f}\n"
            else:
                d50 = float(self.entry_d50.get())
                m = float(self.entry_m.get())
                n = float(self.entry_n.get())
                ntcp = compute_ntcp_lkb(dvh.bins, dvh.counts, d50, m, n)
                result += f"NTCP (LKB) = {ntcp:.4f}\n"

            result += f"Mean Dose = {mean_dose:.2f} Gy\n"

            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, result)
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == '__main__':
    root = tk.Tk()
    app = NTCPApp(root)
    root.mainloop()
