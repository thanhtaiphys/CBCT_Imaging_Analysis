import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk  # pip install pillow
from dicompylercore import dicomparser, dvhcalc
import numpy as np
from scipy.stats import norm

# ====== HÀM TÍNH NTCP LKB ======
def compute_ntcp_lkb(dose_bins, counts, d50, m, n):
    total_counts = np.sum(counts)
    if total_counts == 0:
        return 0.0
    v_fraction = np.array(counts) / total_counts
    doses = np.array(dose_bins[:len(v_fraction)])
    valid = v_fraction > 0
    if not np.any(valid):
        return 0.0
    deff = np.power(np.sum(v_fraction[valid] * (doses[valid] ** n)), 1 / n)
    t = (deff - d50) / (m * d50)
    return float(norm.cdf(t))


# ====== GIAO DIỆN CHÍNH ======
root = tk.Tk()
root.title("NTCP Calculator")
root.geometry("700x400")
root.resizable(False, False)

# ====== CHÈN LOGO ======
logo_path = "C:/RT_Project/VMPC.png"
logo_img = Image.open(logo_path).resize((80, 80))
logo_photo = ImageTk.PhotoImage(logo_img)
logo_label = tk.Label(root, image=logo_photo)
logo_label.grid(row=0, column=0, padx=10, pady=10, sticky="nw", rowspan=2)


# ====== CHÈN LOGO RÕ NÉT & GIỮ TỈ LỆ ======
logo_path = "C:/RT_Project/VMPC.png"
logo_img = Image.open(logo_path)

# Giới hạn kích thước tối đa (giữ tỉ lệ)
max_size = (120, 120)
logo_img.thumbnail(max_size, Image.LANCZOS)

logo_photo = ImageTk.PhotoImage(logo_img)
logo_label = tk.Label(root, image=logo_photo)
logo_label.grid(row=1, column=2, padx=(10, 20), pady=(10, 5), sticky="ne")


# ====== CHỌN FOLDER ======
def browse_folder():
    folder = filedialog.askdirectory()
    if folder:
        folder_entry.delete(0, tk.END)
        folder_entry.insert(0, folder)
        load_rois(folder)

tk.Label(root, text="Select Patient Folder:").grid(row=0, column=1, sticky="w")
folder_entry = tk.Entry(root, width=60)
folder_entry.grid(row=0, column=2, padx=5)
browse_button = tk.Button(root, text="Browse", command=browse_folder)
browse_button.grid(row=0, column=3)

# ====== DANH SÁCH ROI ======
tk.Label(root, text="Available ROIs:").grid(row=1, column=1, sticky="nw", pady=(10, 0))
roi_listbox = tk.Listbox(root, height=15, width=40)
roi_listbox.grid(row=1, column=2, padx=5, sticky="nw", pady=(10, 0))

roi_id_map = {}
rtstruct_parser = None
rtdose_parser = None

def load_rois(folder):
    global rtstruct_parser, rtdose_parser, roi_id_map
    files = os.listdir(folder)
    rs_file = [f for f in files if f.startswith("RS")][0]
    rd_file = [f for f in files if f.startswith("RD") or "CBCT" in f][0]

    rtstruct_parser = dicomparser.DicomParser(os.path.join(folder, rs_file))
    rtdose_parser = dicomparser.DicomParser(os.path.join(folder, rd_file))

    structs = rtstruct_parser.GetStructures()
    roi_listbox.delete(0, tk.END)
    roi_id_map = {}

    for id_, info in structs.items():
        name = info['name']
        roi_id_map[f"ID {id_}: {name}"] = id_
        roi_listbox.insert(tk.END, f"ID {id_}: {name}")


# ====== THÔNG SỐ SINH HỌC ======
param_frame = tk.Frame(root)
param_frame.grid(row=2, column=2, pady=10)

tk.Label(param_frame, text="TD50:").grid(row=0, column=0)
td50_entry = tk.Entry(param_frame, width=5)
td50_entry.insert(0, "45")
td50_entry.grid(row=0, column=1)

tk.Label(param_frame, text="m:").grid(row=0, column=2)
m_entry = tk.Entry(param_frame, width=5)
m_entry.insert(0, "0.3")
m_entry.grid(row=0, column=3)

tk.Label(param_frame, text="n:").grid(row=0, column=4)
n_entry = tk.Entry(param_frame, width=5)
n_entry.insert(0, "0.87")
n_entry.grid(row=0, column=5)

# ====== TÍNH TOÁN NTCP ======
def calculate_ntcp():
    try:
        selected = roi_listbox.get(roi_listbox.curselection())
        roi_id = int(selected.split()[1].replace(":", ""))

        td50 = float(td50_entry.get())
        m = float(m_entry.get())
        n = float(n_entry.get())

        dvh = dvhcalc.get_dvh(rtstruct_parser.ds, rtdose_parser.ds, roi_id)
        if dvh is None or not hasattr(dvh, "bins") or len(dvh.bins) == 0:
            raise ValueError("Không tìm thấy DVH.")

        ntcp = compute_ntcp_lkb(dvh.bins, dvh.counts, td50, m, n)
        messagebox.showinfo("NTCP Result", f"NTCP = {ntcp:.4f}")
    except Exception as e:
        messagebox.showerror("Lỗi tính NTCP", str(e))

calc_button = tk.Button(root, text="Calculate NTCP", command=calculate_ntcp)
calc_button.grid(row=2, column=3)

# ====== KHỞI CHẠY ======
root.mainloop()
