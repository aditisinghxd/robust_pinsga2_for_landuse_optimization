import numpy as np
import sys
import os
import matplotlib
matplotlib.use("TkAgg")
import tkinter as tk
import matplotlib.pyplot as plt


import os
from datetime import datetime

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

from helper.genome_to_full_map import genome_to_full_landuse_map
from helper.plot_landuse_share import plot_landuse_share
from helper.timestamp_utils import timestamp_decorator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import colors

# --- Robustness imports ---
from helper.SAR_SYM_model import SAR_SYM_model
from helper.robust_adapter_ema import EmaRobustnessAdapter, RobustnessConfig
from helper.mapping import candidate_to_policy

# --- Optional: define uncertainties explicitly to match your tutorial ---
from ema_workbench import RealParameter, ScalarOutcome

import configparser

config = configparser.ConfigParser()
config.read("config.ini")

n_scenarios = config.getint("ROBUSTNESS", "n_scenarios")

def parse_range(value):
    low, high = map(float, value.split(","))
    return low, high


uncertainties = []

for name, value in config["UNCERTAINTIES"].items():
    low, high = parse_range(value)
    uncertainties.append(RealParameter(name, low, high))


'''uncertainties = [
    RealParameter("c", 3.0, 7.0),
    RealParameter("z", 0.1, 0.3),
    RealParameter("fet",  0.8, 1.2),
]'''

outcomes = [
    ScalarOutcome("S"),
    ScalarOutcome("Yield")
]

# --- Lazy singleton so GUI can call robustness anywhere without re-initializing ---
_ROBUST = None
candidate = None

def get_robust(candidate = candidate):
    global _ROBUST
    if _ROBUST is None:
        cfg = RobustnessConfig(
            n_scenarios= n_scenarios,          
            use_multiprocessing=False 
        )
        _ROBUST = EmaRobustnessAdapter(
            model_function=SAR_SYM_model,
            candidate_to_policy= candidate_to_policy, 
            uncertainties=uncertainties,
            #outcomes=outcomes,
            config=cfg,
            #candidate_maps= candidate
        )
    return _ROBUST

def robustness_of(candidate, policy_name):
    return get_robust()(candidate, policy_name)



original_map = np.load("processing/original_map.npy")
patchID_map= np.load("processing/patchID.npy")

landuse_labels = ["Cropland 1", "Cropland 2", "Cropland 3", "Cropland 4", "Cropland 5", "Pasture", "Forest", "Urban"]
landuse_colors = ["#fff68f", "#eee685", "#ffd700", "#eec800", "#ccaa01", "#228b22", "#a2cd5a", "#696969"]


cmap = colors.ListedColormap(landuse_colors)
bounds = [0.5 + i for i in range(9)]  # 0.5 to 8.5
norm = colors.BoundaryNorm(bounds, cmap.N)





@timestamp_decorator
def show_pairwise_decision(f1, f2, eta_F, x1, x2):
    #print(">> show_pairwise_decision called", flush=True)
    def on_decision(val):
        nonlocal decision
        decision = val
        #win.destroy()
        win.after(0, win.quit)      # cleanly exit mainloop()
        win.after(1, win.destroy)   # then destroy the window
        

    

    decision = None
    win = tk.Tk()
    win.title("Pairwise Comparison")
    

    x1_map = genome_to_full_landuse_map(genome=x1, patchID_map=patchID_map, original_map=original_map)
    x2_map = genome_to_full_landuse_map(genome=x2, patchID_map=patchID_map, original_map=original_map)
    x1_chart = plot_landuse_share(x1_map, "Land Use Share: Option A") 
    x2_chart = plot_landuse_share(x2_map, "Land Use Share: Option B") 

    x1_chart.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0)
    x2_chart.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0)


    #r1 , exp1, out1 = robustness_of(x1_map)
    #r2, exp2, out2 = robustness_of(x2_map)
    #print(r1)
    #print(r2)

    r1, exp1, out1 = robustness_of(x1_map, "Option A")
    r2, exp2, out2 = robustness_of(x2_map, "Option B")

    metric_order = [
    "regret_S_max", "regret_Yield_max",
    "S_median", "Yield_median",
    "regret_S_p90", "regret_Yield_p90"
    ]

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)

    lines = []
    lines.append(f"{'Metric':<22} {'Option A':>14} {'Option B':>14}")
    lines.append("-" * 52)

    for key in metric_order:
        val_a = fmt(r1.get(key, ""))
        val_b = fmt(r2.get(key, ""))
        lines.append(f"{key:<22} {val_a:>14} {val_b:>14}")

    robustness_table = "\n".join(lines)

    #text_a = f"Option A — Yield mean: {r1['Yield_mean']:.2f}, S mean: {r1['S_mean']:.2f}"
    #text_b = f"Option B — Yield mean: {r2['Yield_mean']:.2f}, S mean: {r2['S_mean']:.2f}"



    # --- Create frame for composite layout ---
    main_frame = tk.Frame(win)
    main_frame.grid()

    # -- Left: Pareto Plot -- Column 0, Row 0
    fig_pf, ax_pf = plt.subplots(figsize=(5, 4), constrained_layout=True)
    #fig_pf.subplots_adjust(bottom=0.15, left=0.15, right=0.95, top=0.9)
    ax_pf.scatter(-eta_F[:, 0], -eta_F[:, 1], color='#B0B0B0', label="Pareto Front")
    ax_pf.scatter(-f1[0], -f1[1], color="#BE0524", marker='s', label="Option A")
    ax_pf.scatter(-f2[0], -f2[1], color="#0A089C", marker='^',label="Option B")
    ax_pf.set_title("Pareto Front with Highlighted Choices")
    ax_pf.set_xlabel("Crop Yield")
    ax_pf.set_ylabel("Forest Species Richness")
    ax_pf.legend()
    ax_pf.set_xlim(0, 120)
    ax_pf.set_ylim(8, 15)

    canvas_pf = FigureCanvasTkAgg(fig_pf, master=main_frame)
    canvas_pf.draw()
    canvas_pf.get_tk_widget().grid(row=0, column=0)

    # -- Right: Map A and Map B stacked vertically -- Column 1, row 0
    map_frame = tk.Frame(main_frame)
    map_frame.grid(row=0, column=1)

    fig_maps, (ax_map1, ax_map2) = plt.subplots(2, 1, figsize=(4, 4))
    fig_maps.subplots_adjust(left=0, right=1, top=0.9, bottom=0)


    ax_map1.imshow(x1_map, cmap=cmap, norm=norm)
    ax_map1.set_title("Option A Map")
    ax_map1.axis('off')


    ax_map2.imshow(x2_map, cmap=cmap, norm=norm)
    ax_map2.set_title("Option B Map")
    ax_map2.axis('off')

    canvas_maps = FigureCanvasTkAgg(fig_maps, master=map_frame)
    canvas_maps.draw()
    canvas_maps.get_tk_widget().pack()



    # -- Legend on the right -- Column 2, row 0
    legend_frame = tk.Frame(main_frame)
    legend_frame.grid(row=0, column=2)
    tk.Label(legend_frame, text="Legend", font=('Arial', 10, 'bold')).pack()

    for color, label in zip(landuse_colors, landuse_labels):
        entry = tk.Frame(legend_frame)
        entry.pack(anchor='w', pady=2)
        color_box = tk.Label(entry, width=2, height=1, bg=color)
        color_box.pack(side='left', padx=(0, 5))
        tk.Label(entry, text=label).pack(side='left')


    # -- Pie Charts on far right -- Column 3, row 0
    #x1_chart.set_size_inches(2, 2)  
    #x2_chart.set_size_inches(2, 2)

    pie1_frame = tk.Frame(main_frame)
    pie1_frame.grid(row=0, column=3, sticky='n', pady=40)

    canvas_pie1 = FigureCanvasTkAgg(x1_chart, master=pie1_frame)
    canvas_pie1.draw()
    canvas_pie1.get_tk_widget().pack()

    pie2_frame = tk.Frame(main_frame)
    pie2_frame.grid(row=0, column=4, sticky='n', pady=40)

    canvas_pie2 = FigureCanvasTkAgg(x2_chart, master=pie2_frame)
    canvas_pie2.draw()
    canvas_pie2.get_tk_widget().pack()

    # Fitness display under Pareto front and selection buttons-- column 0, row 1
    info_frame = tk.Frame(main_frame)
    info_frame.grid(row=1, column=0)

    tk.Label(info_frame, text=f"Option A: {-f1}").pack()
    tk.Label(info_frame, text=f"Option B: {-f2}").pack()
    #tk.Label(info_frame, text=text_a).pack(anchor="w")
    #tk.Label(info_frame, text=text_b).pack(anchor="w")

    tk.Button(info_frame, text="Prefer A", command=lambda: on_decision("a")).pack(side='left')
    tk.Button(info_frame, text="Prefer B", command=lambda: on_decision("b")).pack(side='left')
    tk.Button(info_frame, text="Can't make a decision", command=lambda: on_decision("c")).pack(side='left')

    from ema_workbench.analysis import pairs_plotting

    robust_frame_A = tk.Frame(main_frame)
    robust_frame_A.grid(row=2, column=1, padx=10, pady=10, sticky='n')

    plt.figure()
    figA, axesA = pairs_plotting.pairs_scatter(
        exp1, out1, group_by="policy", legend=True
    )

    figA.set_size_inches(4.5, 4.5)

    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # #figA.savefig(os.path.join(OUTPUT_DIR, f"{timestamp}_pairwise_A.png"), dpi=300, bbox_inches="tight")

    canvasA = FigureCanvasTkAgg(figA, master=robust_frame_A)
    canvasA.draw()
    canvasA.get_tk_widget().pack()


    robust_frame_B = tk.Frame(main_frame)
    robust_frame_B.grid(row=2, column=3, padx=10, pady=10, sticky='n')

    plt.figure()
    figB, axesB = pairs_plotting.pairs_scatter(
        exp2, out2, group_by="policy", legend=True
    )

    figB.set_size_inches(4.5, 4.5)

    #figB.savefig(os.path.join(OUTPUT_DIR, f"{timestamp}_pairwise_B.png"), dpi=300, bbox_inches="tight" )


    # --- Robustness table display (Row 2, Column 0) ---
    table_frame = tk.Frame(main_frame)
    table_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nw")

    tk.Label(
        table_frame,
        text=robustness_table,
        font=("Courier New", 9),
        justify="left",
        anchor="w"
    ).pack()

    canvasB = FigureCanvasTkAgg(figB, master=robust_frame_B)
    canvasB.draw()
    canvasB.get_tk_widget().pack()

    
    win.mainloop()
    plt.close('all')
    return decision

