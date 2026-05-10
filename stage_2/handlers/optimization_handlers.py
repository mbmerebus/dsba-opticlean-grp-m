import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, '..')
from opticlean_optimization import solve_location_model  # noqa: E402

DATA_PATH = "../data/case-opticlean-data-dsba"

_SITE_COLORS = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
    "#FF7F00", "#A65628", "#F781BF", "#AAAAAA",
]
_UNASSIGNED_COLOR = "#DDDDDD"


def load_inputs():
    potential     = pd.read_excel(f"{DATA_PATH}/potential_clients.xlsx")
    sites         = pd.read_excel(f"{DATA_PATH}/candidate_sites.xlsx")
    demand_matrix = np.load("demand_matrix.npy")
    return demand_matrix, potential, sites


def run_optimization(demand_matrix, sites, budget=850, verbose=True):
    return solve_location_model(demand_matrix, sites, budget=budget, verbose=verbose)


def print_summary(result):
    ss = result["selected_sites"]
    print(f"Total captured demand : {result['objective_value']:.1f} units/week")
    print(f"Budget used           : {result['budget_used']:.0f} / 850 k€")
    print(f"Clients assigned      : {len(result['assignments'])} / 200")
    print(f"Clients unassigned    : {len(result['unassigned_clients'])}")
    print()
    display_cols = ["site_id", "format", "capacity", "cost", "used_capacity", "capacity_utilization"]
    print(ss[display_cols].to_string(index=False))


def plot_assignment_map(result, potential_df, sites_df):
    """Map of client assignments. Square = large store, triangle = small store, X = closed site."""
    assignments = result["selected_sites"]
    site_list   = sorted(assignments["site_id"].tolist())
    color_map   = {s: _SITE_COLORS[i] for i, s in enumerate(site_list)}

    assigned_to = {
        row["client_id"]: row["site_id"]
        for _, row in result["assignments"].iterrows()
    }

    fig, ax = plt.subplots(figsize=(9, 8))

    # unassigned clients
    if result["unassigned_clients"]:
        sub = potential_df.loc[result["unassigned_clients"]]
        ax.scatter(sub["lon"], sub["lat"], color=_UNASSIGNED_COLOR, s=18, alpha=0.7,
                   zorder=2, label="Unassigned")

    # assigned clients, colored by site
    for sid in site_list:
        client_ids = [c for c, s in assigned_to.items() if s == sid]
        sub = potential_df.loc[client_ids]
        ax.scatter(sub["lon"], sub["lat"], color=color_map[sid], s=18, alpha=0.75, zorder=3)

    # all candidate sites
    for j, row in sites_df.iterrows():
        site_row = result["selected_sites"][result["selected_sites"]["site_id"] == j]
        if site_row.empty:
            ax.scatter(row["lon"], row["lat"], marker="x", s=80, color="black", zorder=5)
        else:
            fmt    = site_row.iloc[0]["format"]
            marker = "s" if fmt == "large" else "^"
            ax.scatter(row["lon"], row["lat"], marker=marker, s=220,
                       color=color_map[j], edgecolors="black", linewidths=1.5, zorder=6)
            ax.annotate(
                f"S{j + 1} ({fmt[0].upper()})",
                (row["lon"], row["lat"]),
                textcoords="offset points", xytext=(6, 5), fontsize=8,
            )

    patches = [mpatches.Patch(color=color_map[s], label=f"Site {s + 1}") for s in site_list]
    patches.append(mpatches.Patch(color=_UNASSIGNED_COLOR, label="Unassigned"))
    ax.legend(handles=patches, loc="upper left", fontsize=8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Stage 2: Client Assignments and Opened Stores")
    plt.tight_layout()
    return fig


def plot_capacity_utilization(result):
    ss     = result["selected_sites"]
    labels = [f"Site {r['site_id'] + 1}\n({r['format'][0].upper()})" for _, r in ss.iterrows()]
    x      = range(len(ss))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, ss["capacity"],      color="#BBBBBB", alpha=0.6, label="Total capacity")
    ax.bar(x, ss["used_capacity"], color="#4878D0", alpha=0.85, label="Used capacity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Weekly demand units")
    ax.set_title("Capacity Utilization per Opened Site")
    ax.legend()
    plt.tight_layout()
    return fig


def compare_results(r1, r2, label1="Stage 1", label2="Stage 2"):
    """Print a side-by-side summary comparing two optimization results."""
    def sites_str(r):
        return ", ".join(f"S{s+1}({fmt[0].upper()})"
                         for _, row in r["selected_sites"].iterrows()
                         for s, fmt in [(row["site_id"], row["format"])])

    rows = [
        ("Total demand (units/week)", f"{r1['objective_value']:.1f}", f"{r2['objective_value']:.1f}"),
        ("Budget used (k€)",          f"{r1['budget_used']:.0f} / 850", f"{r2['budget_used']:.0f} / 850"),
        ("Clients assigned",          f"{len(r1['assignments'])} / 200", f"{len(r2['assignments'])} / 200"),
        ("Clients unassigned",        str(len(r1['unassigned_clients'])), str(len(r2['unassigned_clients']))),
        ("Stores opened",             str(len(r1['selected_sites'])), str(len(r2['selected_sites']))),
        ("Sites selected",            sites_str(r1), sites_str(r2)),
    ]

    col_w = max(len(label1), len(label2), 22)
    header = f"{'Metric':<30}  {label1:<{col_w}}  {label2:<{col_w}}"
    print(header)
    print("-" * len(header))
    for metric, v1, v2 in rows:
        print(f"{metric:<30}  {v1:<{col_w}}  {v2:<{col_w}}")


def plot_comparison(r1, r2, label1="Stage 1", label2="Stage 2"):
    """Bar chart comparing key KPIs between two optimization results."""
    metrics = ["Total demand\n(units/week)", "Budget used\n(k€)", "Clients\nassigned"]
    v1 = [r1["objective_value"], r1["budget_used"], len(r1["assignments"])]
    v2 = [r2["objective_value"], r2["budget_used"], len(r2["assignments"])]

    x   = np.arange(len(metrics))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, v1, w, label=label1, color="#4878D0", alpha=0.85)
    ax.bar(x + w / 2, v2, w, label=label2, color="#EE854A", alpha=0.85)

    for xi, (a, b) in zip(x, zip(v1, v2)):
        ax.text(xi - w / 2, a + max(v1 + v2) * 0.01, f"{a:.0f}", ha="center", va="bottom", fontsize=9)
        ax.text(xi + w / 2, b + max(v1 + v2) * 0.01, f"{b:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("Stage 1 vs Stage 2: Key Performance Indicators")
    ax.legend()
    plt.tight_layout()
    return fig
