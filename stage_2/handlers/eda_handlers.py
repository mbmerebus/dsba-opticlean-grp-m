import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Handler function "toolbox" to make the notebook cleaner

DATA_PATH = "../data"

_COLORS = {
    0: "#4878D0",
    1: "#EE854A"
    }

_LABELS = {
    0: "Non-Loyal",
    1: "Loyal"
    }


def load_historic():
    return pd.read_excel(f"{DATA_PATH}/historic_clients.xlsx")


def demand_summary(df):
    """Descriptive stats for weekly demand split by loyalty group."""
    return (
        df.groupby("loyal")["demand"]
        .agg(n="count", mean="mean", median="median", std="std", min="min", max="max")
        .rename(index=_LABELS)
        .round(2)
    )


def plot_demand_distribution(df):
    """Histogram and box plot of weekly demand by loyalty group."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for v, label in _LABELS.items():
        axes[0].hist(
            df[df["loyal"] == v]["demand"],
            bins=30, alpha=0.6, color=_COLORS[v], label=label
        )
    axes[0].set_title("Demand Distribution")
    axes[0].set_xlabel("Weekly Demand")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    groups = [df[df["loyal"] == v]["demand"].values for v in [0, 1]]
    bp = axes[1].boxplot(groups, labels=list(_LABELS.values()), patch_artist=True)
    for patch, color in zip(bp["boxes"], _COLORS.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_title("Demand Box Plot")
    axes[1].set_ylabel("Weekly Demand")

    plt.tight_layout()
    return fig


def distance_sensitivity_table(df):
    """Pearson correlation and linear slope between demand and distance per group."""
    rows = []
    for v, label in _LABELS.items():
        sub = df[df["loyal"] == v]
        x, y = sub["dist_retail_point"].values, sub["demand"].values
        slope = np.polyfit(x, y, 1)[0]
        r = np.corrcoef(x, y)[0, 1]
        rows.append({
            "Group": label,
            "Pearson r": round(r, 3),
            "Slope (demand / dist unit)": round(slope, 3),
        })
    return pd.DataFrame(rows).set_index("Group")


def plot_demand_vs_distance(df):
    """Scatter plot of demand vs distance with per-group regression lines."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for v, label in _LABELS.items():
        sub = df[df["loyal"] == v]
        ax.scatter(
            sub["dist_retail_point"], sub["demand"],
            alpha=0.35, s=18, color=_COLORS[v], label=label
        )
        coef = np.polyfit(sub["dist_retail_point"], sub["demand"], 1)
        x_line = np.linspace(sub["dist_retail_point"].min(), sub["dist_retail_point"].max(), 200)
        r = np.corrcoef(sub["dist_retail_point"], sub["demand"])[0, 1]
        ax.plot(
            x_line, np.polyval(coef, x_line),
            color=_COLORS[v], linewidth=2,
            label=f"{label} fit (r = {r:.2f})"
        )

    ax.set_xlabel("Distance to Retail Point")
    ax.set_ylabel("Weekly Demand")
    ax.set_title("Demand vs Distance by Loyalty Group")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_geographic_distribution(df):
    """Spatial scatter of historic clients colored by loyalty."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for v, label in _LABELS.items():
        sub = df[df["loyal"] == v]
        ax.scatter(sub["lon"], sub["lat"], alpha=0.5, s=20, color=_COLORS[v], label=label)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Geographic Distribution by Loyalty")
    ax.legend()
    plt.tight_layout()
    return fig


def feature_means(df):
    """Mean value of each client feature per loyalty group."""
    features = ["employes", "surface", "firm_age", "monthly_volume", "purchase_staff", "dist_retail_point"]
    return df.groupby("loyal")[features].mean().rename(index=_LABELS).round(2)


def plot_feature_comparison(df):
    """Box plots comparing client features across loyalty groups."""
    features = ["employes", "surface", "firm_age", "monthly_volume", "purchase_staff"]
    fig, axes = plt.subplots(1, len(features), figsize=(15, 5))

    for ax, feat in zip(axes, features):
        groups = [df[df["loyal"] == v][feat].values for v in [0, 1]]
        bp = ax.boxplot(groups, labels=list(_LABELS.values()), patch_artist=True)
        for patch, color in zip(bp["boxes"], _COLORS.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(feat.replace("_", " ").title())

    plt.suptitle("Client Features by Loyalty Group", fontsize=13)
    plt.tight_layout()
    return fig
