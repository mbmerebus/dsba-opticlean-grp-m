import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error

from classifier_handlers import compute_distance_matrix, load_data  # noqa: F401

FEATURES = ["employes", "surface", "firm_age", "monthly_volume", "purchase_staff", "dist_retail_point"]
TARGET   = "demand"

_COLORS = {"Loyal": "#EE854A", "Non-Loyal": "#4878D0"}


def split_by_loyalty(historic):
    loyal    = historic[historic["loyal"] == 1].reset_index(drop=True)
    nonloyal = historic[historic["loyal"] == 0].reset_index(drop=True)
    return loyal, nonloyal


def get_splits(loyal, nonloyal, test_size=0.2, random_state=42):
    """Return (X_train, X_test, y_train, y_test) for each group."""
    splits = {}
    for name, df in [("Loyal", loyal), ("Non-Loyal", nonloyal)]:
        X, y = df[FEATURES], df[TARGET]
        splits[name] = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return splits


def train_demand_models(splits):
    """Train one LinearRegression per loyalty group. Returns {group: model}."""
    models = {}
    for name, (X_train, _, y_train, _) in splits.items():
        models[name] = LinearRegression().fit(X_train, y_train)
    return models


def evaluation_report(models, splits):
    """Print R² and RMSE on the test set for each group."""
    print(f"{'Group':<12} {'R²':>6}  {'RMSE':>7}")
    print("-" * 28)
    for name, model in models.items():
        _, X_test, _, y_test = splits[name]
        y_pred = model.predict(X_test)
        r2   = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        print(f"{name:<12} {r2:>6.3f}  {rmse:>7.2f}")


def plot_predictions_vs_actual(models, splits):
    """Predicted vs actual scatter for each group."""
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    for ax, (name, model) in zip(axes, models.items()):
        _, X_test, _, y_test = splits[name]
        y_pred = np.clip(model.predict(X_test), 0, None)
        ax.scatter(y_test, y_pred, alpha=0.5, s=20, color=_COLORS[name])
        lim = (min(y_test.min(), y_pred.min()) - 2, max(y_test.max(), y_pred.max()) + 2)
        ax.plot(lim, lim, "k--", linewidth=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.set_title(f"{name} model")
    plt.suptitle("Predicted vs Actual Demand", fontsize=13)
    plt.tight_layout()
    return fig


def build_demand_matrix(models, loyalty_matrix, potential_df, sites_df):
    """
    200×8 demand matrix. For each (client, site) pair, selects the model
    matching the predicted loyalty status. Negatives are clipped to 0.
    """
    dist_matrix = compute_distance_matrix(potential_df, sites_df)
    base        = potential_df[["employes", "surface", "firm_age", "monthly_volume", "purchase_staff"]].values
    n_clients, n_sites = len(potential_df), len(sites_df)
    demand_matrix = np.zeros((n_clients, n_sites))

    for j in range(n_sites):
        X = pd.DataFrame(
            np.column_stack([base, dist_matrix[:, j]]),
            columns=FEATURES,
        )
        loyal_mask = loyalty_matrix[:, j] == 1
        if loyal_mask.any():
            demand_matrix[loyal_mask, j]  = models["Loyal"].predict(X[loyal_mask])
        if (~loyal_mask).any():
            demand_matrix[~loyal_mask, j] = models["Non-Loyal"].predict(X[~loyal_mask])

    return np.clip(demand_matrix, 0, None)


def plot_demand_matrix_summary(demand_matrix):
    """Mean and total predicted demand per candidate site."""
    x      = np.arange(demand_matrix.shape[1])
    labels = [f"Site {j+1}" for j in range(len(x))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(x, demand_matrix.mean(axis=0), color=_COLORS["Non-Loyal"], alpha=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Mean predicted demand")
    axes[0].set_title("Average demand per site (across all clients)")

    axes[1].bar(x, demand_matrix.sum(axis=0), color=_COLORS["Loyal"], alpha=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Total predicted demand")
    axes[1].set_title("Total potential demand per site")

    plt.tight_layout()
    return fig
