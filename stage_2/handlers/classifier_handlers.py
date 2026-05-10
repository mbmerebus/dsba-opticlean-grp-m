import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
)

DATA_PATH = "../data/case-opticlean-data-dsba"

# purchased_products is excluded: present in historic but absent from potential_clients
FEATURES = ["employes", "surface", "firm_age", "monthly_volume", "purchase_staff", "dist_retail_point"]
TARGET = "loyal"

_COLORS = {0: "#4878D0", 1: "#EE854A"}
_LABELS = {0: "Non-Loyal", 1: "Loyal"}


def load_data():
    historic  = pd.read_excel(f"{DATA_PATH}/historic_clients.xlsx")
    potential = pd.read_excel(f"{DATA_PATH}/potential_clients.xlsx")
    sites     = pd.read_excel(f"{DATA_PATH}/candidate_sites.xlsx")
    return historic, potential, sites


def compute_distance_matrix(clients_df, sites_df):
    """Euclidean distance matrix of shape (n_clients, n_sites)."""
    lats = clients_df["lat"].values[:, None]
    lons = clients_df["lon"].values[:, None]
    site_lats = sites_df["lat"].values[None, :]
    site_lons = sites_df["lon"].values[None, :]
    return np.sqrt((lats - site_lats) ** 2 + (lons - site_lons) ** 2)


def split_data(historic, test_size=0.2, random_state=42):
    """Return train/test splits for features and target."""
    X = historic[FEATURES]
    y = historic[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_classifiers(X_train, y_train):
    """
    Train logistic regression and random forest.
    Both use class_weight='balanced' to handle the 79/21 class imbalance.
    """
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    )
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    return lr, rf


def evaluation_report(models, names, X_test, y_test):
    """Print classification report and ROC AUC for each model."""
    for model, name in zip(models, names):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print(f"{'='*40}")
        print(f" {name}   (ROC AUC = {auc:.3f})")
        print(f"{'='*40}")
        print(classification_report(y_test, y_pred, target_names=["Non-Loyal", "Loyal"]))


def plot_confusion_matrices(models, names, X_test, y_test):
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]
    for model, name, ax in zip(models, names, axes):
        cm = confusion_matrix(y_test, model.predict(X_test))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-Loyal", "Loyal"])
        ax.set_yticklabels(["Non-Loyal", "Loyal"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{name}")
        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                        color="white" if cm[i, j] > thresh else "black")
    plt.suptitle("Confusion Matrices", fontsize=13)
    plt.tight_layout()
    return fig


def plot_roc_curves(models, names, X_test, y_test):
    fig, ax = plt.subplots(figsize=(6, 5))
    for model, name in zip(models, names):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name}  (AUC = {auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    plt.tight_layout()
    return fig


def build_loyalty_matrix(model, potential_df, sites_df):
    """
    Predict loyalty for every (client, site) pair.
    Returns an int array of shape (n_clients, n_sites).
    """
    dist_matrix = compute_distance_matrix(potential_df, sites_df)
    base = potential_df[["employes", "surface", "firm_age", "monthly_volume", "purchase_staff"]].values
    n_clients, n_sites = len(potential_df), len(sites_df)
    loyalty_matrix = np.zeros((n_clients, n_sites), dtype=int)
    for j in range(n_sites):
        X = pd.DataFrame(
            np.column_stack([base, dist_matrix[:, j]]),
            columns=FEATURES,
        )
        loyalty_matrix[:, j] = model.predict(X)
    return loyalty_matrix


def plot_loyalty_matrix_summary(loyalty_matrix):
    """Stacked bar chart: predicted loyal / non-loyal clients per candidate site."""
    n_sites = loyalty_matrix.shape[1]
    loyal_counts    = loyalty_matrix.sum(axis=0)
    nonloyal_counts = loyalty_matrix.shape[0] - loyal_counts
    x = np.arange(n_sites)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, nonloyal_counts, label="Non-Loyal", color=_COLORS[0], alpha=0.8)
    ax.bar(x, loyal_counts, bottom=nonloyal_counts, label="Loyal", color=_COLORS[1], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Site {j+1}" for j in range(n_sites)])
    ax.set_ylabel("Potential Clients")
    ax.set_title("Predicted Loyalty Distribution per Candidate Site")
    ax.legend()
    plt.tight_layout()
    return fig
