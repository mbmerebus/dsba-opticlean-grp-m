from __future__ import annotations

import gurobipy as gp
import pandas as pd
from gurobipy import GRB


def solve_location_model(
    demand_matrix,
    candidate_sites: pd.DataFrame,
    budget: float = 850,
    time_limit: int | None = None,
    verbose: bool = True,
) -> dict:
    """
    Solve the OptiClean store location and client assignment model.

    Parameters
    ----------
    demand_matrix:
        Matrix where entry [i, j] is the predicted weekly demand captured if
        potential client i is assigned to candidate site j. Negative predictions
        should already be replaced by zero before calling this function.
    candidate_sites:
        DataFrame with one row per candidate site and these columns:
        cap_small, cost_small, cap_large, cost_large.
    budget:
        Total opening budget, expressed in the same unit as candidate-site costs.
        The case uses 850 thousand euros.
    time_limit:
        Optional solver time limit in seconds.
    verbose:
        If False, suppress Gurobi output.

    Returns
    -------
    dict
        Contains the Gurobi model, selected stores, assignments, unassigned clients,
        objective value, and budget used.
    """
    sites_df = candidate_sites.copy()
    demand_df = pd.DataFrame(demand_matrix).copy()

    required_columns = {"cap_small", "cost_small", "cap_large", "cost_large"}
    missing_columns = required_columns - set(sites_df.columns)
    if missing_columns:
        raise ValueError(f"candidate_sites is missing columns: {sorted(missing_columns)}")

    if demand_df.shape[1] != len(sites_df):
        raise ValueError("demand_matrix must have one column for each candidate site")

    site_ids = list(sites_df.index)
    client_ids = list(demand_df.index)
    demand_df.columns = site_ids

    model = gp.Model("opticlean_location_model")
    model.Params.OutputFlag = 1 if verbose else 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    # Decision variables:
    # s_j = 1 if candidate site j is opened as a small store, 0 otherwise.
    # l_j = 1 if candidate site j is opened as a large store, 0 otherwise.
    # x_ij = 1 if client i is assigned to candidate site j, 0 otherwise.
    open_small = model.addVars(site_ids, vtype=GRB.BINARY, name="open_small")
    open_large = model.addVars(site_ids, vtype=GRB.BINARY, name="open_large")
    assign = model.addVars(client_ids, site_ids, vtype=GRB.BINARY, name="assign")

    # Objective:
    # max sum_i sum_j d_ij * x_ij
    # where d_ij is the predicted demand captured from client i at site j.
    model.setObjective(
        gp.quicksum(
            float(demand_df.loc[i, j]) * assign[i, j]
            for i in client_ids
            for j in site_ids
        ),
        GRB.MAXIMIZE,
    )

    # Budget constraint:
    # sum_j cost_small_j * s_j + cost_large_j * l_j <= B
    model.addConstr(
        gp.quicksum(
            float(sites_df.loc[j, "cost_small"]) * open_small[j]
            + float(sites_df.loc[j, "cost_large"]) * open_large[j]
            for j in site_ids
        )
        <= budget,
        name="budget",
    )

    for j in site_ids:
        # Store-format constraint:
        # s_j + l_j <= 1 for every site j
        # A site can be closed, opened small, or opened large, but not both.
        model.addConstr(
            open_small[j] + open_large[j] <= 1,
            name=f"one_format[{j}]",
        )

    for i in client_ids:
        # Assignment constraint:
        # sum_j x_ij <= 1 for every client i
        # A client can be assigned to at most one opened site.
        model.addConstr(
            gp.quicksum(assign[i, j] for j in site_ids) <= 1,
            name=f"assign_once[{i}]",
        )

    for j in site_ids:
        # Capacity constraint:
        # sum_i d_ij * x_ij <= cap_small_j * s_j + cap_large_j * l_j
        # Captured demand assigned to a site cannot exceed its chosen capacity.
        model.addConstr(
            gp.quicksum(float(demand_df.loc[i, j]) * assign[i, j] for i in client_ids)
            <= float(sites_df.loc[j, "cap_small"]) * open_small[j]
            + float(sites_df.loc[j, "cap_large"]) * open_large[j],
            name=f"capacity[{j}]",
        )

    for i in client_ids:
        for j in site_ids:
            # Opening-link constraint:
            # x_ij <= s_j + l_j for every client-site pair (i, j)
            # A client can only be assigned to a site that is actually opened.
            model.addConstr(
                assign[i, j] <= open_small[j] + open_large[j],
                name=f"assignment_requires_open_site[{i},{j}]",
            )

    model.optimize()

    if model.SolCount == 0:
        raise RuntimeError("No feasible solution found.")

    selected_rows = []
    for j in site_ids:
        if open_small[j].X > 0.5:
            selected_rows.append(
                {
                    "site_id": j,
                    "format": "small",
                    "capacity": float(sites_df.loc[j, "cap_small"]),
                    "cost": float(sites_df.loc[j, "cost_small"]),
                }
            )
        elif open_large[j].X > 0.5:
            selected_rows.append(
                {
                    "site_id": j,
                    "format": "large",
                    "capacity": float(sites_df.loc[j, "cap_large"]),
                    "cost": float(sites_df.loc[j, "cost_large"]),
                }
            )

    assignment_rows = []
    assigned_clients = set()
    for i in client_ids:
        for j in site_ids:
            if assign[i, j].X > 0.5:
                assigned_clients.add(i)
                assignment_rows.append(
                    {
                        "client_id": i,
                        "site_id": j,
                        "captured_demand": float(demand_df.loc[i, j]),
                    }
                )

    selected_sites = pd.DataFrame(selected_rows)
    assignments = pd.DataFrame(assignment_rows)
    unassigned_clients = [i for i in client_ids if i not in assigned_clients]
    budget_used = float(selected_sites["cost"].sum()) if not selected_sites.empty else 0.0

    if not assignments.empty:
        site_demand = (
            assignments.groupby("site_id")["captured_demand"]
            .sum()
            .rename("used_capacity")
            .reset_index()
        )
        selected_sites = selected_sites.merge(site_demand, on="site_id", how="left")
        selected_sites["used_capacity"] = selected_sites["used_capacity"].fillna(0.0)
        selected_sites["capacity_utilization"] = (
            selected_sites["used_capacity"] / selected_sites["capacity"]
        )

    return {
        "model": model,
        "selected_sites": selected_sites,
        "assignments": assignments,
        "unassigned_clients": unassigned_clients,
        "objective_value": float(model.ObjVal),
        "budget_used": budget_used,
        "status": model.Status,
    }
