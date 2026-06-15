from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from problem_setup import investment_cost, rockafellar_cvar


def _follower_slot_loss_samples(
    a_ti_samples: np.ndarray,
    h_i_t: float,
    h_t: np.ndarray,
    i: int,
    theta: float,
    delta: float,
    gamma: float,
) -> np.ndarray:
    """
    Loss samples for follower i in slot t evaluated at equilibrium h_t.

    L_{i,s}^t(h^t) =
        - a_{i,s}^t * ln(1 + h_i^t)
        + theta * h_i^t
        + delta/2 * (h_i^t)^2
        + gamma * h_i^t * sum_{j!=i} h_j^t
    """
    rivals = float(np.sum(h_t) - h_i_t)
    payment = theta * h_i_t
    congestion = 0.5 * delta * h_i_t**2 + gamma * h_i_t * rivals

    return -a_ti_samples * np.log(1.0 + h_i_t) + payment + congestion


def compute_follower_equilibrium_breakdown(
    results: dict,
    params: dict,
    follower_index: int = 0,
) -> dict:
    """
    Compute follower equilibrium decomposition for one chosen follower i.
    """
    h_star = results["h_star"]
    theta_star = results["theta_star"]
    a_samples = results["a_samples"]
    w = float(params["w"])
    i = follower_index
    beta_i = float(params["beta"][i])
    alpha_i = float(params["alpha"][i])
    delta = float(params["delta"])
    gamma = float(params["gamma"])
    T = params["T"]

    expected_revenue = 0.0
    total_payment = 0.0
    total_congestion = 0.0
    total_expected_payoff = 0.0
    total_cvar = 0.0
    total_cvar_penalty = 0.0

    per_slot = []

    for t in range(T):
        h_t = h_star[t, :]
        h_i_t = float(h_t[i])
        a_ti_samples = a_samples[t, i, :]
        mean_a_ti = float(np.mean(a_ti_samples))

        revenue_t = mean_a_ti * np.log(1.0 + h_i_t)
        payment_t = theta_star * h_i_t
        rivals_t = float(np.sum(h_t) - h_i_t)
        congestion_t = 0.5 * delta * h_i_t**2 + gamma * h_i_t * rivals_t
        expected_payoff_t = revenue_t - payment_t - congestion_t

        loss_samples_t = _follower_slot_loss_samples(
            a_ti_samples=a_ti_samples,
            h_i_t=h_i_t,
            h_t=h_t,
            i=i,
            theta=theta_star,
            delta=delta,
            gamma=gamma,
        )
        cvar_t = rockafellar_cvar(loss_samples_t, alpha_i)
        cvar_penalty_t = beta_i * cvar_t

        expected_revenue += revenue_t
        total_payment += payment_t
        total_congestion += congestion_t
        total_expected_payoff += expected_payoff_t
        total_cvar += cvar_t
        total_cvar_penalty += cvar_penalty_t

        per_slot.append(
            {
                "slot": t,
                "h_i_t": h_i_t,
                "expected_revenue": revenue_t,
                "payment": payment_t,
                "congestion": congestion_t,
                "expected_payoff": expected_payoff_t,
                "cvar": cvar_t,
                "cvar_penalty": cvar_penalty_t,
            }
        )

    risk_adjusted_objective = w * total_expected_payoff - total_cvar_penalty

    return {
        "follower_index": i,
        "expected_revenue": expected_revenue,
        "payment": total_payment,
        "congestion": total_congestion,
        "expected_payoff": total_expected_payoff,
        "cvar": total_cvar,
        "cvar_penalty": total_cvar_penalty,
        "risk_adjusted_objective": risk_adjusted_objective,
        "per_slot": per_slot,
    }


def compute_system_summary(results: dict, params: dict) -> dict:
    """
    Compute one system-level summary for the optimal equilibrium.
    """
    C_star = float(results["C_star"])
    theta_star = float(results["theta_star"])
    obj_star = float(results["objective_value"])
    h_star = np.asarray(results["h_star"], dtype=float)
    eta_star = np.asarray(results["eta_star"], dtype=float)
    a_samples = np.asarray(results["a_samples"], dtype=float)

    T, N = h_star.shape
    beta = np.asarray(params["beta"], dtype=float)
    alpha = np.asarray(params["alpha"], dtype=float)
    delta = float(params["delta"])
    gamma = float(params["gamma"])
    w = float(params["w"])

    total_follower_demand = float(np.sum(h_star))
    slot_total_demand = np.sum(h_star, axis=1)
    utilization_by_slot = slot_total_demand / C_star
    active_providers_by_slot = np.sum(h_star > 1e-10, axis=1)

    follower_expected_profit = np.zeros(N, dtype=float)
    follower_expected_risk_adjusted_profit = np.zeros(N, dtype=float)
    follower_cvar = np.zeros(N, dtype=float)
    follower_cvar_penalty = np.zeros(N, dtype=float)

    for t in range(T):
        h_t = h_star[t, :]
        total_slot_load = float(np.sum(h_t))

        for i in range(N):
            h_i_t = float(h_t[i])
            a_ti_samples = a_samples[t, i, :]
            mean_a_ti = float(np.mean(a_ti_samples))

            expected_revenue_ti = mean_a_ti * np.log(1.0 + h_i_t)
            payment_ti = theta_star * h_i_t
            rivals_ti = total_slot_load - h_i_t
            congestion_ti = 0.5 * delta * h_i_t**2 + gamma * h_i_t * rivals_ti
            expected_profit_ti = expected_revenue_ti - payment_ti - congestion_ti

            loss_samples_ti = _follower_slot_loss_samples(
                a_ti_samples=a_ti_samples,
                h_i_t=h_i_t,
                h_t=h_t,
                i=i,
                theta=theta_star,
                delta=delta,
                gamma=gamma,
            )
            cvar_ti = rockafellar_cvar(loss_samples_ti, alpha[i])
            cvar_penalty_ti = beta[i] * cvar_ti
            risk_adjusted_profit_ti = w * expected_profit_ti - cvar_penalty_ti

            follower_expected_profit[i] += expected_profit_ti
            follower_expected_risk_adjusted_profit[i] += risk_adjusted_profit_ti
            follower_cvar[i] += cvar_ti
            follower_cvar_penalty[i] += cvar_penalty_ti

    summary = {
        "C_star": C_star,
        "theta_star": theta_star,
        "total_follower_demand": total_follower_demand,
        "utilization_min": float(np.min(utilization_by_slot)),
        "utilization_avg": float(np.mean(utilization_by_slot)),
        "utilization_max": float(np.max(utilization_by_slot)),
        "eta_min": float(np.min(eta_star)),
        "eta_avg": float(np.mean(eta_star)),
        "eta_max": float(np.max(eta_star)),
        "active_min": int(np.min(active_providers_by_slot)),
        "active_avg": float(np.mean(active_providers_by_slot)),
        "active_max": int(np.max(active_providers_by_slot)),
        "leader_profit": obj_star,
        "follower_expected_profit": follower_expected_profit,
        "follower_expected_profit_total": float(np.sum(follower_expected_profit)),
        "follower_expected_risk_adjusted_profit": follower_expected_risk_adjusted_profit,
        "follower_expected_risk_adjusted_profit_total": float(
            np.sum(follower_expected_risk_adjusted_profit)
        ),
        "follower_cvar": follower_cvar,
        "follower_cvar_total": float(np.sum(follower_cvar)),
        "follower_cvar_penalty": follower_cvar_penalty,
        "follower_cvar_penalty_total": float(np.sum(follower_cvar_penalty)),
    }

    return summary



def print_main_summary_table(results: dict, params: dict) -> None:
    """
    Print one main table with no decimals for values.
    """
    s = compute_system_summary(results, params)

    rows = [
        ("Optimal capacity C*", f"{round(s['C_star'])}"),
        ("Optimal tariff theta*", f"{round(s['theta_star'])}"),
        ("Total follower demand", f"{round(s['total_follower_demand'])}"),
        (
            "Utilization by slot (min/avg/max)",
            f"{round(s['utilization_min'])} / {round(s['utilization_avg'])} / {round(s['utilization_max'])}",
        ),
        (
            "Scarcity multipliers eta (min/avg/max)",
            f"{round(s['eta_min'])} / {round(s['eta_avg'])} / {round(s['eta_max'])}",
        ),
        (
            "Active service providers (min/avg/max)",
            f"{s['active_min']} / {round(s['active_avg'])} / {s['active_max']}",
        ),
        ("Leader profit", f"{round(s['leader_profit'])}"),
        (
            "Follower expected profit",
            np.array2string(np.rint(s["follower_expected_profit"]).astype(int), separator=" "),
        ),
        ("Follower expected profit total", f"{round(s['follower_expected_profit_total'])}"),
        (
            "Follower expected risk-adjusted profit",
            np.array2string(
                np.rint(s["follower_expected_risk_adjusted_profit"]).astype(int),
                separator=" "
            ),
        ),
        ("Follower risk-adjusted total", f"{round(s['follower_expected_risk_adjusted_profit_total'])}"),
        (
            "Follower CVaR",
            np.array2string(np.rint(s["follower_cvar"]).astype(int), separator=" "),
        ),
        ("Follower CVaR total", f"{round(s['follower_cvar_total'])}"),
        (
            "Follower CVaR penalty",
            np.array2string(np.rint(s["follower_cvar_penalty"]).astype(int), separator=" "),
        ),
        ("Follower CVaR penalty total", f"{round(s['follower_cvar_penalty_total'])}"),
    ]

    left_width = max(len(r[0]) for r in rows) + 2
    right_width = max(len(r[1]) for r in rows) + 2
    total_width = left_width + right_width + 3

    print("\n" + "=" * total_width)
    print("MAIN EQUILIBRIUM SUMMARY TABLE")
    print("=" * total_width)
    print(f"{'Metric'.ljust(left_width)} | {'Value'.ljust(right_width)}")
    print("-" * total_width)

    for metric, value in rows:
        print(f"{metric.ljust(left_width)} | {value.ljust(right_width)}")

    print("=" * total_width)


def print_results(results: dict, params: dict) -> None:
    """
    Print the main summary table.
    """
    np.set_printoptions(suppress=True)
    print_main_summary_table(results, params)


def print_risk_type_info(params: dict) -> None:
    print("\nRISK TYPE SETUP")
    print("=" * 60)
    print("Selected follower type =", params["follower_type"])
    print("beta_L  =", round(params["beta_L"], 2))
    print("alpha_L =", round(params["alpha_L"], 2))
    print("alpha_H =", round(params["alpha_H"], 2))
    print("alpha =", np.round(np.asarray(params["alpha"], dtype=float), 2))
    print("beta  =", np.round(np.asarray(params["beta"], dtype=float), 2))

def print_follower_diagnostics(results: dict, params: dict) -> None:
    import numpy as np

    h_star = np.asarray(results["h_star"], dtype=float)
    a_samples = np.asarray(results["a_samples"], dtype=float)
    theta_star = float(results["theta_star"])
    delta = float(params["delta"])
    gamma = float(params["gamma"])

    mu = np.asarray(params["mu"], dtype=float)
    sigma = np.asarray(params["sigma"], dtype=float)
    weekly_factors = np.asarray(params["weekly_factors"], dtype=float)
    annual_amplitude = np.asarray(params["annual_amplitude"], dtype=float)

    T, N = h_star.shape

    headers = (
        "Follower",
        "mu",
        "sigma",
        "Avg demand",
        "Total demand",
        "Avg revenue",
        "Avg payment",
        "Avg congestion",
        "Weekly avg",
        "Annual ampl",
    )

    rows = []

    for i in range(N):
        total_revenue = 0.0
        total_payment = 0.0
        total_congestion = 0.0

        for t in range(T):
            h_t = h_star[t, :]
            h_i_t = float(h_t[i])
            mean_a_ti = float(np.mean(a_samples[t, i, :]))
            rivals_t = float(np.sum(h_t) - h_i_t)

            revenue_t = mean_a_ti * np.log(1.0 + h_i_t)
            payment_t = theta_star * h_i_t
            congestion_t = 0.5 * delta * h_i_t**2 + gamma * h_i_t * rivals_t

            total_revenue += revenue_t
            total_payment += payment_t
            total_congestion += congestion_t

        avg_demand = float(np.mean(h_star[:, i]))
        total_demand = float(np.sum(h_star[:, i]))
        avg_revenue = total_revenue / T
        avg_payment = total_payment / T
        avg_congestion = total_congestion / T
        weekly_avg = float(np.mean(weekly_factors[i, :]))

        rows.append((
            f"Follower {i+1}",
            f"{mu[i]:.2f}",
            f"{sigma[i]:.2f}",
            f"{avg_demand:.2f}",
            f"{round(total_demand)}",
            f"{round(avg_revenue)}",
            f"{round(avg_payment)}",
            f"{round(avg_congestion)}",
            f"{weekly_avg:.2f}",
            f"{annual_amplitude[i]:.2f}",
        ))

    widths = [len(h) for h in headers]
    for row in rows:
        for j, value in enumerate(row):
            widths[j] = max(widths[j], len(str(value)))

    total_width = sum(widths) + 3 * (len(widths) - 1)

    print("\nFOLLOWER DIAGNOSTICS")
    print("=" * total_width)
    print(" | ".join(headers[j].ljust(widths[j]) for j in range(len(headers))))
    print("-" * total_width)

    for row in rows:
        print(" | ".join(str(row[j]).ljust(widths[j]) for j in range(len(row))))

    print("=" * total_width)