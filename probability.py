from __future__ import annotations

import numpy as np

from problem_setup import rockafellar_cvar


def cantelli_lower_bound(mean_value: float, variance_value: float, threshold: float) -> float:
    """
    Cantelli-type lower bound for P(X >= threshold),
    using only E[X] and Var(X).
    """
    if threshold <= 0:
        return 1.0
    if threshold >= mean_value:
        return 0.0

    gap = mean_value - threshold
    return gap**2 / (gap**2 + variance_value)


def format_prob_value(value: float) -> str:
    if np.isnan(value):
        return "NA"
    return f"{value:.2f}"


def compute_probability_guarantees(results: dict, params: dict) -> dict:
    """
    Compute probabilistic guarantees at equilibrium.

    If K_i* = 0:
        nu_hat_i = NA
        empirical probability = NA

    Empirical probability is computed as:
        P(Pi_i* > 0)
    """
    h_star = np.asarray(results["h_star"], dtype=float)
    a_samples = np.asarray(results["a_samples"], dtype=float)
    theta_star = float(results["theta_star"])

    T, N = h_star.shape
    S = a_samples.shape[2]

    alpha = np.asarray(params["alpha"], dtype=float)
    beta = np.asarray(params["beta"], dtype=float)
    delta = float(params["delta"])
    gamma = float(params["gamma"])

    revenue_samples = np.zeros((N, S), dtype=float)
    K = np.zeros(N, dtype=float)
    Gamma = np.zeros(N, dtype=float)
    mean_R = np.zeros(N, dtype=float)
    var_R = np.zeros(N, dtype=float)

    nu_profit = np.zeros(N, dtype=float)
    nu_hat = np.full(N, np.nan, dtype=float)

    prob_Pi_positive_empirical = np.full(N, np.nan, dtype=float)

    for i in range(N):
        K_i = 0.0
        Gamma_i = 0.0

        for t in range(T):
            h_t = h_star[t, :]
            h_i_t = float(h_t[i])
            a_ti_samples = a_samples[t, i, :]

            rivals_t = float(np.sum(h_t) - h_i_t)

            revenue_samples[i, :] += a_ti_samples * np.log(1.0 + h_i_t)

            payment_t = theta_star * h_i_t
            congestion_t = 0.5 * delta * h_i_t**2 + gamma * h_i_t * rivals_t
            K_i += payment_t + congestion_t

            loss_samples_t = (
                -a_ti_samples * np.log(1.0 + h_i_t)
                + payment_t
                + congestion_t
            )

            cvar_t = rockafellar_cvar(loss_samples_t, alpha[i])
            Gamma_i += beta[i] * cvar_t

        R_i_samples = revenue_samples[i, :]
        mean_R_i = float(np.mean(R_i_samples))
        var_R_i = float(np.var(R_i_samples))

        K[i] = K_i
        Gamma[i] = Gamma_i
        mean_R[i] = mean_R_i
        var_R[i] = var_R_i

        nu_profit_i = cantelli_lower_bound(
            mean_value=mean_R_i,
            variance_value=var_R_i,
            threshold=K_i,
        )
        nu_profit[i] = nu_profit_i

        Pi_samples = R_i_samples - K_i

        if np.isclose(K_i, 0.0):
            nu_hat[i] = np.nan
            prob_Pi_positive_empirical[i] = np.nan
        else:
            if Gamma_i < 0:
                nu_hat[i] = max(nu_profit_i, alpha[i])
            else:
                nu_hat[i] = nu_profit_i

            prob_Pi_positive_empirical[i] = float(np.mean(Pi_samples > 0.0))

    return {
        "revenue_samples": revenue_samples,
        "K": K,
        "Gamma": Gamma,
        "mean_R": mean_R,
        "var_R": var_R,
        "nu_profit": nu_profit,
        "nu_hat": nu_hat,
        "prob_Pi_positive_empirical": prob_Pi_positive_empirical,
    }


def print_probability_guarantees(prob_results: dict, params: dict) -> None:
    """
    Print one table with the probabilistic guarantee terms.
    """
    headers = (
        "Follower",
        "alpha",
        "beta",
        "E[R_i*]",
        "Var(R_i*)",
        "K_i*",
        "Gamma_i*",
        "nu_i",
        "nu_hat_i",
        "Empirical P(Pi_i*>0)",
    )

    N = len(prob_results["K"])
    rows = []

    for i in range(N):
        rows.append((
            f"Follower {i + 1}",
            f"{params['alpha'][i]:.2f}",
            f"{params['beta'][i]:.2f}",
            f"{round(prob_results['mean_R'][i])}",
            f"{round(prob_results['var_R'][i])}",
            f"{round(prob_results['K'][i])}",
            f"{round(prob_results['Gamma'][i])}",
            format_prob_value(prob_results["nu_profit"][i]),
            format_prob_value(prob_results["nu_hat"][i]),
            format_prob_value(prob_results["prob_Pi_positive_empirical"][i]),
        ))

    widths = [len(h) for h in headers]
    for row in rows:
        for j, value in enumerate(row):
            widths[j] = max(widths[j], len(str(value)))

    total_width = sum(widths) + 3 * (len(widths) - 1)

    print("\nPROBABILISTIC PERFORMANCE GUARANTEE")
    print("=" * total_width)
    print(" | ".join(headers[j].ljust(widths[j]) for j in range(len(headers))))
    print("-" * total_width)

    for row in rows:
        print(" | ".join(str(row[j]).ljust(widths[j]) for j in range(len(row))))

    print("=" * total_width)