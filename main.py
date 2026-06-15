from problem_setup import build_params, generate_a_samples, compute_A_matrix
from leader_optimization import optimize_leader
from postprocessing import (
    print_risk_type_info,
    compute_system_summary,
    print_follower_diagnostics,
)
from probability import compute_probability_guarantees, print_probability_guarantees
import numpy as np
import time
import json

from figure_leader import plot_leader_figure
from figure_proba import plot_probability_figure
import sys

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for f in self.files:
            f.write(text)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

def to_serializable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, dict):
        return {k: to_serializable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_serializable(v) for v in x]
    return x


def compute_follower_profit_samples(results: dict, params: dict) -> np.ndarray:
    h_star = np.asarray(results["h_star"], dtype=float)
    a_samples = np.asarray(results["a_samples"], dtype=float)
    theta_star = float(results["theta_star"])

    T, N = h_star.shape
    S = a_samples.shape[2]

    delta = float(params["delta"])
    gamma = float(params["gamma"])

    profit_samples = np.zeros((N, S), dtype=float)

    for i in range(N):
        for t in range(T):
            h_t = h_star[t, :]
            h_i_t = float(h_t[i])
            a_ti_samples = a_samples[t, i, :]

            rivals_t = float(np.sum(h_t) - h_i_t)

            revenue_t_samples = a_ti_samples * np.log(1.0 + h_i_t)
            payment_t = theta_star * h_i_t
            congestion_t = 0.5 * delta * h_i_t**2 + gamma * h_i_t * rivals_t

            profit_samples[i, :] += revenue_t_samples - payment_t - congestion_t

    return profit_samples


def main():
    np.set_printoptions(suppress=True)

    # These mean 50%, 100%, 150%
    cv_list = [0.001, 0.25, 0.5, 0.75, 1]

    all_case_data = []

    for cv in cv_list:
        print("\n" + "=" * 120)
        print(f"STARTING ALL FOLLOWER TYPES FOR CV = {100 * cv:.0f}%")
        print("=" * 120)

        for follower_type in range(1, 5):
            print("\n" + "#" * 120)
            print(f"RUN FOR FOLLOWER TYPE = {follower_type}, CV = {100 * cv:.0f}%")
            print("#" * 120)
            start_time = time.perf_counter()
            follower_types = [follower_type] * 3
            print("follower_types =", follower_types)
            print("CV =", f"{100 * cv:.0f}%")

            params = build_params(
                follower_types=follower_types,
                reuse_fixed_market=False,
                cv=cv,
            )

            print("mu =", params["mu"])
            print("sigma =", params["sigma"])
            print("annual_amplitude =", params["annual_amplitude"])
            print("weekly_factors[0] =", params["weekly_factors"][0])

            a_samples = generate_a_samples(params)

            print("\nA-SAMPLES DIAGNOSTICS")
            print("=" * 60)
            print("Input CV =", cv)
            print("Input CV percent =", f"{100 * cv:.0f}%")
            print("Lognormal sigma =", params["sigma"])

            empirical_cv_a = np.zeros(params["N"])

            for i in range(params["N"]):
                a_i = a_samples[:, i, :].reshape(-1)

                mean_a = np.mean(a_i)
                std_a = np.std(a_i)
                cv_a = std_a / mean_a

                empirical_cv_a[i] = cv_a

                print(f"\nFollower {i + 1}")
                print("Mean(a) =", mean_a)
                print("Std(a)  =", std_a)
                print("CV(a)   =", cv_a)
                print("CV(a)%  =", 100 * cv_a)
                print("Min(a)  =", np.min(a_i))
                print("Max(a)  =", np.max(a_i))

            print("\nOverall")
            print("Mean(a) =", np.mean(a_samples))
            print("Std(a)  =", np.std(a_samples))
            print("CV(a)   =", np.std(a_samples) / np.mean(a_samples))
            print("CV(a)%  =", 100 * np.std(a_samples) / np.mean(a_samples))

            A = compute_A_matrix(a_samples=a_samples, params=params)

            results = optimize_leader(A=A, params=params)
            results["A"] = A
            results["a_samples"] = a_samples

            end_time = time.perf_counter()
            runtime_seconds = end_time - start_time

            print_risk_type_info(params)

            print("C_star =", round(results["C_star"]))
            print("theta_star =", round(results["theta_star"]))
            print("objective =", round(results["objective_value"]))

            summary = compute_system_summary(results, params)

            print("\nSUMMARY")
            print("=" * 60)
            print("Leader profit =", round(summary["leader_profit"]))
            print(
                "Active service providers (min, max) =",
                summary["active_min"],
                summary["active_max"]
            )
            print(
                "Utilization (min, avg, max) =",
                round(summary["utilization_min"], 2),
                round(summary["utilization_avg"], 2),
                round(summary["utilization_max"], 2)
            )

            print("\nFOLLOWER SUMMARY")
            print("=" * 110)

            headers = (
                "Follower",
                "alpha",
                "beta",
                "Expected profit",
                "Risk-adjusted profit",
                "CVaR",
                "CVaR penalty",
            )

            rows = []
            N = params["N"]
            for i in range(N):
                rows.append((
                    f"Follower {i+1}",
                    f"{params['alpha'][i]:.2f}",
                    f"{params['beta'][i]:.2f}",
                    f"{round(summary['follower_expected_profit'][i])}",
                    f"{round(summary['follower_expected_risk_adjusted_profit'][i])}",
                    f"{round(summary['follower_cvar'][i])}",
                    f"{round(summary['follower_cvar_penalty'][i])}",
                ))

            widths = [len(h) for h in headers]
            for row in rows:
                for j, value in enumerate(row):
                    widths[j] = max(widths[j], len(str(value)))

            total_width = sum(widths) + 3 * (len(widths) - 1)

            print(" | ".join(headers[j].ljust(widths[j]) for j in range(len(headers))))
            print("-" * total_width)

            for row in rows:
                print(" | ".join(str(row[j]).ljust(widths[j]) for j in range(len(row))))

            prob_results = compute_probability_guarantees(results, params)
            print_probability_guarantees(prob_results, params)
            print_follower_diagnostics(results, params)

            profit_samples = compute_follower_profit_samples(results, params)
            print("profit_samples shape =", profit_samples.shape)

            print("Run time (minutes) =", round(runtime_seconds / 60, 2))
            print()
            total_demand = float(np.sum(results["h_star"]))
            avg_demand = float(np.mean(results["h_star"]))

            case_data = {
                "follower_type": follower_type,
                "follower_types": follower_types,
                "cv": float(cv),
                "cv_percent": float(100 * cv),
                "sigma": to_serializable(params["sigma"]),
                "mu": to_serializable(params["mu"]),
                "annual_amplitude": to_serializable(params["annual_amplitude"]),
                "weekly_factors": to_serializable(params["weekly_factors"]),
                "C_star": float(results["C_star"]),
                "theta_star": float(results["theta_star"]),
                "objective_value": float(results["objective_value"]),
                "leader_profit": float(summary["leader_profit"]),

                "total_demand": total_demand,
                "avg_demand": avg_demand,

                "nu_hat": to_serializable(prob_results["nu_hat"]),
                "empirical_probability": to_serializable(prob_results["prob_Pi_positive_empirical"]),
                "profit_samples": to_serializable(profit_samples),
                "summary": to_serializable(summary),
            }

            all_case_data.append(case_data)

    cv_label = "_".join([f"{100 * cv:.0f}" for cv in cv_list])
    filename = f"all_type_cv_results_CV_{cv_label}.json"

    with open(filename, "w") as f:
        json.dump(all_case_data, f, indent=2)

    print("\nSaved results to", filename)

    print("\nSaved results to all_type_cv_results.json")

    # Automatically generate figures
    plot_leader_figure(filename)
    plot_probability_figure(filename)


if __name__ == "__main__":
    log_filename = "command_output.txt"

    with open(log_filename, "w", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, log_file)

        try:
            main()
        finally:
            sys.stdout = original_stdout

    print("Saved command output to", log_filename)