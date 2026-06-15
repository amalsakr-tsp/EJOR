from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

FIXED_PARAMS_FILE = Path("fixed_market_params.npz")


def sigma_from_cv(cv: float) -> float:
    return float(np.sqrt(np.log(1.0 + cv**2)))


def generate_structural_params(N: int, rng: np.random.Generator, cv: float = 0.15) -> dict:
    if N != 3:
        raise ValueError("This setup assumes N = 5.")

    # Two small, one moderate, two high
    mu = np.array([1000.0, 2000.0, 3000.0], dtype=float)

    weekly_factors = np.array([
        [1.05, 1.03, 1.00, 0.90, 0.90, 0.80, 0.70],  # small 1
        [1.10, 1.05, 1.00, 0.95, 0.94, 0.87, 0.75],  # small 2
        [1.15, 1.10, 1.05, 1.00, 0.95, 0.87, 0.75],  # moderate
    ], dtype=float)

    sigma_value = sigma_from_cv(cv)
    sigma = np.full(N, sigma_value, dtype=float)

    annual_amplitude = np.array([0.10, 0.14, 0.18], dtype=float)

    return {
        "mu": mu,
        "sigma": sigma,
        "weekly_factors": weekly_factors,
        "annual_amplitude": annual_amplitude,
    }


def save_structural_params(structural: dict) -> None:
    np.savez(
        FIXED_PARAMS_FILE,
        mu=structural["mu"],
        sigma=structural["sigma"],
        weekly_factors=structural["weekly_factors"],
        annual_amplitude=structural["annual_amplitude"],
    )


def load_structural_params() -> dict:
    data = np.load(FIXED_PARAMS_FILE)
    return {
        "mu": data["mu"],
        "sigma": data["sigma"],
        "weekly_factors": data["weekly_factors"],
        "annual_amplitude": data["annual_amplitude"],
    }


def generate_alpha_beta_by_type(
    N: int,
    follower_type: int,
    beta_L: float,
    alpha_L: float,
    alpha_H: float,
) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-8

    alpha_zero = np.full(N, eps)
    alpha_moderate = np.random.default_rng(100).uniform(alpha_L, alpha_H, N)
    alpha_high = np.random.default_rng(150).uniform(alpha_H + eps, 1.0 - eps, N)

    beta_zero = np.zeros(N, dtype=float)
    beta_moderate = np.random.default_rng(200).uniform(eps, beta_L, N)
    beta_high = np.random.default_rng(250).uniform(beta_L + eps, 2.0 * beta_L, N)
    beta_extremely_high = np.random.default_rng(300).uniform(2 * beta_L + eps, 10, N)

    if follower_type == 1:
        alpha = alpha_zero
        beta = beta_zero
    elif follower_type == 2:
        alpha = alpha_moderate
        beta = beta_moderate
    elif follower_type == 3:
        alpha = alpha_moderate
        beta = beta_high
    elif follower_type == 4:
        alpha = alpha_high
        beta = beta_extremely_high
    else:
        raise ValueError("follower_type must be an integer from 1 to 4.")

    return alpha.copy(), beta.copy()


def generate_alpha_beta_from_list(
    follower_types: list[int] | np.ndarray,
    beta_L: float,
    alpha_L: float,
    alpha_H: float,
) -> tuple[np.ndarray, np.ndarray]:
    follower_types = np.asarray(follower_types, dtype=int)
    N = len(follower_types)

    alpha = np.zeros(N, dtype=float)
    beta = np.zeros(N, dtype=float)

    for i, ftype in enumerate(follower_types):
        alpha_vec, beta_vec = generate_alpha_beta_by_type(
            N=N,
            follower_type=int(ftype),
            beta_L=beta_L,
            alpha_L=alpha_L,
            alpha_H=alpha_H,
        )
        alpha[i] = alpha_vec[i]
        beta[i] = beta_vec[i]

    return alpha, beta


def build_params(
    follower_type: int | None = 1,
    follower_types: list[int] | np.ndarray | None = None,
    alpha: np.ndarray | None = None,
    beta: np.ndarray | None = None,
    reuse_fixed_market: bool = True,
    cv: float = 0.15,
) -> dict:
    N = 3
    rng = np.random.default_rng(42)

    beta_L = 1.0
    alpha_L = 0.9
    alpha_H = 0.95

    if alpha is not None or beta is not None:
        if alpha is None or beta is None:
            raise ValueError("Both alpha and beta must be provided together.")

        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)

        if alpha.shape != (N,) or beta.shape != (N,):
            raise ValueError(f"alpha and beta must both have shape ({N},).")

        follower_type_info = "manual"
        follower_types_info = None

    elif follower_types is not None:
        follower_types = np.asarray(follower_types, dtype=int)

        if follower_types.shape != (N,):
            raise ValueError(f"follower_types must have shape ({N},).")

        alpha, beta = generate_alpha_beta_from_list(
            follower_types=follower_types,
            beta_L=beta_L,
            alpha_L=alpha_L,
            alpha_H=alpha_H,
        )

        follower_type_info = "mixed"
        follower_types_info = follower_types.copy()

    else:
        if follower_type is None:
            raise ValueError("Provide either follower_type, follower_types, or manual alpha and beta.")

        alpha, beta = generate_alpha_beta_by_type(
            N=N,
            follower_type=follower_type,
            beta_L=beta_L,
            alpha_L=alpha_L,
            alpha_H=alpha_H,
        )

        follower_type_info = int(follower_type)
        follower_types_info = None

    gamma = 0.01

    xi = np.array([
        (
            N
            + ((N + 1) / 2.0) * beta[i]
            + 0.5 * np.sum(beta[np.arange(N) != i])
        ) / (1.0 + beta[i])
        for i in range(N)
    ])

    delta_min = gamma * np.max(xi)
    delta = 0.0365

    hours = 24 * 365 * 5
    month = hours / (24 * 30)
    season_period = 24 * 30 * 3

    if reuse_fixed_market and FIXED_PARAMS_FILE.exists():
        structural = load_structural_params()
    else:
        structural = generate_structural_params(N=N, rng=rng, cv=cv)
        if reuse_fixed_market:
            save_structural_params(structural)

    #L_phi = params["T"] * (
     #       params["C_max"]
      #      + params["theta_max"] * params["N"] / (params["delta"] - params["gamma"])
    #)

    #nTheta_implicit = ceil(1 + L_phi * (theta_max - theta_min) / (2 * eta))

    return {
        "N": N,
        "T": hours,
        "n_samples": 1000,

        "alpha": alpha,
        "beta": beta,

        "beta_L": beta_L,
        "alpha_L": alpha_L,
        "alpha_H": alpha_H,
        "cv": cv,

        "follower_type": follower_type_info,
        "follower_types": follower_types_info,

        "w": 1,
        "gamma": gamma,
        "xi": xi,
        "delta_min": float(delta_min),
        "delta": float(delta),

        "mu": structural["mu"],
        "sigma": structural["sigma"],
        "weekly_factors": structural["weekly_factors"],
        "annual_amplitude": structural["annual_amplitude"],
        "season_period": season_period,

        "C_min": 0.0,
        "C_max": 5000.0,
        "theta_min": 0.0,
        "theta_max": 200.0,

        "nC": 40,
        "nTheta": 40,

        "tol": 1e-8,
        "max_iter": 1000,

        "step_size": 1e-3,
        "step_decay": 0.5,
        "min_step_size": 1e-8,

        "kappa1": 10 + 16 * month,
        "kappa2": 0.02,
        "F": 2000,

        "seed": 42,
        "eta_max": 1e5,
    }


def generate_a_samples(params: dict) -> np.ndarray:
    rng = np.random.default_rng(params["seed"])

    T = params["T"]
    N = params["N"]
    S = params["n_samples"]

    mu = params["mu"]
    sigma = params["sigma"]
    target_cv = float(params["cv"])
    weekly_factors = params["weekly_factors"]
    annual_amplitude = params["annual_amplitude"]
    season_period = params["season_period"]

    samples = np.zeros((T, N, S), dtype=float)

    # One annual shock per follower and sample path.
    # Fixed across all hours, but different across followers.
    annual_shock = np.zeros((N, S), dtype=float)

    for i in range(N):
        annual_shock[i, :] = rng.lognormal(
            mean=-0.5 * sigma[i]**2,
            sigma=sigma[i],
            size=S,
        )

    for t in range(T):
        day_of_week = (t // 24) % 7
        weekly_factor = weekly_factors[:, day_of_week]

        annual_factor = 1.0 + annual_amplitude * np.sin(
            2.0 * np.pi * t / season_period
        )

        seasonal_factor = weekly_factor * annual_factor

        for i in range(N):
            samples[t, i, :] = (
                mu[i]
                * seasonal_factor[i]
                * annual_shock[i, :]
                * 1.0
            )

    return samples


def rockafellar_cvar(loss_samples: np.ndarray, alpha: float) -> float:
    loss_samples = np.asarray(loss_samples, dtype=float).ravel()

    if loss_samples.size == 0:
        raise ValueError("loss_samples must be nonempty.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1).")

    eta_candidates = np.unique(np.sort(loss_samples))
    scale = 1.0 / ((1.0 - alpha) * loss_samples.size)

    best_value = np.inf
    for eta in eta_candidates:
        value = eta + scale * np.maximum(loss_samples - eta, 0.0).sum()
        if value < best_value:
            best_value = value

    return float(best_value)


def compute_A_matrix(a_samples: np.ndarray, params: dict) -> np.ndarray:
    alpha = params["alpha"]
    beta = params["beta"]
    w = params["w"]

    T, N, _ = a_samples.shape
    A = np.zeros((T, N), dtype=float)

    for t in range(T):
        for i in range(N):
            a_ti = a_samples[t, i, :]
            mean_val = np.mean(a_ti)
            cvar_val = rockafellar_cvar(loss_samples=-a_ti, alpha=alpha[i])
            A[t, i] = w * mean_val - beta[i] * cvar_val

    return A


def investment_cost(C: float, params: dict) -> float:
    return params["F"] + params["kappa1"] * C + 0.5 * params["kappa2"] * C**2


def print_input_summary(params: dict, A: np.ndarray) -> None:
    np.set_printoptions(precision=4, suppress=True)

    print("\n" + "=" * 70)
    print("MODEL SETUP")
    print("=" * 70)
    print(f"N          = {params['N']}")
    print(f"T          = {params['T']}")
    print(f"n_samples  = {params['n_samples']}")
    print(f"delta      = {params['delta']}")
    print(f"gamma      = {params['gamma']}")

    print("\n" + "=" * 70)
    print("RISK INFO")
    print("=" * 70)
    print(f"follower_type   = {params['follower_type']}")
    print(f"follower_types  = {params['follower_types']}")
    print(f"alpha           = {params['alpha']}")
    print(f"beta            = {params['beta']}")

    print("\n" + "=" * 70)
    print("A MATRIX")
    print("=" * 70)
    print(A)
    print(f"mu         = {params['mu']}")
    print(f"sigma      = {params['sigma']}")
    print(f"weekly     = {params['weekly_factors']}")
    print(f"annual amp = {params['annual_amplitude']}")