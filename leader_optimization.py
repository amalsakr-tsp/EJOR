"""
leader_optimization.py

Upper-level optimization for the Stackelberg model.

This module provides four leader solvers:

1) method="grid"
   Full 2D grid search over (C, theta), using the actual follower solver.

2) method="implicit_grid_theta"
   Grid over theta only. For each theta:
       - compute unconstrained aggregate demands Dhat^t(theta)
       - generate admissible C-candidates from the characterization
       - maximize Phi(C, theta) over those C-candidates

3) method="implicit_root_theta"
   Use regime-specific first-order conditions in theta together with root-finding.
   This avoids a theta grid for interior candidates, while still adding boundary
   candidates and box-boundary candidates.

4) method="implicit_regime_candidates"
   Regime-by-regime candidate construction, closest in spirit to the theorem:
       - mixed regime candidates
       - all-slack regime candidates
       - all-binding regime candidates
       - switching-boundary candidates
       - box-boundary candidates

Important:
- The "implicit" methods use the representation
      Phi(C, theta) = theta * sum_t min(Dhat^t(theta), C) - Cost(C)
  where Dhat^t(theta) is the unconstrained aggregate demand in slot t.
- After the best (C, theta) is found, the true follower equilibrium is recovered
  using the provided follower solver so the returned output is comparable to the
  grid method.

Default follower solver:
    follower_solver.solve_follower_equilibrium

You may also pass:
    follower_solver_vi.solve_follower_equilibrium
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from follower_solver import solve_follower_equilibrium as default_follower_solver

from problem_setup import investment_cost

# =============================================================================
# Cost
# =============================================================================

#def investment_cost(C: float, params: dict) -> float:
 #   """
  #  Cost(C) = kappa1 * C + 0.5 * kappa2 * C^2
   # """
    #F = float(params["F"])
    #kappa1 = float(params["kappa1"])
    #kappa2 = float(params["kappa2"])
    #return F + kappa1 * C + 0.5 * kappa2 * C**2


def investment_cost_prime(C: float, params: dict) -> float:
    """
    Cost'(C) = kappa1 + kappa2 * C
    """
    kappa1 = float(params["kappa1"])
    kappa2 = float(params["kappa2"])
    return kappa1 + kappa2 * C


def inverse_investment_cost_prime(y: float, params: dict) -> float | None:
    """
    Solve Cost'(C) = y for C.

    For quadratic cost:
        kappa1 + kappa2 * C = y
        C = (y - kappa1) / kappa2

    Returns None if not uniquely invertible.
    """
    kappa1 = float(params["kappa1"])
    kappa2 = float(params["kappa2"])
    tol = float(params.get("tol", 1e-8))

    if abs(kappa2) <= tol:
        if abs(y - kappa1) <= 10.0 * tol:
            return 0.0
        return None

    return (y - kappa1) / kappa2


# =============================================================================
# True leader objective using actual follower solver
# =============================================================================

def leader_revenue(theta: float, h_star: np.ndarray) -> float:
    return float(theta * np.sum(h_star))


def leader_objective(
    C: float,
    theta: float,
    A: np.ndarray,
    params: dict,
    follower_solver: Callable | None = None,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    True leader objective evaluated with the actual follower solver.
    """
    if follower_solver is None:
        follower_solver = default_follower_solver

    h_star, eta_star, diagnostics = follower_solver(
        C=C,
        theta=theta,
        A=A,
        params=params,
    )

    revenue = leader_revenue(theta, h_star)
    cost = investment_cost(C, params)
    value = revenue - cost

    diagnostics = dict(diagnostics)
    diagnostics["revenue"] = revenue
    diagnostics["cost"] = cost

    return float(value), h_star, eta_star, diagnostics


# =============================================================================
# Unconstrained follower system used by implicit leader methods
# =============================================================================

def _unconstrained_best_response(
    rivals: float,
    theta: float,
    A_i_t: float,
    b_i: float,
    delta: float,
    gamma: float,
) -> float:
    """
    Closed-form unconstrained best response solving the eta=0 stationarity.

    Formula is the positive root corresponding to your implicit characterization.
    """
    denom = 2.0 * b_i * delta
    if denom <= 0.0:
        raise ValueError("Need delta > 0 and b_i > 0 for the implicit formula.")

    q = b_i * theta + b_i * gamma * rivals
    disc = (q - b_i * delta) ** 2 + 4.0 * b_i * delta * A_i_t
    disc = max(0.0, float(disc))

    val = (-(q + b_i * delta) + np.sqrt(disc)) / denom
    return max(0.0, float(val))


def solve_unconstrained_slot_requests(
    theta: float,
    A_t: np.ndarray,
    params: dict,
    h_init: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Solve unconstrained slot requests hhat^t(theta) via damped best-response.
    """
    A_t = np.asarray(A_t, dtype=float)
    beta = np.asarray(params["beta"], dtype=float)
    delta = float(params["delta"])
    gamma = float(params["gamma"])
    tol = float(params.get("tol", 1e-8))

    max_iter = int(params.get("max_iter_unc", params.get("max_iter", 2000)))
    damping = float(params.get("unc_damping", 0.7))

    if delta <= 0.0:
        raise ValueError("Implicit methods require delta > 0.")

    N = A_t.size
    b = 1.0 + beta

    if h_init is None:
        h = np.zeros(N, dtype=float)
    else:
        h = np.maximum(np.asarray(h_init, dtype=float), 0.0)

    converged = False
    residual = np.inf
    it_used = 0

    for k in range(max_iter):
        h_old = h.copy()
        h_new = np.zeros_like(h_old)

        total_old = float(np.sum(h_old))
        for i in range(N):
            rivals = total_old - h_old[i]
            h_new[i] = _unconstrained_best_response(
                rivals=rivals,
                theta=theta,
                A_i_t=float(A_t[i]),
                b_i=float(b[i]),
                delta=delta,
                gamma=gamma,
            )

        h = damping * h_new + (1.0 - damping) * h_old
        residual = float(np.max(np.abs(h - h_old)))
        it_used = k + 1

        if residual <= tol:
            converged = True
            break

    diagnostics = {
        "iterations": it_used,
        "residual": residual,
        "converged": converged,
    }
    return h, diagnostics


def compute_unconstrained_demands(
    theta: float,
    A: np.ndarray,
    params: dict,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Compute hhat^t(theta) and Dhat^t(theta) for all slots.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be 2D of shape (T, N).")

    T, N = A.shape
    if T != int(params["T"]) or N != int(params["N"]):
        raise ValueError(
            f"A has shape {(T, N)}, but params specify ({params['T']}, {params['N']})."
        )

    hhat = np.zeros((T, N), dtype=float)
    Dhat = np.zeros(T, dtype=float)

    slot_diagnostics: list[dict[str, Any]] = []
    total_iterations = 0

    h_prev = np.zeros(N, dtype=float)

    for t in range(T):
        h_t, info_t = solve_unconstrained_slot_requests(
            theta=theta,
            A_t=A[t, :],
            params=params,
            h_init=h_prev,
        )
        hhat[t, :] = h_t
        Dhat[t] = float(np.sum(h_t))
        slot_diagnostics.append(info_t)
        total_iterations += int(info_t.get("iterations", 0))
        h_prev = h_t

    diagnostics = {
        "slot_diagnostics": slot_diagnostics,
        "total_iterations": total_iterations,
    }
    return hhat, Dhat, diagnostics


def compute_unconstrained_demands_derivative_fd(
    theta: float,
    A: np.ndarray,
    params: dict,
    eps: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finite-difference approximation of Dhat'(theta).

    Returns
    -------
    tuple
        (Dhat(theta), Dhat_plus, Dhat_prime)
    """
    theta_min = float(params["theta_min"])
    theta_max = float(params["theta_max"])

    if eps is None:
        eps = float(params.get("theta_fd_eps", 1e-4))

    eps = min(eps, 0.25 * max(theta_max - theta_min, 1.0))

    # central difference if possible, otherwise one-sided
    if theta - eps >= theta_min and theta + eps <= theta_max:
        _, D0, _ = compute_unconstrained_demands(theta, A, params)
        _, Dp, _ = compute_unconstrained_demands(theta + eps, A, params)
        _, Dm, _ = compute_unconstrained_demands(theta - eps, A, params)
        Dprime = (Dp - Dm) / (2.0 * eps)
        return D0, Dp, Dprime

    if theta + eps <= theta_max:
        _, D0, _ = compute_unconstrained_demands(theta, A, params)
        _, Dp, _ = compute_unconstrained_demands(theta + eps, A, params)
        Dprime = (Dp - D0) / eps
        return D0, Dp, Dprime

    _, Dm, _ = compute_unconstrained_demands(theta - eps, A, params)
    _, D0, _ = compute_unconstrained_demands(theta, A, params)
    Dprime = (D0 - Dm) / eps
    return D0, D0, Dprime


# =============================================================================
# Implicit leader objective and regimes
# =============================================================================

def implicit_leader_objective_from_Dhat(
    C: float,
    theta: float,
    Dhat: np.ndarray,
    params: dict,
) -> float:
    """
    Phi(C, theta) = theta * sum_t min(Dhat_t(theta), C) - Cost(C)
    """
    served = np.minimum(np.asarray(Dhat, dtype=float), float(C))
    revenue = float(theta * np.sum(served))
    cost = investment_cost(C, params)
    return revenue - cost


def regime_partition(C: float, Dhat: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Partition slots into:
        T_> : Dhat_t > C
        T_= : |Dhat_t - C| <= tol
        T_< : Dhat_t < C
    """
    Dhat = np.asarray(Dhat, dtype=float)
    gt = Dhat > C + tol
    eq = np.abs(Dhat - C) <= tol
    lt = Dhat < C - tol
    return gt, eq, lt


def add_candidate_value(
    values: set[float],
    x: float,
    x_min: float,
    x_max: float,
    tol: float,
) -> None:
    if not np.isfinite(x):
        return
    if x < x_min - tol or x > x_max + tol:
        return
    x_clip = min(max(float(x), x_min), x_max)
    values.add(round(x_clip, 12))


# =============================================================================
# C-candidates for fixed theta
# =============================================================================

def candidate_capacities_from_characterization(
    theta: float,
    Dhat: np.ndarray,
    params: dict,
) -> np.ndarray:
    """
    Build C-candidates for fixed theta using the characterization.

    Included:
    - C = 0, C = C_max
    - C = Dhat_t(theta)
    - all-slack candidate: C = max_t Dhat_t(theta)
    - interior mixed/all-binding candidates from Cost'(C) = theta * m

    If Dhat sorted descending is d[0] >= d[1] >= ... >= d[T-1], then:
    - exactly m strict binding slots requires:
          d[m-1] > C > d[m]
      for m = 1,...,T-1
    - all-binding requires:
          C < d[T-1]
    """
    tol = float(params.get("tol", 1e-8))
    C_min = float(params.get("C_min", 0.0))
    C_max = float(params["C_max"])

    Dhat = np.asarray(Dhat, dtype=float)
    T = Dhat.size
    Ddesc = np.sort(Dhat)[::-1]

    cand: set[float] = set()

    # box boundaries
    add_candidate_value(cand, C_min, C_min, C_max, tol)
    add_candidate_value(cand, C_max, C_min, C_max, tol)

    # switching boundaries
    for val in Dhat:
        add_candidate_value(cand, float(val), C_min, C_max, tol)

    # all-slack minimal feasible candidate
    add_candidate_value(cand, float(np.max(Dhat)), C_min, C_max, tol)

    # interior mixed / all-binding from Cost'(C)=theta*m
    for m in range(1, T + 1):
        C_m = inverse_investment_cost_prime(theta * m, params)
        if C_m is None:
            continue
        C_m = float(C_m)

        if C_m < C_min - tol or C_m > C_max + tol:
            continue

        if m < T:
            upper = float(Ddesc[m - 1])
            lower = float(Ddesc[m])
            if (upper > C_m + tol) and (C_m > lower + tol):
                add_candidate_value(cand, C_m, C_min, C_max, tol)
        else:
            if C_m < float(Ddesc[-1]) - tol:
                add_candidate_value(cand, C_m, C_min, C_max, tol)

    return np.array(sorted(cand), dtype=float)


# =============================================================================
# Grid baseline
# =============================================================================

def build_leader_grids(params: dict) -> tuple[np.ndarray, np.ndarray]:
    C_grid = np.linspace(float(params["C_min"]), float(params["C_max"]), int(params["nC"]))
    theta_grid = np.linspace(float(params["theta_min"]), float(params["theta_max"]), int(params["nTheta"]))
    return C_grid, theta_grid


def optimize_leader_grid(
    A: np.ndarray,
    params: dict,
    follower_solver: Callable | None = None,
) -> dict[str, Any]:
    """
    Full 2D grid search over (C, theta).
    """
    if follower_solver is None:
        follower_solver = default_follower_solver

    A = np.asarray(A, dtype=float)
    C_grid, theta_grid = build_leader_grids(params)

    objective_grid = np.full((len(C_grid), len(theta_grid)), -np.inf, dtype=float)

    best_value = -np.inf
    best_C = None
    best_theta = None
    best_h = None
    best_eta = None
    best_diag = None

    for i, C in enumerate(C_grid):
        for j, theta in enumerate(theta_grid):
            value, h_star, eta_star, diag = leader_objective(
                C=float(C),
                theta=float(theta),
                A=A,
                params=params,
                follower_solver=follower_solver,
            )
            objective_grid[i, j] = value

            if value > best_value:
                best_value = value
                best_C = float(C)
                best_theta = float(theta)
                best_h = np.array(h_star, copy=True)
                best_eta = np.array(eta_star, copy=True)
                best_diag = dict(diag)

    if best_C is None:
        raise RuntimeError("Grid leader optimization failed.")

    return {
        "method": "grid",
        "C_star": best_C,
        "theta_star": best_theta,
        "objective_value": float(best_value),
        "h_star": best_h,
        "eta_star": best_eta,
        "C_grid": C_grid,
        "theta_grid": theta_grid,
        "objective_grid": objective_grid,
        "diagnostics": best_diag,
    }


# =============================================================================
# Method 2: implicit C + theta grid
# =============================================================================

def build_theta_grid_implicit(params: dict) -> np.ndarray:
    n_theta = int(params.get("nTheta_implicit", max(200, int(params.get("nTheta", 40)))))
    return np.linspace(float(params["theta_min"]), float(params["theta_max"]), n_theta)


def optimize_leader_implicit_grid_theta(
    A: np.ndarray,
    params: dict,
    follower_solver: Callable | None = None,
) -> dict[str, Any]:
    """
    Option 2:
    - grid only over theta
    - generate C-candidates from the characterization
    """
    if follower_solver is None:
        follower_solver = default_follower_solver

    theta_grid = build_theta_grid_implicit(params)

    best_value = -np.inf
    best_C = None
    best_theta = None
    best_hhat = None
    best_Dhat = None
    best_unc_diag = None

    objective_theta = np.full(theta_grid.shape, -np.inf, dtype=float)
    n_candidates_by_theta = np.zeros(theta_grid.shape, dtype=int)

    for k, theta in enumerate(theta_grid):
        hhat, Dhat, unc_diag = compute_unconstrained_demands(float(theta), A, params)
        C_candidates = candidate_capacities_from_characterization(float(theta), Dhat, params)
        n_candidates_by_theta[k] = len(C_candidates)

        if len(C_candidates) == 0:
            continue

        local_best = -np.inf

        for C in C_candidates:
            value = implicit_leader_objective_from_Dhat(float(C), float(theta), Dhat, params)
            local_best = max(local_best, value)

            if value > best_value:
                best_value = value
                best_C = float(C)
                best_theta = float(theta)
                best_hhat = np.array(hhat, copy=True)
                best_Dhat = np.array(Dhat, copy=True)
                best_unc_diag = dict(unc_diag)

        objective_theta[k] = local_best

    if best_C is None or best_theta is None:
        raise RuntimeError("implicit_grid_theta failed.")

    # Recover true follower equilibrium at the winner
    true_value, h_star, eta_star, true_diag = leader_objective(
        C=best_C,
        theta=best_theta,
        A=A,
        params=params,
        follower_solver=follower_solver,
    )

    return {
        "method": "implicit_grid_theta",
        "C_star": best_C,
        "theta_star": best_theta,
        "objective_value": float(true_value),
        "h_star": h_star,
        "eta_star": eta_star,
        "theta_grid": theta_grid,
        "objective_theta": objective_theta,
        "diagnostics": {
            "best_unconstrained_h": best_hhat,
            "best_unconstrained_Dhat": best_Dhat,
            "unconstrained_diagnostics": best_unc_diag,
            "n_candidates_by_theta": n_candidates_by_theta,
            "true_follower_diagnostics": true_diag,
        },
    }


# =============================================================================
# Root utilities for methods 1 and 3
# =============================================================================

def bisect_root(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float | None:
    """
    Bisection root finder on [a, b] when sign change exists.
    """
    fa = f(a)
    fb = f(b)

    if not np.isfinite(fa) or not np.isfinite(fb):
        return None

    if abs(fa) <= tol:
        return float(a)
    if abs(fb) <= tol:
        return float(b)

    if fa * fb > 0.0:
        return None

    left = float(a)
    right = float(b)
    f_left = float(fa)
    f_right = float(fb)

    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        f_mid = f(mid)

        if not np.isfinite(f_mid):
            return None

        if abs(f_mid) <= tol or abs(right - left) <= tol:
            return float(mid)

        if f_left * f_mid <= 0.0:
            right = mid
            f_right = f_mid
        else:
            left = mid
            f_left = f_mid

    return float(0.5 * (left + right))


def find_roots_on_grid(
    f: Callable[[float], float],
    theta_min: float,
    theta_max: float,
    n_scan: int = 200,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> list[float]:
    """
    Find sign-change roots by scanning a grid and bisecting intervals.
    """
    grid = np.linspace(theta_min, theta_max, n_scan)
    vals = []

    for x in grid:
        try:
            vals.append(float(f(float(x))))
        except Exception:
            vals.append(np.nan)

    roots: list[float] = []

    for i in range(len(grid) - 1):
        a, b = float(grid[i]), float(grid[i + 1])
        fa, fb = vals[i], vals[i + 1]

        if not np.isfinite(fa) or not np.isfinite(fb):
            continue

        if abs(fa) <= tol:
            roots.append(a)
            continue

        if fa * fb < 0.0:
            r = bisect_root(f, a, b, tol=tol, max_iter=max_iter)
            if r is not None:
                roots.append(r)

    # deduplicate
    out: list[float] = []
    for r in sorted(roots):
        if not out or abs(r - out[-1]) > 10.0 * tol:
            out.append(r)
    return out


# =============================================================================
# Method 1: implicit C + root-finding in theta
# =============================================================================

def mixed_regime_theta_foc(
    theta: float,
    C: float,
    Dhat: np.ndarray,
    Dprime: np.ndarray,
    tol: float,
) -> float:
    gt, eq, lt = regime_partition(C, Dhat, tol)
    if np.any(eq):
        return np.nan
    return float(
        np.sum(gt) * C
        + np.sum(Dhat[lt])
        + theta * np.sum(Dprime[lt])
    )


def slack_regime_theta_foc(
    theta: float,
    Dhat: np.ndarray,
    Dprime: np.ndarray,
) -> float:
    return float(np.sum(Dhat) + theta * np.sum(Dprime))


def optimize_leader_implicit_root_theta(
    A: np.ndarray,
    params: dict,
    follower_solver: Callable | None = None,
) -> dict[str, Any]:
    """
    Option 1:
    Use regime-specific root finding in theta from the first-order conditions.

    Practical implementation:
    - still relies on finite-difference Dhat'(theta)
    - scans theta to bracket roots, then bisects
    - adds regime-switching and box-boundary candidates
    """
    if follower_solver is None:
        follower_solver = default_follower_solver

    tol = float(params.get("tol", 1e-8))
    theta_min = float(params["theta_min"])
    theta_max = float(params["theta_max"])
    n_scan = int(params.get("nTheta_root_scan", 200))
    root_tol = float(params.get("theta_root_tol", 1e-6))
    root_iter = int(params.get("max_iter_theta_root", 100))

    # Candidate theta values will be gathered here
    theta_candidates: set[float] = set()
    add_candidate_value(theta_candidates, theta_min, theta_min, theta_max, tol)
    add_candidate_value(theta_candidates, theta_max, theta_min, theta_max, tol)

    # Use a scan over theta only to build root brackets, not optimize directly
    theta_scan = np.linspace(theta_min, theta_max, n_scan)

    # Cache Dhat and Dprime at scan points for later boundary evaluation
    cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    for th in theta_scan:
        Dhat, _, Dprime = compute_unconstrained_demands_derivative_fd(float(th), A, params)
        cache[float(th)] = (Dhat, Dprime)

        # All-slack theta FOC candidate
        g_slack = slack_regime_theta_foc(float(th), Dhat, Dprime)
        if abs(g_slack) <= max(root_tol, 1e-5):
            add_candidate_value(theta_candidates, float(th), theta_min, theta_max, tol)

    # Root-finding for all-slack FOC
    def f_slack(theta: float) -> float:
        Dhat, _, Dprime = compute_unconstrained_demands_derivative_fd(theta, A, params)
        return slack_regime_theta_foc(theta, Dhat, Dprime)

    for r in find_roots_on_grid(
        f_slack,
        theta_min,
        theta_max,
        n_scan=n_scan,
        tol=root_tol,
        max_iter=root_iter,
    ):
        add_candidate_value(theta_candidates, r, theta_min, theta_max, tol)

    # For mixed/all-binding, theta enters together with C.
    # We use regime-generated C(theta) candidates from the characterization,
    # then check theta FOC and keep roots/bracket points.
    for th in theta_scan:
        Dhat, Dprime = cache[float(th)]
        C_candidates = candidate_capacities_from_characterization(float(th), Dhat, params)

        for C in C_candidates:
            gt, eq, lt = regime_partition(float(C), Dhat, tol)

            if np.any(eq):
                # switching boundary candidate
                add_candidate_value(theta_candidates, float(th), theta_min, theta_max, tol)
                continue

            # mixed regime candidate
            if np.any(gt) and np.any(lt):
                val = mixed_regime_theta_foc(float(th), float(C), Dhat, Dprime, tol)
                if np.isfinite(val) and abs(val) <= max(root_tol, 1e-5):
                    add_candidate_value(theta_candidates, float(th), theta_min, theta_max, tol)

    # Evaluate all theta candidates by maximizing over C candidates
    best_value = -np.inf
    best_C = None
    best_theta = None
    best_Dhat = None

    theta_candidates_sorted = np.array(sorted(theta_candidates), dtype=float)
    objective_theta = np.full(theta_candidates_sorted.shape, -np.inf, dtype=float)

    for k, theta in enumerate(theta_candidates_sorted):
        _, Dhat, _ = compute_unconstrained_demands(float(theta), A, params)
        C_candidates = candidate_capacities_from_characterization(float(theta), Dhat, params)

        local_best = -np.inf
        for C in C_candidates:
            value = implicit_leader_objective_from_Dhat(float(C), float(theta), Dhat, params)
            local_best = max(local_best, value)

            if value > best_value:
                best_value = value
                best_C = float(C)
                best_theta = float(theta)
                best_Dhat = np.array(Dhat, copy=True)

        objective_theta[k] = local_best

    if best_C is None or best_theta is None:
        raise RuntimeError("implicit_root_theta failed.")

    true_value, h_star, eta_star, true_diag = leader_objective(
        C=best_C,
        theta=best_theta,
        A=A,
        params=params,
        follower_solver=follower_solver,
    )

    return {
        "method": "implicit_root_theta",
        "C_star": best_C,
        "theta_star": best_theta,
        "objective_value": float(true_value),
        "h_star": h_star,
        "eta_star": eta_star,
        "theta_candidates": theta_candidates_sorted,
        "objective_theta": objective_theta,
        "diagnostics": {
            "best_Dhat": best_Dhat,
            "true_follower_diagnostics": true_diag,
        },
    }


# =============================================================================
# Method 3: theorem-style regime candidate construction
# =============================================================================

def optimize_leader_implicit_regime_candidates(
    A: np.ndarray,
    params: dict,
    follower_solver: Callable | None = None,
) -> dict[str, Any]:
    """
    Option 3:
    Regime-by-regime candidate construction.

    Practical implementation:
    - uses a theta scan only to collect regime-switching and admissibility candidates
    - for each theta, generates:
        * mixed regime interior C candidates
        * all-slack candidate
        * all-binding candidate
        * switching boundaries C = Dhat_t(theta)
        * box boundaries
    - compares all resulting candidates
    """
    if follower_solver is None:
        follower_solver = default_follower_solver

    tol = float(params.get("tol", 1e-8))
    theta_min = float(params["theta_min"])
    theta_max = float(params["theta_max"])
    n_theta = int(params.get("nTheta_regime", max(300, int(params.get("nTheta", 40)))))

    theta_grid = np.linspace(theta_min, theta_max, n_theta)

    best_value = -np.inf
    best_C = None
    best_theta = None
    best_Dhat = None

    all_candidates: list[tuple[float, float]] = []

    for theta in theta_grid:
        _, Dhat, _ = compute_unconstrained_demands(float(theta), A, params)
        Ddesc = np.sort(Dhat)[::-1]

        C_candidates: set[float] = set()
        C_min = float(params["C_min"])
        C_max = float(params["C_max"])

        # box boundary candidates
        add_candidate_value(C_candidates, C_min, C_min, C_max, tol)
        add_candidate_value(C_candidates, C_max, C_min, C_max, tol)

        # switching boundaries
        for d in Dhat:
            add_candidate_value(C_candidates, float(d), C_min, C_max, tol)

        # all-slack candidate
        add_candidate_value(C_candidates, float(np.max(Dhat)), C_min, C_max, tol)

        # mixed and all-binding interior candidates from Cost'(C)=theta*m
        T = len(Dhat)
        for m in range(1, T + 1):
            C_m = inverse_investment_cost_prime(float(theta) * m, params)
            if C_m is None:
                continue
            C_m = float(C_m)

            if C_m < C_min - tol or C_m > C_max + tol:
                continue

            if m < T:
                upper = float(Ddesc[m - 1])
                lower = float(Ddesc[m])
                if (upper > C_m + tol) and (C_m > lower + tol):
                    add_candidate_value(C_candidates, C_m, C_min, C_max, tol)
            else:
                if C_m < float(Ddesc[-1]) - tol:
                    add_candidate_value(C_candidates, C_m, C_min, C_max, tol)

        for C in sorted(C_candidates):
            all_candidates.append((float(C), float(theta)))
            value = implicit_leader_objective_from_Dhat(float(C), float(theta), Dhat, params)

            if value > best_value:
                best_value = value
                best_C = float(C)
                best_theta = float(theta)
                best_Dhat = np.array(Dhat, copy=True)

    if best_C is None or best_theta is None:
        raise RuntimeError("implicit_regime_candidates failed.")

    true_value, h_star, eta_star, true_diag = leader_objective(
        C=best_C,
        theta=best_theta,
        A=A,
        params=params,
        follower_solver=follower_solver,
    )

    return {
        "method": "implicit_regime_candidates",
        "C_star": best_C,
        "theta_star": best_theta,
        "objective_value": float(true_value),
        "h_star": h_star,
        "eta_star": eta_star,
        "all_candidates": all_candidates,
        "diagnostics": {
            "best_Dhat": best_Dhat,
            "n_total_candidates": len(all_candidates),
            "true_follower_diagnostics": true_diag,
        },
    }


# =============================================================================
# Unified wrapper
# =============================================================================

def optimize_leader(
    A: np.ndarray,
    params: dict,
    method: str = "implicit_grid_theta",
    follower_solver: Callable | None = None,
) -> dict[str, Any]:
    """
    Unified leader optimizer.

    Supported methods:
    - "grid"
    - "implicit_grid_theta"
    - "implicit_root_theta"
    - "implicit_regime_candidates"
    """
    method = method.lower().strip()

    if method == "grid":
        return optimize_leader_grid(A=A, params=params, follower_solver=follower_solver)

    if method == "implicit_grid_theta":
        return optimize_leader_implicit_grid_theta(
            A=A,
            params=params,
            follower_solver=follower_solver,
        )

    if method == "implicit_root_theta":
        return optimize_leader_implicit_root_theta(
            A=A,
            params=params,
            follower_solver=follower_solver,
        )

    if method == "implicit_regime_candidates":
        return optimize_leader_implicit_regime_candidates(
            A=A,
            params=params,
            follower_solver=follower_solver,
        )

    raise ValueError(
        "method must be one of: "
        "'grid', 'implicit_grid_theta', 'implicit_root_theta', "
        "'implicit_regime_candidates'"
    )