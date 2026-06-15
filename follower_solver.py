"""
follower_solver.py

Lower-level follower equilibrium solver for the specialized Stackelberg model.

This module solves, for fixed (C, theta), the follower variational equilibrium
slot by slot.

Expected use:
    h_star, eta_star, diagnostics = solve_follower_equilibrium(C, theta, A, params)

Inputs
------
C : float
    Installed capacity selected by the leader.
theta : float
    Tariff selected by the leader.
A : np.ndarray
    Matrix of shape (T, N) with entries
        A_i^t = E[a_i^t] - beta_i * CVaR_{alpha_i}(-a_i^t)
params : dict
    Dictionary containing model and numerical parameters.

Outputs
-------
h_star : np.ndarray
    Follower equilibrium requests, shape (T, N).
eta_star : np.ndarray
    Common multiplier for shared capacity, shape (T,).
diagnostics : dict
    Solver diagnostics.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _compute_b(beta: np.ndarray) -> np.ndarray:
    """
    Compute b_i = 1 + beta_i.
    """
    return 1.0 + beta


def best_response_i(
    i: int,
    h: np.ndarray,
    theta: float,
    eta: float,
    A_t: np.ndarray,
    beta: np.ndarray,
    delta: float,
    gamma: float,
) -> float:
    """
    Best response of follower i in one time slot, holding rivals fixed.

    For active h_i > 0, the first-order condition is:
        A_i / (1 + h_i) = b_i * theta + b_i * delta * h_i
                          + b_i * gamma * sum_{j != i} h_j + eta

    Rearranged into a quadratic:
        b_i * delta * h_i^2
      + (b_i * (theta + gamma * S_i + delta) + eta) * h_i
      + (b_i * (theta + gamma * S_i) + eta - A_i) = 0

    We take the nonnegative root and then apply max(0, ·).

    Parameters
    ----------
    i : int
        Follower index.
    h : np.ndarray
        Current slot vector, shape (N,).
    theta : float
        Tariff.
    eta : float
        Common multiplier for the shared capacity constraint.
    A_t : np.ndarray
        Vector A^t of shape (N,) for the current time slot.
    beta : np.ndarray
        Risk aversion coefficients, shape (N,).
    delta : float
        Own quadratic congestion parameter.
    gamma : float
        Cross congestion parameter.

    Returns
    -------
    float
        Updated best response of follower i.
    """
    b_i = 1.0 + beta[i]
    A_i = A_t[i]
    S_i = float(np.sum(h) - h[i])

    a_quad = b_i * delta
    b_quad = b_i * (theta + gamma * S_i + delta) + eta
    c_quad = b_i * (theta + gamma * S_i) + eta - A_i

    disc = b_quad**2 - 4.0 * a_quad * c_quad
    disc = max(disc, 0.0)

    root = (-b_quad + np.sqrt(disc)) / (2.0 * a_quad)
    return float(max(0.0, root))


def solve_slot_given_eta(
    theta: float,
    eta: float,
    A_t: np.ndarray,
    beta: np.ndarray,
    delta: float,
    gamma: float,
    tol: float,
    max_iter: int,
    h_init: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Solve one time-slot follower equilibrium for a fixed eta by Gauss-Seidel
    best-response iteration.

    Parameters
    ----------
    theta : float
        Tariff.
    eta : float
        Common multiplier.
    A_t : np.ndarray
        Current time-slot A-vector, shape (N,).
    beta : np.ndarray
        Beta vector, shape (N,).
    delta : float
        Own congestion parameter.
    gamma : float
        Cross congestion parameter.
    tol : float
        Iteration tolerance.
    max_iter : int
        Maximum number of fixed-point iterations.
    h_init : np.ndarray | None
        Optional warm start.

    Returns
    -------
    tuple
        (h_t, diagnostics)
    """
    N = A_t.size

    if h_init is None:
        h = np.zeros(N, dtype=float)
    else:
        h = np.array(h_init, dtype=float, copy=True)

    residual = np.inf
    it_used = 0

    for k in range(max_iter):
        h_old = h.copy()

        for i in range(N):
            h[i] = best_response_i(
                i=i,
                h=h,
                theta=theta,
                eta=eta,
                A_t=A_t,
                beta=beta,
                delta=delta,
                gamma=gamma,
            )

        residual = float(np.max(np.abs(h - h_old)))
        it_used = k + 1

        if residual <= tol:
            break

    diagnostics = {
        "iterations": it_used,
        "residual": residual,
        "converged": residual <= tol,
    }
    return h, diagnostics


def capacity_residual(
    eta: float,
    C: float,
    theta: float,
    A_t: np.ndarray,
    beta: np.ndarray,
    delta: float,
    gamma: float,
    tol: float,
    max_iter: int,
    h_init: np.ndarray | None = None,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """
    Compute:
        G(eta) = sum_i h_i(eta) - C

    Returns
    -------
    tuple
        (G(eta), h_t(eta), diagnostics)
    """
    h_t, info = solve_slot_given_eta(
        theta=theta,
        eta=eta,
        A_t=A_t,
        beta=beta,
        delta=delta,
        gamma=gamma,
        tol=tol,
        max_iter=max_iter,
        h_init=h_init,
    )

    residual = float(np.sum(h_t) - C)
    return residual, h_t, info


def solve_slot_equilibrium(
    C: float,
    theta: float,
    A_t: np.ndarray,
    params: dict,
    h_init: np.ndarray | None = None,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """
    Solve the follower equilibrium for one time slot.

    Logic
    -----
    1. Solve with eta = 0.
    2. If sum_i h_i <= C, capacity is slack and eta* = 0.
    3. Otherwise solve G(eta) = sum_i h_i(eta) - C = 0 by bisection.

    Parameters
    ----------
    C : float
        Capacity.
    theta : float
        Tariff.
    A_t : np.ndarray
        A-vector for one time slot, shape (N,).
    params : dict
        Parameter dictionary.
    h_init : np.ndarray | None
        Optional warm start.

    Returns
    -------
    tuple
        (h_t_star, eta_t_star, diagnostics)
    """
    beta = np.asarray(params["beta"], dtype=float)
    delta = float(params["delta"])
    gamma = float(params["gamma"])
    tol = float(params["tol"])
    max_iter = int(params["max_iter"])
    eta_max = float(params["eta_max"])

    # Step 1: unconstrained solve with eta = 0
    h_unc, info_unc = solve_slot_given_eta(
        theta=theta,
        eta=0.0,
        A_t=A_t,
        beta=beta,
        delta=delta,
        gamma=gamma,
        tol=tol,
        max_iter=max_iter,
        h_init=h_init,
    )

    load_unc = float(np.sum(h_unc))
    if load_unc <= C + tol:
        diagnostics = {
            "binding": False,
            "eta_iterations": 0,
            "slot_solver_iterations": info_unc["iterations"],
            "slot_solver_converged": info_unc["converged"],
            "slot_solver_residual": info_unc["residual"],
        }
        return h_unc, 0.0, diagnostics

    # Step 2: capacity binds, solve scalar root in eta
    eta_lo = 0.0
    eta_hi = eta_max

    G_lo, h_lo, info_lo = capacity_residual(
        eta=eta_lo,
        C=C,
        theta=theta,
        A_t=A_t,
        beta=beta,
        delta=delta,
        gamma=gamma,
        tol=tol,
        max_iter=max_iter,
        h_init=h_unc,
    )

    G_hi, h_hi, info_hi = capacity_residual(
        eta=eta_hi,
        C=C,
        theta=theta,
        A_t=A_t,
        beta=beta,
        delta=delta,
        gamma=gamma,
        tol=tol,
        max_iter=max_iter,
        h_init=np.zeros_like(h_unc),
    )

    # Expand eta_hi if needed
    expand_count = 0
    while G_hi > 0.0 and expand_count < 20:
        eta_hi *= 2.0
        G_hi, h_hi, info_hi = capacity_residual(
            eta=eta_hi,
            C=C,
            theta=theta,
            A_t=A_t,
            beta=beta,
            delta=delta,
            gamma=gamma,
            tol=tol,
            max_iter=max_iter,
            h_init=h_hi,
        )
        expand_count += 1

    if G_lo < 0.0:
        # This should not happen if the unconstrained solution is actually binding.
        diagnostics = {
            "binding": True,
            "warning": "Unexpected sign at eta_lo.",
            "eta_iterations": 0,
            "slot_solver_iterations": info_lo["iterations"],
            "slot_solver_converged": info_lo["converged"],
            "slot_solver_residual": info_lo["residual"],
        }
        return h_lo, eta_lo, diagnostics

    if G_hi > 0.0:
        raise RuntimeError(
            "Failed to bracket the capacity-binding multiplier eta. "
            "Increase eta_max or check model parameters."
        )

    eta_mid = None
    h_mid = None
    info_mid: dict[str, Any] = {}
    eta_iterations = 0

    for k in range(max_iter):
        eta_mid = 0.5 * (eta_lo + eta_hi)

        G_mid, h_mid, info_mid = capacity_residual(
            eta=eta_mid,
            C=C,
            theta=theta,
            A_t=A_t,
            beta=beta,
            delta=delta,
            gamma=gamma,
            tol=tol,
            max_iter=max_iter,
            h_init=0.5 * (h_lo + h_hi),
        )

        eta_iterations = k + 1

        if abs(G_mid) <= tol or (eta_hi - eta_lo) <= tol:
            diagnostics = {
                "binding": True,
                "eta_iterations": eta_iterations,
                "slot_solver_iterations": info_mid["iterations"],
                "slot_solver_converged": info_mid["converged"],
                "slot_solver_residual": info_mid["residual"],
                "capacity_residual": G_mid,
            }
            return h_mid, float(eta_mid), diagnostics

        if G_mid > 0.0:
            eta_lo = eta_mid
            h_lo = h_mid
        else:
            eta_hi = eta_mid
            h_hi = h_mid

    # Fallback return if bisection hits max_iter
    if eta_mid is None or h_mid is None:
        eta_mid = eta_hi
        h_mid = h_hi

    diagnostics = {
        "binding": True,
        "eta_iterations": eta_iterations,
        "slot_solver_iterations": info_mid.get("iterations", 0),
        "slot_solver_converged": info_mid.get("converged", False),
        "slot_solver_residual": info_mid.get("residual", np.inf),
        "capacity_residual": float(np.sum(h_mid) - C),
        "warning": "Bisection reached max_iter before full convergence.",
    }
    return h_mid, float(eta_mid), diagnostics


def solve_follower_equilibrium(
    C: float,
    theta: float,
    A: np.ndarray,
    params: dict,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Solve the full follower equilibrium for all time slots.

    Since the specialized follower problem separates by slot once (C, theta)
    and A are fixed, we solve each slot independently.

    Parameters
    ----------
    C : float
        Installed capacity.
    theta : float
        Tariff.
    A : np.ndarray
        Matrix of shape (T, N).
    params : dict
        Parameter dictionary.

    Returns
    -------
    tuple
        (h_star, eta_star, diagnostics)
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array of shape (T, N).")

    T, N = A.shape
    if T != params["T"] or N != params["N"]:
        raise ValueError(
            f"A has shape {(T, N)}, but params specify (T, N)=({params['T']}, {params['N']})."
        )

    h_star = np.zeros((T, N), dtype=float)
    eta_star = np.zeros(T, dtype=float)

    slot_diagnostics: list[dict[str, Any]] = []
    total_eta_iterations = 0
    total_slot_iterations = 0
    n_binding_slots = 0

    h_prev = np.zeros(N, dtype=float)

    for t in range(T):
        h_t, eta_t, info_t = solve_slot_equilibrium(
            C=C,
            theta=theta,
            A_t=A[t, :],
            params=params,
            h_init=h_prev,
        )

        h_star[t, :] = h_t
        eta_star[t] = eta_t
        slot_diagnostics.append(info_t)

        total_eta_iterations += int(info_t.get("eta_iterations", 0))
        total_slot_iterations += int(info_t.get("slot_solver_iterations", 0))
        n_binding_slots += int(bool(info_t.get("binding", False)))

        h_prev = h_t

    diagnostics = {
        "slot_diagnostics": slot_diagnostics,
        "total_eta_iterations": total_eta_iterations,
        "total_slot_iterations": total_slot_iterations,
        "n_binding_slots": n_binding_slots,
    }

    return h_star, eta_star, diagnostics