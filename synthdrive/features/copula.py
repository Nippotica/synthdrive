"""
Gaussian copula for SynthDrive v0.1.

The Gaussian copula captures the rank-correlation structure among continuous
feature variables while leaving each variable's marginal distribution free to
be specified independently.

Approach
--------
1. Convert Spearman rank correlations to Pearson correlations using the
   bivariate normal approximation:

       rho_Pearson = 2 * sin(pi * rho_Spearman / 6)

2. Verify positive definiteness; apply Higham (2002) correction if needed.
3. Sample from a zero-mean multivariate normal with the Pearson correlation
   matrix using the Cholesky decomposition.
4. Convert to uniform marginals via the standard normal CDF.
5. The uniform samples are passed to the feature sampler, which applies the
   variable-specific inverse CDFs.

References
----------
Iman, R. L., and Conover, W. J. (1982). A distribution-free approach to
    inducing rank correlation among input variables. Communications in
    Statistics—Simulation and Computation, 11(3), 311–334.

Higham, N. J. (2002). Computing the nearest correlation matrix.
    IMA Journal of Numerical Analysis, 22(3), 329–343.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Correlation matrix utilities
# ---------------------------------------------------------------------------


def spearman_to_pearson(rho_s: float) -> float:
    """
    Convert a Spearman rank correlation to its Pearson equivalent for a
    Gaussian copula, using the bivariate normal approximation.

    Formula: rho_P = 2 * sin(pi * rho_S / 6)
    """
    return 2.0 * np.sin(np.pi * rho_s / 6.0)


def spearman_matrix_to_pearson(rho_s: np.ndarray) -> np.ndarray:
    """
    Apply spearman_to_pearson element-wise to a correlation matrix,
    preserving the unit diagonal.
    """
    rho_p = 2.0 * np.sin(np.pi * rho_s / 6.0)
    np.fill_diagonal(rho_p, 1.0)
    return rho_p


def nearest_positive_definite(A: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Return the nearest positive definite matrix to A using eigenvalue flooring.

    The result has eigenvalues >= epsilon and is rescaled so the diagonal
    is all ones (i.e., returned as a valid correlation matrix).

    Reference: Higham (2002).
    """
    # Symmetrize
    B = (A + A.T) / 2.0

    # Eigendecomposition (eigh is stable for symmetric matrices)
    eigvals, eigvecs = eigh(B)

    # Floor eigenvalues at epsilon
    eigvals_clipped = np.maximum(eigvals, epsilon)

    # Reconstruct
    A_pd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

    # Rescale to correlation matrix (unit diagonal)
    d = np.sqrt(np.diag(A_pd))
    A_corr = A_pd / np.outer(d, d)
    return A_corr


def validate_correlation_matrix(R: np.ndarray, tol: float = 1e-8) -> bool:
    """Return True if R is a valid correlation matrix (symmetric, PD, unit diagonal)."""
    if not np.allclose(R, R.T, atol=tol):
        return False
    if not np.allclose(np.diag(R), 1.0, atol=tol):
        return False
    eigvals = np.linalg.eigvalsh(R)
    if eigvals.min() < -tol:
        return False
    return True


# ---------------------------------------------------------------------------
# Gaussian copula class
# ---------------------------------------------------------------------------


class GaussianCopula:
    """
    Gaussian copula for joint sampling of continuous variables.

    Parameters
    ----------
    spearman_matrix : array-like of shape (d, d)
        Spearman rank correlation matrix for the d variables.
    variable_names : tuple of str, optional
        Names of the variables, used for diagnostics only.

    Attributes
    ----------
    pearson_matrix_ : ndarray of shape (d, d)
        Pearson correlation matrix (after Spearman → Pearson conversion and
        any nearest-PD correction).
    cholesky_ : ndarray of shape (d, d)
        Lower Cholesky factor of pearson_matrix_.
    d_ : int
        Number of variables.
    """

    def __init__(
        self,
        spearman_matrix: np.ndarray,
        variable_names: Optional[Tuple[str, ...]] = None,
    ) -> None:
        spearman_matrix = np.asarray(spearman_matrix, dtype=float)
        if spearman_matrix.ndim != 2 or spearman_matrix.shape[0] != spearman_matrix.shape[1]:
            raise ValueError("spearman_matrix must be a square 2-D array.")

        self.d_ = spearman_matrix.shape[0]
        self.variable_names = variable_names

        # Convert Spearman to Pearson
        pearson = spearman_matrix_to_pearson(spearman_matrix)

        # Check and correct positive definiteness
        if not validate_correlation_matrix(pearson):
            warnings.warn(
                "Pearson correlation matrix (converted from Spearman) is not "
                "positive definite.  Applying nearest-PD correction.",
                UserWarning,
                stacklevel=2,
            )
            pearson = nearest_positive_definite(pearson)

        self.pearson_matrix_: np.ndarray = pearson

        # Cholesky factorization for sampling
        try:
            self.cholesky_: np.ndarray = np.linalg.cholesky(pearson)
        except np.linalg.LinAlgError as exc:
            # Apply stronger correction and retry
            pearson = nearest_positive_definite(pearson, epsilon=1e-6)
            self.pearson_matrix_ = pearson
            self.cholesky_ = np.linalg.cholesky(pearson)

    def sample_uniform(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Draw n samples from the copula and return as uniform marginals.

        Parameters
        ----------
        n : int
            Number of samples to draw.
        rng : numpy.random.Generator
            Random number generator (from numpy.random.default_rng).

        Returns
        -------
        U : ndarray of shape (n, d)
            Each column is a uniform sample on (0, 1) for one variable.
            The rank correlations among columns approximate the Spearman
            correlations specified at construction.
        """
        # Sample standard normals
        Z = rng.standard_normal((n, self.d_))

        # Introduce correlation via the Cholesky factor
        # Z_corr[i] = L @ z[i]
        Z_corr = Z @ self.cholesky_.T  # shape (n, d)

        # Convert to uniform via standard normal CDF; clip away extremes
        U = norm.cdf(Z_corr)
        U = np.clip(U, 1e-9, 1.0 - 1e-9)

        return U

    def __repr__(self) -> str:
        names = (
            f"variables={self.variable_names}" if self.variable_names else ""
        )
        return f"GaussianCopula(d={self.d_}, {names})"


# ---------------------------------------------------------------------------
# Convenience constructor from CopulaParams
# ---------------------------------------------------------------------------


def copula_from_params(params: object) -> GaussianCopula:
    """
    Build a GaussianCopula from a CopulaParams dataclass instance.

    Parameters
    ----------
    params : CopulaParams
        Must have attributes ``spearman_corr`` (nested tuple/list) and
        ``copula_vars`` (tuple of str).
    """
    spearman = np.array(params.spearman_corr, dtype=float)
    return GaussianCopula(
        spearman_matrix=spearman,
        variable_names=params.copula_vars,
    )


# ---------------------------------------------------------------------------
# Seed-mode copula fitting
# ---------------------------------------------------------------------------


def fit_copula_from_data(
    data: np.ndarray,
    variable_names: Optional[Tuple[str, ...]] = None,
) -> GaussianCopula:
    """
    Fit a Gaussian copula from observed continuous data.

    The fitting procedure:
    1. Compute empirical Spearman rank correlations.
    2. Convert to Pearson correlations.
    3. Construct and return a GaussianCopula.

    Parameters
    ----------
    data : ndarray of shape (n_obs, d)
        Observed values.  Each column is one variable.
    variable_names : tuple of str, optional
        Names of columns.

    Returns
    -------
    GaussianCopula
    """
    from scipy.stats import spearmanr

    n_obs, d = data.shape
    if n_obs < 30:
        warnings.warn(
            f"Fitting Gaussian copula from only {n_obs} observations.  "
            "Correlation estimates will be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # scipy.stats.spearmanr returns a correlation matrix when given a 2-D array
    corr_result = spearmanr(data)
    if d == 1:
        spearman = np.array([[1.0]])
    elif d == 2:
        spearman = np.array([[1.0, corr_result.statistic],
                             [corr_result.statistic, 1.0]])
    else:
        spearman = np.array(corr_result.statistic)

    # Ensure symmetry and unit diagonal
    spearman = (spearman + spearman.T) / 2.0
    np.fill_diagonal(spearman, 1.0)

    return GaussianCopula(spearman_matrix=spearman, variable_names=variable_names)
