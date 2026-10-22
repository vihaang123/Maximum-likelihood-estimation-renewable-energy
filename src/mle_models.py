"""Maximum Likelihood Estimation of candidate probability distributions.

For an i.i.d. sample x_1..x_n from a density f(x; theta) the

    likelihood      L(theta)  = prod_i f(x_i; theta)
    log-likelihood  l(theta)  = sum_i log f(x_i; theta)
    MLE             theta_hat = argmax_theta l(theta)

Each distribution below implements its own log-density from the closed-form formula (documented in the
class docstring).  The maximisation is done numerically with BFGS on an UNCONSTRAINED re-parametrisation
(log for positive parameters, logit for a mixing weight), which enforces the parameter constraints
automatically.  After optimisation we check convergence (gradient ~ 0), that the observed information matrix is
positive definite (a proper maximum) and we derive standard errors from the inverse observed information.
SciPy is used only to (i) evaluate CDFs / quantiles / random draws and (ii) independently VALIDATE our results.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, special, stats


# --------------------------------------------------------------------------- #
# Numerical helpers
# --------------------------------------------------------------------------- #
def numerical_gradient(func, x: np.ndarray, step: float = 1e-6) -> np.ndarray:
    """Central-difference gradient."""
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        e = np.zeros_like(x, dtype=float)
        e[i] = step
        grad[i] = (func(x + e) - func(x - e)) / (2 * step)
    return grad


def numerical_hessian(func, x: np.ndarray, rel_step: float = 1e-3) -> np.ndarray:
    """Central-difference Hessian with a step proportional to each coordinate's magnitude."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    h = rel_step * np.maximum(np.abs(x), 1e-2)
    hess = np.zeros((n, n))
    f0 = func(x)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h[i]
        hess[i, i] = (func(x + ei) - 2 * f0 + func(x - ei)) / h[i] ** 2
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = h[j]
            value = (func(x + ei + ej) - func(x + ei - ej) - func(x - ei + ej) + func(x - ei - ej)) / (4 * h[i] * h[j])
            hess[i, j] = hess[j, i] = value
    return hess


@dataclass
class MLEFit:
    """Container for one fitted model."""
    name: str
    param_names: tuple
    params: dict
    loglik: float
    k: int
    n: int
    converged: bool
    optimizer_success: bool
    n_iter: int
    max_abs_gradient_per_obs: float
    information_pd: bool
    std_errors: dict = field(default_factory=dict)
    message: str = ""

    def as_row(self) -> dict:
        row = {"model": self.name, "n": self.n, "k": self.k, "loglik": self.loglik,
               "converged": self.converged, "optimizer_success": self.optimizer_success,
               "iterations": self.n_iter, "max_abs_grad_per_obs": self.max_abs_gradient_per_obs,
               "information_matrix_pd": self.information_pd}
        return row


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class Distribution:
    """Common MLE machinery.  Subclasses define logpdf, the parameter transforms and start values."""

    name = "base"
    param_names: tuple = ()
    positive_support = False          # True if the density is only defined for x > 0

    # -- to be provided by subclasses ------------------------------------------------------------
    def logpdf(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def to_natural(self, theta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def to_unconstrained(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def start_candidates(self, x: np.ndarray) -> list:
        raise NotImplementedError

    def frozen(self, p: np.ndarray):
        """scipy frozen distribution with the same parametrisation (for CDF / ppf / rvs only)."""
        raise NotImplementedError

    # -- shared ----------------------------------------------------------------------------------
    def check_support(self, x: np.ndarray) -> None:
        if not np.all(np.isfinite(x)):
            raise ValueError(f"{self.name}: data contain non-finite values.")
        if self.positive_support and np.any(x <= 0):
            raise ValueError(f"{self.name} requires strictly positive data (found {np.sum(x <= 0)} values <= 0).")

    @property
    def k(self) -> int:
        return len(self.param_names)

    def loglik(self, x: np.ndarray, p: np.ndarray) -> float:
        value = float(np.sum(self.logpdf(x, p)))
        return value if np.isfinite(value) else -1e300

    def cdf(self, x, p):
        return self.frozen(p).cdf(x)

    def ppf(self, q, p):
        return self.frozen(p).ppf(q)

    def rvs(self, n: int, p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return self.frozen(p).rvs(size=n, random_state=rng)

    def implied_mean_sd(self, p: np.ndarray) -> tuple:
        dist = self.frozen(p)
        return float(dist.mean()), float(dist.std())

    def fit(self, x: np.ndarray, start: np.ndarray | None = None) -> MLEFit:
        """Numerical MLE (multi-start BFGS on the unconstrained scale) with convergence diagnostics."""
        x = np.asarray(x, dtype=float)
        self.check_support(x)
        n = len(x)
        starts = [np.asarray(start, dtype=float)] if start is not None else self.start_candidates(x)

        def objective(theta):                       # mean negative log-likelihood (well scaled for BFGS)
            return -self.loglik(x, self.to_natural(theta)) / n

        best, best_res = None, None
        for s in starts:
            theta0 = self.to_unconstrained(s)
            try:
                res = optimize.minimize(objective, theta0, method="BFGS", options={"gtol": 1e-9, "maxiter": 5000})
            except (FloatingPointError, ValueError):
                continue
            if not np.isfinite(res.fun):
                continue
            # polish with Nelder-Mead if BFGS reports trouble, keep whichever is better
            if not res.success:
                alt = optimize.minimize(objective, res.x, method="Nelder-Mead",
                                        options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 20000})
                if alt.fun < res.fun:
                    res = alt
            if best_res is None or res.fun < best_res.fun:
                best_res, best = res, res.x
        if best is None:
            raise RuntimeError(f"{self.name}: optimisation failed from every starting value.")

        p_hat = self.to_natural(best)
        loglik = self.loglik(x, p_hat)

        grad = numerical_gradient(objective, best)
        max_grad = float(np.max(np.abs(grad)))
        # observed information in NATURAL parameters: -d^2 l / d p^2
        neg_ll = lambda p: -self.loglik(x, p)
        info = numerical_hessian(neg_ll, p_hat)
        try:
            eigvals = np.linalg.eigvalsh(0.5 * (info + info.T))
            pd_ok = bool(np.all(eigvals > 0))
            cov = np.linalg.inv(info) if pd_ok else None
        except np.linalg.LinAlgError:
            pd_ok, cov = False, None
        se = {}
        if cov is not None and np.all(np.diag(cov) > 0):
            se = {name: float(np.sqrt(v)) for name, v in zip(self.param_names, np.diag(cov))}

        converged = bool(np.isfinite(loglik) and max_grad < 1e-5 and pd_ok)
        return MLEFit(
            name=self.name, param_names=self.param_names,
            params={name: float(v) for name, v in zip(self.param_names, p_hat)},
            loglik=loglik, k=self.k, n=n, converged=converged,
            optimizer_success=bool(getattr(best_res, "success", False)),
            n_iter=int(getattr(best_res, "nit", -1)), max_abs_gradient_per_obs=max_grad,
            information_pd=pd_ok, std_errors=se, message=str(getattr(best_res, "message", "")),
        )

    def params_array(self, fit: MLEFit) -> np.ndarray:
        return np.array([fit.params[name] for name in self.param_names], dtype=float)


# --------------------------------------------------------------------------- #
# Candidate distributions
# --------------------------------------------------------------------------- #
class Normal(Distribution):
    """f(x) = 1/sqrt(2 pi s^2) exp(-(x-m)^2 / (2 s^2)).   MLE: m = mean(x), s^2 = mean((x-m)^2)."""
    name = "Normal"
    param_names = ("mu", "sigma")

    def logpdf(self, x, p):
        mu, sigma = p
        return -0.5 * np.log(2 * np.pi * sigma ** 2) - (x - mu) ** 2 / (2 * sigma ** 2)

    def to_natural(self, theta):
        return np.array([theta[0], np.exp(theta[1])])

    def to_unconstrained(self, p):
        return np.array([p[0], np.log(p[1])])

    def start_candidates(self, x):
        return [np.array([np.median(x), 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-6])]

    def frozen(self, p):
        return stats.norm(loc=p[0], scale=p[1])


class StudentT(Distribution):
    """Location-scale Student t:
    log f(x) = lnG((v+1)/2) - lnG(v/2) - 0.5 ln(v pi) - ln s - (v+1)/2 ln(1 + ((x-m)/s)^2 / v)."""
    name = "Student-t"
    param_names = ("mu", "sigma", "nu")

    def logpdf(self, x, p):
        mu, sigma, nu = p
        z = (x - mu) / sigma
        return (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(nu * np.pi)
                - np.log(sigma) - (nu + 1) / 2 * np.log1p(z ** 2 / nu))

    def to_natural(self, theta):
        return np.array([theta[0], np.exp(theta[1]), np.exp(theta[2])])

    def to_unconstrained(self, p):
        return np.array([p[0], np.log(p[1]), np.log(p[2])])

    def start_candidates(self, x):
        med = np.median(x)
        scale = 1.4826 * np.median(np.abs(x - med)) + 1e-6
        return [np.array([med, scale, nu]) for nu in (3.0, 8.0, 30.0)]

    def frozen(self, p):
        return stats.t(df=p[2], loc=p[0], scale=p[1])

    def implied_mean_sd(self, p):
        mu, sigma, nu = p
        sd = sigma * np.sqrt(nu / (nu - 2)) if nu > 2 else float("inf")
        return float(mu if nu > 1 else float("nan")), float(sd)


class LogNormal(Distribution):
    """ln X ~ N(m, s^2):  log f(x) = -ln x - 0.5 ln(2 pi s^2) - (ln x - m)^2 / (2 s^2).
    MLE: m = mean(ln x), s^2 = mean((ln x - m)^2)."""
    name = "Log-normal"
    param_names = ("mu_log", "sigma_log")
    positive_support = True

    def logpdf(self, x, p):
        m, s = p
        lx = np.log(x)
        return -lx - 0.5 * np.log(2 * np.pi * s ** 2) - (lx - m) ** 2 / (2 * s ** 2)

    def to_natural(self, theta):
        return np.array([theta[0], np.exp(theta[1])])

    def to_unconstrained(self, p):
        return np.array([p[0], np.log(p[1])])

    def start_candidates(self, x):
        lx = np.log(x)
        return [np.array([lx.mean(), lx.std() + 1e-6])]

    def frozen(self, p):
        return stats.lognorm(s=p[1], scale=np.exp(p[0]))


class Gamma(Distribution):
    """shape a, scale t:  log f(x) = -ln G(a) - a ln t + (a-1) ln x - x/t.
    Profile condition: t = mean(x)/a and  ln a - psi(a) = ln mean(x) - mean(ln x)."""
    name = "Gamma"
    param_names = ("shape", "scale")
    positive_support = True

    def logpdf(self, x, p):
        a, t = p
        return -special.gammaln(a) - a * np.log(t) + (a - 1) * np.log(x) - x / t

    def to_natural(self, theta):
        return np.exp(theta)

    def to_unconstrained(self, p):
        return np.log(p)

    def start_candidates(self, x):
        m, v = x.mean(), x.var()
        return [np.array([m ** 2 / v, v / m])]

    def frozen(self, p):
        return stats.gamma(a=p[0], scale=p[1])


class Weibull(Distribution):
    """shape k, scale l:  log f(x) = ln k - k ln l + (k-1) ln x - (x/l)^k."""
    name = "Weibull"
    param_names = ("shape", "scale")
    positive_support = True

    def logpdf(self, x, p):
        k, lam = p
        return np.log(k) - k * np.log(lam) + (k - 1) * np.log(x) - (x / lam) ** k

    def to_natural(self, theta):
        return np.exp(theta)

    def to_unconstrained(self, p):
        return np.log(p)

    def start_candidates(self, x):
        lx = np.log(x)
        k0 = 1.2 / (lx.std() + 1e-9)               # moment-type approximation: sd(ln x) ~ 1.28 / k
        return [np.array([k0, np.exp(lx.mean() + 0.5772 / k0)])]

    def frozen(self, p):
        return stats.weibull_min(c=p[0], scale=p[1])


class Beta(Distribution):
    """Beta on y = x / c with a FIXED known upper bound c (c is a constant, not estimated):
       log f_Y(y) = -ln B(a,b) + (a-1) ln y + (b-1) ln(1-y)
       log f_X(x) = log f_Y(x/c) - ln c       (Jacobian term, so log-likelihoods are comparable with the
                                               other candidates, which are densities of x itself)."""
    name = "Beta (scaled)"
    param_names = ("alpha", "beta")
    positive_support = True

    def __init__(self, upper_bound: float):
        if not upper_bound > 0:
            raise ValueError("upper_bound must be positive.")
        self.c = float(upper_bound)

    def check_support(self, x):
        super().check_support(x)
        if np.any(x >= self.c):
            raise ValueError(f"{self.name}: values must lie below the fixed upper bound c={self.c}.")

    def logpdf(self, x, p):
        a, b = p
        y = x / self.c
        return -special.betaln(a, b) + (a - 1) * np.log(y) + (b - 1) * np.log1p(-y) - np.log(self.c)

    def to_natural(self, theta):
        return np.exp(theta)

    def to_unconstrained(self, p):
        return np.log(p)

    def start_candidates(self, x):
        y = x / self.c
        m, v = y.mean(), y.var()
        common = m * (1 - m) / v - 1
        return [np.array([max(m * common, 0.1), max((1 - m) * common, 0.1)])]

    def frozen(self, p):
        return stats.beta(a=p[0], b=p[1], scale=self.c)


class GaussianMixture2(Distribution):
    """Two-component normal mixture:  f(x) = w N(x; m1, s1^2) + (1-w) N(x; m2, s2^2),  params (w, m1, s1, m2, s2).

    Fitted by the EM algorithm (Dempster, Laird & Rubin, 1977) from several starting points, then polished by
    BFGS on the exact log-likelihood.  Components are ordered m1 < m2 to remove label switching.
    """
    name = "2-component normal mixture"
    param_names = ("weight1", "mu1", "sigma1", "mu2", "sigma2")

    def logpdf(self, x, p):
        w, m1, s1, m2, s2 = p
        a = np.log(w) + stats.norm.logpdf(x, m1, s1)
        b = np.log1p(-w) + stats.norm.logpdf(x, m2, s2)
        return np.logaddexp(a, b)

    def to_natural(self, theta):
        w = special.expit(theta[0])
        w = min(max(w, 1e-9), 1 - 1e-9)
        return np.array([w, theta[1], np.exp(theta[2]), theta[3], np.exp(theta[4])])

    def to_unconstrained(self, p):
        w = min(max(p[0], 1e-9), 1 - 1e-9)
        return np.array([special.logit(w), p[1], np.log(p[2]), p[3], np.log(p[4])])

    # -- EM -------------------------------------------------------------------------------------
    @staticmethod
    def _em(x, p0, max_iter=2000, tol=1e-11):
        w, m1, s1, m2, s2 = p0
        previous = -np.inf
        for _ in range(max_iter):
            a = np.log(w) + stats.norm.logpdf(x, m1, s1)           # E-step
            b = np.log1p(-w) + stats.norm.logpdf(x, m2, s2)
            total = np.logaddexp(a, b)
            r1 = np.exp(a - total)
            r2 = 1 - r1
            w = np.clip(r1.mean(), 1e-6, 1 - 1e-6)                   # M-step
            m1 = np.sum(r1 * x) / np.sum(r1)
            m2 = np.sum(r2 * x) / np.sum(r2)
            s1 = max(np.sqrt(np.sum(r1 * (x - m1) ** 2) / np.sum(r1)), 1e-4)
            s2 = max(np.sqrt(np.sum(r2 * (x - m2) ** 2) / np.sum(r2)), 1e-4)
            current = total.sum()
            if abs(current - previous) < tol * max(1.0, abs(current)):
                break
            previous = current
        return np.array([w, m1, s1, m2, s2])

    def start_candidates(self, x):
        q25, q50, q75 = np.quantile(x, [0.25, 0.5, 0.75])
        sd = x.std()
        seeds = [np.array([0.3, q25 - 0.5 * sd, sd / 2, q75, sd / 3]),
                 np.array([0.5, x.mean() - sd, sd / 2, x.mean() + 0.5 * sd, sd / 2]),
                 np.array([0.2, np.quantile(x, 0.1), sd / 2, q75, sd / 3])]
        return [self._em(x, s) for s in seeds]

    def fit(self, x, start=None):
        fitted = super().fit(x, start)
        p = self.params_array(fitted)
        if p[1] > p[3]:                                  # order the components (label switching)
            p = np.array([1 - p[0], p[3], p[4], p[1], p[2]])
            fitted.params = {n: float(v) for n, v in zip(self.param_names, p)}
        return fitted

    def frozen(self, p):
        return _FrozenMixture(p)

    def implied_mean_sd(self, p):
        w, m1, s1, m2, s2 = p
        mean = w * m1 + (1 - w) * m2
        var = w * (s1 ** 2 + m1 ** 2) + (1 - w) * (s2 ** 2 + m2 ** 2) - mean ** 2
        return float(mean), float(np.sqrt(var))


class _FrozenMixture:
    """Minimal frozen-distribution wrapper (cdf, ppf, rvs) for the two-component mixture."""

    def __init__(self, p):
        self.w, self.m1, self.s1, self.m2, self.s2 = p

    def cdf(self, x):
        return self.w * stats.norm.cdf(x, self.m1, self.s1) + (1 - self.w) * stats.norm.cdf(x, self.m2, self.s2)

    def ppf(self, q):
        q = np.atleast_1d(q)
        lo = min(self.m1 - 12 * self.s1, self.m2 - 12 * self.s2)
        hi = max(self.m1 + 12 * self.s1, self.m2 + 12 * self.s2)
        return np.array([optimize.brentq(lambda v, qq=qq: self.cdf(v) - qq, lo, hi, xtol=1e-12) for qq in q])

    def rvs(self, size, random_state):
        rng = random_state
        first = rng.random(size) < self.w
        return np.where(first, rng.normal(self.m1, self.s1, size), rng.normal(self.m2, self.s2, size))


# --------------------------------------------------------------------------- #
# Fitting several candidates on the SAME observations
# --------------------------------------------------------------------------- #
def build_candidates(x: np.ndarray, include_mixture: bool = True) -> list:
    """Candidates suited to the support of x.  The scaled Beta uses c = 1.05 * max(x): a fixed constant slightly
    above the largest observation (the clear-sky index is bounded above by physics, up to envelope error)."""
    candidates = [Normal(), StudentT()]
    if np.all(x > 0):
        candidates += [LogNormal(), Gamma(), Weibull(), Beta(upper_bound=1.05 * float(np.max(x)))]
    if include_mixture:
        candidates.append(GaussianMixture2())
    return candidates


def fit_all(x: np.ndarray, candidates: list) -> dict:
    """Return {name: (distribution, MLEFit)} for every candidate; failures are recorded, not hidden."""
    results = {}
    for dist in candidates:
        try:
            results[dist.name] = (dist, dist.fit(x))
        except (ValueError, RuntimeError) as exc:
            print(f"  [warning] {dist.name} could not be fitted: {exc}")
    return results


# --------------------------------------------------------------------------- #
# Independent validation against SciPy's own fitters
# --------------------------------------------------------------------------- #
def validate_against_scipy(x: np.ndarray, dist: Distribution, fit: MLEFit) -> dict:
    """Compare our MLE with scipy.stats.<dist>.fit (and, where it exists, the closed-form solution)."""
    n = len(x)
    ours = dist.params_array(fit)
    out = {"model": dist.name, "our_loglik": fit.loglik}

    if isinstance(dist, Normal):
        loc, scale = stats.norm.fit(x)
        ref_params = np.array([loc, scale])
        ref_ll = float(np.sum(stats.norm.logpdf(x, loc, scale)))
        closed = np.array([x.mean(), x.std()])
        out["max_abs_diff_vs_closed_form"] = float(np.max(np.abs(ours - closed)))
    elif isinstance(dist, LogNormal):
        s, loc, scale = stats.lognorm.fit(x, floc=0)
        ref_params = np.array([np.log(scale), s])
        ref_ll = float(np.sum(stats.lognorm.logpdf(x, s, loc, scale)))
        closed = np.array([np.log(x).mean(), np.log(x).std()])
        out["max_abs_diff_vs_closed_form"] = float(np.max(np.abs(ours - closed)))
    elif isinstance(dist, Gamma):
        a, loc, scale = stats.gamma.fit(x, floc=0)
        ref_params = np.array([a, scale])
        ref_ll = float(np.sum(stats.gamma.logpdf(x, a, loc, scale)))
        # profile-likelihood identity: ln a - psi(a) = ln mean(x) - mean(ln x)
        out["profile_equation_residual"] = float(np.log(ours[0]) - special.digamma(ours[0])
                                                  - (np.log(x.mean()) - np.log(x).mean()))
        out["scale_equals_mean_over_shape"] = float(abs(ours[1] - x.mean() / ours[0]))
    elif isinstance(dist, Weibull):
        c, loc, scale = stats.weibull_min.fit(x, floc=0)
        ref_params = np.array([c, scale])
        ref_ll = float(np.sum(stats.weibull_min.logpdf(x, c, loc, scale)))
    elif isinstance(dist, StudentT):
        df, loc, scale = stats.t.fit(x)
        ref_params = np.array([loc, scale, df])
        ref_ll = float(np.sum(stats.t.logpdf(x, df, loc, scale)))
    elif isinstance(dist, Beta):
        y = x / dist.c
        a, b, loc, scale = stats.beta.fit(y, floc=0, fscale=1)
        ref_params = np.array([a, b])
        ref_ll = float(np.sum(stats.beta.logpdf(y, a, b)) - n * np.log(dist.c))
    elif isinstance(dist, GaussianMixture2):
        w, m1, s1, m2, s2 = ours
        comp = np.logaddexp(np.log(w) + stats.norm.logpdf(x, m1, s1), np.log1p(-w) + stats.norm.logpdf(x, m2, s2))
        ref_params, ref_ll = ours, float(np.sum(comp))     # independent re-computation of the same likelihood
    else:
        raise ValueError("unknown distribution")

    out["scipy_loglik"] = ref_ll
    out["loglik_difference_ours_minus_scipy"] = fit.loglik - ref_ll
    out["max_abs_param_difference"] = float(np.max(np.abs(ours - ref_params)))
    out["max_rel_param_difference"] = float(np.max(np.abs(ours - ref_params) / np.maximum(np.abs(ref_params), 1e-12)))
    # "valid" means our optimiser reached a likelihood at least as high as the reference (within 1e-3)
    out["validated"] = bool(fit.loglik >= ref_ll - 1e-3)
    return out
