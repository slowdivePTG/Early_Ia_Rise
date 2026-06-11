import os

# Tell JAX to treat your CPU as 4 separate devices for parallel chains.
# This MUST happen before importing jax!
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"


import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az
import numpy as np

from astropy.table import Table
from pathlib import Path

from snia_rise.ztf_lc import ZTFLib, SampleConfig


def add_eti_range(summary_df: az.summary):
    up = summary_df["eti_84%"] - summary_df["median"]
    lo = summary_df["median"] - summary_df["eti_16%"]

    summary_df.insert(1, "eti_lo", lo)
    summary_df.insert(2, "eti_hi", up)

    return summary_df


def prepare_data(x_samples):
    """
    Collapses multi-D sample arrays down to means and standard deviations
    for the N objects. Assumes the last dimension is the object index.
    """
    # Flatten all sample dimensions (e.g., chains and draws) into one
    # Shape becomes (total_samples, N_objects)
    x_flat = np.reshape(x_samples, (-1, x_samples.shape[-1]))

    x_mean, x_err = jnp.mean(x_flat, axis=0), jnp.std(x_flat, axis=0)

    return x_mean, x_err


def hierarchical_linear_model(x_mean, x_err, y_mean, y_err):
    N = len(x_mean)

    # --- Population Priors for Latent X ---
    # This regularizes the true X values based on the overall data distribution
    mu_x_pop = numpyro.sample(
        "mu_x_pop", dist.Normal(jnp.mean(x_mean), jnp.std(x_mean))
    )
    sig_x_pop = numpyro.sample("sig_x_pop", dist.HalfNormal(jnp.std(x_mean)))

    # Latent true X values for each object
    x_true = numpyro.sample(
        "x_true", dist.Normal(mu_x_pop, sig_x_pop), sample_shape=(N,)
    )

    # --- Physical Model Parameters (User Priors) ---
    beta0 = numpyro.sample("beta0", dist.Uniform(15, 25))
    beta1 = numpyro.sample("beta1", dist.Uniform(-5, 5))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1))

    # --- The True Physical Relation ---
    mu_y = beta0 + beta1 * x_true

    # Robustness parameter (degrees of freedom for Student-T)
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    # Latent true Y values with robust Student-T and broken intrinsic scatter
    y_true = numpyro.sample("y_true", dist.StudentT(df=nu, loc=mu_y, scale=sigma))

    # y_true = numpyro.sample("y_true", dist.Normal(loc=mu_y, scale=sigma))

    # --- Measurement Error Model (Likelihood) ---
    numpyro.sample("obs_x", dist.Normal(x_true, x_err), obs=x_mean)
    numpyro.sample("obs_y", dist.Normal(y_true, y_err), obs=y_mean)


def hierarchical_broken_linear_model(
    x_mean, x_err, y_mean, y_err, xb_mu=None, xb_sigma=None, xb_min=0, xb_max=1
):
    N = len(x_mean)

    # --- Population Priors for Latent X ---
    mu_x_pop = numpyro.sample(
        "mu_x_pop", dist.Normal(jnp.mean(x_mean), jnp.std(x_mean))
    )
    sig_x_pop = numpyro.sample("sig_x_pop", dist.HalfNormal(jnp.std(x_mean)))
    x_true = numpyro.sample(
        "x_true", dist.Normal(mu_x_pop, sig_x_pop), sample_shape=(N,)
    )

    # --- Physical Model Parameters (Mean Relation) ---
    beta0 = numpyro.sample(
        "beta0",
        dist.Uniform(y_mean.min() - 5 * y_err.max(), y_mean.max() + 5 * y_err.max()),
    )
    beta1 = numpyro.sample("beta1", dist.Uniform(-5, 5))
    beta2 = numpyro.sample("beta2", dist.Uniform(-5, 5))
    if xb_mu is None or xb_sigma is None:
        xb = numpyro.sample("xb", dist.Uniform(xb_min, xb_max))
    else:
        xb = numpyro.sample("xb", dist.TruncatedNormal(xb_mu, xb_sigma, low=xb_min, high=xb_max))

    # --- Physical Model Parameters (Scatter) ---
    sigma = numpyro.sample("sigma", dist.HalfNormal(1))

    # --- The True Physical Relation (Broken Mean) ---
    mu_y = jnp.where(
        x_true < xb, beta0 + beta1 * x_true, beta0 + beta1 * xb + beta2 * (x_true - xb)
    )

    # Robustness parameter (degrees of freedom for Student-T)
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    # Latent true Y values with robust Student-T and broken intrinsic scatter
    y_true = numpyro.sample("y_true", dist.StudentT(df=nu, loc=mu_y, scale=sigma))

    # y_true = numpyro.sample("y_true", dist.Normal(loc=mu_y, scale=sigma))

    # --- Measurement Error Model (Likelihood) ---
    numpyro.sample("obs_x", dist.Normal(x_true, x_err), obs=x_mean)
    numpyro.sample("obs_y", dist.Normal(y_true, y_err), obs=y_mean)


def get_chains(
    x_samples=None,
    y_samples=None,
    x_mean=None,
    x_err=None,
    models=["lin", "brok"],
    **kwargs,
):
    """
    Utility function to extract and prepare MCMC chains for both models.
    """

    from numpyro.infer import init_to_median

    # 1. Summarize the multi-D posteriors down to object-level means and errors
    if x_mean is None and x_err is None:
        assert x_samples is not None, (
            "x_samples must be provided if x_mu and x_sigma are not given"
        )
        x_mean, x_err = prepare_data(x_samples)
    y_mean, y_err = prepare_data(y_samples)

    # 2. Run MCMC for Linear Model
    if "lin" in models:
        print("Fitting Hierarchical Linear Model...")
        mcmc_lin = MCMC(
            NUTS(hierarchical_linear_model, init_strategy=init_to_median),
            num_warmup=2000,
            num_samples=2000,
            num_chains=4,
            thinning=2,
            chain_method="parallel",
            progress_bar=False,
        )
        mcmc_lin.run(
            jax.random.PRNGKey(0),
            x_mean=x_mean,
            x_err=x_err,
            y_mean=y_mean,
            y_err=y_err,
        )
        idata_lin = az.from_numpyro(mcmc_lin)
    else:
        idata_lin = None

    if "brok" in models:
        # 3. Run MCMC for Broken Linear Model
        print("Fitting Hierarchical Broken Linear Model...")
        mcmc_brok = MCMC(
            NUTS(hierarchical_broken_linear_model, init_strategy=init_to_median),
            num_warmup=2000,
            num_samples=2000,
            num_chains=4,
            thinning=2,
            chain_method="parallel",
            progress_bar=False,
        )
        mcmc_brok.run(
            jax.random.PRNGKey(1),
            x_mean=x_mean,
            x_err=x_err,
            y_mean=y_mean,
            y_err=y_err,
            **kwargs,
        )
        idata_brok = az.from_numpyro(mcmc_brok)
        idata_brok.posterior["beta1_2"] = (
            idata_brok.posterior["beta1"] - idata_brok.posterior["beta2"]
        )
    else:
        idata_brok = None

    return idata_lin, idata_brok


def run_comparison(idata_lin, idata_brok):
    """
    Executes the hierarchical MCMC fits and LOO-CV model comparison.
    """

    # 4. Compare Models
    print("Computing LOO-CV...")
    comp_df = az.compare(
        {"Linear": idata_lin, "Broken_Linear": idata_brok}, ic="loo", var_name="obs_y"
    )

    print("\n--- Model Comparison Results ---")
    print(comp_df[["rank", "elpd_loo", "p_loo", "elpd_diff", "weight", "warning"]])

    best_model = comp_df.index[0]

    if best_model == "Broken_Linear":
        # Safely extract the posterior for the breakpoint
        xb_posterior = idata_brok.posterior["xb"].values.flatten()
        return comp_df, xb_posterior
    else:
        return comp_df, None


####### Posterior predictive checks and plotting functions #######
def _flatten_posterior(idata):
    """
    Convert (chain, draw) dimensions into a single sample dimension.
    """
    posterior = {}

    for var in idata.posterior.data_vars:
        posterior[var] = idata.posterior[var].stack(sample=("chain", "draw")).values

    return posterior


def _evaluate_linear(samples, x_grid):

    beta0 = samples["beta0"][:, None]
    beta1 = samples["beta1"][:, None]

    return beta0 + beta1 * x_grid[None, :]


def _evaluate_broken(samples, x_grid):

    beta0 = samples["beta0"][:, None]
    beta1 = samples["beta1"][:, None]
    beta2 = samples["beta2"][:, None]
    xb = samples["xb"][:, None]

    left = beta0 + beta1 * x_grid[None, :]
    right = beta0 + beta1 * xb + beta2 * (x_grid[None, :] - xb)

    return np.where(x_grid[None, :] < xb, left, right)


def posterior_predictive(
    idata,
    model="lin",
    x_grid=None,
    include_intrinsic_scatter=True,
):
    """
    Returns posterior predictive draws evaluated on x_grid.

    Shape:
        (n_draws, len(x_grid))
    """

    samples = _flatten_posterior(idata)

    if model == "lin":
        mu = _evaluate_linear(samples, x_grid)

    elif model == "brok":
        mu = _evaluate_broken(samples, x_grid)

    else:
        raise ValueError(model)

    if include_intrinsic_scatter:
        sigma = samples["sigma"][:, None]
        rng = np.random.default_rng(1234)
        y = mu + rng.normal(scale=sigma, size=mu.shape)

        return y

    return mu


def plot_posterior_predictive(
    idata,
    model="lin",
    include_intrinsic_scatter=True,
    ax=None,
    color="C0",
    x_grid=None,
):
    """
    Plot posterior predictive relation.

    Parameters
    ----------
    idata : arviz.InferenceData

    model : {"lin", "brok"}

    include_intrinsic_scatter : bool
        If True, band includes sigma_int.

    ax : matplotlib axis

    color : str

    x_grid : array-like
        Grid on which predictions are evaluated.
    """

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()

    samples = _flatten_posterior(idata)

    if x_grid is None:
        if model == "brok":
            x0 = np.median(samples["xb"])
            x_grid = np.linspace(x0 - 3, x0 + 3, 500)
        else:
            x_grid = np.linspace(-3, 3, 500)

    # --------------------------------------------------
    # posterior predictive draws
    # --------------------------------------------------

    draws = posterior_predictive(
        idata,
        model=model,
        x_grid=x_grid,
        include_intrinsic_scatter=include_intrinsic_scatter,
    )

    n_draws = draws.shape[0]
    rng = np.random.default_rng(42)
    idx = rng.choice(n_draws, size=min(50, n_draws), replace=False)

    # --------------------------------------------------
    # median-parameter model
    # --------------------------------------------------

    median_samples = {k: np.array([np.median(v)]) for k, v in samples.items()}

    if model == "lin":
        model_line = _evaluate_linear(median_samples, x_grid)[0]

    else:
        model_line = _evaluate_broken(median_samples, x_grid)[0]

    # --------------------------------------------------
    # plotting
    # --------------------------------------------------

    for i in idx:
        ax.plot(x_grid, draws[i], "-", color=color, lw=0.75, alpha=0.2, zorder=9)
    ax.plot(x_grid, model_line, "--", color=color, lw=3.0, zorder=11)

    return ax
