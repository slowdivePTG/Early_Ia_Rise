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
    x_mean, x_err, y_mean, y_err, xb_mu=0.0, xb_sigma=0.5
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
    xb = numpyro.sample("xb", dist.TruncatedNormal(xb_mu, xb_sigma, low=-1, high=1))

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


def get_chains(x_samples=None, y_samples=None, x_mean=None, x_err=None, **kwargs):
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
    print("Fitting Hierarchical Linear Model...")
    mcmc_lin = MCMC(
        NUTS(hierarchical_linear_model, init_strategy=init_to_median),
        num_warmup=2000,
        num_samples=1000,
        num_chains=4,
        chain_method="parallel",
        progress_bar=False,
    )
    mcmc_lin.run(
        jax.random.PRNGKey(0), x_mean=x_mean, x_err=x_err, y_mean=y_mean, y_err=y_err
    )
    idata_lin = az.from_numpyro(mcmc_lin)

    # 3. Run MCMC for Broken Linear Model
    print("Fitting Hierarchical Broken Linear Model...")
    mcmc_brok = MCMC(
        NUTS(hierarchical_broken_linear_model, init_strategy=init_to_median),
        num_warmup=2000,
        num_samples=1000,
        num_chains=4,
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
