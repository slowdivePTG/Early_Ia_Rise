import numpy as np
import numpyro
import jax
import jax.numpy as jnp
from numpyro import distributions as dist

def f_t(t, tfl, C, A, alpha):
    """
    Calculate the flux with a power-law rise model.

    Parameters:
    -----------
    t : float
        Time value.
    tfl : float
        Time of the first light.
    C : float
        Baseline flux.
    A : float
        Proportionality factor.
    alpha : float
        Rising power-law index.

    Returns:
    --------
    float
        The calculated value of f(t).
    """
    det = jnp.heaviside(t - tfl, 0)
    return C + A * (det * (t - tfl)) ** alpha

def single_model(t, flux_err, n_fcqfid, idx_fcqfid, n_filt, idx_filt, flux=None):
    """
    Function to model the single light curve of a supernova
    in multiple fields, CCDs, quadrants, as well as filters.

    Each measurement has a unique fcqf ID defined in Yao et al. (2019)
        (fcqf ID) = (field ID) * 10000 + (CCD ID) * 100
                  + (quadrant ID) * 10 + (filter ID)

    Parameters
    ----------
    t : array-like
        Time values (phase) of the light curve.
        Phase = (t_obs - t_max) / (1 + z)
    flux : array-like
        Flux values of the light curve.
    flux_err : array-like
        Flux error values of the light curve.
    n_fcqfid, idx_fcqfid : int
        Number of unique fcqf IDs and indices used to index the fcqf IDs
        for each measurement.
    n_filt, idx_filt : int
        Number of unique filters and their indices.

    Returns
    -------
    None
    """

    # t_fl : Time of the first light
    tfl = numpyro.sample("t_fl", dist.Uniform(-50, 0))

    # Parameters specific to each fcqf ID (n_fcqfid)
    # C : Baseline flux
    C = numpyro.sample(
        "C",
        dist.Uniform(-1e2 * jnp.ones(n_fcqfid), 1e2 * jnp.ones(n_fcqfid)),
    )
    # Uncertainty scale factor
    log_beta = numpyro.sample(
        "log_beta",
        dist.Uniform(jnp.zeros(n_fcqfid), 1 * jnp.ones(n_fcqfid)),
    )
    beta = numpyro.deterministic("beta", 10**log_beta)

    # Parameters specific to each filter (n_filt)
    # alpha : Rising power-law index
    alpha = numpyro.sample("alpha", dist.Uniform(jnp.zeros(n_filt), 5 * jnp.ones(n_filt)))
    # Aprime : Proportionality factor
    log_A = numpyro.sample(
        "log_A",
        dist.Uniform(-5 * jnp.ones(n_filt), 5 * jnp.ones(n_filt)),
    )
    A = numpyro.deterministic("A", 10 ** log_A)
    numpyro.deterministic("Aprime", 10 ** (log_A + alpha))

    with numpyro.plate("data", len(t)):
        det = jnp.heaviside(t - tfl, 0)
        numpyro.sample(
            "flux",
            dist.Normal(
                C[idx_fcqfid] + A[idx_filt] * (det * (t - tfl)) ** alpha[idx_filt],
                beta[idx_fcqfid] * flux_err,
            ),
            obs=flux,
        )


def hierarchical_model(t, flux_err, n_fcqfid, idx_fcqfid, n_filt, idx_filt, flux=None):
    """
    Function to model the single light curve of a supernova
    in multiple fields, CCDs, quadrants, as well as filters.

    Each measurement has a unique fcqf ID defined in Yao et al. (2019)
        (fcqf ID) = (field ID) * 10000 + (CCD ID) * 100
                  + (quadrant ID) * 10 + (filter ID)

    Parameters
    ----------
    t : array-like
        Time values (phase) of the light curve.
        Phase = (t_obs - t_max) / (1 + z)
    flux : array-like
        Flux values of the light curve.
    flux_err : array-like
        Flux error values of the light curve.
    n_fcqfid, idx_fcqfid : int
        Number of unique fcqf IDs and indices used to index the fcqf IDs
        for each measurement.
    n_filt, idx_filt : int
        Number of unique filters and their indices.

    Returns
    -------
    None
    """

    pass

    # # t_fl : Time of the first light
    # tfl = numpyro.sample("t_fl", dist.Uniform(-50, 0))

    # # Parameters specific to each fcqf ID (n_fcqfid)
    # # C : Baseline flux
    # C = numpyro.sample(
    #     "C",
    #     dist.Normal(0, 1e2),
    #     sample_shape=(n_fcqfid,),
    # )
    # # Uncertainty scale factor
    # log_beta = numpyro.sample(
    #     "log_beta",
    #     dist.Uniform(jnp.zeros(n_fcqfid), 1 * jnp.ones(n_fcqfid)),
    # )
    # beta = numpyro.deterministic("beta", 10**log_beta)

    # # Parameters specific to each filter (n_filt)
    # # alpha : Rising power-law index
    # alpha = numpyro.sample("alpha", dist.Uniform(jnp.zeros(n_filt), 5 * jnp.ones(n_filt)))
    # # Aprime : Proportionality factor
    # log_Aprime = numpyro.sample(
    #     "log_A_prime",
    #     dist.Uniform(0 * jnp.ones(n_filt), 5 * jnp.ones(n_filt)),
    # )
    # A =