import numpy as np
import pandas as pd
import sncosmo
from redback.sed import RedbackTimeSeriesSource

from jax._src.typing import ArrayLike

MPC_TO_CM = 3.086e24


def f_t(
    t: float | ArrayLike,
    t_fl: float | ArrayLike,
    alpha_0: float | ArrayLike,
    alpha_1: float | ArrayLike = 0.0,
    eps: float = 1e-10,
):
    """
    Calculate the flux with a curved power-law rise model.

    Parameters:
    -----------
    t : float or array-like
        Time value.
    t_fl : float or array-like
        Time of the first light.
    amp : float or array-like
        Proportionality factor.
    alpha_0 : float or array-like
        Rising power-law index.
    alpha_1 : float or array-like, optional, default = 0.0
        Correction factor for the power-law rise (default is 0, i.e., simple power-law).
    eps : float, optional, default = 1e-10
        Small value to avoid numerical issues when t - t_fl is small and alpha_0 < 1

    Returns:
    --------
    float | ArrayLike
        The calculated value of f(t).
    """
    du = np.maximum(t - t_fl, eps)
    return np.where(t < t_fl, 0, np.power(du, alpha_0 * (1 + alpha_1 * du)))


def power_law_rise_flat_sed(
    time: float | ArrayLike,
    peak_luminosity: float,
    alpha_0: float,
    alpha_1: float,
    dist_lum: float,
    redshift: float,
    force_power_law: bool = False,
    **kwargs,
):
    """
    A transient model with a curved power-law rise, and a flat SED.

    Parameters:
    -----------
    time : float or array-like
        Time array in the observor frame (days since explosion).
    peak_luminosity : float
        Peak luminosity (erg/s).
    alpha_0 : float
        Rising power-law index.
    alpha_1 : float
        Correction factor for the power-law rise.
    dist_lum : float
        Luminosity distance (Mpc).
    force_power_law : bool, optional, default = False
        If True, use a simple power-law rise (alpha_1 = 0), but keep other parameters unchanged.
        Determine the flux normalization at peak time based on the curved power-law model.
    """
    from scipy.special import lambertw

    MPC_TO_CM = 3.086e24
    SPEED_OF_LIGHT = 2.99792458e10

    # Helper to extract scalar
    def to_scalar(val):
        if isinstance(val, pd.Series):
            return float(val.iloc[0])
        elif isinstance(val, np.ndarray):
            return float(val.flat[0])
        else:
            return float(val)

    # Ensure time is 1D
    phase = np.atleast_1d(time) / (1 + redshift)

    # Convert to scalars
    peak_luminosity = to_scalar(peak_luminosity)
    alpha_0 = to_scalar(alpha_0)
    alpha_1 = to_scalar(alpha_1)
    dist_lum = to_scalar(dist_lum)
    t_rise = np.exp(lambertw(-np.exp(1) / alpha_1).real - 1)

    # Calculate distance and redshift
    dist_lum_cm = dist_lum * MPC_TO_CM

    # Calculate flux
    flux_density_cgs = peak_luminosity / (4 * np.pi * dist_lum_cm**2)
    flux_density_jy = flux_density_cgs / 1e-23

    flux_max = f_t(
        t=t_rise,
        t_fl=0,
        alpha_0=alpha_0,
        alpha_1=alpha_1,
    )

    flux_norm = (
        f_t(
            t=phase,
            t_fl=0,
            alpha_0=alpha_0,
            alpha_1=0.0 if force_power_law else alpha_1,
        )
        / flux_max
    )

    flux = np.where(flux_norm > 1, 1, flux_norm) * flux_density_jy

    # Wavelength array
    lambda_array = np.linspace(3000, 10000, 100)

    # Flat SED
    flux_2d = np.tile(flux[:, np.newaxis], (1, len(lambda_array)))

    # Convert to erg/cm^2/s/A
    flux_density_cgs = (
        flux_2d * 1e-23 * SPEED_OF_LIGHT * 1e8 / lambda_array[np.newaxis, :] ** 2
    )

    # Always return sncosmo source (simpler for redback to handle)
    source = RedbackTimeSeriesSource(
        phase=phase, wave=lambda_array, flux=flux_density_cgs
    )
    return source


def snf_2011fe_sed(
    time: float | ArrayLike,
    dist_lum: float,
    redshift: float,
    t_rise: float = 0.0,
    **kwargs,
):
    """
    A transient model using the 'snf-2011fe' SED template from sncosmo.

    This wrapper converts physical parameters (peak_luminosity, dist_lum) to
    sncosmo format (amplitude, redshift) and applies proper redshift corrections.

    Parameters:
    -----------
    time : float or array-like
        Time array (days in observer frame).
    peak_luminosity : float
        Peak luminosity (erg/s).
    dist_lum : float
        Luminosity distance (Mpc).
    t_rise : float, optional, default = 0.0
        Time of B-band maximum (days).
    """

    # Calculate redshift from luminosity distance using astropy
    from astropy import units as u

    # Helper to extract scalar
    def to_scalar(val):
        if isinstance(val, pd.Series):
            return float(val.iloc[0])
        elif isinstance(val, np.ndarray):
            return float(val.flat[0])
        else:
            return float(val)

    # Ensure time is 1D
    phase = np.atleast_1d(time) / (1 + redshift)

    # Convert to scalars
    peak_luminosity = to_scalar(peak_luminosity)
    dist_lum = to_scalar(dist_lum)
    t_rise = to_scalar(t_rise)

    # Create sncosmo model with snf-2011fe source
    model = sncosmo.Model(source="snf-2011fe")
    model.set(z=redshift)
    model.set(t0=t_rise)

    # Set B-band absolute magnitude
    model.set_source_peakabsmag(-19.0, "bessellb", "ab")

    # Define wavelength array in observer frame
    lambda_array = np.linspace(3000, 10000, 100) * (1 + redshift)

    # Get the SED at each time point
    flux_2d = np.zeros((len(phase), len(lambda_array)))

    for i, t in enumerate(phase):
        try:
            # sncosmo automatically handles redshift corrections
            sed_flux = model.flux(t, lambda_array)
            flux_2d[i, :] = sed_flux

        except Exception:
            flux_2d[i, :] = np.zeros_like(lambda_array)

    # Convert to Redback source
    source_redback = RedbackTimeSeriesSource(
        phase=phase, wave=lambda_array, flux=flux_2d
    )

    return source_redback
