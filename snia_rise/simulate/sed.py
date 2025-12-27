import numpy as np
import pandas as pd
import sncosmo
from redback.sed import RedbackTimeSeriesSource
import astropy.units as u

from jax._src.typing import ArrayLike

MPC_TO_CM = 3.086e24
SPEED_OF_LIGHT = 2.99792458e10  # cm/s


def f_t(
    t: float | ArrayLike,
    t_fl: float | ArrayLike,
    alpha_0: float | ArrayLike,
    alpha_1: float | ArrayLike = 0.0,
    t_b: float = None,
    s: float = None,
    eps: float = 1e-10,
):
    """
    Calculate the flux with a curved/broken power-law rise model.

    Parameters:
    -----------
    t : float or array-like
        Time value.
    t_fl : float or array-like
        Time of the first light.
    amp : float or array-like
        Proportionality factor.
    alpha_0 :  float or array-like
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
    if t_b is None or s is None:
        # Simple curved power-law rise
        flux = np.power(du, alpha_0 * (1 + alpha_1 * du))
    else:
        # Broken power-law rise
        flux = (du / t_b) ** alpha_0 * (1 + (du / t_b) ** (s * alpha_1)) ** (-2 / s)
    return np.where(t < t_fl, 0, flux)


# Helper to extract scalar
def to_scalar(val):
    if isinstance(val, pd.Series):
        return float(val.iloc[0])
    elif isinstance(val, np.ndarray):
        return float(val.flat[0])
    else:
        return float(val)


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
    Observer-frame SED model for redback simulations.

    Parameters:
    -----------
    time : float or array-like
        Time array in the OBSERVER frame (days since explosion in observer frame).
    peak_luminosity : float
        Peak luminosity in rest frame (erg/s).
    alpha_0 : float
        Rising power-law index.
    alpha_1 : float
        Correction factor for the power-law rise.
    dist_lum :  float
        Luminosity distance (Mpc).
    redshift : float
        Cosmological redshift.
    """
    from scipy.special import lambertw

    # Ensure time is 1D (observer frame from redback)
    time_obs = np.atleast_1d(time)

    # Convert to scalars
    peak_luminosity = to_scalar(peak_luminosity)
    alpha_0 = to_scalar(alpha_0)
    alpha_1 = to_scalar(alpha_1)
    dist_lum = to_scalar(dist_lum)
    z = to_scalar(redshift)

    # Calculate peak time in rest frame
    assert alpha_1 < 0, "alpha_1 must be negative for a peak to exist"
    t_peak_rest = np.exp(lambertw(-np.exp(1) / alpha_1).real - 1)

    # Calculate luminosity distance
    dist_lum_cm = dist_lum * MPC_TO_CM

    # Calculate flux with proper cosmological dilution
    # F = L / (4π D_L² (1+z))
    flux_factor = 1.0 / (4.0 * np.pi * dist_lum_cm**2 * (1 + z))
    flux_density_cgs = peak_luminosity * flux_factor  # erg/s/cm²
    flux_density_jy = flux_density_cgs / 1e-23

    # Evaluate light curve in REST FRAME
    time_rest = time_obs / (1 + z)

    # Evaluate flux at rest-frame times
    flux_at_times = f_t(
        t=time_rest,
        t_fl=0,
        alpha_0=alpha_0,
        alpha_1=0.0 if force_power_law else alpha_1,
    )

    # Find the actual maximum flux
    flux_peak = f_t(t=t_peak_rest, t_fl=0, alpha_0=alpha_0, alpha_1=alpha_1)

    # Normalize to the peak flux
    flux_norm = flux_at_times / flux_peak

    if force_power_law:
        # Further calibrate, such that it reaches 40% of peak at the same time as curved power-law
        t_template = np.linspace(0, t_peak_rest, 100)
        flux_template = f_t(
            t=t_template,
            t_fl=0,
            alpha_0=alpha_0,
            alpha_1=0.0 if force_power_law else alpha_1,
        )
        t_30_template = np.interp(0.3 * flux_peak, flux_template, t_template)

        flux_norm *= f_t(t=t_30_template, t_fl=0, alpha_0=alpha_0, alpha_1=0.0) / (
            0.3 * flux_peak
        )
        flux = np.where(flux_norm > 1, 1, flux_norm) * flux_density_jy

    else:
        flux = flux_norm * flux_density_jy

    # Observer-frame wavelength grid that covers ZTF bandpasses
    lambda_obs = np.linspace(3000, 10000, 300)  # Observer-frame wavelengths (Å)

    # Flat SED in F_nu
    flux_nu_2d = np.tile(flux[:, np.newaxis], (1, len(lambda_obs)))  # Jy

    # Convert F_nu (Jy) to F_lambda (erg/s/cm²/Å) in OBSERVER frame
    flux_lambda_cgs = (
        flux_nu_2d * 1e-23 * SPEED_OF_LIGHT * 1e8 / (lambda_obs[np.newaxis, :] ** 2)
    )

    # Return RedbackTimeSeriesSource with OBSERVER-FRAME quantities
    # phase = observer-frame time, wave = observer-frame wavelengths
    source = RedbackTimeSeriesSource(
        phase=time_obs,  # Observer frame
        wave=lambda_obs,  # Observer frame
        flux=flux_lambda_cgs,  # Observer frame
    )

    return source


def broken_power_law_rise_flat_sed(
    time: float | ArrayLike,
    peak_luminosity: float,
    alpha_0: float,
    alpha_1: float,
    t_b: float,
    s: float,
    dist_lum: float,
    redshift: float,
    **kwargs,
):
    """
    Observer-frame SED model for redback simulations using a broken power-law rise.

    Parameters:
    -----------
    time : float or array-like
        Time array in the OBSERVER frame (days since explosion in observer frame).
    peak_luminosity : float
        Peak luminosity in rest frame (erg/s).
    alpha_0 : float
        Rising power-law index before the break.
    alpha_1 : float
        Rising power-law index after the break.
    t_b : float
        Break time in rest frame (days).
    s : float
        Smoothness parameter for the break.
    dist_lum :  float
        Luminosity distance (Mpc).
    redshift : float
        Cosmological redshift.
    """

    # Ensure time is 1D (observer frame from redback)
    time_obs = np.atleast_1d(time)

    # Convert to scalars
    peak_luminosity = to_scalar(peak_luminosity)
    alpha_0 = to_scalar(alpha_0)
    alpha_1 = to_scalar(alpha_1)
    dist_lum = to_scalar(dist_lum)
    z = to_scalar(redshift)

    alpha_v_1 = alpha_0 / 2 - 1
    alpha_v_2 = alpha_v_1 - alpha_1

    t_peak_rest = t_b * np.power(
        -(alpha_v_1 + 1) / (alpha_v_2 + 1), 1 / (s * (alpha_v_1 - alpha_v_2))
    )
    # Calculate luminosity distance
    dist_lum_cm = dist_lum * MPC_TO_CM

    # Calculate flux with proper cosmological dilution
    # F = L / (4π D_L² (1+z))
    flux_factor = 1.0 / (4.0 * np.pi * dist_lum_cm**2 * (1 + z))
    flux_density_cgs = peak_luminosity * flux_factor  # erg/s/cm²
    flux_density_jy = flux_density_cgs / 1e-23

    # Evaluate light curve in REST FRAME
    time_rest = time_obs / (1 + z)

    # Evaluate flux at rest-frame times
    flux_at_times = f_t(
        t=time_rest, t_fl=0, alpha_0=alpha_0, alpha_1=alpha_1, t_b=t_b, s=s
    )

    # Find the actual maximum flux
    flux_peak = f_t(
        t=t_peak_rest, t_fl=0, alpha_0=alpha_0, alpha_1=alpha_1, t_b=t_b, s=s
    )

    # Normalize to the peak flux
    flux_norm = flux_at_times / flux_peak
    flux = flux_norm * flux_density_jy

    # Observer-frame wavelength grid that covers ZTF bandpasses
    lambda_obs = np.linspace(3000, 10000, 300)  # Observer-frame wavelengths (Å)

    # Flat SED in F_nu
    flux_nu_2d = np.tile(flux[:, np.newaxis], (1, len(lambda_obs)))  # Jy

    # Convert F_nu (Jy) to F_lambda (erg/s/cm²/Å) in OBSERVER frame
    flux_lambda_cgs = (
        flux_nu_2d * 1e-23 * SPEED_OF_LIGHT * 1e8 / (lambda_obs[np.newaxis, :] ** 2)
    )

    # Return RedbackTimeSeriesSource with OBSERVER-FRAME quantities
    # phase = observer-frame time, wave = observer-frame wavelengths
    source = RedbackTimeSeriesSource(
        phase=time_obs,  # Observer frame
        wave=lambda_obs,  # Observer frame
        flux=flux_lambda_cgs,  # Observer frame
    )

    return source


def snf_2011fe_sed(
    time: float | ArrayLike,
    dist_lum: float,
    redshift: float,
    **kwargs,
):
    """
    A transient model using the 'snf-2011fe' SED template from sncosmo.

    Leverages sncosmo's built-in redshift handling.

    Parameters:
    -----------
    time : float or array-like
        Time array - days since t0_mjd_transient in observer frame (from redback).
    t0_mjd_transient : float
        MJD of the transient (not used in flux calculation, just for reference).
    dist_lum : float
        Luminosity distance (Mpc).
    redshift : float
        Cosmological redshift.
    """

    # Ensure time is 1D (observer frame from redback)
    time_obs = np.atleast_1d(time)

    # Convert to scalars
    dist_lum = to_scalar(dist_lum)
    z = to_scalar(redshift)

    # Create sncosmo model with snf-2011fe source
    model = sncosmo.Model(source="snf-2011fe")

    # Set redshift - sncosmo will handle all transformations
    model.set(z=z)

    model.set(t0=20.0 * (1 + z))  # Set t0 in observer frame

    # Set B-band absolute magnitude
    model.set_source_peakabsmag(-19.0, "bessellb", "ab")

    # Check template phase range
    source = model.source
    phase_min, phase_max = source.minphase(), source.maxphase()

    # Create a dense wavelength grid in observer frame
    # Cover optical range relevant for ZTF
    lambda_obs = np.linspace(3500, 9500, 500)  # Angstroms

    # Get the SED at each time point
    flux_2d = np.zeros((len(time_obs), len(lambda_obs)))

    # Only evaluate within the valid phase range
    valid_mask = (time_obs >= phase_min) & (time_obs <= phase_max)

    for i, t in enumerate(time_obs):
        if not valid_mask[i]:
            # Outside template range, leave as zero
            continue

        try:
            sed_flux = model.flux(t, lambda_obs)
            flux_2d[i, :] = sed_flux

        except Exception as e:
            # If still fails, leave as zero
            if i < 3:
                print(f"  WARNING at t={t:.2f}: {e}")
            continue

    # Return RedbackTimeSeriesSource with observer-frame quantities
    source_redback = RedbackTimeSeriesSource(
        phase=time_obs,
        wave=lambda_obs,
        flux=flux_2d,
    )

    return source_redback
