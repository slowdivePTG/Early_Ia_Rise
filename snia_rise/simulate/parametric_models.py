import numpy as np
import pandas as pd
from jax._src.typing import ArrayLike
from redback.sed import RedbackTimeSeriesSource

from ..constants import EPS, T_PIVOT

MPC_TO_CM = 3.086e24
SPEED_OF_LIGHT = 2.99792458e10  # cm/s


def f_t_norm(
    t: float | ArrayLike,
    t_fl: float | ArrayLike,
    alpha_0: float | ArrayLike,
    alpha_1: float | ArrayLike = 0.0,
    t_pivot: float | ArrayLike = T_PIVOT,
    t_b: float = None,
    s: float = None,
    eps: float = EPS,
):
    """
    Calculate the flux with a curved/broken power-law rise model.

    Parameters:
    -----------
    t : float or array-like
        Time value.
    t_fl : float or array-like
        Time of the first light.
    alpha_0 :  float or array-like
        Rising power-law index.
    alpha_1 : float or array-like, optional, default = 0.0
        Correction factor for the power-law rise (default is 0, i.e., simple power-law).
    t_pivot : float or array-like, optional, default = 7.0
        Pivot time for the (broken) power-law rise.
    t_b : float or array-like, optional, default = None
        Break time for the broken power-law rise.
    s : float or array-like, optional, default = None
        Slope parameter for the broken power-law rise.
    eps : float, optional, default = EPS
        Small value to avoid numerical issues when t - t_fl is small and alpha_0 < 1

    Returns:
    --------
    float | ArrayLike
        The calculated value of f(t).
    """
    du = np.maximum(t - t_fl, eps)
    if t_b is None or s is None:
        # Simple curved power-law rise
        flux = np.power(du / t_pivot, alpha_0 * (1 + alpha_1 * du / t_pivot))
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
    dist_lum: float,
    redshift: float,
    amp_prime: float = None,
    t_pivot: float = T_PIVOT,
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
    dist_lum = to_scalar(dist_lum)
    z = to_scalar(redshift)

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
    flux_at_times = f_t_norm(
        t=time_rest,
        t_fl=0,
        alpha_0=alpha_0,
        alpha_1=0.0,
        t_pivot=t_pivot,
    )
    # amp_prime: flux at t_pivot
    flux_norm = (amp_prime / 100) * flux_at_times
    flux = np.where(flux_norm > 1, 1, flux_norm) * flux_density_jy

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


def power_law_plus_gaussian_bump_flat_sed(
    time: float | ArrayLike,
    peak_luminosity: float,
    alpha_0: float,
    dist_lum: float,
    redshift: float,
    amp_prime: float = None,
    t_cen: float = 2.0,
    t_sigma: float = 0.5,
    t_fwhm: float = None,
    amp: float = 1.0,
    t_pivot: float = T_PIVOT,
    **kwargs,
):
    """
    Observer-frame SED model for redback simulations: power-law rise with an
    additive Gaussian bump in normalized flux space.

    Parameters:
    -----------
    time : float or array-like
        Time array in the OBSERVER frame (days since explosion in observer frame).
    peak_luminosity : float
        Peak luminosity in rest frame (erg/s).
    alpha_0 : float
        Rising power-law index.
    dist_lum : float
        Luminosity distance (Mpc).
    redshift : float
        Cosmological redshift.
    amp_prime : float, optional
        Baseline power-law normalization in normalized flux units (%).
    t_cen : float, optional
        Center of Gaussian bump in rest-frame days.
    t_sigma : float, optional
        Width of Gaussian bump in rest-frame days. Must be > 0.
    t_fwhm : float, optional
        Full-width at half-maximum of Gaussian bump in rest-frame days.
        If provided, overrides t_sigma/t_cen via:
        t_sigma = t_fwhm / (2 * sqrt(2 * ln 2)), t_cen = 2 * t_sigma.
    amp : float, optional
        Amplitude of Gaussian bump in normalized flux units where 100
        corresponds to the baseline peak flux.
    t_pivot : float, optional
        Pivot time for power-law rise.
    """

    # Ensure time is 1D (observer frame from redback)
    time_obs = np.atleast_1d(time)

    # Convert to scalars
    peak_luminosity = to_scalar(peak_luminosity)
    alpha_0 = to_scalar(alpha_0)
    dist_lum = to_scalar(dist_lum)
    z = to_scalar(redshift)
    t_cen = to_scalar(t_cen)
    t_sigma = to_scalar(t_sigma)
    amp = to_scalar(amp)

    if t_fwhm is not None:
        t_fwhm = to_scalar(t_fwhm)
        if t_fwhm <= 0:
            raise ValueError("t_fwhm must be positive for Gaussian bump.")
        t_sigma = t_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        t_cen = 2.0 * t_sigma

    if t_sigma <= 0:
        raise ValueError("t_sigma must be positive for Gaussian bump.")

    # Calculate luminosity distance
    dist_lum_cm = dist_lum * MPC_TO_CM

    # Calculate flux with proper cosmological dilution
    # F = L / (4π D_L² (1+z))
    flux_factor = 1.0 / (4.0 * np.pi * dist_lum_cm**2 * (1 + z))
    flux_density_cgs = peak_luminosity * flux_factor  # erg/s/cm²
    flux_density_jy = flux_density_cgs / 1e-23

    # Evaluate light curve in REST FRAME
    time_rest = time_obs / (1 + z)

    # Baseline power-law component in normalized flux units
    flux_at_times = f_t_norm(
        t=time_rest,
        t_fl=0,
        alpha_0=alpha_0,
        alpha_1=0.0,
        t_pivot=t_pivot,
    )
    flux_norm_base = (amp_prime / 100) * flux_at_times

    # Additive Gaussian bump in normalized flux units
    bump_norm = (amp / 100) * np.exp(-0.5 * ((time_rest - t_cen) / t_sigma) ** 2)

    # Total normalized flux, clipped to preserve existing normalization behavior
    flux_norm = flux_norm_base + bump_norm
    flux = np.where(flux_norm > 1, 1, flux_norm) * flux_density_jy

    # Observer-frame wavelength grid that covers ZTF bandpasses
    lambda_obs = np.linspace(3000, 10000, 300)  # Observer-frame wavelengths (Å)

    # Flat SED in F_nu
    flux_nu_2d = np.tile(flux[:, np.newaxis], (1, len(lambda_obs)))  # Jy

    # Convert F_nu (Jy) to F_lambda (erg/s/cm²/Å) in OBSERVER frame
    flux_lambda_cgs = (
        flux_nu_2d * 1e-23 * SPEED_OF_LIGHT * 1e8 / (lambda_obs[np.newaxis, :] ** 2)
    )

    # Return RedbackTimeSeriesSource with OBSERVER-FRAME quantities
    source = RedbackTimeSeriesSource(
        phase=time_obs,  # Observer frame
        wave=lambda_obs,  # Observer frame
        flux=flux_lambda_cgs,  # Observer frame
    )

    return source


def curved_power_law_rise_flat_sed(
    time: float | ArrayLike,
    peak_luminosity: float,
    alpha_0: float,
    alpha_1: float,
    dist_lum: float,
    redshift: float,
    amp_prime: float = None,
    t_pivot: float = T_PIVOT,
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
    t_peak_rest = np.exp(lambertw(-np.exp(1) / alpha_1).real - 1) * t_pivot

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
    flux_at_times = f_t_norm(
        t=time_rest,
        t_fl=0,
        alpha_0=alpha_0,
        alpha_1=alpha_1,
        t_pivot=t_pivot,
    )

    # Find the actual maximum flux
    flux_peak = f_t_norm(
        t=t_peak_rest,
        t_fl=0,
        alpha_0=alpha_0,
        alpha_1=alpha_1,
        t_pivot=t_pivot,
    )
    # Normalize to the peak flux
    flux = flux_at_times / flux_peak * flux_density_jy

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
    flux_at_times = f_t_norm(
        t=time_rest, t_fl=0, alpha_0=alpha_0, alpha_1=alpha_1, t_b=t_b, s=s
    )

    # Find the actual maximum flux
    flux_peak = f_t_norm(
        t=t_peak_rest, t_fl=0, alpha_0=alpha_0, alpha_1=alpha_1, t_b=t_b, s=s
    )

    # Normalize to the peak flux
    flux = flux_at_times / flux_peak * flux_density_jy

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
