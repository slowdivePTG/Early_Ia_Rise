"""
Template model loaders and photometry engines for synthetic SN Ia workflows.

This module provides:
1) A unified schema for template-based models.
2) Loaders for:
   - TURTLS synthetic multi-band light curves (rest-frame absolute magnitudes).
   - Shen+2021 time-dependent SED models (rest-frame spectral luminosity-like grids).
   - Observed spectrophotometric models (2011fe)
3) Two photometry engines behind a common interface:
   - TURTLSBandPhotometryEngine: band-template, no K-correction approximation.
   - SEDPhotometryEngine: full SED-based synthetic photometry with redshifting.

Design notes
------------
- TURTLS files contain precomputed z=0 rest-frame band magnitudes. We approximate observed
  magnitudes as:
      m_obs(t_obs) = M_rest(t_rest) + DM(z),   t_rest = t_obs / (1+z)
  without explicit K-correction (as requested by project workflow).
- Shen SED models provide time-dependent spectra. We compute observer-frame AB magnitudes by
  redshifting the SED and integrating through observer filters, which naturally includes
  cross-filter K-correction behavior.
- Shen+2021 models are expected in the HDF5 structure used by `model_Shen.ipynb`
  with datasets: `Lnu`, `time`, and `nu`.

Dependencies
------------
Core: numpy, pandas, sncosmo
Optional cosmology object: astropy.cosmology instance with distmod(z).value and luminosity_distance(z).cgs.value
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import sncosmo

# -----------------------------
# Constants / utilities
# -----------------------------

C_AA_PER_S = 2.99792458e18  # speed of light in Angstrom / s
AB_ZEROPOINT = 48.60  # m_AB = -2.5 log10(f_nu[cgs]) - 48.60


def _as_path(path_or_name: str | Path) -> Path:
    return path_or_name if isinstance(path_or_name, Path) else Path(path_or_name)


def _distance_modulus_from_cosmo(z: float, cosmo: Any | None) -> float:
    if z <= 0:
        return 0.0
    if cosmo is None:
        raise ValueError(
            "A cosmology object is required for z > 0. "
            "Pass an astropy cosmology with distmod(z).value."
        )
    return float(cosmo.distmod(z).value)


def _match_native_time_indices(
    t_native: np.ndarray, t_query: np.ndarray, atol: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match query times to native grid times without interpolation.

    Returns
    -------
    matched : np.ndarray[bool]
        Boolean mask over t_query indicating exact (within atol) native-grid matches.
    idx : np.ndarray[int]
        For each query time, the nearest native index (valid only where matched is True).
    """
    t_native = np.asarray(t_native, dtype=float)
    t_query = np.asarray(t_query, dtype=float)

    if t_native.ndim != 1:
        raise ValueError("t_native must be 1D")
    if t_query.ndim != 1:
        raise ValueError("t_query must be 1D")
    if t_native.size == 0:
        raise ValueError("t_native cannot be empty")

    idx = np.searchsorted(t_native, t_query)
    idx = np.clip(idx, 0, t_native.size - 1)

    idx_left = np.clip(idx - 1, 0, t_native.size - 1)
    left_dist = np.abs(t_query - t_native[idx_left])
    right_dist = np.abs(t_query - t_native[idx])

    use_left = left_dist <= right_dist
    idx_nearest = np.where(use_left, idx_left, idx)

    matched = np.abs(t_query - t_native[idx_nearest]) <= float(atol)
    return matched, idx_nearest.astype(int)


def normalize_band_name(name: str) -> str:
    """
    Normalize heterogeneous band labels to internal canonical names.

    Canonical names used here:
    - "ztfg", "ztfr", "ztfi"
    - "bessellu", "bessellb", "bessellv", "bessellr", "besselli"
    """
    n = str(name).strip().lower()

    aliases = {
        "g": "ztfg",
        "gs": "ztfg",
        "g_s": "ztfg",
        "ztfg": "ztfg",
        "r": "ztfr",
        "rs": "ztfr",
        "r_s": "ztfr",
        "ztfr": "ztfr",
        "i": "ztfi",
        "is": "ztfi",
        "i_s": "ztfi",
        "ztfi": "ztfi",
        "u": "bessellu",
        "b": "bessellb",
        "v": "bessellv",
        "rc": "bessellr",
        "r_c": "bessellr",
        "ic": "besselli",
        "i_c": "besselli",
        "bessellu": "bessellu",
        "bessellb": "bessellb",
        "bessellv": "bessellv",
        "bessellr": "bessellr",
        "besselli": "besselli",
    }
    return aliases.get(n, n)


def _infer_turtls_metadata_from_name(stem: str) -> dict[str, Any]:
    """
    Parse e.g. DPL_Ni0.4_KE1.40_P3 into metadata.
    """
    meta: dict[str, Any] = {"raw_name": stem}
    parts = stem.split("_")
    if len(parts) >= 1:
        meta["profile"] = parts[0]
    for p in parts[1:]:
        if p.startswith("Ni"):
            try:
                meta["ni_mass"] = float(p.replace("Ni", ""))
            except ValueError:
                meta["ni_mass"] = p.replace("Ni", "")
        elif p.startswith("KE"):
            try:
                meta["kinetic_energy"] = float(p.replace("KE", ""))
            except ValueError:
                meta["kinetic_energy"] = p.replace("KE", "")
        elif p.startswith("P"):
            try:
                meta["scale_p"] = float(p.replace("P", ""))
            except ValueError:
                meta["scale_p"] = p.replace("P", "")
    return meta


# -----------------------------
# Data model schema
# -----------------------------


@dataclass
class TemplateSEDModel:
    """
    Unified schema for both template-band and full-SED models.

    For band-template models (TURTLS):
      - time_rest and band_abs_mag are required.
      - wavelength_rest / sed_flux are None.

    For full-SED models (Shen):
      - time_rest, wavelength_rest, sed_flux are required.
      - band_abs_mag can be empty.
    """

    model_id: str
    family: str
    time_rest: np.ndarray
    band_abs_mag: dict[str, np.ndarray] = field(default_factory=dict)

    # For full SED models
    wavelength_rest: np.ndarray | None = None  # Angstrom
    sed_flux: np.ndarray | None = (
        None  # shape (n_time, n_wave), rest-frame L_lambda-like
    )

    # Metadata and cached peak
    meta: dict[str, Any] = field(default_factory=dict)
    t_peak_b: float | None = None
    M_peak_b: float | None = None

    def is_sed_model(self) -> bool:
        return self.wavelength_rest is not None and self.sed_flux is not None

    def is_band_template(self) -> bool:
        return len(self.band_abs_mag) > 0


def validate_template_model(model: TemplateSEDModel) -> None:
    t = np.asarray(model.time_rest, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("time_rest must be a 1D array with at least 2 points")
    if np.any(~np.isfinite(t)):
        raise ValueError("time_rest contains non-finite values")
    if np.any(np.diff(t) <= 0):
        raise ValueError("time_rest must be strictly increasing")

    if model.is_band_template():
        for b, m in model.band_abs_mag.items():
            arr = np.asarray(m, dtype=float)
            if arr.shape != t.shape:
                raise ValueError(
                    f"Band '{b}' has shape {arr.shape}, expected {t.shape}"
                )
            if np.any(~np.isfinite(arr)):
                raise ValueError(f"Band '{b}' contains non-finite values")

    if model.is_sed_model():
        w = np.asarray(model.wavelength_rest, dtype=float)
        f = np.asarray(model.sed_flux, dtype=float)
        if w.ndim != 1 or w.size < 2:
            raise ValueError("wavelength_rest must be 1D with at least 2 points")
        if np.any(np.diff(w) <= 0):
            raise ValueError("wavelength_rest must be strictly increasing")
        if f.ndim != 2:
            raise ValueError("sed_flux must be a 2D array (n_time, n_wave)")
        if f.shape != (t.size, w.size):
            raise ValueError(
                f"sed_flux shape {f.shape} incompatible with "
                f"(n_time, n_wave)=({t.size}, {w.size})"
            )
        if np.any(~np.isfinite(f)):
            raise ValueError("sed_flux contains non-finite values")

    if not model.is_band_template() and not model.is_sed_model():
        raise ValueError("Model must provide either band_abs_mag or full SED arrays")


# -----------------------------
# TURTLS loader
# -----------------------------


def load_turtls_model(
    model_name_or_path: str,
    base_dir: str | Path | None = None,
) -> TemplateSEDModel:
    """
    Load a TURTLS band-template light curve file.

    Expected columns:
      Time, U, B, V, R, I, gs, rs, is
    where magnitudes are absolute AB mags in rest frame.
    """
    if base_dir is None:
        base_dir = Path("./data/TURTLS-Light-curves/56Ni_distributions/LightCurves")
    else:
        base_dir = _as_path(base_dir)

    path_candidate = _as_path(model_name_or_path)
    if path_candidate.exists():
        path = path_candidate
    else:
        name = model_name_or_path
        if not name.endswith(".dat"):
            name = f"{name}.dat"
        path = base_dir / name

    if not path.exists():
        raise FileNotFoundError(f"TURTLS model file not found: {path}")

    # TURTLS files are often whitespace-delimited with possible comment lines
    df = pd.read_csv(
        path,
        delim_whitespace=True,
        comment="#",
        header=None,
        engine="python",
    )

    # Handle either explicit headers or plain numeric table
    # If first row is non-numeric header-like, re-read with header=0.
    if not isinstance(df.iloc[0, 0], (int, float, np.number)):
        df = pd.read_csv(path, delim_whitespace=True, comment="#", engine="python")

    # Try to infer columns
    # Common order: Time U B V R I gs rs is
    if df.shape[1] < 9:
        raise ValueError(
            f"TURTLS file has {df.shape[1]} columns; expected at least 9 columns"
        )

    # If unnamed columns, assign canonical names by position
    if all(str(c).isdigit() for c in df.columns):
        df.columns = ["Time", "U", "B", "V", "R", "I", "gs", "rs", "is"][: df.shape[1]]

    time_rest = df["Time"].to_numpy(dtype=float)

    band_map = {
        "U": "bessellu",
        "B": "bessellb",
        "V": "bessellv",
        "R": "bessellr",
        "I": "besselli",
        "gs": "ztfg",
        "rs": "ztfr",
        "is": "ztfi",
    }

    band_abs_mag: dict[str, np.ndarray] = {}
    for col, bname in band_map.items():
        if col in df.columns:
            # Replace non-finite values with faint magnitude (0.0)
            data = df[col].to_numpy(dtype=float)
            data[~np.isfinite(data)] = 0.0
            band_abs_mag[bname] = data

    stem = path.stem
    meta = _infer_turtls_metadata_from_name(stem)
    meta["source_file"] = str(path)
    meta["photometry_mode"] = "band_template_no_kcorr"

    model = TemplateSEDModel(
        model_id=f"turtls:{stem}",
        family="turtls",
        time_rest=time_rest,
        band_abs_mag=band_abs_mag,
        meta=meta,
    )
    validate_template_model(model)
    return model


# -----------------------------
# Shen loader
# -----------------------------


def _load_shen_hdf5(path: Path) -> TemplateSEDModel:
    """
    Load Shen+2021 time-dependent SED model from strict HDF5 format used in
    model_Shen.ipynb.

    Required datasets
    -----------------
    - Lnu  : spectral luminosity density, shape (n_time, n_nu, n_angle) or (n_time, n_nu)
    - time : seconds since explosion
    - nu   : frequency grid in Hz
    """
    import h5py

    with h5py.File(path, "r") as f:
        if "Lnu" not in f or "time" not in f or "nu" not in f:
            missing = [k for k in ["Lnu", "time", "nu"] if k not in f]
            raise ValueError(
                f"Shen HDF5 file missing required dataset(s): {missing}. "
                "Expected datasets: 'Lnu', 'time', 'nu'."
            )

        Lnu = np.asarray(f["Lnu"], dtype=float)
        time_sec = np.asarray(f["time"], dtype=float)
        nu_hz = np.asarray(f["nu"], dtype=float)

    if Lnu.ndim == 2:
        # Promote to 3D with a single viewing angle for consistent handling.
        Lnu = Lnu[:, :, np.newaxis]
    elif Lnu.ndim != 3:
        raise ValueError(
            f"Expected Lnu to have 2 or 3 dimensions, got shape {Lnu.shape}."
        )

    # Convert time to days, matching notebook workflow.
    time_rest = time_sec / 86400.0

    # Convert frequency to wavelength in Angstrom and enforce increasing wavelength.
    # notebook equivalent: wv0 = np.flip((const.c / nu / u.Hz).to(u.AA).value)
    wavelength_rest = C_AA_PER_S / np.asarray(nu_hz, dtype=float)
    if np.any(~np.isfinite(wavelength_rest)) or np.any(wavelength_rest <= 0):
        raise ValueError(
            "Computed wavelength grid contains non-finite or non-positive values."
        )

    # Flip spectral axis so wavelength increases.
    wavelength_rest = np.flip(wavelength_rest)

    # Convert Lnu to an absolute-flux-like Fnu at 10 pc, then to Flambda.
    # This matches the notebook convention used before creating sncosmo.TimeSeriesSource.
    ten_pc_cm = 10.0 * 3.085677581491367e18
    Fnu = Lnu / (4.0 * np.pi * ten_pc_cm * ten_pc_cm)  # erg s^-1 cm^-2 Hz^-1 (at 10 pc)

    # Flip frequency axis to correspond to increasing wavelength.
    Fnu = np.flip(Fnu, axis=1)

    # F_lambda = F_nu * c / lambda^2  with c in Angstrom/s and lambda in Angstrom.
    sed_flux = np.empty_like(Fnu, dtype=float)  # (n_time, n_wave, n_angle)
    lam2 = np.clip(wavelength_rest**2, 1e-300, None)
    for i in range(Fnu.shape[2]):
        sed_flux[:, :, i] = Fnu[:, :, i] * (C_AA_PER_S / lam2[np.newaxis, :])

    stem = path.stem
    meta = {
        "source_file": str(path),
        "photometry_mode": "full_sed",
        "format": "shen2021_hdf5",
        "num_viewing_angles": int(sed_flux.shape[2]),
    }

    # Store first viewing angle in the base model arrays for compatibility.
    # Full angle cube is retained in metadata for downstream selection.
    model = TemplateSEDModel(
        model_id=f"shen2021:{stem}",
        family="shen2021",
        time_rest=time_rest,
        wavelength_rest=wavelength_rest,
        sed_flux=sed_flux[:, :, 0],
        band_abs_mag={},
        meta=meta,
    )
    model.meta["sed_flux_all_angles"] = sed_flux
    validate_template_model(model)
    return model


def load_shen2021_model(
    model_name_or_path: str,
    base_dir: str | Path | None = None,
) -> TemplateSEDModel:
    """
    Load Shen+2021 time-dependent SED model from strict HDF5 format.

    This follows the same dataset assumptions as model_Shen.ipynb.
    """
    if base_dir is None:
        base_dir = Path("/Users/chang/Desktop/SN/Shen2021/")
    else:
        base_dir = _as_path(base_dir)

    path_candidate = _as_path(model_name_or_path)
    if path_candidate.exists():
        path = path_candidate
    else:
        stem = model_name_or_path
        candidates = [
            base_dir / stem,
            base_dir / f"{stem}.h5",
            base_dir / f"{stem}.hdf5",
        ]
        path = next((p for p in candidates if p.exists()), None)

    if path is None or not path.exists():
        raise FileNotFoundError(f"Shen model file not found for: {model_name_or_path}")

    suffix = path.suffix.lower()
    if suffix not in {".h5", ".hdf5"}:
        raise ValueError(
            f"Unsupported Shen model format '{suffix}'. "
            "Expected strict HDF5 file with extension .h5 or .hdf5."
        )

    return _load_shen_hdf5(path)


def load_observed_model(
    model_name: str,
    base_dir: str | Path | None = None,
) -> TemplateSEDModel:
    """
    Load observed SN Ia spectral time series (e.g., 2011fe).

    Note: No wavelength regridding/interpolation is performed. We keep spectra on their
    native wavelength grid and require all epochs to have an identical wavelength grid
    to form a single 2D (time, wavelength) SED array. Epochs with mismatched wavelength
    sampling are dropped.
    """

    if base_dir is None:
        base_dir_path = Path("./data")
    else:
        base_dir_path = _as_path(base_dir)

    name_lower = model_name.lower()

    times = []
    spectra = []  # list of (wave, flux) tuples (rest frame)

    Z_OBJ = 0.0
    DIST_MPC = 10.0 / 1e6  # default to avoid division by zero, though overwritten

    if name_lower == "2011fe":
        # Pereira 2013
        data_dir = base_dir_path / "Pereira_2013"
        spec_files = sorted(list(data_dir.glob("*.dat")))
        if not spec_files:
            raise FileNotFoundError(f"No .dat files found in {data_dir} for 2011fe")

        Z_OBJ = 0.000804
        DIST_MPC = 7.0  # Mpc

        for spec_file in spec_files:
            with open(spec_file, "r") as f:
                header = f.readlines()
            phase = None
            for line in header:
                if "TMAX" in line:
                    phase = float(line.split("=")[1].strip().split()[0])
                    break

            if phase is None:
                continue

            # Load spectrum
            try:
                spec = np.loadtxt(spec_file)
            except Exception:
                continue

            # De-redshift
            # spec columns: wave, flux
            wave_rest = spec[:, 0] / (1 + Z_OBJ)
            flux_rest = spec[:, 1] * (1 + Z_OBJ)

            times.append(phase)
            spectra.append((wave_rest, flux_rest))

    else:
        raise ValueError(f"Unknown observation model: {model_name}")

    if not spectra:
        raise ValueError(f"No valid spectra loaded for {model_name}")

    # NOTE: No wavelength regridding / interpolation is performed here by design.
    # Observed spectra may have different native wavelength grids per epoch; we keep
    # the first epoch's native grid and REQUIRE all other epochs to match it exactly.
    #
    # This preserves the "native grid only" requirement, but it also means any
    # spectrum whose wavelength sampling differs will be dropped.
    #
    # Sort by time first so "first epoch" is deterministic.
    sorter = np.argsort(times)
    times = np.array(times)[sorter]
    spectra = [spectra[i] for i in sorter]

    # Use the first spectrum's wavelength grid as the shared native grid
    common_wave = np.asarray(spectra[0][0], dtype=float)
    if common_wave.ndim != 1 or common_wave.size < 2:
        raise ValueError("Invalid wavelength grid in observed spectrum")

    # Distance scaling to 10pc (absolute flux)
    # F_10pc = F_dist * (dist / 10pc)^2
    dist_factor = (DIST_MPC * 1e6 / 10.0) ** 2

    # Keep only epochs whose wavelength grid matches exactly
    kept_times = []
    kept_fluxes = []

    for t, (w, f) in zip(times, spectra):
        w = np.asarray(w, dtype=float)
        f = np.asarray(f, dtype=float)

        if w.shape != common_wave.shape:
            continue
        if not np.allclose(w, common_wave, rtol=0.0, atol=0.0):
            continue

        kept_times.append(t)
        kept_fluxes.append(f * dist_factor)

    if len(kept_times) < 2:
        raise ValueError(
            "Observed SED epochs do not share an identical wavelength grid; "
            "cannot build a 2D (time, wavelength) SED without interpolation."
        )

    times = np.asarray(kept_times, dtype=float)
    sed_flux = np.vstack([np.asarray(ff, dtype=float) for ff in kept_fluxes])

    meta = {
        "source": f"observation_{name_lower}",
        "photometry_mode": "full_sed",
        "num_viewing_angles": 1,
        "original_dist_mpc": DIST_MPC,
    }

    return TemplateSEDModel(
        model_id=f"observation:{name_lower}",
        family="observation",
        time_rest=times,
        wavelength_rest=common_wave,
        sed_flux=sed_flux,
        band_abs_mag={},
        meta=meta,
    )


def load_template_model(
    model_id: str,
    registry: dict[str, dict[str, Any]] | None = None,
) -> TemplateSEDModel:
    """
    Dispatch loader by model_id:
      - "turtls:<name>"
      - "shen2021:<name>"
    Optionally resolve through registry first.
    """
    if registry is not None and model_id in registry:
        entry = registry[model_id]
        family = entry.get("family", "")
        source = entry.get("source", entry.get("filename", entry.get("path", "")))
        base_dir = entry.get("base_dir", None)
        if family == "turtls":
            return load_turtls_model(source, base_dir=base_dir)
        if family in {"shen", "shen2021"}:
            return load_shen2021_model(source, base_dir=base_dir)
        if family in {"observation"}:
            return load_observed_model(source, base_dir=base_dir)
        raise ValueError(f"Unsupported family '{family}' in registry for {model_id}")

    if ":" not in model_id:
        raise ValueError(
            "model_id must include family prefix, e.g. 'turtls:DPL_Ni0.4_KE1.40_P3' "
            "or 'shen2021:model_name'"
        )

    family, name = model_id.split(":", 1)
    family = family.strip().lower()
    name = name.strip()

    if family == "turtls":
        return load_turtls_model(name)
    if family in {"shen", "shen2021"}:
        return load_shen2021_model(name)
    if family in {"observation"}:
        return load_observed_model(name)

    raise ValueError(f"Unsupported model family prefix: {family}")


# -----------------------------
# Peak-finding helper
# -----------------------------


def find_peak_poly4(
    time: np.ndarray,
    mag: np.ndarray,
    window: float = 10.0,
) -> tuple[float, float]:
    """
    Find minimum magnitude epoch using local 4th-order polynomial fit.
    Returns (t_peak, m_peak).
    """
    t = np.asarray(time, dtype=float)
    m = np.asarray(mag, dtype=float)

    i0 = int(np.argmin(m))
    t0 = t[i0]
    sel = np.abs(t - t0) <= float(window)
    t_fit = t[sel]
    m_fit = m[sel]

    if t_fit.size < 6:
        # fallback discrete
        return float(t0), float(m[i0])

    # Fit m(t) with poly4 and find critical points
    coef = np.polyfit(t_fit, m_fit, deg=4)
    p = np.poly1d(coef)
    dp = np.polyder(p)
    ddp = np.polyder(dp)

    roots = np.roots(dp)
    roots = roots[np.isreal(roots)].real
    # Keep roots in fitting range and minima only
    in_range = (roots >= t_fit.min()) & (roots <= t_fit.max())
    roots = roots[in_range]
    roots = roots[ddp(roots) > 0]

    if roots.size == 0:
        return float(t0), float(m[i0])

    vals = p(roots)
    j = int(np.argmin(vals))
    return float(roots[j]), float(vals[j])


def compute_band_peak(
    model: TemplateSEDModel,
    band: str = "bessellb",
    method: str = "poly4",
) -> tuple[float, float]:
    b = normalize_band_name(band)
    if b not in model.band_abs_mag:
        raise ValueError(f"Band '{b}' not available in model {model.model_id}")

    t = model.time_rest
    m = model.band_abs_mag[b]

    if method.lower() == "poly4":
        return find_peak_poly4(t, m)
    i = int(np.argmin(m))
    return float(t[i]), float(m[i])


# -----------------------------
# Photometry engine interface
# -----------------------------


class BasePhotometryEngine(Protocol):
    model: TemplateSEDModel

    def get_obs_mag(
        self,
        filters: list[str],
        t_obs: np.ndarray,
        z: float,
        cosmo: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]: ...

    def get_peak(
        self,
        band: str = "bessellb",
        method: str = "poly4",
        z: float = 0.0,
        cosmo: Any | None = None,
        **kwargs: Any,
    ) -> tuple[float, float]: ...


# -----------------------------
# TURTLS photometry engine
# -----------------------------


class TURTLSBandPhotometryEngine:
    """
    Band-template photometry engine (no K-correction).
    """

    def __init__(self, model: TemplateSEDModel) -> None:
        if model.family != "turtls":
            raise ValueError("TURTLSBandPhotometryEngine requires family='turtls'")
        self.model = model

    @classmethod
    def from_model(cls, model_name_or_path: str, base_dir: str | Path | None = None):
        return cls(load_turtls_model(model_name_or_path, base_dir=base_dir))

    @staticmethod
    def _map_obs_filter_to_rest_band(filter_name: str) -> str:
        """
        Map observer filters to available TURTLS rest-frame bands.
        """
        f = normalize_band_name(filter_name)
        # direct mapping for current workflow
        mapping = {
            "ztfg": "ztfg",
            "ztfr": "ztfr",
            "ztfi": "ztfi",
            # allow direct request of Bessell bands
            "bessellb": "bessellb",
            "bessellv": "bessellv",
            "bessellr": "bessellr",
            "besselli": "besselli",
            "bessellu": "bessellu",
        }
        if f not in mapping:
            raise ValueError(f"Unsupported TURTLS filter mapping for '{filter_name}'")
        return mapping[f]

    def get_obs_mag(
        self,
        filters: list[str],
        t_obs: np.ndarray,
        z: float,
        cosmo: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        t_obs = np.asarray(t_obs, dtype=float)
        t_rest = t_obs / (1.0 + float(z))
        dm = _distance_modulus_from_cosmo(float(z), cosmo)

        out: dict[str, np.ndarray] = {}
        for f in filters:
            f_norm = normalize_band_name(f)
            b_rest = self._map_obs_filter_to_rest_band(f_norm)
            if b_rest not in self.model.band_abs_mag:
                raise ValueError(
                    f"Model {self.model.model_id} missing band '{b_rest}' "
                    f"required for filter '{f}'"
                )
            mags = np.full(t_rest.shape, np.nan, dtype=float)
            matched, idx_native = _match_native_time_indices(
                self.model.time_rest, t_rest
            )
            mags[matched] = self.model.band_abs_mag[b_rest][idx_native[matched]] + dm
            out[f_norm] = mags
        return out

    def get_peak(
        self,
        band: str = "bessellb",
        method: str = "poly4",
        z: float = 0.0,
        cosmo: Any | None = None,
        **kwargs: Any,
    ) -> tuple[float, float]:
        t_peak, M_peak = compute_band_peak(self.model, band=band, method=method)
        if z > 0:
            dm = _distance_modulus_from_cosmo(float(z), cosmo)
            return t_peak * (1.0 + z), M_peak + dm
        return t_peak, M_peak


# -----------------------------
# SED photometry engine (Shen+2021 and observation-based)
# -----------------------------


class SEDPhotometryEngine:
    """
    Full-SED synthetic photometry engine.

    Uses sncosmo for AB mags.
    """

    def __init__(self, model: TemplateSEDModel) -> None:
        if model.family not in {"shen2021", "observation"}:
            raise ValueError(
                "SEDPhotometryEngine requires family='shen2021' or 'observation'"
            )
        if not model.is_sed_model():
            raise ValueError("Model must provide wavelength_rest and sed_flux")
        self.model = model

    @classmethod
    def from_model(cls, model_name_or_path: str, base_dir: str | Path | None = None):
        return cls(load_shen2021_model(model_name_or_path, base_dir=base_dir))

    def get_num_viewing_angles(self) -> int:
        return self.model.meta.get("num_viewing_angles", 1)

    def _get_sed_grid(self, angle_idx: int | None = None) -> np.ndarray:
        """
        Return native SED grid at model time samples (no time interpolation).
        Shape: (n_time, n_wave)
        """
        if angle_idx is not None:
            if "sed_flux_all_angles" not in self.model.meta:
                if angle_idx == 0:
                    sed = self.model.sed_flux
                else:
                    raise ValueError("Model does not contain multi-angle flux data.")
            else:
                sed_all = self.model.meta["sed_flux_all_angles"]
                if not (0 <= angle_idx < sed_all.shape[2]):
                    raise ValueError(
                        f"angle_idx {angle_idx} out of range [0, {sed_all.shape[2]})"
                    )
                sed = sed_all[:, :, angle_idx]
        else:
            sed = self.model.sed_flux
        return np.asarray(sed, dtype=float)

    def get_obs_mag(
        self,
        filters: list[str],
        t_obs: np.ndarray,
        z: float,
        cosmo: Any | None = None,
        *,
        angle_idx: int | None = None,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        """
        Compute observer-frame AB magnitudes for requested filters.

        Parameters
        ----------
        filters : list[str]
            Observer filters, e.g. ["ztfg", "ztfr"].
        t_obs : array-like
            Observer-frame time (days since explosion reference).
        z : float
            Redshift.
        cosmo : astropy cosmology or compatible object
            Must provide luminosity_distance(z).cgs.value for full flux scaling.
        angle_idx : int, optional
            Index of viewing angle to use for SED flux.
        """
        t_obs = np.asarray(t_obs, dtype=float)
        z = float(z)
        t_rest = t_obs / (1.0 + z)

        wave_rest = self.model.wavelength_rest
        sed_native = self._get_sed_grid(angle_idx=angle_idx)  # (n_native, n_wave)
        matched, idx_native = _match_native_time_indices(self.model.time_rest, t_rest)

        # Convert rest-frame F_lambda at 10pc to observed f_lambda:
        # f_lambda_obs(λ_obs) = F_lambda_10pc(λ_rest) * (10pc / D_L)^2 / (1+z)
        # with λ_obs = λ_rest (1+z)
        if z > 0 and cosmo is None:
            raise ValueError(
                "cosmo is required for full-SED photometry at z > 0 "
                "(needs luminosity distance)."
            )

        ten_pc_cm = 10.0 * 3.085677581491367e18
        if z > 0:
            D_L = float(cosmo.luminosity_distance(z).cgs.value)  # cm
            geom = (ten_pc_cm / D_L) ** 2 / (1.0 + z)
        else:
            # If z=0, we are effectively at 10pc (absolute magnitude).
            geom = 1.0

        wave_obs = wave_rest * (1.0 + z)

        out: dict[str, np.ndarray] = {}
        matched_idx = np.where(matched)[0]
        for f in filters:
            f_norm = normalize_band_name(f)
            mags = np.full(t_obs.size, np.nan, dtype=float)

            if matched_idx.size > 0:
                t_obs_matched = t_obs[matched_idx]
                flux_obs_native = sed_native[idx_native[matched_idx]] * geom
                src = sncosmo.TimeSeriesSource(
                    phase=t_obs_matched,
                    wave=wave_obs,
                    flux=flux_obs_native,
                    zero_before=True,
                )

                for j, t in enumerate(t_obs_matched):
                    try:
                        mags[matched_idx[j]] = src.bandmag(f_norm, "ab", t)
                    except Exception:
                        mags[matched_idx[j]] = np.nan

            out[f_norm] = mags

        return out

    def get_peak(
        self,
        band: str = "bessellb",
        method: str = "poly4",
        z: float = 0.0,
        cosmo: Any | None = None,
        *,
        t_grid: np.ndarray | None = None,
        angle_idx: int | None = None,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """
        Compute band peak by synthesizing band magnitudes from SED then finding minimum.
        """
        b = normalize_band_name(band)
        if t_grid is None:
            t_grid = self.model.time_rest * (1.0 + z)

        mags = self.get_obs_mag(
            filters=[b],
            t_obs=np.asarray(t_grid, dtype=float),
            z=z,
            cosmo=cosmo,
            angle_idx=angle_idx,
        )[b]

        if method.lower() == "poly4":
            mask_valid = (t_grid < 30) & (np.isfinite(mags))
            return find_peak_poly4(
                np.asarray(t_grid[mask_valid], dtype=float), mags[mask_valid]
            )
        i = int(np.nanargmin(mags))
        return float(t_grid[i]), float(mags[i])


# -----------------------------
# Engine factory
# -----------------------------


def build_photometry_engine(
    model_id: str,
    registry: dict[str, dict[str, Any]] | None = None,
) -> TURTLSBandPhotometryEngine | SEDPhotometryEngine:
    model = load_template_model(model_id, registry=registry)
    if model.family == "turtls":
        return TURTLSBandPhotometryEngine(model)
    if model.family in {"shen2021", "observation", "observed"}:
        return SEDPhotometryEngine(model)
    raise ValueError(f"Unsupported model family: {model.family}")
