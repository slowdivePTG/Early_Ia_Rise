# Early_Ia_Rise

Hierarchical Bayesian modeling of early-time Type Ia supernova light curves.

## Single-SN Fitting

Individual light curves are fit from a portable bundle containing:

- `manifest.yaml` with `schema_version` and `filter_order`
- `objects.ecsv` with one row per supernova
- `photometry.ecsv` with model-ready normalized photometry

For the early/late ZTF sample, first export a raw calibrated-photometry bundle:

```bash
export-ztf-early-late-bundle \
  --object ZTF24aahgaov \
  --include-external-gr \
  --output bundles/ZTF24aahgaov_raw
```

Then run SALT2 followed by BayeSN to estimate the maximum flux in each native
filter and create normalized rise-model photometry:

```bash
snia-rise-calibrate \
  --bundle bundles/ZTF24aahgaov_raw \
  --output bundles/ZTF24aahgaov_calibrated \
  --filter-yaml data/ztf_snia_early_late/bayesn_filters/external_filters.yaml \
  --salt2-model-dir data/ztf_snia_early_late/salt2_models/salt2-T21 \
  --sncosmo-filter-dir data/ztf_snia_early_late/sncosmo_filters
```

BayeSN peak-flux p16/median/p84 values are saved under `calibration/`, but the
rise model normalizes by the posterior median and does not add the correlated
normalization uncertainty to each epoch's independent photometric error.

List packaged population-prior profiles:

```bash
snia-rise-fit --list-priors
```

Fit one object from a bundle:

```bash
snia-rise-fit \
  --bundle bundles/ZTF24aahgaov_calibrated \
  --object ZTF24aahgaov \
  --prior ztf-dr2-normal-frac40-mvn \
  --output results/ZTF24aahgaov
```

Use a base unpooled prior instead of a population prior:

```bash
snia-rise-fit \
  --bundle bundles/SN_TEST \
  --prior-type maximum_entropy \
  --output results/SN_TEST
```

The output directory contains a copy of the selected input bundle, the resolved
prior, run settings, posterior inference NetCDF, a multiband light-curve
diagnostic (`light_curve.pdf`), and a compact summary table (`summary.ecsv`).
The summary table includes the posterior rise-time estimate (`t_rise`) in
rest-frame days, along with per-filter parameters such as `alpha_0` and
`Aprime`.

The same functionality is available from Python:

```python
from snia_rise import (
    calibrate_single_sn_bundle,
    export_ztf_early_late_bundle,
    fit_single_sn_bundle,
)

export_ztf_early_late_bundle(
    "data/ztf_snia_early_late",
    "ZTF24aahgaov",
    "bundles/ZTF24aahgaov_raw",
    include_external_gr=True,
)
calibrate_single_sn_bundle(
    "bundles/ZTF24aahgaov_raw",
    "bundles/ZTF24aahgaov_calibrated",
    filter_yaml="data/ztf_snia_early_late/bayesn_filters/external_filters.yaml",
    salt2_model_dir="data/ztf_snia_early_late/salt2_models/salt2-T21",
    sncosmo_filter_dir="data/ztf_snia_early_late/sncosmo_filters",
)
fit_single_sn_bundle(
    "bundles/ZTF24aahgaov_calibrated",
    "results/ZTF24aahgaov",
    prior="ztf-dr2-normal-frac40-mvn",
)
```
