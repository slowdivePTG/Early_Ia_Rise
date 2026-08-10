# Individual SN Fit Workspace

Workspace for single-object SN Ia fitting. Organize each object in its own
directory:

```text
individual_sne_fit/
  <SN_NAME>/
    raw_bundle/
    calibrated_bundle/
    fit_result/
    notes.md
```

Use the object identifier as `<SN_NAME>`, for example `ZTF24aahgaov`.

Recommended names:

- `raw_bundle/`: exported unnormalized bundle for the object
- `calibrated_bundle/`: SALT2/BayeSN-calibrated and normalized bundle
- `fit_result/`: rise-model posterior outputs, summaries, and run metadata
- `notes.md`: optional notes about source choices, calibration settings, or caveats

Example raw photometry table:

| mjd | native_filter | model_filter | flux | flux_err | zp | magsys | origin | source | stream_id |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 60310.421 | ztfg | ztfg | 12.41 | 3.02 | 30.0 | ab | ztf | ZTF | 743_6_2_ztfg |
| 60311.418 | ztfr | ztfr | 18.76 | 2.81 | 30.0 | ab | ztf | ZTF | 743_6_2_ztfr |
| 60312.305 | sdssg | ztfg | 15.33 | 1.95 | 30.0 | ab | external | Las Cumbres 1m | Las Cumbres 1m_sdssg |

The raw table is stored inside `raw_bundle/photometry.ecsv` when using the
package bundle format.

Large generated data products should stay out of Git unless they are small
fixtures needed for tests or examples.
