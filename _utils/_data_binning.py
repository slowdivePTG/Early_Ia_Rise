import numpy as np


def data_binning(
    data, size=2, min_bin=1, spec_resolution=0, sigma_clip=3
):  # size - Angstrom
    """
    binning spectroscopic data with a finite spectral resolution (FWHM)
    within the FWHM, the noise is expected to be correlated

    data : array_like
        np.array([[wavelength], [flux], [flux_err]]).T
    size : float, default = 2
        bin size in same units as that of the wavelength
    min_bin : int, default = 1
        minimum number of data points with a bin
    spec_resolution : float, default = 0
        the typical scale of noise correlation
        roughly the spectroscopic resolution
        default - noise is independent
    sigma_clip : float, default = 3
        treat data points beyond this threshold as outliers
    """
    from astropy.stats import mad_std

    data_bin = []
    i = 0
    while i < len(data):
        j = i
        while j < len(data):
            if data[j, 0] < data[i, 0] + size:
                j += 1
            else:
                break
        temp = data[i:j, :]
        if len(temp) >= min_bin:
            if len(temp) > 1:  # if there are more than 1 data points in the bin
                arg = np.arange(len(temp))
                X, Y, Yerr = (
                    temp[arg, 0].reshape(-1, 1),
                    temp[arg, 1].reshape(-1, 1),
                    temp[arg, 2].reshape(-1, 1),
                )
                if sigma_clip != None:
                    clip = np.abs(Y - np.median(Y)) <= sigma_clip * mad_std(Y)
                    X, Y, Yerr = X[clip], Y[clip], Yerr[clip]

                # mu = weight^T * Y
                # var = weight^T * Cov * weight
                weight = Yerr ** (-2.0) / (Yerr ** (-2.0)).sum()

                if spec_resolution > 0:
                    X_A = np.repeat(X, len(X)).reshape(len(X), len(X))
                    X_B = X_A.T
                    rho = np.exp(
                        -((X_A - X_B) ** 2) / (2 * (spec_resolution / 2.355) ** 2)
                    )  # correlation coefficient = exp(-r^2/2 sigma^2)
                    cov = rho * np.outer(Yerr, Yerr)
                else:
                    cov = np.diag(Yerr.flatten() ** 2)
                X_bin = X.mean()
                Y_bin = weight.T @ Y
                Yerr_bin = (weight.T @ cov @ weight) ** 0.5
                i = j
            else:  # if there is only 1 data point in the bin
                X_bin = data[i, 0]
                Y_bin = data[i, 1]
                Yerr_bin = data[i, 2]
                i += 1
            data_bin.append([X_bin, Y_bin, Yerr_bin])
        else:  # if there are less than min_bin data points in the bin
            for t in temp.reshape(len(temp.flatten()) // 3, 3):
                data_bin.append(t)
            i = j
    return np.array(data_bin)


def get_box_spec(wave, flux):
    flux_plot = np.repeat(flux, 2)
    wv_plot = wave.copy()
    wv_plot[:-1] += np.diff(wave) / 2
    wv_plot = np.append(
        wave[0] - (wave[1] - wave[0]) / 2,
        np.append(np.repeat(wv_plot[0:-1], 2), wave[-1] + (wave[-1] - wave[-2]) / 2),
    )

    # Detect large gaps and set flux to NaN within those gaps
    if len(wave) > 1:
        delta_wv = np.diff(wave)
        median_delta = np.median(delta_wv)
        gap_threshold = median_delta * 3

        # Find indices where gaps are larger than threshold
        large_gaps = delta_wv > gap_threshold

        if np.any(large_gaps):
            # For each large gap, set corresponding flux values to NaN
            for i, is_gap in enumerate(large_gaps):
                if is_gap:
                    # The gap is between wave[i] and wave[i+1]
                    # In flux_plot, this corresponds to indices 2*i+1 and 2*i+2
                    # (the right edge of bin i and left edge of bin i+1)
                    if 2 * i + 1 < len(flux_plot):
                        flux_plot[2 * i + 1] = np.nan
                    if 2 * i + 2 < len(flux_plot):
                        flux_plot[2 * i + 2] = np.nan

    return wv_plot, flux_plot
