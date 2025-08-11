import numpy as np
import jax.numpy as jnp
import corner

from fit_early_lc import f_t, SNLightCurveLib
from _utils import plt


def generate_flux_err(flux, zp=0, method="broken_power_law"):
    """
    Estimate the flux error based on the broken power law relation between magnitude and S/N
    """
    mag = -2.5 * jnp.log10(flux) + zp
    if method == "broken_power_law":

        def log_broken_powerlaw(x, x0=18.2123, y0=-8.9946, k1=-0.3044, k2=0):
            """
            Broken power law function - mag v.s. S/N (default values adopted from the fit to the r-band data in Yao et al. 2019)
            """
            log_err = 0.5 * jnp.log10(
                (
                    jnp.power(10, (y0 + k1 * (x - x0)) * 2)
                    + jnp.power(10, (y0 + k2 * (x - x0)) * 2)
                )
            )
            log_sky_noise = y0
            return np.where(np.isfinite(x) & ~np.isnan(x), log_err, log_sky_noise)

        return 10 ** (log_broken_powerlaw(mag) + 0.4 * zp)
    else:
        raise ValueError("Method not recognized")


class MockLightCurveLib(SNLightCurveLib):
    def __init__(
        self,
        cadence: float = 1,
        n_lc: int = 10,
        var_true: dict = dict(t_fl=-20.0, C=0.0, amp_prime=50, alpha=2.0),
        var_mean: dict = dict(t_fl=-20.0, C=0.0, amp_prime=50, alpha=2.0),
        var_std: dict = dict(t_fl=1.0, C=0.1, amp_prime=5, alpha=0.1),
        fix_values: bool = True,
        mag_peak: float = 18,
        realistic_mag: bool = False,
    ) -> None:
        t_sample = jnp.arange(-100, 0, step=cadence)
        n_sample = len(t_sample)

        lc_early_lib = np.empty(shape=n_lc, dtype=object)

        np.random.seed(n_lc * int(cadence) + 114514)

        if fix_values:
            t_fl = var_true.get("t_fl", -20.0) * jnp.ones(n_lc)
            base = var_true.get("C", 0.0) * jnp.ones(n_lc)
            amp_prime = var_true.get("Aprime", 50) * jnp.ones(n_lc)
            alpha = var_true.get("alpha", 2.0) * jnp.ones(n_lc)
            self.var_true = var_true
        else:
            t_fl = np.random.randn(n_lc) * (var_std.get("t_fl", 1.0)) + var_mean.get(
                "t_fl", -20.0
            )
            base = np.random.randn(n_lc) * (var_std.get("C", 0.1)) + var_mean.get(
                "C", 0.0
            )
            amp_prime = np.random.randn(n_lc) * (
                var_std.get("Aprime", 0.1)
            ) + var_mean.get("Aprime", 50)
            alpha = np.random.randn(n_lc) * (var_std.get("alpha", 0.1)) + var_mean.get(
                "alpha", 2.0
            )
            self.var_true = dict(t_fl=t_fl, C=C, amp_prime=amp_prime, alpha=alpha)
            self.var_true["mean_alpha"] = var_mean.get("alpha", 2.0)
            self.var_true["std_alpha"] = var_std.get("alpha", 0.1)
            self.var_true["mean_t_fl"] = var_mean.get("t_fl", -20.0)
            self.var_true["std_t_fl"] = var_std.get("t_fl", 1.0)

        amp = amp_prime / jnp.power(10, alpha)

        if realistic_mag:
            # n(mag) ~ 10^(0.6*mag)
            # dmag = mag_peak - mag_peak_0 ~ Exponential(0.6)
            dmag = np.random.exponential(0.6, n_lc)
        else:
            # mag_peak = constant
            dmag = np.zeros(n_lc)

        self.mag_peak = mag_peak - dmag

        for k in range(n_lc):
            t_jitter = np.random.randn(n_sample) * 0.05  # 0.05 days = 1.2 hours
            t_jitter -= t_jitter[
                np.argmin(np.abs(t_sample - t_fl[k]))
            ]  # ensure jitter=0 when t=t_fl -> SN observed right after the exposion
            t_shift = cadence / n_lc * k  # delay the first detection time
            self.t_first_det = t_shift
            t_mock = t_sample + t_jitter + t_shift
            flux_true = f_t(
                t=t_mock, t_fl=t_fl[k], base=base[k], amp=amp[k], alpha=alpha[k]
            )

            mag_40 = self.mag_peak[k] - 2.5 * np.log10(0.4)  # 40% of peak
            t_40 = (40 / amp[k]) ** (1 / alpha[k]) + t_fl[
                k
            ]  # 40% of maximum flux is achieved at t_40
            idx_early = t_mock <= t_40
            zp_mock = 2.5 * np.log10(40) + mag_40

            t_mock_early = t_mock[idx_early]
            flux_true_early = flux_true[idx_early]
            flux_err_mock_early = generate_flux_err(
                flux_true_early, zp=zp_mock, method="broken_power_law"
            )
            flux_mock_early = (
                flux_true_early + np.random.randn(idx_early.sum()) * flux_err_mock_early
            )

            lc_early_lib[k] = dict(
                phase=t_mock_early, flux=flux_mock_early, flux_err=flux_err_mock_early
            )

        super().__init__(lc_early_lib=lc_early_lib)
        self.n_lc = n_lc
        self.var_name = dict(
            t_fl=r"$t_\mathrm{fl}$",
            base=r"$C$",
            amp=r"$A$",
            alpha=r"$\alpha$",
            mean_alpha=r"$\mu_\alpha$",
            std_alpha=r"$\sigma_\alpha$",
            mean_t_fl=r"$\mu_{t_\mathrm{fl}}$",
            std_t_fl=r"$\sigma_{t_\mathrm{fl}}$",
        )

        self.inf_data = None

    def plot_prior_posterior(
        self, var_name: list = ["alpha", "t_fl"], var_range: dict = None
    ):
        if var_range is None:
            var_range = dict(alpha=(0, 4), t_fl=(-30, -10), base=(-1, 1), amp=(0, 10))
        _, ax = plt.subplots(
            2,
            len(var_name),
            sharex="col",
            sharey="row",
            constrained_layout=True,
            figsize=(3 * len(var_name), 6),
        )
        cmap = plt.get_cmap("coolwarm")  # Choose a colormap

        for i, var in enumerate(var_name):
            # overall
            if self.inf_data is None:
                prior, posterior = [], []
                for lc in self.lc_library:
                    prior = np.append(prior, lc.inf_data.prior[var].data)
                    posterior = np.append(posterior, lc.inf_data.posterior[var].data)
                prior = np.array(prior).ravel()
                posterior = np.array(posterior).ravel()
            else:
                prior = self.inf_data.prior[var].data.ravel()
                posterior = self.inf_data.posterior[var].data.ravel()
            ax[0, i].hist(
                prior,
                histtype="step",
                bins=50,
                color="k",
                lw=3,
                weights=np.ones_like(np.array(prior).ravel()) / len(prior),
                range=var_range[var],
            )
            ax[1, i].hist(
                posterior,
                histtype="step",
                bins=25 * self.n_lc,
                color="k",
                lw=2,
                range=var_range[var],
            )

            # posterior for each light curve
            for k in range(self.n_lc):
                if self.inf_data is None:
                    posterior_k = (
                        self.lc_library[k].inf_data.posterior[var].data.ravel()
                    )
                else:
                    posterior_k = self.post_sample[var].data[:, :, k].ravel()
                ax[1, i].hist(
                    posterior_k,  # only valid for single filter (as assumed in the mock data)
                    histtype="step",
                    bins=50,
                    color=cmap(k / self.n_lc),
                    range=var_range[var],
                    zorder=-1,
                )
            try:
                ax[1, i].axvline(self.var_true[var], color="crimson", ls="--")
            except ValueError:
                ax[1, i].axvline(self.var_true["mean_" + var], color="crimson", ls="--")
            ax[1, i].axvline(np.median(np.array(posterior).ravel()), color="k", ls="--")
            ax[0, i].set_title(f"Prior: {self.var_name[var]}")
            ax[1, i].set_title(f"Posterior: {self.var_name[var]}")
            ax[1, i].set_xlabel(self.var_name[var])
        ax[0, 0].set_yticks([])
        ax[1, 0].set_yticks([])
        ax[0, 0].set_ylabel("Normalized count")
        ax[1, 0].set_ylabel("Normalized count")

        return ax

    def plot_corner(self, save: bool = False, filename: str = None, **kwargs):
        """
        Plot the corner plot of the posterior samples.

        Parameters
        ----------
        save : bool, optional
            Save the figure if True (default: False).
        filename : str, optional, default=self.ID
            Filename to save the figure.

        Returns
        -------
        None
        """

        var_name = kwargs.pop("var_name", ["alpha", "t_fl"])

        corner.corner(
            self.post_sample,
            show_titles=True,
            title_kwargs={"fontsize": 12},
            quantiles=[0.05, 0.5, 0.95],
            title_quantiles=[0.05, 0.5, 0.95],
            var_names=var_name,
            # truths=[self.var_true[var] for var in var_name],
            **kwargs,
        )

        if save:
            if filename is None:
                filename = self.ID
            plt.savefig(filename + "_corner.pdf", bbox_inches="tight")
