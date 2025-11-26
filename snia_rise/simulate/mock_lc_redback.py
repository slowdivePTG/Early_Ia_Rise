import redback
from redback.simulate_transients import SimulateOpticalTransient
import bilby
from snia_rise._utils import plt

# 1. Import your corrected custom model
from .sed import power_law_rise_flat_sed

# 2. Define the prior for your model parameters
priors = bilby.core.prior.PriorDict()
# Prior on intrinsic peak luminosity (erg/s/Hz)
priors["peak_luminosity"] = 2e28
priors["t_fl"] = bilby.core.prior.Gaussian(-18, 2, "t_fl", latex_label="$t_{fl}$")
priors["alpha_1"] = bilby.core.prior.Gaussian(2.0, 0.2, "rise_index_1")
priors["alpha_2"] = bilby.core.prior.Uniform(-0.02, 0.002, "rise_index_2")
priors["dist_lum"] = bilby.core.prior.PowerLaw(
    alpha=2,
    minimum=10,
    maximum=250,
    name="dist_lum",
    latex_label="$d_{lum}$",
    unit="Mpc",
)

# ... (the rest of the script is the same)
# 3. Sample the prior to generate parameters
num_events = 5
population_parameters = priors.sample(num_events)
population_parameters["t_peak"] += population_parameters["t0"]

# 4. Simulate the population
population_simulation = SimulateOpticalTransient.simulate_transient_population_in_ztf(
    model=power_law_rise_flat_sed,
    survey="ztf",
    parameters=population_parameters,
    end_transient_time=30,
    snr_threshold=0.0,
    add_source_noise=True,
)

breakpoint()

# 5. Save and plot the results
population_simulation.save_transient_population()
print("Population data saved. Now plotting detected events.")

# Plot the light curves
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(num_events):
    try:
        transient_obj = redback.transient.Transient.from_simulated_optical_data(
            name=f"event_{i}", data_mode="magnitude"
        )
        if transient_obj.x.shape[0] > 0:
            transient_obj.plot_data(
                ax=ax,
                data_mode="magnitude",
                show=False,
                label=f"Event {i} (z={population_parameters['redshift'][i]:.2f})",
            )
    except FileNotFoundError:
        print(f"Event {i} had no detections and was not saved.")

ax.invert_yaxis()
ax.set_xlabel("Time (MJD)")
ax.set_ylabel("Apparent Magnitude")
ax.legend()
ax.set_title("Simulated Population with Physical Power-Law Model")
plt.show()
