import os
import numpyro

numpyro.set_host_device_count(4)

# numpyro.enable_x64()

from astropy.table import Table
from ztf_lc import ZTFLib

ztfid_early = Table.read(
    "./Data/ztf_snia_dr2/tables/snia_early_data.csv", format="ascii.csv"
)["ztfname"].data

ztflib_dr2 = ZTFLib(ztfid_lib=ztfid_early, source="DR2")
ztflib_dr2.sampling(num_warmup=5000, num_samples=1000, num_chains=4, random_seed=114514)

if os.path.exists("./Data/ztf_snia_dr2/results/") is False:
    os.makedirs("./Data/ztf_snia_dr2/results/")

for k in range(len(ztflib_dr2.lc_library)):
    lc = ztflib_dr2.lc_library[k]
    posterior = lc.post_sample
    # save the posterior
    posterior.to_netcdf(f"./Data/ztf_snia_dr2/results/posterior_{ztfid_early[k]}.nc")
