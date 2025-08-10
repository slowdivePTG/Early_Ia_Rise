# To-do list

1. Filter out the 127 SNe Ia in Yao+2019
    - 120 normal (including 91T and 99aa-like)
    - only 86 in Adam's BTS light curve sample - rest are non-BTS? (indeed they are way fainter)
    - should we include only BTS targets in our analysis (by requiring early detections we'll more or less fall into the category)?
2. Reproduce the results in Miller+2020 with ~~`pymc`~~ `NumPyro`
3. Build the hierarchival Bayesian model
    - Estimate the potential computational cost
4. Construct a mock dataset -- the question to answer: how much will the large sample size of ZTF help?
    - Test different cadence (e.g., TESS v.s. ZTF)
    - Test different S/N (low-z v.s. high-z)
    - Test with a simulated volume-complete ZTF Ia sample (`redback`?)
        - Fixed $\alpha$
        - $\alpha$ with given mean and variance (Gamma-distributed)
    - Try inferring population properties (i.e., mean and variance of $\alpha$) with different priors
5. Test $K$-corrections
    - SALT3+ results from D'Arcy
    - 2D GP model
6. Define the most recent ZTF Ia sample
    - Till when?
7. Filter out early SNe Ia in the new sample to which we apply the hierarchical model
8. Joint study with TESS? [TESSreduce](https://github.com/CheerfulUser/TESSreduce)