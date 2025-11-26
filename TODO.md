# To-do list

- [x] Filter out the 127 SNe Ia in Yao+2019
    - 120 normal (including 91T and 99aa-like)
    - only 86 in Adam's BTS light curve sample - rest are non-BTS? (indeed they are way fainter)
    - should we include only BTS targets in our analysis (by requiring early detections we'll more or less fall into the category)?
- [x] Reproduce the results in Miller+2020 with ~~`pymc`~~ `NumPyro`
- [x] Build the hierarchival Bayesian model
    - Estimate the potential computational cost
- [x] Construct a mock dataset -- the question to answer: how much will the large sample size of ZTF help?
    - Test different cadence (e.g., TESS v.s. ZTF)
    - Test different S/N (low-z v.s. high-z)
    - Test with a simulated volume-complete ZTF Ia sample
        - Fixed $\alpha$
        - $\alpha$ with given mean and variance (Gamma-distributed)
    - Try inferring population properties (i.e., mean and variance of $\alpha$) with different priors
- [ ] Test alternative rise models
    - Modified fireball (Firth+2015, Vallely+2021, Fausnaugh+2023)
    - Broken power-law (Zheng+2013)

    The test focuses on how well we can
    1. recover the input parameters with the correct model
    2. recover the rise time and color evolution with an incorrect model

- [ ] Define the new ZTF Ia sample (ZTF DR2)
    - How the cutoff of the rise to fit affects the results?
- [ ] Joint study with TESS? [TESSreduce](https://github.com/CheerfulUser/TESSreduce)