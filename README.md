# nsides
Analysis notebooks and database interaction scripts for the nsides project

# Steps

1. Create file map
2. Create reformatted matrices of exposures and outcomes, also saving the id-to-index keys as vectors
3. Compute all propensity scores (by averaging across the 20 bootstrap iterations, and only those iterations where AUC > 0.5)
4. Compute all disproportionality statistics (PRR, PRR_error, A, B, C, D, and mean (reporting frequency))
