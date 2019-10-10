# nsides
Analysis notebooks and database interaction scripts for the nsides project

# Setup

The notebooks and scripts in this repository expect that certain source files are properly located.
The `data/` layout I employed is the following:

```
.
+-- _config.yml
+-- data
|   +-- aeolus
|   |   +-- AEOLUS_all_reports_IN_0.npy
|   |   +-- ... (all-inclusive)
|   |   +-- AEOLUS_all_reports_IN_54.npy
|   +-- archives
|   |   +-- scores_1.tgz
|   |   +-- ... (all-inclusive)
|   |   +-- scores_220.tgz
|   +-- scores
|   |   +-- 0.npz
|   |   +-- ... (not all-inclusive)
|   |   +-- 4391.npz
|   +-- prr
|   |   +-- 0.csv.xz
|   |   +-- ... (not all-inclusive)
|   |   +-- 4391.csv.xz
|   +-- meta
```

### `aeolus`

`aeolus` files are (reports x drug exposures).
The split into 55 files is apparently to reduce the size of individual files was done before I joined the project.
In `nb/2.reformat_exposures_outcomes.ipynb` I load and combine all these files, in order, and I save the resulting array as `data/meta/all_drug_exposures.npz`, a `scipy.sparse.csc_matrix` of dimension (4694086 x 4396).
I had to truncate the combined array to have the correct number of reports, as `AEOLUS_all_reports_IN_54.npy` had excessive all-zero rows at the end, making the combination of all `AEOLUS_all_reports_IN_**.npy` files have more rows than reports.
The number of possible exposures in this array, 4396, is incorrect, because it contains duplicates (3453 is the correct number).
However, to maintain consistency with the work previously done I did not correct this error, but simply used the first column corresponding to each ingredient as the true values for that ingredient.

It is possible, of course, to use the code here when files are located elsewhere, but care must be taken.
When possible, I have attempted to make these path assignments in obvious locations, such as the first cell in a notebook or one of the first lines in `main` for scripts, though some paths may still be irregularly relative.

### `archives`

`archives` contains `.tgz` (`.tar.gz`) archives of propensity scores for each drug.
The computation of these involved 20 bootstrap iterations per drug, meaning that within these archives are 20 propensity score files for each drug, which should be averaged to find the final propensity scores used for later computation.
Each archive contains files of these bootstrap iteration propensity scores (`scores_lrc_<drug>__<boostrap>.npy`), as well as performance metrics for each of these files (`log_lrc_<drug>__<bootstrap>.npy`).
In computing the average, I used only those bootstrap iterations which had an AUROC > 0.5.

### `scores`

This directory is initially empty but comes to be filled with files for each drug.
Because I only averaged those bootstrap iterations with AUC > 0.5, some drugs do not have corresponding propensity score files.
Of the 3453 unique drugs, only ultimately 2757 have propensity score files.
The files in this directory are compressed `numpy.ndarray`s stored in the `.npz` format.
Because this format is intended to enable the storage of multiple arrays, the scores can be accessed by loading the `.npz` file with `numpy.load()`, and the extracting scores using an attribute of the loaded file (ie. `loaded['scores']`).

### `prr`

`prr` is also initially empty, and it also comes to be filled with one file per drug (the same 2757 as in `scores`).
These files correspond to disproportionality statistics for a given drug and all (MedDRA) outcomes.
Each file is a `.csv` file compressed using the LZMA algorithm (result is an `.xz` file).
The columns of these files are the following: drug_id (RxNorm ID), outcome_id (MedDRA ID), A, B, C, D, PRR, and PRR_error.

These values correspond to the following:

* A is the number of reports with exposure to the drug who had the outcome
* B is the number of reports with exposure to the drug who did not have the outcome
* C is the number of reports without exposure to the drug who had the outcome
* D is the number of reports without exposure to the drug who did not have the outcome

For C and D, the number of unexposed reports was determined by binned propensity score matching.
That is, the propensity scores were binned, and the number of exposed reports in the bin was paired to 10x the number of unexposed reports in the bin, sampled with replacement.
In cases where every bin having an exposed report also had at least one unexposed report, there are 10x the total number of exposed reports in the combined unexposed group.
However, there were some cases in which a bin had only exposed reports.
In these cases simply no unexposed cases were added for the bin.

# Steps

1. Create file map
2. Create reformatted matrices of exposures and outcomes, also saving the id-to-index keys as vectors
3. Compute all propensity scores (by averaging across the 20 bootstrap iterations, and only those iterations where AUC > 0.5)
4. Compute all disproportionality statistics (PRR, PRR_error, A, B, C, D, and mean (reporting frequency))
