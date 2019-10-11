# nsides
Analysis notebooks and database interaction scripts for the nsides project

# Overview - Steps to run everything

### Preprocessing

1. Download from the MYSQL database all outcomes (MedDRA concepts) (`nb/pre/1.get_outcomes_meddra.ipynb`)
2. Create reformatted matrices of exposures and outcomes, also saving the id-to-index keys as vectors (`nb/pre/2.reformat_exposures_outcomes.ipynb`)

### Computation

1. Create file maps for OFFSIDES and TWOSIDES (`nb/1.create_file_maps.ipynb`)
2. Compute all propensity scores (by averaging across the 20 bootstrap iterations, and only those iterations where AUC > 0.5) (`nb/2.compute_average_propensity_scores_offsides.py`)
3. Compute all disproportionality statistics for OFFSIDES (PRR, PRR_error, A, B, C, D, and mean (reporting frequency)) (`nb/3.compute_prr_offsides.ipynb`)
4. Create a file map for TWOSIDES (`nb/6.create_twosides_file_map.ipynb`)
5. Compute all disproportionality statistics for TWOSIDES (PRR, PRR_error, A, B, C, D, and mean (reporting frequency)) (`nb/7.compute_prr.py`)
6. Combine all disproportionality data into single files, one for each `n` (ie. `offsides_prr.csv.xz`, `twosides.csv.xz`). (The PRR files were originally split to allow parallelization) (`nb/8.combine_prr.ipynb`)

# Method notes

### PRR

A contingency table can be drawn using exposed and unexposed cohorts produced by propensity score matching.

|  | Had outcome | Didn't have outcome |
| -- | -- | -- |
| **Drug exposed** | A | B |
| **Not drug exposed** | C | D |

Using these definitions,

<img src="https://latex.codecogs.com/svg.latex?PRR&space;=&space;\frac{\frac{A}{A&plus;B}}{\frac{C}{C&plus;D}}" title="PRR = \frac{\frac{A}{A+B}}{\frac{C}{C+D}}" />

and the error is

<img src="https://latex.codecogs.com/svg.latex?PRR_s&space;=&space;\sqrt{\frac{1}{A}&space;&plus;&space;\frac{1}{C}&space;-&space;\frac{1}{A&plus;B}&space;-&space;\frac{1}{C&plus;D}}" title="PRR_s = \sqrt{\frac{1}{A} + \frac{1}{C} - \frac{1}{A+B} - \frac{1}{C+D}}" />

Several consequences of these definitions should be taken into account when inspecting the data.

* PRR is `NaN` when both A and C are zero.
* PRR is `Inf` when C is zero but A is greater than zero.
* PRR is zero when A is zero and C is not zero.
* PRR_s is `Inf` when A or C or both is zero.

# Setup

The notebooks and scripts in this repository expect that certain source files are properly located.
The `data/` layout I employed is the following:

```
.
+-- data
|   +-- aeolus
|   |   +-- AEOLUS_all_reports_IN_0.npy
|   |   +-- ... (all-inclusive)
|   |   +-- AEOLUS_all_reports_IN_54.npy
|   +-- archives
|   |   +-- 1
|   |   |   +-- scores_1.tgz
|   |   |   +-- ... (all-inclusive)
|   |   |   +-- scores_220.tgz
|   |   +-- 2
|   |   |   +-- scores_1.tgz
|   |   |   +-- ... (all-inclusive)
|   |   |   +-- scores_?.tgz
|   +-- scores
|   |   +-- 1
|   |   |   +-- 0.npz
|   |   |   +-- ... (not all-inclusive)
|   |   |   +-- 4391.npz
|   |   +-- 2
|   |   |   +-- ?_?.npz
|   |   |   +-- ... (not all-inclusive)
|   |   |   +-- ?_?.npz
|   +-- prr
|   |   +-- 1
|   |   |   +-- 0.csv.xz
|   |   |   +-- ... (not all-inclusive)
|   |   |   +-- 4391.csv.xz
|   |   +-- 2
|   +-- meta
|   +-- tables
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
The first subdirectories tell whether the archives are for OFFSIDES, TWOSIDES, ...
The computation of these involved 20 bootstrap iterations per drug, meaning that within these archives are 20 propensity score files for each drug, which should be averaged to find the final propensity scores used for later computation.
Each archive contains files of these bootstrap iteration propensity scores (`scores_lrc_<drug>__<boostrap>.npy`), as well as performance metrics for each of these files (`log_lrc_<drug>__<bootstrap>.npy`).
In computing the average, I used only those bootstrap iterations which had an AUROC > 0.5.

### `scores`

This directory is initially empty but comes to be filled with files for each drug.
The first subdirectories tell whether the archives are for OFFSIDES, TWOSIDES, ...
Because I only averaged those bootstrap iterations with AUC > 0.5, some drugs do not have corresponding propensity score files.
Of the 3453 unique drugs, only ultimately 2757 have propensity score files.
The files in this directory are compressed `numpy.ndarray`s stored in the `.npz` format.
Because this format is intended to enable the storage of multiple arrays, the scores can be accessed by loading the `.npz` file with `numpy.load()`, and the extracting scores using an attribute of the loaded file (ie. `loaded['scores']`).

These files are each between 50 KB and 11 MB.

### `prr`

`prr` is also initially empty, and it also comes to be filled with one file per drug (the same 2757 as in `scores`).
The first subdirectories tell whether the archives are for OFFSIDES, TWOSIDES, ...
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

These files are each between 8 and 160 KB.
The combined, `data/full_prr.csv.xz` file is 52 MB, though it excludes rows with $PRR = `NaN`$.

### `meta`

This directory is for a number of files, including the following:

* `all_auc.csv`
    * Built in `nb/4.compute_scores.py` from the `log_lrc_**__**.npy` files in `archives/scores_*.tgz` archives.
* `all_drug_exposures.npz`
    * Matrix of reports by drugs. Built in `nb/3.reformat_exposures_outcomes.ipynb`.
* `all_outcomes_meddra.npz`
    * Matrix of reports by outcomes. Built in `nb/3.reformat_exposures_outcomes.ipynb`.
* `all_reportids_IN.npy`
    * Vector giving the report ID at each index in the matrix
* `drugs_vector.npy`
    * Vector giving the drug ID at each index in the matrix. Built in `nb/3.reformat_exposures_outcomes.ipynb`.
* `file_map.csv`
    * Computed in `nb/1.create_file_map.ipynb`, this file shows where score and log files are located.
* `outcomes_table.csv.xz`
    * Equivalent to the `standard_case_outcome` table from `effect_aeolus` on mimir. Built in `nb/2.get_outcomes_meddra.ipynb`.
* `outcomes_vector_meddra.npy`
    * Vector giving the outcome ID at each index in the matrix. Built in `nb/3.reformat_exposures_outcomes.ipynb`.
* `reports_outcomes.csv.xz`
    * Essentially the information that we use from `outcomes_table.csv.xz` along with matrix indices. Built in `nb/3.reformat_exposures_outcomes.ipynb`.

### `tables`

This directory is for locally saving tables that will be later inserted into the `effect_nsides` MYSQL database.
