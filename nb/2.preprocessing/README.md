# Preprocessing

The goal of the notebooks in directory are to generate files in `data/meta_unformatted/` that do not already exist and to convert these to files in `data/meta_formatted/` which are used in computation.

## Unchanged files

To begin, the following files should be in the `data/meta_unformatted/` directory without modification:

* `all_reportids_IN.npy`
* `unique_ingredients.npy`

## Procedure

1. The first notebook, `1.get_outcomes_meddra.ipynb` should be run on the `mimir` server, as that is where the relevant database is located.
This produces `outcomes_table.csv.xz`, which is the entire database table for outcomes, and which should be copied to `data/meta_unformatted` on the local machine.
2. The second notebook, `2.format_outcomes_data.ipynb`, creates the following: a matrix of reports by outcomes (in MedDRA coding), a vector giving the MedDRA ID corresponding to each index in the matrix, and a vector of report IDs.
Respectively, these are `data/meta_formatted/outcome_matrix.npz`, `data/meta_formatted/outcome_id_vector.npy`, and `data/meta_formatted/report_id_vector.npy`.
3. `3.format_drug_exposures_data.ipynb` creates a matrix of reports by drug exposures and a vector giving the RxNorm ID corresponding to each index in the matrix.
The drug exposure matrix uses the same index for reports as the outcomes matrix.

## Formatted files

After running these notebooks, `data/meta_formatted/` will contain the following files:

* `report_id_vector.npy`
* `drug_id_vector.npy`
* `outcome_id_vector.npy`

* `outcome_matrix.npz`
    * Reports by outcomes matrix, with the ID for each index specified by the `*id_vector.npy` files
* `drug_exposure_matrix.npz`
    * Reports by drug exposures matrix, with the ID for each index specified by the `*id_vector.npy` files
