# Computing NSIDES on AWS

## Scripts

* scripts/1.compute_file_maps.py
* scripts/2.compute_extract_propensity_scores.py
* scripts/3.compute_prr.py
* scripts/4.combine_prr_clean.py

## Required data files

* data/meta/drug_exposure_matrix.npz
* data/meta/drug_id_vector.npz
* data/meta/outcome_matrix.npz
* data/meta/outcome_id_vector.npz
* data/meta/report_id_vector.npz

* data/archives/1/scores_*.tgz
* data/archives/2/scores_*.tgz

## Files to-be-computed

* data/meta/file_map_offsides.csv
* data/meta/file_map_twosides.csv
* data/scores/1/*.npz
* data/scores/2/*_*.npz
* data/prr/1/*.csv.xz
* data/prr/2/*.csv.xz
* data/tables/offsides.csv.xz
* data/tables/twosides.csv.xz

## Files to-be-output (not deleted)

* data/meta/file_map_offsides.csv
* data/meta/file_map_twosides.csv
* data/tables/offsides.csv.xz
* data/tables/twosides.csv.xz
* data/output_archives/offsides_propensity_scores.tar.xz
* data/output_archives/twosides_propensity_scores.tar.xz
