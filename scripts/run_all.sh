export PATH="~/miniconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate nsides

echo "Starting file 1 / 4"

python 1.compute_file_maps.py

echo "Finished file 1"

echo "Starting file 2 / 4"

python 2.compute_propensity_scores.py

echo "Starting file 3 / 4"

python 3.compute_prr.py

echo "Starting file 4 / 4"

python 4.combine_prr_clean.py

echo "Finished all!"
