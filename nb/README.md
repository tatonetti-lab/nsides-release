# Notebooks

## Overall process

The overall process of these notebooks is determined by the directory structure.

1. `1.mimir/` - **Run only on `mimir` server**
    - Save MySQL tables as compressed `.csv` files, as needed
    - Copy created and existing files from `mimir` to the local machine (and AWS) as described in `1.mimir/README.md`
2. `2.preprocessing/`
    - Convert data to usable formats. For example, tables to matrices for exposures and outcomes.

---
**Compute propensity scores -> PRR, etc. using `scripts/` directory**
This is not done using notebooks at all, and the computation is run on AWS rather than on `mimir` or the local machine.

---

3. `3.format_tables/`
    - Prepare data for entry into MySQL database by reformatting to compressed `.csv` files that can be transferred to `mimir`
    - Copy created and existing files from the local machine (and AWS) to `mimir` as described in `3.format_tables/README.md`
4. `4.insert_mimir_tables/` - **Run only on `mimir` server**
    - Load compressed `.csv` files and insert them into MySQL database `effect_nsides`
