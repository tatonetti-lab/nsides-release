import concurrent.futures
import functools
import pathlib
import re
import sys
import tarfile

import pandas as pd
import tqdm

sys.path.insert(0, '../src/')
import utils  # noqa:E402


def get_subfiles(archive_file_path, n_drugs):
    """
    Find all score, interaction, and log files within a `.tar.gz` archive.
    Each subfile is represented as a list.

    For OFFSIDES, files are [drug, bootstrap_iteration, file_type,
    subfile_name, archive_path].

    For TWOSIDES, files are [drug_1, drug_2, file_type, subfile_name,
    archive_path]

    For THREESIDES, files are [drug_1, drug_2, drug_3, file_type, subfile_name,
    archive_path]
    """
    file_locations = list()
    try:
        tar = tarfile.open(archive_file_path, mode='r:gz')
        subfiles = tar.getnames()
    except tarfile.ReadError:
        return None
    except EOFError:
        return None

    if n_drugs == 1:
        for subfile in subfiles:
            if 'interaction' in subfile:
                drug = re.match(r'(?:interactions__)([0-9]+)(?:\.npy)', subfile)
                if not drug:
                    raise ValueError(f'{archive_file_path.name} contained {subfile} not matched')
                drug = drug.group(1)
                bootstrap = None
                file_type = 'interaction'
            else:
                drug, bootstrap = utils.extract_indices(subfile)
                file_type = re.match('^[a-z]+(?=_.+)', subfile).group()
            file_locations.append([drug, bootstrap, file_type, subfile,
                                   archive_file_path.name])
    elif n_drugs == 2:
        for subfile in subfiles:
            file_name_match = re.match(r'([a-z]+)(?:.*?__)([0-9]+)(?:_)([0-9]+)(?=\.npy)', subfile)
            if not file_name_match:
                raise ValueError(f'{archive_file_path.name} contained {subfile} not matched')
            file_type, drug_1, drug_2 = file_name_match.groups()
            drug_1, drug_2 = int(drug_1), int(drug_2)
            file_locations.append([drug_1, drug_2, file_type, subfile,
                                   archive_file_path.name])
    elif n_drugs == 3:
        for subfile in subfiles:
            file_name_match = re.match(r'([a-z]+)(?:.*?__)([0-9]+)(?:_)([0-9]+)(?:_)([0-9]+)(?=\.npy)', subfile)
            if not file_name_match:
                raise ValueError(f'{archive_file_path.name} contained {subfile} not matched')
            file_type, drug_1, drug_2, drug_3 = file_name_match.groups()
            drug_1, drug_2, drug_3 = int(drug_1), int(drug_2), int(drug_3)
            file_locations.append([drug_1, drug_2, drug_3, file_type, subfile,
                                   archive_file_path.name])
    return file_locations


def compute_file_map(n_drugs, archives_path):
    """Compute a file map for an entire directory of archives"""
    get_subfiles_partial = functools.partial(get_subfiles, n_drugs=n_drugs)

    archive_files = list(archives_path.glob('scores_*.tgz'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        file_locations = list(tqdm.tqdm(
            executor.map(get_subfiles_partial, archive_files),
            total=len(archive_files)
        ))

    # Flatten list of lists of tuples to a list of tuples
    flattened_file_locations = [l for l in file_locations if l is not None]
    flattened_file_locations = [i for l in flattened_file_locations for i in l]

    # First two column names depend on the number of drugs
    n_drug_to_column_names = {
        1: ('drug', 'bootstrap'),
        2: ('drug_index_1', 'drug_index_2'),
        3: ('drug_index_1', 'drug_index_2', 'drug_index_3')
    }
    column_names = [*n_drug_to_column_names[n_drugs], 'file_type', 'file_name',
                    'archive_file']

    # Instantiate DataFrame and save to disk immediately
    return pd.DataFrame(flattened_file_locations, columns=column_names)


def compute_all_filemaps():
#     # Path to where the `.tgz` archives are stored
#     archives_path = pathlib.Path('/data/archives/')
# 
#     # Path where the file maps will be saved
#     meta_path = pathlib.Path('/data/meta')
# 
#     # Compute and save OFFSIDES file map
#     offsides_file_map = compute_file_map(1, archives_path.joinpath('1/'))
#     offsides_file_map.to_csv(meta_path.joinpath('file_map_offsides.csv'),
#                              index=False)
# 
#     # Compute and save TWOSIDES file map
#     twosides_file_map = compute_file_map(2, archives_path.joinpath('2/'))
#     twosides_file_map.to_csv(meta_path.joinpath('file_map_twosides.csv'),
#                              index=False)

    # Compute and save THREESIDES file map
    nsides_root = pathlib.Path('/data2/nsides/')
    threesides_file_map = compute_file_map(3, nsides_root.joinpath('dddi/'))
    threesides_file_map.to_csv(
        nsides_root.joinpath('meta/file_map_twosides.csv'), index=False)


if __name__ == "__main__":
    compute_all_filemaps()
