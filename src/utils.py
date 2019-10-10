import re


def extract_indices(filename):
    match = re.match('(?:.+_lrc_)([0-9]+)(?:__)([0-9]+)(?:\.npy)', filename)
    if match:
        bootstrap, drug = match.groups()
        return int(drug), int(bootstrap)
    else:
        return False, False
