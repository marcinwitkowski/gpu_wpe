import sys
import numpy as np
import logging
import os
import shutil
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from typing import Union

from tqdm import tqdm
import soundfile as sf

Pathlike = Union[Path, str]


def make_non_zero(x):
    if np.any(x == 0):
        x += sys.float_info.epsilon
    return x


def read_files_list_as_matrix(path_to_files: list):
    data = []
    fss = []
    for j in range(len(path_to_files)):
        signal, fs = sf.read(path_to_files[j])
        if len(signal.shape) == 1:
            signal = np.expand_dims(signal, axis=1)
        data.append(signal)
        fss.append(fs)

    if fss.count(fss[0]) != len(fss):
        raise Exception("Audio files have different sampling rate")

    data = np.hstack(data)
    return data, fss[0]


def tqdm_urlretrieve_hook(t):
    """Wraps tqdm instance.
    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    Example
    -------
    >>> from urllib.request import urlretrieve
    >>> with tqdm(...) as t:
    ...     reporthook = tqdm_urlretrieve_hook(t)
    ...     urlretrieve(..., reporthook=reporthook)
    Source: https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] or -1,
            remains unchanged.
        """
        if tsize not in (None, -1):
            t.total = tsize
        displayed = t.update((b - last_b[0]) * bsize)
        last_b[0] = b
        return displayed

    return update_to


def urlretrieve_progress(url, filename=None, data=None, desc=None):
    """
    Works exactly like urllib.request.urlretrieve, but attaches a tqdm hook to display
    a progress bar of the download.
    Use "desc" argument to display a user-readable string that informs what is being downloaded.
    """
    from urllib.request import urlretrieve

    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=desc) as t:
        reporthook = tqdm_urlretrieve_hook(t)
        return urlretrieve(url=url, filename=filename, reporthook=reporthook, data=data)


def download_and_unzip(url: str, target_dir: Pathlike, force_download: bool = False):
    parsed_url = urlparse(url)
    target_dir = Path(target_dir)
    zip_name = os.path.basename(parsed_url.path)[:-4]
    zip_target_dir = target_dir / zip_name
    completed_detector = zip_target_dir / ".completed"
    if completed_detector.is_file():
        logging.info(f"Skipping {zip_name} because {completed_detector} exists.")
        return zip_target_dir

    # Maybe-download the archive.
    zip_filename = f"{zip_name}.zip"
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / zip_filename
    if force_download or not zip_path.is_file():
        urlretrieve_progress(
            f"{url}", filename=zip_path, desc=f"Downloading {zip_path}"
        )
    # Remove partial unpacked files, if any, and unpack everything.
    shutil.rmtree(zip_target_dir, ignore_errors=True)
    with zipfile.ZipFile(zip_path, "r") as zip_f:
        zip_f.extractall(path=zip_target_dir)
    completed_detector.touch()
    return zip_target_dir
