# -*- coding: utf-8 -*-
# License: BSD 3-Clause
# Author: LKouadio (a.k.a. @Daniel) <etanoyau@gmail.com>

"""
The `gofast-datasets-rload` module provides tools for remotely loading and
managing datasets stored in Zenodo and GitHub repositories. This module
includes functions and classes to retrieve data archives, extract files, and
organize datasets for easy access and integration into processing workflows.

This module is essential for users needing datasets not included in the
package distribution, enabling remote dataset fetching, setup, and management.

Bagoue Dataset:
---------------
The Bagoue dataset is not included directly within the `gofast` package. Instead,
it is retrieved from the `watex` package, which hosts the dataset and related
documentation. To learn more about the `watex` package and its dataset, visit:
  - GitHub Repository: https://github.com/earthai-tech/watex
  - Documentation: https://watex.readthedocs.io/

Configuration details for the Bagoue dataset, including repository paths, Zenodo
records, and GitHub links, are specified below in `DEFAULT_DATA_CONFIG`. For
minimal pre-processed Bagoue data, users can load it directly within `gofast`:

    >>> from gofast.datasets import load_bagoue
    >>> bagoue_dataset = load_bagoue()
    >>> print(bagoue_dataset.DESCR)  # Provides details on the data

Published studies utilizing this dataset can be found at:
  - https://doi.org/10.1029/2021WR031623
  - https://doi.org/10.1007/s11269-023-03562-5

Key Features:
-------------
- **Remote Data Loading**: Fetch datasets from remote repositories such as
  Zenodo and GitHub.
- **Archive Extraction**: Support for extracting files from various archive
  formats, including `.zip`, `.tar.gz`, and `.rar`.
- **Dataset Organization**: Tools to move and structure datasets for seamless
  integration with data processing pipelines.

Examples:
---------
>>> from gofast.datasets.rload import load_dataset
>>> data = load_dataset('my_dataset_name')
>>> print(data.head())

Notes:
------
- The module handles dependencies gracefully, providing informative messages
  if required packages (e.g., `tqdm`) are unavailable.
- Logging is configured to offer detailed information during the data loading
  process, aiding in debugging and tracking.

References:
-----------
- `GoFast GitHub Repository <https://github.com/earthai-tech/gofast>`_
- `WATex GitHub Repository <https://github.com/earthai-tech/watex>`_
- `Zenodo <https://zenodo.org/>`_

"""

from __future__ import print_function, annotations

import os
import sys
import subprocess
import shutil
import zipfile

try:
    import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from .._gofastlog import gofastlog
from ..decorators import RunReturn, smartFitRun
from ..api.types import Optional
from ..api.property import BaseClass
from ..tools.depsutils import import_optional_dependency

# Initialize logging
_logger = gofastlog().get_gofast_logger(__name__)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Default Configuration for Bagoue Dataset
# --------------------------------------------------------------------------
_DATA = 'data/geodata/main.bagciv.data.csv'
_ZENODO_RECORD = '10.5281/zenodo.5571534'

# Paths for Tar and CSV files within the dataset archive
_TGZ_DICT = {
    'tgz_f': 'data/__tar.tgz/fmain.bagciv.data.tar.gz',
    'csv_f': '/__tar.tgz_files__/___fmain.bagciv.data.csv'
}

# GitHub Repository details for remote data retrieval
_GIT_DICT = {
    'root': 'https://raw.githubusercontent.com/earthai-tech/watex/master/',
    'repo': 'https://github.com/earthai-tech/watex',
    'blob_root': 'https://github.com/earthai-tech/watex/blob/master/'
}
_GIT_DICT['url_tgz'] = f"{_GIT_DICT['root']}{_TGZ_DICT['tgz_f']}"

# Comprehensive default configuration for the Bagoue dataset
DEFAULT_DATA_CONFIG = {
    'data_path': 'data/geodata/main.bagciv.data.csv',
    'zenodo_record': '10.5281/zenodo.5571534',
    'tgz_file': 'data/__tar.tgz/fmain.bagciv.data.tar.gz',
    'csv_file': '/__tar.tgz_files__/___fmain.bagciv.data.csv',
    'git_root': 'https://raw.githubusercontent.com/earthai-tech/watex/master/',
    'repo_url': 'https://github.com/earthai-tech/watex',
    'blob_root': 'https://github.com/earthai-tech/watex/blob/master/',
    'url_tgz': 'https://raw.githubusercontent.com/earthai-tech/watex/master/data/__tar.tgz/fmain.bagciv.data.tar.gz',
    'zip_or_rar_file': 'BagoueCIV__dataset__main.rar'
}

__all__ = [
    "load_dataset", "GFRemoteLoader", "extract_and_move_archive_file",
    "move_data", "extract_from_rar", "extract_from_zip"
]

def load_dataset(
    data_config: dict = DEFAULT_DATA_CONFIG
):
    """
    Load a dataset based on the provided configuration. By default, it loads
    the Bagoue dataset.

    This function leverages the `GFRemoteLoader` to download, extract, and
    organize datasets from remote repositories, including GitHub and Zenodo.
    It is especially useful for datasets not bundled directly within the
    `gofast` package.

    Parameters
    ----------
    data_config : dict, optional
        A configuration dictionary for the dataset, containing details on
        file paths, URLs, and repository records. Defaults to
        `DEFAULT_DATA_CONFIG`, which provides the configuration for the
        Bagoue dataset.
        
        Required keys include:
        
        - `zenodo_record` : str
            Zenodo record ID for the dataset.
        - `git_root` : str
            Base URL for the GitHub raw content, used for direct access.
        - `repo_url` : str
            URL for the main GitHub repository page.
        - `url_tgz` : str
            URL for the compressed `.tgz` file.
        - `blob_root` : str
            URL for GitHub blob content, used for file navigation.
        - `zip_or_rar_file` : str
            Filename of the compressed dataset archive.
        - `csv_file` : str
            Name of the CSV file within the archive.
        - `data_path` : str
            Path where the extracted data will be stored locally.

    Returns
    -------
    None
        The function does not return any value but loads the dataset and
        saves it to the specified location.

    Raises
    ------
    FileNotFoundError
        Raised if the dataset cannot be found or retrieved from the remote
        repository.
    ValueError
        Raised if the configuration dictionary is missing required keys.

    Notes
    -----
    The function automatically handles dataset retrieval, including data
    extraction and caching, where applicable. It uses `GFRemoteLoader`,
    which manages the complexities of remote file handling, including
    retries, decompression, and data structure organization.

    Examples
    --------
    >>> from gofast.datasets.rload import load_dataset
    >>> # Load the default Bagoue dataset
    >>> load_dataset()
    dataset:   0%|                                          | 0/1 [00:00<?, ?B/s]
    ### -> Decompressing 'fmain.bagciv.data.tar.gz' file...
    --- -> Decompression failed for 'fmain.bagciv.data.tar.gz'
    --- -> 'main.bagciv.data.csv' not found locally
    ### -> Fetching data from GitHub...
    +++ -> Data successfully loaded from GitHub!
    dataset: 100%|##################################| 1/1 [00:03<00:00,  3.38s/B]

    >>> # Using a custom configuration
    >>> custom_data_config = {
    ...     'zenodo_record': '10.5281/zenodo.1234567',
    ...     'git_root': 'https://raw.githubusercontent.com/earthai-tech/customrepo/master/',
    ...     'repo_url': 'https://github.com/earthai-tech/customrepo',
    ...     'url_tgz': 'https://raw.githubusercontent.com/earthai-tech/customrepo/master/data.tar.gz',
    ...     'blob_root': 'https://github.com/earthai-tech/customrepo/blob/master/',
    ...     'zip_or_rar_file': 'CustomDataset.rar',
    ...     'csv_file': '/data_files/custom_data.csv',
    ...     'data_path': 'data/custom_data.csv'
    ... }
    >>> load_dataset(custom_data_config)

    See Also
    --------
    GFRemoteLoader : Handles dataset retrieval from remote repositories,
                     including decompression and data structuring.

    References
    ----------
    .. [1] `GoFast GitHub Repository <https://github.com/earthai-tech/gofast>`_
    .. [2] `Zenodo <https://zenodo.org/>`_
    
    """
    GFRemoteLoader(
        zenodo_record=data_config['zenodo_record'],
        content_url=data_config['git_root'],
        repo_url=data_config['repo_url'],
        tgz_file=data_config['url_tgz'],
        blobcontent_url=data_config['blob_root'],
        zip_or_rar_file=data_config['zip_or_rar_file'],
        csv_file=data_config['csv_file'],
        verbose=10
    ).run(data_config['data_path'])

@smartFitRun
class GFRemoteLoader(BaseClass):
    """
    Load `GoFast` package data from online sources such as Zenodo, GitHub,
    or local files.

    This class provides methods to retrieve datasets stored remotely in 
    repositories or Zenodo records, with functionality for downloading,
    decompressing, and structuring datasets in a local environment.

    Parameters
    ----------
    zenodo_record : str, optional
        A Zenodo digital object identifier (DOI) or filepath to a Zenodo 
        record, used for fetching data from Zenodo.
    content_url : str, optional
        URL to access the repository user content directly, such as:
        'https://raw.githubusercontent.com/user/repo/branch/' for GitHub.
    repo_url : str, optional
        URL for the main repository page that hosts the dataset.
    tgz_file : str, optional
        URL to the TGZ (or TAR.GZ) file if the data is stored in this format.
    blobcontent_url : str, optional
        URL to the blob root for accessing raw GitHub data files.
    zip_or_rar_file : str, optional
        Filename of the ZIP or RAR file, if the data is compressed in either 
        of these formats.
    csv_file : str, optional
        Path to the primary CSV file to retrieve within the dataset record.
    verbose : int, optional
        Level of verbosity for logging information. Higher values result in
        more detailed messages, with a default of `0`.

    Attributes
    ----------
    zenodo_record : str
        Zenodo DOI or identifier for the dataset.
    content_url : str
        URL for accessing repository content.
    repo_url : str
        URL for the hosting repository.
    tgz_file : str
        Path or URL to the TGZ archive.
    blobcontent_url : str
        URL to the blob content root.
    zip_or_rar_file : str
        Filename of ZIP or RAR data archive.
    csv_file : str
        Path to the CSV file within the dataset.
    verbose : int
        Verbose output level for logging.
    f : str
        Path to the main file used by the instance, set during operations.

    Methods
    -------
    run(f: str = None) -> 'GFRemoteLoader'
        Retrieves the dataset from the configured source, trying local, 
        GitHub, and Zenodo in that order.

    from_zenodo(zenodo_record: Optional[str] = None, ...)
        Fetches data directly from Zenodo by record ID, with options for
        specifying files within the archive.

    from_git_repo(f: Optional[str] = None, ...)
        Downloads the dataset from a GitHub repository to the specified
        local location.

    from_local_machine(f: str = None) -> bool
        Checks for the local existence of the dataset and loads it if
        available, also extracting any TGZ/TAR files.

    Examples
    --------
    >>> from gofast.datasets.rload import GFRemoteLoader
    >>> loader = GFRemoteLoader(
            zenodo_record='10.5281/zenodo.5571534',
            content_url='https://raw.githubusercontent.com/earthai-tech/gofast/master/',
            repo_url='https://github.com/WEgeophysics/gofast',
            tgz_file='https://raw.githubusercontent.com/earthai-tech/gofast/master/data/__tar.tgz/fmain.bagciv.data.tar.gz',
            blobcontent_url='https://github.com/WEgeophysics/gofast/blob/master/',
            zip_or_rar_file='BagoueCIV__dataset__main.rar',
            csv_file='/__tar.tgz_files__/___fmain.bagciv.data.csv',
            verbose=10
        )
    >>> loader.run('data/geodata/main.bagciv.data.csv')

    Notes
    -----
    The `run` method attempts a multi-source retrieval, prioritizing local
    files first, then GitHub, and finally Zenodo if no local or GitHub data
    is found. This approach allows for efficient data handling across 
    various hosting platforms and formats.


    References
    ----------
    .. [1] `GoFast GitHub Repository <https://github.com/earthai-tech/gofast>`_
    .. [2] `Zenodo <https://zenodo.org/>`_
    """

    def __init__(
        self,
        zenodo_record: Optional[str] = None,
        content_url: Optional[str] = None,
        repo_url: Optional[str] = None,
        tgz_file: Optional[str] = None,
        blobcontent_url: Optional[str] = None,
        zip_or_rar_file: Optional[str] = None,
        csv_file: Optional[str] = None,
        verbose: int = 0
    ):
        self.zenodo_record = zenodo_record
        self.content_url = content_url
        self.blobcontent_url = blobcontent_url
        self.repo_url = repo_url
        self.tgz_file = tgz_file
        self.zip_or_rar_file = zip_or_rar_file
        self.csv_file = csv_file
        self.verbose = verbose
        self._f = None

    @property
    def zenodo_record(self) -> str:
        """str: Zenodo record identifier."""
        return self._zenodo_record

    @zenodo_record.setter
    def zenodo_record(self, value: str):
        self._zenodo_record = value

    @property 
    def f(self): 
        return self.f_ 
    @f.setter 
    def f (self, file): 
        """ assert the file exists"""
        self.f_ = file 

    @RunReturn 
    def run(self, f: str = None) -> 'GFRemoteLoader':
        """
        Retrieve dataset from GitHub repository, Zenodo record, or local file.

        Parameters
        ----------
        f : str, optional
            Path-like string to the main file containing the data.

        Returns
        -------
        Loader
            The instance of the Loader class.

        Notes
        -----
        Retrieving a dataset like the Bagoue dataset from GitHub or Zenodo 
        can take a while during the first fetch. The method attempts to fetch 
        from local storage first, then GitHub, and finally Zenodo.

        Examples
        --------
        >>> from gofast.datasets.rload import RemoteLoader 
        >>> loader = RemoteLoader(...)
        >>> loader.run('data/geodata/main.bagciv.data.csv')
        """
        if f is not None:
            self.f = f

        if not os.path.isdir(os.path.dirname(self.f)):
            os.makedirs(os.path.dirname(self.f))

        if self._try_load_from_local() or self._try_load_from_github() \
            or self._try_load_from_zenodo():
            _logger.info(f"{os.path.basename(self.f)!r} was successfully loaded.")
        else:
            _logger.error(f"Unable to load {os.path.basename(self.f)!r}"
                          " from any source.")
        

    def _try_load_from_local(self) -> bool:
        """
        Try to load the dataset from a local file.

        Returns
        -------
        bool
            True if the file was successfully loaded, False otherwise.
        """
        if os.path.exists(self.f):
            _logger.info(f"Found {self.f} locally.")
            # Load the data from self.f
            # Example: data = pd.read_csv(self.f)
            return True
        _logger.info(f"{self.f} not found locally.")
        return False

    def _try_load_from_github(self) -> bool:
        """
        Try to load the dataset from a GitHub repository.

        Returns
        -------
        bool
            True if the file was successfully loaded, False otherwise.
        """
        import_optional_dependency("requests")
        import requests 
        try:
            response = requests.get(self.content_url + self.f)
            if response.status_code == 200:
                _logger.info(f"Successfully fetched {self.f} from GitHub.")
                # Load the data from response.content
                # Example: data = pd.read_csv(io.StringIO(response.text))
                return True
        except Exception as e:
            _logger.error(f"Error fetching {self.f} from GitHub: {e}")
        return False

    def _try_load_from_zenodo(self) -> bool:
        """
        Try to load the dataset from a Zenodo record.

        Returns
        -------
        bool
            True if the file was successfully loaded, False otherwise.
        """
        import_optional_dependency("requests")
        import requests 
        # Assuming zenodo_record is a URL or identifier to fetch the data from Zenodo
        try:
            response = requests.get(self.zenodo_record)
            if response.status_code == 200:
                _logger.info("Successfully fetched data from Zenodo "
                             f"record {self.zenodo_record}.")
                # Load the data from response.content
                # Example: data = pd.read_csv(io.StringIO(response.text))
                return True
        except Exception as e:
            _logger.error(f"Error fetching data from Zenodo "
                          f"record {self.zenodo_record}: {e}")
        return False

    def _initialize_progress_bar(self):
        """
        Initialize a tqdm progress bar if tqdm is available.

        Returns
        -------
        tqdm.tqdm or range
            A tqdm progress bar object or a range object if tqdm is not available.
        """
        if TQDM_AVAILABLE:
            return tqdm.tqdm(range(1), ascii=True, unit='B', desc="dataset", ncols=77)
        return range(1)

    def from_zenodo(self,  
        zenodo_record: Optional[str] = None, 
        f: Optional[str] = None,  
        zip_or_rar_file: Optional[str] = None,
        csv_file: Optional[str] = None
        ) -> Optional[str]: 
        """
        Fetch data from Zenodo records.

        Parameters
        ----------
        zenodo_record : str, optional
            Record ID or DOI from Zenodo database.
        f : str, optional
            Path-like object to the main file containing the data.
        zip_or_rar_file : str, optional
            Path to a .zip or .rar file in the record.
        csv_file : str, optional
            Path to the main CSV file to retrieve from the record.

        Returns
        -------
        str or None
            Path to the retrieved file, or None if the process fails.

        Examples
        --------
        >>> loader = RemoteLoader(verbose=10)
        >>> loader._fetch_data_from_zenodo(
                zenodo_record='10.5281/zenodo.1234567',
                zip_or_rar_file='dataset_main.zip',
                csv_file='data.csv'
            )
        """
        if zenodo_record is not None: 
            self.zenodo_record = zenodo_record

        if f is not None: 
            self.f = f 

        if not self.zenodo_record:
            raise ValueError("Zenodo record ID or DOI is required.")

        # Ensure the directory for `f` exists
        if f and not os.path.isdir(os.path.dirname(f)):
            os.makedirs(os.path.dirname(f))

        # Try importing zenodo_get, install if not present
        try:
            import zenodo_get
        except ImportError:
            if self._install_zenodo_get():
                import zenodo_get  # noqa
            else:
                _logger.error("Failed to install `zenodo_get`.")
                return None
        # Download data from Zenodo
        if not self._download_from_zenodo():
            return None

        # Process the downloaded archive
        return self._process_downloaded_archive(zip_or_rar_file, csv_file)

    def _install_zenodo_get(self) -> bool:
        """Install zenodo_get package if not already installed."""
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'zenodo_get'])
            _logger.info("'zenodo_get' installation complete.")
            return True
        except subprocess.CalledProcessError:
            _logger.error("Failed to install `zenodo_get`.")
            return False

    def _download_from_zenodo(self) -> bool:
        """Download data from Zenodo using zenodo_get."""
        try:
            subprocess.check_call([sys.executable, '-m', 'zenodo_get', self.zenodo_record])
            _logger.info(f"Record {self.zenodo_record!r} successfully downloaded.")
            return True
        except subprocess.CalledProcessError:
            _logger.error(f"Failed to download record {self.zenodo_record!r}.")
            return False

    def _process_downloaded_archive(
            self, zip_or_rar_file: Optional[str], csv_file: Optional[str]
            ) -> Optional[str]:
        """
        Process the downloaded archive file.

        Parameters
        ----------
        zip_or_rar_file : str, optional
            Path to the downloaded ZIP or RAR file.
        csv_file : str, optional
            Name of the CSV file to extract from the archive.

        Returns
        -------
        str or None
            Path to the extracted CSV file, or None if the process fails.
        """
        archive_path = os.path.join(os.getcwd(), zip_or_rar_file
                                    ) if zip_or_rar_file else None
        if archive_path.endswith(".rar"): 
            import_optional_dependency("rarfile")
            import  rarfile
            
        if archive_path and os.path.isfile(archive_path):
            extracted_dir = os.path.join(os.path.dirname(archive_path), "extracted_files")
            os.makedirs(extracted_dir, exist_ok=True)

            if zipfile.is_zipfile(archive_path):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extracted_dir)
            elif rarfile.is_rarfile(archive_path):
                with rarfile.RarFile(archive_path, 'r') as rar_ref:
                    rar_ref.extractall(extracted_dir)
            else:
                _logger.error(f"Unsupported archive format for {archive_path}.")
                return None

            if csv_file:
                csv_path = self._find_and_move_csv(csv_file, extracted_dir)
                return csv_path

            return extracted_dir

        return None

    def _find_and_move_csv(self, csv_file: str, extracted_dir: str) -> Optional[str]:
        """
        Find and move the CSV file from the extracted directory to the desired location.

        Parameters
        ----------
        csv_file : str
            Name of the CSV file to find.
        extracted_dir : str
            Directory where files are extracted.

        Returns
        -------
        str or None
            Path to the moved CSV file, or None if not found.
        """
        for root, dirs, files in os.walk(extracted_dir):
            if csv_file in files:
                csv_path = os.path.join(root, csv_file)
                desired_location = os.path.join(os.path.dirname(extracted_dir), csv_file)
                shutil.move(csv_path, desired_location)
                return desired_location

        _logger.error(f"{csv_file} not found in the extracted files.")
        return None

    def from_git_repo(self, f: Optional[str] = None,
                                content_url: Optional[str] = None
                    ) -> Optional[str]:
        """
        Fetch data from a GitHub repository and save it to the local machine.

        Parameters
        ----------
        f : str, optional
            Path-like object representing the main file containing the data.
        content_url : str, optional
            URL to the repository user content. For example, it can be
            'https://raw.githubusercontent.com/user/repo/branch/'.

        Returns
        -------
        str or None
            Path to the downloaded file, or None if the download fails.
        """
        if f is not None: 
            self.f = f 
            
        if content_url is not None: 
            self.content_url = content_url

        if not self.content_url or not self.f:
            _logger.error("Content URL and file path are required.")
            return None

        # Ensure the directory for `f` exists
        if not os.path.isdir(os.path.dirname(self.f)): 
            os.makedirs(os.path.dirname(self.f))

        file_url = os.path.join(self.content_url, self.f)
        success = self._attempt_download(file_url)

        if not success:
            _logger.error(f"Failed to download data from {file_url}")
            return None

        _logger.info(f"Successfully downloaded {file_url}")
        return self.f

    def _attempt_download(self, file_url: str) -> bool:
        """
        Attempt to download a file from the given URL.

        Parameters
        ----------
        file_url : str
            URL of the file to be downloaded.

        Returns
        -------
        bool
            True if the download was successful, False otherwise.
        """
        import_optional_dependency("requests")
        import requests 
        
        try:
            response = requests.get(file_url)
            if response.status_code == 200:
                with open(self.f, 'wb') as file:
                    file.write(response.content)
                return True
        except requests.RequestException as e:
            _logger.error(f"Request error: {e}")
        return False

    def from_local_machine(self, f: str = None) -> bool:
        """
        Check whether the local file exists and return file name.

        It also reads and extracts .tgz and .tar files if they exist locally.

        Parameters
        ----------
        f : str
            Path-like object representing the main file containing the data.

        Returns
        -------
        bool
            True if the file or extracted file is available, False otherwise.
        """
        if f is not None:
            self.f = f

        if os.path.isfile(self.f):
            return True

        return self._attempt_extract_tgz_or_tar()

    def _attempt_extract_tgz_or_tar(self) -> bool:
        """
        Attempt to extract a .tgz or .tar file locally to find the required file.

        This method also displays a progress bar during the extraction process.

        Returns
        -------
        bool
            True if the extraction is successful and the required file is found, 
            False otherwise.
        """
        import_optional_dependency("tarfile")
        import tarfile 
        if self.tgz_file and os.path.isfile(self.tgz_file):
            try:
                with tarfile.open(self.tgz_file, 'r:*') as tar:
                    members = tar.getmembers()
                    # Setting up the progress bar
                    with tqdm(total=len(members), desc="Extracting",
                              unit="file", ascii=True, leave=False, ncols=77) as pbar:
                        for member in members:
                            tar.extract(member, path=os.path.dirname(self.f))
                            pbar.update(1)
                extracted_file = os.path.join(os.path.dirname(self.f),
                                              os.path.basename(self.f))
                if os.path.isfile(extracted_file):
                    _logger.info(f"Successfully decompressed and found {extracted_file}.")
                    self.f = extracted_file
                    return True
            except Exception as e:
                _logger.error(f"Failed to decompress {self.tgz_file}: {e}")

        return False


def extract_and_move_archive_file(
        archive_file: str, target_file: str, destination_dir: str, 
        new_name: str = None) -> str:
    """
    Extract a specific file from a ZIP or RAR archive and optionally rename it.

    Parameters
    ----------
    archive_file : str
        Path to the .zip or .rar archive.
    target_file : str
        File to extract from the archive.
    destination_dir : str
        Directory where the extracted file will be placed.
    new_name : str, optional
        New name for the extracted file.

    Returns
    -------
    str
        Path to the extracted (and possibly renamed) file.

    Raises
    ------
    FileNotFoundError
        If the archive file does not exist.
    ValueError
        If the archive format is not supported.

    Example
    -------
    >>> from gofast.datasets.rload import extract_and_move_archive_file 
    >>> extract_and_move_archive_file('path/to/archive.zip', 'file_inside.zip',
                                      'destination_dir', 'new_file_name.csv')
    'destination_dir/new_file_name.csv'
    """
    if not os.path.isfile(archive_file):
        raise FileNotFoundError(f"Archive file not found: {archive_file}")

    _, ext = os.path.splitext(archive_file)
    extracted_file_path = None

    if ext.lower() == '.zip':
        extracted_file_path = extract_from_zip(archive_file, target_file, destination_dir)
    elif ext.lower() == '.rar':
        extracted_file_path = extract_from_rar(archive_file, target_file, destination_dir)
    else:
        raise ValueError("Unsupported archive format")

    if new_name:
        new_path = os.path.join(destination_dir, new_name)
        shutil.move(extracted_file_path, new_path)
        return new_path

    return extracted_file_path

def extract_from_zip(zip_file: str, file_to_extract: str, destination: str) -> str:
    """
    Extract a single file from a ZIP archive.

    Parameters
    ----------
    zip_file : str
        Path to the ZIP file.
    file_to_extract : str
        File to extract from the ZIP archive.
    destination : str
        Destination directory for the extracted file.

    Returns
    -------
    str
        Path to the extracted file.

    Example
    -------
    >>> from gofast.datasets.rload import extract_from_zip
    >>> extract_from_zip('path/to/archive.zip', 'file_inside.zip', 'destination_dir')
    'destination_dir/file_inside.zip'
    """
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extract(file_to_extract, destination)
        return os.path.join(destination, file_to_extract)

def extract_from_rar(rar_file: str, file_to_extract: str, destination: str) -> str:
    """
    Extract a single file from a RAR archive.

    Parameters
    ----------
    rar_file : str
        Path to the RAR file.
    file_to_extract : str
        File to extract from the RAR archive.
    destination : str
        Destination directory for the extracted file.

    Returns
    -------
    str
        Path to the extracted file.

    Example
    -------
    >>> from gofast.datasets.rload import extract_from_rar
    >>> extract_from_rar('path/to/archive.rar', 'file_inside.rar', 'destination_dir')
    'destination_dir/file_inside.rar'
    """
    import_optional_dependency("rarfile")
    import rarfile 
    with rarfile.RarFile(rar_file, 'r') as rar_ref:
        rar_ref.extract(file_to_extract, destination)
        return os.path.join(destination, file_to_extract)

def move_data(source: str, destination: str) -> None:
    """
    Move a data file from one location to another.

    Parameters
    ----------
    source : str
        Path to the source file.
    destination : str
        Path to the destination directory.

    Returns
    -------
    None

    Example
    -------
    >>> from gofast.datasets.rload import move_data
    >>> move_data('path/to/source_file.txt', 'path/to/destination_dir')
    """

    if os.path.isfile(source):
        shutil.move(source, destination)
        _logger.info(f"File {source} moved to {destination}")
    else:
        _logger.error(f"File not found: {source}")
