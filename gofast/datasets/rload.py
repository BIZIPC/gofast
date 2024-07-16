# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com> 

"""
Fetch data online from zenodo record or repository.  
Provides functions and classes for loading and managing datasets, including 
remote loading, extracting from archives, and moving

"""
from __future__ import print_function , annotations 
import os 
import sys 
import subprocess 
import shutil  
import zipfile

from ..api.types import Optional
try: 
    import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from ..tools._dependency import import_optional_dependency
from .._gofastlog import  gofastlog

_logger = gofastlog().get_gofast_logger(__name__)

##### config repo data ################################################

_DATA = 'data/geodata/main.bagciv.data.csv'
_ZENODO_RECORD= '10.5281/zenodo.5571534'

_TGZ_DICT = dict (
    # path = 'data/__tar.tgz', 
    tgz_f = 'data/__tar.tgz/fmain.bagciv.data.tar.gz', 
    csv_f = '/__tar.tgz_files__/___fmain.bagciv.data.csv'
 )

_GIT_DICT = dict(
    root  = 'https://raw.githubusercontent.com/WEgeophysics/gofast/master/' , 
    repo = 'https://github.com/WEgeophysics/gofast' , 
    blob_root = 'https://github.com/WEgeophysics/gofast/blob/master/'
 )
_GIT_DICT ['url_tgz'] = _GIT_DICT.get ('root') + _TGZ_DICT.get('tgz_f')

# Default configuration for the Bagoue dataset
DEFAULT_DATA_CONFIG = {
    'data_path': 'data/geodata/main.bagciv.data.csv',
    'zenodo_record': '10.5281/zenodo.5571534',
    'tgz_file': 'data/__tar.tgz/fmain.bagciv.data.tar.gz',
    'csv_file': '/__tar.tgz_files__/___fmain.bagciv.data.csv',
    'git_root': 'https://raw.githubusercontent.com/WEgeophysics/gofast/master/',
    'repo_url': 'https://github.com/WEgeophysics/gofast',
    'blob_root': 'https://github.com/WEgeophysics/gofast/blob/master/',
    'url_tgz': 'https://raw.githubusercontent.com/WEgeophysics/gofast/master/data/__tar.tgz/fmain.bagciv.data.tar.gz',
    'zip_or_rar_file': 'BagoueCIV__dataset__main.rar'
}

__all__=["load_dataset", "RemoteLoader","extract_and_move_archive_file", 
         "move_data", "extract_from_rar", "extract_from_zip", 
         ]

def load_dataset(data_config: dict=DEFAULT_DATA_CONFIG):
    """
    Load a dataset based on the provided configuration. Defaults to the Bagoue dataset.

    Parameters
    ----------
    data_config : dict
        Configuration dictionary for the dataset.

    Example
    -------
    # Usage example with default Bagoue dataset
    # load_dataset()
    # Usage example with a different dataset (user needs to provide a configuration dictionary)
    # custom_data_config = { ... }  # Custom configuration
    # load_dataset(custom_data_config)
    >>> from gofast.datasets.rload import load_dataset

    >>> load_dataset()
    ... dataset:   0%|                                          | 0/1 [00:00<?, ?B/s]
    ... ### -> Wait while decompressing 'fmain.bagciv.data.tar.gz' file ...
    ... --- -> Fail to decompress 'fmain.bagciv.data.tar.gz' file
    ... --- -> 'main.bagciv.data.csv' not found in the local machine
    ... ### -> Wait while fetching data from GitHub...
    ... +++ -> Load data from GitHub successfully done!
    ... dataset: 100%|##################################| 1/1 [00:03<00:00,  3.38s/B]
    """
    RemoteLoader(
        zenodo_record=data_config['zenodo_record'],
        content_url=data_config['git_root'],
        repo_url=data_config['repo_url'],
        tgz_file=data_config['url_tgz'],
        blobcontent_url=data_config['blob_root'],
        zip_or_rar_file=data_config['zip_or_rar_file'],
        csv_file=data_config['csv_file'],
        verbose=10
    ).fit(data_config['data_path'])


class RemoteLoader:
    """
    Load data from online sources like Zenodo, GitHub, or local files.

    Parameters
    ----------
    zenodo_record : str, optional
        A Zenodo digital object identifier (DOI) or filepath to a Zenodo record.
    content_url : str, optional
        URL to the repository user content. For GitHub, it can be in the format
        'https://raw.githubusercontent.com/user/repo/branch/'.
    repo_url : str, optional
        URL for the repository hosting the project.
    tgz_file : str, optional
        If data is saved in a TGZ file format, provide the URL to fetch the data.
    blobcontent_url : str, optional
        Root URL to blob content for accessing raw data in GitHub.
    zip_or_rar_file : str, optional
        If data is in a ZIP or RAR file, provide the file name.
    csv_file : str, optional
        Path to the main CSV file to retrieve in the record.
    verbose : int, optional
        Level of verbosity. Higher values mean more messages (default is 0).

    Examples
    --------
    >>> from gofast.datasets.rload import RemoteLoader 
    >>> loader = RemoteLoader(
            zenodo_record='10.5281/zenodo.5571534',
            content_url='https://raw.githubusercontent.com/WEgeophysics/gofast/master/',
            repo_url='https://github.com/WEgeophysics/gofast',
            tgz_file='https://raw.githubusercontent.com/WEgeophysics/gofast/master/data/__tar.tgz/fmain.bagciv.data.tar.gz',
            blobcontent_url='https://github.com/WEgeophysics/gofast/blob/master/',
            zip_or_rar_file='BagoueCIV__dataset__main.rar',
            csv_file='/__tar.tgz_files__/___fmain.bagciv.data.csv',
            verbose=10
        )
    >>> loader.fit('data/geodata/main.bagciv.data.csv')
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

    def fit(self, f: str = None) -> 'RemoteLoader':
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
        >>> loader.fit('data/geodata/main.bagciv.data.csv')
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

        return self

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
