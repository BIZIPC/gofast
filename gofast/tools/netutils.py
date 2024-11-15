# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Network utilities module for handling network operations, URL validation, 
and data fetching.

This module includes functions for downloading data, tracking download 
progress, validating URLs, and fetching JSON data from APIs or web resources.
"""
import os
import zipfile
import tarfile
import sys
import re
import json 
import shutil
import warnings 
from six.moves import urllib 
from typing import Optional, List

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Flag to indicate if 'rarfile' is available
try:
    import rarfile
    RAR_AVAILABLE = True
except ImportError:
    RAR_AVAILABLE = False
    
from .._gofastlog import gofastlog 
from ..api.property import BaseClass
from ..compat.sklearn import validate_params 
from ..decorators import EnsureMethod
from .depsutils import is_installing, is_module_installed, ensure_pkg


__all__= [
    "ArchiveExtractor", 
    "RemoteDataLoader", 
    "download_progress_hook", 
    "url_checker", 
    "validate_url", 
    "validate_url_by_validators", 
    "fetch_json_data_from_url"
]

# Initialize logging for the extractor
logger = gofastlog.get_gofast_logger(__name__)


@EnsureMethod("fetch", error="ignore", mode="soft")
class RemoteDataLoader(BaseClass):
    """
    A robust class for downloading datasets from remote sources, with
    support for Zenodo, GitHub, and other URLs, along with caching,
    extraction, and SSL verification options.

    Parameters
    ----------
    source_url : str
        The URL of the remote dataset to download.

    destination_path : str
        The local path to save the downloaded file.

    extract : bool, default=False
        If ``True``, extracts the downloaded file if it’s a supported
        archive format (e.g., ``.zip``, ``.tar.gz``).

    overwrite : bool, default=False
        If ``True``, overwrites the existing file if it already exists
        in ``destination_path``.

    progress : bool, default=True
        If ``True``, displays a progress bar during download if `tqdm`
        is available.

    verify_ssl : bool, default=True
        If ``True``, verifies SSL certificates during download. Disable
        for self-signed certificates.

    cache : bool, default=False
        If ``True``, stores the downloaded file in a cache directory
        for reuse, reducing redundant downloads.

    cache_dir : str, default='.cache'
        Directory for storing cached files if ``cache=True``.

    Attributes
    ----------
    destination_path_ : str
        The resolved destination path after downloading and extraction.

    Methods
    -------
    run()
        Executes the download and extraction process.

    clear_cache()
        Clears the cache directory, removing all cached files.

    Examples
    --------
    >>> from gofast.tools.netutils import RemoteDataLoader
    >>> loader = RemoteDataLoader(
    ...     source_url='https://example.com/data.zip',
    ...     destination_path='data/data.zip',
    ...     extract=True,
    ...     cache=True
    ... )
    >>> loader.run()

    Notes
    -----
    The `run` method must be called before invoking any other methods.
    This class supports downloading from any URL accessible via HTTP
    or HTTPS. If the dataset is an archive, it can be automatically
    extracted if `extract` is set to ``True``. The `requests` package
    is required for downloading datasets.

    See Also
    --------
    requests.get : Sends a GET request.
    shutil.copy2 : Copies a file preserving metadata.
    zipfile.ZipFile : Tools for working with ZIP archives.
    tarfile.TarFile : Read and write tar archive files.

    References
    ----------
    .. [1] `Requests: HTTP for Humans <https://docs.python-requests.org/>`_
    .. [2] `shutil — High-level file operations <https://docs.python.org/3/library/shutil.html>`_
    .. [3] `zipfile — Work with ZIP archives <https://docs.python.org/3/library/zipfile.html>`_
    .. [4] `tarfile — Read and write tar archive files <https://docs.python.org/3/library/tarfile.html>`_
    """

    @validate_params({
        'source_url': [str],
        'destination_path': [str],
        'extract': [bool],
        'overwrite': [bool],
        'progress': [bool],
        'verify_ssl': [bool],
        'cache': [bool],
        'cache_dir': [str],
    })
    def __init__(
        self,
        source_url: str,
        destination_path: str,
        extract: bool = False,
        overwrite: bool = False,
        progress: bool = True,
        verify_ssl: bool = True,
        cache: bool = False,
        cache_dir: str = ".cache"
    ):
        self.source_url = source_url
        self.destination_path = destination_path
        self.extract = extract
        self.overwrite = overwrite
        self.progress = progress and TQDM_AVAILABLE
        self.verify_ssl = verify_ssl
        self.cache = cache
        self.cache_dir = cache_dir
        self._is_runned = False  # Internal flag to indicate if run has been called

        # Ensure cache directory exists if caching is enabled
        if self.cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.debug(f"Cache directory created at {self.cache_dir}")

    def fetch(self):
        """
        Executes or run the download and extraction process.

        This method downloads the dataset from the specified source URL,
        utilizing caching if enabled, and extracts the file if specified.

        Returns
        -------
        self : RemoteDataLoader
            Returns self.

        Examples
        --------
        >>> loader = RemoteDataLoader(...)
        >>> loader.run()

        Notes
        -----
        The `run` method must be called before invoking any other methods.
        It sets up the necessary state for the object.
        """
        self._fetch()
        self._is_runned = True  # Mark as runned
        
        return self 

    def clear_cache(self):
        """
        Clears the cache directory, removing all cached files.

        Examples
        --------
        >>> loader = RemoteDataLoader(...)
        >>> loader.clear_cache()
        """
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Cleared cache at {self.cache_dir}")
        else:
            logger.warning(f"No cache found at {self.cache_dir}")

    def _fetch(self):
        """Private method to handle downloading and extraction."""
        # Use ensure_pkg to ensure 'requests' is installed
        self._download_dataset()
        if self.extract:
            self._extract_file()
        else:
            self.destination_path_ = self.destination_path

    @ensure_pkg(
        "requests",
        extra="The 'requests' package is required to download data.",
        auto_install=False
    )
    def _download_dataset(self):
        """Downloads the dataset, using cache if enabled."""
        import requests  # noqa Ensure 'requests' is imported

        cached_file_path = os.path.join(
            self.cache_dir, os.path.basename(self.destination_path)
        )

        if self.cache and os.path.exists(cached_file_path) and not self.overwrite:
            logger.info(f"Loading dataset from cache at {cached_file_path}")
            shutil.copy2(cached_file_path, self.destination_path)
        else:
            if os.path.exists(self.destination_path) and not self.overwrite:
                logger.info(f"File {self.destination_path} already exists; skipping download.")
            else:
                logger.info(f"Downloading dataset from {self.source_url}")
                self._download_file(self.source_url, self.destination_path)
            if self.cache:
                shutil.copy2(self.destination_path, cached_file_path)
                logger.debug(f"Cached dataset at {cached_file_path}")

    @ensure_pkg(
        "requests",
        extra="The 'requests' package is required to download data.",
        auto_install=False
    )
    def _download_file(self, url: str, dest_path: str):
        """Handles the download of a file with optional progress tracking."""
        import requests  # Ensure 'requests' is imported

        response = requests.get(url, stream=True, verify=self.verify_ssl)
        response.raise_for_status()  # Raise an error on bad status codes

        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 1024  # Download in 1 KB chunks
        progress_bar = tqdm(
            total=total_size, unit='B', unit_scale=True,
            desc="Downloading"
        ) if self.progress else None

        with open(dest_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # Filter out keep-alive chunks
                    file.write(chunk)
                    if progress_bar:
                        progress_bar.update(len(chunk))

        if progress_bar:
            progress_bar.close()

        logger.debug(f"Downloaded {url} to {dest_path}")

    def _extract_file(self):
        """Extracts the downloaded file if it is an archive (zip, tar)."""
        extract_dir = os.path.dirname(self.destination_path)
        if zipfile.is_zipfile(self.destination_path):
            with zipfile.ZipFile(self.destination_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                logger.debug(f"Extracted zip archive at {self.destination_path}")
        elif tarfile.is_tarfile(self.destination_path):
            with tarfile.open(self.destination_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_dir)
                logger.debug(f"Extracted tar archive at {self.destination_path}")
        else:
            logger.warning(f"No extraction performed; unsupported archive format: {self.destination_path}")
            self.destination_path_ = self.destination_path  # No extraction, use original path
            return
        # Update destination_path_ to extraction directory
        self.destination_path_ = extract_dir


@EnsureMethod("extract", error="ignore", mode="soft")
class ArchiveExtractor(BaseClass):
    """
    A class for extracting archive files of various formats.

    Supports extraction of `.zip`, `.tar`, `.tar.gz`, `.tgz`, and
    `.rar` archives.

    Parameters
    ----------
    archive_path : str
        The path to the archive file to extract.

    extract_to : str, optional
        The directory to extract the files to. If ``None``, files are
        extracted to a directory with the same name as the archive
        (without extension). Default is ``None``.

    overwrite : bool, optional
        If ``True``, existing files will be overwritten. If ``False``,
        existing files are skipped. Default is ``False``.

    file_types : list of str, optional
        A list of file extensions to extract (e.g., ``['.txt', '.csv']``).
        If ``None``, all files are extracted. Default is ``None``.

    include_hidden : bool, optional
        If ``True``, hidden files are included in the extraction. If
        ``False``, hidden files are skipped. Default is ``False``.

    show_progress : bool, optional
        If ``True``, displays a progress bar during extraction.
        Requires the `tqdm` package to be installed. Default is
        ``False``.

    Attributes
    ----------
    extract_to : str
        The directory where files are extracted.

    Methods
    -------
    extract()
        Extracts the archive based on its format.

    Examples
    --------
    >>> from gofast.tools.netutils import ArchiveExtractor
    >>> extractor = ArchiveExtractor('sample.zip', extract_to='output_dir',
    ...                              overwrite=True)
    >>> extractor.extract()

    Notes
    -----
    This class supports extraction of `.zip`, `.tar`, `.tar.gz`, `.tgz`,
    and `.rar` archives. The `.rar` format requires the `rarfile` package
    to be installed.

    See Also
    --------
    shutil.unpack_archive : Function to unpack an archive.

    References
    ----------
    .. [1] Python documentation on zipfile module.
    .. [2] Python documentation on tarfile module.
    """

    @validate_params({
        'archive_path': [str],
        'extract_to': [str, None],
        'overwrite': [bool],
        'file_types': [list, None],
        'include_hidden': [bool],
        'show_progress': [bool]
    })
    def __init__(
            self, archive_path: str, extract_to: Optional[str] = None,
            overwrite: bool = False,
            file_types: Optional[List[str]] = None, include_hidden: bool = False,
            show_progress: bool = False):
        self.archive_path = archive_path
        self.extract_to = extract_to or os.path.splitext(archive_path)[0]
        self.overwrite = overwrite
        self.file_types = file_types
        self.include_hidden = include_hidden
        self.show_progress = show_progress and TQDM_AVAILABLE

        # Ensure the extraction directory exists
        os.makedirs(self.extract_to, exist_ok=True)
        logger.debug(f"Initialized ArchiveExtractor with archive: {self.archive_path}, "
                     f"extract_to: {self.extract_to}, overwrite: {self.overwrite}")

    def extract(self):
        """
        Extracts the archive based on its format.

        This method determines the archive format and calls the
        appropriate extraction method.

        Raises
        ------
        RuntimeError
            If extraction fails.

        Examples
        --------
        >>> extractor = ArchiveExtractor('sample.zip')
        >>> extractor.extract()
        """
        try:
            extract_method = self._check_archive_format()
            logger.info(f"Starting extraction of {self.archive_path} to {self.extract_to}")
            extract_method()
            logger.info(f"Extraction complete for {self.archive_path}")
        except Exception as e:
            logger.error(f"Failed to extract {self.archive_path}: {str(e)}")
            raise RuntimeError(f"Extraction failed: {str(e)}") from e

    def _check_archive_format(self):
        """Determines the archive type and returns the appropriate
        extraction method."""
        if zipfile.is_zipfile(self.archive_path):
            return self._extract_zip
        elif tarfile.is_tarfile(self.archive_path):
            return self._extract_tar
        elif RAR_AVAILABLE and rarfile.is_rarfile(self.archive_path):
            return self._extract_rar
        else:
            raise ValueError(f"Unsupported archive format: {self.archive_path}")

    def _extract_zip(self):
        """Extracts `.zip` archives."""
        with zipfile.ZipFile(self.archive_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            members = self._filter_members(members)
            iterator = tqdm(members, desc="Extracting ZIP"
                            ) if self.show_progress else members
            for member in iterator:
                self._safe_extract(zip_ref, member)

    def _extract_tar(self):
        """Extracts `.tgz` and `.tar.gz` archives."""
        with tarfile.open(self.archive_path, 'r:*') as tar_ref:
            members = tar_ref.getmembers()
            members = self._filter_members(members)
            iterator = tqdm(members, desc="Extracting TAR"
                            ) if self.show_progress else members
            for member in iterator:
                self._safe_extract(tar_ref, member)

    @ensure_pkg(
        "rarfile",
        extra="The 'rarfile' package is required to extract .rar archives.",
        auto_install=False
    )
    def _extract_rar(self):
        """Extracts `.rar` archives."""
        import rarfile  # Ensure rarfile is imported
        with rarfile.RarFile(self.archive_path, 'r') as rar_ref:
            members = rar_ref.namelist()
            members = self._filter_members(members)
            iterator = tqdm(members, desc="Extracting RAR"
                            ) if self.show_progress else members
            for member in iterator:
                self._safe_extract(rar_ref, member)

    def _safe_extract(self, archive, member):
        """Safely extracts a single file from the archive, handling
        overwrites and logging."""
        target_path = os.path.join(
            self.extract_to,
            member if isinstance(member, str) else member.name
        )

        # Skip extraction if file exists and overwrite is False
        if os.path.exists(target_path) and not self.overwrite:
            logger.info(f"File {target_path} already exists; skipping.")
            return

        # Extract the file
        try:
            if isinstance(archive, zipfile.ZipFile):
                archive.extract(member, self.extract_to)
            elif isinstance(archive, tarfile.TarFile):
                archive.extract(member, self.extract_to)
            elif 'rarfile' in sys.modules and isinstance(archive, rarfile.RarFile):
                archive.extract(member, self.extract_to)
            logger.debug(f"Extracted {member} to {target_path}")
        except Exception as e:
            logger.warning(f"Could not extract {member} due to: {str(e)}")

    def _filter_members(self, members):
        """Filters members of the archive based on file type, hidden
        status, and other criteria."""
        filtered_members = []
        for member in members:
            # Handle tarfile member structure
            name = member if isinstance(member, str) else member.name

            # Check for file type
            if self.file_types and not any(name.endswith(ext) for ext in self.file_types):
                continue

            # Skip hidden files if not included
            if not self.include_hidden and os.path.basename(name).startswith('.'):
                continue

            filtered_members.append(member)

        return filtered_members


def url_checker (url: str , install:bool = False, 
                 raises:str ='ignore')-> bool : 
    """
    check whether the URL is reachable or not. 
    
    function uses the requests library. If not install, set the `install`  
    parameter to ``True`` to subprocess install it. 
    
    Parameters 
    ------------
    url: str, 
        link to the url for checker whether it is reachable 
    install: bool, 
        Action to install the 'requests' module if module is not install yet.
    raises: str 
        raise errors when url is not recheable rather than returning ``0``.
        if `raises` is ``ignore``, and module 'requests' is not installed, it 
        will use the django url validator. However, the latter only assert 
        whether url is right but not validate its reachability. 
              
    Returns
    --------
        ``True``{1} for reacheable and ``False``{0} otherwise. 
        
    Example
    ----------
    >>> from gofast.tools.netutils import url_checker 
    >>> url_checker ("http://www.example.com")
    ...  0 # not reacheable 
    >>> url_checker ("https://gofast.readthedocs.io/en/latest/api/gofast.html")
    ... 1 
    
    """
    isr =0 ; success = False 
    
    regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        #domain...
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )
    
    try : 
        import requests 
    except ImportError: 
        if install: 
            success  = is_installing('requests', DEVNULL=True) 
        if not success: 
            if raises=='raises': 
                raise ModuleNotFoundError(
                    "auto-installation of 'requests' failed."
                    " Install it mannually.")
                
    else : success=True  
    
    if success: 
        try:
            get = requests.get(url) #Get Url
            if get.status_code == 200: # if the request succeeds 
                isr =1 # (f"{url}: is reachable")
                
            else:
                warnings.warn(
                    f"{url}: is not reachable, status_code: {get.status_code}")
                isr =0 
        
        except requests.exceptions.RequestException as e:
            if raises=='raises': 
                raise SystemExit(f"{url}: is not reachable \nErr: {e}")
            else: isr =0 
            
    if not success : 
        # use django url validation regex
        # https://github.com/django/django/blob/stable/1.3.x/django/core/validators.py#L45
        isr = 1 if re.match(regex, url) is not None else 0 
        
    return isr 

def download_progress_hook(t):
    """
    Hook to update the tqdm progress bar during a download.

    Parameters
    ----------
    t : tqdm
        An instance of tqdm to update as the download progresses.
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        Updates the progress bar.

        Parameters
        ----------
        b : int
            Number of blocks transferred so far [default: 1].
        bsize : int
            Size of each block (in tqdm-compatible units) [default: 1].
        tsize : int, optional
            Total size (in tqdm-compatible units). If None, remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to 

def fetch_json_data_from_url (url:str , todo:str ='load'): 
    """ Retrieve JSON data from url 
    :param url: Universal Resource Locator .
    :param todo:  Action to perform with JSON:
        - load: Load data from the JSON file 
        - dump: serialize data from the Python object and create a JSON file
    """
    with urllib.request.urlopen(url) as jresponse :
        source = jresponse.read()
    data = json.loads(source)
    if todo .find('load')>=0:
        todo , json_fn  ='loads', source 
        
    if todo.find('dump')>=0:  # then collect the data and dump it
        # set json default filename 
        todo, json_fn = 'dumps',  '_urlsourcejsonf.json'  
        
    return todo, json_fn, data 


def validate_url(url: str) -> bool:
    """
    Check if the provided string is a valid URL.

    Parameters
    ----------
    url : str
        The string to be checked as a URL.

    Raises
    ------
    ValueError
        If the provided string is not a valid URL.

    Returns
    -------
    bool
        True if the URL is valid, False otherwise.

    Examples
    --------
    >>> validate_url("https://www.example.com")
    True
    >>> validate_url("not_a_url")
    ValueError: The provided string is not a valid URL.
    """
    from urllib.parse import urlparse
    
    if is_module_installed("validators"): 
        return validate_url_by_validators (url)
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError("The provided string is not a valid URL.")
    return True

def validate_url_by_validators(url: str):
    """
    Check if the provided string is a valid URL using `validators` packages.

    Parameters
    ----------
    url : str
        The string to be checked as a URL.

    Raises
    ------
    ValueError
        If the provided string is not a valid URL.

    Returns
    -------
    bool
        True if the URL is valid, False otherwise.

    Examples
    --------
    >>> validate_url("https://www.example.com")
    True
    >>> validate_url("not_a_url")
    ValueError: The provided string is not a valid URL.
    """
    import validators
    if not validators.url(url):
        raise ValueError("The provided string is not a valid URL.")
    return True

