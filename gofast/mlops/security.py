# -*- coding: utf-8 -*-
#   Licence: BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Incorporate security best practices into machine learning workflows, 
ensuring models and data are protected throughout their lifecycle.
"""
import os
import hashlib
import hmac
import json
import threading
import zlib
from numbers import Integral 
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable

from sklearn.utils._param_validation import StrOptions

from ._config import INSTALL_DEPENDENCIES, USE_CONDA 
from ..api.property import BaseLearner
from ..compat.sklearn import validate_params, Interval 
from ..decorators import ( 
    smartFitRun, 
    RunReturn, 
    EnsureFileExists
   )
from ..tools.depsutils import ensure_pkg 
from ..tools.validator import check_is_runned

from .._gofastlog import gofastlog 
logger=gofastlog.get_gofast_logger(__name__)

__all__ = [
    'BaseSecurity',
    'DataEncryption',
    'ModelProtection',
    'SecureDeployment',
    'AuditTrail',
    'AccessControl'
]

@smartFitRun
class BaseSecurity(BaseLearner):
    """
    Provides security functionalities such as data encryption,
    decryption, and secure logging. Supports key management,
    encryption algorithms, and secure log handling.

    Parameters
    ----------
    encryption_key : bytes or None, default=None
        The encryption key to use. If `None`, a new key is generated
        using ``Fernet.generate_key()``.

    log_file : str, default='security_log.json'
        The path to the log file where security events are recorded.

    encryption_algorithm : {'fernet', 'sha256'}, default='fernet'
        The encryption algorithm to use. Options are `'fernet'` for
        symmetric encryption or `'sha256'` for hashing.

    log_encryption_enabled : bool, default=False
        If `True`, the log entries are encrypted using the specified
        encryption algorithm.

    log_retention_days : int or None, default=None
        The number of days to retain logs. If `None`, logs are not
        pruned based on age.
        
    initialize_cipher: bool, default=True 
        Initialize cipher when calling `BaseSecurityClass`. This is turned 
        to ``False`` when subclass implements other kind of initialization. 
        
        
    Attributes
    ----------
    encryption_key_ : bytes
        The encryption key used for encryption and decryption.

    cipher_ : object
        The cipher object used for encryption and decryption.

    encryption_algorithm_ : str
        The encryption algorithm in use.

    log_encryption_enabled_ : bool
        Indicates if log encryption is enabled.

    log_file_ : str
        The path to the log file.

    log_retention_days_ : int or None
        The number of days to retain logs.

    Methods
    -------
    encrypt(data)
        Encrypts data using the selected encryption algorithm.

    decrypt(encrypted_data)
        Decrypts data using the selected encryption algorithm.

    export_key(file_path, encrypted=False)
        Exports the encryption key to a file, with optional encryption.

    import_key(file_path, encrypted=False)
        Imports the encryption key from a file, with optional decryption.

    rotate_key(new_key=None)
        Rotates the encryption key and re-encrypts the log file if necessary.

    log_event(event_type, details)
        Logs an event for security auditing, with optional encryption.

    retrieve_logs(decrypt=True)
        Retrieves the logs stored in the log file, with optional decryption.

    prune_old_logs()
        Prunes logs that exceed the log retention policy.

    backup_logs(backup_path)
        Backs up the log file to a specified path.

    Notes
    -----
    The class uses the `cryptography` library for encryption and
    decryption operations. When using the `'fernet'` algorithm, data
    is encrypted symmetrically. The `'sha256'` algorithm is used for
    hashing purposes and does not support decryption.

    Examples
    --------
    >>> from gofast.mlops.security import BaseSecurity
    >>> security = BaseSecurity()
    >>> encrypted_data = security.encrypt(b'secret data')
    >>> decrypted_data = security.decrypt(encrypted_data)
    >>> print(decrypted_data)
    b'secret data'

    See Also
    --------
    cryptography.fernet.Fernet : Symmetric encryption using Fernet.
    hashlib.sha256 : Secure hashing using SHA-256.

    References
    ----------
    .. [1] D. Eastlake and P. Jones, "US Secure Hash Algorithm 1 (SHA1)",
           RFC 3174, 2001.

    """

    @validate_params({
        'encryption_key': [bytes, str, None],
        'log_file': [str],
        'encryption_algorithm': [StrOptions({'fernet', 'sha256'})],
        'log_encryption_enabled': [bool],
        'log_retention_days': [Interval(Integral, 1, None, closed='left'), None],
    })
    def __init__(
        self,
        encryption_key: Optional[bytes] = None,
        log_file: str = 'security_log.json',
        encryption_algorithm: str = 'fernet',
        log_encryption_enabled: bool = False,
        log_retention_days: Optional[int] = None,
    ):
        super().__init__()
        self.encryption_key = encryption_key or self._generate_key()
        self.encryption_algorithm = encryption_algorithm
        self.log_file = log_file
        self.log_encryption_enabled = log_encryption_enabled
        self.log_retention_days = log_retention_days
        self.cipher = self._initialize_cipher()

    @RunReturn 
    def run (self): 
        """ Does nothing, just for API purpose"""
        
    @ensure_pkg(
        'cryptography',
        extra="The 'cryptography' package is required for encryption.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def _generate_key(self) -> bytes:
        """Generates a new encryption key using Fernet."""
        from cryptography.fernet import Fernet
        return Fernet.generate_key()

    @ensure_pkg(
        'cryptography',
        extra="The 'cryptography' package is required for encryption.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def _initialize_cipher(self):
        """Initialize cipher based on the selected encryption algorithm."""
        if self.encryption_algorithm == 'fernet':
            from cryptography.fernet import Fernet
            return Fernet(self.encryption_key)
        elif self.encryption_algorithm == 'sha256':
            # For hashing, no cipher object is needed
            return None
        else:
            raise ValueError("Unsupported encryption algorithm")

    
    @ensure_pkg(
        'cryptography',
        extra="The 'cryptography' package is required for encryption.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def encrypt(self, data: bytes) -> bytes:
        """
        Encrypts data using the selected encryption algorithm.

        Parameters
        ----------
        data : bytes
            The data to encrypt.

        Returns
        -------
        encrypted_data : bytes
            The encrypted data.

        Raises
        ------
        ValueError
            If the encryption algorithm is unsupported.

        Notes
        -----
        When using the `'fernet'` algorithm, the data is encrypted
        symmetrically using the Fernet cipher. For the `'sha256'`
        algorithm, the data is hashed using SHA-256 and cannot be
        decrypted.

        Examples
        --------
        >>> encrypted_data = security.encrypt(b'secret data')

        """
        if self.encryption_algorithm == 'fernet':
            return self.cipher.encrypt(data)
        elif self.encryption_algorithm == 'sha256':
            import hashlib
            return hashlib.sha256(data).hexdigest().encode()
        else:
            raise ValueError("Unsupported encryption algorithm for encryption")

    @ensure_pkg(
        'cryptography',
        extra="The 'cryptography' package is required for decryption.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypts data using the selected encryption algorithm.

        Parameters
        ----------
        encrypted_data : bytes
            The data to decrypt.

        Returns
        -------
        data : bytes
            The decrypted data.

        Raises
        ------
        ValueError
            If the encryption algorithm does not support decryption.

        Notes
        -----
        Decryption is only supported when using the `'fernet'`
        encryption algorithm.

        Examples
        --------
        >>> decrypted_data = security.decrypt(encrypted_data)

        """
        if self.encryption_algorithm == 'fernet':
            return self.cipher.decrypt(encrypted_data)
        else:
            raise ValueError("Decryption is only supported for Fernet algorithm")

    def export_key(self, file_path: str, encrypted: bool = False):
        """
        Exports the encryption key to a file, with optional encryption.

        Parameters
        ----------
        file_path : str
            The path to the file where the key will be saved.

        encrypted : bool, default=False
            If `True`, the key is encrypted before saving.

        Raises
        ------
        ValueError
            If encryption is requested but the algorithm does not support it.

        Notes
        -----
        When `encrypted` is `True`, the key is encrypted using the
        current cipher before being saved to the file.

        Examples
        --------
        >>> security.export_key('keyfile.key', encrypted=True)

        """
        key_data = self.encryption_key
        if encrypted:
            if self.encryption_algorithm == 'fernet':
                key_data = self.cipher.encrypt(self.encryption_key)
            else:
                raise ValueError("Key encryption is only supported for Fernet algorithm")
        with open(file_path, 'wb') as key_file:
            key_file.write(key_data)

    def import_key(self, file_path: str, encrypted: bool = False):
        """
        Imports the encryption key from a file, with optional decryption.

        Parameters
        ----------
        file_path : str
            The path to the file from which the key will be loaded.

        encrypted : bool, default=False
            If `True`, the key is decrypted after loading.

        Raises
        ------
        ValueError
            If decryption is requested but the algorithm does not support it.

        Notes
        -----
        When `encrypted` is `True`, the key is decrypted using the
        current cipher after being loaded from the file.

        Examples
        --------
        >>> security.import_key('keyfile.key', encrypted=True)

        """
        with open(file_path, 'rb') as key_file:
            key_data = key_file.read()
            if encrypted:
                if self.encryption_algorithm == 'fernet':
                    key_data = self.cipher.decrypt(key_data)
                else:
                    raise ValueError("Key decryption is only supported for Fernet algorithm")
            self.encryption_key = key_data
            self.cipher = self._initialize_cipher()

    @ensure_pkg(
        'cryptography',
        extra="The 'cryptography' package is required for key rotation.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def rotate_key(self, new_key: Optional[bytes] = None):
        """
        Rotates the encryption key and re-encrypts the log file if necessary.

        Parameters
        ----------
        new_key : bytes or None, default=None
            The new encryption key. If `None`, a new key is generated
            using ``Fernet.generate_key()``.

        Notes
        -----
        When the key is rotated and log encryption is enabled, the
        existing logs are re-encrypted using the new key.

        Examples
        --------
        >>> security.rotate_key()

        """
        old_key = self.encryption_key
        self.encryption_key = new_key or self._generate_key()
        self.cipher = self._initialize_cipher()
        logger.info("Encryption key rotated.")

        # Re-encrypt logs if enabled
        if self.log_encryption_enabled:
            self._reencrypt_logs(old_key)

    @ensure_pkg(
        'cryptography',
        extra="The 'cryptography' package is required for re-encrypting logs.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def _reencrypt_logs(self, old_key: bytes):
        """
        Re-encrypts the log file with the new key.

        Parameters
        ----------
        old_key : bytes
            The old encryption key used to decrypt the existing logs.

        Notes
        -----
        This method is called internally when the encryption key is rotated
        and `log_encryption_enabled` is `True`.

        """
        from cryptography.fernet import Fernet
        old_cipher = Fernet(old_key)
        pruned_logs = []
        with open(self.log_file, 'rb') as f:
            for line in f:
                line_data = line.strip()
                # Decrypt using old cipher
                decrypted_line = old_cipher.decrypt(line_data)
                # Re-encrypt using new cipher
                reencrypted_line = self.encrypt(decrypted_line)
                pruned_logs.append(reencrypted_line + b'\n')
        with open(self.log_file, 'wb') as f:
            f.writelines(pruned_logs)
        logger.info("Re-encrypted logs with new encryption key.")

    def log_event(self, event_type: str, details: Dict[str, Any]):
        """
        Logs an event for security auditing, with optional encryption.

        Parameters
        ----------
        event_type : str
            The type of the event to log.

        details : dict of str to Any
            Additional details about the event.

        Notes
        -----
        The log entry includes a timestamp in ISO format, the event type,
        and the event details. If log encryption is enabled, the log
        entry is encrypted before being written to the log file.

        Examples
        --------
        >>> security.log_event('login_attempt', {'user': 'admin'})

        """
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'details': details
        }
        log_data = json.dumps(log_entry).encode()
        if self.log_encryption_enabled:
            log_data = self.encrypt(log_data)

        with open(self.log_file, 'ab') as f:
            f.write(log_data + b'\n')

        logger.info(f"Logged event: {event_type} - {details}")

    def retrieve_logs(self, decrypt: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieves the logs stored in the log file, with optional decryption.

        Parameters
        ----------
        decrypt : bool, default=True
            If `True`, decrypts the log entries if they are encrypted.

        Returns
        -------
        logs : list of dict
            A list of log entries.

        Raises
        ------
        ValueError
            If decryption is requested but the algorithm does not support it.

        Notes
        -----
        If log encryption is enabled and `decrypt` is `True`, the log
        entries are decrypted before being parsed.

        Examples
        --------
        >>> logs = security.retrieve_logs()

        """
        logs = []
        with open(self.log_file, 'rb') as f:
            for line in f:
                line = line.strip()
                if self.log_encryption_enabled and decrypt:
                    if self.encryption_algorithm == 'fernet':
                        line = self.decrypt(line)
                    else:
                        raise ValueError(
                            "Decryption is only supported for Fernet algorithm")
                logs.append(json.loads(line))
        return logs

    def prune_old_logs(self):
        """
        Prunes logs that exceed the log retention policy.

        Notes
        -----
        Removes log entries that are older than the retention period
        specified in `log_retention_days_`. If `log_retention_days_`
        is `None`, no pruning is performed.

        Examples
        --------
        >>> security.prune_old_logs()

        """
        if not self.log_retention_days:
            return
        cutoff_date = datetime.utcnow() - timedelta(days=self.log_retention_days)
        pruned_logs = []
        with open(self.log_file, 'rb') as f:
            for line in f:
                line_data = line.strip()
                if self.log_encryption_enabled:
                    if self.encryption_algorithm == 'fernet':
                        line_data = self.decrypt(line_data)
                    else:
                        raise ValueError(
                            "Decryption is only supported for Fernet algorithm")
                log = json.loads(line_data)
                log_date = datetime.fromisoformat(log['timestamp'])
                if log_date >= cutoff_date:
                    if self.log_encryption_enabled:
                        line_to_write = self.encrypt(json.dumps(log).encode()) + b'\n'
                    else:
                        line_to_write = json.dumps(log).encode() + b'\n'
                    pruned_logs.append(line_to_write)
        with open(self.log_file, 'wb') as f:
            f.writelines(pruned_logs)
        logger.info(f"Pruned logs older than {self.log_retention_days} days.")

    @validate_params({"backup_path": [str]})
    def backup_logs(self, backup_path: str):
        """
        Backs up the log file to a specified path.

        Parameters
        ----------
        backup_path : str
            The destination path for the backup file.

        Notes
        -----
        The method creates the backup directory if it does not exist.

        Examples
        --------
        >>> security.backup_logs('backups/security_log_backup.json')

        """
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        with open(self.log_file, 'rb') as src, open(backup_path, 'wb') as dst:
            dst.write(src.read())
        logger.info(f"Backed up logs to {backup_path}")

@smartFitRun
class DataEncryption(BaseSecurity):
    """
    A class for handling data encryption and decryption with optional
    compression, supporting various encryption algorithms.

    Parameters
    ----------
    encryption_algorithm : {'fernet', 'aes', 'chacha20'}, default='fernet'
        The encryption algorithm to use. Supported algorithms include
        `'fernet'`, `'aes'`, and `'chacha20'`.

    compression : bool, default=False
        If `True`, the data is compressed before encryption and
        decompressed after decryption.

    encryption_key : str or None, default=None
        The key used for encryption. Required for certain encryption
        algorithms like `'aes'` and `'chacha20'`. If `None`, a new key
        is generated automatically.

    Attributes
    ----------
    encryption_algorithm : str
        The algorithm used for encryption.

    compression : bool
        Indicates whether data compression is enabled.

    encryption_key_ : bytes
        The encryption key used for encryption and decryption.

    is_runned_ : bool
        Indicates whether the `run` method has been called.

    Methods
    -------
    run()
        Initializes the encryption system based on the specified algorithm.

    encrypt(data)
        Encrypts data using the specified encryption algorithm.

    decrypt(encrypted_data)
        Decrypts data using the specified encryption algorithm.

    encrypt_file(file_path, output_path)
        Encrypts a file and saves the encrypted content to a new file.

    decrypt_file(encrypted_file_path, output_path)
        Decrypts an encrypted file and saves the decrypted content to a new file.

    change_encryption_algorithm(algorithm)
        Changes the encryption algorithm used by the class.

    Notes
    -----
    This class supports various encryption methods, including symmetric
    encryption (e.g., AES and ChaCha20) and symmetric encryption using
    the `'fernet'` algorithm.

    The `run` method must be called before using `encrypt_data` or
    `decrypt_data`. It initializes the necessary ciphers and keys.

    Examples
    --------
    >>> from gofast.mlops.security import DataEncryption
    >>> encryptor = DataEncryption(
    ...     encryption_algorithm='aes',
    ...     compression=True,
    ...     encryption_key='my_secret_key_1234567890123456'
    ... )
    >>> encryptor.run()
    >>> encrypted_data = encryptor.encrypt_data(b"Secret data")
    >>> decrypted_data = encryptor.decrypt_data(encrypted_data)
    >>> print(decrypted_data)
    b'Secret data'

    See Also
    --------
    BaseSecurity : The base class that provides core security functionality.

    References
    ----------
    .. [1] D. Eastlake and P. Jones, "US Secure Hash Algorithm 1 (SHA1)",
           RFC 3174, 2001.

    """

    @validate_params({
        'encryption_algorithm': [StrOptions({'fernet', 'aes', 'chacha20'})],
        'compression': [bool],
        'encryption_key': [str, None]
    })
    def __init__(
        self,
        encryption_algorithm: str = 'fernet',
        compression: bool = False,
        encryption_key: Optional[str] = None
    ):
        self.encryption_algorithm = encryption_algorithm
        self.compression = compression
        self.encryption_key = encryption_key
        
        self.encryption_key_ = None
        self.is_runned_ = False
        self.cipher_ = None
        
        self.log_file = 'security_log.json'
        self.log_encryption_enabled=False
  

    @RunReturn 
    def run(self):
        """
        Initializes the encryption system based on the specified algorithm.

        Notes
        -----
        This method must be called before using `encrypt_data` or
        `decrypt_data`. It initializes the necessary ciphers and keys.

        Raises
        ------
        ValueError
            If the encryption key is invalid for the chosen algorithm.

        """
        # Generate key if not provided
        if self.encryption_key is None:
            self.encryption_key_ = self._generate_key()
        else:
            self.encryption_key_ = self.encryption_key.encode()

        self._initialize_cipher()
        self.is_runned_ = True

    @ensure_pkg(
        'cryptography',
        extra="The 'cryptography' package is required for encryption.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def _initialize_cipher(self):
        """
        Initializes cipher based on the selected encryption algorithm.

        Raises
        ------
        ValueError
            If the encryption algorithm is unsupported or if the encryption
            key is invalid.

        """
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend
        from cryptography.fernet import Fernet

        if self.encryption_algorithm == 'fernet':
            self.cipher_ = Fernet(self.encryption_key_)
        elif self.encryption_algorithm == 'aes':
            if len(self.encryption_key_) not in (16, 24, 32):
                raise ValueError(
                    "Invalid AES key size. Key must be 16, 24, or 32 bytes long."
                )
            self.cipher_ = Cipher(
                algorithms.AES(self.encryption_key_),
                modes.CFB(b'0' * 16),
                backend=default_backend()
            )
        elif self.encryption_algorithm == 'chacha20':
            if len(self.encryption_key_) != 32:
                raise ValueError(
                    "ChaCha20 key must be 32 bytes long."
                )
            nonce = b'0' * 16  # In production, use a secure random nonce
            self.cipher_ = Cipher(
                algorithms.ChaCha20(self.encryption_key_, nonce),
                mode=None,
                backend=default_backend()
            )
        else:
            raise ValueError(
                f"Unsupported encryption algorithm: {self.encryption_algorithm}"
            )

    @ensure_pkg(
        'cryptography',
        extra="The 'cryptography' package is required for encryption.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def _generate_key(self) -> bytes:
        """
        Generates a new encryption key.

        Returns
        -------
        key : bytes
            The generated encryption key.

        """
        from cryptography.fernet import Fernet
        return Fernet.generate_key()

    def _apply_compression(self, data: bytes) -> bytes:
        """Apply compression to the data if compression is enabled."""
        if self.compression:
            import zlib
            return zlib.compress(data)
        return data

    def _apply_decompression(self, data: bytes) -> bytes:
        """Decompress the data if compression is enabled."""
        if self.compression:
            import zlib
            return zlib.decompress(data)
        return data

    def encrypt(self, data: bytes) -> bytes:
        """
        Encrypts data using the specified encryption algorithm.

        Parameters
        ----------
        data : bytes
            The data to be encrypted.

        Returns
        -------
        encrypted_data : bytes
            The encrypted data.

        Raises
        ------
        ValueError
            If the encryption system is not initialized or if an
            unsupported encryption algorithm is specified.

        Notes
        -----
        The `run` method must be called before using this method.

        """
        check_is_runned(
            self,
            msg="Encryption system is not initialized. Call `run` first."
        )
        data = self._apply_compression(data)

        if self.encryption_algorithm == 'fernet':
            encrypted_data = self.cipher_.encrypt(data)
        else:
            encryptor = self.cipher_.encryptor()
            encrypted_data = encryptor.update(data) + encryptor.finalize()

        self.log_event('data_encryption', {'data_length': len(data)})
        return encrypted_data

    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypts data using the specified encryption algorithm.

        Parameters
        ----------
        encrypted_data : bytes
            The encrypted data to be decrypted.

        Returns
        -------
        data : bytes
            The decrypted data.

        Raises
        ------
        ValueError
            If the encryption system is not initialized or if an
            unsupported encryption algorithm is specified.

        Notes
        -----
        The `run` method must be called before using this method.

        """
        check_is_runned(
            self,
            msg="Encryption system is not initialized. Call `run` first."
        )

        if self.encryption_algorithm == 'fernet':
            decrypted_data = self.cipher_.decrypt(encrypted_data)
        else:
            decryptor = self.cipher_.decryptor()
            decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()

        decrypted_data = self._apply_decompression(decrypted_data)
        self.log_event('data_decryption', {'data_length': len(decrypted_data)})
        return decrypted_data

    def encrypt_file(self, file_path: str, output_path: str):
        """
        Encrypts a file and saves the encrypted content to a new file.

        Parameters
        ----------
        file_path : str
            The path of the file to be encrypted.

        output_path : str
            The path where the encrypted file will be saved.

        Notes
        -----
        The `run` method must be called before using this method.

        Examples
        --------
        >>> encryptor.encrypt_file('plain.txt', 'encrypted.dat')

        """
        check_is_runned(
            self,
            msg="Encryption system is not initialized. Call `run` first."
        )
        with open(file_path, 'rb') as f:
            file_data = f.read()
        encrypted_data = self.encrypt_data(file_data)
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
        self.log_event(
            'file_encryption',
            {'file': file_path, 'output': output_path}
        )
 
    @validate_params({"output_path": [str]})
    @EnsureFileExists
    def decrypt_file(self, encrypted_file_path: str, output_path: str):
        """
        Decrypts an encrypted file and saves the decrypted content to a new file.

        Parameters
        ----------
        encrypted_file_path : str
            The path of the encrypted file.

        output_path : str
            The path where the decrypted file will be saved.

        Notes
        -----
        The `run` method must be called before using this method.

        Examples
        --------
        >>> encryptor.decrypt_file('encrypted.dat', 'decrypted.txt')

        """
        check_is_runned(
            self,
            msg="Encryption system is not initialized. Call `run` first."
        )
        with open(encrypted_file_path, 'rb') as f:
            encrypted_data = f.read()
        decrypted_data = self.decrypt_data(encrypted_data)
        with open(output_path, 'wb') as f:
            f.write(decrypted_data)
        self.log_event(
            'file_decryption',
            {'file': encrypted_file_path, 'output': output_path}
        )

    def change_encryption_algorithm(self, algorithm: str):
        """
        Changes the encryption algorithm used by the class.

        Parameters
        ----------
        algorithm : {'fernet', 'aes', 'chacha20'}
            The new encryption algorithm to use.

        Raises
        ------
        ValueError
            If an unsupported algorithm is specified.

        Notes
        -----
        After changing the algorithm, you must call `run` again to
        reinitialize the encryption system.

        Examples
        --------
        >>> encryptor.change_encryption_algorithm('chacha20')
        >>> encryptor.run()

        """
        if algorithm not in {'fernet', 'aes', 'chacha20'}:
            raise ValueError(
                "Supported algorithms are 'fernet', 'aes', and 'chacha20'."
            )
        self.encryption_algorithm = algorithm
        self.is_runned_ = False  # Need to reinitialize
        self.log_event(
            'encryption_algorithm_change',
            {'algorithm': algorithm}
        )

@smartFitRun
class ModelProtection(BaseSecurity):
    """
    A class for securing machine learning models by offering functionality
    for encryption, signing, verification, and backup. This class allows
    users to protect their models using HMAC-based signing and secure
    encryption algorithms.

    Parameters
    ----------
    secret_key : bytes or None, default=None
        A secret key used for signing the model. If not provided, a
        default key is generated using SHA-256.

    encryption_key : str or None, default=None
        The key used for encrypting the model data. If `None`, a new key
        is generated automatically.

    hash_algorithm : {'sha256', 'sha512', 'md5'}, default='sha256'
        The hash algorithm used for signing the model. Supported algorithms
        include `'sha256'`, `'sha512'`, and `'md5'`.

    compression : bool, default=False
        If `True`, the model data is compressed before encryption and
        decompressed after decryption.

    Attributes
    ----------
    secret_key_ : bytes
        The secret key used for signing and verifying the model.

    hash_algorithm_ : str
        The algorithm used for hashing and signing the model.

    compression : bool
        Indicates whether data compression is enabled.

    is_runned_ : bool
        Indicates whether the `run` method has been called.

    Methods
    -------
    run()
        Initializes the model protection system.

    sign_model(model_data)
        Signs the model data using HMAC with the selected hash algorithm.

    verify_model(model_data, signature)
        Verifies the integrity of the model by checking its signature.

    encrypt_model(model_data)
        Encrypts the model data after optional compression.

    decrypt_model(encrypted_model_data)
        Decrypts the model data and decompresses it if necessary.

    track_model_version(model_data, version)
        Tracks and logs the version of the model, returning a version ID.

    rotate_secret_key(new_secret_key=None)
        Rotates the secret key used for signing and verifying the model.

    integrity_check(model_data)
        Performs a basic integrity check on the model data.

    backup_model(model_data, backup_path)
        Backs up the model by encrypting and saving it to a file.

    restore_model(backup_path)
        Restores the model from an encrypted backup file.

    Notes
    -----
    This class extends `BaseSecurity` to provide advanced security
    features specifically for machine learning models. It ensures that
    models are encrypted, signed, and can be verified for integrity,
    supporting secure model deployment and storage.

    Examples
    --------
    >>> from gofast.mlops.security import ModelProtection
    >>> protector = ModelProtection(
    ...     secret_key=b'super_secret_key',
    ...     compression=True,
    ...     encryption_key='my_encryption_key'
    ... )
    >>> protector.run()
    >>> model_data = b"model binary data"
    >>> signature = protector.sign_model(model_data)
    >>> verified = protector.verify_model(model_data, signature)
    >>> print(f"Signature verified: {verified}")
    Signature verified: True
    >>> encrypted_model = protector.encrypt_model(model_data)
    >>> decrypted_model = protector.decrypt_model(encrypted_model)
    >>> print(decrypted_model == model_data)
    True

    See Also
    --------
    DataEncryption : Class for handling general data encryption and decryption.
    BaseSecurity : The base class that provides core security functionality.

    References
    ----------
    .. [1] "HMAC: Keyed-Hashing for Message Authentication",
           RFC 2104, 1997.

    """

    @validate_params({
        'secret_key': [bytes, None],
        'encryption_key': [str, None],
        'hash_algorithm': [StrOptions({'sha256', 'sha512', 'md5'})],
        'compression': [bool]
    })
    def __init__(
        self,
        secret_key: Optional[bytes] = None,
        encryption_key: Optional[str] = None,
        hash_algorithm: str = 'sha256',
        compression: bool = False
    ):

        super().__init__(encryption_key=encryption_key)
        self.secret_key = secret_key
        self.hash_algorithm = hash_algorithm.lower()
        self.compression = compression
        self.is_runned_ = False

    @RunReturn
    def run(self):
        """
        Initializes the model protection system.

        Notes
        -----
        This method must be called before using other methods like
        `sign_model`, `encrypt_model`, etc. It sets up the necessary keys
        and configurations.

        """
        # Initialize secret key
        if self.secret_key is None:
            self.secret_key_ = hashlib.sha256(b'default_key').digest()
        else:
            self.secret_key_ = self.secret_key

        # Validate hash algorithm
        if self.hash_algorithm not in {'sha256', 'sha512', 'md5'}:
            raise ValueError(
                f"Unsupported hash algorithm: {self.hash_algorithm}"
            )
        self.hash_algorithm_ = self.hash_algorithm
        self.is_runned_ = True

    def _apply_compression(self, data: bytes) -> bytes:
        """Compresses the data if compression is enabled."""
        if self.compression:
            return zlib.compress(data)
        return data

    def _apply_decompression(self, data: bytes) -> bytes:
        """Decompresses the data if compression is enabled."""
        if self.compression:
            return zlib.decompress(data)
        return data

    def sign_model(self, model_data: bytes) -> str:
        """
        Signs the model data using HMAC with the selected hash algorithm.

        Parameters
        ----------
        model_data : bytes
            The binary data of the model to be signed.

        Returns
        -------
        signature : str
            The hexadecimal HMAC signature of the model data.

        Raises
        ------
        ValueError
            If the model protection system is not initialized.

        Notes
        -----
        The `run` method must be called before using this method.

        Examples
        --------
        >>> signature = protector.sign_model(model_data)

        """
        check_is_runned(
            self,
            msg="Model protection system is not initialized. Call `run` first."
        )

        hash_function = getattr(hashlib, self.hash_algorithm_)
        signature = hmac.new(
            self.secret_key_, model_data, hash_function
        ).hexdigest()

        self.log_event('model_sign', {'signature': signature})
        return signature

    def verify_model(self, model_data: bytes, signature: str) -> bool:
        """
        Verifies the integrity of the model by checking its signature.

        Parameters
        ----------
        model_data : bytes
            The binary data of the model to be verified.

        signature : str
            The expected signature of the model.

        Returns
        -------
        verified : bool
            `True` if the model's signature matches, `False` otherwise.

        Raises
        ------
        ValueError
            If the model protection system is not initialized.

        Notes
        -----
        The `run` method must be called before using this method.

        Examples
        --------
        >>> verified = protector.verify_model(model_data, signature)

        """
        check_is_runned(
            self,
            msg="Model protection system is not initialized. Call `run` first."
        )

        hash_function = getattr(hashlib, self.hash_algorithm_)
        calculated_signature = hmac.new(
            self.secret_key_, model_data, hash_function
        ).hexdigest()

        verified = hmac.compare_digest(calculated_signature, signature)
        self.log_event('model_verification', {'verified': verified})
        return verified

    def encrypt_model(self, model_data: bytes) -> bytes:
        """
        Encrypts the model data after optional compression.

        Parameters
        ----------
        model_data : bytes
            The binary data of the model to be encrypted.

        Returns
        -------
        encrypted_model : bytes
            The encrypted model data.

        Raises
        ------
        ValueError
            If the model protection system is not initialized.

        Notes
        -----
        The `run` method must be called before using this method.

        Examples
        --------
        >>> encrypted_model = protector.encrypt_model(model_data)

        """
        check_is_runned(
            self,
            msg="Model protection system is not initialized. Call `run` first."
        )

        compressed_data = self._apply_compression(model_data)
        encrypted_model = self.encrypt(compressed_data)
        self.log_event('model_encryption', {'model_size': len(compressed_data)})
        return encrypted_model

    def decrypt_model(self, encrypted_model_data: bytes) -> bytes:
        """
        Decrypts the model data and decompresses it if necessary.

        Parameters
        ----------
        encrypted_model_data : bytes
            The encrypted model data to be decrypted.

        Returns
        -------
        model_data : bytes
            The decrypted model data.

        Raises
        ------
        ValueError
            If the model protection system is not initialized.

        Notes
        -----
        The `run` method must be called before using this method.

        Examples
        --------
        >>> decrypted_model = protector.decrypt_model(encrypted_model)

        """
        check_is_runned(
            self,
            msg="Model protection system is not initialized. Call `run` first."
        )

        decrypted_data = self.decrypt(encrypted_model_data)
        model_data = self._apply_decompression(decrypted_data)
        self.log_event('model_decryption', {'model_size': len(model_data)})
        return model_data

    def track_model_version(self, model_data: bytes, version: str) -> str:
        """
        Tracks and logs the version of the model, returning a version ID.

        Parameters
        ----------
        model_data : bytes
            The binary data of the model.

        version : str
            The version of the model.

        Returns
        -------
        version_id : str
            A unique version ID for the model based on its content and version.

        Notes
        -----
        This method generates a version ID by hashing the model data and
        version string.

        Examples
        --------
        >>> version_id = protector.track_model_version(model_data, 'v1.0')

        """
        version_id = hashlib.sha256(
            model_data + version.encode()
        ).hexdigest()
        self.log_event(
            'model_version_track',
            {'version': version, 'version_id': version_id}
        )
        return version_id

    def rotate_secret_key(self, new_secret_key: Optional[bytes] = None):
        """
        Rotates the secret key used for signing and verifying the model.

        Parameters
        ----------
        new_secret_key : bytes or None, default=None
            A new secret key. If not provided, a new default key will be
            generated using SHA-256.

        Notes
        -----
        After rotating the secret key, existing signatures will no longer
        be valid.

        Examples
        --------
        >>> protector.rotate_secret_key()

        """
        self.secret_key_ = new_secret_key or hashlib.sha256(
            b'new_default_key'
        ).digest()
        self.log_event('secret_key_rotation', {'new_key': True})

    def integrity_check(self, model_data: bytes) -> bool:
        """
        Performs a basic integrity check on the model data.

        Parameters
        ----------
        model_data : bytes
            The binary data of the model to be checked.

        Returns
        -------
        is_valid : bool
            `True` if the integrity check passes, `False` otherwise.

        Notes
        -----
        This method checks if the model size is within expected bounds.
        Adjust the expected size and tolerance as needed.

        Examples
        --------
        >>> is_valid = protector.integrity_check(model_data)

        """
        expected_size = 5_000_000  # Expected model size in bytes (e.g., 5MB)
        tolerance = 0.2  # 20% tolerance
        lower_bound = expected_size * (1 - tolerance)
        upper_bound = expected_size * (1 + tolerance)

        model_size = len(model_data)
        if not (lower_bound <= model_size <= upper_bound):
            self.log_event(
                'integrity_check',
                {'status': 'failed', 'size': model_size}
            )
            return False

        self.log_event(
            'integrity_check',
            {'status': 'passed', 'size': model_size}
        )
        return True

    def backup_model(self, model_data: bytes, backup_path: str):
        """
        Backs up the model by encrypting and saving it to a file.

        Parameters
        ----------
        model_data : bytes
            The binary data of the model to be backed up.

        backup_path : str
            The file path where the encrypted model backup will be saved.

        Notes
        -----
        The `run` method must be called before using this method.

        Examples
        --------
        >>> protector.backup_model(model_data, 'model_backup.dat')

        """
        check_is_runned(
            self,
            msg="Model protection system is not initialized. Call `run` first."
        )

        encrypted_model = self.encrypt_model(model_data)
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        with open(backup_path, 'wb') as backup_file:
            backup_file.write(encrypted_model)
        self.log_event('model_backup', {'backup_path': backup_path})

    def restore_model(self, backup_path: str) -> bytes:
        """
        Restores the model from an encrypted backup file.

        Parameters
        ----------
        backup_path : str
            The path to the encrypted backup file.

        Returns
        -------
        model_data : bytes
            The decrypted model data.

        Notes
        -----
        The `run` method must be called before using this method.

        Examples
        --------
        >>> model_data = protector.restore_model('model_backup.dat')

        """
        check_is_runned(
            self,
            msg="Model protection system is not initialized. Call `run` first."
        )

        with open(backup_path, 'rb') as backup_file:
            encrypted_model = backup_file.read()
        model_data = self.decrypt_model(encrypted_model)
        self.log_event('model_restore', {'backup_path': backup_path})

        return model_data

@smartFitRun
class SecureDeployment(BaseSecurity):
    """
    Manages secure deployment workflows, including token-based authentication,
    role-based access control (RBAC), IP whitelisting, and auditing. Provides
    functionality for generating, verifying, and revoking authentication tokens
    and refresh tokens.

    Parameters
    ----------
    secret_key : str
        A secret key used for signing tokens (JWT).

    auth_method : {'token', 'password'}, default='token'
        The authentication method to use.

    encryption_key : str or None, default=None
        The encryption key used for secure data transmission (if needed).

    allow_refresh_tokens : bool, default=True
        If `True`, allows the generation and verification of refresh tokens.

    token_blacklist : list of str or None, default=None
        A list of tokens that are revoked or blacklisted.

    Attributes
    ----------
    secret_key_ : str
        The secret key used to sign and verify tokens.

    auth_method_ : str
        The authentication method in use.

    allow_refresh_tokens_ : bool
        Indicates if refresh tokens are enabled.

    token_blacklist_ : list of str
        A list containing blacklisted tokens.

    is_runned_ : bool
        Indicates whether the `run` method has been called.

    Methods
    -------
    run()
        Initializes the secure deployment system.

    generate_token(user_id, expires_in=3600, roles=None, custom_claims=None)
        Generates a JWT token for a user with optional expiration and roles.

    generate_refresh_token(user_id, expires_in=86400)
        Generates a refresh token for a user.

    verify_token(token)
        Verifies the validity of a token.

    revoke_token(token)
        Revokes a token by adding it to the blacklist.

    enforce_rbac(required_roles, token)
        Enforces role-based access control (RBAC) for a user.

    ip_whitelisting(allowed_ips, current_ip)
        Checks if the current IP is in the whitelist.

    audit_event(event, metadata, ip_address=None, user_agent=None)
        Logs an audit event with optional IP address and user agent.

    Notes
    -----
    This class extends `BaseSecurity` to provide security features essential
    for deploying machine learning models securely. It supports token-based
    authentication using JSON Web Tokens (JWT), which are industry-standard
    for secure information exchange.

    The `run` method must be called before using other methods to initialize
    the secure deployment system.

    Examples
    --------
    >>> from gofast.mlops.security import SecureDeployment
    >>> deploy = SecureDeployment(secret_key="super_secret_key")
    >>> deploy.run()
    >>> token = deploy.generate_token(user_id="user123", roles=["admin"])
    >>> valid = deploy.verify_token(token)
    >>> print(f"Token is valid: {valid}")
    Token is valid: True

    See Also
    --------
    ModelProtection : Class for securing models with encryption and signing.
    DataEncryption : Class for handling general data encryption and decryption.

    References
    ----------
    .. [1] JSON Web Tokens (JWT) - RFC 7519
           https://tools.ietf.org/html/rfc7519

    """

    @validate_params({
        'secret_key': [str],
        'auth_method': [StrOptions({'token', 'password'})],
        'encryption_key': [str, None],
        'allow_refresh_tokens': [bool],
        'token_blacklist': [list, None]
    })
    def __init__(
        self,
        secret_key: str,
        auth_method: str = 'token',
        encryption_key: Optional[str] = None,
        allow_refresh_tokens: bool = True,
        token_blacklist: Optional[List[str]] = None
    ):
        """
        Initializes the SecureDeployment class with the given secret key,
        authentication method, and token handling options.

        Parameters
        ----------
        secret_key : str
            A secret key used for signing tokens (JWT).

        auth_method : {'token', 'password'}, default='token'
            The authentication method to use.

        encryption_key : str or None, default=None
            The encryption key used for secure data transmission (if needed).

        allow_refresh_tokens : bool, default=True
            If `True`, allows the generation and verification of refresh tokens.

        token_blacklist : list of str or None, default=None
            A list of tokens that are revoked or blacklisted.

        """
        super().__init__(encryption_key=encryption_key)
        self.secret_key = secret_key
        self.auth_method = auth_method
        self.allow_refresh_tokens = allow_refresh_tokens
        self.token_blacklist = token_blacklist or []
        self.is_runned_ = False

    @RunReturn
    def run(self):
        """
        Initializes the secure deployment system.

        Notes
        -----
        This method must be called before using other methods like
        `generate_token`, `verify_token`, etc. It sets up the necessary
        configurations and validates the authentication method.

        """
        if self.auth_method not in {'token', 'password'}:
            raise ValueError(
                f"Unsupported authentication method: {self.auth_method}"
            )
        self.secret_key_ = self.secret_key
        self.auth_method_ = self.auth_method
        self.allow_refresh_tokens_ = self.allow_refresh_tokens
        self.token_blacklist_ = self.token_blacklist
        self.is_runned_ = True

    @ensure_pkg(
        'pyjwt',
        extra="The 'pyjwt' package is required for token operations.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def generate_token(
        self,
        user_id: str,
        expires_in: int = 3600,
        roles: Optional[List[str]] = None,
        custom_claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generates a JWT token for a user with optional expiration and roles.

        Parameters
        ----------
        user_id : str
            The user ID for whom the token is being generated.

        expires_in : int, default=3600
            The time (in seconds) until the token expires.

        roles : list of str or None, default=None
            A list of roles to include in the token.

        custom_claims : dict of str to Any or None, default=None
            Custom claims to add to the token payload.

        Returns
        -------
        token : str
            The encoded JWT token.

        Raises
        ------
        ValueError
            If the secure deployment system is not initialized.

        Notes
        -----
        The `run` method must be called before using this method.

        Examples
        --------
        >>> token = deploy.generate_token(user_id="user123", roles=["admin"])

        """
        check_is_runned(
            self,
            msg="Secure deployment system is not initialized. Call `run` first."
        )
        import jwt

        payload = {
            'user_id': user_id,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=expires_in),
            'roles': roles or []
        }

        if custom_claims:
            payload.update(custom_claims)

        token = jwt.encode(
            payload,
            self.secret_key_,
            algorithm='HS256'
        )

        self.log_event(
            'token_generation',
            {'user_id': user_id, 'roles': roles}
        )
        return token

    @ensure_pkg(
        'pyjwt',
        extra="The 'pyjwt' package is required for token operations.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def generate_refresh_token(
        self,
        user_id: str,
        expires_in: int = 86400
    ) -> str:
        """
        Generates a refresh token for a user.

        Parameters
        ----------
        user_id : str
            The user ID for whom the refresh token is being generated.

        expires_in : int, default=86400
            The time (in seconds) until the refresh token expires.

        Returns
        -------
        refresh_token : str
            The encoded refresh token.

        Raises
        ------
        ValueError
            If refresh tokens are not enabled or the system is not initialized.

        Notes
        -----
        The `run` method must be called before using this method.

        Examples
        --------
        >>> refresh_token = deploy.generate_refresh_token(user_id="user123")

        """
        check_is_runned(
            self,
            msg="Secure deployment system is not initialized. Call `run` first."
        )
        if not self.allow_refresh_tokens_:
            raise ValueError("Refresh tokens are not enabled.")

        import jwt

        payload = {
            'user_id': user_id,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=expires_in),
            'type': 'refresh_token'
        }

        refresh_token = jwt.encode(
            payload,
            self.secret_key_,
            algorithm='HS256'
        )

        self.log_event(
            'refresh_token_generation',
            {'user_id': user_id}
        )
        return refresh_token

    @ensure_pkg(
        'pyjwt',
        extra="The 'pyjwt' package is required for token operations.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def verify_token(self, token: str) -> bool:
        """
        Verifies the validity of a token.

        Parameters
        ----------
        token : str
            The JWT token to be verified.

        Returns
        -------
        is_valid : bool
            `True` if the token is valid and not expired, `False` otherwise.

        Raises
        ------
        ValueError
            If the secure deployment system is not initialized.

        Notes
        -----
        The `run` method must be called before using this method.

        Examples
        --------
        >>> is_valid = deploy.verify_token(token)

        """
        check_is_runned(
            self,
            msg="Secure deployment system is not initialized. Call `run` first."
        )
        if token in self.token_blacklist_:
            self.log_event(
                'token_verification',
                {'valid': False, 'error': 'Revoked'}
            )
            return False

        import jwt

        try:
            jwt.decode(
                token,
                self.secret_key_,
                algorithms=['HS256']
            )
            self.log_event('token_verification', {'valid': True})
            return True
        except jwt.ExpiredSignatureError:
            self.log_event(
                'token_verification',
                {'valid': False, 'error': 'Expired'}
            )
            return False
        except jwt.InvalidTokenError:
            self.log_event(
                'token_verification',
                {'valid': False, 'error': 'Invalid'}
            )
            return False

    def revoke_token(self, token: str):
        """
        Revokes a token by adding it to the blacklist.

        Parameters
        ----------
        token : str
            The token to be revoked.

        Raises
        ------
        ValueError
            If the secure deployment system is not initialized.

        Notes
        -----
        Revoked tokens are stored in `token_blacklist_` and will fail
        verification in `verify_token`.

        The `run` method must be called before using this method.

        Examples
        --------
        >>> deploy.revoke_token(token)

        """
        check_is_runned(
            self,
            msg="Secure deployment system is not initialized. Call `run` first."
        )
        self.token_blacklist_.append(token)
        self.log_event('token_revocation', {'token': token})

    @ensure_pkg(
        'pyjwt',
        extra="The 'pyjwt' package is required for token operations.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def enforce_rbac(
        self,
        required_roles: List[str],
        token: str
    ) -> bool:
        """
        Enforces role-based access control (RBAC) for a user.

        Parameters
        ----------
        required_roles : list of str
            The roles required for accessing a resource.

        token : str
            The JWT token to verify the user's roles.

        Returns
        -------
        has_access : bool
            `True` if the user has at least one of the required roles,
            `False` otherwise.

        Raises
        ------
        ValueError
            If the secure deployment system is not initialized.

        Notes
        -----
        The `run` method must be called before using this method.

        Examples
        --------
        >>> has_access = deploy.enforce_rbac(['admin'], token)

        """
        check_is_runned(
            self,
            msg="Secure deployment system is not initialized. Call `run` first."
        )
        import jwt

        try:
            decoded = jwt.decode(
                token,
                self.secret_key_,
                algorithms=['HS256']
            )
            user_roles = decoded.get('roles', [])
            if any(role in required_roles for role in user_roles):
                self.log_event(
                    'rbac_check',
                    {'status': 'success', 'user_roles': user_roles}
                )
                return True
            self.log_event(
                'rbac_check',
                {
                    'status': 'failure',
                    'required_roles': required_roles,
                    'user_roles': user_roles
                }
            )
            return False
        except jwt.InvalidTokenError:
            self.log_event(
                'rbac_check',
                {'status': 'failure', 'error': 'Invalid Token'}
            )
            return False

    def ip_whitelisting(
        self,
        allowed_ips: List[str],
        current_ip: str
    ) -> bool:
        """
        Checks if the current IP is in the whitelist.

        Parameters
        ----------
        allowed_ips : list of str
            A list of IPs that are allowed to access the resource.

        current_ip : str
            The IP address of the current request.

        Returns
        -------
        is_allowed : bool
            `True` if the current IP is whitelisted, `False` otherwise.

        Raises
        ------
        ValueError
            If the secure deployment system is not initialized.

        Notes
        -----
        The `run` method must be called before using this method.

        Examples
        --------
        >>> is_allowed = deploy.ip_whitelisting(['192.168.1.1'], '192.168.1.2')

        """
        check_is_runned(
            self,
            msg="Secure deployment system is not initialized. Call `run` first."
        )
        if current_ip in allowed_ips:
            self.log_event(
                'ip_whitelist_check',
                {'status': 'allowed', 'ip': current_ip}
            )
            return True
        else:
            self.log_event(
                'ip_whitelist_check',
                {'status': 'denied', 'ip': current_ip}
            )
            return False

    def audit_event(
        self,
        event: str,
        metadata: Dict[str, Any],
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """
        Logs an audit event with optional IP address and user agent.

        Parameters
        ----------
        event : str
            The event to audit.

        metadata : dict of str to Any
            Additional metadata related to the event.

        ip_address : str or None, default=None
            The IP address of the user.

        user_agent : str or None, default=None
            The user agent of the browser or client.

        Raises
        ------
        ValueError
            If the secure deployment system is not initialized.

        Notes
        -----
        The `run` method must be called before using this method.

        Examples
        --------
        >>> deploy.audit_event('login_attempt', {'user_id': 'user123'})

        """
        check_is_runned(
            self,
            msg="Secure deployment system is not initialized. Call `run` first."
        )
        log_data = {
            'event': event,
            'metadata': metadata,
            'ip_address': ip_address,
            'user_agent': user_agent
        }
        self.log_event('audit', log_data)


@smartFitRun
class AuditTrail(BaseSecurity):
    """
    A class to handle audit trail logging, allowing flexible configurations
    for logging levels, batch logging, and external logger integrations.
    Provides functionality to log events with metadata, flush logs at
    specified intervals, and send logs to external services if configured.

    Parameters
    ----------
    logging_level : {'INFO', 'WARNING', 'ERROR'}, default='INFO'
        The logging level to use.

    external_logger : callable or None, default=None
        A custom external logger function to log events. It should accept
        two parameters: `event_type` (str) and `event_details` (dict).

    batch_logging : bool, default=False
        If `True`, enables batch logging where events are logged in batches.

    batch_size : int, default=10
        The number of events to accumulate before flushing the log when
        `batch_logging` is `True`.

    flush_interval : int, default=60
        The time interval in seconds to flush the batch log when
        `batch_logging` is enabled.

    include_metadata : bool, default=True
        If `True`, includes metadata such as `user_id`, `ip_address`, and
        `user_agent` in the event logs.

    Attributes
    ----------
    logging_level_ : str
        The logging level in use.

    external_logger_ : callable or None
        The external logger function used for logging events.

    batch_logging_ : bool
        Indicates whether batch logging is enabled.

    batch_size_ : int
        The batch size for logging events.

    flush_interval_ : int
        The time interval for flushing logs in batch mode.

    include_metadata_ : bool
        Indicates whether metadata is included in the event logs.

    event_log_ : list of dict
        A list to store event logs when `batch_logging` is enabled.

    last_flush_time_ : datetime.datetime
        Tracks the last time the event log was flushed.

    is_runned_ : bool
        Indicates whether the `run` method has been called.

    Methods
    -------
    run()
        Initializes the audit trail system.

    log_event(event_type, details, user_id=None, ip_address=None,
              user_agent=None, metadata=None)
        Logs an individual event with optional metadata.

    log_batch_events(events)
        Logs multiple events at once.

    integrate_with_cloud_logging(cloud_logging_service)
        Integrates the audit trail with an external cloud logging service.

    change_logging_level(new_level)
        Changes the logging level for future events.

    Notes
    -----
    This class extends `BaseSecurity` to provide advanced logging capabilities
    for auditing purposes. It supports batch logging, external logging
    integrations, and customizable logging levels.

    The `run` method must be called before using other methods to initialize
    the audit trail system.

    Examples
    --------
    >>> from gofast.mlops.security import AuditTrail
    >>> audit_trail = AuditTrail(logging_level='INFO', batch_logging=True)
    >>> audit_trail.run()
    >>> audit_trail.log_event(
    ...     event_type='login_attempt',
    ...     details={'success': True},
    ...     user_id='user123'
    ... )
    >>> audit_trail.change_logging_level('ERROR')

    See Also
    --------
    SecureDeployment : Class for managing secure deployment and token generation.

    """

    @validate_params({
        'logging_level': [StrOptions({'INFO', 'WARNING', 'ERROR'})],
        'external_logger': [callable,  None],
        'batch_logging': [bool],
        'batch_size': [Interval(Integral, 1, None, closed='left')],
        'flush_interval': [Interval(Integral, 1, None, closed='left')],
        'include_metadata': [bool]
    })
    def __init__(
        self,
        logging_level: str = 'INFO',
        external_logger: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        batch_logging: bool = False,
        batch_size: int = 10,
        flush_interval: int = 60,
        include_metadata: bool = True,
    ):
        """
        Initializes the `AuditTrail` class with the given configuration options
        for logging levels, external logging services, and batch logging.

        Parameters
        ----------
        logging_level : {'INFO', 'WARNING', 'ERROR'}, default='INFO'
            The logging level to use.

        external_logger : callable or None, default=None
            A custom external logger function to log events. It should accept
            two parameters: `event_type` (str) and `event_details` (dict).

        batch_logging : bool, default=False
            If `True`, enables batch logging where events are logged in batches.

        batch_size : int, default=10
            The number of events to accumulate before flushing the log when
            `batch_logging` is `True`.

        flush_interval : int, default=60
            The time interval in seconds to flush the batch log when
            `batch_logging` is enabled.

        include_metadata : bool, default=True
            If `True`, includes metadata such as `user_id`, `ip_address`, and
            `user_agent` in the event logs.

        """
        super().__init__()
        self.logging_level = logging_level.upper()
        self.external_logger = external_logger
        self.batch_logging = batch_logging
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.include_metadata = include_metadata
        self.is_runned_ = False

    @RunReturn
    def run(self):
        """
        Initializes the audit trail system.

        Notes
        -----
        This method must be called before using other methods like `log_event`.
        It sets up the necessary configurations and initializes the event log.

        """
        self.logging_level_ = self.logging_level
        self.external_logger_ = self.external_logger
        self.batch_logging_ = self.batch_logging
        self.batch_size_ = self.batch_size
        self.flush_interval_ = self.flush_interval
        self.include_metadata_ = self.include_metadata
        self.event_log_ = []
        self.last_flush_time_ = datetime.datetime.now()
        self.is_runned_ = True

    @validate_params({
        'event_type':[str],
        'details': [dict],
        'user_id': [str, None],
        'ip_address': [str, None],
        'user_agent': [str, None],
        'metadata': [dict, None]
    })
    def log_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Logs an individual event with optional metadata.

        Parameters
        ----------
        event_type : str
            The type of event being logged (e.g., `'login_attempt'`).

        details : dict of str to Any
            Specific details of the event.

        user_id : str or None, default=None
            The ID of the user associated with the event.

        ip_address : str or None, default=None
            The IP address of the user associated with the event.

        user_agent : str or None, default=None
            The user agent of the browser or client used by the user.

        metadata : dict of str to Any or None, default=None
            Additional metadata related to the event.

        Raises
        ------
        ValueError
            If the audit trail system is not initialized.

        Notes
        -----
        This method supports batch logging, console logging, and can send logs
        to an external logger if provided.

        Examples
        --------
        >>> audit_trail.log_event(
        ...     event_type='login_attempt',
        ...     details={'success': True},
        ...     user_id='user123'
        ... )

        """
        check_is_runned(
            self,
            msg="Audit trail system is not initialized. Call `run` first."
        )
        event_details = {
            'event_type': event_type,
            'details': details,
            'timestamp': datetime.datetime.utcnow().isoformat(),
        }

        if self.include_metadata_:
            event_details.update({
                'user_id': user_id,
                'ip_address': ip_address,
                'user_agent': user_agent,
                'metadata': metadata or {}
            })

        if self.batch_logging_:
            self._batch_log_event(event_details)
        else:
            self._log_event_to_console(event_details)

        if self.external_logger_:
            self.external_logger_(event_type, event_details)

    def log_batch_events(self, events: List[Dict[str, Any]]):
        """
        Logs multiple events at once.

        Parameters
        ----------
        events : list of dict
            A list of event dictionaries to log.

        Raises
        ------
        ValueError
            If the audit trail system is not initialized.

        Notes
        -----
        Each event in the `events` list should be a dictionary containing the
        same keys as required by the `log_event` method.

        Examples
        --------
        >>> events = [
        ...     {'event_type': 'login', 'details': {'success': True}},
        ...     {'event_type': 'logout', 'details': {'user_id': 'user123'}}
        ... ]
        >>> audit_trail.log_batch_events(events)

        """
        check_is_runned(
            self,
            msg="Audit trail system is not initialized. Call `run` first."
        )
        for event in events:
            self.log_event(
                event_type=event.get('event_type'),
                details=event.get('details'),
                user_id=event.get('user_id'),
                ip_address=event.get('ip_address'),
                user_agent=event.get('user_agent'),
                metadata=event.get('metadata'),
            )

    def integrate_with_cloud_logging(
        self,
        cloud_logging_service: Callable[[str, Dict[str, Any]], None]
    ):
        """
        Integrates the audit trail with an external cloud logging service.

        Parameters
        ----------
        cloud_logging_service : callable
            A function to handle cloud-based logging of events. It should accept
            two parameters: `event_type` (str) and `event_details` (dict).

        Raises
        ------
        ValueError
            If the audit trail system is not initialized.

        Notes
        -----
        After integration, all events will be sent to the external logging
        service in addition to local logging.

        Examples
        --------
        >>> def cloud_logger(event_type, event_details):
        ...     # Code to send logs to cloud service
        ...     pass
        >>> audit_trail.integrate_with_cloud_logging(cloud_logger)

        """
        check_is_runned(
            self,
            msg="Audit trail system is not initialized. Call `run` first."
        )
        logger.info("Integrating with external cloud logging service.")
        self.external_logger_ = cloud_logging_service

    def change_logging_level(self, new_level: str):
        """
        Changes the logging level for future events.

        Parameters
        ----------
        new_level : {'INFO', 'WARNING', 'ERROR'}
            The new logging level to set.

        Raises
        ------
        ValueError
            If the provided logging level is invalid or if the audit trail
            system is not initialized.

        Notes
        -----
        The logging level controls the severity of messages that are logged.

        Examples
        --------
        >>> audit_trail.change_logging_level('ERROR')

        """
        check_is_runned(
            self,
            msg="Audit trail system is not initialized. Call `run` first."
        )
        valid_levels = {'INFO', 'WARNING', 'ERROR'}
        new_level = new_level.upper()
        if new_level not in valid_levels:
            raise ValueError(
                f"Invalid logging level: {new_level}. Valid options are {valid_levels}."
            )
        self.logging_level_ = new_level
        logger.info(f"Logging level changed to: {self.logging_level_}")

    # Private methods below this line
    def _batch_log_event(self, event_details: Dict[str, Any]):
        """Logs events in batches and flushes the log if necessary."""
        self.event_log_.append(event_details)
        if len(self.event_log_) >= self.batch_size_ or self._should_flush():
            self._flush_log()

    def _flush_log(self):
        """Flushes the current batch of event logs."""
        logger.info(f"Flushing {len(self.event_log_)} audit events.")
        for event in self.event_log_:
            self._log_event_to_console(event)
        self.event_log_.clear()
        self.last_flush_time_ = datetime.datetime.now()

    def _log_event_to_console(self, event_details: Dict[str, Any]):
        """Logs events to the console based on the logging level."""
        if self.logging_level_ == 'INFO':
            logger.info(event_details)
        elif self.logging_level_ == 'WARNING':
            logger.warning(event_details)
        elif self.logging_level_ == 'ERROR':
            logger.error(event_details)

    def _should_flush(self) -> bool:
        """Determines if the event log should be flushed based on the interval."""
        time_since_last_flush = (
            datetime.datetime.now() - self.last_flush_time_
        ).total_seconds()
        return time_since_last_flush >= self.flush_interval_

@smartFitRun
class AccessControl(BaseSecurity):
    """
    A class for managing role-based access control (RBAC), allowing for role
    assignment, permission management, and user-specific permissions. It
    supports dynamic role creation, permission checks, and temporary
    permissions.

    Parameters
    ----------
    encryption_key : str or None, default=None
        The encryption key for security purposes.

    default_roles : dict of str to list of str or None, default=None
        The default roles and associated users.
        Example: ``{'admin': ['admin_user'], 'user': ['user1', 'user2']}``

    default_permissions : dict of str to list of str or None, default=None
        The default permissions and roles that can access them.
        Example: ``{'view': ['admin', 'user'], 'edit': ['admin']}``

    allow_custom_roles : bool, default=True
        Whether to allow the creation of custom roles.

    allow_role_inheritance : bool, default=True
        Whether to allow role inheritance when adding custom roles.

    Attributes
    ----------
    roles_ : dict of str to list of str
        Stores the roles and the list of users associated with each role.

    permissions_ : dict of str to list of str
        Stores the permissions and the roles that are allowed to access them.

    custom_roles_ : dict of str to list of str
        Stores any custom roles dynamically added during runtime.

    allow_custom_roles_ : bool
        Indicates if custom roles are allowed.

    allow_role_inheritance_ : bool
        Indicates if role inheritance is allowed.

    is_runned_ : bool
        Indicates whether the `run` method has been called.

    Methods
    -------
    run()
        Initializes the access control system.

    add_user(username, role)
        Adds a user to a role.

    remove_user(username, role)
        Removes a user from a role.

    add_custom_role(role_name, inherits_from=None)
        Adds a custom role, optionally inheriting users from another role.

    add_permission(permission, roles)
        Adds a permission to specified roles.

    remove_permission(permission, role)
        Removes a permission from a role.

    check_permission(username, permission)
        Checks if a user has a specific permission.

    assign_temporary_permission(username, permission, duration)
        Assigns a temporary permission to a user for a specified duration.

    get_role_users(role)
        Retrieves all users assigned to a specific role.

    get_user_permissions(username)
        Retrieves all permissions assigned to a user.

    log_role_change(username, old_role, new_role)
        Logs a role change for a user.

    Notes
    -----
    The `run` method must be called before using other methods to initialize
    the access control system.

    Examples
    --------
    >>> from gofast.mlops.security import AccessControl
    >>> ac = AccessControl()
    >>> ac.run()
    >>> ac.add_user('user1', 'user')
    >>> ac.check_permission('user1', 'view')
    True

    See Also
    --------
    SecureDeployment : Class for managing secure deployment and token generation.

    """

    @validate_params({
        'encryption_key': [str, None],
        'default_roles': [dict, None],
        'default_permissions': [dict, None],
        'allow_custom_roles': [bool],
        'allow_role_inheritance': [bool],
    })
    def __init__(
        self,
        encryption_key: Optional[str] = None,
        default_roles: Optional[Dict[str, List[str]]] = None,
        default_permissions: Optional[Dict[str, List[str]]] = None,
        allow_custom_roles: bool = True,
        allow_role_inheritance: bool = True
    ):
        """
        Initializes the `AccessControl` class with default roles and permissions,
        and enables custom roles and role inheritance by default.

        Parameters
        ----------
        encryption_key : str or None, default=None
            The encryption key for security purposes.

        default_roles : dict of str to list of str or None, default=None
            The default roles and associated users.
            Example: ``{'admin': ['admin_user'], 'user': ['user1', 'user2']}``

        default_permissions : dict of str to list of str or None, default=None
            The default permissions and roles that can access them.
            Example: ``{'view': ['admin', 'user'], 'edit': ['admin']}``

        allow_custom_roles : bool, default=True
            Whether to allow the creation of custom roles.

        allow_role_inheritance : bool, default=True
            Whether to allow role inheritance when adding custom roles.

        """
        super().__init__(encryption_key=encryption_key)
        self.encryption_key = encryption_key
        self.default_roles = default_roles
        self.default_permissions = default_permissions
        self.allow_custom_roles = allow_custom_roles
        self.allow_role_inheritance = allow_role_inheritance
        self.is_runned_ = False

    @RunReturn
    def run(self):
        """
        Initializes the access control system.

        Notes
        -----
        This method must be called before using other methods like `add_user`,
        `check_permission`, etc. It sets up the default roles and permissions.

        """
        self.roles_ = self.default_roles or {'admin': [], 'user': [], 'guest': []}
        self.permissions_ = self.default_permissions or {
            'deploy': ['admin'],
            'modify': ['admin', 'user'],
            'view': ['admin', 'user', 'guest']
        }
        self.custom_roles_ = {}  # Store any custom roles added dynamically
        self.allow_custom_roles_ = self.allow_custom_roles
        self.allow_role_inheritance_ = self.allow_role_inheritance
        self.is_runned_ = True

    def add_user(self, username: str, role: str):
        """
        Adds a user to a role.

        Parameters
        ----------
        username : str
            The name of the user to add to the role.

        role : str
            The role to assign to the user.

        Raises
        ------
        ValueError
            If the role does not exist or the system is not initialized.

        Notes
        -----
        The `run` method must be called before using this method.

        Examples
        --------
        >>> ac.add_user('user1', 'user')

        """
        check_is_runned(
            self,
            msg="Access control system is not initialized. Call `run` first."
        )
        if role in self.roles_ or (self.allow_custom_roles_ and role in self.custom_roles_):
            self.roles_.setdefault(role, []).append(username)
            self.log_event('add_user', {'username': username, 'role': role})
        else:
            raise ValueError(f"Role '{role}' does not exist.")

    def remove_user(self, username: str, role: str):
        """
        Removes a user from a role.

        Parameters
        ----------
        username : str
            The name of the user to remove from the role.

        role : str
            The role from which to remove the user.

        Raises
        ------
        ValueError
            If the role or user does not exist, or the system is not initialized.

        Notes
        -----
        The `run` method must be called before using this method.

        Examples
        --------
        >>> ac.remove_user('user1', 'user')

        """
        check_is_runned(
            self,
            msg="Access control system is not initialized. Call `run` first."
        )
        if role in self.roles_ and username in self.roles_[role]:
            self.roles_[role].remove(username)
            self.log_event('remove_user', {'username': username, 'role': role})
        else:
            raise ValueError(f"User '{username}' is not in role '{role}'.")

    def add_custom_role(self, role_name: str, inherits_from: Optional[str] = None):
        """
        Adds a custom role, optionally inheriting users from another role.

        Parameters
        ----------
        role_name : str
            The name of the custom role to create.

        inherits_from : str or None, default=None
            The role to inherit users from, if role inheritance is enabled.

        Raises
        ------
        ValueError
            If custom roles are disabled or the system is not initialized.

        Notes
        -----
        The `run` method must be called before using this method.

        Examples
        --------
        >>> ac.add_custom_role('manager', inherits_from='user')

        """
        check_is_runned(
            self,
            msg="Access control system is not initialized. Call `run` first."
        )
        if not self.allow_custom_roles_:
            raise ValueError("Custom roles are disabled.")

        if inherits_from and self.allow_role_inheritance_:
            inherited_users = self.roles_.get(inherits_from, []) + \
                              self.custom_roles_.get(inherits_from, [])
            self.custom_roles_[role_name] = inherited_users.copy()
        else:
            self.custom_roles_[role_name] = []

        self.log_event(
            'add_custom_role',
            {'role_name': role_name, 'inherits_from': inherits_from}
        )

    def add_permission(self, permission: str, roles: List[str]):
        """
        Adds a permission to specified roles.

        Parameters
        ----------
        permission : str
            The name of the permission to add.

        roles : list of str
            The roles that will be granted this permission.

        Raises
        ------
        ValueError
            If a role does not exist or the system is not initialized.

        Notes
        -----
        The `run` method must be called before using this method.

        Examples
        --------
        >>> ac.add_permission('delete', ['admin'])

        """
        check_is_runned(
            self,
            msg="Access control system is not initialized. Call `run` first."
        )
        for role in roles:
            if role in self.roles_ or (
                    self.allow_custom_roles_ and role in self.custom_roles_):
                self.permissions_.setdefault(permission, []).append(role)
                self.log_event(
                    'add_permission',
                    {'permission': permission, 'role': role}
                )
            else:
                raise ValueError(f"Role '{role}' does not exist.")

    def remove_permission(self, permission: str, role: str):
        """
        Removes a permission from a role.

        Parameters
        ----------
        permission : str
            The name of the permission to remove.

        role : str
            The role to remove the permission from.

        Raises
        ------
        ValueError
            If the permission or role does not exist, or the system is 
            not initialized.

        Notes
        -----
        The `run` method must be called before using this method.

        Examples
        --------
        >>> ac.remove_permission('modify', 'user')

        """
        check_is_runned(
            self,
            msg="Access control system is not initialized. Call `run` first."
        )
        if permission in self.permissions_ and role in self.permissions_[permission]:
            self.permissions_[permission].remove(role)
            self.log_event(
                'remove_permission',
                {'permission': permission, 'role': role}
            )
        else:
            raise ValueError(
                f"Permission '{permission}' or role '{role}' does not exist.")

    def check_permission(self, username: str, permission: str) -> bool:
        """
        Checks if a user has a specific permission.

        Parameters
        ----------
        username : str
            The name of the user whose permissions are being checked.

        permission : str
            The permission to check.

        Returns
        -------
        has_permission : bool
            `True` if the user has the permission, `False` otherwise.

        Raises
        ------
        ValueError
            If the system is not initialized.

        Notes
        -----
        The `run` method must be called before using this method.

        Examples
        --------
        >>> ac.check_permission('user1', 'view')
        True

        """
        check_is_runned(
            self,
            msg="Access control system is not initialized. Call `run` first."
        )
        user_roles = []
        for role, users in {**self.roles_, **self.custom_roles_}.items():
            if username in users:
                user_roles.append(role)
        allowed_roles = self.permissions_.get(permission, [])
        has_permission = any(role in allowed_roles for role in user_roles)
        self.log_event(
            'check_permission',
            {
                'username': username,
                'permission': permission,
                'granted': has_permission
            }
        )
        return has_permission

    def assign_temporary_permission(
            self, username: str, 
            permission: str, 
            duration: int
            ):
        """
        Assigns a temporary permission to a user for a specified duration.

        Parameters
        ----------
        username : str
            The user to assign the temporary permission to.

        permission : str
            The permission to assign.

        duration : int
            The duration (in seconds) for the temporary permission.

        Raises
        ------
        ValueError
            If the system is not initialized.

        Notes
        -----
        The `run` method must be called before using this method.

        Examples
        --------
        >>> ac.assign_temporary_permission('user1', 'deploy', 3600)

        """
        check_is_runned(
            self,
            msg="Access control system is not initialized. Call `run` first."
        )
        self.permissions_.setdefault(permission, []).append(username)
        self.log_event(
            'assign_temporary_permission',
            {
                'username': username,
                'permission': permission,
                'duration': duration
            }
        )

        threading.Timer(
            duration,
            self._revoke_temp_permission,
            args=[username, permission]
        ).start()

    def get_role_users(self, role: str) -> List[str]:
        """
        Retrieves all users assigned to a specific role.

        Parameters
        ----------
        role : str
            The role whose users are to be retrieved.

        Returns
        -------
        users : list of str
            A list of usernames assigned to the role.

        Raises
        ------
        ValueError
            If the role does not exist or the system is not initialized.

        Notes
        -----
        The `run` method must be called before using this method.

        Examples
        --------
        >>> users = ac.get_role_users('admin')

        """
        check_is_runned(
            self,
            msg="Access control system is not initialized. Call `run` first."
        )
        users = self.roles_.get(role, []) + self.custom_roles_.get(role, [])
        if users:
            return users
        else:
            raise ValueError(f"Role '{role}' does not exist.")

    def get_user_permissions(self, username: str) -> List[str]:
        """
        Retrieves all permissions assigned to a user.

        Parameters
        ----------
        username : str
            The username whose permissions are to be retrieved.

        Returns
        -------
        permissions : list of str
            A list of permissions assigned to the user.

        Raises
        ------
        ValueError
            If the system is not initialized.

        Notes
        -----
        The `run` method must be called before using this method.

        Examples
        --------
        >>> permissions = ac.get_user_permissions('user1')

        """
        check_is_runned(
            self,
            msg="Access control system is not initialized. Call `run` first."
        )
        user_permissions = []
        for permission, roles in self.permissions_.items():
            for role in roles:
                if username in self.roles_.get(role, []) or \
                   username in self.custom_roles_.get(role, []) or \
                   username == role:
                    user_permissions.append(permission)
        self.log_event(
            'get_user_permissions',
            {'username': username, 'permissions': user_permissions}
        )
        return user_permissions

    def log_role_change(self, username: str, old_role: str, new_role: str):
        """
        Logs a role change for a user.

        Parameters
        ----------
        username : str
            The username whose role is being changed.

        old_role : str
            The old role the user had.

        new_role : str
            The new role assigned to the user.

        Raises
        ------
        ValueError
            If the system is not initialized.

        Notes
        -----
        The `run` method must be called before using this method.

        Examples
        --------
        >>> ac.log_role_change('user1', 'user', 'admin')

        """
        check_is_runned(
            self,
            msg="Access control system is not initialized. Call `run` first."
        )
        self.log_event(
            'role_change',
            {'username': username, 'old_role': old_role, 'new_role': new_role}
        )

    # Private methods
    def _revoke_temp_permission(self, username: str, permission: str):
        """Revokes a temporary permission from a user."""
        if permission in self.permissions_ and username in self.permissions_[permission]:
            self.permissions_[permission].remove(username)
            self.log_event(
                'revoke_temp_permission',
                {'username': username, 'permission': permission}
            )
