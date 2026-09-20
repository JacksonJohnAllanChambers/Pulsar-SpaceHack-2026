"""Authenticated encryption for files held in local communication queues."""

import base64
import os
from pathlib import Path
from typing import Optional, Union

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


_MAGIC = b"PULSAR-ENC-1\x00"
_KEY_BYTES = 32
_NONCE_BYTES = 12
_KEY_ENV = "PULSAR_COMMUNICATIONS_KEY"
_KEY_PATH_ENV = "PULSAR_COMMUNICATIONS_KEY_PATH"


def _default_key_path() -> Path:
    return Path(__file__).resolve().parents[2] / ".keys" / "communications.key"


def communications_key() -> bytes:
    """Load a provisioned AES-256 key or create the local development key once."""
    encoded_key = os.environ.get(_KEY_ENV)
    if encoded_key:
        key = base64.b64decode(encoded_key, validate=True)
        if len(key) != _KEY_BYTES:
            raise ValueError(f"{_KEY_ENV} must decode to a {_KEY_BYTES}-byte key")
        return key

    key_path = Path(os.environ.get(_KEY_PATH_ENV, _default_key_path()))
    if key_path.is_file():
        key = key_path.read_bytes()
        if len(key) != _KEY_BYTES:
            raise ValueError(f"communication key at {key_path} must be {_KEY_BYTES} bytes")
        return key

    key_path.parent.mkdir(parents=True, exist_ok=True)
    key = AESGCM.generate_key(bit_length=256)
    try:
        descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return communications_key()
    with os.fdopen(descriptor, "wb") as key_file:
        key_file.write(key)
    return key


def encrypt_bytes(plaintext: bytes, *, associated_data: Optional[bytes] = None) -> bytes:
    nonce = os.urandom(_NONCE_BYTES)
    return _MAGIC + nonce + AESGCM(communications_key()).encrypt(nonce, plaintext, associated_data)


def decrypt_bytes(ciphertext: bytes, *, associated_data: Optional[bytes] = None) -> bytes:
    if not ciphertext.startswith(_MAGIC):
        raise ValueError("file is not a Pulsar encrypted communication payload")
    nonce = ciphertext[len(_MAGIC):len(_MAGIC) + _NONCE_BYTES]
    if len(nonce) != _NONCE_BYTES:
        raise ValueError("encrypted communication payload is truncated")
    return AESGCM(communications_key()).decrypt(nonce, ciphertext[len(_MAGIC) + _NONCE_BYTES:], associated_data)


def encrypt_file(source: Union[str, Path], destination: Union[str, Path]) -> Path:
    source_path, destination_path = Path(source), Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    ciphertext = encrypt_bytes(source_path.read_bytes(), associated_data=source_path.name.encode("utf-8"))
    destination_path.write_bytes(ciphertext)
    source_path.unlink()
    return destination_path


def decrypt_file(path: Union[str, Path], *, original_name: Optional[str] = None) -> bytes:
    payload_path = Path(path)
    name = original_name or payload_path.name.removesuffix(".enc")
    return decrypt_bytes(payload_path.read_bytes(), associated_data=name.encode("utf-8"))