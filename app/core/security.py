from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Union
from jose import jwt, JWTError
from passlib.context import CryptContext
from cryptography.fernet import Fernet
import base64

from app.core.config import settings

# Password Hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Fernet symmetric encryption for sensitive API keys at rest
def _get_fernet() -> Fernet:
    try:
        # Validate key length (must be 32 url-safe base64-encoded bytes)
        key = settings.FERNET_SECRET_KEY.encode()
        return Fernet(key)
    except Exception:
        # Derive a valid Fernet key from JWT secret via SHA256 hash
        import hashlib
        import logging
        logging.getLogger("aicto").warning(
            "FERNET_SECRET_KEY is invalid — deriving fallback key from JWT_SECRET_KEY. "
            "Generate a proper key with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
        )
        raw_hash = hashlib.sha256(settings.JWT_SECRET_KEY.encode()).digest()
        safe_key = base64.urlsafe_b64encode(raw_hash)
        return Fernet(safe_key)


fernet_cipher = _get_fernet()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "iat": now, "type": "access"})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


def create_refresh_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({"exp": expire, "iat": now, "type": "refresh"})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except JWTError:
        return None


def encrypt_secret(plain_text: str) -> str:
    """Encrypt secret data before persisting to DB."""
    if not plain_text:
        return ""
    return fernet_cipher.encrypt(plain_text.encode()).decode()


def decrypt_secret(encrypted_text: str) -> str:
    """Decrypt encrypted secret data from DB."""
    if not encrypted_text:
        return ""
    try:
        return fernet_cipher.decrypt(encrypted_text.encode()).decode()
    except Exception:
        return ""
