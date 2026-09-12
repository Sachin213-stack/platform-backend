import uuid
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    refresh_token: Optional[str] = None
    business_id: str
    user_id: str


class TokenRefreshRequest(BaseModel):
    refresh_token: str


class TokenPayload(BaseModel):
    sub: str  # user_id
    business_id: str
    role: str = "member"
    type: str = "access"
    jti: Optional[str] = None


class UserRegister(BaseModel):
    business_name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: uuid.UUID
    name: str = ""
    full_name: Optional[str] = None
    email: str
    role: str
    avatar_url: Optional[str] = None
    business_id: uuid.UUID
    business_name: Optional[str] = None
    business_type: Optional[str] = "ecommerce"
    ops_email: Optional[str] = None
    timezone: Optional[str] = "America/New_York"
    currency: Optional[str] = "USD"
    auto_refresh_interval: Optional[str] = "30s"

    class Config:
        from_attributes = True


class UserOrgUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    full_name: Optional[str] = Field(None, min_length=1, max_length=255)
    business_name: Optional[str] = Field(None, min_length=2, max_length=255)
    business_type: Optional[str] = Field(None, max_length=50)
    ops_email: Optional[EmailStr] = None
    timezone: Optional[str] = None
    currency: Optional[str] = None
    auto_refresh_interval: Optional[str] = None
