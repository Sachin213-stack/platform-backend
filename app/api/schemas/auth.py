import uuid
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    refresh_token: Optional[str] = None
    business_id: str
    user_id: str


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
    email: str
    full_name: Optional[str] = None
    role: str
    business_id: uuid.UUID
    business_name: Optional[str] = None

    class Config:
        from_attributes = True
