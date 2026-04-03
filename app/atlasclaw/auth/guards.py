# -*- coding: utf-8 -*-
"""
FastAPI dependency guards for authentication and authorization.

Provides reusable dependency functions for:
- Extracting authenticated user from request state
- Requiring admin privileges for protected endpoints
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from fastapi import Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.atlasclaw.auth.models import UserInfo
from app.atlasclaw.db import get_db_session_dependency as get_db_session
from app.atlasclaw.db.models import UserModel
from app.atlasclaw.db.orm.role import RoleService, build_default_permissions
from app.atlasclaw.db.orm.user import UserService


SKILL_MODULE_PERMISSION_KEYS = {"view", "enable_disable", "manage_permissions"}


@dataclass
class AuthorizationContext:
    """Resolved authorization state for the current request."""

    user: UserInfo
    db_user: Optional[UserModel] = None
    role_identifiers: list[str] = field(default_factory=list)
    permissions: dict[str, Any] = field(default_factory=build_default_permissions)
    is_admin: bool = False


async def get_current_user(request: Request) -> UserInfo:
    """
    Extract authenticated user from request state.

    This dependency retrieves the UserInfo object injected by AuthMiddleware
    and validates that the user is properly authenticated (not anonymous).

    Args:
        request: The FastAPI request object

    Returns:
        UserInfo: The authenticated user's information

    Raises:
        HTTPException: 401 if no user info found or user is anonymous
    """
    user_info = getattr(request.state, "user_info", None)
    if not user_info or user_info.user_id == "anonymous":
        raise HTTPException(status_code=401, detail="Authentication required")
    return user_info


async def require_admin(user: UserInfo = Depends(get_current_user)) -> UserInfo:
    """
    Require admin privileges for the current user.

    This dependency builds on get_current_user and additionally checks
    that the authenticated user has admin privileges.

    Args:
        user: The authenticated user (injected by get_current_user)

    Returns:
        UserInfo: The authenticated admin user's information

    Raises:
        HTTPException: 401 if not authenticated (via get_current_user)
        HTTPException: 403 if user is not an admin
    """
    is_admin = user.extra.get("is_admin", False) if user.extra else False
    if not is_admin:
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return user


def _extract_role_identifiers(raw_roles: Any) -> list[str]:
    """Normalize assigned role identifiers from either dict or list storage."""
    if isinstance(raw_roles, dict):
        return [str(identifier) for identifier, enabled in raw_roles.items() if bool(enabled)]
    if isinstance(raw_roles, list):
        return [str(identifier) for identifier in raw_roles if str(identifier).strip()]
    return []


def _merge_skill_permissions(
    current_entries: list[dict[str, Any]],
    incoming_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge per-skill permissions across roles using OR semantics."""
    merged: dict[str, dict[str, Any]] = {}
    for entry in current_entries + incoming_entries:
        if not isinstance(entry, dict):
            continue

        skill_id = str(entry.get("skill_id") or entry.get("skill_name") or "").strip()
        if not skill_id:
            continue

        existing = merged.get(skill_id)
        if not existing:
            merged[skill_id] = {
                "skill_id": skill_id,
                "skill_name": str(entry.get("skill_name") or skill_id),
                "description": str(entry.get("description") or ""),
                "authorized": bool(entry.get("authorized", False)),
                "enabled": bool(entry.get("enabled", False)),
            }
            continue

        existing["authorized"] = existing["authorized"] or bool(entry.get("authorized", False))
        existing["enabled"] = existing["enabled"] or bool(entry.get("enabled", False))
        if not existing["description"] and entry.get("description"):
            existing["description"] = str(entry.get("description"))
        if not existing["skill_name"] and entry.get("skill_name"):
            existing["skill_name"] = str(entry.get("skill_name"))

    return list(merged.values())


def _merge_permissions(current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Merge role permissions with recursive OR semantics."""
    merged = dict(current)
    for key, value in incoming.items():
        existing = merged.get(key)
        if key == "skill_permissions" and isinstance(existing, list) and isinstance(value, list):
            merged[key] = _merge_skill_permissions(existing, value)
            continue

        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _merge_permissions(existing, value)
            continue

        if isinstance(existing, bool) and isinstance(value, bool):
            merged[key] = existing or value
            continue

        merged[key] = value

    return merged


def _normalize_permission_path(permission_path: str) -> list[str]:
    parts = [segment.strip() for segment in permission_path.split(".") if segment.strip()]
    if len(parts) == 2 and parts[0] == "skills" and parts[1] in SKILL_MODULE_PERMISSION_KEYS:
        return ["skills", "module_permissions", parts[1]]
    return parts


def has_permission(authz: AuthorizationContext, permission_path: str) -> bool:
    """Check whether the current user has a specific effective permission."""
    if authz.is_admin:
        return True

    value: Any = authz.permissions
    for segment in _normalize_permission_path(permission_path):
        if not isinstance(value, dict):
            return False
        value = value.get(segment)

    return value is True


def ensure_permission(
    authz: AuthorizationContext,
    permission_path: str,
    *,
    detail: Optional[str] = None,
) -> None:
    """Raise 403 if the current user lacks a required permission."""
    if has_permission(authz, permission_path):
        return
    raise HTTPException(status_code=403, detail=detail or f"Missing permission: {permission_path}")


def ensure_any_permission(
    authz: AuthorizationContext,
    permission_paths: Sequence[str],
    *,
    detail: str,
) -> None:
    """Raise 403 unless one of the requested permissions is granted."""
    if any(has_permission(authz, permission_path) for permission_path in permission_paths):
        return
    raise HTTPException(status_code=403, detail=detail)


def can_manage_permission_module(authz: AuthorizationContext, module_id: str) -> bool:
    """Check whether the current user can govern a permission module."""
    if has_permission(authz, "rbac.manage_permissions"):
        return True
    if module_id in {"rbac", "roles"}:
        return False
    return has_permission(authz, f"{module_id}.manage_permissions")


def ensure_can_manage_permission_modules(
    authz: AuthorizationContext,
    requested_permissions: Optional[dict[str, Any]],
    *,
    existing_permissions: Optional[dict[str, Any]] = None,
) -> None:
    """Validate permission-matrix edits against module governance permissions."""
    normalized_existing = RoleService.normalize_permissions(existing_permissions)
    normalized_requested = RoleService.normalize_permissions(requested_permissions)
    changed_modules = sorted({
        module_id
        for module_id in set(normalized_existing.keys()) | set(normalized_requested.keys())
        if normalized_existing.get(module_id) != normalized_requested.get(module_id)
    })

    if not changed_modules:
        return

    if has_permission(authz, "rbac.manage_permissions"):
        return

    unauthorized_modules = [
        module_id for module_id in changed_modules if not can_manage_permission_module(authz, module_id)
    ]
    if unauthorized_modules:
        raise HTTPException(
            status_code=403,
            detail=(
                "Missing permission governance access for module(s): "
                + ", ".join(unauthorized_modules)
            ),
        )


async def get_authorization_context(
    request: Request,
    user: UserInfo = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> AuthorizationContext:
    """Resolve effective request permissions from assigned roles."""
    cached = getattr(request.state, "authorization_context", None)
    if isinstance(cached, AuthorizationContext) and cached.user.user_id == user.user_id:
        return cached

    await RoleService.ensure_builtin_roles(session)

    db_user: Optional[UserModel] = None
    db_lookup_name = str(user.user_id or "").strip()
    if db_lookup_name:
        db_user = await UserService.get_by_username(session, db_lookup_name)
        if db_user and not db_user.is_active:
            raise HTTPException(status_code=403, detail="User account is inactive")

    role_identifiers = _extract_role_identifiers(db_user.roles if db_user else user.roles)
    if db_user and db_user.is_admin and "admin" not in role_identifiers:
        role_identifiers.append("admin")

    effective_permissions = build_default_permissions()
    if role_identifiers:
        roles = await RoleService.list_by_identifiers(session, role_identifiers, is_active=True)
        for role in roles:
            effective_permissions = _merge_permissions(
                effective_permissions,
                RoleService.normalize_permissions(role.permissions),
            )

    is_admin = (
        bool(user.extra.get("is_admin", False))
        or bool(getattr(db_user, "is_admin", False))
        or any(identifier.lower() == "admin" for identifier in role_identifiers)
    )

    authz = AuthorizationContext(
        user=user,
        db_user=db_user,
        role_identifiers=role_identifiers,
        permissions=effective_permissions,
        is_admin=is_admin,
    )
    request.state.authorization_context = authz
    return authz
