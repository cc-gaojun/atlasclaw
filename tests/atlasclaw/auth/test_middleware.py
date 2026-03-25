# -*- coding: utf-8 -*-
"""
AuthMiddleware 单元测试

涵盖：Token 认证成功 → 注入 UserInfo、Token 无效 → 401、无 auth 配置 → anonymous、
provider=none → default_user_id。
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.atlasclaw.auth.models import AuthResult, UserInfo, ANONYMOUS_USER, AuthenticationError
from app.atlasclaw.auth.middleware import setup_auth_middleware, AuthMiddleware
from app.atlasclaw.auth.config import AuthConfig

from app.atlasclaw.auth.strategy import AuthStrategy
from app.atlasclaw.auth.shadow_store import ShadowUserStore


def _make_app(auth_config=None, strategy=None):
    """Build a minimal FastAPI app with auth middleware and a test endpoint."""
    app = FastAPI()
    setup_auth_middleware(app, auth_config=auth_config, strategy=strategy)

    @app.get("/test")
    async def test_endpoint(request: "Request"):
        from fastapi import Request
        user_info = getattr(request.state, "user_info", None)
        if user_info is None:
            return {"user_id": None}
        return {"user_id": user_info.user_id, "display_name": user_info.display_name}

    return app


class TestAuthMiddleware:

    @pytest.mark.skip(reason="BaseHTTPMiddleware compatibility issue with TestClient in newer Starlette")
    def test_no_auth_config_injects_anonymous(self):
        """When no auth config is provided, every request gets anonymous user_id."""
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from app.atlasclaw.auth.providers.none import NoneProvider

        app = FastAPI()
        strategy = AuthStrategy(provider=NoneProvider(default_user_id="anonymous"), shadow_store=None)
        app.add_middleware(AuthMiddleware, strategy=strategy, anonymous_fallback=True)

        @app.get("/ping")
        async def ping(request: Request):
            ui = getattr(request.state, "user_info", ANONYMOUS_USER)
            return {"user_id": ui.user_id}

        client = TestClient(app)
        resp = client.get("/ping")
        assert resp.status_code == 200
        assert resp.json()["user_id"] == "anonymous"

    @pytest.mark.skip(reason="BaseHTTPMiddleware compatibility issue with TestClient in newer Starlette")
    def test_provider_none_uses_default_user_id(self, tmp_path):
        """provider=none → default_user_id injected, no credential required."""
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from app.atlasclaw.auth.providers.none import NoneProvider

        store = ShadowUserStore(store_path=str(tmp_path / "users.json"))
        provider = NoneProvider(default_user_id="dev")
        strategy = AuthStrategy(provider=provider, shadow_store=store)

        app = FastAPI()
        app.add_middleware(AuthMiddleware, strategy=strategy, anonymous_fallback=False)

        @app.get("/ping")
        async def ping(request: Request):
            ui = getattr(request.state, "user_info", ANONYMOUS_USER)
            return {"user_id": ui.user_id}

        client = TestClient(app)
        resp = client.get("/ping")
        assert resp.status_code == 200

    @pytest.mark.skip(reason="BaseHTTPMiddleware compatibility issue with TestClient in newer Starlette")
    def test_missing_token_returns_401(self, tmp_path):
        """Real provider, no token → 401."""
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from app.atlasclaw.auth.providers.smartcmp import SmartCMPProvider
        from app.atlasclaw.auth.config import SmartCMPAuthConfig

        store = ShadowUserStore(store_path=str(tmp_path / "users.json"))
        prov = SmartCMPProvider(validate_url="http://example.com/v")
        strategy = AuthStrategy(provider=prov, shadow_store=store)

        app = FastAPI()
        app.add_middleware(AuthMiddleware, strategy=strategy, anonymous_fallback=False)

        @app.get("/ping")
        async def ping(request: Request):
            return {"ok": True}

        client = TestClient(app)
        resp = client.get("/ping")
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid or expired token"

    @pytest.mark.skip(reason="BaseHTTPMiddleware compatibility issue with TestClient in newer Starlette")
    def test_invalid_token_returns_401(self, tmp_path):
        """Provider raises AuthenticationError → 401."""
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from app.atlasclaw.auth.providers.base import AuthProvider

        class FailingProvider(AuthProvider):
            def provider_name(self):
                return "fail"
            async def authenticate(self, credential: str) -> AuthResult:
                raise AuthenticationError("bad token")

        store = ShadowUserStore(store_path=str(tmp_path / "users.json"))
        strategy = AuthStrategy(provider=FailingProvider(), shadow_store=store)

        app = FastAPI()
        app.add_middleware(
            AuthMiddleware,
            strategy=strategy,
            anonymous_fallback=False,
        )

        @app.get("/ping")
        async def ping(request: Request):
            return {"ok": True}

        client = TestClient(app)
        resp = client.get("/ping", headers={"CloudChef-Authenticate": "bad-token"})
        assert resp.status_code == 401
