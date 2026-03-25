# -*- coding: utf-8 -*-
"""
AuthStrategy 单元测试

涵盖：TTL 缓存命中、缓存未命中完整认证流程。
"""

from __future__ import annotations

import time
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.atlasclaw.auth.models import AuthResult, UserInfo, ANONYMOUS_USER
from app.atlasclaw.auth.strategy import AuthStrategy
from app.atlasclaw.auth.providers.base import AuthProvider
from app.atlasclaw.auth.shadow_store import ShadowUserStore


class _MockProvider(AuthProvider):
    def __init__(self, subject: str = "mock-user", call_count: list | None = None):
        self._subject = subject
        self.call_count = call_count if call_count is not None else []

    def provider_name(self) -> str:
        return "mock"

    async def authenticate(self, credential: str) -> AuthResult:
        self.call_count.append(credential)
        return AuthResult(subject=self._subject, display_name="Mock", raw_token=credential)


class TestAuthStrategy:

    @pytest.mark.asyncio
    async def test_full_auth_flow_creates_user_info(self, tmp_path):
        store = ShadowUserStore(store_path=str(tmp_path / "users.json"))
        provider = _MockProvider(subject="alice")
        strategy = AuthStrategy(providers=[provider], shadow_store=store, cache_ttl_seconds=60)

        user_info = await strategy.resolve_user("token-alice")

        assert user_info.user_id  # non-empty UUID
        assert user_info.display_name == "Mock"
        assert user_info.raw_token == "token-alice"

    @pytest.mark.asyncio
    async def test_cache_hit_skips_provider(self, tmp_path):
        calls: list[str] = []
        store = ShadowUserStore(store_path=str(tmp_path / "users.json"))
        provider = _MockProvider(subject="bob", call_count=calls)
        strategy = AuthStrategy(providers=[provider], shadow_store=store, cache_ttl_seconds=60)


        await strategy.resolve_user("token-bob")
        await strategy.resolve_user("token-bob")

        # Provider should have been called only once
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_cache_miss_after_ttl_expiry(self, tmp_path):
        calls: list[str] = []
        store = ShadowUserStore(store_path=str(tmp_path / "users.json"))
        provider = _MockProvider(subject="carol", call_count=calls)
        # TTL = 0 → every call is a cache miss
        strategy = AuthStrategy(providers=[provider], shadow_store=store, cache_ttl_seconds=0)

        await strategy.resolve_user("token-carol")
        await strategy.resolve_user("token-carol")

        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_different_tokens_are_cached_independently(self, tmp_path):
        calls: list[str] = []
        store = ShadowUserStore(store_path=str(tmp_path / "users.json"))
        provider = _MockProvider(subject="dave", call_count=calls)
        strategy = AuthStrategy(providers=[provider], shadow_store=store, cache_ttl_seconds=60)


        await strategy.resolve_user("token-A")
        await strategy.resolve_user("token-B")
        await strategy.resolve_user("token-A")  # cache hit

        assert len(calls) == 2  # token-A and token-B each called once
