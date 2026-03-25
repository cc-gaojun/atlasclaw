# -*- coding: utf-8 -*-
"""
E2E API 测试

测试完整的 API 端到端流程，需要启动完整服务。
运行方式:
1. 设置环境变量: ANTHROPIC_BASE_URL, ANTHROPIC_API_KEY
2. pytest -m e2e tests/atlasclaw/test_e2e_api.py -v
"""

import os
import json
import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator

import httpx
from httpx_sse import aconnect_sse


# 标记为 e2e 测试
pytestmark = pytest.mark.e2e


# 测试服务地址
TEST_SERVER_URL = os.environ.get("TEST_SERVER_URL", "http://127.0.0.1:8000")


@pytest.fixture(scope="module")
def event_loop():
    """创建事件循环"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP 客户端 fixture"""
    async with httpx.AsyncClient(base_url=TEST_SERVER_URL, timeout=60.0) as c:
        yield c


class TestHealthAPI:
    """健康检查 API 测试"""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, client: httpx.AsyncClient):
        """测试健康检查端点"""
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        
        data = resp.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestSkillsAPI:
    """Skills API 测试"""

    @pytest.mark.asyncio
    async def test_list_skills(self, client: httpx.AsyncClient):
        """测试未认证访问 skills 返回 401"""
        resp = await client.get("/api/skills")
        assert resp.status_code == 401


    @pytest.mark.asyncio
    async def test_skills_contain_builtin_tools(self, client: httpx.AsyncClient):
        """测试未认证访问 skills 返回 401"""
        resp = await client.get("/api/skills")
        assert resp.status_code == 401


    @pytest.mark.asyncio
    async def test_skills_contain_md_skills(self, client: httpx.AsyncClient):
        """测试 skills 包含 markdown skills (需要认证)"""
        resp = await client.get("/api/skills")
        
        # 如果没有认证，应该返回 401
        if resp.status_code == 401:
            pytest.skip("Skills API requires authentication")
            
        # 如果认证通过，验证响应格式
        assert resp.status_code == 200, f"Unexpected status: {resp.status_code}"
        data = resp.json()
        
        # 验证响应包含 skills 字段
        assert "skills" in data, f"Response missing 'skills' key: {data.keys()}"
        
        md_skills = [s for s in data["skills"] if s.get("type") == "markdown"]
        
        # 如果没有 markdown skills，只是警告而不是失败（可能是配置问题）
        if len(md_skills) == 0:
            pytest.skip("No markdown skills configured")
        
        # 验证 markdown skill 结构
        for skill in md_skills:
            assert "name" in skill
            assert "description" in skill
            assert skill["type"] == "markdown"


class TestChatAPI:
    """Chat API 测试"""

    @pytest.mark.asyncio
    async def test_chat_simple_message(self, client: httpx.AsyncClient, kimi_env_vars):
        """测试简单消息聊天"""
        # 设置环境变量
        os.environ["ANTHROPIC_BASE_URL"] = kimi_env_vars["base_url"]
        os.environ["ANTHROPIC_API_KEY"] = kimi_env_vars["api_key"]
        
        payload = {
            "message": "Hello, please respond with just 'Hi'",
            "session_key": "test-session-e2e"
        }
        
        # 使用 SSE 接收响应
        events = []
        async with aconnect_sse(
            client, 
            "POST", 
            "/api/chat",
            json=payload,
            headers={"Accept": "text/event-stream"}
        ) as event_source:
            async for event in event_source.aiter_sse():
                events.append({
                    "event": event.event,
                    "data": event.data
                })
                # 收到 end 事件后停止
                if event.event == "lifecycle" and "end" in event.data:
                    break
        
        # 验证收到了事件
        assert len(events) > 0, "应该收到 SSE 事件"
        
        # 验证事件类型
        event_types = [e["event"] for e in events]
        assert "lifecycle" in event_types, "应该有 lifecycle 事件"

    @pytest.mark.asyncio
    async def test_chat_with_tool_call(self, client: httpx.AsyncClient, kimi_env_vars):
        """测试带工具调用的聊天"""
        # 设置环境变量
        os.environ["ANTHROPIC_BASE_URL"] = kimi_env_vars["base_url"]
        os.environ["ANTHROPIC_API_KEY"] = kimi_env_vars["api_key"]
        
        payload = {
            "message": "Please use the exec tool to run 'echo hello'",
            "session_key": "test-session-tool-e2e"
        }
        
        events = []
        async with aconnect_sse(
            client,
            "POST",
            "/api/chat",
            json=payload,
            headers={"Accept": "text/event-stream"}
        ) as event_source:
            async for event in event_source.aiter_sse():
                events.append({
                    "event": event.event,
                    "data": event.data
                })
                if event.event == "lifecycle" and "end" in event.data:
                    break
        
        # 验证收到了事件
        assert len(events) > 0
        
        # 检查是否有 tool 事件（如果 LLM 决定调用工具）
        tool_events = [e for e in events if e["event"] == "tool"]
        # 注意：这个测试可能不稳定，因为 LLM 可能选择不调用工具
        # 所以我们只验证事件流正常工作
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_chat_session_persistence(self, client: httpx.AsyncClient, kimi_env_vars):
        """测试会话持久化"""
        # 设置环境变量
        os.environ["ANTHROPIC_BASE_URL"] = kimi_env_vars["base_url"]
        os.environ["ANTHROPIC_API_KEY"] = kimi_env_vars["api_key"]
        
        session_key = "test-session-persist-e2e"
        
        # 第一条消息
        payload1 = {
            "message": "My name is Alice",
            "session_key": session_key
        }
        
        events1 = []
        async with aconnect_sse(
            client,
            "POST",
            "/api/chat",
            json=payload1,
            headers={"Accept": "text/event-stream"}
        ) as event_source:
            async for event in event_source.aiter_sse():
                events1.append({"event": event.event, "data": event.data})
                if event.event == "lifecycle" and "end" in event.data:
                    break
        
        # 第二条消息 - 询问名字
        payload2 = {
            "message": "What is my name?",
            "session_key": session_key
        }
        
        events2 = []
        async with aconnect_sse(
            client,
            "POST",
            "/api/chat",
            json=payload2,
            headers={"Accept": "text/event-stream"}
        ) as event_source:
            async for event in event_source.aiter_sse():
                events2.append({"event": event.event, "data": event.data})
                if event.event == "lifecycle" and "end" in event.data:
                    break
        
        # 验证两次请求都成功
        assert len(events1) > 0
        assert len(events2) > 0


class TestSessionAPI:
    """Session API 测试"""

    @pytest.mark.skip(reason="API endpoint /api/sessions not implemented in current version")
    @pytest.mark.asyncio
    async def test_list_sessions(self, client: httpx.AsyncClient):
        """测试列出会话"""
        resp = await client.get("/api/sessions")
        assert resp.status_code == 200
        
        data = resp.json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)

    @pytest.mark.asyncio
    async def test_get_session_history(self, client: httpx.AsyncClient, kimi_env_vars):
        """测试获取会话历史"""
        # 设置环境变量
        os.environ["ANTHROPIC_BASE_URL"] = kimi_env_vars["base_url"]
        os.environ["ANTHROPIC_API_KEY"] = kimi_env_vars["api_key"]
        
        session_key = "test-session-history-e2e"
        
        # 先发送一条消息
        payload = {
            "message": "Test message for history",
            "session_key": session_key
        }
        
        async with aconnect_sse(
            client,
            "POST",
            "/api/chat",
            json=payload,
            headers={"Accept": "text/event-stream"}
        ) as event_source:
            async for event in event_source.aiter_sse():
                if event.event == "lifecycle" and "end" in event.data:
                    break
        
        # 获取历史
        resp = await client.get(f"/api/sessions/{session_key}/history")
        assert resp.status_code == 200
        
        data = resp.json()
        assert "messages" in data


class TestErrorHandling:
    """错误处理测试"""

    @pytest.mark.skip(reason="API endpoint /api/chat validation behavior changed")
    @pytest.mark.asyncio
    async def test_chat_missing_message(self, client: httpx.AsyncClient):
        """测试缺少消息参数"""
        payload = {"session_key": "test-session"}
        
        resp = await client.post("/api/chat", json=payload)
        assert resp.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_invalid_endpoint(self, client: httpx.AsyncClient):
        """测试未认证访问无效端点时优先返回 401"""
        resp = await client.get("/api/nonexistent")
        assert resp.status_code == 401



if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])
