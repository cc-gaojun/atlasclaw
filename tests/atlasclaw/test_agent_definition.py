# -*- coding: utf-8 -*-
"""
Agent definition loading and parsing tests.

Tests for AgentDefinitionParser and AgentLoader.
"""

import pytest
from pathlib import Path

from app.atlasclaw.agent.agent_definition import (
    AgentConfig,
    AgentDefinitionParser,
    AgentLoader,
)


class TestAgentDefinitionParser:
    """Test AgentDefinitionParser functionality."""

    def test_parse_frontmatter_basic(self):
        """场景：解析基本的 YAML frontmatter"""
        content = '''---
agent_id: "main"
name: "测试助手"
version: "1.0"
---

## 系统提示词

这是系统提示词内容。
'''
        frontmatter, body = AgentDefinitionParser.parse_frontmatter(content)
        
        assert frontmatter["agent_id"] == "main"
        assert frontmatter["name"] == "测试助手"
        assert frontmatter["version"] == "1.0"
        assert "系统提示词" in body

    def test_parse_frontmatter_no_frontmatter(self):
        """场景：解析没有 frontmatter 的内容"""
        content = '''## 系统提示词

这是系统提示词内容。
'''
        frontmatter, body = AgentDefinitionParser.parse_frontmatter(content)
        
        assert frontmatter == {}
        assert "系统提示词" in body

    def test_parse_soul_md(self):
        """场景：从 SOUL.md 解析系统提示词和 capabilities"""
        content = '''---
agent_id: "main"
---

## 系统提示词

你是企业的智能助手。

## 能力范围

- 回答问题
- 提供技术支持
- 处理文档

## 可用 Providers

- jira
- confluence

## 可用 Skills

- query_knowledge
- create_ticket
'''
        result = AgentDefinitionParser.parse_soul_md(content)
        
        assert "你是企业的智能助手" in result["system_prompt"]
        assert "回答问题" in result["capabilities"]
        assert "提供技术支持" in result["capabilities"]
        assert "jira" in result["allowed_providers"]
        assert "query_knowledge" in result["allowed_skills"]

    def test_parse_identity_md(self):
        """场景：从 IDENTITY.md 解析名称、头像、语气"""
        content = '''---
agent_id: "main"
---

# IDENTITY.md - Agent 身份

## 基本信息

- **显示名称**: 小助手
- **头像**: 🤖
- **语气**: 专业、友好
'''
        result = AgentDefinitionParser.parse_identity_md(content)
        
        assert result["display_name"] == "小助手"
        assert result["avatar"] == "🤖"
        assert result["tone"] == "专业、友好"

    def test_parse_user_md(self):
        """场景：从 USER.md 解析交互方式"""
        content = '''---
agent_id: "main"
---

# USER.md - 用户交互方式

## 个性化设置

- 记住用户偏好
- 根据用户角色调整回答深度
'''
        result = AgentDefinitionParser.parse_user_md(content)
        
        assert "记住用户偏好" in result["interaction_style"]

    def test_parse_memory_md(self):
        """场景：从 MEMORY.md 解析记忆策略"""
        content = '''---
agent_id: "main"
---

# MEMORY.md - 记忆策略

## 上下文管理

- 最大轮数: 25
- 压缩策略：摘要 + 关键决策保留
'''
        result = AgentDefinitionParser.parse_memory_md(content)
        
        assert result["max_context_rounds"] == 25
        assert "最大轮数" in result["memory_strategy"]


class TestAgentLoader:
    """Test AgentLoader functionality."""

    def test_load_agent_from_files(self, tmp_path):
        """场景：从 Markdown 文件加载 Agent 定义"""
        # Create agent directory structure
        agent_dir = tmp_path / ".atlasclaw" / "agents" / "test_agent"
        agent_dir.mkdir(parents=True)
        
        # Create SOUL.md
        (agent_dir / "SOUL.md").write_text('''---
agent_id: "test_agent"
---

## 系统提示词

你是测试助手。

## 能力范围

- 测试功能

## 可用 Providers

- test_provider

## 可用 Skills

- test_skill
''', encoding="utf-8")
        
        # Create IDENTITY.md
        (agent_dir / "IDENTITY.md").write_text('''---
agent_id: "test_agent"
---

## 基本信息

- **显示名称**: 测试助手
- **头像**: 🧪
- **语气**: 专业
''', encoding="utf-8")
        
        # Load agent
        loader = AgentLoader(str(tmp_path))
        config = loader.load_agent("test_agent")
        
        assert config.agent_id == "test_agent"
        assert "测试助手" in config.system_prompt
        assert config.display_name == "测试助手"
        assert config.avatar == "🧪"
        assert "test_provider" in config.allowed_providers
        assert "test_skill" in config.allowed_skills

    def test_load_agent_with_missing_files(self, tmp_path):
        """场景：部分文件缺失时使用默认值"""
        # Create agent directory with only SOUL.md
        agent_dir = tmp_path / ".atlasclaw" / "agents" / "partial_agent"
        agent_dir.mkdir(parents=True)
        
        (agent_dir / "SOUL.md").write_text('''---
agent_id: "partial_agent"
---

## 系统提示词

部分配置的助手。
''', encoding="utf-8")
        
        loader = AgentLoader(str(tmp_path))
        config = loader.load_agent("partial_agent")
        
        assert config.agent_id == "partial_agent"
        assert "部分配置的助手" in config.system_prompt
        # Should use defaults for missing files
        assert config.display_name == "partial_agent"  # fallback to agent_id

    def test_load_agent_with_all_files_missing(self, tmp_path):
        """场景：所有文件缺失时使用内置默认配置"""
        # Create agent directory but no files
        agent_dir = tmp_path / ".atlasclaw" / "agents" / "empty_agent"
        agent_dir.mkdir(parents=True)
        
        loader = AgentLoader(str(tmp_path))
        config = loader.load_agent("empty_agent")
        
        assert config.agent_id == "empty_agent"
        # Should use default system prompt
        assert config.system_prompt == AgentLoader.DEFAULT_CONFIG.system_prompt
        assert config.capabilities == AgentLoader.DEFAULT_CONFIG.capabilities

    def test_load_main_agent_default(self, tmp_path):
        """场景：加载默认 main Agent"""
        # Initialize workspace to create default main agent
        from app.atlasclaw.core.workspace import WorkspaceInitializer
        WorkspaceInitializer(str(tmp_path)).initialize()
        
        loader = AgentLoader(str(tmp_path))
        config = loader.load_agent("main")
        
        assert config.agent_id == "main"
        # Name may be loaded from IDENTITY.md or fallback to agent_id
        assert config.name in ["main", "企业助手"]
        assert "企业助手" in config.system_prompt or config.system_prompt == AgentLoader.DEFAULT_CONFIG.system_prompt

    def test_list_agents(self, tmp_path):
        """场景：列出可用 Agent IDs"""
        # Create multiple agents
        agents_dir = tmp_path / ".atlasclaw" / "agents"
        (agents_dir / "agent1").mkdir(parents=True)
        (agents_dir / "agent2").mkdir(parents=True)
        
        loader = AgentLoader(str(tmp_path))
        agents = loader.list_agents()
        
        assert "agent1" in agents
        assert "agent2" in agents


class TestAgentConfig:
    """Test AgentConfig dataclass."""

    def test_agent_config_defaults(self):
        """Test: Verify AgentConfig default values"""
        config = AgentConfig()
        
        assert config.agent_id == "main"
        assert config.name == "Assistant"
        assert config.version == "1.0"
        assert config.max_context_rounds == 20
        assert config.capabilities == []
        assert config.allowed_providers == []
        assert config.allowed_skills == []

    def test_agent_config_custom_values(self):
        """场景：验证 AgentConfig 自定义值"""
        config = AgentConfig(
            agent_id="custom",
            name="Custom Agent",
            system_prompt="Custom prompt",
            capabilities=["cap1", "cap2"],
            allowed_providers=["provider1"],
            allowed_skills=["skill1"],
        )
        
        assert config.agent_id == "custom"
        assert config.name == "Custom Agent"
        assert config.system_prompt == "Custom prompt"
        assert "cap1" in config.capabilities
        assert "provider1" in config.allowed_providers
        assert "skill1" in config.allowed_skills
