"""Pydantic configuration schema definitions."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field

# Auth config is imported lazily to avoid circular imports at module load time.
# AuthConfig is referenced only in AtlasClawConfig.auth field annotation.


class LogLevel(str, Enum):
    """Supported log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class QueueModeConfig(str, Enum):
    """Supported queue modes."""
    COLLECT = "collect"
    STEER = "steer"
    FOLLOWUP = "followup"
    STEER_BACKLOG = "steer-backlog"
    INTERRUPT = "interrupt"


class DropStrategy(str, Enum):
    """Queue overflow strategies."""
    OLD = "old"
    NEW = "new"
    SUMMARIZE = "summarize"


class ResetMode(str, Enum):
    """Supported session reset policies."""
    DAILY = "daily"
    IDLE = "idle"
    MANUAL = "manual"


class PromptMode(str, Enum):
    """System prompt mode"""
    FULL = "full"
    MINIMAL = "minimal"
    NONE = "none"


class SandboxMode(str, Enum):
    """Sandbox mode"""
    OFF = "off"
    AGENT = "agent"
    SESSION = "session"


class HumanDelayMode(str, Enum):
    """Human-like delay mode"""
    OFF = "off"
    NATURAL = "natural"
    CUSTOM = "custom"


# ============================================================
# configurationmodel
# ============================================================

class QueueConfig(BaseModel):
    """configuration"""
    mode: QueueModeConfig = QueueModeConfig.COLLECT
    debounce_ms: int = Field(default=1000, ge=0, description="防抖等待毫秒数")
    cap: int = Field(default=20, ge=1, description="每会话最大排队消息数")
    drop: DropStrategy = DropStrategy.OLD


class ResetConfig(BaseModel):
    """session configuration"""
    mode: ResetMode = ResetMode.DAILY
    daily_hour: int = Field(default=4, ge=0, le=23, description="每日重置时间（小时）")
    idle_minutes: int = Field(default=60, ge=1, description="空闲重置阈值（分钟）")


class CompactionConfig(BaseModel):
    """Compaction configuration"""
    reserve_tokens_floor: int = Field(default=20000, description="预留给新响应的 token 数")
    soft_threshold_tokens: int = Field(default=4000, description="触发记忆刷新的软阈值")
    context_window: int = Field(default=128000, description="模型上下文窗口大小")
    memory_flush_enabled: bool = True


class BlockChunkerConfig(BaseModel):
    """streaming configuration"""
    min_chars: int = Field(default=800, ge=1, description="最小分块字符数")
    max_chars: int = Field(default=1200, ge=1, description="最大分块字符数")
    break_preference: str = Field(default="paragraph", description="断点偏好")
    idle_ms: int = Field(default=300, ge=0, description="空闲刷新毫秒数")


class HumanDelayConfig(BaseModel):
    """Human-like delay configuration"""
    mode: HumanDelayMode = HumanDelayMode.OFF
    min_ms: int = Field(default=800, ge=0)
    max_ms: int = Field(default=2500, ge=0)


class SandboxConfig(BaseModel):
    """Sandbox configuration"""
    enabled: bool = False
    mode: SandboxMode = SandboxMode.OFF
    workspace_root: str = ""
    elevated_exec: bool = False


class SecurityPolicyConfig(BaseModel):
    """configuration"""
    allowed_tools: list[str] = Field(default_factory=list, description="允许的工具列表（空=全部允许）")
    denied_tools: list[str] = Field(default_factory=list, description="拒绝的工具列表（优先，支持 * 通配符）")
    workspace_access: str = Field(default="rw", description="工作区访问权限：rw | ro | none")


class SkillsConfig(BaseModel):
    """MD Skills configuration"""
    md_skills_max_count: int = Field(default=20, ge=1, description="索引段最多显示的 MD Skills 数量")
    md_skills_desc_max_chars: int = Field(default=200, ge=1, description="单条描述最大字符数")
    md_skills_index_max_chars: int = Field(default=3000, ge=1, description="索引段总字符上限")
    md_skills_max_file_bytes: int = Field(default=262144, ge=1, description="单个 SKILL.md 文件最大字节数（默认 256KB）")


class WebhookSkillSourceConfig(BaseModel):
    """Additional markdown-skill roots addressable by provider-qualified name."""
    provider: str = Field(description="Provider namespace used in provider:skill identifiers")
    root: str = Field(description="Path to the skills root directory")


class WebhookSystemConfig(BaseModel):
    """Per-system webhook access configuration."""
    system_id: str = Field(description="Stable identifier for the external system")
    enabled: bool = True
    sk_env: str = Field(description="Environment variable that stores the shared secret")
    default_agent_id: str = "main"
    allowed_skills: list[str] = Field(default_factory=list)


class WebhookConfig(BaseModel):
    """Inbound webhook dispatch configuration."""
    enabled: bool = False
    header_name: str = "X-AtlasClaw-SK"
    skill_sources: list[WebhookSkillSourceConfig] = Field(default_factory=list)
    systems: list[WebhookSystemConfig] = Field(default_factory=list)


class ModelConfig(BaseModel):
    """modelconfiguration"""
    primary: str = Field(default="doubao-pro-32k", description="主模型（格式: provider/model）")
    fallbacks: list[str] = Field(default_factory=list, description="回退模型列表")
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: Optional[int] = None
    providers: dict[str, Any] = Field(
        default_factory=dict,
        description="LLM 提供商配置，{name: {base_url, api_key, api_type, models}}",
    )


class RetryConfig(BaseModel):
    """Retry configuration"""
    attempts: int = Field(default=3, ge=1)
    min_delay_ms: int = Field(default=1000, ge=0)
    max_delay_ms: int = Field(default=30000, ge=0)
    jitter: float = Field(default=0.1, ge=0, le=1)


class AgentDefaultsConfig(BaseModel):
    """agentdefaultconfiguration"""
    timeout_seconds: int = Field(default=600, ge=1, description="运行超时秒数")
    max_concurrent: int = Field(default=4, ge=1, description="最大并发数")
    max_tool_calls: int = Field(default=50, ge=1, description="单次运行最大工具调用数")
    prompt_mode: PromptMode = PromptMode.FULL
    bootstrap_max_chars: int = Field(default=20000, description="Bootstrap 文件最大字符数")
    block_streaming_default: bool = False
    block_streaming_break: str = "text_end"
    human_delay: HumanDelayConfig = Field(default_factory=HumanDelayConfig)


class MessagesConfig(BaseModel):
    """messageconfiguration"""
    queue: QueueConfig = Field(default_factory=QueueConfig)
    response_prefix: str = ""
    reply_to_mode: str = "auto"
    inbound_debounce_ms: int = Field(default=1000, ge=0)
    dedup_ttl_seconds: int = Field(default=60, ge=1)


class MemoryConfig(BaseModel):
    """configuration"""
    enabled: bool = True
    vector_weight: float = Field(default=0.7, ge=0, le=1, description="向量搜索权重")
    fulltext_weight: float = Field(default=0.3, ge=0, le=1, description="全文搜索权重")
    time_decay_half_life_days: float = Field(default=30.0, ge=1, description="时间衰减半衰期（天）")
    max_results: int = Field(default=6, ge=1)


class WorkspaceConfig(BaseModel):
    """Workspace configuration"""
    path: str = Field(default=".", description="工作区路径，默认为当前目录")
    per_user_isolation: bool = Field(default=True, description="是否按用户隔离数据")


class AtlasClawConfig(BaseModel):
    """AtlasClaw configuration"""
    log_level: LogLevel = LogLevel.INFO
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig, description="工作区配置")
    agents_dir: str = Field(default="~/.atlasclaw/agents", description="智能体目录（向后兼容）")
    
    # subconfiguration
    agent_defaults: AgentDefaultsConfig = Field(default_factory=AgentDefaultsConfig)
    messages: MessagesConfig = Field(default_factory=MessagesConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    block_chunker: BlockChunkerConfig = Field(default_factory=BlockChunkerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    security: SecurityPolicyConfig = Field(default_factory=SecurityPolicyConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    reset: ResetConfig = Field(default_factory=ResetConfig)
    webhook: WebhookConfig = Field(default_factory=WebhookConfig)

    # Auth configuration — loaded from `auth` section of atlasclaw.json.
    # None means no auth config present; runtime falls back to anonymous mode.
    auth: Optional[Any] = Field(
        default=None,
        description="认证配置，对应 atlasclaw.json 的 auth 节，缺少时回退为 anonymous 模式",
    )
    
    # ServiceProvider instanceconfiguration
    # :{provider_type:{instance_name:{param:value}}}
    service_providers: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="企业服务提供者实例配置，{type: {instance: {params}}}",
    )
