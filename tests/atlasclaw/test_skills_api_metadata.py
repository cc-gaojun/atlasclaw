# -*- coding: utf-8 -*-
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.atlasclaw.api.routes import APIContext, create_router, set_api_context
from app.atlasclaw.auth.guards import AuthorizationContext, get_authorization_context
from app.atlasclaw.auth.models import UserInfo
from app.atlasclaw.db.orm.role import build_default_permissions
from app.atlasclaw.session.manager import SessionManager
from app.atlasclaw.session.queue import SessionQueue
from app.atlasclaw.skills.registry import SkillRegistry
from app.atlasclaw.tools.registration import register_builtin_tools


def _build_client(tmp_path) -> TestClient:
    registry = SkillRegistry()
    register_builtin_tools(registry)
    ctx = APIContext(
        session_manager=SessionManager(agents_dir=str(tmp_path / "agents")),
        session_queue=SessionQueue(),
        skill_registry=registry,
    )
    set_api_context(ctx)

    app = FastAPI()
    app.include_router(create_router())

    permissions = build_default_permissions()
    permissions["skills"]["module_permissions"]["view"] = True
    app.dependency_overrides[get_authorization_context] = lambda: AuthorizationContext(
        user=UserInfo(
            user_id="skill-metadata-viewer",
            display_name="Skill Metadata Viewer",
            roles=["viewer"],
            extra={},
            auth_type="test",
        ),
        role_identifiers=["viewer"],
        permissions=permissions,
        is_admin=False,
    )
    return TestClient(app)


def test_skills_api_include_metadata_uses_normalized_tool_snapshot(tmp_path) -> None:
    client = _build_client(tmp_path)

    response = client.get("/api/skills?include_metadata=true")

    assert response.status_code == 200
    payload = response.json()
    skills = {item["name"]: item for item in payload["skills"] if item["type"] == "executable"}

    assert skills["web_search"]["source"] == "builtin"
    assert skills["web_search"]["capability_class"] == "web_search"
    assert set(skills["web_search"]["group_ids"]) == {"group:web", "group:atlasclaw"}

    assert skills["openmeteo_weather"]["source"] == "builtin"
    assert skills["openmeteo_weather"]["capability_class"] == "weather"
    assert set(skills["openmeteo_weather"]["group_ids"]) == {"group:web", "group:atlasclaw"}
