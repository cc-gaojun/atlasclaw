# -*- coding: utf-8 -*-
"""Tests for role CRUD API endpoints."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.atlasclaw.api.api_routes import router as api_router
from app.atlasclaw.api.routes import APIContext, create_router, set_api_context
from app.atlasclaw.auth.config import AuthConfig
from app.atlasclaw.auth.middleware import setup_auth_middleware
from app.atlasclaw.db import get_db_session
from app.atlasclaw.db.database import DatabaseConfig, DatabaseManager, init_database
from app.atlasclaw.db.models import RoleModel
from app.atlasclaw.db.orm.user import UserService
from app.atlasclaw.db.schemas import UserCreate
from app.atlasclaw.session.manager import SessionManager
from app.atlasclaw.session.queue import SessionQueue
from app.atlasclaw.skills.registry import SkillRegistry


_test_db_manager: DatabaseManager = None


async def _test_get_db_session() -> AsyncGenerator[AsyncSession, None]:
    global _test_db_manager
    async with _test_db_manager.get_session() as session:
        yield session


def _build_client(tmp_path: Path, auth_config: AuthConfig) -> TestClient:
    ctx = APIContext(
        session_manager=SessionManager(agents_dir=str(tmp_path / 'agents')),
        session_queue=SessionQueue(),
        skill_registry=SkillRegistry(),
    )
    set_api_context(ctx)

    app = FastAPI()
    app.state.config = SimpleNamespace(auth=auth_config)
    setup_auth_middleware(app, auth_config)
    app.include_router(create_router())
    app.include_router(api_router)
    app.dependency_overrides[get_db_session] = _test_get_db_session
    return TestClient(app)


def _init_database_sync(tmp_path: Path):
    global _test_db_manager

    async def _init():
        global _test_db_manager
        db_path = tmp_path / 'test_role_crud.db'
        _test_db_manager = await init_database(DatabaseConfig(db_type='sqlite', sqlite_path=str(db_path)))
        await _test_db_manager.create_tables()
        async with _test_db_manager.get_session() as session:
            await UserService.create(
                session,
                UserCreate(
                    username='admin',
                    password='adminpass123',
                    display_name='Test Admin',
                    email='admin@test.com',
                    roles={'admin': True},
                    auth_type='local',
                    is_admin=True,
                    is_active=True,
                ),
            )
            await UserService.create(
                session,
                UserCreate(
                    username='regularuser',
                    password='userpass123',
                    display_name='Regular User',
                    email='user@test.com',
                    roles={},
                    auth_type='local',
                    is_admin=False,
                    is_active=True,
                ),
            )
        return _test_db_manager

    return asyncio.run(_init())


def _cleanup_manager(manager):
    asyncio.run(manager.close())


def _get_auth_config() -> AuthConfig:
    return AuthConfig(
        provider='local',
        jwt={
            'secret_key': 'test-secret-key-for-testing',
            'issuer': 'atlasclaw-test',
            'header_name': 'AtlasClaw-Authenticate',
            'cookie_name': 'AtlasClaw-Authenticate',
            'expires_minutes': 60,
        },
    )


def _login_as(client: TestClient, username: str, password: str) -> str:
    response = client.post('/api/auth/local/login', json={'username': username, 'password': password})
    assert response.status_code == 200, f'Login failed: {response.json()}'
    return response.json()['token']


class TestRoleCRUDAPI:
    """Tests for role management endpoints."""

    def test_list_roles_auto_seeds_builtin_roles(self, tmp_path):
        manager = _init_database_sync(tmp_path)
        client = _build_client(tmp_path, _get_auth_config())
        token = _login_as(client, 'admin', 'adminpass123')

        response = client.get('/api/roles?page=1&page_size=20', headers={'AtlasClaw-Authenticate': token})

        assert response.status_code == 200
        payload = response.json()
        identifiers = {role['identifier'] for role in payload['roles']}
        assert {'admin', 'user', 'viewer'}.issubset(identifiers)
        roles_by_identifier = {role['identifier']: role for role in payload['roles']}
        assert roles_by_identifier['admin']['permissions']['agent_configs']['view'] is True
        assert roles_by_identifier['user']['permissions']['agent_configs']['view'] is False
        assert roles_by_identifier['viewer']['permissions']['agent_configs']['view'] is False
        assert roles_by_identifier['viewer']['permissions']['provider_configs']['view'] is False
        assert roles_by_identifier['viewer']['permissions']['model_configs']['view'] is False

        _cleanup_manager(manager)

    def test_list_roles_syncs_builtin_defaults(self, tmp_path):
        manager = _init_database_sync(tmp_path)

        async def _seed_stale_viewer():
            async with _test_db_manager.get_session() as session:
                session.add(
                    RoleModel(
                        name='Viewer',
                        identifier='viewer',
                        description='stale viewer',
                        permissions={
                            'skills': {'module_permissions': {'view': True}, 'skill_permissions': []},
                            'agent_configs': {'view': True},
                            'provider_configs': {'view': True},
                            'model_configs': {'view': True},
                        },
                        is_builtin=True,
                        is_active=False,
                    )
                )
                await session.flush()

        asyncio.run(_seed_stale_viewer())

        client = _build_client(tmp_path, _get_auth_config())
        token = _login_as(client, 'admin', 'adminpass123')

        response = client.get('/api/roles?page=1&page_size=20', headers={'AtlasClaw-Authenticate': token})

        assert response.status_code == 200
        roles_by_identifier = {role['identifier']: role for role in response.json()['roles']}
        viewer = roles_by_identifier['viewer']
        assert viewer['description'] == 'Read-only role for audit and oversight workflows.'
        assert viewer['is_active'] is True
        assert viewer['permissions']['agent_configs']['view'] is False
        assert viewer['permissions']['provider_configs']['view'] is False
        assert viewer['permissions']['model_configs']['view'] is False

        _cleanup_manager(manager)

    def test_create_update_and_delete_custom_role(self, tmp_path):
        manager = _init_database_sync(tmp_path)
        client = _build_client(tmp_path, _get_auth_config())
        token = _login_as(client, 'admin', 'adminpass123')

        create_response = client.post(
            '/api/roles',
            json={
                'name': 'Operations',
                'identifier': 'operations',
                'description': 'Operations role',
                'permissions': {
                    'skills': {'module_permissions': {'view': True, 'enable_disable': False}, 'skill_permissions': []},
                    'channels': {'view': True, 'create': True, 'edit': False, 'delete': False},
                    'tokens': {'view': False, 'create': False, 'edit': False, 'delete': False},
                    'users': {'view': True, 'create': False, 'edit': False, 'delete': False, 'reset_password': False},
                    'roles': {'view': False, 'create': False, 'edit': False, 'delete': False},
                },
                'is_active': True,
            },
            headers={'AtlasClaw-Authenticate': token},
        )

        assert create_response.status_code == 201
        role_id = create_response.json()['id']

        update_response = client.put(
            f'/api/roles/{role_id}',
            json={'description': 'Updated operations role', 'is_active': False},
            headers={'AtlasClaw-Authenticate': token},
        )

        assert update_response.status_code == 200
        assert update_response.json()['description'] == 'Updated operations role'
        assert update_response.json()['is_active'] is False

        delete_response = client.delete(f'/api/roles/{role_id}', headers={'AtlasClaw-Authenticate': token})
        assert delete_response.status_code == 204

        get_response = client.get(f'/api/roles/{role_id}', headers={'AtlasClaw-Authenticate': token})
        assert get_response.status_code == 404

        _cleanup_manager(manager)

    def test_duplicate_role_identifier_returns_409(self, tmp_path):
        manager = _init_database_sync(tmp_path)
        client = _build_client(tmp_path, _get_auth_config())
        token = _login_as(client, 'admin', 'adminpass123')

        client.post(
            '/api/roles',
            json={'name': 'Operations', 'identifier': 'operations', 'permissions': {}, 'is_active': True},
            headers={'AtlasClaw-Authenticate': token},
        )
        duplicate_response = client.post(
            '/api/roles',
            json={'name': 'Operations 2', 'identifier': 'operations', 'permissions': {}, 'is_active': True},
            headers={'AtlasClaw-Authenticate': token},
        )

        assert duplicate_response.status_code == 409

        _cleanup_manager(manager)

    def test_non_admin_cannot_manage_roles(self, tmp_path):
        manager = _init_database_sync(tmp_path)
        client = _build_client(tmp_path, _get_auth_config())
        token = _login_as(client, 'regularuser', 'userpass123')

        response = client.get('/api/roles', headers={'AtlasClaw-Authenticate': token})
        assert response.status_code == 403

        _cleanup_manager(manager)

    def test_builtin_role_cannot_be_deleted(self, tmp_path):
        manager = _init_database_sync(tmp_path)
        client = _build_client(tmp_path, _get_auth_config())
        token = _login_as(client, 'admin', 'adminpass123')

        roles_response = client.get('/api/roles', headers={'AtlasClaw-Authenticate': token})
        admin_role = next(role for role in roles_response.json()['roles'] if role['identifier'] == 'admin')

        delete_response = client.delete(
            f"/api/roles/{admin_role['id']}",
            headers={'AtlasClaw-Authenticate': token},
        )

        assert delete_response.status_code == 400
        assert 'built-in' in delete_response.json()['detail'].lower()

        _cleanup_manager(manager)

    def test_assigned_role_cannot_be_deleted(self, tmp_path):
        manager = _init_database_sync(tmp_path)
        client = _build_client(tmp_path, _get_auth_config())
        token = _login_as(client, 'admin', 'adminpass123')

        create_response = client.post(
            '/api/roles',
            json={'name': 'Support', 'identifier': 'support', 'permissions': {}, 'is_active': True},
            headers={'AtlasClaw-Authenticate': token},
        )
        assert create_response.status_code == 201
        role_id = create_response.json()['id']

        users_response = client.get('/api/users?search=regularuser', headers={'AtlasClaw-Authenticate': token})
        user_id = users_response.json()['users'][0]['id']

        assign_response = client.put(
            f'/api/users/{user_id}',
            json={'roles': {'support': True}},
            headers={'AtlasClaw-Authenticate': token},
        )
        assert assign_response.status_code == 200

        delete_response = client.delete(f'/api/roles/{role_id}', headers={'AtlasClaw-Authenticate': token})
        assert delete_response.status_code == 400
        assert 'assigned' in delete_response.json()['detail'].lower()

        _cleanup_manager(manager)

    def test_module_permission_governor_can_only_edit_managed_modules(self, tmp_path):
        manager = _init_database_sync(tmp_path)
        client = _build_client(tmp_path, _get_auth_config())
        admin_token = _login_as(client, 'admin', 'adminpass123')
        admin_headers = {'AtlasClaw-Authenticate': admin_token}

        create_manager_resp = client.post(
            '/api/users',
            json={
                'username': 'rolemanager',
                'password': 'rolemanagerpass123',
                'display_name': 'Role Manager',
                'email': 'rolemanager@test.com',
                'roles': {},
                'is_active': True,
                'is_admin': False,
            },
            headers=admin_headers,
        )
        assert create_manager_resp.status_code == 201
        manager_user_id = create_manager_resp.json()['id']

        target_role_resp = client.post(
            '/api/roles',
            json={
                'name': 'Target Role',
                'identifier': 'target_role',
                'description': 'Role to be governed',
                'permissions': {},
                'is_active': True,
            },
            headers=admin_headers,
        )
        assert target_role_resp.status_code == 201
        target_role_id = target_role_resp.json()['id']

        governor_role_resp = client.post(
            '/api/roles',
            json={
                'name': 'User Permission Governor',
                'identifier': 'user_permission_governor',
                'description': 'Can govern user permission policies only.',
                'permissions': {
                    'users': {
                        'manage_permissions': True,
                    },
                    'roles': {
                        'view': True,
                    },
                },
                'is_active': True,
            },
            headers=admin_headers,
        )
        assert governor_role_resp.status_code == 201

        assign_governor_resp = client.put(
            f'/api/users/{manager_user_id}',
            json={'roles': {'user_permission_governor': True}},
            headers=admin_headers,
        )
        assert assign_governor_resp.status_code == 200

        manager_token = _login_as(client, 'rolemanager', 'rolemanagerpass123')
        manager_headers = {'AtlasClaw-Authenticate': manager_token}

        list_roles_resp = client.get('/api/roles?page=1&page_size=20', headers=manager_headers)
        assert list_roles_resp.status_code == 200

        manage_users_resp = client.put(
            f'/api/roles/{target_role_id}',
            json={
                'permissions': {
                    'users': {
                        'view': True,
                        'manage_permissions': True,
                    },
                },
            },
            headers=manager_headers,
        )
        assert manage_users_resp.status_code == 200
        assert manage_users_resp.json()['permissions']['users']['view'] is True
        assert manage_users_resp.json()['permissions']['users']['manage_permissions'] is True

        manage_skills_resp = client.put(
            f'/api/roles/{target_role_id}',
            json={
                'permissions': {
                    'skills': {
                        'module_permissions': {
                            'view': True,
                        },
                    },
                },
            },
            headers=manager_headers,
        )
        assert manage_skills_resp.status_code == 403
        assert 'skills' in manage_skills_resp.json()['detail'].lower()

        rename_role_resp = client.put(
            f'/api/roles/{target_role_id}',
            json={'description': 'Renamed by governor'},
            headers=manager_headers,
        )
        assert rename_role_resp.status_code == 403
        assert 'roles.edit' in rename_role_resp.json()['detail'].lower()

        _cleanup_manager(manager)

    def test_user_with_skill_view_can_list_skills(self, tmp_path):
        manager = _init_database_sync(tmp_path)
        client = _build_client(tmp_path, _get_auth_config())
        admin_token = _login_as(client, 'admin', 'adminpass123')
        admin_headers = {'AtlasClaw-Authenticate': admin_token}

        create_user_resp = client.post(
            '/api/users',
            json={
                'username': 'skillviewer',
                'password': 'skillviewerpass123',
                'display_name': 'Skill Viewer',
                'email': 'skillviewer@test.com',
                'roles': {},
                'is_active': True,
                'is_admin': False,
            },
            headers=admin_headers,
        )
        assert create_user_resp.status_code == 201
        skill_viewer_id = create_user_resp.json()['id']

        create_role_resp = client.post(
            '/api/roles',
            json={
                'name': 'Skill Viewer',
                'identifier': 'skill_viewer',
                'description': 'Can browse skill catalog.',
                'permissions': {
                    'skills': {
                        'module_permissions': {
                            'view': True,
                        },
                    },
                },
                'is_active': True,
            },
            headers=admin_headers,
        )
        assert create_role_resp.status_code == 201

        assign_role_resp = client.put(
            f"/api/users/{skill_viewer_id}",
            json={'roles': {'skill_viewer': True}},
            headers=admin_headers,
        )
        assert assign_role_resp.status_code == 200

        viewer_token = _login_as(client, 'skillviewer', 'skillviewerpass123')
        viewer_headers = {'AtlasClaw-Authenticate': viewer_token}

        skills_resp = client.get('/api/skills', headers=viewer_headers)
        assert skills_resp.status_code == 200
        assert isinstance(skills_resp.json()['skills'], list)

        regular_token = _login_as(client, 'regularuser', 'userpass123')
        regular_resp = client.get('/api/skills', headers={'AtlasClaw-Authenticate': regular_token})
        assert regular_resp.status_code == 403

        _cleanup_manager(manager)
