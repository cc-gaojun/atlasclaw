import { translateIfExists, updateContainerTranslations } from '../i18n.js'
import { showToast } from '../components/toast.js'
import { checkAuth } from '../auth.js'

const MODULES = [
  ['rbac', 'shield', 'roles.modules.rbac', 'Permission Governance', 'roles.modules.rbacDescription', 'Control who may edit permission models across workspace modules.'],
  ['skills', 'sparkles', 'roles.modules.skills', 'Skills', 'roles.modules.skillsDescription', 'Manage which skills this role can enable and use.'],
  ['channels', 'channels', 'roles.modules.channels', 'Channels', 'roles.modules.channelsDescription', 'Manage access to connection configuration and lifecycle actions.'],
  ['tokens', 'tokens', 'roles.modules.tokens', 'Tokens', 'roles.modules.tokensDescription', 'Control visibility and maintenance rights for provider credentials.'],
  ['agent_configs', 'sparkles', 'roles.modules.agentConfigs', 'Agent Templates', 'roles.modules.agentConfigsDescription', 'Manage reusable agent definitions, persona blocks, and runtime defaults.'],
  ['provider_configs', 'channels', 'roles.modules.providerConfigs', 'Provider Instances', 'roles.modules.providerConfigsDescription', 'Manage service provider instances and their configuration lifecycle.'],
  ['model_configs', 'tokens', 'roles.modules.modelConfigs', 'Model Configs', 'roles.modules.modelConfigsDescription', 'Control model catalog visibility and maintenance rights.'],
  ['users', 'users', 'roles.modules.users', 'Users', 'roles.modules.usersDescription', 'Define how this role may browse and administer workspace users.'],
  ['roles', 'shield', 'roles.modules.roles', 'Roles', 'roles.modules.rolesDescription', 'Set whether this role can create, edit, and retire other roles.']
].map(([id, icon, titleKey, fallbackTitle, descriptionKey, fallbackDescription]) => ({
  id,
  icon,
  titleKey,
  fallbackTitle,
  descriptionKey,
  fallbackDescription
}))

const MODULE_PERMISSION_DEFINITIONS = {
  rbac: [
    ['manage_permissions', 'roles.permissions.managePermissions', 'Manage Permissions', 'roles.rbacPermissions.managePermissionsDescription', 'Edit permission models across role-governed modules.']
  ],
  skills: [
    ['manage_permissions', 'roles.permissions.managePermissions', 'Manage Permissions', 'roles.skillPermissions.managePermissionsDescription', 'Edit which skill permissions roles may receive.']
  ],
  channels: [
    ['view', 'roles.permissions.view', 'View', 'roles.channelPermissions.viewDescription', 'Review connected channels, owners, and health status.'],
    ['create', 'roles.permissions.create', 'Create', 'roles.channelPermissions.createDescription', 'Create new channel connections for supported platforms.'],
    ['edit', 'roles.permissions.edit', 'Edit', 'roles.channelPermissions.editDescription', 'Update credentials, callback URLs, and runtime settings.'],
    ['delete', 'roles.permissions.delete', 'Delete', 'roles.channelPermissions.deleteDescription', 'Remove obsolete or compromised channel connections.'],
    ['manage_permissions', 'roles.permissions.managePermissions', 'Manage Permissions', 'roles.channelPermissions.managePermissionsDescription', 'Edit the channel permission model available to other roles.']
  ],
  tokens: [
    ['view', 'roles.permissions.view', 'View', 'roles.tokenPermissions.viewDescription', 'Inspect token names, providers, and current health status.'],
    ['create', 'roles.permissions.create', 'Create', 'roles.tokenPermissions.createDescription', 'Register new provider endpoints or API key entries.'],
    ['edit', 'roles.permissions.edit', 'Edit', 'roles.tokenPermissions.editDescription', 'Rotate API keys and update model routing metadata.'],
    ['delete', 'roles.permissions.delete', 'Delete', 'roles.tokenPermissions.deleteDescription', 'Retire tokens that should no longer be used by the workspace.'],
    ['manage_permissions', 'roles.permissions.managePermissions', 'Manage Permissions', 'roles.tokenPermissions.managePermissionsDescription', 'Edit the token permission model available to other roles.']
  ],
  agent_configs: [
    ['view', 'roles.permissions.view', 'View', 'roles.agentConfigPermissions.viewDescription', 'Inspect reusable agent templates, persona blocks, and activation status.'],
    ['create', 'roles.permissions.create', 'Create', 'roles.agentConfigPermissions.createDescription', 'Create new agent templates for assistants and workflow specialists.'],
    ['edit', 'roles.permissions.edit', 'Edit', 'roles.agentConfigPermissions.editDescription', 'Update agent identity, memory presets, and runtime defaults.'],
    ['delete', 'roles.permissions.delete', 'Delete', 'roles.agentConfigPermissions.deleteDescription', 'Remove agent templates that should no longer be available.'],
    ['manage_permissions', 'roles.permissions.managePermissions', 'Manage Permissions', 'roles.agentConfigPermissions.managePermissionsDescription', 'Edit the agent-template permission model available to other roles.']
  ],
  provider_configs: [
    ['view', 'roles.permissions.view', 'View', 'roles.providerConfigPermissions.viewDescription', 'Inspect registered provider instances and their ownership.'],
    ['create', 'roles.permissions.create', 'Create', 'roles.providerConfigPermissions.createDescription', 'Register new provider instances and baseline configuration.'],
    ['edit', 'roles.permissions.edit', 'Edit', 'roles.providerConfigPermissions.editDescription', 'Update provider configuration, credentials, and runtime options.'],
    ['delete', 'roles.permissions.delete', 'Delete', 'roles.providerConfigPermissions.deleteDescription', 'Remove provider instances that should no longer be used.'],
    ['manage_permissions', 'roles.permissions.managePermissions', 'Manage Permissions', 'roles.providerConfigPermissions.managePermissionsDescription', 'Edit the provider instance permission model available to other roles.']
  ],
  model_configs: [
    ['view', 'roles.permissions.view', 'View', 'roles.modelConfigPermissions.viewDescription', 'Inspect model catalog entries and routing metadata.'],
    ['create', 'roles.permissions.create', 'Create', 'roles.modelConfigPermissions.createDescription', 'Register new model definitions for supported providers.'],
    ['edit', 'roles.permissions.edit', 'Edit', 'roles.modelConfigPermissions.editDescription', 'Update model routing, capability metadata, and runtime defaults.'],
    ['delete', 'roles.permissions.delete', 'Delete', 'roles.modelConfigPermissions.deleteDescription', 'Retire model definitions that should no longer be selectable.'],
    ['manage_permissions', 'roles.permissions.managePermissions', 'Manage Permissions', 'roles.modelConfigPermissions.managePermissionsDescription', 'Edit the model configuration permission model available to other roles.']
  ],
  users: [
    ['view', 'roles.permissions.view', 'View', 'roles.userPermissions.viewDescription', 'Browse users, status, and the roles currently assigned to them.'],
    ['create', 'roles.permissions.create', 'Create', 'roles.userPermissions.createDescription', 'Invite or create new users directly from the admin workspace.'],
    ['edit', 'roles.permissions.edit', 'Edit', 'roles.userPermissions.editDescription', 'Change profiles, assigned roles, and authentication options.'],
    ['delete', 'roles.permissions.delete', 'Delete', 'roles.userPermissions.deleteDescription', 'Remove user accounts that no longer need workspace access.'],
    ['reset_password', 'roles.userPermissions.resetPassword', 'Reset Password', 'roles.userPermissions.resetPasswordDescription', 'Reset local account passwords during support and onboarding.'],
    ['assign_roles', 'roles.permissions.assignRoles', 'Assign Roles', 'roles.userPermissions.assignRolesDescription', 'Assign or remove roles for workspace users.'],
    ['manage_permissions', 'roles.permissions.managePermissions', 'Manage Permissions', 'roles.userPermissions.managePermissionsDescription', 'Edit the user permission model available to other roles.']
  ],
  roles: [
    ['view', 'roles.permissions.view', 'View', 'roles.rolePermissions.viewDescription', 'Inspect existing roles, summaries, and saved permission sets.'],
    ['create', 'roles.permissions.create', 'Create', 'roles.rolePermissions.createDescription', 'Create new custom roles for teams and projects.'],
    ['edit', 'roles.permissions.edit', 'Edit', 'roles.rolePermissions.editDescription', 'Adjust permission bundles and module defaults.'],
    ['delete', 'roles.permissions.delete', 'Delete', 'roles.rolePermissions.deleteDescription', 'Retire roles that are no longer assigned anywhere.']
  ]
}

const ICONS = {
  sparkles: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="m12 3 1.6 3.9L18 8.5l-3.6 2.3 1.1 4.2L12 12.9 8.5 15l1.1-4.2L6 8.5l4.4-1.6L12 3Z"></path></svg>',
  channels: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M5 8h14"></path><path d="M5 16h14"></path><circle cx="8" cy="8" r="2"></circle><circle cx="16" cy="16" r="2"></circle></svg>',
  tokens: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M7 14a5 5 0 1 1 0-10h7a4 4 0 1 1 0 8H7"></path><path d="M7 10h10"></path><path d="M7 18h5"></path></svg>',
  users: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M17 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"></path><circle cx="9.5" cy="7" r="3.5"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a3.5 3.5 0 0 1 0 6.74"></path></svg>',
  shield: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3 5 6v6c0 5 3.4 8.6 7 9 3.6-.4 7-4 7-9V6l-7-3Z"></path><path d="m9.5 12 1.7 1.7 3.8-4"></path></svg>',
  plus: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14"></path><path d="M5 12h14"></path></svg>',
  trash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M3 6h18"></path><path d="M8 6V4h8v2"></path><path d="m19 6-1 14H6L5 6"></path></svg>',
  search: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="11" cy="11" r="7"></circle><path d="m20 20-3.5-3.5"></path></svg>'
}

let container = null
let rolesListEl = null
let editorEl = null
let deleteModal = null
let roles = []
let skills = []
let selectedRoleId = null
let roleSearch = ''
let skillSearch = ''
let activeModuleId = 'skills'
let draftRoleState = null
let identifierTouched = false
let eventCleanupFns = []

const PAGE_HTML = `
<div class="role-management-page">
  <div class="role-management-shell">
    <header class="role-management-page-header">
      <div>
        <span class="role-management-eyebrow" data-i18n="roles.eyebrow">Access Design</span>
        <h1 data-i18n="roles.title">Role Management</h1>
        <p data-i18n="roles.description">Design reusable permission bundles for channels, users, tokens, and skills from one editor.</p>
      </div>
      <button type="button" class="btn-primary role-management-create-btn" id="createRoleBtn">
        ${ICONS.plus}
        <span data-i18n="roles.createButton">Create Role</span>
      </button>
    </header>
    <div class="role-management-workspace">
      <aside class="role-management-sidebar">
        <label class="role-search-shell" for="roleSearchInput">
          ${ICONS.search}
          <input type="text" id="roleSearchInput" data-i18n-placeholder="roles.searchPlaceholder" placeholder="Search roles...">
        </label>
        <div class="role-management-list" id="roleList"></div>
      </aside>
      <section class="role-management-editor-shell" id="roleEditor"></section>
    </div>
  </div>
</div>
<div id="deleteRoleModal" class="modal-overlay hidden">
  <div class="modal role-delete-modal">
    <div class="modal-header">
      <div>
        <h2 data-i18n="roles.deleteTitle">Delete Role</h2>
        <p class="modal-description" data-i18n="roles.deleteDescription">This role will be removed permanently if it is no longer assigned to any user.</p>
      </div>
      <button type="button" class="modal-close" id="deleteRoleClose">&times;</button>
    </div>
    <div class="modal-body">
      <p class="confirm-message"><span data-i18n="roles.deleteConfirm">Are you sure you want to delete this role?</span></p>
      <strong class="delete-target-name" id="deleteRoleName"></strong>
      <input type="hidden" id="deleteRoleId" value="">
    </div>
    <div class="modal-footer">
      <button type="button" class="btn-secondary" id="deleteRoleCancel" data-i18n="roles.cancel">Cancel</button>
      <button type="button" class="btn-danger" id="deleteRoleConfirm" data-i18n="roles.deleteButton">Delete Role</button>
    </div>
  </div>
</div>
`

function translateOrFallback(key, fallback) {
  return translateIfExists(key) || fallback
}

function getBuiltinRoleTranslation(role, field) {
  if (!role?.is_builtin || !role?.identifier) return null
  return translateIfExists(`roles.builtinRoleCatalog.${role.identifier}.${field}`)
}

function getRoleDisplayName(role) {
  return getBuiltinRoleTranslation(role, 'name') || role?.name || ''
}

function getRoleDisplayDescription(role) {
  return getBuiltinRoleTranslation(role, 'description') || role?.description || ''
}

function getSkillTranslation(skill, field) {
  const skillName = skill?.skill_id || skill?.skill_name || skill?.name
  if (!skillName) return null
  return (
    translateIfExists(`roles.skillCatalog.${skillName}.${field}`)
    || translateIfExists(`roles.skillCatalog.${skillName.split(':').pop()}.${field}`)
  )
}

function getSkillDisplayName(skill) {
  return getSkillTranslation(skill, 'name') || skill?.skill_name || skill?.name || ''
}

function getSkillDisplayDescription(skill) {
  return (
    getSkillTranslation(skill, 'description')
    || skill?.description
    || translateOrFallback('roles.defaultSkillDescription', 'Workspace skill available to AtlasClaw agents.')
  )
}

function addTrackedListener(element, event, handler, options) {
  if (!element) return
  element.addEventListener(event, handler, options)
  eventCleanupFns.push(() => element.removeEventListener(event, handler, options))
}

function cloneData(value) {
  return JSON.parse(JSON.stringify(value))
}

function escapeHtml(str) {
  if (!str) return ''
  const div = document.createElement('div')
  div.textContent = str
  return div.innerHTML
}

function slugifyIdentifier(value) {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

function buildDefaultPermissions() {
  return {
    rbac: { manage_permissions: false },
    skills: { module_permissions: { view: false, enable_disable: false, manage_permissions: false }, skill_permissions: [] },
    channels: { view: false, create: false, edit: false, delete: false, manage_permissions: false },
    tokens: { view: false, create: false, edit: false, delete: false, manage_permissions: false },
    agent_configs: { view: false, create: false, edit: false, delete: false, manage_permissions: false },
    provider_configs: { view: false, create: false, edit: false, delete: false, manage_permissions: false },
    model_configs: { view: false, create: false, edit: false, delete: false, manage_permissions: false },
    users: { view: false, create: false, edit: false, delete: false, reset_password: false, assign_roles: false, manage_permissions: false },
    roles: { view: false, create: false, edit: false, delete: false }
  }
}

function buildAllEnabledPermissions() {
  const permissions = buildDefaultPermissions()
  Object.entries(permissions).forEach(([moduleId, config]) => {
    if (moduleId === 'rbac') {
      config.manage_permissions = true
      return
    }
    if (moduleId === 'skills') {
      config.module_permissions.view = true
      config.module_permissions.enable_disable = true
      config.module_permissions.manage_permissions = true
      return
    }
    Object.keys(config).forEach(key => {
      config[key] = true
    })
  })
  return permissions
}

function getRoleTemplate(identifier = '') {
  if (identifier === 'admin') return buildAllEnabledPermissions()
  if (identifier === 'user') {
    const permissions = buildDefaultPermissions()
    permissions.skills.module_permissions.view = true
    return permissions
  }
  if (identifier === 'viewer') {
    const permissions = buildDefaultPermissions()
    permissions.skills.module_permissions.view = true
    permissions.channels.view = true
    permissions.tokens.view = true
    permissions.users.view = true
    permissions.roles.view = true
    return permissions
  }
  return buildDefaultPermissions()
}

function normalizeRole(role, skillCatalog = []) {
  const normalized = {
    id: role?.id || '',
    name: role?.name || '',
    identifier: role?.identifier || '',
    description: role?.description || '',
    is_active: role?.is_active !== false,
    is_builtin: role?.is_builtin === true,
    isNew: role?.isNew === true,
    permissions: cloneData(role?.permissions || getRoleTemplate(role?.identifier))
  }
  const basePermissions = buildDefaultPermissions()
  normalized.permissions = {
    ...basePermissions,
    ...normalized.permissions,
    skills: {
      ...basePermissions.skills,
      ...(normalized.permissions.skills || {}),
      module_permissions: {
        ...basePermissions.skills.module_permissions,
        ...(normalized.permissions.skills?.module_permissions || {})
      },
      skill_permissions: Array.isArray(normalized.permissions.skills?.skill_permissions)
        ? normalized.permissions.skills.skill_permissions
        : []
    }
  }
  const storedSkillPermissions = new Map(
    normalized.permissions.skills.skill_permissions.map(skill => [skill.skill_id || skill.skill_name, skill])
  )
  normalized.permissions.skills.skill_permissions = skillCatalog.map(skill => {
    const existing = storedSkillPermissions.get(skill.name) || {}
    return {
      skill_id: skill.name,
      skill_name: skill.name,
      description: skill.description || '',
      authorized: existing.authorized === true,
      enabled: existing.enabled === true
    }
  })
  return normalized
}

function createEmptyRole() {
  return normalizeRole({
    id: 'new-role',
    name: '',
    identifier: '',
    description: '',
    is_active: true,
    is_builtin: false,
    isNew: true,
    permissions: getRoleTemplate()
  }, skills)
}

function fetchJson(url, options = {}) {
  return fetch(url, options).then(async response => {
    if (!response.ok) {
      let detail = translateOrFallback('roles.loadFailed', 'Failed to load role management data')
      try {
        const payload = await response.json()
        detail = payload.detail || detail
      } catch (error) {
        // Ignore parsing failure and use fallback detail.
      }
      throw new Error(detail)
    }
    if (response.status === 204) return null
    return response.json()
  })
}

function countEnabledPermissions(role, moduleId) {
  const permissions = role?.permissions?.[moduleId]
  if (!permissions) return 0
  if (moduleId === 'skills') {
    const moduleFlags = [
      permissions.module_permissions?.view,
      permissions.module_permissions?.manage_permissions
    ].filter(Boolean).length
    const skillFlags = (permissions.skill_permissions || []).filter(skill => skill.enabled).length
    return moduleFlags + skillFlags
  }
  return Object.values(permissions).filter(Boolean).length
}

function countModuleSummaryEnabledPermissions(role, moduleId) {
  const permissions = role?.permissions?.[moduleId]
  if (!permissions) return 0
  if (moduleId === 'skills') {
    const governanceFlags = permissions.module_permissions?.manage_permissions ? 1 : 0
    const skillFlags = (permissions.skill_permissions || []).filter(skill => skill.enabled).length
    return governanceFlags + skillFlags
  }
  return countEnabledPermissions(role, moduleId)
}

function getVisibleRoles() {
  const items = draftRoleState?.isNew ? [draftRoleState, ...roles] : roles
  const search = roleSearch.trim().toLowerCase()
  if (!search) return items
  return items.filter(role => {
    const searchContent = [
      role.name,
      role.identifier,
      role.description || '',
      getRoleDisplayName(role),
      getRoleDisplayDescription(role)
    ].join(' ').toLowerCase()
    return searchContent.includes(search)
  })
}

function renderRoleList() {
  const visibleRoles = getVisibleRoles()
  if (!visibleRoles.length) {
    rolesListEl.innerHTML = `
      <div class="role-list-empty">
        <strong data-i18n="roles.noRolesTitle">No roles found</strong>
        <p data-i18n="roles.noRolesDescription">Try a different search term or create a new custom role.</p>
      </div>
    `
    return
  }

  rolesListEl.innerHTML = visibleRoles.map(role => {
    const isSelected = role.id === selectedRoleId || (role.isNew && selectedRoleId === 'new-role')
    const enabledCount = MODULES.reduce((sum, module) => sum + countEnabledPermissions(role, module.id), 0)
    const roleDisplayName = getRoleDisplayName(role)
    const roleDisplayDescription = getRoleDisplayDescription(role)
    return `
      <button type="button" class="role-list-card ${isSelected ? 'selected' : ''}" data-role-select="${escapeHtml(role.id || 'new-role')}">
        <div class="role-list-card-top">
          <div>
            <strong>${escapeHtml(roleDisplayName || translateOrFallback('roles.newRole', 'New Role'))}</strong>
            <span>${escapeHtml(role.identifier || translateOrFallback('roles.unsavedIdentifier', 'unsaved-role'))}</span>
          </div>
          ${role.is_builtin ? '<span class="role-chip builtin" data-i18n="roles.builtinBadge">Built-in</span>' : ''}
          ${role.isNew ? '<span class="role-chip draft" data-i18n="roles.draftBadge">Draft</span>' : ''}
        </div>
        <p>${escapeHtml(roleDisplayDescription || translateOrFallback('roles.defaultRoleDescription', 'Permission bundle for workspace access control.'))}</p>
        <div class="role-list-card-meta">
          <span>${enabledCount} ${escapeHtml(translateOrFallback('roles.enabledShort', 'enabled'))}</span>
          <span class="role-status ${role.is_active ? 'active' : 'inactive'}">${escapeHtml(role.is_active ? translateOrFallback('roles.statusActive', 'Active') : translateOrFallback('roles.statusInactive', 'Inactive'))}</span>
        </div>
      </button>
    `
  }).join('')
}

function renderPermissionCards(moduleId) {
  const definitions = MODULE_PERMISSION_DEFINITIONS[moduleId] || []
  const permissionState = draftRoleState.permissions[moduleId] || {}
  return `
    <div class="role-permission-grid">
      ${definitions.map(([id, titleKey, fallbackTitle, descriptionKey, fallbackDescription]) => `
        <label class="role-permission-card">
          <div class="role-permission-card-copy">
            <span class="role-permission-badge" data-i18n="roles.permissionBadge">Permission</span>
            <strong>${escapeHtml(translateOrFallback(titleKey, fallbackTitle))}</strong>
            <p>${escapeHtml(translateOrFallback(descriptionKey, fallbackDescription))}</p>
          </div>
          <span class="toggle-switch">
            <input type="checkbox" data-module-toggle="${escapeHtml(moduleId)}" data-permission-toggle="${escapeHtml(id)}" ${permissionState[id] ? 'checked' : ''}>
            <span></span>
          </span>
        </label>
      `).join('')}
    </div>
  `
}

function getFilteredSkillRows(skillPermissions = []) {
  const search = skillSearch.trim().toLowerCase()
  return skillPermissions.filter(skill => {
    if (!search) return true
    const searchContent = [
      skill.skill_name,
      skill.description || '',
      getSkillDisplayName(skill),
      getSkillDisplayDescription(skill)
    ].join(' ').toLowerCase()
    return searchContent.includes(search)
  })
}

function syncSkillModulePermissions() {
  if (!draftRoleState) return
  const modulePermissions = draftRoleState.permissions.skills.module_permissions || {}
  const skillPermissions = draftRoleState.permissions.skills.skill_permissions || []
  const hasConfiguredSkills = skillPermissions.some(skill => skill.authorized || skill.enabled)
  const hasEnabledSkills = skillPermissions.some(skill => skill.enabled)

  modulePermissions.view = hasConfiguredSkills
  modulePermissions.enable_disable = hasEnabledSkills
}

function renderSkillsModule() {
  const permissions = draftRoleState.permissions.skills || { module_permissions: {}, skill_permissions: [] }
  const skillRows = getFilteredSkillRows(permissions.skill_permissions || [])
  const allVisibleSkillsEnabled = skillRows.length > 0 && skillRows.every(skill => skill.enabled)

  return `
    <div class="role-permission-grid role-permission-grid-skills">
      ${MODULE_PERMISSION_DEFINITIONS.skills.map(([id, titleKey, fallbackTitle, descriptionKey, fallbackDescription]) => `
        <label class="role-permission-card">
          <div class="role-permission-card-copy">
            <span class="role-permission-badge" data-i18n="roles.permissionBadge">Permission</span>
            <strong>${escapeHtml(translateOrFallback(titleKey, fallbackTitle))}</strong>
            <p>${escapeHtml(translateOrFallback(descriptionKey, fallbackDescription))}</p>
          </div>
          <span class="toggle-switch">
            <input type="checkbox" data-module-toggle="skills" data-permission-toggle="${escapeHtml(id)}" ${permissions.module_permissions?.[id] ? 'checked' : ''}>
            <span></span>
          </span>
        </label>
      `).join('')}
    </div>
    <section class="role-skill-section">
      <div class="role-skill-toolbar">
        <div>
          <h3 data-i18n="roles.skillsListTitle">Skill Management</h3>
          <p data-i18n="roles.skillsListDescription">Search the live skill catalog and decide which skills this role can enable.</p>
        </div>
        <label class="role-skill-search" for="skillsSearchInput">
          ${ICONS.search}
          <input type="text" id="skillsSearchInput" value="${escapeHtml(skillSearch)}" data-i18n-placeholder="roles.skillSearchPlaceholder" placeholder="Search skills...">
        </label>
      </div>
      <div class="role-skill-note">
        <div class="role-skill-note-copy">
          <strong data-i18n="roles.skillBehaviorTitle">New skill behavior</strong>
          <p data-i18n="roles.skillBehaviorDescription">Use the master switch to enable or disable all visible skills below at once.</p>
        </div>
        <label class="role-skill-master-toggle">
          <span class="role-skill-master-label" data-i18n="roles.enableAllToggle">Enable all</span>
          <span class="toggle-switch">
            <input type="checkbox" data-skill-master-toggle="enabled" ${allVisibleSkillsEnabled ? 'checked' : ''}>
            <span></span>
          </span>
        </label>
      </div>
      <div class="role-skill-list">
        ${skillRows.length ? skillRows.map(skill => {
          const skillDisplayName = getSkillDisplayName(skill)
          const skillDisplayDescription = getSkillDisplayDescription(skill)
          return `
          <div class="role-skill-card">
            <div class="role-skill-copy">
              <div class="role-skill-header">
                <strong>${escapeHtml(skillDisplayName)}</strong>
              </div>
              <p>${escapeHtml(skillDisplayDescription)}</p>
            </div>
            <div class="role-skill-controls">
              <label class="role-inline-toggle">
                <span data-i18n="roles.enableToggle">Enable</span>
                <span class="toggle-switch compact">
                  <input type="checkbox" data-skill-id="${escapeHtml(skill.skill_id)}" data-skill-toggle="enabled" ${skill.enabled ? 'checked' : ''}>
                  <span></span>
                </span>
              </label>
            </div>
          </div>
        `}).join('') : `
          <div class="role-list-empty role-list-empty-compact">
            <strong data-i18n="roles.noSkillsTitle">No skills match</strong>
            <p data-i18n="roles.noSkillsDescription">Adjust the search term to review more skills.</p>
          </div>
        `}
      </div>
    </section>
  `
}

function renderEditor() {
  if (!draftRoleState) {
    editorEl.innerHTML = `
      <div class="role-editor-empty">
        <strong data-i18n="roles.emptyTitle">Select a role to begin</strong>
        <p data-i18n="roles.emptyDescription">Choose a built-in role or create a new one to edit the right-side permission designer.</p>
      </div>
    `
    return
  }

  const module = MODULES.find(item => item.id === activeModuleId) || MODULES[0]
  const moduleMarkup = activeModuleId === 'skills' ? renderSkillsModule() : renderPermissionCards(activeModuleId)
  const roleDisplayName = getRoleDisplayName(draftRoleState) || translateOrFallback('roles.untitledRole', 'Untitled Role')
  const roleDisplayDescription = getRoleDisplayDescription(draftRoleState)
  const roleNameValue = draftRoleState.is_builtin ? roleDisplayName : draftRoleState.name
  const roleDescriptionValue = draftRoleState.is_builtin ? roleDisplayDescription : draftRoleState.description
  const builtinReadonlyAttr = draftRoleState.is_builtin ? 'readonly' : ''

  editorEl.innerHTML = `
    <div class="role-editor-shell">
      <div class="role-editor-topbar">
        <div>
          <span class="role-editor-breadcrumb">${escapeHtml(translateOrFallback('roles.title', 'Role Management'))} / ${escapeHtml(draftRoleState.isNew ? translateOrFallback('roles.createTitle', 'Create Role') : translateOrFallback('roles.editTitle', 'Edit Role'))}</span>
          <h2>${escapeHtml(roleDisplayName)}</h2>
        </div>
      </div>
      <section class="role-summary-card">
        <div class="role-summary-grid">
          <label class="form-field">
            <span data-i18n="roles.roleName">Role Name</span>
            <input type="text" data-role-field="name" value="${escapeHtml(roleNameValue)}" placeholder="${escapeHtml(translateOrFallback('roles.roleNamePlaceholder', 'Role name'))}" ${builtinReadonlyAttr}>
          </label>
          <div class="form-field">
            <span data-i18n="roles.roleStatus">Status</span>
            <div class="role-summary-meta-row role-summary-status-row">
              <span class="role-chip ${draftRoleState.is_active ? 'success' : 'muted'}">${escapeHtml(draftRoleState.is_active ? translateOrFallback('roles.statusActive', 'Active') : translateOrFallback('roles.statusInactive', 'Inactive'))}</span>
              <span class="toggle-switch">
                <input type="checkbox" data-role-field="is_active" ${draftRoleState.is_active ? 'checked' : ''}>
                <span></span>
              </span>
            </div>
          </div>
          <label class="form-field form-field-full">
            <span data-i18n="roles.roleDescription">Description</span>
            <textarea data-role-field="description" rows="3" placeholder="${escapeHtml(translateOrFallback('roles.roleDescriptionPlaceholder', 'Describe when this role should be assigned.'))}" ${builtinReadonlyAttr}>${escapeHtml(roleDescriptionValue)}</textarea>
          </label>
          <div class="role-summary-meta form-field form-field-full">
            <span data-i18n="roles.roleIdentifier">Role Identifier</span>
            ${draftRoleState.is_builtin
              ? `<div class="role-summary-meta-row"><span class="role-chip builtin" data-i18n="roles.builtinBadge">Built-in</span><span>${escapeHtml(translateOrFallback('roles.identifierPrefix', 'Identifier:'))} ${escapeHtml(draftRoleState.identifier)}</span></div>`
              : `<input type="text" data-role-field="identifier" value="${escapeHtml(draftRoleState.identifier)}" placeholder="${escapeHtml(translateOrFallback('roles.identifierPlaceholder', 'role-identifier'))}">`
            }
          </div>
        </div>
      </section>
      <div class="role-designer-layout">
        <aside class="role-module-nav">
          ${MODULES.map(item => `
            <button type="button" class="role-module-nav-item ${item.id === activeModuleId ? 'active' : ''}" data-module-id="${escapeHtml(item.id)}">
              <span class="role-module-icon">${ICONS[item.icon] || ICONS.shield}</span>
              <span class="role-module-copy">
                <strong>${escapeHtml(translateOrFallback(item.titleKey, item.fallbackTitle))}</strong>
                <span>${countModuleSummaryEnabledPermissions(draftRoleState, item.id)} ${escapeHtml(translateOrFallback('roles.enabledShort', 'enabled'))}</span>
              </span>
            </button>
          `).join('')}
        </aside>
        <section class="role-module-panel">
          <div class="role-module-panel-header">
            <div>
              <h3>${escapeHtml(translateOrFallback(module.titleKey, module.fallbackTitle))}</h3>
              <p>${escapeHtml(translateOrFallback(module.descriptionKey, module.fallbackDescription))}</p>
            </div>
            <div class="role-module-panel-actions">
              <button type="button" class="btn-secondary" data-module-action="select-all" data-i18n="roles.selectAll">Select All</button>
              <button type="button" class="btn-secondary" data-module-action="restore-defaults" data-i18n="roles.restoreDefaults">Restore Defaults</button>
            </div>
          </div>
          ${moduleMarkup}
        </section>
      </div>
      <div class="role-editor-footer">
        <div class="role-editor-footer-meta">
          <span class="role-chip ${draftRoleState.is_active ? 'success' : 'muted'}">${escapeHtml(draftRoleState.is_active ? translateOrFallback('roles.statusActive', 'Active') : translateOrFallback('roles.statusInactive', 'Inactive'))}</span>
          <span class="role-editor-footer-note">${escapeHtml(draftRoleState.isNew ? translateOrFallback('roles.createTitle', 'Create Role') : translateOrFallback('roles.editTitle', 'Edit Role'))}</span>
        </div>
        <div class="role-editor-actions role-editor-actions-footer">
          ${!draftRoleState.is_builtin && !draftRoleState.isNew ? `<button type="button" class="btn-secondary role-delete-trigger" id="deleteRoleTrigger">${ICONS.trash}<span data-i18n="roles.deleteButton">Delete Role</span></button>` : ''}
          <button type="button" class="btn-secondary" id="cancelRoleChanges" data-i18n="roles.cancel">Cancel</button>
          <button type="button" class="btn-primary" id="saveRoleChanges" data-i18n="roles.saveButton">Save Role</button>
        </div>
      </div>
    </div>
  `
}

function renderPage() {
  renderRoleList()
  renderEditor()
  updateContainerTranslations(container)
}

async function loadSkills() {
  try {
    const data = await fetchJson('/api/skills')
    skills = Array.isArray(data?.skills) ? data.skills : []
  } catch (error) {
    skills = []
    console.warn('[RoleManagement] Failed to load skills:', error)
  }
}

async function loadRoles(preserveRoleId = selectedRoleId) {
  const data = await fetchJson('/api/roles?page=1&page_size=100')
  roles = Array.isArray(data?.roles) ? data.roles.map(role => normalizeRole(role, skills)) : []

  if (preserveRoleId === 'new-role' && draftRoleState?.isNew) {
    selectedRoleId = 'new-role'
    return
  }

  const currentRole = roles.find(role => role.id === preserveRoleId)
  if (currentRole) {
    selectedRoleId = currentRole.id
    draftRoleState = normalizeRole(currentRole, skills)
    identifierTouched = true
    return
  }

  if (roles.length) {
    selectedRoleId = roles[0].id
    draftRoleState = normalizeRole(roles[0], skills)
    identifierTouched = true
    return
  }

  selectedRoleId = null
  draftRoleState = null
  identifierTouched = false
}

function selectRole(roleId) {
  if (roleId === 'new-role' && draftRoleState?.isNew) {
    selectedRoleId = 'new-role'
    renderPage()
    return
  }

  const role = roles.find(item => item.id === roleId)
  if (!role) return
  selectedRoleId = role.id
  activeModuleId = 'skills'
  skillSearch = ''
  draftRoleState = normalizeRole(role, skills)
  identifierTouched = true
  renderPage()
}

function updateDraftField(field, value, shouldRender = true) {
  if (!draftRoleState) return

  if (field === 'is_active') {
    draftRoleState.is_active = value === true
  } else if (field === 'name') {
    draftRoleState.name = value
    if (!identifierTouched && !draftRoleState.is_builtin) {
      draftRoleState.identifier = slugifyIdentifier(value)
    }
  } else if (field === 'identifier') {
    identifierTouched = true
    draftRoleState.identifier = slugifyIdentifier(value)
  } else {
    draftRoleState[field] = value
  }

  if (shouldRender) renderPage()
}

function toggleModulePermission(moduleId, permissionId, checked) {
  if (!draftRoleState) return
  if (moduleId === 'skills') {
    draftRoleState.permissions.skills.module_permissions[permissionId] = checked
  } else {
    draftRoleState.permissions[moduleId][permissionId] = checked
  }
  renderPage()
}

function toggleSkillPermission(skillId, property, checked) {
  if (!draftRoleState) return
  const skill = draftRoleState.permissions.skills.skill_permissions.find(item => item.skill_id === skillId)
  if (!skill) return
  if (property === 'enabled') {
    skill.enabled = checked
    skill.authorized = checked
  } else {
    if (property === 'authorized' && !checked) skill.enabled = false
    skill[property] = checked
  }
  syncSkillModulePermissions()
  renderPage()
}

function toggleAllVisibleSkills(checked) {
  if (!draftRoleState) return
  const visibleSkillIds = new Set(getFilteredSkillRows(draftRoleState.permissions.skills.skill_permissions).map(skill => skill.skill_id))
  draftRoleState.permissions.skills.skill_permissions = draftRoleState.permissions.skills.skill_permissions.map(skill => {
    if (!visibleSkillIds.has(skill.skill_id)) {
      return skill
    }
    return {
      ...skill,
      authorized: checked,
      enabled: checked
    }
  })
  syncSkillModulePermissions()
  renderPage()
}

function applyModuleAction(action) {
  if (!draftRoleState) return
  if (action === 'restore-defaults') {
    const template = getRoleTemplate(draftRoleState.identifier)
    if (activeModuleId === 'skills') {
      draftRoleState.permissions.skills = normalizeRole({ permissions: { skills: template.skills } }, skills).permissions.skills
    } else {
      draftRoleState.permissions[activeModuleId] = cloneData(template[activeModuleId])
    }
  }
  if (action === 'select-all') {
    if (activeModuleId === 'skills') {
      Object.keys(draftRoleState.permissions.skills.module_permissions).forEach(key => {
        draftRoleState.permissions.skills.module_permissions[key] = true
      })
      draftRoleState.permissions.skills.skill_permissions = draftRoleState.permissions.skills.skill_permissions.map(skill => ({
        ...skill,
        authorized: true,
        enabled: true
      }))
      syncSkillModulePermissions()
    } else {
      Object.keys(draftRoleState.permissions[activeModuleId]).forEach(key => {
        draftRoleState.permissions[activeModuleId][key] = true
      })
    }
  }
  renderPage()
}

function createNewRole() {
  selectedRoleId = 'new-role'
  activeModuleId = 'skills'
  skillSearch = ''
  identifierTouched = false
  draftRoleState = createEmptyRole()
  renderPage()
}

function cancelDraftChanges() {
  if (!draftRoleState) return
  if (draftRoleState.isNew) {
    const firstRole = roles[0] || null
    if (!firstRole) {
      selectedRoleId = null
      draftRoleState = null
      renderPage()
      return
    }
    selectedRoleId = firstRole.id
    draftRoleState = normalizeRole(firstRole, skills)
  } else {
    const original = roles.find(item => item.id === draftRoleState.id)
    if (original) draftRoleState = normalizeRole(original, skills)
  }
  skillSearch = ''
  identifierTouched = true
  renderPage()
}

function openDeleteModal() {
  if (!draftRoleState || draftRoleState.is_builtin || draftRoleState.isNew) return
  container.querySelector('#deleteRoleId').value = draftRoleState.id
  container.querySelector('#deleteRoleName').textContent = draftRoleState.name
  deleteModal.classList.remove('hidden')
}

function closeDeleteModal() {
  deleteModal?.classList.add('hidden')
}

async function saveRole() {
  if (!draftRoleState) return
  const name = draftRoleState.name.trim()
  const identifier = draftRoleState.identifier.trim()
  if (!name || !identifier) {
    showToast(translateOrFallback('roles.validationRequired', 'Role name and identifier are required.'), 'error')
    return
  }

  const payload = {
    name,
    identifier,
    description: draftRoleState.description.trim(),
    is_active: draftRoleState.is_active,
    permissions: draftRoleState.permissions
  }
  const requestOptions = {
    method: draftRoleState.isNew ? 'POST' : 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  }
  const endpoint = draftRoleState.isNew ? '/api/roles' : `/api/roles/${draftRoleState.id}`

  try {
    const savedRole = await fetchJson(endpoint, requestOptions)
    showToast(translateOrFallback(draftRoleState.isNew ? 'roles.createSuccess' : 'roles.saveSuccess', draftRoleState.isNew ? 'Role created successfully' : 'Role saved successfully'), 'success')
    selectedRoleId = savedRole.id
    await loadRoles(savedRole.id)
    draftRoleState = normalizeRole(savedRole, skills)
    identifierTouched = true
    renderPage()
  } catch (error) {
    showToast(error.message || translateOrFallback('roles.saveFailed', 'Failed to save role'), 'error')
  }
}

async function confirmDeleteRole() {
  const roleId = container.querySelector('#deleteRoleId').value
  if (!roleId) return
  try {
    await fetchJson(`/api/roles/${roleId}`, { method: 'DELETE' })
    showToast(translateOrFallback('roles.deleteSuccess', 'Role deleted successfully'), 'success')
    closeDeleteModal()
    selectedRoleId = null
    draftRoleState = null
    identifierTouched = false
    await loadRoles()
    renderPage()
  } catch (error) {
    showToast(error.message || translateOrFallback('roles.deleteFailed', 'Failed to delete role'), 'error')
  }
}

function handleRoleListClick(event) {
  const target = event.target.closest('[data-role-select]')
  if (!target) return
  selectRole(target.getAttribute('data-role-select'))
}

function handleEditorClick(event) {
  if (event.target.closest('#deleteRoleTrigger')) {
    openDeleteModal()
    return
  }
  if (event.target.closest('#saveRoleChanges')) {
    saveRole()
    return
  }
  if (event.target.closest('#cancelRoleChanges')) {
    cancelDraftChanges()
    return
  }

  const moduleButton = event.target.closest('[data-module-id]')
  if (moduleButton) {
    activeModuleId = moduleButton.getAttribute('data-module-id')
    skillSearch = ''
    renderPage()
    return
  }

  const actionButton = event.target.closest('[data-module-action]')
  if (actionButton) {
    applyModuleAction(actionButton.getAttribute('data-module-action'))
  }
}

function handleEditorInput(event) {
  const field = event.target.getAttribute('data-role-field')
  if (field) {
    updateDraftField(field, field === 'is_active' ? event.target.checked : event.target.value, field === 'is_active')
    return
  }
  if (event.target.id === 'skillsSearchInput') {
    skillSearch = event.target.value
    renderPage()
  }
}

function handleEditorChange(event) {
  const field = event.target.getAttribute('data-role-field')
  if (field) {
    updateDraftField(field, field === 'is_active' ? event.target.checked : event.target.value)
    return
  }

  const moduleId = event.target.getAttribute('data-module-toggle')
  const permissionId = event.target.getAttribute('data-permission-toggle')
  if (moduleId && permissionId) {
    toggleModulePermission(moduleId, permissionId, event.target.checked)
    return
  }

  const skillId = event.target.getAttribute('data-skill-id')
  const skillToggle = event.target.getAttribute('data-skill-toggle')
  if (skillId && skillToggle) {
    toggleSkillPermission(skillId, skillToggle, event.target.checked)
    return
  }

  const masterSkillToggle = event.target.getAttribute('data-skill-master-toggle')
  if (masterSkillToggle === 'enabled') {
    toggleAllVisibleSkills(event.target.checked)
  }
}

function setupEventListeners() {
  addTrackedListener(container.querySelector('#createRoleBtn'), 'click', createNewRole)
  addTrackedListener(container.querySelector('#roleSearchInput'), 'input', event => {
    roleSearch = event.target.value
    renderRoleList()
    updateContainerTranslations(container)
  })
  addTrackedListener(rolesListEl, 'click', handleRoleListClick)
  addTrackedListener(editorEl, 'click', handleEditorClick)
  addTrackedListener(editorEl, 'input', handleEditorInput)
  addTrackedListener(editorEl, 'change', handleEditorChange)
  addTrackedListener(container.querySelector('#deleteRoleClose'), 'click', closeDeleteModal)
  addTrackedListener(container.querySelector('#deleteRoleCancel'), 'click', closeDeleteModal)
  addTrackedListener(container.querySelector('#deleteRoleConfirm'), 'click', confirmDeleteRole)
  addTrackedListener(deleteModal, 'click', event => {
    if (event.target === deleteModal) closeDeleteModal()
  })
  addTrackedListener(document, 'keydown', event => {
    if (event.key === 'Escape') closeDeleteModal()
  })
}

export async function mount(containerEl) {
  container = containerEl
  const user = await checkAuth({ redirect: true })
  if (!user) return
  if (!user.is_admin) {
    showToast(translateOrFallback('roles.accessDenied', 'Access denied. Admin privileges required.'), 'error')
    window.location.href = '/'
    return
  }

  if (!document.getElementById('role-management-page-css')) {
    const cssLink = document.createElement('link')
    cssLink.rel = 'stylesheet'
    cssLink.href = '/styles/role-management.css'
    cssLink.id = 'role-management-page-css'
    document.head.appendChild(cssLink)
  }

  container.innerHTML = PAGE_HTML
  rolesListEl = container.querySelector('#roleList')
  editorEl = container.querySelector('#roleEditor')
  deleteModal = container.querySelector('#deleteRoleModal')
  setupEventListeners()
  await loadSkills()
  await loadRoles()
  renderPage()
}

export async function unmount() {
  eventCleanupFns.forEach(fn => fn())
  eventCleanupFns = []
  document.getElementById('role-management-page-css')?.remove()
  container = null
  rolesListEl = null
  editorEl = null
  deleteModal = null
  roles = []
  skills = []
  selectedRoleId = null
  roleSearch = ''
  skillSearch = ''
  activeModuleId = 'skills'
  draftRoleState = null
  identifierTouched = false
}

export default { mount, unmount }
