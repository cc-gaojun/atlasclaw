/**
 * Admin Users Management Module
 * Handles CRUD operations for user management
 */

import { checkAuth, installAuthFetchInterceptor, logout } from './auth.js'

// State management
let currentPage = 1
let currentSearch = ''
let currentPageSize = 20
let totalUsers = 0
let searchDebounceTimer = null

/**
 * Convert roles dict to comma-separated string
 * @param {Object} roles - Roles dict like {admin: true, viewer: true}
 * @returns {string} Comma-separated role names
 */
function rolesDictToString(roles) {
  if (!roles || typeof roles !== 'object') return ''
  return Object.keys(roles).filter(key => roles[key]).join(', ')
}

/**
 * Convert comma-separated string to roles dict
 * @param {string} str - Comma-separated role names
 * @returns {Object} Roles dict with true values
 */
function rolesStringToDict(str) {
  const result = {}
  if (!str) return result
  str.split(',').map(r => r.trim()).filter(r => r).forEach(roleName => {
    result[roleName] = true
  })
  return result
}

// DOM Elements
let usersTableBody = null
let paginationInfo = null
let paginationBtns = null
let searchInput = null
let userModal = null
let deleteModal = null
let toastContainer = null

/**
 * Initialize the admin users page
 */
async function init() {
  // Install auth interceptor to auto-add JWT token
  installAuthFetchInterceptor()

  // Check if user is authenticated and is admin
  const user = await checkAuth({ redirect: true })
  if (!user) {
    return
  }

  if (!user.is_admin) {
    alert('Access denied. Admin privileges required.')
    window.location.href = '/'
    return
  }

  // Cache DOM elements
  usersTableBody = document.getElementById('usersTableBody')
  paginationInfo = document.getElementById('paginationInfo')
  paginationBtns = document.getElementById('paginationBtns')
  searchInput = document.getElementById('searchInput')
  userModal = document.getElementById('userModal')
  deleteModal = document.getElementById('deleteModal')
  toastContainer = document.getElementById('toastContainer')

  // Setup event listeners
  setupEventListeners()

  // Load initial data
  await loadUsers(currentPage, currentSearch)
}

/**
 * Setup all event listeners
 */
function setupEventListeners() {
  // Search input with debounce
  searchInput.addEventListener('input', (e) => {
    handleSearch(e.target.value)
  })

  // Create user button
  document.getElementById('createUserBtn').addEventListener('click', showCreateModal)

  // Modal close buttons
  document.getElementById('modalClose').addEventListener('click', closeModal)
  document.getElementById('modalCancel').addEventListener('click', closeModal)
  document.getElementById('deleteModalClose').addEventListener('click', closeDeleteModal)
  document.getElementById('deleteCancel').addEventListener('click', closeDeleteModal)

  // Form submit
  document.getElementById('userForm').addEventListener('submit', handleFormSubmit)

  // Delete confirm
  document.getElementById('deleteConfirm').addEventListener('click', handleDeleteConfirm)

  // Logout button
  document.getElementById('logoutBtn').addEventListener('click', () => logout())

  // Close modals on overlay click
  userModal.addEventListener('click', (e) => {
    if (e.target === userModal) closeModal()
  })
  deleteModal.addEventListener('click', (e) => {
    if (e.target === deleteModal) closeDeleteModal()
  })

  // Close modals on Escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeModal()
      closeDeleteModal()
    }
  })
}

// ===================
// API Calls
// ===================

/**
 * Load users from API with pagination and search
 */
async function loadUsers(page = 1, search = '') {
  try {
    const params = new URLSearchParams({
      page: page.toString(),
      page_size: currentPageSize.toString()
    })

    if (search) {
      params.append('search', search)
    }

    const response = await fetch(`/api/users?${params}`)

    if (!response.ok) {
      await handleApiError(response)
      return
    }

    const data = await response.json()
    const users = data.users || []
    totalUsers = data.total || 0

    renderUserTable(users, totalUsers)
    renderPagination(page, Math.ceil(totalUsers / currentPageSize))
  } catch (error) {
    console.error('[AdminUsers] Failed to load users:', error)
    showToast(error.message || 'Failed to load users', 'error')
    usersTableBody.innerHTML = '<tr><td colspan="7" class="loading-row" style="color: #e74c3c;">Failed to load users</td></tr>'
  }
}

/**
 * Create a new user
 */
async function createUser(formData) {
  const response = await fetch('/api/users', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(formData)
  })

  if (!response.ok) {
    await handleApiError(response)
    return null
  }

  return await response.json()
}

/**
 * Update an existing user
 */
async function updateUser(userId, formData) {
  const response = await fetch(`/api/users/${userId}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(formData)
  })

  if (!response.ok) {
    await handleApiError(response)
    return null
  }

  return await response.json()
}

/**
 * Delete a user
 */
async function deleteUser(userId) {
  const response = await fetch(`/api/users/${userId}`, {
    method: 'DELETE'
  })

  if (!response.ok) {
    await handleApiError(response)
    return false
  }

  return true
}

/**
 * Handle API errors
 */
async function handleApiError(response) {
  const status = response.status

  if (status === 401) {
    showToast('Session expired. Please login again.', 'error')
    setTimeout(() => {
      window.location.href = '/login.html'
    }, 1500)
    return
  }

  if (status === 403) {
    showToast('Access denied. Admin privileges required.', 'error')
    return
  }

  let errorMessage = 'An error occurred'
  try {
    const data = await response.json()
    errorMessage = data.detail || data.error || data.message || errorMessage
  } catch {
    // ignore parse error
  }

  if (status === 409) {
    errorMessage = 'User already exists with this username or email'
  } else if (status === 404) {
    errorMessage = 'User not found'
  }

  throw new Error(errorMessage)
}

// ===================
// UI Rendering
// ===================

/**
 * Render the user table
 */
function renderUserTable(users, total) {
  if (!users || users.length === 0) {
    usersTableBody.innerHTML = `
      <tr>
        <td colspan="7">
          <div class="empty-state">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
              <circle cx="9" cy="7" r="4"></circle>
              <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
              <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
            </svg>
            <p>No users found</p>
          </div>
        </td>
      </tr>
    `
    return
  }

  usersTableBody.innerHTML = users.map(user => `
    <tr data-user-id="${user.id}">
      <td><strong>${escapeHtml(user.username)}</strong></td>
      <td>${escapeHtml(user.display_name || '-')}</td>
      <td>${escapeHtml(user.email || '-')}</td>
      <td>${renderAuthTypeBadge(user.auth_type)}</td>
      <td>${renderStatusBadge(user.is_active)}</td>
      <td>${renderAdminBadge(user.is_admin)}</td>
      <td>
        <div class="action-btns">
          <button class="btn-icon btn-edit" title="Edit" data-user='${JSON.stringify(user).replace(/'/g, "&#39;")}'>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
              <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
            </svg>
          </button>
          <button class="btn-icon btn-delete" title="Delete" data-user-id="${user.id}" data-username="${escapeHtml(user.username)}">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="3 6 5 6 21 6"></polyline>
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
            </svg>
          </button>
        </div>
      </td>
    </tr>
  `).join('')

  // Attach event listeners to action buttons
  usersTableBody.querySelectorAll('.btn-edit').forEach(btn => {
    btn.addEventListener('click', () => {
      const user = JSON.parse(btn.dataset.user)
      showEditModal(user)
    })
  })

  usersTableBody.querySelectorAll('.btn-delete').forEach(btn => {
    btn.addEventListener('click', () => {
      showDeleteConfirm(btn.dataset.userId, btn.dataset.username)
    })
  })
}

/**
 * Render pagination controls
 */
function renderPagination(page, totalPages) {
  const start = totalUsers === 0 ? 0 : (page - 1) * currentPageSize + 1
  const end = Math.min(page * currentPageSize, totalUsers)

  paginationInfo.textContent = `Showing ${start}-${end} of ${totalUsers} users`

  if (totalPages <= 1) {
    paginationBtns.innerHTML = ''
    return
  }

  let html = ''

  // Previous button
  html += `<button class="pagination-btn" ${page === 1 ? 'disabled' : ''} data-page="${page - 1}">Prev</button>`

  // Page numbers
  const maxVisible = 5
  let startPage = Math.max(1, page - Math.floor(maxVisible / 2))
  let endPage = Math.min(totalPages, startPage + maxVisible - 1)

  if (endPage - startPage < maxVisible - 1) {
    startPage = Math.max(1, endPage - maxVisible + 1)
  }

  if (startPage > 1) {
    html += `<button class="pagination-btn" data-page="1">1</button>`
    if (startPage > 2) {
      html += `<span style="padding: 8px 4px; color: #888;">...</span>`
    }
  }

  for (let i = startPage; i <= endPage; i++) {
    html += `<button class="pagination-btn ${i === page ? 'active' : ''}" data-page="${i}">${i}</button>`
  }

  if (endPage < totalPages) {
    if (endPage < totalPages - 1) {
      html += `<span style="padding: 8px 4px; color: #888;">...</span>`
    }
    html += `<button class="pagination-btn" data-page="${totalPages}">${totalPages}</button>`
  }

  // Next button
  html += `<button class="pagination-btn" ${page === totalPages ? 'disabled' : ''} data-page="${page + 1}">Next</button>`

  paginationBtns.innerHTML = html

  // Attach event listeners
  paginationBtns.querySelectorAll('.pagination-btn:not(:disabled)').forEach(btn => {
    btn.addEventListener('click', () => {
      const newPage = parseInt(btn.dataset.page, 10)
      if (newPage !== currentPage) {
        handlePageChange(newPage)
      }
    })
  })
}

/**
 * Render status badge
 */
function renderStatusBadge(isActive) {
  if (isActive) {
    return '<span class="badge badge-active">Active</span>'
  }
  return '<span class="badge badge-inactive">Inactive</span>'
}

/**
 * Render admin badge
 */
function renderAdminBadge(isAdmin) {
  if (isAdmin) {
    return '<span class="badge badge-admin">Admin</span>'
  }
  return '-'
}

/**
 * Render auth type badge
 */
function renderAuthTypeBadge(authType) {
  const type = (authType || 'local').toLowerCase()
  if (type === 'sso') {
    return '<span class="badge badge-sso">SSO</span>'
  }
  return '<span class="badge badge-local">Local</span>'
}

// ===================
// Modal Management
// ===================

/**
 * Show create user modal
 */
function showCreateModal() {
  document.getElementById('modalTitle').textContent = 'Create User'
  document.getElementById('editUserId').value = ''
  document.getElementById('userForm').reset()
  document.getElementById('formIsActive').checked = true

  // Username is editable in create mode
  document.getElementById('formUsername').disabled = false

  // Password is required in create mode
  document.getElementById('passwordRequired').style.display = 'inline'
  document.getElementById('passwordHint').style.display = 'none'
  document.getElementById('formPassword').required = true

  userModal.classList.remove('hidden')
  document.getElementById('formUsername').focus()
}

/**
 * Show edit user modal with pre-filled data
 */
function showEditModal(user) {
  document.getElementById('modalTitle').textContent = 'Edit User'
  document.getElementById('editUserId').value = user.id

  // Pre-fill form
  document.getElementById('formUsername').value = user.username || ''
  document.getElementById('formDisplayName').value = user.display_name || ''
  document.getElementById('formEmail').value = user.email || ''
  document.getElementById('formPassword').value = ''
  document.getElementById('formAuthType').value = user.auth_type || 'local'
  document.getElementById('formRoles').value = rolesDictToString(user.roles)
  document.getElementById('formIsActive').checked = user.is_active !== false
  document.getElementById('formIsAdmin').checked = user.is_admin === true

  // Username is read-only in edit mode
  document.getElementById('formUsername').disabled = true

  // Password is optional in edit mode
  document.getElementById('passwordRequired').style.display = 'none'
  document.getElementById('passwordHint').style.display = 'block'
  document.getElementById('formPassword').required = false

  userModal.classList.remove('hidden')
  document.getElementById('formDisplayName').focus()
}

/**
 * Close user modal
 */
function closeModal() {
  userModal.classList.add('hidden')
  document.getElementById('userForm').reset()
}

/**
 * Show delete confirmation modal
 */
function showDeleteConfirm(userId, username) {
  document.getElementById('deleteUserId').value = userId
  document.getElementById('deleteUserName').textContent = username
  deleteModal.classList.remove('hidden')
}

/**
 * Close delete modal
 */
function closeDeleteModal() {
  deleteModal.classList.add('hidden')
}

// ===================
// Event Handlers
// ===================

/**
 * Handle search with debounce
 */
function handleSearch(query) {
  if (searchDebounceTimer) {
    clearTimeout(searchDebounceTimer)
  }

  searchDebounceTimer = setTimeout(() => {
    currentSearch = query
    currentPage = 1
    loadUsers(currentPage, currentSearch)
  }, 300)
}

/**
 * Handle page change
 */
function handlePageChange(page) {
  currentPage = page
  loadUsers(currentPage, currentSearch)
}

/**
 * Handle form submission for create/edit
 */
async function handleFormSubmit(event) {
  event.preventDefault()

  const submitBtn = document.getElementById('modalSubmit')
  const editUserId = document.getElementById('editUserId').value
  const isEdit = !!editUserId

  // Gather form data
  const formData = {
    email: document.getElementById('formEmail').value.trim() || null,
    display_name: document.getElementById('formDisplayName').value.trim() || null,
    auth_type: document.getElementById('formAuthType').value,
    roles: rolesStringToDict(document.getElementById('formRoles').value),
    is_active: document.getElementById('formIsActive').checked,
    is_admin: document.getElementById('formIsAdmin').checked
  }

  // Add username only in create mode
  if (!isEdit) {
    formData.username = document.getElementById('formUsername').value.trim()
  }

  // Add password if provided
  const password = document.getElementById('formPassword').value
  if (password) {
    formData.password = password
  } else if (!isEdit) {
    showToast('Password is required for new users', 'error')
    return
  }

  // Validate required fields
  if (!isEdit && !formData.username) {
    showToast('Username is required', 'error')
    return
  }

  // Disable submit button
  submitBtn.disabled = true
  submitBtn.textContent = isEdit ? 'Saving...' : 'Creating...'

  try {
    let result
    if (isEdit) {
      result = await updateUser(editUserId, formData)
    } else {
      result = await createUser(formData)
    }

    if (result) {
      showToast(isEdit ? 'User updated successfully' : 'User created successfully', 'success')
      closeModal()
      await loadUsers(currentPage, currentSearch)
    }
  } catch (error) {
    showToast(error.message, 'error')
  } finally {
    submitBtn.disabled = false
    submitBtn.textContent = 'Save'
  }
}

/**
 * Handle delete confirmation
 */
async function handleDeleteConfirm() {
  const userId = document.getElementById('deleteUserId').value
  const confirmBtn = document.getElementById('deleteConfirm')

  confirmBtn.disabled = true
  confirmBtn.textContent = 'Deleting...'

  try {
    const success = await deleteUser(userId)
    if (success) {
      showToast('User deleted successfully', 'success')
      closeDeleteModal()

      // If we deleted the last user on current page, go to previous page
      const remainingOnPage = totalUsers - 1 - (currentPage - 1) * currentPageSize
      if (remainingOnPage <= 0 && currentPage > 1) {
        currentPage--
      }

      await loadUsers(currentPage, currentSearch)
    }
  } catch (error) {
    showToast(error.message, 'error')
  } finally {
    confirmBtn.disabled = false
    confirmBtn.textContent = 'Delete'
  }
}

// ===================
// Toast Notifications
// ===================

/**
 * Show toast notification
 */
function showToast(message, type = 'success') {
  const toast = document.createElement('div')
  toast.className = `toast-item ${type}`
  toast.textContent = message
  toastContainer.appendChild(toast)

  // Auto-dismiss after 4 seconds
  setTimeout(() => {
    toast.style.opacity = '0'
    toast.style.transform = 'translateX(100%)'
    setTimeout(() => {
      toast.remove()
    }, 300)
  }, 4000)
}

// ===================
// Utilities
// ===================

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(str) {
  if (!str) return ''
  const div = document.createElement('div')
  div.textContent = str
  return div.innerHTML
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', init)
