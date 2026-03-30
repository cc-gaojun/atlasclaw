/**
 * header.js - Header Component
 *
 * Provides:
 * - renderHeader(container, { authInfo }) - Render centered title + user entry
 * - updateHeaderTitle(titleKey) - Update page title using i18n key
 */
import { t } from '../i18n.js'

// Store reference to header element for updates
let headerElement = null
let titleElement = null

/**
 * Render header into container
 * @param {HTMLElement} container - Container element
 */
export function renderHeader(container, { authInfo } = {}) {
  if (!container) {
    console.warn('[Header] No container provided')
    return
  }

  headerElement = container

  // Render header HTML
  const displayName = authInfo?.display_name || authInfo?.username || 'User'
  const initial = displayName.trim().charAt(0).toUpperCase() || 'U'
  container.innerHTML = `
    <div class="chat-header-spacer" aria-hidden="true"></div>
    <h1 id="page-title" class="chat-header-title" data-i18n="app.title">AtlasClaw</h1>
    <div class="header-actions">
      <div class="user-pill" title="${escapeHtml(displayName)}">
        <span class="user-pill-avatar">${escapeHtml(initial)}</span>
        <span class="user-pill-name">${escapeHtml(displayName)}</span>
      </div>
    </div>
  `

  titleElement = container.querySelector('#page-title')
}

export function updateHeaderTitleText(titleText) {
  if (!titleElement) {
    titleElement = document.getElementById('page-title')
  }

  if (!titleElement) {
    return
  }

  titleElement.removeAttribute('data-i18n')
  titleElement.textContent = titleText || 'AtlasClaw'
  document.title = titleElement.textContent
}

/**
 * Update header title
 * @param {string} titleKey - i18n key for title
 */
export function updateHeaderTitle(titleKey) {
  if (!titleElement) {
    titleElement = document.getElementById('page-title')
  }

  if (titleElement) {
    // Update the data-i18n attribute
    titleElement.setAttribute('data-i18n', titleKey)

    // Try to get translated text
    const translated = t(titleKey)
    if (translated && translated !== titleKey) {
      titleElement.textContent = translated
    } else {
      // Fallback to key's last part
      titleElement.textContent = getDefaultTitle(titleKey)
    }

    // Also update document title
    document.title = titleElement.textContent + ' - AtlasClaw'
  }
}

/**
 * Get default title for i18n key (fallback before translations load)
 * @param {string} key - i18n key
 * @returns {string}
 */
function getDefaultTitle(key) {
  const defaults = {
    'app.title': 'AtlasClaw',
    'app.chatTitle': 'Chat',
    'channel.title': 'Channel Management',
    'model.pageTitle': 'Model Management',
    'admin.title': 'User Management',
    'app.channels': 'Channels',
    'app.models': 'Models'
  }
  return defaults[key] || key.split('.').pop()
}

/**
 * Get header element
 * @returns {HTMLElement|null}
 */
export function getHeaderElement() {
  return headerElement
}

export default {
  renderHeader,
  updateHeaderTitle,
  updateHeaderTitleText,
  getHeaderElement
}

function escapeHtml(text) {
  return String(text || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;')
}
