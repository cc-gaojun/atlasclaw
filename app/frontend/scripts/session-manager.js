/**
 * Session State Management
 * Manage session lifecycle and persistence
 */

import { createThreadSession, getSessionHistory } from './api-client.js';

const SESSION_KEY_STORAGE = 'atlasclaw_session_key';

let currentSessionKey = null;

/**
 * Initialize session
 * Restore from sessionStorage or create new session
 * @param {object} params - Session parameters
 * @returns {Promise<string>} Session key
 */
export async function initSession(params = {}) {
    // Try to restore from sessionStorage
    const storedKey = sessionStorage.getItem(SESSION_KEY_STORAGE);
    
    if (storedKey) {
        currentSessionKey = storedKey;
        console.log('[Session] Restored:', currentSessionKey);
        return currentSessionKey;
    }
    
    // Create new session
    const session = await createThreadSession(params);
    currentSessionKey = session.session_key;
    sessionStorage.setItem(SESSION_KEY_STORAGE, currentSessionKey);
    console.log('[Session] Created:', currentSessionKey);
    
    return currentSessionKey;
}

/**
 * Get current session key
 * @returns {string|null} Session key
 */
export function getSessionKey() {
    if (!currentSessionKey) {
        currentSessionKey = sessionStorage.getItem(SESSION_KEY_STORAGE);
    }
    return currentSessionKey;
}

/**
 * Set session key (for session restoration)
 * @param {string} key - Session key
 */
export function setSessionKey(key) {
    currentSessionKey = key;
    if (key) {
        sessionStorage.setItem(SESSION_KEY_STORAGE, key);
    } else {
        sessionStorage.removeItem(SESSION_KEY_STORAGE);
    }
}

/**
 * Check if there is an active session
 * @returns {boolean}
 */
export function hasSession() {
    return !!getSessionKey();
}

/**
 * Clear current session and create a new independent thread
 * @param {boolean} archive - Unused compatibility argument
 * @param {object} params - New session parameters
 * @returns {Promise<string>} New session key
 */
export async function startNewSession(archive = true, params = {}) {
    const activeSessionKey = getSessionKey();

    if (activeSessionKey) {
        try {
            const historyPayload = await getSessionHistory(activeSessionKey);
            const messages = Array.isArray(historyPayload?.messages) ? historyPayload.messages : [];
            if (messages.length === 0) {
                return activeSessionKey;
            }
        } catch (error) {
            console.warn('[Session] Failed to inspect active session history before creating a new thread:', error);
        }
    }

    // Clear storage and create a brand-new thread while preserving history entries
    void archive;
    sessionStorage.removeItem(SESSION_KEY_STORAGE);
    currentSessionKey = null;

    return initSession(params);
}

/**
 * Clear session (local only)
 */
export function clearSession() {
    sessionStorage.removeItem(SESSION_KEY_STORAGE);
    currentSessionKey = null;
    console.log('[Session] Cleared');
}

export default {
    initSession,
    getSessionKey,
    setSessionKey,
    hasSession,
    startNewSession,
    clearSession
};
