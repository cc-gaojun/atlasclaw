beforeEach(() => {
  jest.resetModules()
  document.body.innerHTML = `
    <div id="sidebar-dynamic-content"></div>
    <div id="page-root"></div>
  `
  sessionStorage.clear()
  global.fetch = jest.fn((url, options = {}) => {
    const target = String(url)
    if (target.endsWith('/api/sessions/threads')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ session_key: 'session-a' })
      })
    }
    if (target.endsWith('/api/agent/info')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          name: 'AtlasClaw Enterprise AI Assistant',
          welcome_message: 'Welcome'
        })
      })
    }
    if (target.endsWith('/api/sessions/session-a/history')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ messages: [] })
      })
    }
    if (target.endsWith('/api/sessions')) {
      if (options.method === 'POST') {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ session_key: 'session-a' })
        })
      }
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve([
          { session_key: 'session-a', title: 'Query approvals', title_status: 'final' },
          { session_key: 'session-b', title: 'Create virtual machine', title_status: 'final' }
        ])
      })
    }
    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve({})
    })
  })
})

const sessionStorageMock = (() => {
  let store = {}
  return {
    getItem: jest.fn((key) => store[key] || null),
    setItem: jest.fn((key, value) => { store[key] = value }),
    removeItem: jest.fn((key) => { delete store[key] }),
    clear: jest.fn(() => { store = {} })
  }
})()

Object.defineProperty(global, 'sessionStorage', { value: sessionStorageMock })

describe('chat page', () => {
  test('mount renders searchable session titles without date grouping', async () => {
    const chatPage = await import('../../app/frontend/scripts/pages/chat.js')
    const container = document.getElementById('page-root')

    await chatPage.mount(container)

    const sidebar = document.getElementById('sidebar-dynamic-content')
    expect(sidebar.textContent).toContain('Query approvals')
    expect(sidebar.textContent).toContain('Create virtual machine')
    expect(sidebar.textContent).not.toContain('Today')

    const searchInput = sidebar.querySelector('#session-search-input')
    searchInput.value = 'approvals'
    searchInput.dispatchEvent(new Event('input'))

    expect(sidebar.textContent).toContain('Query approvals')
    expect(sidebar.textContent).not.toContain('Create virtual machine')
  })

  test('user turn hides empty state immediately before assistant response returns', async () => {
    jest.resetModules()

    let capturedCallbacks = null
    jest.unstable_mockModule('../../app/frontend/scripts/chat-ui.js', () => ({
      initChat: jest.fn(async (_element, callbacks = {}) => {
        capturedCallbacks = callbacks
      }),
      activateSession: jest.fn(async () => false),
      refreshActiveSessionHistory: jest.fn(async () => false),
      abortCurrentStream: jest.fn(),
      getCurrentAgentInfo: jest.fn(() => ({ name: 'AtlasClaw Enterprise AI Assistant' }))
    }))

    const chatPage = await import('../../app/frontend/scripts/pages/chat.js')
    const container = document.getElementById('page-root')

    await chatPage.mount(container)

    capturedCallbacks.onConversationStateChange({
      hasMessages: false,
      agentInfo: {
        name: 'AtlasClaw Enterprise AI Assistant',
        welcome_message: 'Welcome'
      }
    })

    const emptyState = container.querySelector('#chat-empty-state')
    expect(emptyState.classList.contains('hidden')).toBe(false)

    capturedCallbacks.onUserTurnStarted({
      sessionKey: 'session-a',
      messageText: '你好'
    })

    expect(emptyState.classList.contains('hidden')).toBe(true)
    expect(container.classList.contains('chat-empty-mode')).toBe(false)
  })
})
