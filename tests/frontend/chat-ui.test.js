/**
 * chat-ui.js regression tests
 */

jest.mock('../../app/frontend/scripts/config.js', () => ({
    buildApiUrl: (path) => path.startsWith('/') ? path : `/${path}`
}));

jest.mock('../../app/frontend/scripts/i18n.js', () => ({
    t: jest.fn((key) => key),
    isLocaleLoaded: jest.fn(() => false)
}));

beforeEach(() => {
    jest.resetModules();
    global.fetch = jest.fn();
    sessionStorageMock.clear();
    MockEventSource.instances = [];
});

const sessionStorageMock = (() => {
    let store = {};
    return {
        getItem: jest.fn((key) => store[key] || null),
        setItem: jest.fn((key, value) => { store[key] = value; }),
        removeItem: jest.fn((key) => { delete store[key]; }),
        clear: jest.fn(() => { store = {}; })
    };
})();

Object.defineProperty(global, 'sessionStorage', { value: sessionStorageMock });

class MockEventSource {
    constructor(url, options = {}) {
        this.url = url;
        this.options = options;
        this.readyState = EventSource.CONNECTING;
        this.listeners = {};
        MockEventSource.instances.push(this);
    }

    addEventListener(type, callback) {
        this.listeners[type] = this.listeners[type] || [];
        this.listeners[type].push(callback);
    }

    close() {
        this.readyState = EventSource.CLOSED;
    }

    simulateEvent(type, data) {
        const callbacks = this.listeners[type] || [];
        callbacks.forEach(cb => cb({ data: JSON.stringify(data) }));
    }
}

MockEventSource.CONNECTING = 0;
MockEventSource.OPEN = 1;
MockEventSource.CLOSED = 2;
MockEventSource.instances = [];

global.EventSource = MockEventSource;

function createChatElement() {
    return {
        addMessage: jest.fn()
    };
}

describe('chat-ui.js', () => {
    test('requestInterceptor rewrites request body for /api/agent/run without sending its own fetch', async () => {
        const { initChat } = await import('../../app/frontend/scripts/chat-ui.js');
        const element = createChatElement();
        global.fetch.mockResolvedValueOnce({
            ok: true,
            json: () => Promise.resolve({ session_key: 'session-123' })
        });

        await initChat(element);
        global.fetch.mockClear();

        const request = {
            body: {
                messages: [{ text: 'hello' }]
            }
        };

        const pendingRequest = element.requestInterceptor(request);

        expect(global.fetch).not.toHaveBeenCalled();
        const intercepted = await pendingRequest;

        expect(intercepted.body).toEqual({
            session_key: 'session-123',
            message: 'hello',
            timeout_seconds: 600
        });
    });

    test('responseInterceptor starts SSE after backend returns run_id', async () => {
        const { initChat } = await import('../../app/frontend/scripts/chat-ui.js');
        const element = createChatElement();
        global.fetch.mockResolvedValueOnce({
            ok: true,
            json: () => Promise.resolve({ session_key: 'session-123' })
        });

        await initChat(element);
        global.fetch.mockClear();

        const response = await element.responseInterceptor({
            run_id: 'run-123',
            status: 'running',
            session_key: 'session-123'
        });

        expect(MockEventSource.instances).toHaveLength(1);
        // URL is relative path since buildApiUrl returns relative paths
        expect(MockEventSource.instances[0].url).toBe('/api/agent/runs/run-123/stream');
        expect(response).toEqual({ text: '' });
    });

    test('assistant stream overwrites placeholder message with streamed content', async () => {
        const { initChat } = await import('../../app/frontend/scripts/chat-ui.js');
        const element = createChatElement();
        global.fetch.mockResolvedValueOnce({
            ok: true,
            json: () => Promise.resolve({ session_key: 'session-123' })
        });

        await initChat(element);
        element.addMessage({ text: '...', role: 'ai' });

        await element.responseInterceptor({
            run_id: 'run-123',
            status: 'running',
            session_key: 'session-123'
        });

        const stream = MockEventSource.instances[0];
        stream.simulateEvent('assistant', { text: 'Hello', is_delta: true });
        stream.simulateEvent('lifecycle', { phase: 'end' });

        expect(element.addMessage).toHaveBeenCalledWith({
            text: 'Hello',
            role: 'ai',
            overwrite: true
        });
    });
});
