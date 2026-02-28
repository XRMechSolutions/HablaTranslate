// Habla Service Worker — app shell caching only
// Does NOT cache API responses, WebSocket data, or audio

const CACHE_NAME = 'habla-v33';
const SHELL_URLS = [
  '/',
  '/static/styles.css?v=33',
  '/static/js/core.js?v=33',
  '/static/js/ui.js?v=33',
  '/static/js/audio.js?v=33',
  '/static/js/settings.js?v=33',
  '/static/js/websocket.js?v=33',
  '/static/js/app.js?v=33',
  '/static/manifest.json',
  '/static/icon-192.png',
  '/static/icon-512.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_URLS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Never cache API calls, WebSocket, health, or vocab API
  if (
    url.pathname.startsWith('/api/') ||
    url.pathname.startsWith('/ws/') ||
    url.pathname === '/health'
  ) {
    return;
  }

  // Network-first for app shell: try network, fall back to cache
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        // Cache successful GET responses
        if (response.ok && event.request.method === 'GET') {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        }
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});
