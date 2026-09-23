(function () {
  'use strict';

  // 1. Locate the tracker script tag
  var currentScript = document.currentScript || (function () {
    var scripts = document.getElementsByTagName('script');
    for (var i = 0; i < scripts.length; i++) {
      var s = scripts[i];
      if (s.src && (s.src.indexOf('tracker.js') !== -1 || s.getAttribute('data-business-id'))) {
        return s;
      }
    }
    return null;
  })();

  if (!currentScript) {
    return;
  }

  var businessId = currentScript.getAttribute('data-business-id');
  var apiKey = currentScript.getAttribute('data-api-key');

  if (!businessId || !apiKey) {
    if (window.console && console.warn) {
      console.warn('[AI-CTO] tracker.js: Missing data-business-id or data-api-key attribute on script tag.');
    }
    return;
  }

  // 2. Dynamically determine API base URL from script tag source
  var apiBase = '';
  try {
    var scriptUrl = new URL(currentScript.src, window.location.href);
    apiBase = scriptUrl.origin;
  } catch (_e) {
    var a = document.createElement('a');
    a.href = currentScript.src;
    apiBase = a.protocol + '//' + a.host;
  }

  var ingestionEndpoint = apiBase + '/api/ingestion/events';

  // 3. Dispatch telemetry / log event to FastAPI ingestion pipeline
  function dispatchEvent(payload) {
    try {
      var fullPayload = {
        business_id: businessId,
        event_type: payload.event_type || 'pageview',
        endpoint: payload.endpoint || window.location.pathname || '/',
        response_time_ms: typeof payload.response_time_ms === 'number' ? payload.response_time_ms : 45.0,
        status_code: payload.status_code || 200,
        payload_metadata: payload.payload_metadata || {},
        timestamp: new Date().toISOString(),
      };

      if (typeof window.fetch === 'function') {
        window.fetch(ingestionEndpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-API-Key': apiKey,
          },
          body: JSON.stringify(fullPayload),
          mode: 'cors',
          credentials: 'omit',
        }).catch(function () {
          // Silent catch to prevent impacting host site
        });
      }
    } catch (_err) {
      // Telemetry capture failure must never disrupt host website
    }
  }

  // 4. Initial Page View & Core Web Vitals
  function reportPageLoad() {
    var latency = 45.0;
    var ttfb = 0;
    var domReady = 0;

    try {
      if (window.performance && typeof window.performance.getEntriesByType === 'function') {
        var navEntries = window.performance.getEntriesByType('navigation');
        if (navEntries && navEntries.length > 0) {
          var n = navEntries[0];
          if (n.responseEnd && n.requestStart) {
            latency = Math.max(1.0, Math.round(n.responseEnd - n.requestStart));
          }
          if (n.responseStart && n.requestStart) {
            ttfb = Math.max(0, Math.round(n.responseStart - n.requestStart));
          }
          if (n.domContentLoadedEventEnd && n.startTime) {
            domReady = Math.max(0, Math.round(n.domContentLoadedEventEnd - n.startTime));
          }
        }
      }
    } catch (_perfErr) {}

    dispatchEvent({
      event_type: 'pageview',
      endpoint: window.location.pathname || '/',
      status_code: 200,
      response_time_ms: latency,
      payload_metadata: {
        title: document.title || '',
        url: window.location.href,
        referrer: document.referrer || '',
        screen: (window.screen ? window.screen.width + 'x' + window.screen.height : ''),
        ttfb_ms: ttfb,
        dom_ready_ms: domReady,
      },
    });
  }

  if (document.readyState === 'complete') {
    reportPageLoad();
  } else {
    window.addEventListener('load', reportPageLoad);
  }

  // 5. JavaScript Error Capture with Stack Trace
  window.addEventListener('error', function (event) {
    try {
      var stack = (event.error && event.error.stack) ? String(event.error.stack).slice(0, 1000) : '';
      dispatchEvent({
        event_type: 'js_error',
        endpoint: window.location.pathname || '/',
        status_code: 500,
        response_time_ms: 0.0,
        payload_metadata: {
          message: event.message || 'Uncaught JavaScript Error',
          filename: event.filename || '',
          lineno: event.lineno || 0,
          colno: event.colno || 0,
          stack: stack,
        },
      });
    } catch (_err) {}
  });

  // 6. Unhandled Promise Rejection Capture
  window.addEventListener('unhandledrejection', function (event) {
    try {
      var reason = event.reason;
      var message = 'Unhandled Promise Rejection';
      var stack = '';

      if (typeof reason === 'string') {
        message = reason;
      } else if (reason && typeof reason === 'object') {
        message = reason.message || reason.statusText || 'Rejected Promise';
        stack = reason.stack ? String(reason.stack).slice(0, 1000) : '';
      }

      dispatchEvent({
        event_type: 'unhandled_rejection',
        endpoint: window.location.pathname || '/',
        status_code: 500,
        response_time_ms: 0.0,
        payload_metadata: {
          reason: message,
          stack: stack,
        },
      });
    } catch (_err) {}
  });

  // 7. Non-Intrusive Network Fetch Error & Latency Hook
  if (typeof window.fetch === 'function') {
    var origFetch = window.fetch;
    window.fetch = function () {
      var args = arguments;
      var url = typeof args[0] === 'string' ? args[0] : (args[0] && args[0].url ? args[0].url : '');

      // Avoid monitoring calls to our own ingestion endpoint
      if (url && url.indexOf('/api/ingestion/events') !== -1) {
        return origFetch.apply(this, args);
      }

      var start = (window.performance && performance.now) ? performance.now() : Date.now();
      return origFetch.apply(this, args).then(function (res) {
        if (!res.ok && res.status >= 400) {
          var duration = Math.round(((window.performance && performance.now) ? performance.now() : Date.now()) - start);
          var pathOnly = url;
          try { pathOnly = new URL(url, window.location.href).pathname; } catch (_u) {}
          dispatchEvent({
            event_type: 'api_error',
            endpoint: pathOnly || '/',
            status_code: res.status,
            response_time_ms: duration,
            payload_metadata: {
              url: url,
              status_text: res.statusText || '',
            },
          });
        }
        return res;
      }).catch(function (fetchErr) {
        var duration = Math.round(((window.performance && performance.now) ? performance.now() : Date.now()) - start);
        var pathOnly = url;
        try { pathOnly = new URL(url, window.location.href).pathname; } catch (_u) {}
        dispatchEvent({
          event_type: 'api_error',
          endpoint: pathOnly || '/',
          status_code: 503,
          response_time_ms: duration,
          payload_metadata: {
            url: url,
            error: fetchErr ? fetchErr.message : 'Network error',
          },
        });
        throw fetchErr;
      });
    };
  }

  // 8. Public Client Logging API for Custom Host Events
  window.aiCto = {
    log: function (level, message, metadata) {
      dispatchEvent({
        event_type: 'client_log',
        endpoint: window.location.pathname || '/',
        status_code: level === 'error' ? 500 : 200,
        payload_metadata: Object.assign({ level: level, message: message }, metadata || {}),
      });
    },
    track: function (eventType, metadata) {
      dispatchEvent({
        event_type: eventType,
        endpoint: window.location.pathname || '/',
        status_code: 200,
        payload_metadata: metadata || {},
      });
    },
  };
})();
