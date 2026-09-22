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

  // 3. Dispatch telemetry event to FastAPI ingestion pipeline
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

  // 4. Initial Page View / Load Event
  function reportPageLoad() {
    var latency = 45.0;
    try {
      if (window.performance && typeof window.performance.getEntriesByType === 'function') {
        var navEntries = window.performance.getEntriesByType('navigation');
        if (navEntries && navEntries.length > 0 && navEntries[0].responseEnd) {
          latency = Math.max(1.0, Math.round(navEntries[0].responseEnd - navEntries[0].requestStart));
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
      },
    });
  }

  if (document.readyState === 'complete') {
    reportPageLoad();
  } else {
    window.addEventListener('load', reportPageLoad);
  }

  // 5. Basic Error Capture
  window.addEventListener('error', function (event) {
    try {
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
        },
      });
    } catch (_err) {}
  });

})();
