/**
 * API_Client for the Room Visualizer service.
 *
 * A dependency-free ES module: no build step, no bundler, no framework. It owns
 * every `fetch` call the widget makes, so `visualizer.js` never touches HTTP
 * directly and the base URL lives in exactly one place (Requirement 10.2).
 *
 * Every method parses the service's shared error envelope on a non-2xx status
 * and throws an `ApiError` carrying the machine-readable `code`, the
 * shopper-facing `message`, and the HTTP `status`, which is the single path by
 * which the component gets its displayable error text (Requirement 10.6).
 *
 * Every method accepts an `AbortSignal` so a superseded render can be
 * cancelled when the shopper clicks a third tile before the second finishes.
 */

/** Error envelope shape: `{ "error": { "code": "...", "message": "..." } }`. */

/**
 * Thrown for every non-2xx response and for transport-level failures.
 *
 * `code` is the service's machine-readable error code (`scene_expired`,
 * `payload_too_large`, `no_usable_plane`, ...) for callers that branch on the
 * failure. `message` is written for shoppers and is displayed verbatim.
 * `status` is the HTTP status, or 0 when the request never reached the server.
 */
export class ApiError extends Error {
  /**
   * @param {string} code Machine-readable error code.
   * @param {string} message Human-readable message, displayed verbatim.
   * @param {number} status HTTP status code, or 0 for transport failures.
   */
  constructor(code, message, status) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
  }
}

/** Fallback text used only when a failure carries no message of its own. */
const FALLBACK_MESSAGES = {
  0: 'Could not reach the visualizer service. Check your connection and try again.',
  404: 'That room photo is no longer available. Upload the photo again to continue.',
  413: 'That image is too large. Try a photo under 12 MB.',
  415: 'That file type is not supported. Use a JPEG, PNG, or WebP photo.',
  422: 'That photo could not be analysed. Try a photo showing more of the floor or wall.',
  500: 'Something went wrong while rendering. Try again in a moment.',
};

/**
 * Strip trailing slashes so `${baseUrl}/api/...` never doubles a separator.
 * An empty or missing value yields `''`, which makes every request
 * same-origin and relative — the right default for the bundled demo page.
 *
 * @param {unknown} value
 * @returns {string}
 */
function normaliseBaseUrl(value) {
  if (value === null || value === undefined) return '';
  return String(value).trim().replace(/\/+$/, '');
}

/**
 * Read a message out of whatever the server sent back.
 *
 * Prefers the shared envelope. Falls back to FastAPI's own `detail` shape,
 * which request-validation failures can produce before the application
 * exception handler sees them, and finally to a generic per-status message.
 *
 * @param {unknown} body Parsed response body, or `null` when unparseable.
 * @param {number} status
 * @returns {{ code: string, message: string }}
 */
function extractError(body, status) {
  let code = `http_${status}`;
  let message = '';

  if (body && typeof body === 'object') {
    const envelope = /** @type {Record<string, any>} */ (body).error;
    if (envelope && typeof envelope === 'object') {
      if (typeof envelope.code === 'string' && envelope.code) code = envelope.code;
      if (typeof envelope.message === 'string' && envelope.message) message = envelope.message;
    }

    if (!message) {
      const detail = /** @type {Record<string, any>} */ (body).detail;
      if (typeof detail === 'string' && detail) {
        message = detail;
      } else if (Array.isArray(detail) && detail.length > 0) {
        const first = detail[0];
        if (first && typeof first.msg === 'string' && first.msg) message = first.msg;
      }
    }
  }

  if (!message) {
    message = FALLBACK_MESSAGES[status] || `The request failed with status ${status}.`;
  }

  return { code, message };
}

/**
 * Parse a response body as JSON, returning `null` rather than throwing when the
 * body is empty or is not JSON at all (an HTML error page from a proxy, say).
 *
 * @param {Response} response
 * @returns {Promise<unknown>}
 */
async function readJson(response) {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

export class ApiClient {
  /**
   * @param {string} [baseUrl] Visualizer_API base URL. Empty means same origin.
   */
  constructor(baseUrl) {
    this.baseUrl = normaliseBaseUrl(baseUrl);
  }

  /**
   * Build an absolute (or same-origin relative) URL for an API path.
   *
   * @param {string} path Path beginning with `/`.
   * @returns {string}
   */
  url(path) {
    return `${this.baseUrl}${path}`;
  }

  /**
   * Issue one request and resolve with its parsed JSON body.
   *
   * Throws `ApiError` on any non-2xx status and on transport failure. An abort
   * propagates as the original `AbortError` so callers can tell a cancellation
   * apart from a real failure and stay silent about it.
   *
   * @param {string} path
   * @param {RequestInit} [init]
   * @returns {Promise<any>}
   */
  async request(path, init = {}) {
    const headers = new Headers(init.headers || {});
    headers.set('Accept', 'application/json');

    let response;
    try {
      response = await fetch(this.url(path), { ...init, headers });
    } catch (cause) {
      // A caller-initiated abort is not a service failure; let it through.
      if (cause && cause.name === 'AbortError') throw cause;
      const error = new ApiError('network_error', FALLBACK_MESSAGES[0], 0);
      error.cause = cause;
      throw error;
    }

    const body = await readJson(response);

    if (!response.ok) {
      const { code, message } = extractError(body, response.status);
      throw new ApiError(code, message, response.status);
    }

    return body;
  }

  /**
   * Upload a room photograph for analysis.
   *
   * @param {File|Blob} file The photograph, sent as the `file` multipart part.
   * @param {{ signal?: AbortSignal }} [options]
   * @returns {Promise<any>} The `SegmentResponse` body: `scene_id`, `width`,
   *   `height`, `segmentation_backend`, `geometry_mode`, `horizon`,
   *   `vanishing_points`, `planes`, `analysis_ms`.
   */
  async segment(file, { signal } = {}) {
    const form = new FormData();
    // The filename matters: the service validates the extension allow-list.
    form.append('file', file, file && file.name ? file.name : 'room.png');
    // No Content-Type header — the browser must set the multipart boundary.
    return this.request('/api/segment', { method: 'POST', body: form, signal });
  }

  /**
   * Composite tile selections onto an already-analysed scene.
   *
   * @param {{ scene_id: string, planes: Record<string, object>, format?: string }} payload
   * @param {{ signal?: AbortSignal }} [options]
   * @returns {Promise<any>} The `RenderResponse` body: `scene_id`, `mime`,
   *   base64 `image`, `width`, `height`, `render_ms`, `warnings`.
   */
  async render(payload, { signal } = {}) {
    return this.request('/api/render', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal,
    });
  }

  /**
   * List the available tile products.
   *
   * @param {{ signal?: AbortSignal }} [options]
   * @returns {Promise<Array<object>>} The catalog entries. Always an array, so
   *   callers can iterate without a shape check.
   */
  async tiles({ signal } = {}) {
    const body = await this.request('/api/tiles', { method: 'GET', signal });
    return body && Array.isArray(body.tiles) ? body.tiles : [];
  }

  /**
   * Read service liveness and runtime facts.
   *
   * @param {{ signal?: AbortSignal }} [options]
   * @returns {Promise<any>} The `HealthResponse` body: `status`,
   *   `segmentation_backend`, `onnx_provider`, and the scene cache counters.
   */
  async health({ signal } = {}) {
    return this.request('/api/health', { method: 'GET', signal });
  }
}

export default ApiClient;
