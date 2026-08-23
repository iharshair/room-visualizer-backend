# AI Room & Tile Visualizer

Upload a photograph of a room, pick a surface, pick a tile, and see the tile applied to that
surface with correct perspective, correct real-world scale, preserved foreground occlusions, and
the photograph's own light and shadow carried across the new material.

The service is a Python FastAPI backend plus a zero-build vanilla JavaScript widget. Processing is
split in two:

- **`POST /api/segment`** — expensive, runs **once per photograph**. Segmentation, camera
  calibration, and lighting decomposition. Seconds.
- **`POST /api/render`** — cheap, runs **on every tile change**. Warps a seamless texture through
  the cached homography and blends it with the cached lighting maps. Tens of milliseconds.

The `Scene_Cache` is the boundary between the two, which is what makes tile swapping feel
immediate on a CPU-only host.

---

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Asset setup](#asset-setup)
- [Running the service](#running-the-service)
- [Tests](#tests)
- [HTTP API](#http-api)
  - [Error envelope](#error-envelope)
  - [`POST /api/segment`](#post-apisegment)
  - [`POST /api/render`](#post-apirender)
  - [`GET /api/tiles`](#get-apitiles)
  - [`GET /api/health`](#get-apihealth)
  - [Status and error code table](#status-and-error-code-table)
- [Embedding the frontend component](#embedding-the-frontend-component)
- [Environment variables](#environment-variables)
- [Adding your own tiles](#adding-your-own-tiles)
- [Security](#security)
- [Deployment](#deployment)
- [How it works](#how-it-works)

---

## Requirements

- Python 3.11 or newer
- No GPU required. `onnxruntime` uses `CPUExecutionProvider` unless CUDA is available.
- No network access required to install assets, run the service, or run the tests. Model weights
  are downloaded on first start when the network is available; when it is not, the service starts
  anyway and serves with the classical segmentation backend.

## Installation

All commands are run from the `room-visualizer/` directory.

```bash
cd room-visualizer

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r backend/requirements.txt
```

`backend/requirements.txt` pins every runtime dependency to an exact version and declares the
development and test dependencies (`pytest`, `pytest-cov`, `hypothesis`, `psutil`) in a labelled
section at the bottom of the same file, so the single command above installs both.

## Asset setup

The starter tile catalog and the sample room image are **generated**, not shipped. Run the setup
tool once after installing:

```bash
python scripts/setup_assets.py
```

It writes, under `assets/`:

- `assets/tiles/*.png` — eight seamless procedural tiles: marble, wood plank, concrete, and
  terrazzo, each in 600 × 600 mm and 600 × 1200 mm.
- `assets/tiles/manifest.json` — the catalog manifest describing each generated tile with its
  millimetre dimensions, finish label, and gloss value.
- `assets/samples/synthetic_room.png`, plus `synthetic_room.occluders.png` and
  `synthetic_room.truth.json` — a synthetic perspective room with a floor, three walls, and two
  occluders, together with its occluder mask and its analytic ground-truth vanishing points,
  horizon, and per-plane homographies. Useful as sample input and as a test fixture.

Everything is computed procedurally: no network access, no third-party imagery.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--out DIR` | `assets` | Assets root to write into. `tiles/` and `samples/` are created inside it. |
| `--seed N` | `7` | Base seed for the **tile rasters**. They are a deterministic function of it, so the same seed reproduces byte-identical tiles. The sample room ignores it — see below. |
| `--force` | off | Overwrite existing assets. Without it, existing files are left untouched. |
| `--quiet` | off | Suppress the progress report. Errors are still reported. |

Re-running without `--force` is safe: existing tile images and an existing manifest are preserved,
so your own product drops survive a re-run.

The sample room is pinned to seed `0` and `--seed` does not move it. That seed is
`make_synthetic_room`'s own default, which is the pose the test suite's `synthetic_room` fixture is
anchored to, so the shipped sample is pixel-for-pixel the scene the geometry, lighting, and
compositing assertions are written against. It is also a pose the classical segmentation backend
handles: the two occluders sit well inside the frame, where foreground recall is around 0.92, so the
demo shows the furniture staying in front of the tiles. Other seeds can push the occluders off the
frame edge and past that backend's envelope, which reads as the composite tiling straight over the
furniture.

## Running the service

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --workers 1
```

Then open <http://127.0.0.1:8000/> — the same process serves the demo frontend from
`frontend/index.html`, the tile images at `/assets/tiles/...`, and the API under `/api/`.

Add `--reload` for development. **Keep `--workers 1`**: the scene cache is process-local, and a
second worker would answer renders for scenes it has never seen. See
[Deployment](#deployment).

## Tests

```bash
pytest
```

The suite runs with no model weights present and no network access — an autouse fixture actively
blocks socket connections, so an accidental network dependency fails rather than passing quietly.
Timing and memory budgets are excluded from the default run and selected explicitly:

```bash
pytest -m perf         # render latency and analysis timing budgets
pytest -m resource     # resident-set-size budgets
```

---

## HTTP API

Four endpoints, all under `/api`. `GET /` and everything not under `/api` or `/assets` is served
from `frontend/`. FastAPI's interactive documentation is at `/docs`, and the OpenAPI schema at
`/openapi.json`.

### Error envelope

**Every** failure — from any endpoint, at any status — uses one body shape:

```json
{
  "error": {
    "code": "no_usable_plane",
    "message": "No wall or floor large enough to tile was found in this photo. Try a photo showing more of the floor."
  }
}
```

`code` is machine-readable and stable; branch on it. `message` is written for shoppers and is
displayed verbatim by the frontend, so it names the corrective action wherever one exists.

### `POST /api/segment`

Analyse one room photograph. Runs segmentation, calibration, and lighting decomposition exactly
once, caches the result, and returns the scene description. No corner points, plane annotations,
or perspective hints are accepted or needed.

**Request** — `multipart/form-data` with a single part named `file`.

```bash
curl -sS -X POST http://127.0.0.1:8000/api/segment \
  -F 'file=@assets/samples/synthetic_room.png;type=image/png'
```

Validation runs in this order and each step short-circuits, so no pipeline stage is reachable by
input that has not passed all four:

1. streamed byte count against `RV_MAX_UPLOAD_BYTES` → **413** `payload_too_large`
2. declared MIME type against `RV_ALLOWED_MIME_TYPES` → **415** `unsupported_media_type`
3. filename extension against `RV_ALLOWED_EXTENSIONS` → **415** `unsupported_media_type`
4. the bytes actually decode as a raster image → **415** `unsupported_media_type`

The size cap is enforced against the bytes actually read, not against `Content-Length`, so an
understated header does not get a payload past it. An accepted photograph whose longest edge
exceeds `RV_MAX_LONGEST_EDGE` (default 2048 px) is downscaled to that limit with the aspect ratio
preserved; every coordinate in the response is in the coordinate space of that processed image.

**Response `200`**

```json
{
  "scene_id": "8f14e45fceea167a5a36dedd4bea2543",
  "width": 1600,
  "height": 1200,
  "segmentation_backend": "classical",
  "geometry_mode": "vanishing_points",
  "horizon": { "a": 0.0, "b": 1.0, "c": -512.4, "y_at_center": 512.4 },
  "vanishing_points": {
    "VPx": [-2140.5, 511.8],
    "VPy": [802.1, -9840.2],
    "VPz": [4102.7, 513.0]
  },
  "planes": [
    {
      "name": "floor",
      "area_fraction": 0.312,
      "contour": [[12, 780], [1588, 802], [1140, 1199], [430, 1199]],
      "bounding_points": [[12, 780], [1588, 802], [1140, 1199], [430, 1199]],
      "centroid": [800.0, 980.4],
      "reprojection_rmse_px": 0.41
    }
  ],
  "analysis_ms": 2140
}
```

| Field | Meaning |
| --- | --- |
| `scene_id` | Cache key. Pass it to `/api/render`. |
| `width`, `height` | Dimensions of the **processed** image, in pixels. |
| `segmentation_backend` | `mobilesam-onnx` when the neural backend is active, `classical` when it fell back. |
| `geometry_mode` | `vanishing_points` when three orthogonal vanishing points were recovered, `planar_fallback` when a four-point planar homography was used instead. |
| `horizon` | Homogeneous line `(a, b, c)` plus `y_at_center`, its row at the image centre column. |
| `vanishing_points` | `VPx`, `VPy`, `VPz` in image pixels. A label that could not be recovered is `null`. |
| `planes[]` | One entry per **detected** plane. Undetected planes are omitted, never returned empty. |
| `planes[].name` | `floor`, `wall_left`, `wall_right`, or `wall_back`. |
| `planes[].area_fraction` | Mask pixels divided by total pixels, in `(0, 1]`. |
| `planes[].contour` | Simplified polygon, `[[x, y], ...]`, at least three points, for the filled highlight. |
| `planes[].bounding_points` | Exactly four points, for hit-testing and the selection outline. |
| `planes[].reprojection_rmse_px` | Homography round-trip error in pixels. |
| `analysis_ms` | Server-side analysis time. |

Plane masks are mutually disjoint, and no foreground (furniture, fixtures, people) pixel belongs to
any plane mask.

### `POST /api/render`

Apply tile selections to an already-analysed scene. Reuses the cached masks, homographies, and
lighting maps; performs no segmentation and no vanishing point estimation.

**Request** — `application/json`

```json
{
  "scene_id": "8f14e45fceea167a5a36dedd4bea2543",
  "planes": {
    "floor":     { "tile_id": "marble-polished-600x600", "rotation_deg": 0, "grout_mm": 3 },
    "wall_back": { "tile_id": "concrete-matte-600x600", "rotation_deg": 45 }
  },
  "format": "png"
}
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `scene_id` | string | required | The id returned by `/api/segment`. |
| `planes` | object | `{}` | Plane name → spec. A plane not listed keeps its original photographic appearance. |
| `planes[].tile_id` | string | required | A tile id from `/api/tiles`. |
| `planes[].rotation_deg` | number | `0` | Tile rotation within the metric plane. |
| `planes[].grout_mm` | number \| null | inherit | Grout width in millimetres. `null` inherits the tile value, then `RV_DEFAULT_GROUT_MM`. |
| `planes[].grout_rgb` | `[r, g, b]` \| null | inherit | Grout colour, each channel `0-255`. `null` inherits `RV_DEFAULT_GROUT_RGB`. |
| `planes[].offset_mm` | `[u, v]` | `[0, 0]` | Tiling origin offset in millimetres. |
| `format` | `"png"` \| `"jpeg"` \| null | `null` | `null` uses the configured `RV_RENDER_FORMAT`. |

**Response `200`**

```json
{
  "scene_id": "8f14e45fceea167a5a36dedd4bea2543",
  "mime": "image/png",
  "image": "iVBORw0KGgoAAAANSUhEUg...",
  "width": 1600,
  "height": 1200,
  "render_ms": 61,
  "warnings": []
}
```

`image` is the encoded image, base64-encoded. Adding `?binary=1` returns the raw encoded bytes
with `Content-Type: image/png` (or `image/jpeg`) and the timing in an `X-Render-Ms` header, which
avoids base64's 33 percent overhead for callers that do not need the JSON envelope.

Grout is drawn in metric space alongside the tiles, so grout lines foreshorten with the surface
instead of staying a constant pixel width.

### `GET /api/tiles`

Every valid tile in the catalog.

```json
{
  "tiles": [
    {
      "id": "marble-polished-600x600",
      "name": "Polished Marble 600x600",
      "width_mm": 600.0,
      "height_mm": 600.0,
      "finish": "polished",
      "gloss": 0.85,
      "thumbnail_url": "/assets/tiles/marble_600x600.png"
    }
  ]
}
```

The manifest is re-read whenever its modification stamp changes, so a new tile appears here with
no restart and no Python edit. Entries that fail validation are absent from the list and were
logged at `WARNING` naming the entry and the specific failure.

### `GET /api/health`

Liveness plus the runtime facts an operator needs.

```json
{
  "status": "ok",
  "segmentation_backend": "classical",
  "onnx_provider": "n/a",
  "scene_cache_entries": 3,
  "scene_cache_max_entries": 32,
  "scene_cache_ttl_seconds": 1800
}
```

`onnx_provider` is `CPUExecutionProvider`, `CUDAExecutionProvider`, or `n/a` when no onnxruntime
session is open — which is the case whenever `segmentation_backend` is `classical`.

### Status and error code table

| Status | `code` | Endpoint | Condition |
| --- | --- | --- | --- |
| 413 | `payload_too_large` | `/api/segment` | Upload exceeds `RV_MAX_UPLOAD_BYTES`. |
| 415 | `unsupported_media_type` | `/api/segment` | Disallowed MIME type, disallowed filename extension, or bytes that do not decode as a raster image. |
| 422 | `no_usable_plane` | `/api/segment` | No plane reaches `RV_MIN_PLANE_AREA_FRACTION`, or no detected plane could be given geometry. |
| 422 | `analysis_failed` | `/api/segment` | A pipeline stage raised. The failing stage is named in `message`. |
| 404 | `scene_expired` | `/api/render` | `scene_id` is unknown, was evicted by the LRU bound, or aged past the TTL. The three are indistinguishable to a client and the recovery is the same: re-upload the photo. |
| 422 | `unknown_tile` | `/api/render` | `tile_id` is not a valid catalog entry. |
| 422 | `unknown_plane` | `/api/render` | The named plane was not detected in this scene. |
| 422 | `invalid_request` | any | Request body or query parameters failed validation. |
| 500 | `render_failed` | `/api/render` | Compositing raised. |
| 404 | `not_found` | any | No such path. |
| 405 | `method_not_allowed` | any | Method not allowed for this path. |
| 500 | `internal_error` | any | Unhandled failure. |

Failures that the service can degrade around never reach the client at all. Missing weights, a
checksum mismatch, or a failed onnxruntime session initialisation each log a `WARNING` and bind the
classical backend. An OpenCV build without the line segment detector falls back to Hough. Fewer
than three orthogonal vanishing points falls back to a four-point planar homography and reports
`geometry_mode: "planar_fallback"`. Only genuinely unusable input produces a 4xx.

---

## Embedding the frontend component

`frontend/js/visualizer.js` is a plain ES module — no build step, no bundler, no framework, and no
dependency beyond its sibling `api.js`. A host page needs one stylesheet link, one script, and one
constructor call. This is `frontend/index.html`, which doubles as the working example:

```html
<link rel="stylesheet" href="css/visualizer.css">

<div id="container"></div>

<script type="module">
  import { RoomVisualizer } from './js/visualizer.js';

  const visualizer = new RoomVisualizer('#container', {
    // Empty means same origin. Point it at 'http://localhost:8000' when
    // embedding on a page the API does not serve.
    apiBaseUrl: '',

    defaultPlane: 'floor',
    renderFormat: 'png',
    initialTileId: null,

    onRenderComplete(result) {
      console.log(`Rendered in ${result.render_ms} ms.`);
    },
  });
</script>
```

Copy `frontend/css/`, `frontend/js/`, and set `apiBaseUrl` to wherever the API is served. The class
is also assigned to `window.RoomVisualizer`, so a page that loads the file with
`<script type="module" src="js/visualizer.js"></script>` can construct it from ordinary inline
script afterwards.

Serving the widget from a different origin than the API requires that origin in
`RV_CORS_ALLOW_ORIGINS`.

### Configuration keys

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `apiBaseUrl` | string | `''` | Visualizer_API base URL. Empty means same origin. Trailing slashes are trimmed. |
| `initialTileId` | string \| null | `null` | Tile applied to the default plane as soon as a photo is analysed. |
| `defaultPlane` | string | `'floor'` | Plane selected on load: `floor`, `wall_left`, `wall_right`, or `wall_back`. |
| `renderFormat` | `'png'` \| `'jpeg'` | `'png'` | `format` sent with each render request. |
| `onRenderComplete` | function | — | Called with the `RenderResponse` after each successful render. |
| `onSceneReady` | function | — | Called with the `SegmentResponse` after a photo is analysed. |
| `onError` | function | — | Called with the `ApiError` on any failure. |
| `labels` | object | — | Overrides for any user-facing string, for localisation. |

The first argument is a CSS selector string or an `HTMLElement`.

### Methods

| Method | Behaviour |
| --- | --- |
| `loadPhoto(file)` | Upload and analyse. Resolves with the scene, or `null` when dropped. |
| `selectPlane(name)` | Make a plane the active tile target. Returns `true` when it exists in the scene. |
| `applyTile(tileId, planeName?)` | Set a plane's tile and render. |
| `setRotation(deg, planeName?)` | Update rotation and re-render. |
| `setGrout(mm, planeName?)` | Update grout width and re-render. |
| `render()` | Force a render with the current selections. |
| `reset()` | Clear scene, selections, and canvas. |
| `destroy()` | Remove listeners, abort in-flight requests, empty the container. |
| `getState()` | Snapshot: `{ sceneId, planes, selections, activePlane, busy }`. |

### Events

The same three callbacks are also dispatched as `CustomEvent`s on the container, so a host page can
integrate through listeners instead:

```js
document.querySelector('#container')
  .addEventListener('rv:render-complete', (event) => console.log(event.detail));
```

`rv:scene-ready`, `rv:render-complete`, `rv:error`, `rv:busy-change`.

### Styling and accessibility

Every element the widget injects lives under a single `.rv-root` inside your container, every class
and id is prefixed `rv-`, and every selector in `visualizer.css` is a descendant of `.rv-root`.
The stylesheet ships no reset, no bare element selectors, and no `:root` rules, so host page styles
are never touched. Theme the widget by overriding its custom properties on your container:

```css
#container { --rv-accent: #2f6f4f; --rv-radius: 10px; }
```

Photo selection is a real `<input type="file">` with a `<label>`. Plane selection is a
`role="radiogroup"` with arrow-key traversal. Tile selection is a list of `aria-pressed` buttons
with accessible names combining tile name, size, and finish. The canvas carries `role="img"` with
a label describing the current composition. Progress text goes to a `role="status"` region and
errors to a `role="alert"` region, so screen readers interrupt for failures but not for routine
progress. While a request is in flight, `.rv-root` carries `aria-busy="true"` and controls are
disabled; a repeat of an identical in-flight render is dropped, and a differing one supersedes it.

---

## Environment variables

Every setting is read from an `RV_`-prefixed environment variable, or from a `.env` file in the
process working directory. Sequence-valued settings accept either a comma-separated list
(`RV_CORS_ALLOW_ORIGINS='https://a.example,https://b.example'`) or a JSON array.

### Upload hardening

| Variable | Type | Default | Meaning |
| --- | --- | --- | --- |
| `RV_MAX_UPLOAD_BYTES` | int > 0 | `12582912` (12 MB) | Maximum accepted upload size. Over this, 413. |
| `RV_MAX_LONGEST_EDGE` | int ≥ 64 | `2048` | Accepted photographs are downscaled so the longest edge is at most this, preserving aspect ratio. |
| `RV_ALLOWED_MIME_TYPES` | list | `image/jpeg,image/png,image/webp` | Declared MIME types accepted. |
| `RV_ALLOWED_EXTENSIONS` | list | `.jpg,.jpeg,.png,.webp` | Filename extensions accepted. A leading dot is added if omitted. |
| `RV_CORS_ALLOW_ORIGINS` | list | `*` | CORS allow-list. `*` is a local-development default; narrow it for any deployment. |

### Scene cache

| Variable | Type | Default | Meaning |
| --- | --- | --- | --- |
| `RV_SCENE_CACHE_MAX_ENTRIES` | int ≥ 1 | `32` | Cached scenes retained. Least-recently-used entries are evicted past this. |
| `RV_SCENE_CACHE_TTL_SECONDS` | int > 0 | `1800` | Maximum scene age. Older scenes are evicted and render returns `scene_expired`. |

### Geometry

| Variable | Type | Default | Meaning |
| --- | --- | --- | --- |
| `RV_MIN_PLANE_AREA_FRACTION` | float in (0, 1) | `0.02` | Smallest plane, as a fraction of image area, that is worth tiling. Below this for every plane, 422 `no_usable_plane`. |
| `RV_VP_MIN_CLUSTER_SIZE` | int ≥ 2 | `8` | Smallest directional line cluster accepted for vanishing point estimation. |
| `RV_VP_RANSAC_ITERATIONS` | int ≥ 1 | `400` | RANSAC iterations per vanishing point. |
| `RV_VP_INLIER_THRESHOLD_PX` | float > 0 | `2.0` | Inlier distance threshold for vanishing point RANSAC, in pixels. |
| `RV_ORTHOGONALITY_TOLERANCE` | float > 0 | `0.25` | Maximum mutual-orthogonality residual for a vanishing point triple. Exceeding it routes to the planar fallback. |
| `RV_ASSUMED_CAMERA_HEIGHT_MM` | float > 0 | `1500.0` | Assumed camera height above the floor. Absolute scale is unobservable from one uncalibrated photograph, so this convention fixes the millimetre-per-unit factor and makes tile scale internally consistent across each surface. |

### Lighting and compositing

| Variable | Type | Default | Meaning |
| --- | --- | --- | --- |
| `RV_SHADING_SIGMA_PX` | int ≥ 1 | `31` | Spatial sigma for the low-frequency shading map. Rounded up to the next odd value. |
| `RV_USE_BILATERAL_SHADING` | bool | `true` | Run a bilateral pass before the Gaussian, which keeps cast-shadow edges sharp. Disable on hosts where it is too slow. |
| `RV_FEATHER_WIDTH_PX` | int ≥ 0 | `2` | Width of the alpha ramp at plane mask edges. |
| `RV_DEFAULT_GROUT_MM` | float ≥ 0 | `3.0` | Grout width used when neither the request nor the tile specifies one. |
| `RV_DEFAULT_GROUT_RGB` | `r,g,b` | `168,168,164` | Grout colour, each channel `0-255`. |

### Model loader

| Variable | Type | Default | Meaning |
| --- | --- | --- | --- |
| `RV_ENABLE_NEURAL_BACKEND` | bool | `true` | When false, the classical backend is used without touching the network or the weights directory. |
| `RV_WEIGHTS_DIR` | path | `~/.cache/room-visualizer/weights` | Local cache for the ONNX weights. |
| `RV_MOBILESAM_ENCODER_URL` | https URL | HuggingFace mirror | MobileSAM encoder source. Must be `https://`. |
| `RV_MOBILESAM_ENCODER_SHA256` | 64 hex chars | pinned | Expected encoder digest. A mismatch discards the download. |
| `RV_MOBILESAM_DECODER_URL` | https URL | HuggingFace mirror | MobileSAM decoder source. Must be `https://`. |
| `RV_MOBILESAM_DECODER_SHA256` | 64 hex chars | pinned | Expected decoder digest. |
| `RV_MODEL_DOWNLOAD_TIMEOUT_S` | float > 0 | `30.0` | Per-request timeout for weight download. |

Point the URL pair at an internal mirror and set the matching digests to serve the neural backend
without egress to a public host.

### Assets and output

| Variable | Type | Default | Meaning |
| --- | --- | --- | --- |
| `RV_ASSETS_DIR` | path | `<project>/assets` | Assets root. `tiles/` and `samples/` live inside it, and it is what `/assets` serves. |
| `RV_TILES_MANIFEST_NAME` | string | `manifest.json` | Manifest filename inside `assets/tiles/`. |
| `RV_RENDER_FORMAT` | `png` \| `jpeg` | `png` | Default render encoding when a request does not specify one. |
| `RV_RENDER_JPEG_QUALITY` | int 1-100 | `90` | JPEG quality when the render format is `jpeg`. |

## Adding your own tiles

No Python changes are needed. Drop the image into `assets/tiles/` and add an entry to
`assets/tiles/manifest.json`:

```json
{
  "version": 1,
  "tiles": [
    {
      "id": "my-tile-600x1200",
      "name": "My Plank 600×1200",
      "file": "my_tile_600x1200.png",
      "width_mm": 600,
      "height_mm": 1200,
      "finish": "satin",
      "gloss": 0.35
    }
  ]
}
```

Every field is required: `id`, `name`, `file`, a positive `width_mm` and `height_mm`, a non-empty
`finish`, and a `gloss` between `0.0` and `1.0`. `file` is resolved inside `assets/tiles/` and must
exist and decode. An entry failing any of these is excluded from `/api/tiles` and the reason is
logged at `WARNING`.

Declare the tile's true real-world millimetre dimensions. They are what set its rendered scale and
aspect ratio: the tiling is laid out in millimetres and projected, never stretched in pixel space,
so a 600 × 1200 plank keeps its 1:2 ratio on every surface and in either geometry mode. Product
photography does not need to be pre-tiled — the service synthesises a seamless pattern from a
single image on first use and memoises it.

`gloss` scales the specular highlight the photograph's own light contributes to the tile: `0.0`
for a fully matte finish, up to `0.85` or so for polished stone.

---

## Security

**This service is unauthenticated by design.** It provides:

- **no authentication** — any caller reaching the port can upload and render
- **no authorization** — there are no users, roles, or ownership of scenes
- **no rate limiting** — there is no per-caller quota of any kind
- **no multi-tenant isolation** — a `scene_id` is a bearer capability; any caller holding one can
  render against that scene

The scope of the project is local development and embeddable-module distribution.

> **Any public deployment requires an authenticating reverse proxy in front of the Visualizer_API,
> configured with rate limiting and request-size limits at the proxy.** Without one, `/api/segment`
> is an open, CPU-expensive compute endpoint available to anonymous callers — a straightforward
> denial-of-service and denial-of-wallet target. Do not expose this service directly to the
> internet.

Narrow `RV_CORS_ALLOW_ORIGINS` from its `*` default at the same time. The default is a
local-development convenience, not a deployment setting.

What the service does harden, within that scope:

- MIME type and filename extension allow-lists, plus an actual image decode, so a renamed
  executable or a polyglot file is rejected before any pipeline stage runs.
- A size cap enforced against bytes actually read, aborting mid-upload rather than trusting
  `Content-Length`.
- A longest-edge clamp that bounds every downstream allocation regardless of the dimensions the
  upload declares, which also defeats decompression-bomb style input.
- A bounded cache with LRU and TTL eviction, so repeated uploads cannot grow memory without limit.
- Model weights fetched only over HTTPS from a pinned URL and verified against a pinned SHA-256,
  with partial downloads discarded, so a truncated or substituted artifact is never loaded.
- Manifest `file` values resolved and confirmed to sit inside `assets/tiles/`, so a manifest entry
  cannot read arbitrary filesystem paths.
- No user-supplied value reaches a shell, a filesystem path outside the assets directory, or an
  eval-like construct.

## Deployment

**Run a single uvicorn worker.**

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 1
```

The reasons are structural:

- **The `Scene_Cache` is process-local.** It is an in-memory `OrderedDict` inside one Python
  process. Nothing is shared between workers.
- **`Scene_State` is lost on process restart.** There is no persistence. Every cached scene is gone
  after a restart, a redeploy, or a worker recycle, and the next `/api/render` for those scenes
  returns 404 `scene_expired`. Clients recover by re-uploading the photograph — the frontend
  surfaces this as a message telling the shopper to upload the photo again.
- **More than one worker breaks tile swapping.** `/api/segment` populates the cache of whichever
  worker handled the upload. A subsequent `/api/render` load-balanced to a different worker finds no
  such `scene_id` and returns 404 `scene_expired`. With N workers, roughly `(N-1)/N` of renders fail
  this way.

**Run more than one worker only after introducing a shared cache** — Redis, or another out-of-process
store — behind the `SceneCache` interface in `backend/cache.py`. Until then, scale vertically, or
run multiple single-worker processes behind a load balancer with sticky sessions keyed so that a
client's segment and render requests reach the same process.

Other operational notes:

- Size the cache for your memory budget. A cached 2048 px scene is roughly 31 MB of 8-bit masks
  and lighting maps, so the default 32 entries is roughly 1 GB. `RV_SCENE_CACHE_MAX_ENTRIES` and
  `RV_SCENE_CACHE_TTL_SECONDS` are the two levers.
- Analysis is CPU-bound and runs in Starlette's threadpool, so concurrent uploads compete for
  cores. Analysis of one 2048 px photograph stays under 2 GB resident on a CPU-only host.
- Poll `GET /api/health` for liveness. It also reports which segmentation backend is actually in
  use, so a host that fell back to `classical` is visible without reading startup logs.
- Set `RV_RENDER_FORMAT=jpeg` if PNG encoding dominates render time on a slow host.
- The API process also serves `frontend/` and `/assets`. In production, prefer serving both from
  the reverse proxy or a CDN and letting the API handle only `/api/`.

---

## How it works

**Segmentation** produces disjoint masks for `floor`, `wall_left`, `wall_right`, and `wall_back`,
plus a foreground mask covering furniture, fixtures, and people. MobileSAM through onnxruntime when
weights are available; an OpenCV geometry-and-colour backend otherwise. Both feed one shared
post-processing pass, so the disjointness and foreground-exclusion guarantees hold either way.
Planes not present in the photograph are omitted rather than returned empty.

**Calibration** detects line segments, clusters them by direction, and recovers up to three mutually
orthogonal vanishing points by RANSAC plus total-least-squares refinement. From those it derives the
horizon and, per plane, a homography mapping millimetre plane coordinates to image pixels. Metric
millimetres are the single source of truth for scale, which is what makes tile size consistent from
the front of a floor to the back and foreshortening automatic rather than special-cased.

**Lighting decomposition** converts the photograph to CIELAB, keeps `L*`, and splits it into a
low-frequency shading map (soft gradients and cast shadows) and a 128-centred high-frequency detail
map (local variation and specular highlights). A per-plane median of the shading map is stored as
that surface's neutral point.

**Compositing** inverse-maps each destination pixel through `H⁻¹` into millimetres, applies rotation
and offset there, reduces modulo the tile pitch, and samples the seamless texture in one
`cv2.remap`. No Python loop runs per tile or per pixel. The result is blended with the lighting
maps — multiply below the plane median, soft-light above it, plus a gloss-scaled highlight — then
alpha-composited over the photograph with a feathered plane edge. Foreground pixels are redrawn
from the original photograph last, so occluding objects stay in front of the tiles.

Every stage degrades rather than fails: neural to classical segmentation, line segment detector to
Hough, three-vanishing-point calibration to four-point planar, vanishing-point horizon to
contour-derived horizon. The only hard failure is a photograph with no usable surface.
