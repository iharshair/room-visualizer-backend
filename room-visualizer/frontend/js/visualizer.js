/**
 * RoomVisualizer — the embeddable Room Visualizer widget.
 *
 * A framework-free ES module: no build step, no bundler, no dependency beyond
 * `./api.js`. A host page needs one script tag and one constructor call
 * (Requirement 10.1):
 *
 *     import { RoomVisualizer } from './js/visualizer.js';
 *     new RoomVisualizer('#container', { apiBaseUrl: 'http://localhost:8000' });
 *
 * The class is also assigned to `window.RoomVisualizer`, so a page that loads
 * this file with `<script type="module" src="js/visualizer.js">` can construct
 * it from ordinary inline script afterwards.
 *
 * Everything the component injects lives under a single `.rv-root` element
 * created inside the supplied container, and every injected class and id is
 * prefixed `rv-`, so host page styles are never touched (Requirement 10.8).
 */

import { ApiClient } from './api.js';

/** Structural plane names, in the draw order used for the overlay. */
const PLANE_NAMES = ['wall_back', 'wall_left', 'wall_right', 'floor'];

/** Default user-facing strings. Every key is overridable via `config.labels`. */
const DEFAULT_LABELS = {
  photoSection: 'Room photo',
  planeSection: 'Surface',
  tileSection: 'Tiles',
  controlsSection: 'Adjust',

  uploadLabel: 'Choose room photo',
  uploadHint: 'JPEG, PNG, or WebP, up to 12 MB.',
  noFileName: 'No photo selected yet.',

  stageEmpty: 'Upload a photo of your room to preview tiles on the floor and walls.',
  planesEmpty: 'Surfaces appear here once a photo has been analysed.',
  tilesEmpty: 'No tiles are available from the catalogue.',
  tilesLoading: 'Loading tiles…',

  rotationLabel: 'Rotation',
  rotationValue: '{value}°',
  groutLabel: 'Grout',
  groutValue: '{value} mm',

  analysing: 'Analysing photo…',
  sceneReady: 'Found {count} surface(s). Pick a surface, then pick a tile.',
  rendering: 'Applying tiles…',
  renderDone: 'Updated in {ms} ms.',
  tilesLoaded: '{count} tiles available.',

  canvasEmpty: 'No room photo loaded yet.',
  canvasPhoto: 'Room photo with no tiles applied.',
  canvasComposition: 'Room photo with {selections}.',
  canvasSelection: '{tile} applied to the {plane}',

  tileAriaLabel: '{name}, {width} by {height} millimetres, {finish}',
  tileMeta: '{width} × {height} mm · {finish}',

  planeFloor: 'Floor',
  planeWallLeft: 'Left wall',
  planeWallRight: 'Right wall',
  planeWallBack: 'Back wall',

  noSceneError: 'Upload a room photo before applying a tile.',
  unknownTileError: 'That tile is not in the catalogue.',
  unknownPlaneError: 'That surface was not detected in this photo.',
  genericError: 'Something went wrong. Please try again.',
};

/** Human-readable label key for each structural plane name. */
const PLANE_LABEL_KEYS = {
  floor: 'planeFloor',
  wall_left: 'planeWallLeft',
  wall_right: 'planeWallRight',
  wall_back: 'planeWallBack',
};

/** Rotation slider bounds, in degrees within the metric plane. */
const ROTATION_MIN = 0;
const ROTATION_MAX = 90;
const ROTATION_STEP = 1;

/** Grout slider bounds, in millimetres. */
const GROUT_MIN = 0;
const GROUT_MAX = 12;
const GROUT_STEP = 0.5;

/** Fallback accent colour, used when the stylesheet is not loaded. */
const FALLBACK_ACCENT = '#b4552d';

let instanceCounter = 0;

/**
 * Substitute `{token}` placeholders in a label template.
 *
 * @param {string} template
 * @param {Record<string, string|number>} [vars]
 * @returns {string}
 */
function format(template, vars) {
  if (!vars) return template;
  return String(template).replace(/\{(\w+)\}/g, (match, key) =>
    Object.prototype.hasOwnProperty.call(vars, key) ? String(vars[key]) : match
  );
}

/**
 * Ray-casting point-in-polygon test.
 *
 * @param {number} x
 * @param {number} y
 * @param {Array<Array<number>>} polygon `[[x, y], ...]`, at least three points.
 * @returns {boolean}
 */
function pointInPolygon(x, y, polygon) {
  if (!Array.isArray(polygon) || polygon.length < 3) return false;
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    const straddles = yi > y !== yj > y;
    if (straddles && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

/**
 * Trim a number for display without trailing zeros (`3.0` -> `3`, `2.5` stays).
 *
 * @param {number} value
 * @returns {string}
 */
function trimNumber(value) {
  const rounded = Math.round(Number(value) * 100) / 100;
  return String(rounded);
}

export class RoomVisualizer {
  /** In-flight requests keyed by kind: `'segment'` or `'render'`. */
  #inflight = new Map();

  /** Aborts every DOM listener this instance registered, in one call. */
  #listeners = new AbortController();

  /** Aborts the catalogue fetch, which is not part of the busy state. */
  #catalogController = null;

  /** Cached DOM references, all created by `#build`. */
  #els = {};

  /** The `SegmentResponse` for the loaded photo, or `null`. */
  #scene = null;

  /** Plane name -> `{ tile_id, rotation_deg, grout_mm }`. */
  #selections = new Map();

  /** Tile id -> catalogue entry. */
  #tiles = new Map();

  /** Plane names present in the scene, ascending by `area_fraction`. */
  #hitOrder = [];

  #activePlane = null;
  #hoverPlane = null;
  #displayScale = 1;
  #busy = false;
  #destroyed = false;
  #photoUrl = null;
  #photoName = '';
  #resizeObserver = null;

  /**
   * @param {string|HTMLElement} containerSelectorOrElement Container selector
   *   or element. The component appends one `.rv-root` inside it.
   * @param {{
   *   apiBaseUrl?: string,
   *   initialTileId?: string,
   *   defaultPlane?: string,
   *   renderFormat?: string,
   *   onRenderComplete?: (result: any) => void,
   *   onSceneReady?: (scene: any) => void,
   *   onError?: (error: Error) => void,
   *   labels?: Record<string, string>,
   * }} [config]
   */
  constructor(containerSelectorOrElement, config = {}) {
    const container =
      typeof containerSelectorOrElement === 'string'
        ? (typeof document !== 'undefined'
            ? document.querySelector(containerSelectorOrElement)
            : null)
        : containerSelectorOrElement;

    if (!container || typeof container.appendChild !== 'function') {
      throw new Error(
        `RoomVisualizer: container not found for ${String(containerSelectorOrElement)}`
      );
    }

    this.container = container;
    this.config = {
      apiBaseUrl: '',
      initialTileId: null,
      defaultPlane: 'floor',
      renderFormat: 'png',
      ...config,
    };
    this.labels = { ...DEFAULT_LABELS, ...(config.labels || {}) };

    this.document = container.ownerDocument || (typeof document !== 'undefined' ? document : null);
    if (!this.document) throw new Error('RoomVisualizer: no document available');

    this.api = new ApiClient(this.config.apiBaseUrl);

    instanceCounter += 1;
    this.#uid = `rv-${instanceCounter}-${Math.random().toString(36).slice(2, 8)}`;

    this.#activePlane = this.config.defaultPlane || null;
    if (this.config.initialTileId && this.#activePlane) {
      this.#selections.set(this.#activePlane, {
        tile_id: this.config.initialTileId,
        rotation_deg: 0,
        grout_mm: null,
      });
    }

    this.#build();
    this.#loadTiles();
  }

  /** Per-instance id prefix, so several widgets can coexist on one page. */
  #uid = '';

  // ---------------------------------------------------------------- public API

  /**
   * Upload a photograph, analyse it, and draw the result.
   *
   * A second call while an upload is in flight is dropped, since re-uploading
   * the same photo has no useful semantics (Requirement 10.5).
   *
   * @param {File|Blob} file
   * @returns {Promise<any|null>} The scene, or `null` when dropped or failed.
   */
  loadPhoto(file) {
    if (!file) return Promise.resolve(null);
    return this.#run('segment', 'segment', async (signal) => {
      this.#setStatus(this.labels.analysing);
      this.#clearError();
      const scene = await this.api.segment(file, { signal });
      this.#photoName = (file && file.name) || '';
      this.#adoptScene(scene, file);
      this.#setStatus(
        format(this.labels.sceneReady, { count: (scene.planes || []).length })
      );
      this.#notify('scene-ready', scene, this.config.onSceneReady);
      await this.#maybeAutoRender();
      return scene;
    });
  }

  /**
   * Make a Structural_Plane the active target for tile selection.
   *
   * @param {string} name
   * @returns {boolean} `true` when the plane exists in the current scene.
   */
  selectPlane(name) {
    if (!this.#scene || !this.#planeByName(name)) return false;
    this.#activePlane = name;
    this.#syncPlaneButtons();
    this.#syncTileButtons();
    this.#syncControls();
    this.#drawOverlay();
    return true;
  }

  /**
   * Select a tile for a plane and re-render.
   *
   * @param {string} tileId
   * @param {string} [planeName] Defaults to the active plane.
   * @returns {Promise<any|null>} The render result, or `null`.
   */
  applyTile(tileId, planeName) {
    const plane = planeName || this.#activePlane;
    if (!plane) {
      this.#fail(new Error(this.labels.unknownPlaneError));
      return Promise.resolve(null);
    }
    if (this.#tiles.size > 0 && !this.#tiles.has(tileId)) {
      this.#fail(new Error(this.labels.unknownTileError));
      return Promise.resolve(null);
    }
    const current = this.#selections.get(plane) || { rotation_deg: 0, grout_mm: null };
    this.#selections.set(plane, { ...current, tile_id: tileId });
    if (planeName && planeName !== this.#activePlane && this.#planeByName(planeName)) {
      this.#activePlane = planeName;
      this.#syncPlaneButtons();
      this.#drawOverlay();
    }
    this.#syncTileButtons();
    this.#syncControls();
    return this.render();
  }

  /**
   * Set the in-plane rotation for a plane, in degrees, and re-render.
   *
   * @param {number} deg
   * @param {string} [planeName]
   * @returns {Promise<any|null>}
   */
  setRotation(deg, planeName) {
    return this.#setPlaneOption(planeName, 'rotation_deg', Number(deg) || 0);
  }

  /**
   * Set the grout width for a plane, in millimetres, and re-render.
   *
   * @param {number} mm
   * @param {string} [planeName]
   * @returns {Promise<any|null>}
   */
  setGrout(mm, planeName) {
    const value = mm === null || mm === undefined || mm === '' ? null : Number(mm);
    return this.#setPlaneOption(planeName, 'grout_mm', value);
  }

  /**
   * Composite the current selections against the cached scene.
   *
   * A render whose selection signature matches the in-flight render is dropped;
   * one that differs aborts the in-flight request and supersedes it, so rapid
   * tile clicking converges on the last choice (Requirement 10.5).
   *
   * @returns {Promise<any|null>}
   */
  render() {
    if (!this.#scene) return Promise.resolve(null);

    const payload = this.#buildPayload();
    if (!payload) {
      this.#drawPhoto();
      return Promise.resolve(null);
    }

    const signature = JSON.stringify(payload);
    return this.#run('render', signature, async (signal) => {
      this.#setStatus(this.labels.rendering);
      const result = await this.api.render(payload, { signal });
      this.#clearError();
      await this.#drawEncodedImage(result.mime, result.image);
      this.#syncCanvasLabel();
      this.#setStatus(format(this.labels.renderDone, { ms: result.render_ms }));
      this.#notify('render-complete', result, this.config.onRenderComplete);
      return result;
    });
  }

  /**
   * Clear the scene, the selections, and the canvas.
   *
   * A no-op after `destroy()`, since the DOM this would touch is gone.
   */
  reset() {
    if (this.#destroyed) return;
    this.#abortAll();
    this.#scene = null;
    this.#selections.clear();
    this.#hitOrder = [];
    this.#activePlane = this.config.defaultPlane || null;
    this.#hoverPlane = null;
    this.#photoName = '';
    this.#releasePhotoUrl();

    const { canvasImage, canvasOverlay, uploadInput, fileName, stageEmpty } = this.#els;
    this.#clearCanvas(canvasImage);
    this.#clearCanvas(canvasOverlay);
    if (canvasImage) canvasImage.classList.add('rv-hidden');
    if (canvasOverlay) canvasOverlay.classList.add('rv-hidden');
    if (stageEmpty) stageEmpty.classList.remove('rv-hidden');
    if (uploadInput) uploadInput.value = '';
    if (fileName) fileName.textContent = this.labels.noFileName;

    this.#syncPlaneButtons();
    this.#syncTileButtons();
    this.#syncControls();
    this.#syncCanvasLabel();
    this.#setStatus('');
    this.#clearError();
  }

  /** Remove listeners, abort in-flight requests, and empty the container. */
  destroy() {
    if (this.#destroyed) return;
    this.#destroyed = true;
    this.#abortAll();
    this.#listeners.abort();
    if (this.#resizeObserver) {
      this.#resizeObserver.disconnect();
      this.#resizeObserver = null;
    }
    this.#releasePhotoUrl();
    this.#scene = null;
    this.#selections.clear();
    this.#tiles.clear();
    if (this.#els.root && this.#els.root.parentNode) {
      this.#els.root.parentNode.removeChild(this.#els.root);
    }
    this.#els = {};
  }

  /**
   * A plain snapshot of the component state, safe to serialise.
   *
   * @returns {{ sceneId: string|null, planes: string[], selections: Record<string, object>, activePlane: string|null, busy: boolean }}
   */
  getState() {
    const selections = {};
    for (const [plane, spec] of this.#selections) selections[plane] = { ...spec };
    return {
      sceneId: this.#scene ? this.#scene.scene_id : null,
      planes: this.#scene ? (this.#scene.planes || []).map((p) => p.name) : [],
      selections,
      activePlane: this.#activePlane,
      busy: this.#busy,
    };
  }

  // ------------------------------------------------------------- construction

  /** Build the whole DOM subtree. Every class is `rv-`-prefixed. */
  #build() {
    const root = this.#el('div', 'rv-root');
    root.setAttribute('aria-busy', 'false');

    // --- stage: two stacked canvases plus the empty hint and the spinner ---
    const stage = this.#el('div', 'rv-stage');

    const canvasImage = this.#el('canvas', 'rv-canvas rv-canvas-image');
    canvasImage.setAttribute('role', 'img');
    canvasImage.setAttribute('aria-label', this.labels.canvasEmpty);
    canvasImage.classList.add('rv-hidden');

    const canvasOverlay = this.#el('canvas', 'rv-canvas rv-canvas-overlay');
    // The plane radiogroup is the accessible route to this affordance, so the
    // overlay itself is decorative (Requirement 10.7).
    canvasOverlay.setAttribute('aria-hidden', 'true');
    canvasOverlay.classList.add('rv-hidden');

    const stageEmpty = this.#el('div', 'rv-stage-empty');
    stageEmpty.textContent = this.labels.stageEmpty;

    const spinner = this.#el('div', 'rv-spinner');
    spinner.setAttribute('aria-hidden', 'true');

    stage.append(canvasImage, canvasOverlay, stageEmpty, spinner);

    // --- panel: upload, planes, tiles, controls ---
    const panel = this.#el('div', 'rv-panel');

    // Photo section. The input comes first in document order because the
    // stylesheet draws the input's focus ring on the following label.
    const photoSection = this.#el('div', 'rv-section');
    const photoTitle = this.#el('h3', 'rv-section-title');
    photoTitle.id = `${this.#uid}-photo-title`;
    photoTitle.textContent = this.labels.photoSection;

    const upload = this.#el('div', 'rv-upload');
    const uploadInput = this.#el('input', 'rv-upload-input');
    uploadInput.type = 'file';
    uploadInput.id = `${this.#uid}-file`;
    uploadInput.accept = 'image/jpeg,image/png,image/webp';
    const uploadLabel = this.#el('label', 'rv-upload-label');
    uploadLabel.htmlFor = uploadInput.id;
    uploadLabel.setAttribute('for', uploadInput.id);
    uploadLabel.textContent = this.labels.uploadLabel;
    const uploadHint = this.#el('p', 'rv-upload-hint');
    uploadHint.id = `${this.#uid}-upload-hint`;
    uploadHint.textContent = this.labels.uploadHint;
    uploadInput.setAttribute('aria-describedby', uploadHint.id);
    const fileName = this.#el('p', 'rv-file-name');
    fileName.textContent = this.labels.noFileName;

    upload.append(uploadInput, uploadLabel, uploadHint, fileName);
    photoSection.append(photoTitle, upload);

    // Plane section: a radiogroup of radio buttons with roving tabindex.
    const planeSection = this.#el('div', 'rv-section');
    const planeTitle = this.#el('h3', 'rv-section-title');
    planeTitle.id = `${this.#uid}-plane-title`;
    planeTitle.textContent = this.labels.planeSection;
    const planes = this.#el('div', 'rv-planes');
    planes.setAttribute('role', 'radiogroup');
    planes.setAttribute('aria-labelledby', planeTitle.id);
    const planesEmpty = this.#el('p', 'rv-planes-empty');
    planesEmpty.textContent = this.labels.planesEmpty;
    planeSection.append(planeTitle, planes, planesEmpty);

    // Tile section: a list of aria-pressed toggle buttons.
    const tileSection = this.#el('div', 'rv-section');
    const tileTitle = this.#el('h3', 'rv-section-title');
    tileTitle.id = `${this.#uid}-tile-title`;
    tileTitle.textContent = this.labels.tileSection;
    const tiles = this.#el('ul', 'rv-tiles');
    tiles.setAttribute('aria-labelledby', tileTitle.id);
    const tilesEmpty = this.#el('p', 'rv-tiles-empty');
    tilesEmpty.textContent = this.labels.tilesLoading;
    tileSection.append(tileTitle, tiles, tilesEmpty);

    // Controls section: rotation and grout.
    const controlsSection = this.#el('div', 'rv-section');
    const controlsTitle = this.#el('h3', 'rv-section-title');
    controlsTitle.textContent = this.labels.controlsSection;
    const controls = this.#el('div', 'rv-controls');

    const rotation = this.#buildControl({
      id: `${this.#uid}-rotation`,
      labelText: this.labels.rotationLabel,
      min: ROTATION_MIN,
      max: ROTATION_MAX,
      step: ROTATION_STEP,
      value: 0,
    });
    const grout = this.#buildControl({
      id: `${this.#uid}-grout`,
      labelText: this.labels.groutLabel,
      min: GROUT_MIN,
      max: GROUT_MAX,
      step: GROUT_STEP,
      value: 3,
    });
    controls.append(rotation.control, grout.control);
    controlsSection.append(controlsTitle, controls);

    panel.append(photoSection, planeSection, tileSection, controlsSection);

    // --- live regions ---
    const status = this.#el('div', 'rv-status');
    status.setAttribute('role', 'status');
    status.setAttribute('aria-live', 'polite');

    const error = this.#el('div', 'rv-error');
    error.setAttribute('role', 'alert');

    root.append(stage, panel, status, error);
    this.container.appendChild(root);

    this.#els = {
      root,
      stage,
      canvasImage,
      canvasOverlay,
      stageEmpty,
      spinner,
      panel,
      uploadInput,
      uploadLabel,
      fileName,
      planes,
      planesEmpty,
      tiles,
      tilesEmpty,
      rotationInput: rotation.input,
      rotationValue: rotation.value,
      groutInput: grout.input,
      groutValue: grout.value,
      status,
      error,
    };

    this.#bindEvents();
    this.#syncControls();
  }

  /**
   * Build one labelled range control.
   *
   * @param {{id: string, labelText: string, min: number, max: number, step: number, value: number}} spec
   */
  #buildControl(spec) {
    const control = this.#el('div', 'rv-control');
    const label = this.#el('label', 'rv-control-label');
    label.htmlFor = spec.id;
    label.setAttribute('for', spec.id);
    const labelText = this.#el('span', 'rv-control-text');
    labelText.textContent = spec.labelText;
    const value = this.#el('span', 'rv-control-value');
    label.append(labelText, value);

    const input = this.#el('input', 'rv-control-input');
    input.type = 'range';
    input.id = spec.id;
    input.min = String(spec.min);
    input.max = String(spec.max);
    input.step = String(spec.step);
    input.value = String(spec.value);

    control.append(label, input);
    return { control, input, value };
  }

  /** Register every DOM listener against `#listeners`, so destroy is one call. */
  #bindEvents() {
    const { signal } = this.#listeners;
    const {
      uploadInput,
      canvasOverlay,
      planes,
      tiles,
      rotationInput,
      groutInput,
      stage,
    } = this.#els;

    uploadInput.addEventListener(
      'change',
      () => {
        const file = uploadInput.files && uploadInput.files[0];
        if (file) this.loadPhoto(file);
      },
      { signal }
    );

    canvasOverlay.addEventListener(
      'pointermove',
      (event) => this.#onPointerMove(event),
      { signal }
    );
    canvasOverlay.addEventListener(
      'pointerleave',
      () => {
        if (this.#hoverPlane !== null) {
          this.#hoverPlane = null;
          this.#drawOverlay();
        }
      },
      { signal }
    );
    canvasOverlay.addEventListener('click', (event) => this.#onCanvasClick(event), { signal });

    planes.addEventListener('keydown', (event) => this.#onPlaneKeydown(event), { signal });
    tiles.addEventListener('keydown', (event) => this.#onTileKeydown(event), { signal });

    rotationInput.addEventListener(
      'input',
      () => {
        this.#els.rotationValue.textContent = format(this.labels.rotationValue, {
          value: trimNumber(rotationInput.value),
        });
      },
      { signal }
    );
    rotationInput.addEventListener(
      'change',
      () => this.setRotation(Number(rotationInput.value)),
      { signal }
    );

    groutInput.addEventListener(
      'input',
      () => {
        this.#els.groutValue.textContent = format(this.labels.groutValue, {
          value: trimNumber(groutInput.value),
        });
      },
      { signal }
    );
    groutInput.addEventListener(
      'change',
      () => this.setGrout(Number(groutInput.value)),
      { signal }
    );

    const view = this.document.defaultView;
    if (view && typeof view.ResizeObserver === 'function') {
      this.#resizeObserver = new view.ResizeObserver(() => this.#onResize());
      this.#resizeObserver.observe(stage);
    } else if (view) {
      view.addEventListener('resize', () => this.#onResize(), { signal });
    }
  }

  // ------------------------------------------------------------- catalogue

  /** Fetch the tile catalogue and build the tile buttons. */
  #loadTiles() {
    this.#catalogController = new AbortController();
    const signal = this.#catalogController.signal;
    Promise.resolve()
      .then(() => this.api.tiles({ signal }))
      .then((list) => {
        if (this.#destroyed) return;
        this.#tiles = new Map((list || []).map((tile) => [tile.id, tile]));
        this.#renderTileList();
        if (this.#tiles.size > 0) {
          this.#setStatus(format(this.labels.tilesLoaded, { count: this.#tiles.size }));
        }
        return this.#maybeAutoRender();
      })
      .catch((err) => {
        if (this.#destroyed) return;
        this.#fail(err);
      })
      .finally(() => {
        this.#catalogController = null;
      });
  }

  /** Rebuild the `<ul>` of tile buttons from the catalogue. */
  #renderTileList() {
    const { tiles, tilesEmpty } = this.#els;
    tiles.textContent = '';

    if (this.#tiles.size === 0) {
      tilesEmpty.textContent = this.labels.tilesEmpty;
      tilesEmpty.classList.remove('rv-hidden');
      return;
    }
    tilesEmpty.classList.add('rv-hidden');

    for (const tile of this.#tiles.values()) {
      const item = this.#el('li', 'rv-tile-item');
      const button = this.#el('button', 'rv-tile-button');
      button.type = 'button';
      button.setAttribute('aria-pressed', 'false');
      button.setAttribute(
        'aria-label',
        format(this.labels.tileAriaLabel, {
          name: tile.name || tile.id,
          width: trimNumber(tile.width_mm),
          height: trimNumber(tile.height_mm),
          finish: tile.finish || '',
        })
      );
      button.dataset.rvTileId = tile.id;

      const thumb = this.#el('span', 'rv-tile-thumb');
      thumb.setAttribute('aria-hidden', 'true');
      if (tile.thumbnail_url) {
        const url = /^https?:/i.test(tile.thumbnail_url)
          ? tile.thumbnail_url
          : this.api.url(tile.thumbnail_url);
        thumb.style.backgroundImage = `url("${url}")`;
      }

      const name = this.#el('span', 'rv-tile-name');
      name.textContent = tile.name || tile.id;
      const meta = this.#el('span', 'rv-tile-meta');
      meta.textContent = format(this.labels.tileMeta, {
        width: trimNumber(tile.width_mm),
        height: trimNumber(tile.height_mm),
        finish: tile.finish || '',
      });

      button.append(thumb, name, meta);
      button.addEventListener('click', () => this.applyTile(tile.id), {
        signal: this.#listeners.signal,
      });
      item.appendChild(button);
      tiles.appendChild(item);
    }

    this.#syncTileButtons();
    if (this.#busy) this.#applyBusyToControls(true);
  }

  // ----------------------------------------------------------------- scene

  /**
   * Adopt an analysis response: store it, draw the photo, build plane buttons.
   *
   * @param {any} scene
   * @param {File|Blob} file
   */
  #adoptScene(scene, file) {
    this.#scene = scene;
    const planeList = scene.planes || [];
    this.#hitOrder = planeList
      .slice()
      .sort((a, b) => (a.area_fraction || 0) - (b.area_fraction || 0))
      .map((plane) => plane.name);

    // Drop selections for planes this photo does not have.
    for (const name of Array.from(this.#selections.keys())) {
      if (!this.#planeByName(name)) this.#selections.delete(name);
    }

    // Prefer the configured default plane; otherwise the largest plane.
    if (!this.#activePlane || !this.#planeByName(this.#activePlane)) {
      this.#activePlane =
        this.#hitOrder.length > 0 ? this.#hitOrder[this.#hitOrder.length - 1] : null;
    }
    if (this.config.initialTileId && this.#activePlane && !this.#selections.has(this.#activePlane)) {
      this.#selections.set(this.#activePlane, {
        tile_id: this.config.initialTileId,
        rotation_deg: 0,
        grout_mm: null,
      });
    }

    this.#els.fileName.textContent = this.#photoName || this.labels.noFileName;
    this.#renderPlaneList();
    this.#syncTileButtons();
    this.#syncControls();
    this.#showPhoto(file);
  }

  /** Build the plane radiogroup for the current scene. */
  #renderPlaneList() {
    const { planes, planesEmpty } = this.#els;
    planes.textContent = '';

    const present = PLANE_NAMES.filter((name) => this.#planeByName(name));
    if (present.length === 0) {
      planesEmpty.classList.remove('rv-hidden');
      return;
    }
    planesEmpty.classList.add('rv-hidden');

    for (const name of present) {
      const button = this.#el('button', 'rv-plane-button');
      button.type = 'button';
      button.setAttribute('role', 'radio');
      button.setAttribute('aria-checked', String(name === this.#activePlane));
      button.dataset.rvPlane = name;
      button.textContent = this.#planeLabel(name);
      button.addEventListener('click', () => this.selectPlane(name), {
        signal: this.#listeners.signal,
      });
      planes.appendChild(button);
    }
    this.#syncPlaneButtons();
  }

  /**
   * Look up a plane entry by name in the current scene.
   *
   * @param {string} name
   * @returns {any|null}
   */
  #planeByName(name) {
    if (!this.#scene || !name) return null;
    const list = this.#scene.planes || [];
    return list.find((plane) => plane.name === name) || null;
  }

  /**
   * Display label for a plane name.
   *
   * @param {string} name
   * @returns {string}
   */
  #planeLabel(name) {
    const key = PLANE_LABEL_KEYS[name];
    return key && this.labels[key] ? this.labels[key] : name;
  }

  // ------------------------------------------------------------- rendering

  /**
   * Build the `/api/render` payload, or `null` when nothing is selected.
   *
   * @returns {object|null}
   */
  #buildPayload() {
    if (!this.#scene) return null;
    const planes = {};
    for (const name of PLANE_NAMES) {
      const spec = this.#selections.get(name);
      if (!spec || !spec.tile_id || !this.#planeByName(name)) continue;
      const entry = { tile_id: spec.tile_id, rotation_deg: spec.rotation_deg || 0 };
      if (spec.grout_mm !== null && spec.grout_mm !== undefined) {
        entry.grout_mm = spec.grout_mm;
      }
      planes[name] = entry;
    }
    if (Object.keys(planes).length === 0) return null;
    return {
      scene_id: this.#scene.scene_id,
      planes,
      format: this.config.renderFormat || 'png',
    };
  }

  /** Render when a scene, a selection, and a resolved catalogue all exist. */
  #maybeAutoRender() {
    if (!this.#scene) return Promise.resolve(null);
    if (!this.#buildPayload()) return Promise.resolve(null);
    return this.render();
  }

  /**
   * Update a per-plane option and re-render when that plane has a tile.
   *
   * @param {string|undefined} planeName
   * @param {'rotation_deg'|'grout_mm'} key
   * @param {number|null} value
   * @returns {Promise<any|null>}
   */
  #setPlaneOption(planeName, key, value) {
    const plane = planeName || this.#activePlane;
    if (!plane) return Promise.resolve(null);
    const current = this.#selections.get(plane) || {
      tile_id: null,
      rotation_deg: 0,
      grout_mm: null,
    };
    this.#selections.set(plane, { ...current, [key]: value });
    this.#syncControls();
    if (!this.#selections.get(plane).tile_id) return Promise.resolve(null);
    return this.render();
  }

  /**
   * Draw the freshly uploaded photograph, before any tiles are applied.
   *
   * @param {File|Blob} file
   */
  #showPhoto(file) {
    const { canvasImage, canvasOverlay, stageEmpty } = this.#els;
    canvasImage.width = this.#scene.width;
    canvasImage.height = this.#scene.height;
    canvasImage.classList.remove('rv-hidden');
    canvasOverlay.classList.remove('rv-hidden');
    stageEmpty.classList.add('rv-hidden');
    this.#syncCanvasLabel();

    this.#releasePhotoUrl();
    const view = this.document.defaultView;
    if (view && view.URL && typeof view.URL.createObjectURL === 'function' && file) {
      this.#photoUrl = view.URL.createObjectURL(file);
      this.#drawUrl(this.#photoUrl).catch(() => {
        /* A failed decode leaves the previous pixels; the render will replace them. */
      });
    }
    this.#onResize();
  }

  /** Redraw the un-tiled photograph, used when every selection is cleared. */
  #drawPhoto() {
    if (this.#photoUrl) {
      this.#drawUrl(this.#photoUrl).catch(() => {});
    }
    this.#syncCanvasLabel();
  }

  /**
   * Draw a base64 payload returned by `/api/render`.
   *
   * @param {string} mime
   * @param {string} base64
   */
  #drawEncodedImage(mime, base64) {
    return this.#drawUrl(`data:${mime || 'image/png'};base64,${base64}`);
  }

  /**
   * Draw an image URL onto the base canvas at scene resolution.
   *
   * @param {string} url
   * @returns {Promise<void>}
   */
  #drawUrl(url) {
    const view = this.document.defaultView;
    const { canvasImage } = this.#els;
    if (!view || typeof view.Image !== 'function' || !canvasImage) return Promise.resolve();

    return new Promise((resolve, reject) => {
      const img = new view.Image();
      img.onload = () => {
        const ctx = canvasImage.getContext ? canvasImage.getContext('2d') : null;
        if (ctx) {
          canvasImage.width = img.naturalWidth || canvasImage.width;
          canvasImage.height = img.naturalHeight || canvasImage.height;
          ctx.clearRect(0, 0, canvasImage.width, canvasImage.height);
          ctx.drawImage(img, 0, 0, canvasImage.width, canvasImage.height);
        }
        this.#onResize();
        resolve();
      };
      img.onerror = () => reject(new Error('Image could not be decoded for display.'));
      img.src = url;
    });
  }

  /**
   * Clear a canvas.
   *
   * @param {HTMLCanvasElement} canvas
   */
  #clearCanvas(canvas) {
    if (!canvas || !canvas.getContext) return;
    const ctx = canvas.getContext('2d');
    if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  // --------------------------------------------------------------- overlay

  /**
   * Recompute `displayScale` and resize the overlay's backing store.
   *
   * `displayScale = canvasClientWidth / sceneWidth`: one factor for both axes,
   * multiplied into contour points when drawing and divided out of pointer
   * coordinates when hit-testing.
   */
  #onResize() {
    const { canvasOverlay } = this.#els;
    if (!canvasOverlay || !this.#scene) return;

    const clientWidth = canvasOverlay.clientWidth || this.#measuredWidth(canvasOverlay);
    const scale = this.#scene.width > 0 && clientWidth > 0 ? clientWidth / this.#scene.width : 1;
    this.#displayScale = scale > 0 ? scale : 1;

    const width = Math.max(1, Math.round(this.#scene.width * this.#displayScale));
    const height = Math.max(1, Math.round(this.#scene.height * this.#displayScale));
    if (canvasOverlay.width !== width) canvasOverlay.width = width;
    if (canvasOverlay.height !== height) canvasOverlay.height = height;

    this.#drawOverlay();
  }

  /**
   * Fall back to the measured box when `clientWidth` is unavailable.
   *
   * @param {HTMLElement} element
   * @returns {number}
   */
  #measuredWidth(element) {
    if (typeof element.getBoundingClientRect !== 'function') return 0;
    const rect = element.getBoundingClientRect();
    return rect && rect.width ? rect.width : 0;
  }

  /** Draw plane outlines plus the hover and selection highlights. */
  #drawOverlay() {
    const { canvasOverlay } = this.#els;
    if (!canvasOverlay || !canvasOverlay.getContext) return;
    const ctx = canvasOverlay.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);
    if (!this.#scene) return;

    const accent = this.#accentColour();
    const scale = this.#displayScale;

    for (const name of PLANE_NAMES) {
      const plane = this.#planeByName(name);
      if (!plane) continue;

      const isActive = name === this.#activePlane;
      const isHover = name === this.#hoverPlane;
      if (!isActive && !isHover) continue;

      // Filled highlight from the full contour.
      const contour = plane.contour || plane.bounding_points;
      if (Array.isArray(contour) && contour.length >= 3) {
        ctx.beginPath();
        contour.forEach(([x, y], index) => {
          const px = x * scale;
          const py = y * scale;
          if (index === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.closePath();
        ctx.globalAlpha = isActive ? 0.22 : 0.12;
        ctx.fillStyle = accent;
        ctx.fill();
        ctx.globalAlpha = 1;
      }

      // Selection outline from the four bounding points.
      const quad = plane.bounding_points;
      if (Array.isArray(quad) && quad.length >= 3) {
        ctx.beginPath();
        quad.forEach(([x, y], index) => {
          const px = x * scale;
          const py = y * scale;
          if (index === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.closePath();
        ctx.lineWidth = isActive ? 3 : 2;
        ctx.strokeStyle = accent;
        ctx.setLineDash(isActive ? [] : [6, 5]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  }

  /**
   * Resolve `--rv-accent` from the root, so the overlay follows host theming.
   *
   * @returns {string}
   */
  #accentColour() {
    const view = this.document.defaultView;
    if (!view || typeof view.getComputedStyle !== 'function') return FALLBACK_ACCENT;
    try {
      const value = view.getComputedStyle(this.#els.root).getPropertyValue('--rv-accent');
      return value && value.trim() ? value.trim() : FALLBACK_ACCENT;
    } catch {
      return FALLBACK_ACCENT;
    }
  }

  /**
   * Convert a pointer event to scene coordinates by inverting `displayScale`.
   *
   * @param {PointerEvent|MouseEvent} event
   * @returns {{x: number, y: number}|null}
   */
  #eventToScene(event) {
    const { canvasOverlay } = this.#els;
    if (!canvasOverlay || typeof canvasOverlay.getBoundingClientRect !== 'function') return null;
    const rect = canvasOverlay.getBoundingClientRect();
    const scale = this.#displayScale || 1;
    return {
      x: (event.clientX - (rect.left || 0)) / scale,
      y: (event.clientY - (rect.top || 0)) / scale,
    };
  }

  /**
   * Hit-test the plane quads in ascending `area_fraction` order, so a small
   * plane nested inside a larger one still wins the pointer.
   *
   * @param {number} x Scene x.
   * @param {number} y Scene y.
   * @returns {string|null}
   */
  #planeAt(x, y) {
    for (const name of this.#hitOrder) {
      const plane = this.#planeByName(name);
      if (!plane) continue;
      const quad = plane.bounding_points || plane.contour;
      if (pointInPolygon(x, y, quad || [])) return name;
    }
    return null;
  }

  /** @param {PointerEvent} event */
  #onPointerMove(event) {
    if (!this.#scene) return;
    const point = this.#eventToScene(event);
    if (!point) return;
    const hit = this.#planeAt(point.x, point.y);
    if (hit !== this.#hoverPlane) {
      this.#hoverPlane = hit;
      this.#drawOverlay();
    }
  }

  /** @param {MouseEvent} event */
  #onCanvasClick(event) {
    if (!this.#scene) return;
    const point = this.#eventToScene(event);
    if (!point) return;
    const hit = this.#planeAt(point.x, point.y);
    if (hit) this.selectPlane(hit);
  }

  // -------------------------------------------------------------- keyboard

  /**
   * Arrow-key traversal for the plane radiogroup: moving focus also selects,
   * which is the expected behaviour for a radio group.
   *
   * @param {KeyboardEvent} event
   */
  #onPlaneKeydown(event) {
    const buttons = this.#planeButtons();
    if (buttons.length === 0) return;
    const currentIndex = buttons.indexOf(event.target);
    if (currentIndex === -1) return;

    let next = currentIndex;
    switch (event.key) {
      case 'ArrowRight':
      case 'ArrowDown':
        next = (currentIndex + 1) % buttons.length;
        break;
      case 'ArrowLeft':
      case 'ArrowUp':
        next = (currentIndex - 1 + buttons.length) % buttons.length;
        break;
      case 'Home':
        next = 0;
        break;
      case 'End':
        next = buttons.length - 1;
        break;
      default:
        return;
    }
    event.preventDefault();
    const button = buttons[next];
    this.selectPlane(button.dataset.rvPlane);
    if (typeof button.focus === 'function') button.focus();
  }

  /**
   * Arrow-key traversal for the tile list: focus only, so browsing tiles by
   * keyboard does not fire a render per key press.
   *
   * @param {KeyboardEvent} event
   */
  #onTileKeydown(event) {
    const buttons = this.#tileButtons();
    if (buttons.length === 0) return;
    const currentIndex = buttons.indexOf(event.target);
    if (currentIndex === -1) return;

    let next = currentIndex;
    switch (event.key) {
      case 'ArrowRight':
      case 'ArrowDown':
        next = Math.min(currentIndex + 1, buttons.length - 1);
        break;
      case 'ArrowLeft':
      case 'ArrowUp':
        next = Math.max(currentIndex - 1, 0);
        break;
      case 'Home':
        next = 0;
        break;
      case 'End':
        next = buttons.length - 1;
        break;
      default:
        return;
    }
    event.preventDefault();
    const button = buttons[next];
    if (button && typeof button.focus === 'function') button.focus();
  }

  /** @returns {HTMLButtonElement[]} */
  #planeButtons() {
    if (!this.#els.planes) return [];
    return Array.from(this.#els.planes.querySelectorAll('.rv-plane-button'));
  }

  /** @returns {HTMLButtonElement[]} */
  #tileButtons() {
    if (!this.#els.tiles) return [];
    return Array.from(this.#els.tiles.querySelectorAll('.rv-tile-button'));
  }

  // ------------------------------------------------------------------ state

  /** Reflect the active plane in `aria-checked` and the roving tabindex. */
  #syncPlaneButtons() {
    const buttons = this.#planeButtons();
    let hasChecked = false;
    for (const button of buttons) {
      const checked = button.dataset.rvPlane === this.#activePlane;
      button.setAttribute('aria-checked', String(checked));
      button.tabIndex = checked ? 0 : -1;
      if (checked) hasChecked = true;
    }
    // A radiogroup must stay reachable by Tab even with nothing selected.
    if (!hasChecked && buttons.length > 0) buttons[0].tabIndex = 0;
  }

  /** Reflect the active plane's tile selection in `aria-pressed`. */
  #syncTileButtons() {
    const selected = this.#activePlane ? this.#selections.get(this.#activePlane) : null;
    const activeTile = selected ? selected.tile_id : null;
    for (const button of this.#tileButtons()) {
      button.setAttribute('aria-pressed', String(button.dataset.rvTileId === activeTile));
    }
  }

  /** Push the active plane's rotation and grout into the range inputs. */
  #syncControls() {
    const { rotationInput, rotationValue, groutInput, groutValue } = this.#els;
    if (!rotationInput || !groutInput) return;

    const spec = this.#activePlane ? this.#selections.get(this.#activePlane) : null;
    const rotation = spec && spec.rotation_deg ? spec.rotation_deg : 0;
    const grout = spec && spec.grout_mm !== null && spec.grout_mm !== undefined ? spec.grout_mm : 3;

    rotationInput.value = String(rotation);
    rotationValue.textContent = format(this.labels.rotationValue, {
      value: trimNumber(rotation),
    });
    groutInput.value = String(grout);
    groutValue.textContent = format(this.labels.groutValue, { value: trimNumber(grout) });

    const enabled = Boolean(this.#scene && this.#activePlane);
    rotationInput.disabled = !enabled;
    groutInput.disabled = !enabled;
  }

  /** Describe the current composition on the `role="img"` canvas. */
  #syncCanvasLabel() {
    const { canvasImage } = this.#els;
    if (!canvasImage) return;

    if (!this.#scene) {
      canvasImage.setAttribute('aria-label', this.labels.canvasEmpty);
      return;
    }

    const parts = [];
    for (const name of PLANE_NAMES) {
      const spec = this.#selections.get(name);
      if (!spec || !spec.tile_id || !this.#planeByName(name)) continue;
      const tile = this.#tiles.get(spec.tile_id);
      parts.push(
        format(this.labels.canvasSelection, {
          tile: tile && tile.name ? tile.name : spec.tile_id,
          plane: this.#planeLabel(name).toLowerCase(),
        })
      );
    }

    canvasImage.setAttribute(
      'aria-label',
      parts.length === 0
        ? this.labels.canvasPhoto
        : format(this.labels.canvasComposition, { selections: parts.join(', ') })
    );
  }

  // ------------------------------------------------------------ concurrency

  /**
   * Run one request of the given kind, enforcing the duplicate-submission and
   * supersession rules of Requirement 10.5.
   *
   * @param {'segment'|'render'} kind
   * @param {string} signature Identity of the requested work.
   * @param {(signal: AbortSignal) => Promise<any>} task
   * @returns {Promise<any|null>} Never rejects: failures land in `#fail`.
   */
  #run(kind, signature, task) {
    if (this.#destroyed) return Promise.resolve(null);

    const existing = this.#inflight.get(kind);
    if (existing) {
      // Same work, or a second upload: drop it and join the in-flight promise.
      if (kind === 'segment' || existing.signature === signature) return existing.promise;
      // Different work: the newest selection wins.
      existing.controller.abort();
      this.#inflight.delete(kind);
    }

    const controller = new AbortController();
    const entry = { controller, signature, promise: null };
    this.#inflight.set(kind, entry);
    this.#syncBusy();

    entry.promise = Promise.resolve()
      .then(() => task(controller.signal))
      .then(
        (value) => value,
        (err) => {
          this.#fail(err);
          return null;
        }
      )
      .finally(() => {
        if (this.#inflight.get(kind) === entry) {
          this.#inflight.delete(kind);
          this.#syncBusy();
        }
      });

    return entry.promise;
  }

  /** Abort every in-flight request without reporting the aborts as errors. */
  #abortAll() {
    for (const entry of this.#inflight.values()) entry.controller.abort();
    this.#inflight.clear();
    if (this.#catalogController) {
      this.#catalogController.abort();
      this.#catalogController = null;
    }
    this.#syncBusy();
  }

  /** Recompute the busy state from `#inflight` and apply it once on change. */
  #syncBusy() {
    const busy = this.#inflight.size > 0;
    if (busy === this.#busy) return;
    this.#busy = busy;

    const { root, spinner } = this.#els;
    if (root) root.setAttribute('aria-busy', String(busy));
    if (spinner) spinner.classList.toggle('rv-is-visible', busy);
    this.#applyBusyToControls(busy);
    this.#emit('busy-change', { busy, kinds: Array.from(this.#inflight.keys()) });
  }

  /**
   * Disable the controls that would submit new work while a request is running.
   *
   * @param {boolean} busy
   */
  #applyBusyToControls(busy) {
    if (this.#els.uploadInput) this.#els.uploadInput.disabled = busy;
    for (const button of this.#tileButtons()) button.disabled = busy;
  }

  // ------------------------------------------------------- status and events

  /**
   * Set the polite status text. An empty string collapses the region.
   *
   * @param {string} text
   */
  #setStatus(text) {
    if (this.#els.status) this.#els.status.textContent = text || '';
  }

  /** Clear the alert region. */
  #clearError() {
    if (this.#els.error) this.#els.error.textContent = '';
  }

  /**
   * Report a failure: the message is displayed verbatim, because `ApiError`
   * messages are written for shoppers (Requirement 10.6). Aborts are silent —
   * a superseded render is not a failure.
   *
   * @param {Error} err
   */
  #fail(err) {
    if (err && err.name === 'AbortError') return;
    const message = err && err.message ? err.message : this.labels.genericError;
    if (this.#els.error) this.#els.error.textContent = message;
    this.#setStatus('');
    this.#notify('error', err, this.config.onError);
  }

  /**
   * Invoke a config callback and dispatch the matching `CustomEvent`.
   *
   * @param {string} name Event name without the `rv:` prefix.
   * @param {any} detail
   * @param {Function} [callback]
   */
  #notify(name, detail, callback) {
    if (typeof callback === 'function') {
      try {
        callback(detail);
      } catch (err) {
        // A host callback must not break the widget's own state machine.
        if (this.document.defaultView && this.document.defaultView.console) {
          this.document.defaultView.console.error('RoomVisualizer callback failed', err);
        }
      }
    }
    this.#emit(name, detail);
  }

  /**
   * Dispatch `rv:<name>` on the supplied container.
   *
   * @param {string} name
   * @param {any} detail
   */
  #emit(name, detail) {
    const view = this.document.defaultView;
    if (!this.container || !view || typeof view.CustomEvent !== 'function') return;
    this.container.dispatchEvent(
      new view.CustomEvent(`rv:${name}`, { detail, bubbles: true })
    );
  }

  // ---------------------------------------------------------------- helpers

  /**
   * Create an element with a class list. Every class is `rv-`-prefixed.
   *
   * @param {string} tag
   * @param {string} className
   * @returns {any}
   */
  #el(tag, className) {
    const element = this.document.createElement(tag);
    if (className) element.className = className;
    return element;
  }

  /** Revoke the object URL held for the uploaded photograph. */
  #releasePhotoUrl() {
    const view = this.document ? this.document.defaultView : null;
    if (this.#photoUrl && view && view.URL && typeof view.URL.revokeObjectURL === 'function') {
      view.URL.revokeObjectURL(this.#photoUrl);
    }
    this.#photoUrl = null;
  }
}

if (typeof window !== 'undefined') {
  window.RoomVisualizer = RoomVisualizer;
}

export default RoomVisualizer;
