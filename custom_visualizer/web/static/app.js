const state = {
  captured: false,
  page: 0,
  totalPages: 1,
  selectedFrameIdx: 0,
  framePickerPage: 0,
  framePickerTotalPages: 1,
  pendingFrameIdx: 0,
  layout: null,
  sourceImages: [],
  imageKeys: [],
  playing: false,
  timer: null,
  playToken: 0,
  frameIndex: 0,
  frameCount: 0,
  dynamicFrames: [],
  dynamicAxisName: '',
  renderToken: 0,
};

const FRAME_PICKER_PAGE_SIZE = 16;
const FRAME_PICKER_STRIDE = 16;
const imageCache = new Map();

const el = (id) => document.getElementById(id);

function setStatus(text, kind = 'idle') {
  const badge = el('statusBadge');
  badge.textContent = text;
  badge.className = `status-badge ${kind}`;
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const data = await res.json();
  if (!res.ok || data.ok === false) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return data;
}

function fillSelect(select, values) {
  select.innerHTML = '<option value="all">全部</option>';
  values.forEach((value) => {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  });
  select.disabled = false;
}

function fillDatalist(datalist, values) {
  datalist.innerHTML = '';
  (values || []).forEach((value) => {
    if (!value) return;
    const option = document.createElement('option');
    option.value = value;
    datalist.appendChild(option);
  });
}

function applyPathHistory(settings) {
  el('checkpoint').value = settings.checkpoint || '';
  el('dataset').value = settings.dataset || '';
  fillDatalist(el('checkpointHistory'), settings.checkpoint_history || [settings.checkpoint]);
  fillDatalist(el('datasetHistory'), settings.dataset_history || [settings.dataset]);
}

function updateSelectedFrameText() {
  el('selectedFrameText').textContent = `Frame ${state.selectedFrameIdx}`;
}
function updateAttentionLegend() {
  const legend = el('attentionLegend');
  if (!legend) return;
  legend.hidden = !state.captured || el('overlayToggle').disabled || !el('overlayToggle').checked;
}

function selections() {
  const mode = el('modeSelect').value;
  const params = new URLSearchParams();
  params.set('mode', mode === 'static' ? 'static' : mode);
  params.set('step', el('stepSelect').value);
  params.set('layer', el('layerSelect').value);
  params.set('head', el('headSelect').value);
  params.set('overlay', el('overlayToggle').checked ? '1' : '0');
  return params;
}

function decodeFloat32(payload) {
  const binary = atob(payload.data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return { values: new Float32Array(bytes.buffer), shape: payload.shape };
}

function viridis(t) {
  const stops = [
    [0.00, 68, 1, 84], [0.13, 71, 44, 122], [0.25, 59, 81, 139],
    [0.38, 44, 113, 142], [0.50, 33, 144, 141], [0.63, 39, 173, 129],
    [0.75, 92, 200, 99], [0.88, 170, 220, 50], [1.00, 253, 231, 37],
  ];
  const x = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0));
  for (let i = 1; i < stops.length; i += 1) {
    if (x <= stops[i][0]) {
      const a = stops[i - 1];
      const b = stops[i];
      const f = (x - a[0]) / (b[0] - a[0]);
      return [
        Math.round(a[1] + (b[1] - a[1]) * f),
        Math.round(a[2] + (b[2] - a[2]) * f),
        Math.round(a[3] + (b[3] - a[3]) * f),
      ];
    }
  }
  return [253, 231, 37];
}

function normalizeValues(values) {
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    if (value < min) min = value;
    if (value > max) max = value;
  }
  const span = max > min ? max - min : 1;
  return { min, span };
}

function drawMatrix(canvas, item) {
  const decoded = decodeFloat32(item.probs);
  const rows = decoded.shape[0];
  const cols = decoded.shape[1];
  const values = decoded.values;
  canvas.width = 520;
  canvas.height = 370;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const margin = { left: 48, top: 14, right: 12, bottom: 42 };
  const plotW = canvas.width - margin.left - margin.right;
  const plotH = canvas.height - margin.top - margin.bottom;
  const imageData = ctx.createImageData(cols, rows);
  const { min, span } = normalizeValues(values);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const idx = r * cols + c;
      const [red, green, blue] = viridis((values[idx] - min) / span);
      const p = idx * 4;
      imageData.data[p] = red;
      imageData.data[p + 1] = green;
      imageData.data[p + 2] = blue;
      imageData.data[p + 3] = 255;
    }
  }
  const offscreen = document.createElement('canvas');
  offscreen.width = cols;
  offscreen.height = rows;
  offscreen.getContext('2d').putImageData(imageData, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(offscreen, margin.left, margin.top, plotW, plotH);
  ctx.imageSmoothingEnabled = true;

  drawPartitions(ctx, item, margin, plotW, plotH);
  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 1;
  ctx.strokeRect(margin.left, margin.top, plotW, plotH);
  ctx.fillStyle = '#475569';
  ctx.font = '12px "Microsoft YaHei UI", sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Key tokens', margin.left + plotW / 2, canvas.height - 14);
  ctx.save();
  ctx.translate(15, margin.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Action tokens', 0, 0);
  ctx.restore();
}

function drawPartitions(ctx, item, margin, plotW, plotH) {
  if (!state.layout) return;
  const layout = state.layout;
  const cols = item.probs.shape[1];
  const drawLine = (token, color, width, dash = []) => {
    if (token <= 0 || token >= cols) return;
    const x = margin.left + (token / cols) * plotW;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(x, margin.top);
    ctx.lineTo(x, margin.top + plotH);
    ctx.stroke();
    ctx.restore();
  };
  drawLine(layout.image_tokens, '#ffffff', 1.4, [5, 4]);
  drawLine(layout.image_tokens + layout.language_tokens, '#ffffff', 1.4, [5, 4]);
  if (item.type === 'self_attn') drawLine(layout.prefix_tokens, '#ef4444', 2.0);
}

function meanImageAttention(item) {
  const decoded = decodeFloat32(item.probs);
  const rows = decoded.shape[0];
  const cols = decoded.shape[1];
  const imageTokens = state.layout.image_tokens;
  const scores = new Float32Array(imageTokens);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < imageTokens && c < cols; c += 1) {
      scores[c] += decoded.values[r * cols + c];
    }
  }
  for (let c = 0; c < imageTokens; c += 1) scores[c] /= Math.max(1, rows);
  return scores;
}

function loadImage(src) {
  if (imageCache.has(src)) return imageCache.get(src);
  const promise = new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = src;
  });
  imageCache.set(src, promise);
  return promise;
}

async function drawOverlay(canvas, item, renderToken = state.renderToken) {
  const layout = state.layout;
  if (!layout || layout.overlay_reason || !layout.image_layouts || !layout.image_layouts.length) {
    drawMatrix(canvas, item);
    return;
  }
  const loaded = await Promise.all(state.sourceImages.map((image) => loadImage(image.src)));
  const scores = meanImageAttention(item);
  canvas.width = 520;
  canvas.height = 360;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const gap = 6;
  const count = Math.min(loaded.length, layout.image_layouts.length);
  const slotW = (canvas.width - gap * (count - 1)) / Math.max(1, count);
  for (let i = 0; i < count; i += 1) {
    const image = loaded[i];
    const imageLayout = layout.image_layouts[i];
    const slotX = i * (slotW + gap);
    const scale = Math.min(slotW / image.width, canvas.height / image.height);
    const drawW = image.width * scale;
    const drawH = image.height * scale;
    const x = slotX + (slotW - drawW) / 2;
    const y = (canvas.height - drawH) / 2;
    ctx.drawImage(image, x, y, drawW, drawH);
    drawHeatmapOverlay(ctx, scores, imageLayout, x, y, drawW, drawH);
  }
}

function drawHeatmapOverlay(ctx, scores, imageLayout, x, y, width, height) {
  const start = imageLayout.token_start;
  const count = imageLayout.token_count;
  const viewScores = scores.slice(start, start + count);
  const { min, span } = normalizeValues(viewScores);
  const gridW = imageLayout.grid_width;
  const gridH = imageLayout.grid_height;
  const patchW = width / gridW;
  const patchH = height / gridH;

  ctx.save();
  ctx.globalAlpha = 0.56;
  for (let row = 0; row < gridH; row += 1) {
    for (let col = 0; col < gridW; col += 1) {
      const idx = row * gridW + col;
      if (idx >= count) continue;
      const [red, green, blue] = viridis((viewScores[idx] - min) / span);
      ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
      ctx.fillRect(x + col * patchW, y + row * patchH, Math.ceil(patchW), Math.ceil(patchH));
    }
  }
  ctx.restore();

  ctx.save();
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.28)';
  ctx.lineWidth = 0.8;
  for (let col = 1; col < gridW; col += 1) {
    const lineX = x + col * patchW;
    ctx.beginPath();
    ctx.moveTo(lineX, y);
    ctx.lineTo(lineX, y + height);
    ctx.stroke();
  }
  for (let row = 1; row < gridH; row += 1) {
    const lineY = y + row * patchH;
    ctx.beginPath();
    ctx.moveTo(x, lineY);
    ctx.lineTo(x + width, lineY);
    ctx.stroke();
  }
  ctx.restore();
}

function renderCards(items) {
  updateAttentionLegend();
  const grid = el('mapGrid');
  const renderToken = ++state.renderToken;
  const existingCards = Array.from(grid.querySelectorAll('.map-card'));
  const canReuse = items.length > 0 && existingCards.length === items.length;

  grid.classList.toggle('empty', items.length === 0);
  if (!items.length) {
    grid.innerHTML = '';
    const empty = document.createElement('div');
    empty.className = 'empty-state';
    empty.textContent = '当前条件没有匹配的 attention map';
    grid.appendChild(empty);
    return;
  }

  if (!canReuse) {
    grid.innerHTML = '';
    items.forEach(() => {
      const card = document.createElement('article');
      card.className = 'map-card';
      card.innerHTML = '<div class="map-title"><span></span></div><canvas></canvas>';
      grid.appendChild(card);
    });
  }

  Array.from(grid.querySelectorAll('.map-card')).forEach((card, idx) => {
    const item = items[idx];
    const title = card.querySelector('.map-title span');
    const canvas = card.querySelector('canvas');
    title.textContent = item.title;
    canvas.setAttribute('aria-label', item.title);
    if (el('overlayToggle').checked) {
      drawOverlay(canvas, item, renderToken).then(() => {
        if (renderToken !== state.renderToken) return;
      }).catch(() => {
        if (renderToken === state.renderToken) drawMatrix(canvas, item);
      });
    } else {
      drawMatrix(canvas, item);
    }
  });
}

function updatePager() {
  el('pageInfo').textContent = `${state.page + 1} / ${state.totalPages}`;
  el('pagePrev').disabled = !state.captured || state.page <= 0 || el('modeSelect').value !== 'static';
  el('pageNext').disabled = !state.captured || state.page + 1 >= state.totalPages || el('modeSelect').value !== 'static';
}

function enforceDynamicInputs() {
  ['stepSelect', 'layerSelect', 'headSelect'].forEach((id) => el(id).disabled = !state.captured);
  const mode = el('modeSelect').value;
  if (!state.captured || mode === 'static') return;
  const axisByMode = { steps: 'stepSelect', layers: 'layerSelect', heads: 'headSelect' };
  const axisId = axisByMode[mode];
  el(axisId).value = 'all';
  el(axisId).disabled = true;
}

async function loadStatic(resetPage = true) {
  stopPlayback();
  if (!state.captured) return;
  if (resetPage) state.page = 0;
  const params = selections();
  params.set('page', state.page);
  const data = await api(`/api/items?${params.toString()}`);
  state.totalPages = data.total_pages;
  renderCards(data.items);
  el('positionText').textContent = `当前显示 ${data.items.length} / ${data.total} 张 · 第 ${state.page + 1}/${state.totalPages} 页`;
  updatePager();
}

async function prepareDynamicFrames() {
  const params = selections();
  const data = await api(`/api/dynamic?${params.toString()}`);
  state.dynamicFrames = data.frames || [];
  state.dynamicAxisName = data.axis_name || '';
  state.frameCount = state.dynamicFrames.length;
  state.frameIndex = 0;
  el('playBtn').disabled = state.frameCount === 0;
  el('prevBtn').disabled = true;
  el('nextBtn').disabled = state.frameCount <= 1;
  renderDynamicFrame();
}

function renderDynamicFrame() {
  if (!state.dynamicFrames.length) {
    renderCards([]);
    return;
  }
  const frame = state.dynamicFrames[state.frameIndex];
  renderCards(frame.items || []);
  el('positionText').textContent = `第 ${state.frameIndex + 1}/${state.frameCount} 帧 · ${state.dynamicAxisName}=${frame.axis_value} · ${(frame.items || []).length} 张`;
  el('prevBtn').disabled = state.frameIndex <= 0;
  el('nextBtn').disabled = state.frameIndex + 1 >= state.frameCount && !el('loopToggle').checked;
}

function moveFrame(delta) {
  if (!state.frameCount) return;
  let next = state.frameIndex + delta;
  if (next >= state.frameCount) next = el('loopToggle').checked ? 0 : state.frameCount - 1;
  if (next < 0) next = 0;
  state.frameIndex = next;
  renderDynamicFrame();
}

function stopPlayback() {
  state.playing = false;
  state.playToken += 1;
  if (state.timer) clearTimeout(state.timer);
  state.timer = null;
  el('playBtn').textContent = '播放';
}

function startPlayback() {
  if (!state.captured || state.frameCount <= 0) return;
  state.playing = true;
  const playToken = ++state.playToken;
  el('playBtn').textContent = '暂停';
  const tick = () => {
    if (!state.playing || playToken !== state.playToken) return;
    if (state.frameIndex + 1 >= state.frameCount && !el('loopToggle').checked) {
      stopPlayback();
      return;
    }
    moveFrame(1);
    if (!state.playing || playToken !== state.playToken) return;
    const interval = Math.max(100, Number(el('intervalInput').value || 600));
    state.timer = setTimeout(tick, interval);
  };
  const interval = Math.max(100, Number(el('intervalInput').value || 600));
  state.timer = setTimeout(tick, interval);
}

async function onModeOrFilterChange() {
  if (!state.captured) return;
  enforceDynamicInputs();
  stopPlayback();
  if (el('modeSelect').value === 'static') {
    state.dynamicFrames = [];
    state.frameCount = 0;
    el('playBtn').disabled = true;
    el('prevBtn').disabled = true;
    el('nextBtn').disabled = true;
    await loadStatic(true);
  } else {
    state.totalPages = 1;
    updatePager();
    try {
      await prepareDynamicFrames();
    } catch (err) {
      renderCards([]);
      el('positionText').textContent = err.message;
      el('playBtn').disabled = true;
      el('prevBtn').disabled = true;
      el('nextBtn').disabled = true;
    }
  }
}


function renderFrameList(frames) {
  const list = el('frameList');
  list.innerHTML = '';
  if (!frames.length) {
    const empty = document.createElement('div');
    empty.className = 'empty-state';
    empty.textContent = '当前页没有数据帧';
    list.appendChild(empty);
    return;
  }
  frames.forEach((frame) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `frame-card${frame.frame_idx === state.pendingFrameIdx ? ' selected' : ''}`;
    const images = (frame.images || []).map((image) => `<img src="${image.src}" alt="frame ${frame.frame_idx}">`).join('');
    button.innerHTML = `
      <div class="frame-card-header"><span>Frame ${frame.frame_idx}</span><span>${(frame.images || []).length} image</span></div>
      <div class="frame-preview">${images}</div>
    `;
    button.addEventListener('click', () => {
      state.pendingFrameIdx = frame.frame_idx;
      renderFrameList(frames);
    });
    list.appendChild(button);
  });
}

async function loadFramePreviewPage(page = 0) {
  const dataset = el('dataset').value.trim();
  if (!dataset) throw new Error('请先填写数据集路径');
  el('frameModalStatus').textContent = '正在加载数据帧预览...';
  const params = new URLSearchParams({ dataset, page, page_size: FRAME_PICKER_PAGE_SIZE, stride: FRAME_PICKER_STRIDE });
  params.set('checkpoint', el('checkpoint').value.trim());
  const data = await api(`/api/dataset/frames?${params.toString()}`);
  state.framePickerPage = data.page;
  state.framePickerTotalPages = data.total_pages;
  renderFrameList(data.frames || []);
  el('framePageInfo').textContent = `${data.page + 1} / ${data.total_pages}`;
  el('framePagePrev').disabled = data.page <= 0;
  el('framePageNext').disabled = data.page + 1 >= data.total_pages;
  el('frameModalStatus').textContent = `共 ${data.num_frames} 帧`;
}

async function openFrameModal() {
  stopPlayback();
  state.pendingFrameIdx = state.selectedFrameIdx;
  el('frameModal').hidden = false;
  try {
    await loadFramePreviewPage(Math.floor(state.selectedFrameIdx / (FRAME_PICKER_PAGE_SIZE * FRAME_PICKER_STRIDE)));
  } catch (err) {
    el('frameModalStatus').textContent = err.message;
    renderFrameList([]);
  }
}

function closeFrameModal() {
  el('frameModal').hidden = true;
}

async function init() {
  updateSelectedFrameText();
  updateAttentionLegend();
  try {
    const settings = await api('/api/settings');
    applyPathHistory(settings);
  } catch (err) {
    setStatus('设置加载失败', 'error');
    el('summaryText').textContent = err.message;
  }

  el('captureBtn').addEventListener('click', async () => {
    stopPlayback();
    setStatus('采集中', 'busy');
    el('captureBtn').disabled = true;
    el('summaryText').textContent = '正在加载模型、数据集并采集 cache...';
    try {
      const data = await api('/api/capture', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          checkpoint: el('checkpoint').value,
          dataset: el('dataset').value,
          frame_idx: state.selectedFrameIdx,
        }),
      });
      state.captured = true;
      state.layout = data.layout;
      state.sourceImages = data.source_images || [];
      state.imageKeys = data.image_keys || [];
      applyPathHistory({
        checkpoint: data.checkpoint || el('checkpoint').value,
        dataset: data.dataset || el('dataset').value,
        checkpoint_history: data.checkpoint_history || [el('checkpoint').value],
        dataset_history: data.dataset_history || [el('dataset').value],
      });
      fillSelect(el('stepSelect'), data.steps);
      fillSelect(el('layerSelect'), data.layers);
      fillSelect(el('headSelect'), data.heads);
      el('modeSelect').disabled = false;
      el('overlayToggle').disabled = !data.overlay_available;
      el('overlayToggle').checked = false;
      updateAttentionLegend();
      el('summaryText').textContent = `采集完成：${data.summary}${data.overlay_reason ? '；叠加不可用：' + data.overlay_reason : ''}`;
      setStatus('已采集', 'ok');
      await loadStatic(true);
    } catch (err) {
      state.captured = false;
      updateAttentionLegend();
      setStatus('采集失败', 'error');
      el('summaryText').textContent = err.message;
      renderCards([]);
    } finally {
      el('captureBtn').disabled = false;
    }
  });

  ['stepSelect', 'layerSelect', 'headSelect', 'modeSelect', 'overlayToggle'].forEach((id) => {
    el(id).addEventListener('change', () => onModeOrFilterChange().catch((err) => {
      setStatus('渲染失败', 'error');
      el('summaryText').textContent = err.message;
    }));
  });

  el('pagePrev').addEventListener('click', () => { state.page = Math.max(0, state.page - 1); loadStatic(false); });
  el('pageNext').addEventListener('click', () => { state.page = Math.min(state.totalPages - 1, state.page + 1); loadStatic(false); });
  el('prevBtn').addEventListener('click', () => { stopPlayback(); moveFrame(-1); });
  el('nextBtn').addEventListener('click', () => { stopPlayback(); moveFrame(1); });
  el('playBtn').addEventListener('click', () => state.playing ? stopPlayback() : startPlayback());

  el('openFramePickerBtn').addEventListener('click', () => openFrameModal());
  el('closeFrameModalBtn').addEventListener('click', closeFrameModal);
  document.querySelector('[data-close-frame-modal]').addEventListener('click', closeFrameModal);
  el('framePagePrev').addEventListener('click', () => loadFramePreviewPage(state.framePickerPage - 1).catch((err) => { el('frameModalStatus').textContent = err.message; }));
  el('framePageNext').addEventListener('click', () => loadFramePreviewPage(state.framePickerPage + 1).catch((err) => { el('frameModalStatus').textContent = err.message; }));
  el('confirmFrameBtn').addEventListener('click', () => {
    state.selectedFrameIdx = state.pendingFrameIdx;
    updateSelectedFrameText();
    closeFrameModal();
  });
}

init();