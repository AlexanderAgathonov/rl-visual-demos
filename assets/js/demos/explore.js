/* Автор: А. А. Агафонов (КФУ). Лицензия: MIT. Часть материалов к статье об изобразительных средствах обучения основам RL. */
/* ==== js/demos/explore.js (demo-kit/explore.html) ==== */
(function () {
  'use strict';

  const DEMO_KEY = 'explore';
  const ACC = Engine.ACCENT_HEX;
  const MONO = Engine.FONT_MONO;
  const GRID = 420;

  const gameCanvas = document.getElementById('explore-game');
  const g = gameCanvas.getContext('2d');
  const netCanvas = document.getElementById('explore-net');
  const nctx = netCanvas.getContext('2d');
  const statsEl = document.getElementById('explore-stats');
  const speedBtn = document.getElementById('explore-speedBtn');
  const strategyBtn = document.getElementById('explore-strategyBtn');
  const resetBtn = document.getElementById('explore-resetBtn');
  const weightsBtn = document.getElementById('explore-weightsBtn');
  const infoBtn = document.getElementById('explore-infoBtn');
  const modalOverlay = document.getElementById('explore-modalOverlay');

  /* ---------- карта 6×5: старая, «зрелищная» — стена, ловушка,
     7 малых рядом с домом, большое за стеной ---------- */
  const MAP = [
    [0, 5, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 0, 3],
    [0, 0, 0, 0, 0],
    [2, 2, 2, 2, 2],
    [2, 0, -1, 0, 2]
  ];
  const ROWS = 6, COLS = 5, CELL = 62, OX = 55, OY = 10;
  const DOTS_Y = OY + ROWS * CELL + 18;
  const START = { y: 5, x: 2 };
  const BIG_POS = { y: 0, x: 1 };
  const MAX_MOVES = 12;   /* кратчайший путь к большому в обход стены — 8 клеток; 12 ходов оставляют запас на пробы, иначе «дальний» режим учится ненадёжно */
  const SMALL = 1, BIG = 10, TRAP = -5;
  const GAMMA = 0.95, ALPHA = 0.25;
  /* шейпинг к ближайшему малому — мягкий, к дальнему большому —
     заметно сильнее одного шага к мелкому (иначе локальная выгода
     «взять что под ногами» всегда перевешивает, и агент вслепую
     ни разу не проходит весь путь в обход стены — проверено
     прогоном: без этого «дальний» режим 0 раз находил большое) */
  const SHAPE_K = 0.2, SHAPE_CLIP = 0.3;
  const SHAPE_BIG_K = 1.5, SHAPE_BIG_CLIP = 1.5;
  const DIRS = [{ dy: -1, dx: 0 }, { dy: 0, dx: 1 }, { dy: 1, dx: 0 }, { dy: 0, dx: -1 }];
  const ACTION_SHORT = ['↑', '→', '↓', '←'];
  const NORMAL_DELAY = 220, FAST_DELAY = 40;

  /* ---------- два режима: своя мотивация на каждый ---------- */
  const MODES = {
    greedy: {
      label: 'Жадность (рядом)', short: 'Жадность',
      epsStart: 0.25, epsDecay: 0.97, epsEnd: 0.03,
      curiosity: 0, shapeSmall: true, shapeBig: false
    },
    far: {
      label: 'Дальний расчёт (за стеной)', short: 'Далеко',
      epsStart: 0.60, epsDecay: 0.985, epsEnd: 0.02,
      curiosity: 1.0, shapeSmall: false, shapeBig: true
    }
  };

  function clip(v) { return Math.max(-SHAPE_CLIP, Math.min(SHAPE_CLIP, v)); }
  function isWall(y, x) { return y < 0 || y >= ROWS || x < 0 || x >= COLS || MAP[y][x] === 1; }
  function cellKey(y, x) { return y + ',' + x; }
  function stateKey(y, x, k) { return y + ',' + x + '|' + k; }

  /* расстояние до большого в обход стены (BFS по проходимым клеткам) —
     без этого линейная «дистанция по прямой» врёт прямо у стены и
     сбивает агента с толку ровно в критической точке маршрута */
  function bfsDistTo(ty, tx) {
    const dist = Array.from({ length: ROWS }, () => new Array(COLS).fill(-1));
    dist[ty][tx] = 0;
    const q = [[ty, tx]]; let qi = 0;
    while (qi < q.length) {
      const [y, x] = q[qi++];
      for (const d of DIRS) {
        const ny = y + d.dy, nx = x + d.dx;
        if (ny < 0 || ny >= ROWS || nx < 0 || nx >= COLS) continue;
        if (MAP[ny][nx] === 1) continue;
        if (dist[ny][nx] !== -1) continue;
        dist[ny][nx] = dist[y][x] + 1;
        q.push([ny, nx]);
      }
    }
    return dist;
  }
  const distToBig = bfsDistTo(BIG_POS.y, BIG_POS.x);

  let mode = 'greedy';
  const Qmap = { greedy: new Map(), far: new Map() };
  const hist = { greedy: [], far: [] };
  const epCount = { greedy: 0, far: 0 };
  const bigFound = { greedy: 0, far: 0 };
  const lastEpiSample = { greedy: -1, far: -1 };

  function getQ(k) { const m = Qmap[mode]; if (!m.has(k)) m.set(k, [0, 0, 0, 0]); return m.get(k); }
  function currentEpsilon() {
    const m = MODES[mode];
    return Math.max(m.epsEnd, m.epsStart * Math.pow(m.epsDecay, epCount[mode]));
  }

  let agent = { y: START.y, x: START.x };
  let stepIdx = 0, epReward = 0, epBigGot = false;
  let collected = new Set();
  let visitedThisEpisode = new Set();
  let lastAction = 0, lastResult = '', fastMode = false, mainStarted = false;
  let lastUpd = null;

  function validActions() {
    return [0, 1, 2, 3].filter(a => {
      const ny = agent.y + DIRS[a].dy, nx = agent.x + DIRS[a].dx;
      return !isWall(ny, nx);
    });
  }
  function chooseAction() {
    const valid = validActions();
    if (valid.length === 0) return Math.floor(Math.random() * 4);
    if (Math.random() < currentEpsilon()) return valid[Math.floor(Math.random() * valid.length)];
    const q = getQ(stateKey(agent.y, agent.x, collected.size));
    let maxQ = q[0];
    for (let a = 1; a < 4; a++) if (q[a] > maxQ) maxQ = q[a];
    const best = [];
    for (let a = 0; a < 4; a++) if (q[a] === maxQ) best.push(a);
    return best[Math.floor(Math.random() * best.length)];
  }
  function nearestSmall() {
    let best = null, bd = Infinity;
    for (let y = 0; y < ROWS; y++) for (let x = 0; x < COLS; x++) {
      if (MAP[y][x] === 2 && !collected.has(cellKey(y, x))) {
        const d = Math.abs(y - agent.y) + Math.abs(x - agent.x);
        if (d < bd) { bd = d; best = { y, x }; }
      }
    }
    return best;
  }
  function resetEpisode() {
    agent = { y: START.y, x: START.x };
    stepIdx = 0; epReward = 0; epBigGot = false;
    collected = new Set(); visitedThisEpisode = new Set(); lastResult = '';
  }
  function stepGame(a) {
    const oldPos = { y: agent.y, x: agent.x };
    const ny = agent.y + DIRS[a].dy, nx = agent.x + DIRS[a].dx;
    let reward = 0, result = 'шаг', done = false;
    if (isWall(ny, nx)) { result = 'стена'; }
    else {
      agent.y = ny; agent.x = nx;
      const key = cellKey(ny, nx), code = MAP[ny][nx];
      if (code === 2 && !collected.has(key)) { collected.add(key); reward += SMALL; result = 'малое'; }
      else if (code === 5 && !collected.has(key)) { collected.add(key); reward += BIG; result = 'большое'; epBigGot = true; }
      else if (code === 3) { reward += TRAP; result = 'ловушка'; done = true; }
    }
    const m = MODES[mode];
    if (!done) {
      const ck = cellKey(agent.y, agent.x);
      if (!visitedThisEpisode.has(ck)) {
        visitedThisEpisode.add(ck);
        if (m.curiosity > 0) reward += m.curiosity * currentEpsilon();
      }
      if (result === 'шаг') {
        if (m.shapeSmall) {
          const t = nearestSmall();
          if (t) {
            const od = Math.abs(oldPos.y - t.y) + Math.abs(oldPos.x - t.x);
            const nd = Math.abs(agent.y - t.y) + Math.abs(agent.x - t.x);
            reward += clip(SHAPE_K * (od - nd));
          }
        }
        if (m.shapeBig && !collected.has(cellKey(BIG_POS.y, BIG_POS.x))) {
          const od = distToBig[oldPos.y][oldPos.x];
          const nd = distToBig[agent.y][agent.x];
          reward += Math.max(-SHAPE_BIG_CLIP, Math.min(SHAPE_BIG_CLIP, SHAPE_BIG_K * (od - nd)));
        }
      }
    }
    stepIdx++;
    if (stepIdx >= MAX_MOVES) { done = true; if (result === 'шаг') result = 'конец'; }
    return { reward, done, result };
  }
  let endHold = false;
  function stepOnce() {
    if (endHold) { endHold = false; resetEpisode(); return; }
    const oy = agent.y, ox = agent.x, k = collected.size;
    const s = stateKey(oy, ox, k);
    const a = chooseAction(); lastAction = a;
    const res = stepGame(a);
    epReward += res.reward; lastResult = res.result;
    const k2 = collected.size;
    const s2 = res.done ? null : stateKey(agent.y, agent.x, k2);
    const q = getQ(s);
    const boot = res.done ? 0 : Math.max.apply(null, getQ(s2));
    const old = q[a];
    const target = res.reward + GAMMA * boot;
    q[a] = old + ALPHA * (target - old);
    lastUpd = { y: oy, x: ox, k, a, r: res.reward, boot, old, target, neu: q[a] };
    if (res.done) {
      epCount[mode]++;
      hist[mode].push(epReward);
      if (epBigGot) bigFound[mode]++;
      endHold = true;          /* последний ход остаётся на экране: сброс — на следующем тике */
    }
  }

  /* ---------- ЛЕВАЯ ПАНЕЛЬ ---------- */
  function drawGame() {
    g.clearRect(0, 0, GRID, GRID); g.fillStyle = '#fff'; g.fillRect(0, 0, GRID, GRID);
    g.strokeStyle = '#e5e5e5'; g.lineWidth = 1;
    for (let i = 0; i <= COLS; i++) { g.beginPath(); g.moveTo(OX + i * CELL + 0.5, OY); g.lineTo(OX + i * CELL + 0.5, OY + ROWS * CELL); g.stroke(); }
    for (let j = 0; j <= ROWS; j++) { g.beginPath(); g.moveTo(OX, OY + j * CELL + 0.5); g.lineTo(OX + COLS * CELL, OY + j * CELL + 0.5); g.stroke(); }
    const blink = Math.floor(performance.now() / 600) % 2 === 0;
    for (let y = 0; y < ROWS; y++) for (let x = 0; x < COLS; x++) {
      const px = OX + x * CELL, py = OY + y * CELL, code = MAP[y][x], key = cellKey(y, x);
      if (code === 1) { g.fillStyle = '#111'; g.fillRect(px + 1, py + 1, CELL - 2, CELL - 2); continue; }
      if (code === 3) { Engine.drawTrapCell(g, px, py, CELL); continue; }
      if (code === -1) {
        g.strokeStyle = '#000'; g.lineWidth = 2;
        g.beginPath(); g.arc(px + CELL / 2, py + CELL / 2, CELL / 4.2, 0, Math.PI * 2); g.stroke();
      }
      if (!collected.has(key)) {
        if (code === 2) {
          const s = CELL * 0.35;
          if (blink) { g.fillStyle = ACC; g.fillRect(px + CELL / 2 - s / 2, py + CELL / 2 - s / 2, s, s); }
          else { g.strokeStyle = ACC; g.lineWidth = 2.5; g.strokeRect(px + CELL / 2 - s / 2, py + CELL / 2 - s / 2, s, s); }
        } else if (code === 5) {
          const s = CELL * 0.62;
          if (blink) { g.fillStyle = ACC; g.fillRect(px + CELL / 2 - s / 2, py + CELL / 2 - s / 2, s, s); }
          else { g.strokeStyle = ACC; g.lineWidth = 3; g.strokeRect(px + CELL / 2 - s / 2, py + CELL / 2 - s / 2, s, s); }
        }
      }
    }
    const ax = OX + agent.x * CELL + CELL / 2, ay = OY + agent.y * CELL + CELL / 2;
    g.beginPath(); g.arc(ax, ay, CELL / 3.1, 0, Math.PI * 2);
    g.fillStyle = ACC; g.fill(); g.strokeStyle = '#000'; g.lineWidth = 2; g.stroke();
    for (let i = 0; i < MAX_MOVES; i++) {
      g.fillStyle = i < stepIdx ? ACC : '#ddd';
      g.beginPath(); g.arc(OX + 10 + i * ((COLS * CELL - 20) / (MAX_MOVES - 1)), DOTS_Y, 4, 0, Math.PI * 2); g.fill();
    }
  }

  /* ---------- ПРАВАЯ ПАНЕЛЬ ---------- */
  function sliceFor(y, x, k) {
    if ((MAP[y][x] === 2 || MAP[y][x] === 5) && !collected.has(cellKey(y, x))) return k + 1;
    return k;
  }
  function drawRight() {
    nctx.clearRect(0, 0, GRID, GRID);
    const k = collected.size;
    nctx.fillStyle = '#000'; nctx.font = '13px Inter, sans-serif'; nctx.textAlign = 'center';
    nctx.fillText('Режим «' + MODES[mode].short + '» · собрано ' + k + '/7', 210, 16);

    const cell = 50, gx = 16, gy = 26;
    let maxAbs = Math.abs(TRAP);
    for (let y = 0; y < ROWS; y++) for (let x = 0; x < COLS; x++) {
      const code = MAP[y][x];
      if (code === 1 || code === 3) continue;
      const q = Qmap[mode].get(stateKey(y, x, sliceFor(y, x, k)));
      if (q) for (let a = 0; a < 4; a++) { const v = Math.abs(q[a]); if (v > maxAbs) maxAbs = v; }
    }
    for (let y = 0; y < ROWS; y++) for (let x = 0; x < COLS; x++) {
      const px = gx + x * cell, py = gy + y * cell, code = MAP[y][x];
      if (code === 1) {
        nctx.fillStyle = '#111'; nctx.fillRect(px + 1, py + 1, cell - 2, cell - 2);
        nctx.fillStyle = '#fff'; nctx.font = '11px ' + MONO; nctx.textAlign = 'center';
        nctx.fillText('0.00', px + cell / 2, py + cell / 2 + 4);
        continue;
      }
      if (code === 3) {
        nctx.fillStyle = Engine.shadeFor(TRAP, maxAbs); nctx.fillRect(px + 1, py + 1, cell - 2, cell - 2);
        nctx.fillStyle = '#fff'; nctx.font = '11px ' + MONO; nctx.textAlign = 'center';
        nctx.fillText(Engine.fmt(TRAP), px + cell / 2, py + cell / 2 + 4);
        continue;
      }
      const sl = sliceFor(y, x, k);
      const q = Qmap[mode].get(stateKey(y, x, sl));
      nctx.fillStyle = '#fff'; nctx.fillRect(px + 1, py + 1, cell - 2, cell - 2);
      let v = 0, bestA = 0, learned = false;
      if (q && q.some(val => Math.abs(val) > 1e-6)) {
        learned = true; v = q[0]; bestA = 0;
        for (let a = 1; a < 4; a++) if (q[a] > v) { v = q[a]; bestA = a; }
        nctx.fillStyle = Engine.shadeFor(v, maxAbs); nctx.fillRect(px + 1, py + 1, cell - 2, cell - 2);
      }
      const light = learned && Math.abs(v) / maxAbs > 0.55;
      nctx.fillStyle = learned ? (light ? '#fff' : '#000') : '#999';
      nctx.font = '11px ' + MONO; nctx.textAlign = 'center';
      nctx.fillText(Engine.fmt(v), px + cell / 2, py + 20);
      if (learned) Engine.drawArrow(nctx, px + cell / 2, py + 36, bestA, 9, light ? '#fff' : '#000');
      if (agent.y === y && agent.x === x) {
        nctx.strokeStyle = '#000'; nctx.lineWidth = 2.5;
        nctx.strokeRect(px + 1.5, py + 1.5, cell - 3, cell - 3);
      }
    }
    nctx.strokeStyle = '#ccc'; nctx.lineWidth = 1;
    for (let i = 0; i <= COLS; i++) { nctx.beginPath(); nctx.moveTo(gx + i * cell + 0.5, gy); nctx.lineTo(gx + i * cell + 0.5, gy + ROWS * cell); nctx.stroke(); }
    for (let j = 0; j <= ROWS; j++) { nctx.beginPath(); nctx.moveTo(gx, gy + j * cell + 0.5); nctx.lineTo(gx + COLS * cell, gy + j * cell + 0.5); nctx.stroke(); }

    /* бары переоценки четырёх действий текущего состояния */
    const qCur = Qmap[mode].get(stateKey(agent.y, agent.x, k)) || [0, 0, 0, 0];
    let barMax = 0.01;
    for (let a = 0; a < 4; a++) if (Math.abs(qCur[a]) > barMax) barMax = Math.abs(qCur[a]);
    const bx = 280, bw = 124;
    nctx.fillStyle = '#000'; nctx.font = '11px Inter, sans-serif'; nctx.textAlign = 'left';
    nctx.fillText('Q(s,a) текущего s', bx, 34);
    for (let a = 0; a < 4; a++) {
      const ly = 46 + a * 40;
      const v = qCur[a];
      nctx.fillStyle = '#000'; nctx.font = '12px ' + MONO; nctx.textAlign = 'left';
      nctx.fillText(ACTION_SHORT[a] + ' ' + Engine.fmt(v), bx, ly);
      const by = ly + 5;
      nctx.fillStyle = '#f0f2f5'; nctx.fillRect(bx, by, bw, 14);
      if (Math.abs(v) > 1e-6) {
        nctx.fillStyle = v >= 0 ? 'rgba(59,111,224,0.85)' : 'rgba(70,70,75,0.85)';
        nctx.fillRect(bx, by, Math.max(2, Math.abs(v) / barMax * bw), 14);
      }
      if (a === lastAction) { nctx.strokeStyle = '#000'; nctx.lineWidth = 2; nctx.strokeRect(bx - 1, by - 1, bw + 2, 16); }
      else { nctx.strokeStyle = '#ddd'; nctx.lineWidth = 1; nctx.strokeRect(bx + 0.5, by + 0.5, bw - 1, 13); }
    }

    /* TD-разбор с числами */
    nctx.font = '11px ' + MONO; nctx.textAlign = 'left';
    let ty = gy + ROWS * cell + 18;
    if (lastUpd) {
      const t = lastUpd;
      nctx.fillStyle = '#000';
      nctx.fillText('s=(' + t.y + ',' + t.x + ') k=' + t.k + ' a=' + ACTION_SHORT[t.a] + '  r=' + Engine.fmt(t.r), 16, ty); ty += 15;
      nctx.fillText('maxQ(s′)=' + Engine.fmt(t.boot) + '  цель=r+γ·maxQ′=' + Engine.fmt(t.target), 16, ty); ty += 15;
      nctx.fillText('Q старое=' + Engine.fmt(t.old), 16, ty); ty += 15;
      nctx.fillStyle = ACC;
      nctx.fillText('Q новое=' + Engine.fmt(t.old) + '+' + ALPHA + '·(' + Engine.fmt(t.target) + '−' + Engine.fmt(t.old) + ')=' + Engine.fmt(t.neu), 16, ty); ty += 15;
    }
    nctx.fillStyle = '#666'; nctx.font = '11px Inter, sans-serif'; nctx.textAlign = 'left';
    nctx.fillText('ε ' + currentEpsilon().toFixed(3) + ' (' + MODES[mode].epsStart.toFixed(2) + '→' + MODES[mode].epsEnd.toFixed(2) + ')', 16, 412);
  }

  function drawStats() {
    statsEl.innerHTML =
      '<span>Режим: <span class="v">' + MODES[mode].short + '</span></span>' +
      '<span>Эпизод: <span class="v">' + Engine.fmtFixed(epCount[mode] + 1, 5, 0) + '</span></span>' +
      '<span>Ход: <span class="v">' + Engine.fmtFixed(stepIdx, 1, 0) + '/' + MAX_MOVES + '</span></span>' +
      '<span>Награда: <span class="v">' + Engine.fmtFixed(epReward, 2, 1) + '</span></span>' +
      '<span>ε: <span class="v">' + Engine.fmtFixed(currentEpsilon(), 1, 3) + '</span></span>' +
      '<span>Большое: <span class="v">' + Engine.fmtFixed(bigFound[mode], 3, 0) + '</span></span>';
  }

  function mainLoop(ts) {
    if (mainStarted) {
      const delay = endHold ? (fastMode ? 90 : 900) : (fastMode ? FAST_DELAY : NORMAL_DELAY);
      if (ts - lastTs >= delay) { stepOnce(); lastTs = ts; }
      drawGame(); drawRight(); drawStats();
    }
    parker.raf(mainLoop);
  }
  let lastTs = 0;
  function introFrame(ts) {
    if (introFinished) return;
    if (introStart === null) introStart = ts;
    const t = (ts - introStart) / 1000;
    drawIntro(t);
    if (t >= 24) { finishIntro(); return; }
    parker.raf(introFrame);
  }
  const parker = Engine.makeRafParker({
    getMain: function () { return mainLoop; },
    getIntro: function () { return introFrame; }
  });

  /* ---------- кривая обучения: своя история на каждый режим ---------- */
  const curve = Engine.makeCurve(document.getElementById('explore-curve'), {
    primaryLabel: 'средняя награда', secondaryLabel: 'награда эп.'
  });
  Engine.resizeCanvas(document.getElementById('explore-curve'), {});
  function redrawCurveForMode() {
    curve.reset();
    const h = hist[mode];
    let sum = 0;
    for (let i = 0; i < h.length; i++) { sum += h[i]; curve.sample(i + 1, Math.round((sum / (i + 1)) * 100) / 100, h[i]); }
    lastEpiSample[mode] = epCount[mode];
  }
  setInterval(() => {
    if (!parker.isRunning()) return;
    if (epCount[mode] === lastEpiSample[mode]) return;
    lastEpiSample[mode] = epCount[mode];
    const h = hist[mode];
    const sum = h.reduce((a, b) => a + b, 0);
    curve.sample(epCount[mode], Math.round((sum / h.length) * 100) / 100, h[h.length - 1]);
  }, 250);

  strategyBtn.addEventListener('click', () => {
    mode = mode === 'greedy' ? 'far' : 'greedy';
    endHold = false; resetEpisode();
    strategyBtn.textContent = 'Режим: ' + MODES[mode].label;
    redrawCurveForMode();
    drawGame(); drawRight(); drawStats();
  });
  speedBtn.addEventListener('click', () => {
    fastMode = !fastMode;
    speedBtn.textContent = fastMode ? 'Обучение: ускоренное' : 'Обучение: обычное'; speedBtn.classList.toggle('eng-btn-accent', fastMode);
  });
  resetBtn.addEventListener('click', () => {
    Qmap.greedy = new Map(); Qmap.far = new Map();
    hist.greedy = []; hist.far = [];
    epCount.greedy = 0; epCount.far = 0;
    bigFound.greedy = 0; bigFound.far = 0;
    lastEpiSample.greedy = -1; lastEpiSample.far = -1;
    endHold = false; resetEpisode(); lastUpd = null;
    curve.reset();
    drawGame(); drawRight(); drawStats();
  });
  infoBtn.addEventListener('click', () => { modalOverlay.classList.add('is-open'); });
  document.getElementById('explore-modalClose').addEventListener('click', () => { modalOverlay.classList.remove('is-open'); });
  document.getElementById('explore-modalOk').addEventListener('click', () => { modalOverlay.classList.remove('is-open'); });
  modalOverlay.addEventListener('click', e => { if (e.target === modalOverlay) modalOverlay.classList.remove('is-open'); });
  weightsBtn.addEventListener('click', () => {
    Engine.openWeightsModal({
      key: DEMO_KEY,
      exportWeights: exportWeights,
      importWeights: importWeights,
      presetWeights: (window.DemoWeights && window.DemoWeights[DEMO_KEY]) || null
    });
  });

  /* ---------- интро ---------- */
  const introOverlay = document.getElementById('explore-introOverlay');
  const ictx = document.getElementById('explore-introCanvas').getContext('2d');
  let introFinished = false, introStart = null;
  function finishIntro() { if (introFinished) return; introFinished = true; introOverlay.style.display = 'none'; mainStarted = true; }
  introOverlay.addEventListener('click', finishIntro);
  const drawSkull = Engine.drawSkullAndBones;
  const CAPS = [
    [0, 2.5, 'Поле 6×5: дом, семь маленьких сокровищ +1 и одно большое +10 за стеной.'],
    [2.5, 5, 'Эпизод = ровно 12 ходов, потом сброс. Рядом с проходом — ловушка −5.'],
    [5, 9, 'Режим «Жадность»: агент учится обходить все 7 малых — суммарно +7 за эпизод.'],
    [9, 13, 'Режим «Дальний расчёт»: агент учится обходить стену — и ловушку — за большим +10.'],
    [13, 18, 'У каждого режима своя таблица Q — сравнивайте кривые обучения.'],
    [18, 24, 'Q(s,a) ← Q(s,a) + α·[ r + γ·maxQ(s′) − Q(s,a) ]']
  ];
  function drawIntroMap(c, alpha) {
    if (alpha <= 0) return;
    c.save(); c.globalAlpha = alpha;
    const cs = 52, x0 = 30, y0 = 60;
    for (let y = 0; y < ROWS; y++) for (let x = 0; x < COLS; x++) {
      const px = x0 + x * cs, py = y0 + y * cs, code = MAP[y][x];
      c.strokeStyle = '#ddd'; c.strokeRect(px + 0.5, py + 0.5, cs, cs);
      if (code === 1) { c.fillStyle = '#111'; c.fillRect(px + 1, py + 1, cs - 2, cs - 2); }
      if (code === 3) {
        c.fillStyle = '#fff'; c.fillRect(px + 2, py + 2, cs - 4, cs - 4);
        c.strokeStyle = '#000'; c.lineWidth = 1; c.strokeRect(px + 2.5, py + 2.5, cs - 5, cs - 5);
        drawSkull(c, px + cs / 2, py + cs / 2, cs * 0.72);
      }
      if (code === 2) { const s = cs * 0.35; c.fillStyle = ACC; c.fillRect(px + cs / 2 - s / 2, py + cs / 2 - s / 2, s, s); }
      if (code === 5) { const s = cs * 0.62; c.fillStyle = ACC; c.fillRect(px + cs / 2 - s / 2, py + cs / 2 - s / 2, s, s); }
    }
    c.restore();
  }
  function drawIntro(t) {
    const c = ictx; c.clearRect(0, 0, 720, 480); c.fillStyle = '#fff'; c.fillRect(0, 0, 720, 480);
    const cap = CAPS.find(x => t >= x[0] && t < x[1]);
    if (cap) { c.fillStyle = '#000'; c.font = '14px Inter, sans-serif'; c.textAlign = 'left'; c.fillText(cap[2], 14, 24); }
    drawIntroMap(c, Math.min(1, t / 1.2));
    if (t > 18 && t < 24) {
      c.fillStyle = '#000'; c.font = '15px ' + MONO; c.textAlign = 'left';
      c.fillText('Q ← Q + α·[ r + γ·maxQ′ − Q ]', 420, 130);
    }
    c.fillStyle = '#999'; c.font = '11px Inter, sans-serif'; c.textAlign = 'right'; c.fillText('клик — пропустить', 712, 474);
    const fo = Math.max(0, Math.min(1, (t - 22.5) / 1.5));
    if (fo > 0) { c.fillStyle = 'rgba(255,255,255,' + fo + ')'; c.fillRect(0, 0, 720, 480); }
  }

  /* ---------- адаптивные канвасы ---------- */
  let gameScale = 1, netScale = 1, gameOX = 0, gameOY = 0, netOX = 0, netOY = 0;
  Engine.resizeCanvas(gameCanvas, { onResize: (w, h) => { gameScale = Math.min(w, h) / GRID; gameOX = (w - GRID * gameScale) / 2; gameOY = (h - GRID * gameScale) / 2; } });
  Engine.resizeCanvas(netCanvas, { onResize: (w, h) => { netScale = Math.min(w, h) / GRID; netOX = (w - GRID * netScale) / 2; netOY = (h - GRID * netScale) / 2; } });
  const rawDrawGame = drawGame, rawDrawRight = drawRight;
  drawGame = function () { g.save(); g.translate(gameOX, gameOY); g.scale(gameScale, gameScale); g.beginPath(); g.rect(0, 0, GRID, GRID); g.clip(); rawDrawGame(); g.lineWidth = 3 / gameScale; g.strokeStyle = '#1a1c20'; g.strokeRect(0, 0, GRID, GRID); g.restore(); };
  drawRight = function () { nctx.save(); nctx.translate(netOX, netOY); nctx.scale(netScale, netScale); rawDrawRight(); nctx.restore(); };

  /* ---------- веса ---------- */
  function exportWeights() {
    return { greedy: Array.from(Qmap.greedy.entries()), far: Array.from(Qmap.far.entries()) };
  }
  function importWeights(W) {
    try {
      if (!W) return false;
      let ok = false;
      if (Array.isArray(W.greedy)) { Qmap.greedy = new Map(W.greedy); ok = true; }
      if (Array.isArray(W.far)) { Qmap.far = new Map(W.far); ok = true; }
      return ok;
    } catch (e) { return false; }
  }

  /* ---------- регистрация ---------- */
  resetEpisode();
  drawGame(); drawRight(); drawStats();
  parker.raf(mainLoop);
  parker.raf(introFrame);

  Engine.register(DEMO_KEY, {
    key: DEMO_KEY,
    title: 'Жадность или дальний расчёт?',
    root: document.getElementById('demo-' + DEMO_KEY),
    resume: parker.resume,
    pause: parker.pause,
    kick: function () { parker.kick(introFinished); },
    startMain: function () {
      parker.startMain(function () {
        introFinished = true;
        introOverlay.style.display = 'none';
        mainStarted = true;
      });
    },
    exportWeights: function () { try { return JSON.parse(JSON.stringify(exportWeights())); } catch (e) { return null; } },
    importWeights: function (W) { return importWeights(W); },
    isFast: function () { return fastMode; },
    setFast: function (on) { if (fastMode !== !!on) speedBtn.click(); },
    getStrategy: function () { return mode; },
    setStrategy: function (m) { if (MODES[m] && m !== mode) strategyBtn.click(); },
    skipIntro: function () { finishIntro(); },
    showIntro: function () {
      introFinished = false; introStart = null;
      introOverlay.style.display = '';
      parker.showOnly(introFrame);
    },
    reset: function () { resetBtn.click(); }
  });

  parker.resume();
})();
