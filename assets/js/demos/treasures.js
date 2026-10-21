/* Автор: А. А. Агафонов (КФУ). Лицензия: MIT. Часть материалов к статье об изобразительных средствах обучения основам RL. */
/* ==== js/demos/treasures.js (demo-kit/treasures.html) ==== */
(function () {
  'use strict';

  const DEMO_KEY = 'treasures';
  const ACC = Engine.ACCENT_HEX;
  const MONO = Engine.FONT_MONO;
  const GRID = 420;
  const METRIC_LABEL = 'возвратов';

  const gameCanvas = document.getElementById('treasures-game');
  const g = gameCanvas.getContext('2d');
  const netCanvas = document.getElementById('treasures-net');
  const nctx = netCanvas.getContext('2d');
  const statsEl = document.getElementById('treasures-stats');
  const speedBtn = document.getElementById('treasures-speedBtn');
  const resetBtn = document.getElementById('treasures-resetBtn');
  const weightsBtn = document.getElementById('treasures-weightsBtn');
  const infoBtn = document.getElementById('treasures-infoBtn');
  const modalOverlay = document.getElementById('treasures-modalOverlay');

  const drawSkull = Engine.drawSkullAndBones;
  const drawTrapCell = Engine.drawTrapCell;
  const drawDynamicWall = Engine.drawDynamicWall;
  const drawArrow = Engine.drawArrow;
  const shadeFor = Engine.shadeFor;
  const fmt = Engine.fmt;

  /* ---------- карта и параметры среды (без изменений по сути) ---------- */
  const N = 8, CELL = GRID / N;
  const START = { y: 0, x: 0 };
  const TREASURE_POS = [{ y: 1, x: 5 }, { y: 3, x: 7 }, { y: 6, x: 4 }];
  const TRAP_POS = [{ y: 2, x: 6 }, { y: 4, x: 0 }, { y: 7, x: 5 }];
  const BASE_WORLD = [
    [0, 0, 0, 1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0]
  ];
  const ACTIONS = [0, 1, 2, 3];
  const DIRS = [{ dy: -1, dx: 0 }, { dy: 0, dx: 1 }, { dy: 1, dx: 0 }, { dy: 0, dx: -1 }];
  const ACTION_SHORT = ['↑', '→', '↓', '←'];

  const STEP_PENALTY = -0.1, WALL_PENALTY = -1.0, TRAP_PENALTY = -5.0;
  const TREASURE_REWARD = 10.0, HOME_BONUS = 50.0;
  const MAX_STEPS = 80;
  const ALPHA = 0.20, GAMMA_Q = 0.95;
  const EPS_START = 1.0, EPS_END = 0.05, EPS_DECAY = 0.995;
  const SHAPE_K = 0.20, SHAPE_CLIP = 0.30;
  const NORMAL_STEP_DELAY = 90;
  const FAST_TIME_BUDGET_MS = 6, FAST_MAX_STEPS_PER_FRAME = 1000;
  const DYNAMIC_WALL_PROB = 0.15, DYNAMIC_CANDIDATE_PROB = 0.15;
  const SAFE_TRAPS = false;

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function manhattan(a, b) { return Math.abs(a.y - b.y) + Math.abs(a.x - b.x); }
  function inside(y, x) { return y >= 0 && y < N && x >= 0 && x < N; }
  function cellKey(y, x) { return y + ',' + x; }
  function isTreasurePos(y, x) { return TREASURE_POS.some(t => t.y === y && t.x === x); }
  function treasureIndex(y, x) { return TREASURE_POS.findIndex(t => t.y === y && t.x === x); }
  function isTrap(y, x) { return TRAP_POS.some(t => t.y === y && t.x === x); }
  function isSpecialCell(y, x) {
    if (START.y === y && START.x === x) return true;
    if (isTreasurePos(y, x)) return true;
    if (isTrap(y, x)) return true;
    return false;
  }

  let Q = new Map();
  let agent = { y: START.y, x: START.x };
  let treasures = [], treasuresLeft = 3;
  let dynamicCandidates = [], currentDynamicWalls = new Set();
  let episodeSteps = 0, episodeReward = 0, episodeCount = 0, lastEpisodeReward = 0;
  let score = 0;
  let lastAction = 0, lastNotableEvent = '', lastUpd = null;
  let fastMode = false, mainStarted = false;

  function collectedMask() {
    let mask = 0;
    treasures.forEach((t, i) => { if (t.collected) mask |= 1 << i; });
    return mask;
  }
  function stateSuffix() { return String(collectedMask()); }
  function stateKey() { return agent.y + ',' + agent.x + '|' + stateSuffix(); }
  function phaseLabelText() {
    return treasuresLeft > 0
      ? 'осталось ' + treasuresLeft
      : 'идём домой';
  }

  function makeDynamicCandidates() {
    dynamicCandidates = [];
    if (DYNAMIC_WALL_PROB <= 0) return;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      if (BASE_WORLD[y][x] === 1) continue;
      if (isSpecialCell(y, x)) continue;
      if (Math.random() < DYNAMIC_CANDIDATE_PROB) dynamicCandidates.push({ y, x });
    }
  }
  function resetEpisode() {
    agent = { y: START.y, x: START.x };
    treasures = TREASURE_POS.map(t => ({ y: t.y, x: t.x, collected: false }));
    treasuresLeft = 3;
    episodeSteps = 0; episodeReward = 0;
    currentDynamicWalls = new Set();
    makeDynamicCandidates();
  }
  function isWall(y, x) {
    if (!inside(y, x)) return true;
    if (BASE_WORLD[y][x] === 1) return true;
    if (currentDynamicWalls.has(cellKey(y, x))) return true;
    return false;
  }
  function updateDynamicWalls() {
    if (DYNAMIC_WALL_PROB <= 0) return;
    for (const p of dynamicCandidates) {
      if (Math.random() < DYNAMIC_WALL_PROB) {
        const key = cellKey(p.y, p.x);
        if (currentDynamicWalls.has(key)) currentDynamicWalls.delete(key);
        else {
          if (agent.y === p.y && agent.x === p.x) continue;
          if (isTrap(p.y, p.x) || isTreasurePos(p.y, p.x)) continue;
          currentDynamicWalls.add(key);
        }
      }
    }
  }
  function currentTarget() {
    const remaining = treasures.filter(t => !t.collected);
    if (remaining.length > 0) {
      let best = remaining[0], bestD = manhattan(agent, best);
      for (const t of remaining) { const d = manhattan(agent, t); if (d < bestD) { best = t; bestD = d; } }
      return best;
    }
    return START;
  }

  function stepGame(action) {
    episodeSteps++;
    const oldPos = { y: agent.y, x: agent.x };
    const targetBefore = currentTarget();
    const oldDist = targetBefore ? manhattan(oldPos, targetBefore) : 0;
    let reward = STEP_PENALTY, result = 'шаг';
    const ny = agent.y + DIRS[action].dy, nx = agent.x + DIRS[action].dx;
    if (!inside(ny, nx) || isWall(ny, nx)) { reward = WALL_PENALTY; result = 'стена'; }
    else {
      agent.y = ny; agent.x = nx;
      if (isTrap(ny, nx)) return { reward: TRAP_PENALTY, done: true, result: 'ловушка' };
      const treasure = treasures.find(t => !t.collected && t.y === ny && t.x === nx);
      if (treasure) { treasure.collected = true; treasuresLeft--; reward = TREASURE_REWARD; result = 'сокровище'; }
    }
    updateDynamicWalls();
    if (treasuresLeft === 0 && agent.y === START.y && agent.x === START.x) {
      return { reward: HOME_BONUS, done: true, result: 'дом' };
    }
    if (result === 'шаг' && targetBefore) {
      const newDist = manhattan(agent, targetBefore);
      reward += clamp(SHAPE_K * (oldDist - newDist), -SHAPE_CLIP, SHAPE_CLIP);
    }
    if (episodeSteps >= MAX_STEPS) return { reward, done: true, result: 'время' };
    return { reward, done: false, result };
  }

  function getQ(key) { if (!Q.has(key)) Q.set(key, [0, 0, 0, 0]); return Q.get(key); }
  function currentEpsilon() { return Math.max(EPS_END, EPS_START * Math.pow(EPS_DECAY, episodeCount)); }
  function validActions() {
    return ACTIONS.filter(a => {
      const ny = agent.y + DIRS[a].dy, nx = agent.x + DIRS[a].dx;
      if (!inside(ny, nx) || isWall(ny, nx)) return false;
      if (SAFE_TRAPS && isTrap(ny, nx)) return false;
      return true;
    });
  }
  function chooseAction() {
    const valid = validActions();
    if (valid.length === 0) return Math.floor(Math.random() * ACTIONS.length);
    if (Math.random() < currentEpsilon()) return valid[Math.floor(Math.random() * valid.length)];
    const q = getQ(stateKey());
    let bestValue = -Infinity;
    for (const a of valid) if (q[a] > bestValue) bestValue = q[a];
    const bestActions = valid.filter(a => q[a] === bestValue);
    return bestActions[Math.floor(Math.random() * bestActions.length)];
  }
  function stepOnce() {
    const oy = agent.y, ox = agent.x, m = collectedMask();
    const s = stateKey();
    const a = chooseAction(); lastAction = a;
    const res = stepGame(a);
    episodeReward += res.reward;
    if (res.result !== 'шаг') lastNotableEvent = res.result;
    const s2 = res.done ? null : stateKey();
    const q = getQ(s);
    const boot = res.done ? 0 : Math.max.apply(null, getQ(s2));
    const old = q[a];
    const target = res.reward + GAMMA_Q * boot;
    q[a] = old + ALPHA * (target - old);
    lastUpd = { y: oy, x: ox, m, a, r: res.reward, boot, old, target, neu: q[a] };
    if (res.done) {
      episodeCount++;
      lastEpisodeReward = episodeReward;
      if (res.result === 'дом') score++;
      resetEpisode();
    }
  }

  /* ---------- ЛЕВАЯ ПАНЕЛЬ ---------- */
  function drawGame() {
    g.clearRect(0, 0, GRID, GRID); g.fillStyle = '#fff'; g.fillRect(0, 0, GRID, GRID);
    g.strokeStyle = '#e5e5e5'; g.lineWidth = 1;
    for (let i = 1; i < N; i++) {
      g.beginPath(); g.moveTo(i * CELL + 0.5, 0); g.lineTo(i * CELL + 0.5, GRID); g.stroke();
      g.beginPath(); g.moveTo(0, i * CELL + 0.5); g.lineTo(GRID, i * CELL + 0.5); g.stroke();
    }
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const px = x * CELL, py = y * CELL;
      if (BASE_WORLD[y][x] === 1) { g.fillStyle = '#111'; g.fillRect(px + 1, py + 1, CELL - 2, CELL - 2); continue; }
      if (currentDynamicWalls.has(cellKey(y, x))) drawDynamicWall(g, px, py, CELL);
    }
    for (const t of TRAP_POS) drawTrapCell(g, t.x * CELL, t.y * CELL, CELL);

    const sx = START.x * CELL + CELL / 2, sy = START.y * CELL + CELL / 2;
    if (treasuresLeft === 0) {
      g.fillStyle = 'rgba(59,111,224,0.18)';
      g.beginPath(); g.arc(sx, sy, CELL / 3.2, 0, Math.PI * 2); g.fill();
      g.strokeStyle = ACC; g.lineWidth = 3;
      g.beginPath(); g.arc(sx, sy, CELL / 3.2, 0, Math.PI * 2); g.stroke();
    } else {
      g.strokeStyle = '#000'; g.lineWidth = 2;
      g.beginPath(); g.arc(sx, sy, CELL / 4.2, 0, Math.PI * 2); g.stroke();
    }

    const target = currentTarget();
    if (target && !(target.y === agent.y && target.x === agent.x)) {
      const tx = target.x * CELL + CELL / 2, ty = target.y * CELL + CELL / 2;
      const ax = agent.x * CELL + CELL / 2, ay = agent.y * CELL + CELL / 2;
      g.strokeStyle = 'rgba(59,111,224,0.35)'; g.lineWidth = 2; g.setLineDash([6, 5]);
      g.beginPath(); g.moveTo(ax, ay); g.lineTo(tx, ty); g.stroke();
      g.setLineDash([]);
    }

    const blink = Math.floor(performance.now() / 600) % 2 === 0;
    for (const t of treasures) {
      if (t.collected) continue;
      const px = t.x * CELL, py = t.y * CELL, s = CELL * 0.62;
      if (blink) { g.fillStyle = ACC; g.fillRect(px + CELL / 2 - s / 2, py + CELL / 2 - s / 2, s, s); }
      else { g.strokeStyle = ACC; g.lineWidth = 3; g.strokeRect(px + CELL / 2 - s / 2, py + CELL / 2 - s / 2, s, s); }
    }

    const ax = agent.x * CELL + CELL / 2, ay = agent.y * CELL + CELL / 2;
    g.beginPath(); g.arc(ax, ay, CELL / 3.1, 0, Math.PI * 2);
    g.fillStyle = ACC; g.fill(); g.strokeStyle = '#000'; g.lineWidth = 2; g.stroke();
  }

  /* ---------- ПРАВАЯ ПАНЕЛЬ ---------- */
  function sliceFor(y, x, mask) {
    const ti = treasureIndex(y, x);
    if (ti >= 0) return String(mask | (1 << ti));
    return String(mask);
  }
  function drawPolicy() {
    nctx.clearRect(0, 0, GRID, GRID);
    nctx.fillStyle = '#000'; nctx.font = '13px Inter, sans-serif'; nctx.textAlign = 'center';
    nctx.fillText('Политика и Q-значения', GRID / 2, 18);
    nctx.fillStyle = '#666'; nctx.font = '11px Inter, sans-serif';
    nctx.fillText(phaseLabelText() + ' · маска ' + stateSuffix(), GRID / 2, 34);

    const cell = 28, mapX = 16, mapY = 46, mapSize = N * cell;
    const mask = collectedMask();
    let maxAbs = Math.abs(TRAP_PENALTY);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      if (BASE_WORLD[y][x] === 1 || isTrap(y, x)) continue;
      const q = Q.get(y + ',' + x + '|' + sliceFor(y, x, mask));
      if (q) for (let a = 0; a < 4; a++) { const v = Math.abs(q[a]); if (v > maxAbs) maxAbs = v; }
    }
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const px = mapX + x * cell, py = mapY + y * cell;
      nctx.fillStyle = '#fff'; nctx.fillRect(px, py, cell, cell);

      if (BASE_WORLD[y][x] === 1) {
        nctx.fillStyle = '#111'; nctx.fillRect(px + 1, py + 1, cell - 2, cell - 2);
        nctx.fillStyle = '#fff'; nctx.font = '8px ' + MONO; nctx.textAlign = 'center';
        nctx.fillText('0', px + cell / 2, py + cell / 2 + 3);
        continue;
      }
      if (currentDynamicWalls.has(cellKey(y, x))) { drawDynamicWall(nctx, px, py, cell); continue; }
      if (isTrap(y, x)) {
        nctx.fillStyle = shadeFor(TRAP_PENALTY, maxAbs); nctx.fillRect(px + 1, py + 1, cell - 2, cell - 2);
        nctx.fillStyle = '#fff'; nctx.font = '8px ' + MONO; nctx.textAlign = 'center';
        nctx.fillText('−5', px + cell / 2, py + cell / 2 + 3);
        if (agent.y === y && agent.x === x) { nctx.strokeStyle = '#000'; nctx.lineWidth = 2.5; nctx.strokeRect(px + 1.5, py + 1.5, cell - 3, cell - 3); }
        continue;
      }
      const isHome = (y === START.y && x === START.x);
      if (isHome && mask === 7) {
        nctx.fillStyle = shadeFor(HOME_BONUS, maxAbs); nctx.fillRect(px + 1, py + 1, cell - 2, cell - 2);
        nctx.fillStyle = '#fff'; nctx.font = '8px ' + MONO; nctx.textAlign = 'center';
        nctx.fillText('50', px + cell / 2, py + cell / 2 + 3);
        if (agent.y === y && agent.x === x) { nctx.strokeStyle = '#000'; nctx.lineWidth = 2.5; nctx.strokeRect(px + 1.5, py + 1.5, cell - 3, cell - 3); }
        continue;
      }
      const sl = sliceFor(y, x, mask);
      const q = Q.get(y + ',' + x + '|' + sl);
      const ti = treasureIndex(y, x);
      const isTreasureCell = ti >= 0 && !treasures[ti].collected;
      let learned = false, v = 0, bestA = 0;
      if (q && q.some(val => Math.abs(val) > 1e-6)) {
        learned = true; v = q[0];
        for (let a = 1; a < 4; a++) if (q[a] > v) { v = q[a]; bestA = a; }
        nctx.fillStyle = shadeFor(v, maxAbs); nctx.fillRect(px + 1, py + 1, cell - 2, cell - 2);
        const light = Math.abs(v) / maxAbs > 0.55;
        if (isTreasureCell) {
          nctx.fillStyle = light ? '#fff' : '#000'; nctx.font = '8px ' + MONO; nctx.textAlign = 'center';
          nctx.fillText(v.toFixed(1), px + cell / 2, py + 9);
          drawArrow(nctx, px + cell / 2, py + cell / 2 + 4, bestA, cell * 0.28, light ? '#fff' : '#000');
        } else {
          drawArrow(nctx, px + cell / 2, py + cell / 2, bestA, cell * 0.32, light ? '#fff' : '#000');
        }
      } else if (isTreasureCell) {
        nctx.fillStyle = '#999'; nctx.font = '8px ' + MONO; nctx.textAlign = 'center';
        nctx.fillText('+10', px + cell / 2, py + cell / 2 + 3);
      }
      if (agent.y === y && agent.x === x) { nctx.strokeStyle = '#000'; nctx.lineWidth = 2.5; nctx.strokeRect(px + 1.5, py + 1.5, cell - 3, cell - 3); }
    }
    nctx.strokeStyle = '#ddd'; nctx.lineWidth = 1;
    for (let i = 0; i <= N; i++) {
      nctx.beginPath(); nctx.moveTo(mapX + i * cell + 0.5, mapY); nctx.lineTo(mapX + i * cell + 0.5, mapY + mapSize); nctx.stroke();
      nctx.beginPath(); nctx.moveTo(mapX, mapY + i * cell + 0.5); nctx.lineTo(mapX + mapSize, mapY + i * cell + 0.5); nctx.stroke();
    }

    /* бары четырёх действий текущего состояния */
    const qCur = Q.get(stateKey()) || [0, 0, 0, 0];
    let barMax = 0.01;
    for (let a = 0; a < 4; a++) if (Math.abs(qCur[a]) > barMax) barMax = Math.abs(qCur[a]);
    const barX = mapX + mapSize + 26, barW = 118, barH = 16;
    nctx.fillStyle = '#000'; nctx.font = '11px Inter, sans-serif'; nctx.textAlign = 'left';
    nctx.fillText('Q(s,a) текущего s', barX, mapY + 6);
    for (let a = 0; a < 4; a++) {
      const labelY = mapY + 30 + a * 46;
      const val = qCur[a];
      nctx.fillStyle = '#000'; nctx.font = '12px ' + MONO; nctx.textAlign = 'left';
      nctx.fillText(ACTION_SHORT[a] + ' ' + fmt(val), barX, labelY);
      const by = labelY + 6;
      nctx.fillStyle = '#f0f2f5'; nctx.fillRect(barX, by, barW, barH);
      if (Math.abs(val) > 1e-6) {
        nctx.fillStyle = val >= 0 ? 'rgba(59,111,224,0.85)' : 'rgba(70,70,75,0.85)';
        nctx.fillRect(barX, by, Math.max(2, Math.abs(val) / barMax * barW), barH);
      }
      if (a === lastAction) { nctx.strokeStyle = '#000'; nctx.lineWidth = 2; nctx.strokeRect(barX - 1, by - 1, barW + 2, barH + 2); }
      else { nctx.strokeStyle = '#ddd'; nctx.lineWidth = 1; nctx.strokeRect(barX + 0.5, by + 0.5, barW - 1, barH - 1); }
    }

    /* TD-разбор — под всей панелью, во всю ширину канваса
       (в оригинале был прижат к узкой полосе у баров —x там
       случайно схлопывался в barX, и длинные строки обрезались) */
    let ly = Math.max(mapY + mapSize, mapY + 30 + 4 * 46) + 18;
    nctx.font = '10px ' + MONO; nctx.textAlign = 'left';
    if (lastUpd) {
      const t = lastUpd;
      nctx.fillStyle = '#000';
      nctx.fillText('s=(' + t.y + ',' + t.x + ') m=' + t.m + ' a=' + ACTION_SHORT[t.a] + ' r=' + fmt(t.r), mapX, ly); ly += 14;
      nctx.fillText('maxQ(s′)=' + fmt(t.boot) + '  цель=' + fmt(t.target) + '  Q старое=' + fmt(t.old), mapX, ly); ly += 14;
      nctx.fillStyle = ACC;
      nctx.fillText('Q новое=' + fmt(t.old) + '+' + ALPHA + '·(' + fmt(t.target) + '−' + fmt(t.old) + ')=' + fmt(t.neu), mapX, ly);
    }
    nctx.fillStyle = '#999'; nctx.font = '10px Inter, sans-serif'; nctx.textAlign = 'center';
    nctx.fillText('синий = выгодно · серый = невыгодно · рамка = агент', GRID / 2, GRID - 8);
  }

  function drawStats() {
    statsEl.innerHTML =
      '<span>Эпизод: <span class="v">' + Engine.fmtFixed(episodeCount, 5, 0) + '</span></span>' +
      '<span>Ходы: <span class="v">' + Engine.fmtFixed(episodeSteps, 2, 0) + '/' + MAX_STEPS + '</span></span>' +
      '<span>Сокровищ: <span class="v">' + (3 - treasuresLeft) + '/3</span></span>' +
      '<span>Награда: <span class="v">' + Engine.fmtFixed(episodeReward, 3, 1) + '</span></span>' +
      '<span>Возвратов домой: <span class="v">' + Engine.fmtFixed(score, 5, 0) + '</span></span>' +
      '<span>ε: <span class="v">' + Engine.fmtFixed(currentEpsilon(), 1, 3) + '</span></span>';
  }

  /* ---------- главный цикл: обычный — по таймеру, ускоренный — по бюджету времени на кадр ---------- */
  let lastNormalStep = 0;
  function runFastSteps() {
    const t0 = performance.now();
    let steps = 0;
    while (steps < FAST_MAX_STEPS_PER_FRAME) {
      stepOnce(); steps++;
      if ((steps & 7) === 0 && performance.now() - t0 >= FAST_TIME_BUDGET_MS) break;
    }
  }
  function mainLoop(ts) {
    if (mainStarted) {
      if (fastMode) runFastSteps();
      else if (ts - lastNormalStep >= NORMAL_STEP_DELAY) { stepOnce(); lastNormalStep = ts; }
      drawGame(); drawPolicy(); drawStats();
    }
    parker.raf(mainLoop);
  }
  function introFrame(ts) {
    if (introFinished) return;
    if (introStart === null) introStart = ts;
    const t = (ts - introStart) / 1000;
    drawIntro(t);
    if (t >= 28) { finishIntro(); return; }
    parker.raf(introFrame);
  }
  const parker = Engine.makeRafParker({
    getMain: function () { return mainLoop; },
    getIntro: function () { return introFrame; }
  });

  /* ---------- кривая обучения ---------- */
  const curve = Engine.makeCurve(document.getElementById('treasures-curve'), {
    primaryLabel: METRIC_LABEL, secondaryLabel: 'награда эп.'
  });
  Engine.resizeCanvas(document.getElementById('treasures-curve'), {});
  let lastEpiSample = -1;
  setInterval(() => {
    if (!parker.isRunning()) return;
    if (episodeCount === lastEpiSample) return;
    lastEpiSample = episodeCount;
    curve.sample(episodeCount, score, Math.round(lastEpisodeReward * 100) / 100);
  }, 250);

  speedBtn.addEventListener('click', () => {
    fastMode = !fastMode;
    speedBtn.textContent = fastMode ? 'Обучение: ускоренное' : 'Обучение: обычное'; speedBtn.classList.toggle('eng-btn-accent', fastMode);
  });
  resetBtn.addEventListener('click', () => {
    Q.clear(); score = 0; episodeCount = 0; lastEpisodeReward = 0;
    lastAction = 0; lastNotableEvent = ''; lastUpd = null;
    lastEpiSample = -1; curve.reset();
    resetEpisode();
    drawGame(); drawPolicy(); drawStats();
  });
  infoBtn.addEventListener('click', () => { modalOverlay.classList.add('is-open'); });
  document.getElementById('treasures-modalClose').addEventListener('click', () => { modalOverlay.classList.remove('is-open'); });
  document.getElementById('treasures-modalOk').addEventListener('click', () => { modalOverlay.classList.remove('is-open'); });
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
  const introOverlay = document.getElementById('treasures-introOverlay');
  const ictx = document.getElementById('treasures-introCanvas').getContext('2d');
  let introFinished = false, introStart = null;
  function finishIntro() { if (introFinished) return; introFinished = true; introOverlay.style.display = 'none'; mainStarted = true; }
  introOverlay.addEventListener('click', finishIntro);

  function clamp01(v) { return Math.max(0, Math.min(1, v)); }
  function prog(t, a, b) { return clamp01((t - a) / (b - a)); }
  function ease(p) { return p * p * (3 - 2 * p); }

  const CAPS = [
    [0.0, 2.5, 'Карта 8×8: старт, три сокровища, три ловушки, стены.'],
    [2.5, 5.0, 'Четыре действия: вверх, вправо, вниз, влево.'],
    [5.0, 9.0, 'Состояние: позиция агента + маска собранных сокровищ.'],
    [9.0, 13.0, 'Динамические стены могут появляться и исчезать.'],
    [13.0, 18.0, 'Политика: одна карта для текущей маски + Q-значения.'],
    [18.0, 23.0, 'Награды: сокровище +10, дом +50, ловушка −5, стена −1, шаг −0.1.'],
    [23.0, 28.0, 'Q(s,a) ← Q(s,a) + α·[ r + γ·maxQ(s′) − Q(s,a) ]']
  ];

  function drawIntroMap(c, t, alpha) {
    if (alpha <= 0) return;
    c.save(); c.globalAlpha = alpha;
    const cell = 34, x0 = 24, y0 = 64, size = N * cell;
    c.fillStyle = '#fff'; c.fillRect(x0 - 8, y0 - 8, size + 16, size + 16);
    c.strokeStyle = '#000'; c.lineWidth = 1; c.strokeRect(x0 - 0.5, y0 - 0.5, size + 1, size + 1);
    c.strokeStyle = '#ddd';
    for (let i = 1; i < N; i++) {
      c.beginPath(); c.moveTo(x0 + i * cell + 0.5, y0); c.lineTo(x0 + i * cell + 0.5, y0 + size); c.stroke();
      c.beginPath(); c.moveTo(x0, y0 + i * cell + 0.5); c.lineTo(x0 + size, y0 + i * cell + 0.5); c.stroke();
    }
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const px = x0 + x * cell, py = y0 + y * cell;
      if (BASE_WORLD[y][x] === 1) { c.fillStyle = '#111'; c.fillRect(px + 1, py + 1, cell - 2, cell - 2); }
    }
    for (const tpos of TRAP_POS) drawTrapCell(c, x0 + tpos.x * cell, y0 + tpos.y * cell, cell);
    const blink = Math.floor(t * 1000 / 600) % 2 === 0;
    for (const tpos of TREASURE_POS) {
      const px = x0 + tpos.x * cell, py = y0 + tpos.y * cell, s = cell * 0.62;
      if (blink) { c.fillStyle = ACC; c.fillRect(px + cell / 2 - s / 2, py + cell / 2 - s / 2, s, s); }
      else { c.strokeStyle = ACC; c.lineWidth = 3; c.strokeRect(px + cell / 2 - s / 2, py + cell / 2 - s / 2, s, s); }
    }
    c.strokeStyle = '#000'; c.lineWidth = 2;
    c.beginPath(); c.arc(x0 + START.x * cell + cell / 2, y0 + START.y * cell + cell / 2, cell / 4.2, 0, Math.PI * 2); c.stroke();
    c.beginPath(); c.arc(x0 + START.x * cell + cell / 2, y0 + START.y * cell + cell / 2, cell / 3.1, 0, Math.PI * 2);
    c.fillStyle = ACC; c.fill(); c.strokeStyle = '#000'; c.lineWidth = 2; c.stroke();
    c.restore();
  }
  function drawIntroActions(c, t) {
    if (t < 2.5 || t >= 5.2) return;
    const a = Math.min(prog(t, 2.5, 3.1), 1 - prog(t, 4.6, 5.2));
    if (a <= 0) return;
    c.save(); c.globalAlpha = a;
    const cx = 540, cy = 150, L = 46;
    c.beginPath(); c.arc(cx, cy, 16, 0, Math.PI * 2); c.fillStyle = ACC; c.fill(); c.strokeStyle = '#000'; c.lineWidth = 2; c.stroke();
    drawArrow(c, cx, cy, 0, L, '#000'); drawArrow(c, cx, cy, 1, L, '#000');
    drawArrow(c, cx, cy, 2, L, '#000'); drawArrow(c, cx, cy, 3, L, '#000');
    c.fillStyle = '#000'; c.font = '12px ' + MONO; c.textAlign = 'center';
    c.fillText('вверх', cx, cy - L - 12); c.fillText('вправо', cx + L + 28, cy + 4);
    c.fillText('вниз', cx, cy + L + 20); c.fillText('влево', cx - L - 28, cy + 4);
    c.restore();
  }
  function drawIntroState(c, t) {
    if (t < 5.0 || t >= 9.0) return;
    const a = Math.min(prog(t, 5.0, 5.6), 1 - prog(t, 8.4, 9.0));
    if (a <= 0) return;
    c.save(); c.globalAlpha = a;
    c.fillStyle = '#000'; c.font = '13px Inter, sans-serif'; c.textAlign = 'left';
    c.fillText('Состояние:', 450, 110);
    c.fillText('позиция + маска собранных сокровищ', 450, 132);
    c.font = '13px ' + MONO;
    c.fillText('0,0|0', 450, 168); c.fillText('3,7|1', 450, 190); c.fillText('6,4|7', 450, 212);
    c.fillStyle = '#666'; c.font = '13px Inter, sans-serif';
    c.fillText('|0 — ищем сокровища', 540, 168); c.fillText('|7 — идём домой', 540, 212);
    c.restore();
  }
  function drawIntroWalls(c, t) {
    if (t < 9.0 || t >= 13.0) return;
    const a = Math.min(prog(t, 9.0, 9.6), 1 - prog(t, 12.4, 13.0));
    if (a <= 0) return;
    c.save(); c.globalAlpha = a;
    const blink = Math.floor(t * 3) % 2 === 0;
    c.fillStyle = blink ? '#777' : '#ddd'; c.fillRect(450, 120, 44, 44);
    c.strokeStyle = '#000'; c.lineWidth = 2; c.strokeRect(450.5, 120.5, 44, 44);
    c.fillStyle = '#000'; c.font = '13px Inter, sans-serif'; c.textAlign = 'left';
    c.fillText('динамические стены:', 510, 138); c.fillText('могут появляться/исчезать', 510, 158);
    c.fillStyle = '#666'; c.fillText('вероятность: 15%', 450, 192);
    c.restore();
  }
  function drawIntroQ(c, t) {
    if (t < 13.0 || t >= 18.0) return;
    const a = Math.min(prog(t, 13.0, 13.7), 1 - prog(t, 17.3, 18.0));
    if (a <= 0) return;
    c.save(); c.globalAlpha = a;
    c.fillStyle = '#000'; c.font = '13px Inter, sans-serif'; c.textAlign = 'left';
    c.fillText('Текущая политика и Q-значения:', 450, 110);
    const cell = 12, mapX = 450, mapY = 130, mapSize = N * cell;
    c.fillStyle = '#f7f7f7'; c.fillRect(mapX, mapY, mapSize, mapSize);
    c.strokeStyle = '#000'; c.lineWidth = 1; c.strokeRect(mapX + 0.5, mapY + 0.5, mapSize, mapSize);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const px = mapX + x * cell, py = mapY + y * cell;
      if (BASE_WORLD[y][x] === 1) { c.fillStyle = '#111'; c.fillRect(px + 1, py + 1, cell - 2, cell - 2); }
    }
    for (const tpos of TRAP_POS) {
      c.fillStyle = '#fff'; c.fillRect(mapX + tpos.x * cell + 1, mapY + tpos.y * cell + 1, cell - 2, cell - 2);
      drawSkull(c, mapX + tpos.x * cell + cell / 2, mapY + tpos.y * cell + cell / 2, cell * 0.6);
    }
    const barX = mapX + mapSize + 26, barW = 80, barH = 10;
    for (let a2 = 0; a2 < 4; a2++) {
      const labelY = mapY + 20 + a2 * 30;
      const val = [3.2, 1.8, -0.5, 2.4][a2];
      const norm = Math.min(1, Math.abs(val) / 3.2);
      c.fillStyle = '#000'; c.font = '11px ' + MONO; c.textAlign = 'left';
      c.fillText(ACTION_SHORT[a2], barX, labelY);
      const by = labelY + 4;
      c.fillStyle = '#f0f2f5'; c.fillRect(barX, by, barW, barH);
      c.fillStyle = val >= 0 ? 'rgba(59,111,224,0.85)' : 'rgba(70,70,75,0.85)';
      c.fillRect(barX, by, Math.max(2, norm * barW), barH);
      c.strokeStyle = '#ddd'; c.lineWidth = 1; c.strokeRect(barX + 0.5, by + 0.5, barW - 1, barH - 1);
    }
    c.fillStyle = '#666'; c.font = '12px Inter, sans-serif';
    c.fillText('слева карта политики', 450, 330);
    c.fillText('справа Q-значения текущего состояния', 450, 348);
    c.restore();
  }
  function drawIntroRewards(c, t) {
    if (t < 18.0 || t >= 23.0) return;
    const a = Math.min(prog(t, 18.0, 18.6), 1 - prog(t, 22.4, 23.0));
    if (a <= 0) return;
    c.save(); c.globalAlpha = a;
    c.fillStyle = '#000'; c.font = '13px ' + MONO; c.textAlign = 'left';
    c.fillText('сокровище  +10', 450, 130);
    c.fillText('дом        +50', 450, 154);
    c.fillText('ловушка    −5', 450, 178);
    c.fillText('стена      −1', 450, 202);
    c.fillText('шаг        −0.1', 450, 226);
    c.fillStyle = '#666'; c.font = '13px Inter, sans-serif';
    c.fillText('плюс небольшая подсказка расстоянием', 450, 258);
    c.restore();
  }
  function drawIntro(t) {
    const c = ictx;
    c.clearRect(0, 0, 720, 480); c.fillStyle = '#fff'; c.fillRect(0, 0, 720, 480);
    const cap = CAPS.find(x => t >= x[0] && t < x[1]);
    if (cap) { c.fillStyle = '#000'; c.font = '14px Inter, sans-serif'; c.textAlign = 'left'; c.fillText(cap[2], 14, 24); }
    drawIntroMap(c, t, ease(prog(t, 0.2, 1.2)));
    drawIntroActions(c, t);
    drawIntroState(c, t);
    drawIntroWalls(c, t);
    drawIntroQ(c, t);
    drawIntroRewards(c, t);
    c.fillStyle = '#999'; c.font = '11px Inter, sans-serif'; c.textAlign = 'right'; c.fillText('клик — пропустить', 712, 474);
    const fo = prog(t, 26.5, 27.8);
    if (fo > 0) { c.fillStyle = 'rgba(255,255,255,' + fo.toFixed(3) + ')'; c.fillRect(0, 0, 720, 480); }
    const ta = prog(t, 26.7, 27.3);
    if (ta > 0 && t < 27.9) {
      c.fillStyle = 'rgba(0,0,0,' + ta.toFixed(3) + ')';
      c.font = '20px Inter, sans-serif'; c.textAlign = 'center';
      c.fillText('Обучение начинается…', 360, 240);
    }
  }

  /* ---------- адаптивные канвасы ---------- */
  let gameScale = 1, netScale = 1, gameOX = 0, gameOY = 0, netOX = 0, netOY = 0;
  Engine.resizeCanvas(gameCanvas, { onResize: (w, h) => { gameScale = Math.min(w, h) / GRID; gameOX = (w - GRID * gameScale) / 2; gameOY = (h - GRID * gameScale) / 2; } });
  Engine.resizeCanvas(netCanvas, { onResize: (w, h) => { netScale = Math.min(w, h) / GRID; netOX = (w - GRID * netScale) / 2; netOY = (h - GRID * netScale) / 2; } });
  const rawDrawGame = drawGame, rawDrawPolicy = drawPolicy;
  drawGame = function () { g.save(); g.translate(gameOX, gameOY); g.scale(gameScale, gameScale); g.beginPath(); g.rect(0, 0, GRID, GRID); g.clip(); rawDrawGame(); g.lineWidth = 3 / gameScale; g.strokeStyle = '#1a1c20'; g.strokeRect(0, 0, GRID, GRID); g.restore(); };
  drawPolicy = function () { nctx.save(); nctx.translate(netOX, netOY); nctx.scale(netScale, netScale); rawDrawPolicy(); nctx.restore(); };

  /* ---------- веса ---------- */
  function exportWeights() { return { Q: Array.from(Q.entries()) }; }
  function importWeights(W) {
    try {
      const temp = new Map();
      if (W && Array.isArray(W.Q)) {
        for (const item of W.Q) {
          if (!Array.isArray(item) || item.length !== 2) return false;
          const key = item[0], vals = item[1];
          if (typeof key !== 'string' || !Array.isArray(vals) || vals.length !== 4) return false;
          temp.set(key, vals.map(Number));
        }
      } else if (W && typeof W.Q === 'object' && W.Q !== null) {
        for (const [key, vals] of Object.entries(W.Q)) {
          if (typeof key !== 'string' || !Array.isArray(vals) || vals.length !== 4) return false;
          temp.set(key, vals.map(Number));
        }
      } else return false;
      Q = temp;
      return true;
    } catch (e) { return false; }
  }

  /* ---------- регистрация ---------- */
  resetEpisode();
  drawGame(); drawPolicy(); drawStats();
  parker.raf(mainLoop);
  parker.raf(introFrame);

  Engine.register(DEMO_KEY, {
    key: DEMO_KEY,
    title: 'Вернись с сокровищами',
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
