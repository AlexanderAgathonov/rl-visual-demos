/* Автор: А. А. Агафонов (КФУ). Лицензия: MIT. Часть материалов к статье об изобразительных средствах обучения основам RL. */
/* ==== js/demos/snake.js (demo-kit/snake-q.html) ==== */
(function () {
  'use strict';

  /* ============================================================
     КОНФИГУРАЦИЯ
  ============================================================ */
  const DEMO_KEY = 'snake';
  const ACC = Engine.ACCENT_HEX;
  const METRIC_LABEL = 'Яблок';
  const GRID = 420; // внутренняя система координат обоих квадратных канвасов

  const gameCanvas = document.getElementById('snake-game');
  const g = gameCanvas.getContext('2d');
  const netCanvas = document.getElementById('snake-net');
  const nctx = netCanvas.getContext('2d');
  const statsEl = document.getElementById('snake-stats');
  const modeBtn = document.getElementById('snake-modeBtn');
  const infoBtn = document.getElementById('snake-infoBtn');
  const resetBtn = document.getElementById('snake-resetBtn');
  const weightsBtn = document.getElementById('snake-weightsBtn');
  const modalOverlay = document.getElementById('snake-modalOverlay');
  const modalClose = document.getElementById('snake-modalClose');
  const modalOk = document.getElementById('snake-modalOk');

  const N = 20;
  const CELL = GRID / N;
  const BLINK_MS = 600;

  const NUM_DANGER = 8;
  const NUM_FOOD = 4;
  const NUM_STATES = NUM_DANGER * NUM_FOOD;   // 32
  const NUM_ACTIONS = 3;
  const ALPHA = 0.10;
  const GAMMA_Q = 0.95;
  const EPS_START = 1.0;
  const EPS_END = 0.05;
  const EPS_DECAY = 0.995;
  const FOOD_REWARD = 1;
  const DEATH_PENALTY = -10;
  const STEP_PENALTY = -0.01;
  const SHAPE = 0.1;
  const SHAPE_CLIP = 0.2;
  const MAX_STEPS = 3000;
  const MAX_HUNGER_STEPS = 200;
  const FAST_STEPS = 500;
  const NORMAL_STEP_DELAY = 50;
  const ACTION_LABELS = ['налево', 'прямо', 'направо'];
  const ACTION_ARROWS3 = ['←', '↑', '→'];
  const FOOD_DIR_LABELS = ['вперёд', 'вправо', 'назад', 'влево'];

  /* ============================================================
     СОСТОЯНИЕ
  ============================================================ */
  let Q = [];
  let snake = [];
  let dir = 1;
  let food = { x: 0, y: 0 };
  let score = 0, bestScore = 0;
  let episodeReward = 0, episodeCount = 0, episodeSteps = 0;
  let stepsWithoutFood = 0;
  let currentStateIdx = 0;
  let lastAction = 1, lastActionRandom = false;
  let lastEvent = '', lastNotableEvent = '';
  let lastUpd = null;
  let fastMode = false, mainStarted = false;
  let visitedStates = new Set();

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  const fmt1 = Engine.fmt1;
  const fmt2 = Engine.fmt;

  function initQ() {
    Q = Array.from({ length: NUM_STATES }, () => new Array(NUM_ACTIONS).fill(0));
  }
  function currentEpsilon() {
    return Math.max(EPS_END, EPS_START * Math.pow(EPS_DECAY, episodeCount));
  }
  function argmaxQ(q) {
    let a = 0;
    for (let i = 1; i < q.length; i++) if (q[i] > q[a]) a = i;
    return a;
  }
  function chooseAction(s) {
    if (Math.random() < currentEpsilon()) {
      lastActionRandom = true;
      return Math.floor(Math.random() * NUM_ACTIONS);
    }
    lastActionRandom = false;
    return argmaxQ(Q[s]);
  }

  const DIRS = [
    { x: 0, y: -1 },
    { x: 1, y: 0 },
    { x: 0, y: 1 },
    { x: -1, y: 0 }
  ];
  function outside(x, y) { return x < 0 || x >= N || y < 0 || y >= N; }
  function manhattan(a, b) { return Math.abs(a.x - b.x) + Math.abs(a.y - b.y); }
  function isBody(x, y, excludeTail) {
    for (let i = 0; i < snake.length; i++) {
      if (excludeTail && i === snake.length - 1) continue;
      if (snake[i].x === x && snake[i].y === y) return true;
    }
    return false;
  }
  function isDanger(d) {
    const head = snake[0];
    const v = DIRS[d];
    const nx = head.x + v.x, ny = head.y + v.y;
    return outside(nx, ny) || isBody(nx, ny, true);
  }
  function foodDirection() {
    if (food.x < 0) return 0;
    const head = snake[0];
    const f = DIRS[dir], r = DIRS[(dir + 1) % 4];
    const dx = food.x - head.x, dy = food.y - head.y;
    const ahead = dx * f.x + dy * f.y;
    const right = dx * r.x + dy * r.y;
    if (Math.abs(ahead) >= Math.abs(right)) return ahead >= 0 ? 0 : 2;
    return right >= 0 ? 1 : 3;
  }
  function getStateIndex() {
    const dF = isDanger(dir) ? 1 : 0;
    const dL = isDanger((dir + 3) % 4) ? 1 : 0;
    const dR = isDanger((dir + 1) % 4) ? 1 : 0;
    const dangerIdx = dF * 4 + dL * 2 + dR;
    const fd = foodDirection();
    return dangerIdx * NUM_FOOD + fd;
  }
  function dangerBitsOf(s) {
    const d = Math.floor(s / NUM_FOOD);
    return [(d >> 2) & 1, (d >> 1) & 1, d & 1]; // [вперёд, слева, справа]
  }
  function foodDirOf(s) { return s % NUM_FOOD; }

  /* ============================================================
     ИГРА
  ============================================================ */
  function placeFood() {
    if (snake.length >= N * N) { food = { x: -1, y: -1 }; return; }
    let guard = 0;
    while (guard++ < 1000) {
      const p = { x: Math.floor(Math.random() * N), y: Math.floor(Math.random() * N) };
      if (!isBody(p.x, p.y, false)) { food = p; return; }
    }
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      if (!isBody(x, y, false)) { food = { x, y }; return; }
    }
    food = { x: -1, y: -1 };
  }
  function resetGame() {
    const c = Math.floor(N / 2);
    snake = [{ x: c, y: c }, { x: c - 1, y: c }, { x: c - 2, y: c }];
    dir = 1;
    score = 0; episodeReward = 0; stepsWithoutFood = 0; episodeSteps = 0;
    lastAction = 1;
    placeFood();
    currentStateIdx = getStateIndex();
    visitedStates.add(currentStateIdx);
  }
  function stepGame(action) {
    const oldHead = snake[0];
    const oldDist = food.x < 0 ? 0 : manhattan(oldHead, food);
    if (action === 0) dir = (dir + 3) % 4;
    else if (action === 2) dir = (dir + 1) % 4;
    const newHead = { x: oldHead.x + DIRS[dir].x, y: oldHead.y + DIRS[dir].y };
    if (outside(newHead.x, newHead.y))
      return { reward: DEATH_PENALTY, done: true, result: 'стена' };
    const willEat = food.x >= 0 && newHead.x === food.x && newHead.y === food.y;
    if (isBody(newHead.x, newHead.y, !willEat))
      return { reward: DEATH_PENALTY, done: true, result: 'тело' };
    snake.unshift(newHead);
    let reward = STEP_PENALTY, result = 'шаг';
    if (willEat) {
      score++; stepsWithoutFood = 0;
      reward += FOOD_REWARD; result = 'яблоко';
      if (score > bestScore) bestScore = score;
      if (snake.length >= N * N) { reward += 50; return { reward, done: true, result: 'победа' }; }
      placeFood();
    } else {
      snake.pop();
      stepsWithoutFood++;
      if (food.x >= 0) {
        const newDist = manhattan(newHead, food);
        reward += clamp(SHAPE * (oldDist - newDist), -SHAPE_CLIP, SHAPE_CLIP);
      }
    }
    return { reward, done: false, result };
  }
  function stepOnce() {
    const s = currentStateIdx;
    const a = chooseAction(s);
    lastAction = a;
    const res = stepGame(a);
    episodeReward += res.reward;
    lastEvent = res.result;
    let done = res.done, result = res.result;
    episodeSteps++;
    if (!done && episodeSteps >= MAX_STEPS) { done = true; result = 'время'; }
    if (!done && stepsWithoutFood >= MAX_HUNGER_STEPS) { done = true; result = 'голод'; }
    if (result !== 'шаг') lastNotableEvent = result;
    const s2 = done ? -1 : getStateIndex();
    if (!done) visitedStates.add(s2);

    /* TD-обновление + запоминание разбора для правой панели */
    const old = Q[s][a];
    const maxQ2 = done ? 0 : Math.max(Q[s2][0], Q[s2][1], Q[s2][2]);
    const target = res.reward + GAMMA_Q * maxQ2;
    Q[s][a] = old + ALPHA * (target - old);
    lastUpd = { s, a, r: res.reward, maxQ2, target, old, neu: Q[s][a], s2, done };

    if (done) { episodeCount++; resetGame(); }
    else currentStateIdx = s2;
  }

  /* ============================================================
     РЕНДЕР: ИГРА (координаты — во внутренней сетке GRID×GRID,
     canvas масштабируется до реального размера контейнера в
     resize-колбэке через ctx.scale)
  ============================================================ */
  function drawGame() {
    g.clearRect(0, 0, GRID, GRID);
    g.fillStyle = '#fff'; g.fillRect(0, 0, GRID, GRID);
    g.strokeStyle = '#eee'; g.lineWidth = 1;
    for (let i = 1; i < N; i++) {
      g.beginPath(); g.moveTo(i * CELL + 0.5, 0); g.lineTo(i * CELL + 0.5, GRID); g.stroke();
      g.beginPath(); g.moveTo(0, i * CELL + 0.5); g.lineTo(GRID, i * CELL + 0.5); g.stroke();
    }
    if (food.x >= 0) {
      const blink = Math.floor(performance.now() / BLINK_MS) % 2 === 0;
      if (blink) { g.fillStyle = ACC; g.fillRect(food.x * CELL, food.y * CELL, CELL, CELL); }
      else { g.strokeStyle = ACC; g.lineWidth = 2; g.strokeRect(food.x * CELL + 1, food.y * CELL + 1, CELL - 2, CELL - 2); }
    }
    snake.forEach((seg, i) => {
      g.fillStyle = i === 0 ? ACC : '#333';
      g.fillRect(seg.x * CELL + 1, seg.y * CELL + 1, CELL - 2, CELL - 2);
    });
  }

  /* ============================================================
     ПИКТОГРАММЫ СОСТОЯНИЯ (head-frame)
  ============================================================ */
  const drawSmallArrow = Engine.drawSmallArrow;
  function drawPict(c, x, y, cell, s) {
    const bits = dangerBitsOf(s);   // [вперёд, слева, справа]
    const fd = foodDirOf(s);
    c.fillStyle = '#fff';
    c.fillRect(x, y, cell * 3, cell * 3);
    const top = [bits[1], bits[0], bits[2]]; // слева, вперёд, справа
    for (let i = 0; i < 3; i++) {
      c.fillStyle = top[i] ? '#444' : '#eee';
      c.fillRect(x + i * cell + 1, y + 1, cell - 2, cell - 2);
    }
    c.fillStyle = '#eee';
    c.fillRect(x + cell + 1, y + 2 * cell + 1, cell - 2, cell - 2);
    c.strokeStyle = '#ccc'; c.lineWidth = 1;
    for (let i = 0; i <= 3; i++) {
      c.beginPath(); c.moveTo(x + i * cell + 0.5, y); c.lineTo(x + i * cell + 0.5, y + 3 * cell); c.stroke();
    }
    for (let j = 0; j <= 3; j++) {
      c.beginPath(); c.moveTo(x, y + j * cell + 0.5); c.lineTo(x + 3 * cell, y + j * cell + 0.5); c.stroke();
    }
    c.beginPath();
    c.arc(x + 1.5 * cell, y + 1.5 * cell, cell / 2.6, 0, Math.PI * 2);
    c.fillStyle = ACC; c.fill();
    c.strokeStyle = '#000'; c.lineWidth = 1.5; c.stroke();
    const dxy = [[0, -1], [1, 0], [0, 1], [-1, 0]][fd];
    drawSmallArrow(
      c,
      x + 1.5 * cell, y + 1.5 * cell,
      x + 1.5 * cell + dxy[0] * cell * 1.1,
      y + 1.5 * cell + dxy[1] * cell * 1.1,
      ACC
    );
  }
  function drawActionArrowCell(c, cx, cy, a, len, color) {
    const dxy = [[-1, 0], [0, -1], [1, 0]][a]; // налево / прямо / направо
    drawSmallArrow(c, cx, cy, cx + dxy[0] * len, cy + dxy[1] * len, color);
  }

  /* ============================================================
     ПРАВАЯ ПАНЕЛЬ: 3 ЗОНЫ
     (шрифты увеличены относительно исходника ради читаемости —
     требование №6/№2: текст не должен теряться на маленьком экране)
  ============================================================ */
  function drawRight() {
    nctx.clearRect(0, 0, GRID, GRID);
    nctx.fillStyle = '#000'; nctx.font = '13px Inter, sans-serif'; nctx.textAlign = 'center';
    nctx.fillText('Ситуация → код · политика · TD-обновление', 210, 16);

    /* ---------- ЗОНА 1: что видит змейка → код ---------- */
    const cs = 15, cx0 = 14, cy0 = 26;
    const head = snake[0], f = DIRS[dir], r = DIRS[(dir + 1) % 4];
    for (let j = 0; j < 5; j++) {
      for (let i = 0; i < 5; i++) {
        const u = i - 2, v = 2 - j;
        const bx = head.x + r.x * u + f.x * v;
        const by = head.y + r.y * u + f.y * v;
        const px = cx0 + i * cs, py = cy0 + j * cs;
        nctx.fillStyle = '#fff';
        nctx.fillRect(px, py, cs, cs);
        if (outside(bx, by)) {
          nctx.fillStyle = '#111';
          nctx.fillRect(px + 1, py + 1, cs - 2, cs - 2);
        } else if (isBody(bx, by, false)) {
          nctx.fillStyle = '#333';
          nctx.fillRect(px + 1, py + 1, cs - 2, cs - 2);
        } else if (food.x === bx && food.y === by) {
          nctx.fillStyle = ACC;
          nctx.fillRect(px + 2, py + 2, cs - 4, cs - 4);
        }
        nctx.strokeStyle = '#e5e5e5'; nctx.lineWidth = 1;
        nctx.strokeRect(px + 0.5, py + 0.5, cs, cs);
      }
    }
    nctx.beginPath();
    nctx.arc(cx0 + 2 * cs + cs / 2, cy0 + 2 * cs + cs / 2, cs / 2.6, 0, Math.PI * 2);
    nctx.fillStyle = ACC; nctx.fill();
    nctx.strokeStyle = '#000'; nctx.lineWidth = 1.5; nctx.stroke();

    nctx.fillStyle = '#000'; nctx.font = '16px Inter, sans-serif'; nctx.textAlign = 'center';
    nctx.fillText('→', 100, 68);

    drawPict(nctx, 112, 26, 15, currentStateIdx);
    const bits = dangerBitsOf(currentStateIdx);
    const fd = foodDirOf(currentStateIdx);

    /* бары трёх действий текущего состояния (справа от пиктограммы) */
    const q = Q[currentStateIdx];
    let bm = 0.01;
    for (let a = 0; a < 3; a++) if (Math.abs(q[a]) > bm) bm = Math.abs(q[a]);
    for (let a = 0; a < 3; a++) {
      const y = 26 + a * 16;
      nctx.fillStyle = '#000'; nctx.font = '11px Inter, sans-serif'; nctx.textAlign = 'left';
      nctx.fillText(ACTION_ARROWS3[a] + ' ' + ACTION_LABELS[a], 168, y + 9);
      nctx.fillStyle = '#f0f2f5';
      nctx.fillRect(250, y, 100, 12);
      const v = q[a];
      if (Math.abs(v) > 1e-6) {
        nctx.fillStyle = v >= 0 ? 'rgba(59,111,224,0.85)' : 'rgba(70,70,75,0.85)';
        nctx.fillRect(250, y, Math.max(2, Math.abs(v) / bm * 100), 12);
      }
      if (a === lastAction) {
        nctx.strokeStyle = '#000'; nctx.lineWidth = 1.5;
        if (lastActionRandom) nctx.setLineDash([3, 2]);
        nctx.strokeRect(249.5, y - 0.5, 101, 13);
        nctx.setLineDash([]);
      } else {
        nctx.strokeStyle = '#ddd'; nctx.lineWidth = 1;
        nctx.strokeRect(250.5, y + 0.5, 99, 11);
      }
      nctx.fillStyle = '#000'; nctx.textAlign = 'right'; nctx.font = '11px ' + Engine.FONT_MONO;
      nctx.fillText(fmt2(v), 410, y + 9);
    }

    /* подписи ситуации — отдельной строкой во всю ширину, ниже и пиктограммы,
       и кроп-грида (тот заканчивается на y=99), и баров действий */
    nctx.fillStyle = '#000'; nctx.font = '11px Inter, sans-serif'; nctx.textAlign = 'left';
    nctx.fillText('опасность ' + bits.join('') + '  ·  еда: ' + FOOD_DIR_LABELS[fd], 16, 110);
    nctx.fillStyle = '#666'; nctx.font = '10px Inter, sans-serif';
    nctx.fillText('ε=' + currentEpsilon().toFixed(2) + ' · ход: ' + (lastActionRandom ? 'случайный' : 'жадный'), 16, 122);

    /* ---------- ЗОНА 2: карта политики 8×4 ---------- */
    nctx.fillStyle = '#000'; nctx.font = '11px Inter, sans-serif'; nctx.textAlign = 'center';
    nctx.fillText('Политика: оттенок = ценность · стрелка = действие · число = maxQ', 210, 138);

    const gx = 36, gy = 156, colW = 94, rowH = 23;

    /* заголовки колонок: сектор еды */
    for (let c2 = 0; c2 < 4; c2++) {
      const hx = gx + c2 * colW + 16, hy = 148;
      nctx.fillStyle = '#000';
      nctx.beginPath(); nctx.arc(hx, hy, 3, 0, Math.PI * 2); nctx.fill();
      const dxy = [[0, -1], [1, 0], [0, 1], [-1, 0]][c2];
      nctx.fillStyle = ACC;
      nctx.beginPath(); nctx.arc(hx + dxy[0] * 8, hy + dxy[1] * 8, 2.5, 0, Math.PI * 2); nctx.fill();
      nctx.fillStyle = '#666'; nctx.font = '9px Inter, sans-serif'; nctx.textAlign = 'left';
      nctx.fillText(FOOD_DIR_LABELS[c2], hx + 12, hy + 3);
    }

    let maxAbs = 0.01;
    for (let s = 0; s < NUM_STATES; s++)
      for (let a = 0; a < NUM_ACTIONS; a++) {
        const v = Math.abs(Q[s][a]);
        if (v > maxAbs) maxAbs = v;
      }

    for (let d = 0; d < 8; d++) {
      /* заголовок строки: мини-пиктограмма опасности */
      const rb = [(d >> 2) & 1, (d >> 1) & 1, d & 1];
      const ry = gy + d * rowH;
      const s6 = 6;
      for (let i = 0; i < 3; i++) {
        nctx.fillStyle = rb[i] ? '#444' : '#eee';
        nctx.fillRect(12 + i * s6, ry + 3, s6 - 1, s6 - 1);
      }
      nctx.fillStyle = ACC;
      nctx.beginPath(); nctx.arc(12 + 1.5 * s6, ry + 3 + s6 + 4, 2.5, 0, Math.PI * 2); nctx.fill();

      for (let fd2 = 0; fd2 < 4; fd2++) {
        const sIdx = d * 4 + fd2;
        const px = gx + fd2 * colW, py = ry;
        const visited = visitedStates.has(sIdx);
        const q2 = Q[sIdx];
        let maxQ = q2[0], bestA = 0;
        for (let a = 1; a < 3; a++) if (q2[a] > maxQ) { maxQ = q2[a]; bestA = a; }

        if (visited) {
          nctx.fillStyle = Engine.shadeFor(maxQ, maxAbs);
        } else {
          nctx.fillStyle = '#fafafa';
        }
        nctx.fillRect(px + 1, py + 1, colW - 2, rowH - 2);

        if (visited) {
          const light = Math.abs(maxQ) / maxAbs > 0.55;
          drawActionArrowCell(nctx, px + 24, py + rowH / 2, bestA, 7, light ? '#fff' : '#000');
          nctx.fillStyle = light ? '#fff' : '#000';
          nctx.font = '11px ' + Engine.FONT_MONO; nctx.textAlign = 'center';
          nctx.fillText(fmt1(maxQ), px + 62, py + rowH / 2 + 3);
        } else {
          nctx.fillStyle = '#bbb';
          nctx.font = '11px ' + Engine.FONT_MONO; nctx.textAlign = 'center';
          nctx.fillText('·', px + colW / 2, py + rowH / 2 + 3);
        }

        if (sIdx === currentStateIdx) {
          nctx.strokeStyle = '#000'; nctx.lineWidth = 2;
          nctx.strokeRect(px + 1.5, py + 1.5, colW - 3, rowH - 3);
        } else if (lastUpd && lastUpd.s === sIdx) {
          nctx.strokeStyle = ACC; nctx.lineWidth = 1;
          nctx.strokeRect(px + 1.5, py + 1.5, colW - 3, rowH - 3);
        }
      }
    }
    nctx.strokeStyle = '#ddd'; nctx.lineWidth = 1;
    for (let rr = 0; rr <= 8; rr++) {
      nctx.beginPath(); nctx.moveTo(gx, gy + rr * rowH + 0.5); nctx.lineTo(gx + 4 * colW, gy + rr * rowH + 0.5); nctx.stroke();
    }
    for (let cc = 0; cc <= 4; cc++) {
      nctx.beginPath(); nctx.moveTo(gx + cc * colW + 0.5, gy); nctx.lineTo(gx + cc * colW + 0.5, gy + 8 * rowH); nctx.stroke();
    }

    /* ---------- ЗОНА 3: TD-разбор ---------- */
    const zy = 354;
    nctx.strokeStyle = '#eee'; nctx.lineWidth = 1;
    nctx.beginPath(); nctx.moveTo(10, zy - 10); nctx.lineTo(410, zy - 10); nctx.stroke();

    if (lastUpd) {
      drawPict(nctx, 16, zy, 10, lastUpd.s);
      nctx.strokeStyle = '#000'; nctx.fillStyle = '#000'; nctx.lineWidth = 1.5;
      nctx.beginPath(); nctx.moveTo(50, zy + 15); nctx.lineTo(92, zy + 15); nctx.stroke();
      nctx.beginPath();
      nctx.moveTo(92, zy + 15);
      nctx.lineTo(86, zy + 11);
      nctx.lineTo(86, zy + 19);
      nctx.closePath(); nctx.fill();
      nctx.font = '10px ' + Engine.FONT_MONO; nctx.textAlign = 'center';
      nctx.fillText(ACTION_LABELS[lastUpd.a] + '  r=' + fmt1(lastUpd.r), 71, zy - 2);
      if (lastUpd.done) nctx.fillText('терминал', 71, zy + 36);
      if (!lastUpd.done) drawPict(nctx, 100, zy, 10, lastUpd.s2);

      nctx.textAlign = 'left'; nctx.font = '11px ' + Engine.FONT_MONO; nctx.fillStyle = '#000';
      nctx.fillText('Q старое=' + fmt2(lastUpd.old) + '   maxQ(s′)=' + fmt2(lastUpd.maxQ2), 150, zy + 8);
      nctx.fillText('цель = r + γ·maxQ′ = ' + fmt2(lastUpd.target), 150, zy + 22);
      nctx.fillStyle = ACC;
      nctx.fillText(
        'Q новое=' + fmt2(lastUpd.old) + '+' + ALPHA + '·(' + fmt2(lastUpd.target) + '−' + fmt2(lastUpd.old) + ')=' + fmt2(lastUpd.neu),
        150, zy + 36
      );
    } else {
      nctx.fillStyle = '#999'; nctx.font = '11px Inter, sans-serif'; nctx.textAlign = 'left';
      nctx.fillText('Здесь появится разбор TD-обновления…', 16, zy + 20);
    }

    nctx.fillStyle = '#999'; nctx.font = '9px Inter, sans-serif'; nctx.textAlign = 'center';
    nctx.fillText('рамка = текущая ситуация · акцент = обновление · точка = не посещено', 210, 414);
  }

  /* ============================================================
     СТАТИСТИКА (фиксированная ширина чисел — требование №2)
  ============================================================ */
  function drawStats() {
    statsEl.innerHTML =
      '<span>Эпизод: <span class="v">' + Engine.fmtFixed(episodeCount, 5, 0) + '</span></span>' +
      '<span>' + METRIC_LABEL + ': <span class="v">' + Engine.fmtFixed(score, 3, 0) + '</span></span>' +
      '<span>Рекорд: <span class="v">' + Engine.fmtFixed(bestScore, 3, 0) + '</span></span>' +
      '<span>Награда: <span class="v">' + Engine.fmtFixed(episodeReward, 3, 1) + '</span></span>' +
      '<span>ε: <span class="v">' + Engine.fmtFixed(currentEpsilon(), 1, 3) + '</span></span>' +
      '<span>Ситуаций в таблице: <span class="v">' + Engine.fmtFixed(visitedStates.size, 2, 0) + '/' + NUM_STATES + '</span></span>';
  }

  /* ============================================================
     ГЛАВНЫЙ ЦИКЛ + ПАРКОВЩИК rAF
  ============================================================ */
  let lastNormalStep = 0;
  function mainLoop(ts) {
    if (mainStarted) {
      if (fastMode) { for (let i = 0; i < FAST_STEPS; i++) stepOnce(); }
      else if (ts - lastNormalStep >= NORMAL_STEP_DELAY) { stepOnce(); lastNormalStep = ts; }
      drawGame(); drawRight(); drawStats();
    }
    parker.raf(mainLoop);
  }
  function introFrame(ts) {
    if (introFinished) return;
    if (introStart === null) introStart = ts;
    const t = (ts - introStart) / 1000;
    drawIntro(t);
    if (t >= 28.0) { finishIntro(); return; }
    parker.raf(introFrame);
  }
  const parker = Engine.makeRafParker({
    getMain: function () { return mainLoop; },
    getIntro: function () { return introFrame; }
  });

  /* ============================================================
     КНОПКИ
  ============================================================ */
  if (modeBtn) modeBtn.addEventListener('click', () => {
    fastMode = !fastMode;
    modeBtn.textContent = fastMode ? 'Обучение: ускоренное' : 'Обучение: обычное'; modeBtn.classList.toggle('eng-btn-accent', fastMode);
  });
  if (infoBtn) infoBtn.addEventListener('click', () => { if (modalOverlay) modalOverlay.classList.add('is-open'); });
  if (modalClose) modalClose.addEventListener('click', () => { if (modalOverlay) modalOverlay.classList.remove('is-open'); });
  if (modalOk) modalOk.addEventListener('click', () => { if (modalOverlay) modalOverlay.classList.remove('is-open'); });
  if (modalOverlay) modalOverlay.addEventListener('click', e => { if (e.target === modalOverlay) modalOverlay.classList.remove('is-open'); });
  if (resetBtn) resetBtn.addEventListener('click', () => { resetAll(); });
  function resetAll() {
    initQ();
    bestScore = 0; episodeCount = 0;
    lastNotableEvent = '';
    lastUpd = null;
    visitedStates.clear();
    curve.reset();
    resetGame();
    drawGame(); drawRight(); drawStats();
  }
  if (weightsBtn) weightsBtn.addEventListener('click', () => {
    Engine.openWeightsModal({
      key: DEMO_KEY,
      exportWeights: exportWeights,
      importWeights: importWeights,
      presetWeights: (window.DemoWeights && window.DemoWeights[DEMO_KEY]) || null
    });
  });

  document.addEventListener('keydown', (e) => {
    if (!parker.isRunning()) return;
    if (e.code === 'KeyU' && modeBtn) { e.preventDefault(); modeBtn.click(); }
    else if (e.code === 'KeyI' && infoBtn) { e.preventDefault(); infoBtn.click(); }
    else if (e.code === 'Backspace' && resetBtn) { e.preventDefault(); resetBtn.click(); }
  });

  /* ============================================================
     ЗАСТАВКА (технический язык уместен — требование №7;
     в презентацию не идёт — требование №5)
  ============================================================ */
  const introOverlay = document.getElementById('snake-introOverlay');
  const introCanvas = document.getElementById('snake-introCanvas');
  const ictx = introCanvas.getContext('2d');
  let introFinished = false;
  function finishIntro() {
    if (introFinished) return;
    introFinished = true;
    introOverlay.style.display = 'none';
    mainStarted = true;
  }
  introOverlay.addEventListener('click', finishIntro);
  function clamp01(v) { return Math.max(0, Math.min(1, v)); }
  function prog(t, a, b) { return clamp01((t - a) / (b - a)); }
  function ease(p) { return p * p * (3 - 2 * p); }
  const IC = 66, COLS_I = 6, ROWS_I = 6, SX = 10, SY = 50;
  const HEAD_I = { x: 3, y: 2 };
  const BODY_I = [{ x: 3, y: 3 }, { x: 3, y: 4 }, { x: 3, y: 5 }];
  const APPLE_I = { x: 5, y: 1 };
  const CAPS = [
    [0.0, 2.5, 'Фрагмент поля 20×20: змейка, яблоко, стена.'],
    [2.5, 5.0, 'Три действия: налево, прямо, направо.'],
    [5.0, 9.5, 'Опасность: стена или тело в 1 шаге вперёд / слева / справа — три флага.'],
    [9.5, 13.5, 'Куда еда? Вперёд / вправо / назад / влево относительно головы.'],
    [13.5, 21.5, 'Справа: ситуация → код, карта политики 8×4 и разбор TD-обновления.'],
    [21.5, 24.5, 'Награды: яблоко +1, смерть −10, шаг −0.01.'],
    [24.5, 28.0, 'Обучение с подкреплением начинается…']
  ];
  function icx(x) { return SX + (x + 0.5) * IC; }
  function icy(y) { return SY + (y + 0.5) * IC; }
  function drawIntroArrow(c, x1, y1, x2, y2) {
    c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2, y2); c.stroke();
    const ang = Math.atan2(y2 - y1, x2 - x1), s = 7;
    c.beginPath();
    c.moveTo(x2, y2);
    c.lineTo(x2 - s * Math.cos(ang - 0.4), y2 - s * Math.sin(ang - 0.4));
    c.lineTo(x2 - s * Math.cos(ang + 0.4), y2 - s * Math.sin(ang + 0.4));
    c.closePath(); c.fill();
  }
  function drawIntroScene(c, t, alpha) {
    if (alpha <= 0) return;
    c.save(); c.globalAlpha = alpha;
    c.strokeStyle = '#000'; c.lineWidth = 1;
    c.setLineDash([4, 3]);
    c.strokeRect(SX - 0.5, SY - 0.5, COLS_I * IC + 1, ROWS_I * IC + 1);
    c.setLineDash([]);
    c.strokeStyle = '#ddd';
    for (let i = 1; i < COLS_I; i++) { c.beginPath(); c.moveTo(SX + i * IC + 0.5, SY); c.lineTo(SX + i * IC + 0.5, SY + ROWS_I * IC); c.stroke(); }
    for (let j = 1; j < ROWS_I; j++) { c.beginPath(); c.moveTo(SX, SY + j * IC + 0.5); c.lineTo(SX + COLS_I * IC, SY + j * IC + 0.5); c.stroke(); }
    c.fillStyle = '#000';
    c.fillRect(SX - 4, SY - 16, COLS_I * IC + 8, 14);
    c.fillStyle = '#fff'; c.font = '11px Inter, sans-serif'; c.textAlign = 'center';
    c.fillText('стена', SX + COLS_I * IC / 2, SY - 5);
    const ax = SX + APPLE_I.x * IC, ay = SY + APPLE_I.y * IC;
    const blink = Math.floor(t * 1000 / BLINK_MS) % 2 === 0;
    if (blink) { c.fillStyle = ACC; c.fillRect(ax, ay, IC, IC); }
    else { c.strokeStyle = ACC; c.lineWidth = 2; c.strokeRect(ax + 1.5, ay + 1.5, IC - 3, IC - 3); }
    const segs = [HEAD_I, ...BODY_I];
    segs.forEach((s, i) => {
      c.fillStyle = i === 0 ? ACC : '#333';
      c.fillRect(SX + s.x * IC + 1, SY + s.y * IC + 1, IC - 2, IC - 2);
    });
    c.restore();
  }
  function drawIntroActions(c, t) {
    const p = ease(prog(t, 2.6, 3.5));
    const fade = 1 - prog(t, 4.6, 5.2);
    const al = Math.min(p, fade);
    if (al <= 0) return;
    c.save(); c.globalAlpha = al;
    const hx = icx(HEAD_I.x), hy = icy(HEAD_I.y);
    const L = 1.25 * IC * p;
    c.strokeStyle = '#000'; c.fillStyle = '#000'; c.lineWidth = 2;
    drawIntroArrow(c, hx, hy, hx, hy - L);
    drawIntroArrow(c, hx, hy, hx - L, hy);
    drawIntroArrow(c, hx, hy, hx + L, hy);
    c.font = '12px Inter, sans-serif';
    c.textAlign = 'left'; c.fillText('вперёд', hx + 8, hy - 1.25 * IC - 8);
    c.textAlign = 'right'; c.fillText('лево', hx - 1.25 * IC - 8, hy + 4);
    c.textAlign = 'left'; c.fillText('право', hx + 1.25 * IC + 8, hy + 4);
    c.restore();
  }
  function drawIntroDanger(c, t) {
    if (t < 5.0 || t >= 9.5) return;
    const fade = 1 - prog(t, 8.9, 9.5);
    c.save(); c.globalAlpha = fade;
    const hx = icx(HEAD_I.x), hy = icy(HEAD_I.y);
    const grow = ease(prog(t, 5.2, 7.0));
    const L = 2 * IC * grow;
    c.strokeStyle = ACC; c.lineWidth = 2; c.setLineDash([5, 4]);
    c.beginPath(); c.moveTo(hx, hy); c.lineTo(hx, hy - L); c.stroke();
    c.setLineDash([]);
    if (t > 7.2) {
      const fa = prog(t, 7.2, 7.8);
      c.globalAlpha = fade * fa;
      const labels = ['вперёд', 'слева', 'справа'];
      const vals = [true, false, false];
      for (let i = 0; i < 3; i++) {
        const y = 100 + i * 30;
        c.fillStyle = vals[i] ? '#555' : '#ddd';
        c.strokeStyle = vals[i] ? '#333' : '#aaa';
        c.lineWidth = 1;
        c.fillRect(440, y - 8, 14, 14);
        c.strokeRect(440, y - 8, 14, 14);
        c.fillStyle = '#000'; c.font = '12px Inter, sans-serif'; c.textAlign = 'left';
        c.fillText(labels[i] + ': ' + (vals[i] ? '1' : '0'), 462, y + 4);
      }
    }
    c.restore();
  }
  function drawIntroFood(c, t) {
    if (t < 9.0 || t >= 13.5) return;
    const fade = 1 - prog(t, 12.9, 13.5);
    c.save(); c.globalAlpha = fade;
    const hx = icx(HEAD_I.x), hy = icy(HEAD_I.y);
    const ax = icx(APPLE_I.x), ay = icy(APPLE_I.y);
    c.strokeStyle = ACC; c.lineWidth = 2; c.setLineDash([4, 4]);
    c.beginPath(); c.moveTo(hx, hy); c.lineTo(ax, ay); c.stroke();
    c.setLineDash([]);
    if (t > 10.0) {
      c.fillStyle = '#000'; c.font = '12px Inter, sans-serif'; c.textAlign = 'left';
      c.fillText('еда: вперёд-вправо → сектор «вперёд»', 440, 110);
    }
    c.restore();
  }
  function drawIntroMiniTable(c, t) {
    if (t < 13.5) return;
    const alpha = prog(t, 13.5, 14.5);
    c.save(); c.globalAlpha = alpha;
    const x0 = 440, y0 = 150;
    const cw = 36, ch = 22, cols = 4, rows = 8;
    c.fillStyle = '#000'; c.font = '600 13px Inter, sans-serif'; c.textAlign = 'left';
    c.fillText('Карта политики', x0, y0 - 10);
    for (let r = 0; r < rows; r++) {
      for (let cc = 0; cc < cols; cc++) {
        const cx = x0 + cc * cw, cy = y0 + r * ch;
        c.fillStyle = '#f4f5f7';
        c.fillRect(cx, cy, cw - 1, ch - 1);
        const h = 3 + ((r * 7 + cc * 3) % 10);
        c.fillStyle = 'rgba(59,111,224,' + (0.2 + 0.06 * h).toFixed(2) + ')';
        c.fillRect(cx + 1, cy + 1, cw - 3, ch - 3);
        c.strokeStyle = '#000'; c.fillStyle = '#000'; c.lineWidth = 1.2;
        const aa = (r + cc) % 3;
        const dxy = [[-1, 0], [0, -1], [1, 0]][aa];
        c.beginPath();
        c.moveTo(cx + cw / 2, cy + ch / 2);
        c.lineTo(cx + cw / 2 + dxy[0] * 7, cy + ch / 2 + dxy[1] * 7);
        c.stroke();
      }
    }
    c.strokeStyle = '#bbb'; c.lineWidth = 1;
    for (let r = 0; r <= rows; r++) { c.beginPath(); c.moveTo(x0, y0 + r * ch); c.lineTo(x0 + cols * cw, y0 + r * ch); c.stroke(); }
    for (let cc = 0; cc <= cols; cc++) { c.beginPath(); c.moveTo(x0 + cc * cw, y0); c.lineTo(x0 + cc * cw, y0 + rows * ch); c.stroke(); }
    c.font = '11px Inter, sans-serif'; c.textAlign = 'left';
    if (t > 13.8) {
      c.strokeStyle = '#000'; c.lineWidth = 2;
      c.strokeRect(x0 + 1 * cw + 0.5, y0 + 3 * ch + 0.5, cw - 1, ch - 1);
      c.fillStyle = '#000';
      c.fillText('рамка = текущая ситуация', x0, y0 + rows * ch + 18);
      c.fillStyle = '#555';
      c.fillText('оттенок = ценность, стрелка = действие', x0, y0 + rows * ch + 32);
    }
    if (t > 16.5) {
      c.fillStyle = '#000';
      c.fillText('внизу панели — разбор обновления:', x0, y0 + rows * ch + 52);
      c.fillStyle = '#555';
      c.fillText('Q += α·(r + γ·maxQ′ − Q) с числами', x0, y0 + rows * ch + 66);
    }
    c.restore();
  }
  function drawIntroRewards(c, t) {
    if (t < 21.5 || t >= 24.7) return;
    const a = Math.min(prog(t, 21.5, 22.1), 1 - prog(t, 24.1, 24.7));
    c.save(); c.globalAlpha = a;
    c.fillStyle = '#000'; c.font = '13px Inter, sans-serif'; c.textAlign = 'left';
    c.fillText('яблоко  +1', SX + 6, SY + ROWS_I * IC + 14);
    c.fillText('смерть  −10', SX + 6, SY + ROWS_I * IC + 32);
    c.fillText('шаг     −0.01', SX + 6, SY + ROWS_I * IC + 50);
    c.restore();
  }
  function drawIntro(t) {
    const c = ictx;
    c.clearRect(0, 0, 720, 480);
    c.fillStyle = '#fff'; c.fillRect(0, 0, 720, 480);
    const cap = CAPS.find(x => t >= x[0] && t < x[1]);
    if (cap) { c.fillStyle = '#000'; c.font = '14px Inter, sans-serif'; c.textAlign = 'left'; c.fillText(cap[2], 14, 24); }
    drawIntroScene(c, t, ease(prog(t, 0.2, 1.2)));
    drawIntroActions(c, t);
    drawIntroDanger(c, t);
    drawIntroFood(c, t);
    drawIntroMiniTable(c, t);
    drawIntroRewards(c, t);
    c.fillStyle = '#999'; c.font = '11px Inter, sans-serif'; c.textAlign = 'right';
    c.fillText('клик — пропустить', 712, 474);
    const fo = prog(t, 26.0, 27.3);
    if (fo > 0) { c.fillStyle = 'rgba(255,255,255,' + fo + ')'; c.fillRect(0, 0, 720, 480); }
    const ta = prog(t, 26.2, 26.8);
    if (ta > 0 && t < 27.6) {
      c.fillStyle = 'rgba(0,0,0,' + ta + ')'; c.font = '20px Inter, sans-serif'; c.textAlign = 'center';
      c.fillText('Обучение начинается…', 360, 240);
    }
  }
  let introStart = null;

  /* ============================================================
     АДАПТИВНЫЕ КАНВАСЫ (требование №3)
     Внутренняя система координат обоих квадратных канвасов
     фиксирована (GRID×GRID) — реальный размер задаёт CSS/ResizeObserver,
     а рисунок масштабируется одним ctx.scale(k,k) в начале каждого
     кадра, поэтому вся геометрия выше не переписывается под каждый
     экран заново.
  ============================================================ */
  let gameScale = 1, netScale = 1, gameOX = 0, gameOY = 0, netOX = 0, netOY = 0;
  Engine.resizeCanvas(gameCanvas, { onResize: (w, h) => { gameScale = Math.min(w, h) / GRID; gameOX = (w - GRID * gameScale) / 2; gameOY = (h - GRID * gameScale) / 2; } });
  Engine.resizeCanvas(netCanvas, { onResize: (w, h) => { netScale = Math.min(w, h) / GRID; netOX = (w - GRID * netScale) / 2; netOY = (h - GRID * netScale) / 2; } });
  const rawDrawGame = drawGame, rawDrawRight = drawRight;
  drawGame = function () { g.save(); g.translate(gameOX, gameOY); g.scale(gameScale, gameScale); g.beginPath(); g.rect(0, 0, GRID, GRID); g.clip(); rawDrawGame(); g.lineWidth = 3 / gameScale; g.strokeStyle = '#1a1c20'; g.strokeRect(0, 0, GRID, GRID); g.restore(); };
  drawRight = function () { nctx.save(); nctx.translate(netOX, netOY); nctx.scale(netScale, netScale); rawDrawRight(); nctx.restore(); };

  const curve = Engine.makeCurve(document.getElementById('snake-curve'), {
    primaryLabel: 'рекорд яблок', secondaryLabel: 'награда эп.'
  });
  Engine.resizeCanvas(document.getElementById('snake-curve'), {});

  /* ============================================================
     СТАРТ
  ============================================================ */
  initQ();
  resetGame();
  drawGame(); drawRight(); drawStats();
  parker.raf(mainLoop);
  parker.raf(introFrame);

  /* ============================================================
     ЭКСПОРТ / ИМПОРТ ВЕСОВ
  ============================================================ */
  function exportWeights() { return { Q: Q }; }
  function importWeights(W) {
    try {
      if (!W || !W.Q || W.Q.length !== NUM_STATES) return false;
      for (let i = 0; i < NUM_STATES; i++) {
        if (!Array.isArray(W.Q[i]) || W.Q[i].length !== NUM_ACTIONS) return false;
      }
      Q = W.Q.map(r => r.slice());
      return true;
    } catch (e) { return false; }
  }

  /* ============================================================
     КРИВАЯ ОБУЧЕНИЯ: сэмплирование раз в 250мс, пока демо активно
  ============================================================ */
  let lastEpiSample = -1;
  setInterval(() => {
    if (!parker.isRunning()) return;
    if (episodeCount === lastEpiSample) return;
    lastEpiSample = episodeCount;
    curve.sample(episodeCount, bestScore, episodeReward);
  }, 250);

  /* ============================================================
     РЕГИСТРАЦИЯ В КОЛОДЕ
  ============================================================ */
  Engine.register(DEMO_KEY, {
    key: DEMO_KEY,
    title: 'Змейка',
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
    setFast: function (on) {
      if (fastMode !== !!on && modeBtn) modeBtn.click();
    },
    skipIntro: function () { try { finishIntro(); } catch (e) {} },
    showIntro: function () {
      try {
        introFinished = false; introStart = null;
        introOverlay.style.display = '';
        parker.showOnly(introFrame);
      } catch (e) {}
    },
    reset: function () { try { resetAll(); } catch (e) {} }
  });

  parker.resume();
})();
