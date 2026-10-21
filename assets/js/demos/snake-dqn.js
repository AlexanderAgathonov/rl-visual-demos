/* Автор: А. А. Агафонов (КФУ). Лицензия: MIT. Часть материалов к статье об изобразительных средствах обучения основам RL. */
/* ==== js/demos/snake-dqn.js (demo-kit/snake-dqn.html) ==== */
(function () {
  'use strict';

  /* ============================================================
     КОНФИГУРАЦИЯ
  ============================================================ */
  const DEMO_KEY = 'snake-dqn';
  const ACC = Engine.ACCENT_HEX;
  const METRIC_LABEL = 'Яблок';
  const GRID = 420; // внутренняя система координат обоих квадратных канвасов

  const gameCanvas = document.getElementById('snake-dqn-game');
  const g = gameCanvas.getContext('2d');
  const netCanvas = document.getElementById('snake-dqn-net');
  const nctx = netCanvas.getContext('2d');
  const statsEl = document.getElementById('snake-dqn-stats');
  const modeBtn = document.getElementById('snake-dqn-modeBtn');
  const infoBtn = document.getElementById('snake-dqn-infoBtn');
  const resetBtn = document.getElementById('snake-dqn-resetBtn');
  const weightsBtn = document.getElementById('snake-dqn-weightsBtn');
  const modalOverlay = document.getElementById('snake-dqn-modalOverlay');
  const modalClose = document.getElementById('snake-dqn-modalClose');
  const modalOk = document.getElementById('snake-dqn-modalOk');

  const N = 20;
  const CELL = GRID / N;
  const BLINK_MS = 600;

  /* ============================================================
     СЕТЬ 7 → 16 (tanh) → 3 (линейный)
     Входы:
       0..2 — опасность впереди / слева / справа (луч до 6 клеток)
       3..4 — непрерывное направление на еду (cos, sin) в локальных
              координатах головы
       5    — близость еды = 1/(1+манхэттен)
       6    — голод = min(1, шаговБезЕды/100)
  ============================================================ */
  const IN = 7, HID = 16, OUT = 3;
  const INPUT_LABELS = [
    ['опасн.', 'прямо'],
    ['опасн.', 'слева'],
    ['опасн.', 'справа'],
    ['еда', 'cos'],
    ['еда', 'sin'],
    ['близость', 'еды'],
    ['голод', '']
  ];
  const OUTPUT_LABELS = ['налево', 'прямо', 'направо'];

  /* ============================================================
     ГИПЕРПАРАМЕТРЫ
  ============================================================ */
  const FOOD_REWARD = 1;
  const DEATH_PENALTY = -10;
  const STEP_PENALTY = -0.005;
  const SHAPE = 0.1;
  const SHAPE_CLIP = 0.2;

  const GAMMA = 0.97;
  const BATCH = 32;
  const BUFFER_MAX = 10000;
  const N_STEP = 3;
  const TARGET_UPDATE_INTERVAL = 500;
  const MAX_STEPS = 3000;        // защита от затянутого эпизода
  const MAX_HUNGER_STEPS = 200;  // шагов без еды — терминал

  const EPS_START = 0.5, EPS_END = 0.05, EPS_DECAY = 0.995;
  const ADAM_LR = 1e-3, ADAM_B1 = 0.9, ADAM_B2 = 0.999, ADAM_EPS = 1e-8;

  const FAST_STEPS = 500;
  const NORMAL_STEP_DELAY = 50;

  /* ============================================================
     СОСТОЯНИЕ
  ============================================================ */
  let main, target;
  let snake = [];
  let dir = 1;
  let food = { x: 0, y: 0 };

  let score = 0, bestScore = 0;
  let episodeReward = 0, episodeCount = 0, episodeSteps = 0;
  let stepsWithoutFood = 0;

  let currentState = null;
  let lastAction = 1, lastActionRandom = false;
  let lastEvent = '', lastNotableEvent = '';
  let stepCounter = 0, lastLoss = 0, trainCount = 0;

  let fastMode = false, mainStarted = false;

  const buffer = [];
  const pending = [];
  let adamMain = null;

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  const fmt1 = Engine.fmt1;
  const fmt2 = Engine.fmt;

  /* ============================================================
     СЕТЬ
  ============================================================ */
  function randomNet() {
    const s1 = Math.sqrt(2.0 / IN), s2 = Math.sqrt(2.0 / HID);
    return {
      W1: Array.from({ length: HID }, () => Array.from({ length: IN }, () => (Math.random() - 0.5) * s1 * 2)),
      b1: Array.from({ length: HID }, () => (Math.random() - 0.5) * 0.2),
      W2: Array.from({ length: OUT }, () => Array.from({ length: HID }, () => (Math.random() - 0.5) * s2 * 2)),
      b2: Array.from({ length: OUT }, () => (Math.random() - 0.5) * 0.2)
    };
  }
  function copyNet(n) {
    return { W1: n.W1.map(r => r.slice()), b1: n.b1.slice(), W2: n.W2.map(r => r.slice()), b2: n.b2.slice() };
  }
  function forwardNet(net, s) {
    const h = new Array(HID);
    for (let j = 0; j < HID; j++) {
      let sum = net.b1[j];
      for (let i = 0; i < IN; i++) sum += net.W1[j][i] * s[i];
      h[j] = Math.tanh(sum);
    }
    const q = new Array(OUT);
    for (let a = 0; a < OUT; a++) {
      let sum = net.b2[a];
      for (let j = 0; j < HID; j++) sum += net.W2[a][j] * h[j];
      q[a] = sum;
    }
    return { h, q };
  }
  function argmaxQ(q) { let a = 0; for (let i = 1; i < q.length; i++) if (q[i] > q[a]) a = i; return a; }

  function currentEpsilon() { return Math.max(EPS_END, EPS_START * Math.pow(EPS_DECAY, episodeCount)); }
  function chooseAction(net, s) {
    if (Math.random() < currentEpsilon()) { lastActionRandom = true; return Math.floor(Math.random() * OUT); }
    lastActionRandom = false;
    return argmaxQ(forwardNet(net, s).q);
  }

  /* ============================================================
     N-STEP ПАМЯТЬ
  ============================================================ */
  function flushPending() {
    while (pending.length > 0) {
      const head = pending[0];
      let upto = 0, lastDone = head.done;
      if (head.done) {
        upto = 1;
      } else {
        const limit = Math.min(N_STEP, pending.length);
        let firstTerm = -1;
        for (let i = 0; i < limit; i++) { if (pending[i].done) { firstTerm = i; break; } }
        if (firstTerm >= 0) { upto = firstTerm + 1; lastDone = true; }
        else if (pending.length >= N_STEP) { upto = N_STEP; }
        else break;
      }
      let cumR = 0, gamma = 1, lastS2 = head.s2;
      for (let i = 0; i < upto; i++) {
        cumR += gamma * pending[i].r;
        gamma *= GAMMA;
        lastS2 = pending[i].s2;
        if (pending[i].done) break;
      }
      buffer.push({ s: head.s, a: head.a, r: cumR, s2: lastS2, done: lastDone, n: upto });
      if (buffer.length > BUFFER_MAX) buffer.shift();
      pending.splice(0, upto);
    }
  }
  function remember(s, a, r, s2, done) {
    pending.push({ s, a, r, s2, done });
    flushPending();
  }

  /* ============================================================
     ADAM
  ============================================================ */
  function makeAdam(net) {
    const z = r => r.map(() => 0);
    return {
      mW1: net.W1.map(z), mb1: net.b1.map(() => 0),
      mW2: net.W2.map(z), mb2: net.b2.map(() => 0),
      vW1: net.W1.map(z), vb1: net.b1.map(() => 0),
      vW2: net.W2.map(z), vb2: net.b2.map(() => 0),
      t: 0
    };
  }
  function adamApply(net, opt, grads, lr) {
    opt.t++;
    const bc1 = 1 - Math.pow(ADAM_B1, opt.t);
    const bc2 = 1 - Math.pow(ADAM_B2, opt.t);
    function stepOne(p, m, v, gr) {
      const mN = ADAM_B1 * m + (1 - ADAM_B1) * gr;
      const vN = ADAM_B2 * v + (1 - ADAM_B2) * gr * gr;
      return [p - lr * (mN / bc1) / (Math.sqrt(vN / bc2) + ADAM_EPS), mN, vN];
    }
    for (let j = 0; j < HID; j++) {
      let r = stepOne(net.b1[j], opt.mb1[j], opt.vb1[j], grads.gb1[j]);
      net.b1[j] = r[0]; opt.mb1[j] = r[1]; opt.vb1[j] = r[2];
      for (let i = 0; i < IN; i++) {
        r = stepOne(net.W1[j][i], opt.mW1[j][i], opt.vW1[j][i], grads.gW1[j][i]);
        net.W1[j][i] = r[0]; opt.mW1[j][i] = r[1]; opt.vW1[j][i] = r[2];
      }
    }
    for (let a = 0; a < OUT; a++) {
      let r = stepOne(net.b2[a], opt.mb2[a], opt.vb2[a], grads.gb2[a]);
      net.b2[a] = r[0]; opt.mb2[a] = r[1]; opt.vb2[a] = r[2];
      for (let j = 0; j < HID; j++) {
        r = stepOne(net.W2[a][j], opt.mW2[a][j], opt.vW2[a][j], grads.gW2[a][j]);
        net.W2[a][j] = r[0]; opt.mW2[a][j] = r[1]; opt.vW2[a][j] = r[2];
      }
    }
  }

  /* ============================================================
     ОБУЧЕНИЕ: Double DQN + n-step + Adam
  ============================================================ */
  function trainBatch(bs) {
    if (buffer.length < bs) return;
    const gW1 = Array.from({ length: HID }, () => new Array(IN).fill(0)), gb1 = new Array(HID).fill(0);
    const gW2 = Array.from({ length: OUT }, () => new Array(HID).fill(0)), gb2 = new Array(OUT).fill(0);
    let loss = 0;
    for (let n = 0; n < bs; n++) {
      const t = buffer[Math.floor(Math.random() * buffer.length)];
      const f = forwardNet(main, t.s);
      let tgt = t.r;
      if (!t.done && t.s2) {
        const aStar = argmaxQ(forwardNet(main, t.s2).q);
        const tq = forwardNet(target, t.s2).q;
        tgt += Math.pow(GAMMA, t.n || 1) * tq[aStar];
      }
      const err = clamp(f.q[t.a] - tgt, -4, 4);
      loss += err * err;
      const dOut = new Array(OUT).fill(0); dOut[t.a] = err;
      for (let a = 0; a < OUT; a++) { gb2[a] += dOut[a]; for (let j = 0; j < HID; j++) gW2[a][j] += dOut[a] * f.h[j]; }
      const dH = new Array(HID).fill(0);
      for (let j = 0; j < HID; j++) {
        let b = 0; for (let a = 0; a < OUT; a++) b += main.W2[a][j] * dOut[a];
        dH[j] = b * (1 - f.h[j] * f.h[j]);
      }
      for (let j = 0; j < HID; j++) { gb1[j] += dH[j]; for (let i = 0; i < IN; i++) gW1[j][i] += dH[j] * t.s[i]; }
    }
    const inv = 1 / bs;
    for (let a = 0; a < OUT; a++) { gb2[a] *= inv; for (let j = 0; j < HID; j++) gW2[a][j] *= inv; }
    for (let j = 0; j < HID; j++) { gb1[j] *= inv; for (let i = 0; i < IN; i++) gW1[j][i] *= inv; }
    adamApply(main, adamMain, { gW1, gb1, gW2, gb2 }, ADAM_LR);
    lastLoss = loss * inv;
    trainCount++;
    if (trainCount % TARGET_UPDATE_INTERVAL === 0) target = copyNet(main);
  }

  /* ============================================================
     ЛОГИКА ИГРЫ
  ============================================================ */
  const DIRS = [{ x: 0, y: -1 }, { x: 1, y: 0 }, { x: 0, y: 1 }, { x: -1, y: 0 }];

  function outside(x, y) { return x < 0 || x >= N || y < 0 || y >= N; }
  function manhattan(a, b) { return Math.abs(a.x - b.x) + Math.abs(a.y - b.y); }
  function isBody(x, y, excludeTail) {
    for (let i = 0; i < snake.length; i++) {
      if (excludeTail && i === snake.length - 1) continue;
      if (snake[i].x === x && snake[i].y === y) return true;
    }
    return false;
  }
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
    currentState = getState();
  }

  /* Луч опасности: 1/dist, если препятствие/стена в пределах 6 клеток; иначе 0. */
  function rayDanger(d) {
    const head = snake[0];
    const v = DIRS[d];
    let x = head.x + v.x, y = head.y + v.y, dist = 1;
    while (dist <= 6) {
      if (outside(x, y)) return 1 / dist;
      if (isBody(x, y, true)) return 1 / dist;
      x += v.x; y += v.y; dist++;
    }
    return 0;
  }

  /* Непрерывное направление на еду в локальной системе головы:
     (cos, sin) единичного вектора «голова → еда», ось X — «вперёд»,
     ось Y — «вправо». */
  function foodDirectionVector() {
    if (food.x < 0) return [0, 0];
    const head = snake[0];
    const f = DIRS[dir], r = DIRS[(dir + 1) % 4];
    const dx = food.x - head.x, dy = food.y - head.y;
    const ahead = dx * f.x + dy * f.y;
    const right = dx * r.x + dy * r.y;
    const len = Math.hypot(ahead, right);
    if (len < 1e-6) return [0, 0];
    return [ahead / len, right / len];
  }
  function foodProximity() {
    if (food.x < 0) return 0;
    return 1 / (1 + manhattan(snake[0], food));
  }
  function getState() {
    const fd = foodDirectionVector();
    return [
      rayDanger(dir),
      rayDanger((dir + 3) % 4),
      rayDanger((dir + 1) % 4),
      fd[0], fd[1],
      foodProximity(),
      clamp(stepsWithoutFood / 100, 0, 1)
    ];
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
    let reward = STEP_PENALTY;
    let result = 'шаг';

    if (willEat) {
      score++; stepsWithoutFood = 0;
      reward += FOOD_REWARD;
      result = 'яблоко';
      if (score > bestScore) bestScore = score;
      if (snake.length >= N * N) { reward += 50; return { reward, done: true, result: 'победа' }; }
      placeFood();
    } else {
      snake.pop();
      stepsWithoutFood++;
      if (food.x >= 0) {
        const newDist = manhattan(newHead, food);
        let shaping = SHAPE * (oldDist - newDist);
        shaping = clamp(shaping, -SHAPE_CLIP, SHAPE_CLIP);
        reward += shaping;
      }
    }
    return { reward, done: false, result };
  }
  function stepOnce() {
    const s = currentState;
    const a = chooseAction(main, s);
    lastAction = a;
    const res = stepGame(a);
    episodeReward += res.reward;
    lastEvent = res.result;

    let done = res.done;
    let result = res.result;
    episodeSteps++;
    if (!done && episodeSteps >= MAX_STEPS) { done = true; result = 'время'; }
    if (!done && stepsWithoutFood >= MAX_HUNGER_STEPS) { done = true; result = 'голод'; }

    if (result !== 'шаг') lastNotableEvent = result;

    const s2 = done ? null : getState();
    remember(s, a, res.reward, s2, done);
    stepCounter++;
    const trainEvery = fastMode ? 20 : 4;
    if (stepCounter % trainEvery === 0) trainBatch(BATCH);

    if (done) { episodeCount++; resetGame(); }
    else currentState = s2;
  }

  /* ============================================================
     РЕНДЕР: ИГРА (внутренняя сетка GRID×GRID, масштаб — в
     resize-колбэке, как во всех остальных демо)
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
     РЕНДЕР: СЕТЬ
     Толщина/прозрачность линии нормируется не константой, а
     реальным максимумом |веса| в текущей сети на этот кадр — у
     этой сети, в отличие от racing/dino, веса не клампятся жёстко,
     поэтому фиксированный потолок мог бы либо рано насыщаться,
     либо быть завышен большую часть обучения.
  ============================================================ */
  function drawEdge(c, x1, y1, x2, y2, w, wmax) {
    const aw = Math.abs(w);
    if (aw < 0.02) return;
    const t = Math.sqrt(Math.min(1, aw / wmax));
    c.save();
    c.strokeStyle = '#000';
    c.globalAlpha = 0.1 + 0.8 * t;
    c.lineWidth = 0.4 + 2.0 * t;
    if (w < 0) c.setLineDash([2, 2]);
    c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2, y2); c.stroke();
    c.restore();
  }
  function netWMax() {
    let m = 0.05;
    for (let j = 0; j < HID; j++) {
      const ab1 = Math.abs(main.b1[j]); if (ab1 > m) m = ab1;
      for (let i = 0; i < IN; i++) { const v = Math.abs(main.W1[j][i]); if (v > m) m = v; }
    }
    for (let a = 0; a < OUT; a++) {
      const ab2 = Math.abs(main.b2[a]); if (ab2 > m) m = ab2;
      for (let j = 0; j < HID; j++) { const v = Math.abs(main.W2[a][j]); if (v > m) m = v; }
    }
    return m;
  }
  function drawNet() {
    nctx.clearRect(0, 0, GRID, GRID);
    const inputX = 118, hiddenX = 225, outputX = 312;
    const inputY = i => 40 + i * 52;
    const hiddenY = j => 26 + j * 24;
    const outputY = a => 80 + a * 130;
    const f = forwardNet(main, currentState);
    const wmax = netWMax();

    for (let j = 0; j < HID; j++)
      for (let i = 0; i < IN; i++)
        drawEdge(nctx, inputX + 8, inputY(i), hiddenX - 8, hiddenY(j), main.W1[j][i], wmax);
    for (let a = 0; a < OUT; a++)
      for (let j = 0; j < HID; j++)
        drawEdge(nctx, hiddenX + 8, hiddenY(j), outputX - 9, outputY(a), main.W2[a][j], wmax);

    for (let i = 0; i < IN; i++) {
      const x = inputX, y = inputY(i);
      const v = clamp(Math.abs(currentState[i] || 0), 0, 1);
      nctx.beginPath(); nctx.arc(x, y, 7, 0, Math.PI * 2);
      nctx.fillStyle = `rgba(0,0,0,${0.08 + 0.9 * v})`; nctx.fill();
      nctx.strokeStyle = '#000'; nctx.lineWidth = 1; nctx.stroke();
      nctx.fillStyle = '#000'; nctx.font = '10px ' + Engine.FONT_MONO; nctx.textAlign = 'right';
      if (INPUT_LABELS[i][1]) {
        nctx.fillText(INPUT_LABELS[i][0], inputX - 12, y - 2);
        nctx.fillText(INPUT_LABELS[i][1], inputX - 12, y + 10);
      } else {
        nctx.fillText(INPUT_LABELS[i][0], inputX - 12, y + 4);
      }
    }
    for (let j = 0; j < HID; j++) {
      const x = hiddenX, y = hiddenY(j);
      const v = clamp(Math.abs(f.h[j] || 0), 0, 1);
      nctx.beginPath(); nctx.arc(x, y, 6, 0, Math.PI * 2);
      nctx.fillStyle = `rgba(0,0,0,${0.05 + 0.8 * v})`; nctx.fill();
      nctx.strokeStyle = '#000'; nctx.lineWidth = 1; nctx.stroke();
    }
    const mn = Math.min(...f.q), mx = Math.max(...f.q), rg = mx - mn;
    for (let a = 0; a < OUT; a++) {
      const x = outputX, y = outputY(a), norm = rg > 1e-6 ? (f.q[a] - mn) / rg : 0.5;
      nctx.beginPath(); nctx.arc(x, y, 7, 0, Math.PI * 2);
      nctx.fillStyle = `rgba(0,0,0,${0.08 + 0.85 * norm})`; nctx.fill();
      nctx.lineWidth = a === lastAction ? 2.5 : 1;
      if (a === lastAction && lastActionRandom) nctx.setLineDash([3, 2]);
      nctx.strokeStyle = a === lastAction ? Engine.ACCENT_HEX : '#000'; nctx.stroke(); nctx.setLineDash([]);
      nctx.fillStyle = '#000'; nctx.font = '10px ' + Engine.FONT_MONO; nctx.textAlign = 'left';
      nctx.fillText(OUTPUT_LABELS[a] + ' ' + fmt1(f.q[a]), outputX + 16, y + 4);
    }
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
      '<span>ε: <span class="v">' + Engine.fmtFixed(currentEpsilon(), 1, 3) + '</span></span>';
  }

  /* ============================================================
     ГЛАВНЫЙ ЦИКЛ + ПАРКОВЩИК rAF
  ============================================================ */
  let lastNormalStep = 0;
  function mainLoop(ts) {
    if (mainStarted) {
      if (fastMode) { for (let i = 0; i < FAST_STEPS; i++) stepOnce(); }
      else if (ts - lastNormalStep >= NORMAL_STEP_DELAY) { stepOnce(); lastNormalStep = ts; }
      drawGame(); drawNet(); drawStats();
    }
    parker.raf(mainLoop);
  }
  function introFrame(ts) {
    if (introFinished) return;
    if (introStart === null) introStart = ts;
    const t = (ts - introStart) / 1000;
    drawIntro(t);
    if (t >= 31.0) { finishIntro(); return; }
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
    main = randomNet();
    target = copyNet(main);
    adamMain = makeAdam(main);
    bestScore = 0;
    episodeCount = 0; stepCounter = 0; trainCount = 0; lastLoss = 0;
    lastNotableEvent = '';
    buffer.length = 0; pending.length = 0;
    curve.reset();
    resetGame();
    drawGame(); drawNet(); drawStats();
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
     ЗАСТАВКА
  ============================================================ */
  const introOverlay = document.getElementById('snake-dqn-introOverlay');
  const introCanvas = document.getElementById('snake-dqn-introCanvas');
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

  const IC = 66;
  const COLS = 6, ROWS = 6;
  const SX = 10, SY = 50;
  const HEAD = { x: 3, y: 2 };
  const SNAKE_BODY = [{ x: 3, y: 3 }, { x: 3, y: 4 }, { x: 3, y: 5 }];
  const APPLE = { x: 5, y: 1 };

  const CAPS = [
    [0.0, 2.5, 'Та же змейка, что и в демо с таблицей: поле 20×20, яблоко, стена.'],
    [2.5, 5.5, 'Три действия: налево, прямо, направо.'],
    [5.5, 10.0, 'Луч опасности вперёд: 1/расстояние, если препятствие близко.'],
    [10.0, 13.0, 'Лучи влево и вправо — то же самое.'],
    [13.0, 17.0, 'Направление на еду (cos, sin) в системе головы — раньше был 1 из 4 секторов, теперь точное число.'],
    [17.0, 21.0, 'Близость еды = 1 / (1 + расстояние).'],
    [21.0, 24.0, 'Голод — сколько шагов змейка не ела.'],
    [24.0, 29.2, 'Вариантов бесконечно много — таблицей не обойтись. Сеть 7 → 16 → 3 выбирает действие сама.'],
    [29.2, 31.5, 'Обучение с подкреплением начинается…']
  ];
  const ROWS_INFO = [
    { label: 'опасность прямо', value: '0.33', appear: 8.8, win: [5.5, 10.0] },
    { label: 'опасность слева', value: '0', appear: 11.3, win: [10.0, 11.4] },
    { label: 'опасность справа', value: '0', appear: 12.7, win: [11.4, 12.9] },
    { label: 'еда cos', value: '+0.97', appear: 14.0, win: [13.0, 17.0] },
    { label: 'еда sin', value: '−0.24', appear: 14.6, win: [13.0, 17.0] },
    { label: 'близость еды', value: '0.25', appear: 20.0, win: [17.0, 21.0] },
    { label: 'голод', value: '0.12', appear: 22.6, win: [21.0, 24.0] }
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
    c.strokeRect(SX - 0.5, SY - 0.5, COLS * IC + 1, ROWS * IC + 1);
    c.setLineDash([]);

    c.strokeStyle = '#ddd';
    for (let i = 1; i < COLS; i++) { c.beginPath(); c.moveTo(SX + i * IC + 0.5, SY); c.lineTo(SX + i * IC + 0.5, SY + ROWS * IC); c.stroke(); }
    for (let j = 1; j < ROWS; j++) { c.beginPath(); c.moveTo(SX, SY + j * IC + 0.5); c.lineTo(SX + COLS * IC, SY + j * IC + 0.5); c.stroke(); }

    c.fillStyle = '#000';
    c.fillRect(SX - 4, SY - 16, COLS * IC + 8, 14);
    c.fillStyle = '#fff'; c.font = '11px Inter, sans-serif'; c.textAlign = 'center';
    c.fillText('стена', SX + COLS * IC / 2, SY - 5);

    const ax = SX + APPLE.x * IC, ay = SY + APPLE.y * IC;
    const blink = Math.floor(t * 1000 / BLINK_MS) % 2 === 0;
    if (blink) { c.fillStyle = ACC; c.fillRect(ax, ay, IC, IC); }
    else { c.strokeStyle = ACC; c.lineWidth = 2; c.strokeRect(ax + 1.5, ay + 1.5, IC - 3, IC - 3); }

    const segs = [HEAD, ...SNAKE_BODY];
    segs.forEach((s, i) => {
      c.fillStyle = i === 0 ? ACC : '#333';
      c.fillRect(SX + s.x * IC + 1, SY + s.y * IC + 1, IC - 2, IC - 2);
    });
    c.restore();
  }
  function drawIntroActions(c, t) {
    const p = ease(prog(t, 2.6, 3.5));
    const fade = 1 - prog(t, 5.6, 6.4);
    const al = Math.min(p, fade);
    if (al <= 0) return;
    c.save(); c.globalAlpha = al;
    const hx = icx(HEAD.x), hy = icy(HEAD.y);
    const Lfull = 1.25 * IC;
    const L = Lfull * p;
    c.strokeStyle = '#000'; c.fillStyle = '#000'; c.lineWidth = 2;
    drawIntroArrow(c, hx, hy, hx, hy - L);
    drawIntroArrow(c, hx, hy, hx - L, hy);
    drawIntroArrow(c, hx, hy, hx + L, hy);
    c.font = '12px Inter, sans-serif';
    c.textAlign = 'left'; c.fillText('вперёд', hx + 8, hy - Lfull - 8);
    c.textAlign = 'right'; c.fillText('лево', hx - Lfull - 8, hy + 4);
    c.textAlign = 'left'; c.fillText('право', hx + Lfull + 8, hy + 4);
    c.restore();
  }
  function drawIntroRays(c, t) {
    const hx = icx(HEAD.x), hy = icy(HEAD.y);

    if (t >= 5.8 && t < 10.6) {
      const grow = ease(prog(t, 5.8, 8.4));
      const fade = 1 - prog(t, 10.0, 10.6);
      c.save(); c.globalAlpha = fade;
      const L = 3 * IC * grow;
      c.strokeStyle = ACC; c.lineWidth = 2;
      c.setLineDash([5, 4]);
      c.beginPath(); c.moveTo(hx, hy); c.lineTo(hx, hy - L); c.stroke();
      c.setLineDash([]);
      c.fillStyle = ACC;
      c.beginPath(); c.arc(hx, hy - L, 3, 0, Math.PI * 2); c.fill();
      c.fillStyle = '#000'; c.font = '12px Inter, sans-serif'; c.textAlign = 'right';
      if (grow < 1) c.fillText('пройдено: ' + Math.floor(grow * 3 + 0.05), hx - 10, hy - L + 4);
      else { c.fillText('стена!', hx - 10, SY + 24); c.fillText('опасность = 1/3 ≈ 0.33', hx - 10, SY + 40); }
      c.restore();
    }
    if (t >= 10.1 && t < 12.2) {
      const grow = ease(prog(t, 10.2, 11.0));
      const fade = 1 - prog(t, 11.6, 12.2);
      c.save(); c.globalAlpha = fade;
      const L = 3 * IC * grow;
      c.strokeStyle = ACC; c.lineWidth = 2;
      c.setLineDash([5, 4]);
      c.beginPath(); c.moveTo(hx, hy); c.lineTo(hx - L, hy); c.stroke();
      c.setLineDash([]);
      if (grow < 1) { c.fillStyle = '#000'; c.font = '12px Inter, sans-serif'; c.textAlign = 'left'; c.fillText('пройдено: ' + Math.floor(grow * 3 + 0.05), SX + 6, hy + 22); }
      c.restore();
    }
    if (t >= 11.0 && t < 12.4) {
      const fade = 1 - prog(t, 11.9, 12.4);
      c.save(); c.globalAlpha = fade;
      c.fillStyle = '#000'; c.font = '12px Inter, sans-serif'; c.textAlign = 'left';
      c.fillText('слева препятствий нет (луч до 6 клеток) → 0', SX + 6, SY + ROWS * IC + 17);
      c.restore();
    }
    if (t >= 11.5 && t < 13.4) {
      const grow = ease(prog(t, 11.6, 12.4));
      const fade = 1 - prog(t, 12.8, 13.4);
      c.save(); c.globalAlpha = fade;
      const L = 2 * IC * grow;
      c.strokeStyle = ACC; c.lineWidth = 2;
      c.setLineDash([5, 4]);
      c.beginPath(); c.moveTo(hx, hy); c.lineTo(hx + L, hy); c.stroke();
      c.setLineDash([]);
      if (grow < 1) { c.fillStyle = '#000'; c.font = '12px Inter, sans-serif'; c.textAlign = 'left'; c.fillText('пройдено: ' + Math.floor(grow * 2 + 0.05), 295, hy - 12); }
      c.restore();
    }
    if (t >= 12.5 && t < 14.4) {
      const fade = 1 - prog(t, 13.8, 14.4);
      c.save(); c.globalAlpha = fade;
      c.fillStyle = '#000'; c.font = '12px Inter, sans-serif'; c.textAlign = 'left';
      c.fillText('справа препятствий нет → 0', SX + 6, SY + ROWS * IC + 17);
      c.restore();
    }
  }
  function drawIntroFood(c, t) {
    const hx = icx(HEAD.x), hy = icy(HEAD.y);
    const ax = icx(APPLE.x), ay = icy(APPLE.y);

    if (t >= 13.0 && t < 17.2) {
      const fade = 1 - prog(t, 16.6, 17.2);
      c.save(); c.globalAlpha = fade;
      c.strokeStyle = ACC; c.lineWidth = 2;
      c.setLineDash([4, 4]);
      c.beginPath(); c.moveTo(hx, hy); c.lineTo(ax, ay); c.stroke();
      c.setLineDash([]);
      c.strokeStyle = '#999'; c.lineWidth = 1;
      c.beginPath(); c.moveTo(hx, hy); c.lineTo(hx, hy - 30); c.stroke();
      c.beginPath(); c.moveTo(hx, hy); c.lineTo(hx + 30, hy); c.stroke();
      c.fillStyle = '#999'; c.font = '10px Inter, sans-serif'; c.textAlign = 'center';
      c.fillText('вперёд', hx, hy - 36);
      c.textAlign = 'left';
      c.fillText('вправо', hx + 34, hy + 3);
      c.strokeStyle = ACC; c.fillStyle = ACC; c.lineWidth = 2;
      drawIntroArrow(c, hx, hy, hx + (ax - hx) * 0.55, hy + (ay - hy) * 0.55);
      c.fillStyle = '#000'; c.font = '12px Inter, sans-serif'; c.textAlign = 'left';
      c.fillText('cos ≈ +0.97 · sin ≈ −0.24', hx + 40, hy + 20);
      c.restore();
    }

    if (t >= 17.0 && t < 21.4) {
      const fade = 1 - prog(t, 20.8, 21.4);
      c.save(); c.globalAlpha = fade;
      const path = [{ x: 4, y: 2 }, { x: 5, y: 2 }, { x: 5, y: 1 }];
      path.forEach((cell, i) => {
        const p = prog(t, 17.4 + i * 0.8, 18.1 + i * 0.8);
        if (p <= 0) return;
        c.fillStyle = 'rgba(59,111,224,0.25)';
        c.fillRect(SX + cell.x * IC + 2, SY + cell.y * IC + 2, IC - 4, IC - 4);
        c.fillStyle = '#000'; c.font = '14px Inter, sans-serif'; c.textAlign = 'center';
        c.fillText(String(i + 1), icx(cell.x), icy(cell.y) + 5);
      });
      if (t > 19.9) {
        c.fillStyle = '#000'; c.font = '12px Inter, sans-serif'; c.textAlign = 'left';
        c.fillText('расстояние = 3 → близость = 1/(1+3) = 0.25', SX + 6, SY + ROWS * IC + 17);
      }
      c.restore();
    }
    if (t >= 21.0 && t < 24.2) {
      const fade = 1 - prog(t, 23.6, 24.2);
      const n = Math.floor(12 * ease(prog(t, 21.2, 22.4)));
      c.save(); c.globalAlpha = fade;
      c.fillStyle = '#000'; c.font = '12px Inter, sans-serif'; c.textAlign = 'left';
      c.fillText('шагов без еды: ' + n, SX + 6, SY + ROWS * IC + 10);
      if (t > 22.5) c.fillText('голод = 12/100 = 0.12', SX + 6, SY + ROWS * IC + 24);
      c.restore();
    }
  }
  function drawIntroPanel(c, t) {
    const x0 = 430;
    c.fillStyle = '#000'; c.font = '12px Inter, sans-serif'; c.textAlign = 'left';
    c.fillText('Вход нейросети:', x0, 84);

    ROWS_INFO.forEach(row => {
      const y = 110 + ROWS_INFO.indexOf(row) * 26;
      const active = t >= row.win[0] && t < row.win[1];
      if (active) { c.fillStyle = 'rgba(0,0,0,0.06)'; c.fillRect(x0 - 6, y - 13, 286, 20); }
      if (t >= row.appear) {
        const alpha = prog(t, row.appear, row.appear + 0.4);
        c.save(); c.globalAlpha = alpha;
        c.fillStyle = '#000'; c.font = '12px Inter, sans-serif';
        c.textAlign = 'left'; c.fillText(row.label, x0, y);
        c.textAlign = 'right'; c.fillText(row.value, x0 + 280, y);
        c.restore();
      }
    });

    const nnA = prog(t, 24.0, 25.0);
    if (nnA > 0) {
      c.save(); c.globalAlpha = nnA;
      drawIntroArrow(c, x0 + 140, 352, x0 + 140, 372);
      c.restore();
      drawIntroMiniNet(c, nnA);
    }
  }
  function drawIntroMiniNet(c, alpha) {
    c.save(); c.globalAlpha = alpha;
    const x1 = 470, x2 = 560, x3 = 660;
    const yIn = i => 345 + i * 10;
    const yHid = j => 340 + j * 8;
    const yOut = a => 365 + a * 22;
    c.strokeStyle = 'rgba(0,0,0,0.12)'; c.lineWidth = 1;
    for (let i = 0; i < 7; i++) for (let j = 0; j < 16; j++) { c.beginPath(); c.moveTo(x1, yIn(i)); c.lineTo(x2, yHid(j)); c.stroke(); }
    for (let j = 0; j < 16; j++) for (let a = 0; a < 3; a++) { c.beginPath(); c.moveTo(x2, yHid(j)); c.lineTo(x3, yOut(a)); c.stroke(); }
    c.fillStyle = '#000';
    for (let i = 0; i < 7; i++) { c.beginPath(); c.arc(x1, yIn(i), 3, 0, Math.PI * 2); c.fill(); }
    for (let j = 0; j < 16; j++) { c.beginPath(); c.arc(x2, yHid(j), 3, 0, Math.PI * 2); c.fill(); }
    for (let a = 0; a < 3; a++) { c.beginPath(); c.arc(x3, yOut(a), 3.5, 0, Math.PI * 2); c.fill(); }
    c.font = '12px Inter, sans-serif'; c.textAlign = 'center';
    c.fillText('7 → 16 → 3', (x1 + x3) / 2, 452);
    c.restore();
  }
  function drawIntro(t) {
    const c = ictx;
    c.clearRect(0, 0, 720, 480);
    c.fillStyle = '#fff'; c.fillRect(0, 0, 720, 480);
    const cap = CAPS.find(x => t >= x[0] && t < x[1]);
    if (cap) { c.fillStyle = '#000'; c.font = '14px Inter, sans-serif'; c.textAlign = 'left'; c.fillText(cap[2], 14, 24); }
    const sceneA = ease(prog(t, 0.2, 1.2));
    drawIntroScene(c, t, sceneA);
    drawIntroActions(c, t);
    drawIntroRays(c, t);
    drawIntroFood(c, t);
    drawIntroPanel(c, t);
    c.fillStyle = '#999'; c.font = '11px Inter, sans-serif'; c.textAlign = 'right';
    c.fillText('клик — пропустить', 712, 474);
    const fo = prog(t, 29.4, 30.7);
    if (fo > 0) { c.fillStyle = `rgba(255,255,255,${fo})`; c.fillRect(0, 0, 720, 480); }
    const ta = prog(t, 29.5, 30.1);
    if (ta > 0 && t < 30.9) {
      c.fillStyle = `rgba(0,0,0,${ta})`; c.font = '20px Inter, sans-serif'; c.textAlign = 'center';
      c.fillText('Обучение начинается…', 360, 240);
    }
  }
  let introStart = null;

  /* ============================================================
     АДАПТИВНЫЕ КАНВАСЫ (требование №3)
  ============================================================ */
  let gameScale = 1, netScale = 1, gameOX = 0, gameOY = 0, netOX = 0, netOY = 0;
  Engine.resizeCanvas(gameCanvas, { onResize: (w, h) => { gameScale = Math.min(w, h) / GRID; gameOX = (w - GRID * gameScale) / 2; gameOY = (h - GRID * gameScale) / 2; } });
  Engine.resizeCanvas(netCanvas, { onResize: (w, h) => { netScale = Math.min(w, h) / GRID; netOX = (w - GRID * netScale) / 2; netOY = (h - GRID * netScale) / 2; } });
  const rawDrawGame = drawGame, rawDrawNet = drawNet;
  drawGame = function () { g.save(); g.translate(gameOX, gameOY); g.scale(gameScale, gameScale); g.beginPath(); g.rect(0, 0, GRID, GRID); g.clip(); rawDrawGame(); g.lineWidth = 3 / gameScale; g.strokeStyle = '#1a1c20'; g.strokeRect(0, 0, GRID, GRID); g.restore(); };
  drawNet = function () { nctx.save(); nctx.translate(netOX, netOY); nctx.scale(netScale, netScale); rawDrawNet(); nctx.restore(); };

  const curve = Engine.makeCurve(document.getElementById('snake-dqn-curve'), {
    primaryLabel: 'рекорд яблок', secondaryLabel: 'награда эп.'
  });
  Engine.resizeCanvas(document.getElementById('snake-dqn-curve'), {});

  /* ============================================================
     СТАРТ
  ============================================================ */
  main = randomNet();
  target = copyNet(main);
  adamMain = makeAdam(main);
  resetGame();
  drawGame(); drawNet(); drawStats();
  parker.raf(mainLoop);
  parker.raf(introFrame);

  /* ============================================================
     ЭКСПОРТ / ИМПОРТ ВЕСОВ
  ============================================================ */
  function exportWeights() { return { main: main }; }
  function importWeights(W) {
    try {
      main = W.main;
      target = copyNet(main);
      adamMain = makeAdam(main);
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
    title: 'Змейка · нейросеть',
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
