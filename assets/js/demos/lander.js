/* Автор: А. А. Агафонов (КФУ). Лицензия: MIT. Часть материалов к статье об изобразительных средствах обучения основам RL. */
/* ==== js/demos/lander.js (demo-kit/lunar.html) ==== */
(function () {
  'use strict';

  /* ============================================================
  КОНФИГУРАЦИЯ ДЕМО
  ============================================================ */
  const DEMO_KEY = 'lander';
  const ACC = Engine.ACCENT_HEX;
  const MONO = Engine.FONT_MONO;
  const GRID = 420;
  const METRIC_LABEL = 'Посадок';

  /* ============================================================
  DOM
  ============================================================ */
  const gameCanvas = document.getElementById('lander-game');
  const g          = gameCanvas.getContext('2d');
  const netCanvas  = document.getElementById('lander-net');
  const nctx       = netCanvas.getContext('2d');
  const statsEl    = document.getElementById('lander-stats');
  const speedBtn   = document.getElementById('lander-speedBtn');
  const infoBtn    = document.getElementById('lander-infoBtn');
  const resetBtn   = document.getElementById('lander-resetBtn');
  const weightsBtn = document.getElementById('lander-weightsBtn');
  const modalOverlay = document.getElementById('lander-modalOverlay');
  const modalClose   = document.getElementById('lander-modalClose');
  const modalOk      = document.getElementById('lander-modalOk');

  /* ============================================================
  МИР: 40 × 40 м → внутренняя система координат канваса GRID×GRID
  (10.5 px/м), реальный размер задаёт CSS/ResizeObserver
  ============================================================ */
  const GW = GRID, GH = GRID;
  const WORLD = 40;
  const S = GW / WORLD;                 // 10.5 px/м
  const M = 25;                         // узлов высотной карты
  const GRAV = -12;
  const THR_MAIN = 26;
  const ANG_ACC = 5;
  const LAT_ACC = 2.0;
  const DT = 1 / 50;
  const MAX_STEPS = 350;
  const V_SAFE = 4, H_SAFE = 3.0, ANG_SAFE = 0.35, OMEGA_SAFE = 2.0;
  const WIND_BY_STAGE = [0, 0.5, 1.0];
  const PAD_HALF_BY_STAGE = [5, 4, 3];
  const ROUGH_BY_STAGE = [1.2, 2.0, 2.8];

  /* ============================================================
  СЕТЬ 8 → 32 (tanh) → 4 (linear)
  ============================================================ */
  const IN = 8, HID = 32, OUT = 4;
  const INPUT_LABELS = [
    ['до пад.', 'по X'],
    ['высота', 'над пад.'],
    ['радар', 'до грунта'],
    ['скор.', 'по X'],
    ['скор.', 'по Y'],
    ['наклон', ''],
    ['вращ.', ''],
    ['ветер', '']
  ];
  const OUTPUT_LABELS = ['выкл', 'главный', 'влево', 'вправо'];

  /* ============================================================
  ГИПЕРПАРАМЕТРЫ RL
  ============================================================ */
  const GAMMA = 0.99;
  const BATCH = 64;
  const BUFFER_MAX = 20000;
  const N_STEP = 3;
  const TARGET_UPDATE_INTERVAL = 500;
  const LEARN_START = 500;
  /* ε по эпизодам: исследование живёт достаточно долго,
     чтобы сеть успела увидеть посадки */
  const EPS_FLOOR = 0.05;
  const EPS_DECAY_EP = 1200;
  const BOOST_EPS = 0.30;
  const BOOST_EP = 200;
  /* награды и формирование */
  const SHAPE_K = 25;          // усилитель потенциального шейпинга
  const SHAPE_CLIP = 4;        // кламп шейпинга за шаг
  const TIMEOUT_PENALTY = -25; // зависание до таймаута невыгодно
  const TERM_BONUS = 12;       // бонус за завершение эпизода близ площадки
  const TERM_BONUS_POT = 2.5;  // нормировка потенциала в бонусе
  const GATE_H = 8;            // «ворота»: над площадкой ниже этой высоты
  const GATE_DESCEND_LO = -4;  // окно правильного снижения (vy)
  const GATE_DESCEND_HI = -0.6;
  const GATE_DESCEND_BONUS = 0.2;
  const GATE_HOVER_PEN = 0.15;
  const GATE_HOVER_MIN_H = 1.5;
  const ADAM_LR = 1e-3, ADAM_B1 = 0.9, ADAM_B2 = 0.999, ADAM_EPS = 1e-8;
  const EVAL_EVERY = 10;
  const EVAL_WINDOW = 8;
  const PROMOTE_RATE = 0.75;
  const FAST_STEPS = 300;
  const NORMAL_STEP_DELAY = 20;

  let main, target, bestNet;
  let adamMain = null;

  /* ============================================================
  СОСТОЯНИЕ МИРА
  ============================================================ */
  let heights = [];
  let pad = { x1: 15, x2: 25, y: 10, cx: 20, half: 5 };
  let wind = 0;
  let stars = [];
  let ship = { x: 20, y: 30, vx: 0, vy: 0, angle: 0, omega: 0 };
  let trail = [];
  let particles = [];
  let flameMain = false, flameSide = 0;
  let score = 0;
  let bestReward = -Infinity;
  let episodeReward = 0, lastEpisodeReward = 0;
  let episodeCount = 0;
  let episodeSteps = 0;
  let curStage = 1;
  let boostUntilEp = BOOST_EP;
  let evalMode = false;
  let evalWindow = [];
  let currentState = null;
  let lastAction = 0, lastActionRandom = false;
  let lastEvent = '', lastNotableEvent = '';
  let stepCounter = 0, trainCount = 0;
  let fastMode = false, mainStarted = false;
  const buffer = [];
  const pending = [];
  const seedBase = (Math.random() * 1e9) | 0;

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  /* ============================================================
  ГЕНЕРАТОР УРОВНЯ
  ============================================================ */
  function groundOf(arr, x) {
    const u = clamp(x, 0, WORLD) / WORLD * (M - 1);
    const i = Math.floor(u), f = u - i;
    if (i >= M - 1) return arr[M - 1];
    return arr[i] * (1 - f) + arr[i + 1] * f;
  }
  function ground(x) { return groundOf(heights, x); }

  function genWorld() {
    const rnd = mulberry32(seedBase + episodeCount * 7919 + curStage * 104729);
    stars = [];
    for (let i = 0; i < 50; i++) stars.push({ x: rnd() * GW, y: rnd() * 240, r: rnd() < 0.3 ? 1.5 : 1 });
    const rough = ROUGH_BY_STAGE[curStage - 1];
    const base = 8 + rnd() * 3;
    const ph1 = rnd() * 6.283, ph2 = rnd() * 6.283, ph3 = rnd() * 6.283;
    heights = [];
    for (let i = 0; i < M; i++) {
      const u = i / (M - 1);
      let h = base
        + rough * 0.9 * Math.sin(u * 6.283 * 1.3 + ph1)
        + rough * 0.5 * Math.sin(u * 6.283 * 2.7 + ph2)
        + rough * 0.3 * Math.sin(u * 6.283 * 5.1 + ph3);
      heights.push(clamp(h, 3, 16));
    }
    const half = PAD_HALF_BY_STAGE[curStage - 1];
    const cx = 12 + rnd() * 16;
    const pl = groundOf(heights, cx);
    pad = { x1: cx - half, x2: cx + half, y: pl, cx: cx, half: half };
    for (let i = 0; i < M; i++) {
      const x = i * (WORLD / (M - 1));
      if (x >= pad.x1 && x <= pad.x2) heights[i] = pl;
      else {
        const d = x < pad.x1 ? pad.x1 - x : x - pad.x2;
        if (d < 3) heights[i] = pl + (heights[i] - pl) * (d / 3);
      }
    }
    wind = WIND_BY_STAGE[curStage - 1] * (rnd() < 0.5 ? -1 : 1);
    if (curStage === 1) wind = 0;
  }

  function resetEpisode() {
    genWorld();
    const r = Math.random;
    let x, y, vm;
    if (curStage === 1)      { x = pad.cx + (r() * 2 - 1) * 3; y = pad.y + 9 + r() * 4; vm = 1; }
    else if (curStage === 2) { x = 6 + r() * 28;               y = 24 + r() * 8;         vm = 3; }
    else                     { x = 4 + r() * 32;               y = 20 + r() * 12;        vm = 4.5; }
    if (y < ground(x) + 8) y = ground(x) + 10;
    ship = {
      x: x, y: y,
      vx: (r() * 2 - 1) * vm,
      vy: (r() * 2 - 1) * vm * 0.5,
      angle: (r() * 2 - 1) * 0.2,
      omega: (r() * 2 - 1) * 0.3
    };
    episodeReward = 0; episodeSteps = 0;
    trail.length = 0; particles.length = 0;
    flameMain = false; flameSide = 0;
    lastAction = 0;
    currentState = getState();
  }

  /* ============================================================
  НАБЛЮДЕНИЕ (8 чисел, нормы под мир 40 м)
  ============================================================ */
  function getState() {
    return [
      clamp((pad.cx - ship.x) / 20, -1, 1),
      clamp((ship.y - pad.y) / 25, -1, 1),
      clamp((ship.y - ground(ship.x)) / 15, 0, 1),
      clamp(ship.vx / 12, -1, 1),
      clamp(ship.vy / 12, -1, 1),
      clamp(ship.angle / Math.PI, -1, 1),
      clamp(ship.omega / 3, -1, 1),
      clamp(wind / 1.5, -1, 1)
    ];
  }
  /* Потенциал для формирования награды: расстояние + скорость + вращение */
  function potential(s) {
    return 1.2 * Math.abs(s[0])
         + 1.0 * clamp(s[1], 0, 1)
         + 0.8 * clamp(Math.hypot(s[3], s[4]), 0, 1)
         + 0.5 * Math.abs(s[6]);
  }
  /* Бонус за завершение эпизода вблизи площадки (в «воротах») */
  function termBonus() {
    const potEnd = potential(getState());
    return TERM_BONUS * (1 - clamp(potEnd / TERM_BONUS_POT, 0, 1));
  }

  /* ============================================================
  ФИЗИКА И ШАГ ИГРЫ
  ============================================================ */
  function bodyVectors() {
    const s = Math.sin(ship.angle), c = Math.cos(ship.angle);
    return { ux: s, uy: c, rx: c, ry: -s };
  }
  function contactPoints() {
    const c = Math.cos(ship.angle), s = Math.sin(ship.angle);
    const mk = (lx, ly, leg) => ({ x: ship.x + lx * c + ly * s, y: ship.y - lx * s + ly * c, leg: leg });
    return [
      mk(-2.75, -1.25, true), mk(2.75, -1.25, true),
      mk(-1.7, -1.0, false), mk(1.7, -1.0, false)
    ];
  }

  function stepGame(action) {
    const sBefore = currentState;
    const potB = potential(sBefore);
    const bv = bodyVectors();
    let ax = wind, ay = GRAV, torque = 0;
    flameMain = false; flameSide = 0;
    if (action === 1) { ax += THR_MAIN * bv.ux; ay += THR_MAIN * bv.uy; flameMain = true; }
    if (action === 2) { torque -= ANG_ACC; ax -= LAT_ACC * bv.rx; ay -= LAT_ACC * bv.ry; flameSide = -1; }
    if (action === 3) { torque += ANG_ACC; ax += LAT_ACC * bv.rx; ay += LAT_ACC * bv.ry; flameSide = 1; }
    ship.vx = clamp(ship.vx + ax * DT, -40, 40);
    ship.vy = clamp(ship.vy + ay * DT, -40, 40);
    ship.omega = clamp(ship.omega + torque * DT, -4, 4);
    ship.angle += ship.omega * DT;
    ship.x += ship.vx * DT;
    ship.y += ship.vy * DT;

    let reward = -0.02;
    if (action === 1) reward -= 0.12;
    if (action === 2 || action === 3) reward -= 0.04;

    if (ship.x < -2 || ship.x > WORLD + 2) {
      return { reward: reward - 30, done: true, result: 'край', s2: null };
    }
    if (ship.y > WORLD + 6) {
      return { reward: reward - 30, done: true, result: 'высота', s2: null };
    }

    const pts = contactPoints();
    let bodyHit = false, legPt = null;
    for (const p of pts) {
      if (p.y <= ground(p.x)) {
        if (p.leg) { if (!legPt) legPt = p; }
        else bodyHit = true;
      }
    }
    if (bodyHit) {
      return { reward: reward - 100 + termBonus(), done: true, result: 'авария', s2: null };
    }
    if (legPt) {
      const onPad = legPt.x >= pad.x1 - 1 && legPt.x <= pad.x2 + 1;
      const ok = onPad
        && Math.abs(ship.vy) <= V_SAFE
        && Math.abs(ship.vx) <= H_SAFE
        && Math.abs(ship.angle) <= ANG_SAFE
        && Math.abs(ship.omega) <= OMEGA_SAFE;
      if (ok) {
        const soft = 30 * (1 - Math.min(1, Math.abs(ship.vy) / V_SAFE));
        const cent = 10 * (1 - Math.min(1, Math.abs(legPt.x - pad.cx) / pad.half));
        return { reward: reward + 100 + soft + cent, done: true, result: 'посадка', s2: null };
      }
      return { reward: reward - 100 + termBonus(), done: true, result: onPad ? 'жёстко' : 'авария', s2: null };
    }

    /* «ворота»: над площадкой ниже 8 м —
       плавное снижение поощряем, висение штрафуем */
    const hGate = ship.y - pad.y;
    if (ship.x >= pad.x1 && ship.x <= pad.x2 && hGate > 0 && hGate < GATE_H) {
      if (ship.vy <= GATE_DESCEND_HI && ship.vy >= GATE_DESCEND_LO) reward += GATE_DESCEND_BONUS;
      else if (Math.abs(ship.vy) < 0.6 && hGate > GATE_HOVER_MIN_H) reward -= GATE_HOVER_PEN;
    }

    if (hGate < 12) reward -= 0.02 * clamp(Math.abs(ship.angle) / 0.5, 0, 1);

    const sAfter = getState();
    const potA = potential(sAfter);
    reward += clamp(SHAPE_K * (potB - GAMMA * potA), -SHAPE_CLIP, SHAPE_CLIP);

    episodeSteps++;
    let done = false, result = 'шаг';
    if (episodeSteps >= MAX_STEPS) {
      done = true; result = 'время';
      reward += TIMEOUT_PENALTY + termBonus();
    }
    return { reward: reward, done: done, result: result, s2: sAfter };
  }

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
  function currentEpsilon() {
    let eps = Math.max(EPS_FLOOR, 1 - episodeCount / EPS_DECAY_EP);
    if (episodeCount < boostUntilEp) eps = Math.max(eps, BOOST_EPS);
    return eps;
  }
  function chooseAction(net, s) {
    if (evalMode) return argmaxQ(forwardNet(net, s).q);
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
      if (head.done) { upto = 1; }
      else {
        const limit = Math.min(N_STEP, pending.length);
        let firstTerm = -1;
        for (let i = 0; i < limit; i++) if (pending[i].done) { firstTerm = i; break; }
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
    if (buffer.length < LEARN_START || buffer.length < bs) return;
    const gW1 = Array.from({ length: HID }, () => new Array(IN).fill(0)), gb1 = new Array(HID).fill(0);
    const gW2 = Array.from({ length: OUT }, () => new Array(HID).fill(0)), gb2 = new Array(OUT).fill(0);
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
    trainCount++;
    if (trainCount % TARGET_UPDATE_INTERVAL === 0) target = copyNet(main);
  }

  /* ============================================================
  ШАГ ОБУЧЕНИЯ
  ============================================================ */
  function stepOnce() {
    const s = currentState;
    const a = chooseAction(main, s);
    lastAction = a;
    const res = stepGame(a);
    episodeReward += res.reward;
    lastEvent = res.result;

    const s2 = res.done ? null : (res.s2 || getState());
    remember(s, a, res.reward, s2, res.done);
    stepCounter++;
    const trainEvery = fastMode ? 6 : 2;
    if (stepCounter % trainEvery === 0) trainBatch(BATCH);
    if (res.done) {
      lastEpisodeReward = episodeReward;
      episodeCount++;
      if (res.result === 'посадка') score++;
      if (res.result !== 'шаг') lastNotableEvent = res.result;
      if (episodeReward > bestReward) { bestReward = episodeReward; bestNet = copyNet(main); }
      if (evalMode) {
        evalWindow.push(res.result === 'посадка' ? 1 : 0);
        if (evalWindow.length > EVAL_WINDOW) evalWindow.shift();
        if (evalWindow.length === EVAL_WINDOW && curStage < 3) {
          let sum = 0; for (const v of evalWindow) sum += v;
          if (sum / EVAL_WINDOW >= PROMOTE_RATE) {
            curStage++;
            boostUntilEp = episodeCount + BOOST_EP;
            evalWindow.length = 0;
          }
        }
      }
      evalMode = (episodeCount % EVAL_EVERY === 0);
      resetEpisode();
    } else {
      currentState = s2;
      trail.push({ x: ship.x, y: ship.y });
      if (trail.length > 60) trail.shift();
    }
  }

  /* ============================================================
  РЕНДЕР: ИГРА
  ============================================================ */
  function px(x) { return x * S; }
  function py(y) { return GH - y * S; }

  /* Корпус-трапеция + две отдельные ножки (как в референсе Gym) */
  function drawLanderShape(c, cx, cy, angle, scale, withFlame) {
    c.save();
    c.translate(cx, cy);
    c.rotate(angle);
    c.scale(scale, scale);
    const u = S;
    if (withFlame && flameMain) {
      c.fillStyle = 'rgba(255,255,255,0.85)';
      c.beginPath();
      c.moveTo(-0.55 * u, 1.0 * u);
      c.lineTo(0.55 * u, 1.0 * u);
      c.lineTo(0, (2.6 + Math.random() * 0.9) * u);
      c.closePath(); c.fill();
    }
    if (withFlame && flameSide !== 0) {
      c.fillStyle = 'rgba(255,255,255,0.8)';
      const sx = flameSide > 0 ? 1.5 * u : -1.5 * u;
      c.beginPath();
      c.moveTo(sx, -0.5 * u);
      c.lineTo(sx, 0.3 * u);
      c.lineTo(sx + flameSide * (1.4 + Math.random() * 0.5) * u, -0.1 * u);
      c.closePath(); c.fill();
    }
    /* корпус: трапеция, сужающаяся к верху */
    c.fillStyle = ACC;
    c.beginPath();
    c.moveTo(-1.3 * u, -1.1 * u);
    c.lineTo(1.3 * u, -1.1 * u);
    c.lineTo(1.7 * u, 1.0 * u);
    c.lineTo(-1.7 * u, 1.0 * u);
    c.closePath(); c.fill();
    /* ножки: два отдельных наклонных прямоугольника, зазор до корпуса */
    c.strokeStyle = ACC;
    c.lineWidth = 0.4 * u;
    c.lineCap = 'butt';
    c.beginPath(); c.moveTo(-2.05 * u, 0.55 * u); c.lineTo(-2.75 * u, 1.25 * u); c.stroke();
    c.beginPath(); c.moveTo(2.05 * u, 0.55 * u); c.lineTo(2.75 * u, 1.25 * u); c.stroke();
    c.restore();
  }

  function spawnParticles() {
    const bv = bodyVectors();
    if (flameMain) {
      for (let i = 0; i < 2; i++) {
        particles.push({
          x: ship.x - bv.ux * 1.2, y: ship.y - bv.uy * 1.2,
          vx: -bv.ux * 9 + (Math.random() - 0.5) * 2,
          vy: -bv.uy * 9 + (Math.random() - 0.5) * 2,
          life: 0.35
        });
      }
    }
    if (flameSide !== 0) {
      const sx = flameSide > 0 ? 1.3 : -1.3;
      const c = Math.cos(ship.angle), s = Math.sin(ship.angle);
      const wx = ship.x + sx * c, wy = ship.y - sx * s;
      particles.push({
        x: wx, y: wy,
        vx: flameSide * bv.rx * 8, vy: flameSide * bv.ry * 8,
        life: 0.3
      });
    }
    if (particles.length > 90) particles.splice(0, particles.length - 90);
  }

  function gateChecks() {
    const speedOk = Math.abs(ship.vy) <= V_SAFE && Math.abs(ship.vx) <= H_SAFE;
    const tiltOk = Math.abs(ship.angle) <= ANG_SAFE && Math.abs(ship.omega) <= OMEGA_SAFE;
    const posOk = ship.x >= pad.x1 && ship.x <= pad.x2 && (ship.y - pad.y) < 12;
    return [speedOk, tiltOk, posOk];
  }

  function drawGame() {
    g.clearRect(0, 0, GW, GH);
    g.fillStyle = '#000'; g.fillRect(0, 0, GW, GH);
    g.fillStyle = 'rgba(255,255,255,0.8)';
    for (const st of stars) g.fillRect(st.x, st.y, st.r, st.r);

    for (let i = 0; i < trail.length; i++) {
      const a = i / trail.length * 0.35;
      g.fillStyle = 'rgba(255,255,255,' + a.toFixed(2) + ')';
      g.fillRect(px(trail[i].x) - 1, py(trail[i].y) - 1, 2, 2);
    }

    const gc = gateChecks();
    if (gc[0] && gc[1] && gc[2]) {
      g.fillStyle = 'rgba(' + Engine.ACCENT_RGB + ',0.22)';
      g.fillRect(px(pad.x1), py(pad.y) - 16, (pad.x2 - pad.x1) * S, 16);
    }

    g.fillStyle = '#fff';
    g.beginPath();
    g.moveTo(0, GH);
    for (let i = 0; i < M; i++) g.lineTo(px(i * (WORLD / (M - 1))), py(heights[i]));
    g.lineTo(GW, GH);
    g.closePath(); g.fill();

    g.strokeStyle = ACC; g.lineWidth = 2; g.setLineDash([5, 4]);
    g.beginPath(); g.moveTo(px(pad.x1), py(pad.y)); g.lineTo(px(pad.x2), py(pad.y)); g.stroke();
    g.setLineDash([]);
    for (const fx of [pad.x1, pad.x2]) {
      g.strokeStyle = '#fff'; g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(px(fx), py(pad.y)); g.lineTo(px(fx), py(pad.y) - 3.5 * S); g.stroke();
      g.fillStyle = ACC;
      g.beginPath();
      g.moveTo(px(fx), py(pad.y) - 3.5 * S);
      g.lineTo(px(fx) + 1.8 * S, py(pad.y) - 3.5 * S + 0.45 * S);
      g.lineTo(px(fx), py(pad.y) - 3.5 * S + 0.9 * S);
      g.closePath(); g.fill();
    }

    for (const p of particles) {
      const a = clamp(p.life / 0.35, 0, 1) * 0.9;
      g.fillStyle = 'rgba(255,255,255,' + a.toFixed(2) + ')';
      g.beginPath(); g.arc(px(p.x), py(p.y), 1.8, 0, Math.PI * 2); g.fill();
    }

    drawLanderShape(g, px(ship.x), py(ship.y), ship.angle, 1, true);

    if (wind !== 0) {
      g.strokeStyle = '#fff'; g.fillStyle = '#fff'; g.lineWidth = 1.5;
      const x0 = 22, y0 = 22, x1 = 22 + wind * 30;
      g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y0); g.stroke();
      const dir = wind > 0 ? 1 : -1;
      g.beginPath();
      g.moveTo(x1, y0);
      g.lineTo(x1 - dir * 6, y0 - 4);
      g.lineTo(x1 - dir * 6, y0 + 4);
      g.closePath(); g.fill();
      g.font = '10px ' + MONO; g.textAlign = 'left';
      g.fillText('ветер', x0, y0 + 14);
    }

    const labels = ['v', 'φ', 'x'];
    for (let i = 0; i < 3; i++) {
      const bx = 352 + i * 20, by = 14;
      if (gc[i]) { g.fillStyle = '#fff'; g.fillRect(bx, by, 12, 12); }
      else { g.strokeStyle = 'rgba(255,255,255,0.5)'; g.lineWidth = 1; g.strokeRect(bx + 0.5, by + 0.5, 12, 12); }
      g.fillStyle = gc[i] ? '#000' : 'rgba(255,255,255,0.6)';
      g.font = '9px ' + MONO; g.textAlign = 'center';
      g.fillText(labels[i], bx + 6, by + 22);
    }
  }

  /* ============================================================
  РЕНДЕР: СЕТЬ
  ============================================================ */
  function drawEdge(c, x1, y1, x2, y2, w) {
    const aw = Math.abs(w); if (aw < 0.03) return;
    c.save();
    c.strokeStyle = '#000';
    c.globalAlpha = Math.min(1, 0.12 + aw * 0.25);
    c.lineWidth = Math.min(2.5, 0.4 + aw * 0.3);
    if (w < 0) c.setLineDash([2, 2]);
    c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2, y2); c.stroke();
    c.restore();
  }
  function drawNet() {
    nctx.clearRect(0, 0, GRID, GRID);
    const inputX = 118, hiddenX = 225, outputX = 322;
    const inputY = i => 60 + i * 42;
    const hiddenY = j => 14 + j * 12.5;
    const outputY = a => 90 + a * 80;
    const f = forwardNet(main, currentState);
    for (let j = 0; j < HID; j++)
      for (let i = 0; i < IN; i++)
        drawEdge(nctx, inputX + 8, inputY(i), hiddenX - 6, hiddenY(j), main.W1[j][i]);
    for (let a = 0; a < OUT; a++)
      for (let j = 0; j < HID; j++)
        drawEdge(nctx, hiddenX + 6, hiddenY(j), outputX - 9, outputY(a), main.W2[a][j]);
    for (let i = 0; i < IN; i++) {
      const x = inputX, y = inputY(i);
      const v = clamp(Math.abs(currentState[i] || 0), 0, 1);
      nctx.beginPath(); nctx.arc(x, y, 7, 0, Math.PI * 2);
      nctx.fillStyle = 'rgba(0,0,0,' + (0.08 + 0.85 * v) + ')'; nctx.fill();
      nctx.strokeStyle = '#000'; nctx.lineWidth = 1; nctx.stroke();
      nctx.fillStyle = '#000'; nctx.font = '10px ' + MONO; nctx.textAlign = 'right';
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
      nctx.beginPath(); nctx.arc(x, y, 3.5, 0, Math.PI * 2);
      nctx.fillStyle = 'rgba(0,0,0,' + (0.05 + 0.8 * v) + ')'; nctx.fill();
      nctx.strokeStyle = '#000'; nctx.lineWidth = 1; nctx.stroke();
    }
    const mn = Math.min(...f.q), mx = Math.max(...f.q), rg = mx - mn;
    for (let a = 0; a < OUT; a++) {
      const x = outputX, y = outputY(a), norm = rg > 1e-6 ? (f.q[a] - mn) / rg : 0.5;
      nctx.beginPath(); nctx.arc(x, y, 7, 0, Math.PI * 2);
      nctx.fillStyle = 'rgba(0,0,0,' + (0.08 + 0.85 * norm) + ')'; nctx.fill();
      nctx.lineWidth = a === lastAction ? 2.5 : 1;
      if (a === lastAction && lastActionRandom) nctx.setLineDash([3, 2]);
      nctx.strokeStyle = a === lastAction ? Engine.ACCENT_HEX : '#000'; nctx.stroke(); nctx.setLineDash([]);
      nctx.fillStyle = '#000'; nctx.font = '10px ' + MONO; nctx.textAlign = 'left';
      nctx.fillText(OUTPUT_LABELS[a] + ' ' + f.q[a].toFixed(1), outputX + 14, y + 4);
    }
  }

  /* ============================================================
  РЕНДЕР: СТАТИСТИКА
  ============================================================ */
  function drawStats() {
    statsEl.innerHTML =
      '<span>Эпизод: <span class="v">' + Engine.fmtFixed(episodeCount, 5, 0) + '</span></span>' +
      '<span>Стадия: <span class="v">' + curStage + '/3</span></span>' +
      '<span>Успехов: <span class="v">' + (evalWindow.length ? evalWindow.reduce(function (a, b) { return a + b; }, 0) + '/' + evalWindow.length : '—') + '</span></span>' +
      '<span>' + METRIC_LABEL + ': <span class="v">' + Engine.fmtFixed(score, 5, 0) + '</span></span>' +
      '<span>Награда: <span class="v">' + Engine.fmtFixed(episodeReward, 4, 1) + '</span></span>' +
      '<span>Лучшая награда: <span class="v">' + Engine.fmtFixed(bestReward > -Infinity ? Math.round(bestReward) : 0, 4, 0) + '</span></span>' +
      '<span>ε: <span class="v">' + (evalMode ? Engine.fmtFixed(0, 1, 3) : Engine.fmtFixed(currentEpsilon(), 1, 3)) + '</span></span>';
  }

  /* ============================================================
  ГЛАВНЫЙ ЦИКЛ + ПАРКОВЩИК rAF
  ============================================================ */
  let lastNormalStep = 0, lastTs = 0;
  function mainLoop(ts) {
    const dtms = lastTs ? Math.min(100, ts - lastTs) : 16;
    lastTs = ts;
    if (mainStarted) {
      if (fastMode) { for (let i = 0; i < FAST_STEPS; i++) stepOnce(); }
      else if (ts - lastNormalStep >= NORMAL_STEP_DELAY) { stepOnce(); lastNormalStep = ts; }
      spawnParticles();
      const dts = dtms / 1000;
      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.x += p.vx * dts; p.y += p.vy * dts; p.life -= dts;
        if (p.life <= 0) particles.splice(i, 1);
      }
      drawGame(); drawNet(); drawStats();
    }
    parker.raf(mainLoop);
  }
  function introFrame(ts) {
    if (introFinished) return;
    if (introStart === null) introStart = ts;
    const t = (ts - introStart) / 1000;
    drawIntro(t);
    if (t >= 26.5) { finishIntro(); return; }
    parker.raf(introFrame);
  }
  const parker = Engine.makeRafParker({
    getMain: function () { return mainLoop; },
    getIntro: function () { return introFrame; }
  });

  /* ============================================================
  АДАПТИВНЫЕ КАНВАСЫ
  Внутренняя система координат обоих квадратных канвасов
  фиксирована (GRID×GRID) — реальный размер задаёт CSS/ResizeObserver,
  а рисунок масштабируется одним ctx.scale(k,k) в начале каждого
  кадра.
  ============================================================ */
  let gameScale = 1, netScale = 1, gameOX = 0, gameOY = 0, netOX = 0, netOY = 0;
  Engine.resizeCanvas(gameCanvas, { onResize: (w, h) => { gameScale = Math.min(w, h) / GRID; gameOX = (w - GRID * gameScale) / 2; gameOY = (h - GRID * gameScale) / 2; } });
  Engine.resizeCanvas(netCanvas, { onResize: (w, h) => { netScale = Math.min(w, h) / GRID; netOX = (w - GRID * netScale) / 2; netOY = (h - GRID * netScale) / 2; } });
  const rawDrawGame = drawGame, rawDrawNet = drawNet;
  drawGame = function () { g.save(); g.translate(gameOX, gameOY); g.scale(gameScale, gameScale); g.beginPath(); g.rect(0, 0, GRID, GRID); g.clip(); rawDrawGame(); g.lineWidth = 3 / gameScale; g.strokeStyle = '#1a1c20'; g.strokeRect(0, 0, GRID, GRID); g.restore(); };
  drawNet = function () { nctx.save(); nctx.translate(netOX, netOY); nctx.scale(netScale, netScale); rawDrawNet(); nctx.restore(); };

  /* ============================================================
  КРИВАЯ ОБУЧЕНИЯ
  ============================================================ */
  const curve = Engine.makeCurve(document.getElementById('lander-curve'), {
    primaryLabel: 'посадок', secondaryLabel: 'награда эп.'
  });
  Engine.resizeCanvas(document.getElementById('lander-curve'), {});
  let lastEpiSample = -1;
  setInterval(() => {
    if (!parker.isRunning()) return;
    if (episodeCount === lastEpiSample) return;
    lastEpiSample = episodeCount;
    curve.sample(episodeCount, score, Math.round(lastEpisodeReward * 100) / 100);
  }, 250);

  /* ============================================================
  КНОПКИ
  ============================================================ */
  speedBtn.addEventListener('click', () => {
    fastMode = !fastMode;
    speedBtn.textContent = fastMode ? 'Обучение: ускоренное' : 'Обучение: обычное'; speedBtn.classList.toggle('eng-btn-accent', fastMode);
  });
  infoBtn.addEventListener('click', () => { modalOverlay.classList.add('is-open'); });
  modalClose.addEventListener('click', () => { modalOverlay.classList.remove('is-open'); });
  modalOk.addEventListener('click', () => { modalOverlay.classList.remove('is-open'); });
  modalOverlay.addEventListener('click', e => { if (e.target === modalOverlay) modalOverlay.classList.remove('is-open'); });
  resetBtn.addEventListener('click', () => { resetAll(); });

  function resetAll() {
    main = randomNet();
    target = copyNet(main);
    bestNet = copyNet(main);
    adamMain = makeAdam(main);
    bestReward = -Infinity;
    score = 0;
    episodeCount = 0; stepCounter = 0; trainCount = 0;
    curStage = 1; boostUntilEp = BOOST_EP; evalMode = false; evalWindow.length = 0;
    buffer.length = 0; pending.length = 0;
    lastNotableEvent = ''; lastEpisodeReward = 0;
    lastEpiSample = -1; curve.reset();
    resetEpisode();
    drawGame(); drawNet(); drawStats();
  }

  weightsBtn.addEventListener('click', () => {
    Engine.openWeightsModal({
      key: DEMO_KEY,
      exportWeights: exportWeights,
      importWeights: importWeights,
      presetWeights: (window.DemoWeights && window.DemoWeights[DEMO_KEY]) || null
    });
  });

  /* ============================================================
  ЗАСТАВКА
  ============================================================ */
  const introOverlay = document.getElementById('lander-introOverlay');
  const introCanvas  = document.getElementById('lander-introCanvas');
  const ictx = introCanvas.getContext('2d');
  let introFinished = false, introStart = null;

  function finishIntro() { if (introFinished) return; introFinished = true; introOverlay.style.display = 'none'; mainStarted = true; }
  introOverlay.addEventListener('click', finishIntro);

  function clamp01(v) { return Math.max(0, Math.min(1, v)); }
  function prog(t, a, b) { return clamp01((t - a) / (b - a)); }
  function ease(p) { return p * p * (3 - 2 * p); }

  const I_H = [18, 16, 14, 15, 17, 20, 19, 17, 16, 16, 16, 16, 16, 15, 13, 12, 14, 17, 20, 22, 21, 19, 18];
  const I_PAD = { x1: 9, x2: 13, lvl: 16 };
  const I_SHIP = { x: 210, y: 200 };

  const CAPS = [
    [0.0,  2.5,  'Луна: рваный грунт, площадка между двумя флажками.'],
    [2.5,  5.0,  'Четыре действия: выкл · главный · влево · вправо.'],
    [5.0,  9.0,  'Что видит модуль: позиция и скорость относительно площадки.'],
    [9.0,  12.5, 'Плюс наклон, вращение и ветер. Всего 8 чисел.'],
    [12.5, 17.0, 'Сеть 8 → 32 → 4 оценивает каждое действие.'],
    [17.0, 21.0, 'Награды: посадка +100, авария −100, двигатели −.'],
    [21.0, 26.0, 'Обучение с подкреплением начинается…']
  ];
  const INTRO_INPUTS = [
    { label: 'до площадки по X', value: '+0.30', appear: 5.2 },
    { label: 'высота над площадкой', value: '+0.55', appear: 5.8 },
    { label: 'радар до грунта', value: '+0.40', appear: 6.4 },
    { label: 'скорость по X', value: '−0.10', appear: 7.0 },
    { label: 'скорость по Y', value: '−0.35', appear: 7.6 },
    { label: 'наклон корпуса', value: '+0.06', appear: 9.2 },
    { label: 'вращение', value: '−0.02', appear: 10.0 },
    { label: 'ветер', value: '+0.40', appear: 10.8 }
  ];

  function iSceneX(u) { return 10 + u * 420; }
  function iSceneY(h) { return 460 - h * 8; }

  function drawIntroScene(c, t, alpha) {
    if (alpha <= 0) return;
    c.save(); c.globalAlpha = alpha;
    c.fillStyle = '#000';
    c.fillRect(10, 40, 420, 420);
    c.fillStyle = 'rgba(255,255,255,0.8)';
    for (let i = 0; i < 40; i++) {
      const sx = 10 + ((i * 97) % 420), sy = 50 + ((i * 61) % 250);
      c.fillRect(sx, sy, 1.5, 1.5);
    }
    c.fillStyle = '#fff';
    c.beginPath();
    c.moveTo(10, 460);
    for (let i = 0; i < I_H.length; i++) c.lineTo(iSceneX(i / (I_H.length - 1)), iSceneY(I_H[i]));
    c.lineTo(430, 460);
    c.closePath(); c.fill();
    c.strokeStyle = ACC; c.lineWidth = 2; c.setLineDash([5, 4]);
    c.beginPath();
    c.moveTo(iSceneX(I_PAD.x1 / 22), iSceneY(I_PAD.lvl));
    c.lineTo(iSceneX(I_PAD.x2 / 22), iSceneY(I_PAD.lvl));
    c.stroke(); c.setLineDash([]);
    for (const fx of [I_PAD.x1 / 22, I_PAD.x2 / 22]) {
      c.strokeStyle = '#fff'; c.lineWidth = 1.5;
      c.beginPath(); c.moveTo(iSceneX(fx), iSceneY(I_PAD.lvl)); c.lineTo(iSceneX(fx), iSceneY(I_PAD.lvl) - 30); c.stroke();
      c.fillStyle = ACC;
      c.beginPath();
      c.moveTo(iSceneX(fx), iSceneY(I_PAD.lvl) - 30);
      c.lineTo(iSceneX(fx) + 16, iSceneY(I_PAD.lvl) - 26);
      c.lineTo(iSceneX(fx), iSceneY(I_PAD.lvl) - 22);
      c.closePath(); c.fill();
    }
    const flick = Math.floor(t * 12) % 2 === 0;
    flameMain = true; flameSide = 0;
    drawLanderShape(c, I_SHIP.x, I_SHIP.y, 0.06 * Math.sin(t * 1.3), 1, flick);
    flameMain = false;
    c.restore();
  }

  function drawIntroActions(c, t) {
    if (t < 2.5 || t >= 5.2) return;
    const a = Math.min(prog(t, 2.5, 3.1), 1 - prog(t, 4.6, 5.2));
    c.save(); c.globalAlpha = a;
    c.fillStyle = '#000'; c.font = '12px Inter, sans-serif';
    c.textAlign = 'left';
    c.fillText('выкл — ничего не горит', 60, 90);
    c.fillText('главный — тяга вверх по корпусу', 60, 108);
    c.fillText('влево — наклон влево', 60, 126);
    c.fillText('вправо — наклон вправо', 60, 144);
    c.strokeStyle = '#000'; c.lineWidth = 1.5;
    c.beginPath(); c.arc(I_SHIP.x, I_SHIP.y, 34, Math.PI * 1.15, Math.PI * 1.5); c.stroke();
    c.beginPath(); c.arc(I_SHIP.x, I_SHIP.y, 34, Math.PI * 1.85, Math.PI * 1.5, true); c.stroke();
    c.restore();
  }

  function drawIntroPanel(c, t) {
    const x0 = 450;
    c.fillStyle = '#000'; c.font = '12px Inter, sans-serif'; c.textAlign = 'left';
    c.fillText('Вход нейросети (8 чисел):', x0, 84);
    INTRO_INPUTS.forEach((row, i) => {
      if (t < row.appear) return;
      const alpha = prog(t, row.appear, row.appear + 0.4);
      const y = 110 + i * 24;
      c.save(); c.globalAlpha = alpha;
      c.fillStyle = '#000'; c.font = '12px Inter, sans-serif';
      c.textAlign = 'left'; c.fillText(row.label, x0, y);
      c.font = '12px ' + MONO; c.textAlign = 'right'; c.fillText(row.value, x0 + 260, y);
      c.restore();
    });
    const na = prog(t, 12.5, 13.5);
    if (na > 0) {
      c.save(); c.globalAlpha = na;
      const x1 = 470, x2 = 560, x3 = 660;
      const yIn = i => 300 + i * 16;
      const yHid = j => 296 + j * 4.4;
      const yOut = a => 330 + a * 26;
      c.strokeStyle = 'rgba(0,0,0,0.10)'; c.lineWidth = 1;
      for (let i = 0; i < 8; i++) for (let j = 0; j < 32; j++) { c.beginPath(); c.moveTo(x1, yIn(i)); c.lineTo(x2, yHid(j)); c.stroke(); }
      for (let j = 0; j < 32; j++) for (let a = 0; a < 4; a++) { c.beginPath(); c.moveTo(x2, yHid(j)); c.lineTo(x3, yOut(a)); c.stroke(); }
      c.fillStyle = '#000';
      for (let i = 0; i < 8; i++) { c.beginPath(); c.arc(x1, yIn(i), 3, 0, Math.PI * 2); c.fill(); }
      for (let j = 0; j < 32; j++) { c.beginPath(); c.arc(x2, yHid(j), 2, 0, Math.PI * 2); c.fill(); }
      for (let a = 0; a < 4; a++) { c.beginPath(); c.arc(x3, yOut(a), 3.5, 0, Math.PI * 2); c.fill(); }
      c.font = '10px Inter, sans-serif'; c.textAlign = 'left';
      c.fillText('выкл', x3 + 10, yOut(0) + 3);
      c.fillText('главный', x3 + 10, yOut(1) + 3);
      c.fillText('влево', x3 + 10, yOut(2) + 3);
      c.fillText('вправо', x3 + 10, yOut(3) + 3);
      c.font = '12px ' + MONO; c.textAlign = 'center';
      c.fillText('8 → 32 → 4', (x1 + x3) / 2, 400);
      c.restore();
    }
    const ra = prog(t, 17.0, 17.6) * (1 - prog(t, 20.4, 21.0));
    if (ra > 0) {
      c.save(); c.globalAlpha = ra;
      c.fillStyle = '#000'; c.font = '12px Inter, sans-serif'; c.textAlign = 'left';
      c.fillText('посадка +100 (+мягкость, +центр)', x0, 414);
      c.fillText('авария −100 · край −30 · время −25', x0, 430);
      c.fillText('двигатели: −0.12 / −0.04 · шаг −0.02', x0, 446);
      c.fillText('шейпинг: ближе · медленнее · ровнее', x0, 462);
      c.restore();
    }
  }

  function drawIntro(t) {
    const c = ictx;
    c.clearRect(0, 0, 720, 480);
    c.fillStyle = '#fff'; c.fillRect(0, 0, 720, 480);
    const cap = CAPS.find(x => t >= x[0] && t < x[1]);
    if (cap) { c.fillStyle = '#000'; c.font = '14px Inter, sans-serif'; c.textAlign = 'left'; c.fillText(cap[2], 14, 24); }
    drawIntroScene(c, t, ease(prog(t, 0.2, 1.2)));
    drawIntroActions(c, t);
    drawIntroPanel(c, t);
    c.fillStyle = '#999'; c.font = '11px Inter, sans-serif'; c.textAlign = 'right';
    c.fillText('клик — пропустить', 712, 474);
    const fo = prog(t, 24.5, 25.8);
    if (fo > 0) { c.fillStyle = 'rgba(255,255,255,' + fo + ')'; c.fillRect(0, 0, 720, 480); }
    const ta = prog(t, 24.7, 25.3);
    if (ta > 0 && t < 26.2) {
      c.fillStyle = 'rgba(0,0,0,' + ta + ')'; c.font = '20px Inter, sans-serif'; c.textAlign = 'center';
      c.fillText('Обучение начинается…', 360, 240);
    }
  }

  /* ============================================================
  ЭКСПОРТ / ИМПОРТ
  ============================================================ */
  function exportWeights() { return { main: main, best: bestNet }; }
  function importWeights(W) {
    try {
      if (!W || !W.main || !W.main.W1) return false;
      const n = W.main;
      if (n.W1.length !== HID || n.W2.length !== OUT) return false;
      if (n.W1[0].length !== IN || n.W2[0].length !== HID) return false;
      main = n;
      target = copyNet(main);
      if (W.best && W.best.W1) bestNet = copyNet(W.best);
      adamMain = makeAdam(main);
      episodeCount = Math.max(episodeCount, EPS_DECAY_EP); boostUntilEp = 0; evalWindow.length = 0;
      return true;
    } catch (e) { return false; }
  }

  /* ============================================================
  СТАРТ
  ============================================================ */
  main = randomNet();
  target = copyNet(main);
  bestNet = copyNet(main);
  adamMain = makeAdam(main);
  resetEpisode();
  drawGame(); drawNet(); drawStats();
  parker.raf(mainLoop);
  parker.raf(introFrame);

  /* ============================================================
  РЕГИСТРАЦИЯ В КОЛОДЕ
  ============================================================ */
  Engine.register(DEMO_KEY, {
    key: DEMO_KEY,
    title: 'Лунный модуль',
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
