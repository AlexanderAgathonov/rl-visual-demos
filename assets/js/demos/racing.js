/* Автор: А. А. Агафонов (КФУ). Лицензия: MIT. Часть материалов к статье об изобразительных средствах обучения основам RL. */
/* ==== js/demos/racing.js (demo-kit/racing.html) ==== */
(function () {
  'use strict';

  /* ============================================================
  КОНФИГУРАЦИЯ ДЕМО
  ============================================================ */
  const DEMO_KEY = 'racing';
  const ACC = Engine.ACCENT_HEX;
  const ACC_RGB = Engine.ACCENT_RGB;
  const MONO = Engine.FONT_MONO;
  const GRID = 420;
  const METRIC_LABEL = 'Дист';

  /* ============================================================
  DOM
  ============================================================ */
  const gameCanvas = document.getElementById('racing-game');
  const g = gameCanvas.getContext('2d');
  const netCanvas = document.getElementById('racing-net');
  const nctx = netCanvas.getContext('2d');
  const statsEl = document.getElementById('racing-stats');
  const speedBtn = document.getElementById('racing-speedBtn');
  const infoBtn = document.getElementById('racing-infoBtn');
  const resetBtn = document.getElementById('racing-resetBtn');
  const weightsBtn = document.getElementById('racing-weightsBtn');
  const modalOverlay = document.getElementById('racing-modalOverlay');
  const modalClose = document.getElementById('racing-modalClose');
  const modalOk = document.getElementById('racing-modalOk');

  /* ============================================================
  МИР · палитра: оттенки серого + один акцент (тот же, что у
  остальных демо колоды)
  ============================================================ */
  const GW = GRID;
  const GH = GRID;
  const VIEW_ZOOM = 1.8;      // базовых пикселей на мировой юнит
  const CAR_COLOR = ACC;             // машина — акцентным
  const DOT_COLOR = '#ffffff';       // точки-сенсоры — белым
  const WALL_COLOR = '#82868b';      // серые стены
  const HIT_COLOR = 'rgba(' + ACC_RGB + ',0.75)';  // крестики лидаров — акцент
  const SPARK_COLOR = '#454b55';     // искры — тёмно-серые
  const FLASH_COLOR = '22,24,29';    // вспышка рамки — тёмно-серая
  const MAP_START_COLOR = '#000';    // старт на миникарте — чёрная жирная точка
  const CAR_HALF_L = 7;       // полу-длина корпуса
  const CAR_HALF_W = 4;       // полу-ширина корпуса
  const MAX_SPEED = 3.5;
  const MIN_SPEED = 0.8;
  const TURN_RATE = 0.09;
  const LIDAR_ANGLES = [-0.9, -0.45, 0, 0.45, 0.9];
  const LIDAR_RANGE = 110;
  const CRASH_PENALTY = -10;
  const MAX_STEPS = 900;
  const TRACK_MAX_EPISODES = 300;    // страховка: максимум эпизодов на трассе
  const SIGMA_NEWTRACK = 0.22;       // всплеск исследования на новой трассе

  /* ============================================================
  СЕТЬ 5 → 4 → 3 → 2
  ============================================================ */
  const IN = 5, H1 = 4, H2 = 3, OUT = 2;
  const INPUT_LABELS = [
    ['лидар', 'лев-лев'], ['лидар', 'левее'], ['лидар', 'прямо'],
    ['лидар', 'правее'], ['лидар', 'прав-прав']
  ];
  const OUTPUT_LABELS = ['поворот', 'газ'];

  /* ============================================================
  ГИПЕРПАРАМЕТРЫ RL
  ============================================================ */
  const GAMMA = 0.99;
  const LR = 0.02;
  const ADV_CLIP = 5.0;
  const SIGMA_START = 0.35;
  const SIGMA_MIN = 0.12;
  const SIGMA_DECAY = 0.995;
  const FAST_STEPS = 60;
  const NORMAL_STEP_DELAY = 40;
  let W1, b1, W2, b2, W3, b3;
  let sigma = SIGMA_START;

  /* ============================================================
  ТРАССА: процедурное «кольцо-змейка»
  ============================================================ */
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function curvRadius(prev, p, next) {
    const a = Math.hypot(p.x - prev.x, p.y - prev.y);
    const b = Math.hypot(next.x - p.x, next.y - p.y);
    const c = Math.hypot(next.x - prev.x, next.y - prev.y);
    const area = Math.abs((prev.x * (p.y - next.y) + p.x * (next.y - prev.y) + next.x * (prev.y - p.y)) / 2);
    if (area < 1e-6) return 1e9;
    return (a * b * c) / (4 * area);
  }
  function buildTrack() {
    const n = 240;
    const R0 = 300 + Math.random() * 180;
    const ks = [2, 3, 4, 5, 6, 7];
    const rawA = ks.map(k => (0.5 + Math.random()) / k);
    const sumA = rawA.reduce((s, a) => s + a, 0);
    const budget = 0.30 + Math.random() * 0.14;
    const amps = rawA.map(a => a / sumA * budget);
    const phs = ks.map(() => Math.random() * Math.PI * 2);
    const stretch = 1.15 + Math.random() * 0.35;
    const rot = Math.random() * Math.PI * 2;
    const cr = Math.cos(rot), sr = Math.sin(rot);
    const raw = [];
    for (let i = 0; i < n; i++) {
      const th = i / n * 2 * Math.PI;
      let r = 1;
      for (let q = 0; q < ks.length; q++) r += amps[q] * Math.cos(ks[q] * th + phs[q]);
      r *= R0;
      const x = r * Math.cos(th) * stretch, y = r * Math.sin(th);
      raw.push({ x: x * cr - y * sr, y: x * sr + y * cr });
    }
    const hwBase = 22 + Math.random() * 10;
    const hwA = 4 + Math.random() * 5;
    const hwP = Math.random() * Math.PI * 2;
    const half = [];
    for (let i = 0; i < n; i++) {
      const th = i / n * 2 * Math.PI;
      half.push(hwBase + hwA * Math.cos(3 * th + hwP) + hwA * 0.6 * Math.cos(5 * th + hwP * 2));
    }
    for (let i = 0; i < n; i++) {
      const R = curvRadius(raw[(i - 1 + n) % n], raw[i], raw[(i + 1) % n]);
      half[i] = clamp(half[i], 12, Math.min(40, R * 0.7));
    }
    const halfS = half.map((h, i) => (half[(i - 1 + n) % n] + h + half[(i + 1) % n]) / 3);
    const inner = [], outer = [];
    for (let i = 0; i < n; i++) {
      const p = raw[i], prev = raw[(i - 1 + n) % n], next = raw[(i + 1) % n];
      let tx = next.x - prev.x, ty = next.y - prev.y;
      const tl = Math.hypot(tx, ty) || 1;
      tx /= tl; ty /= tl;
      let nx = -ty, ny = tx;
      if (nx * p.x + ny * p.y < 0) { nx = -nx; ny = -ny; }
      outer.push({ x: p.x + nx * halfS[i], y: p.y + ny * halfS[i] });
      inner.push({ x: p.x - nx * halfS[i], y: p.y - ny * halfS[i] });
    }
    const cT = raw;
    let cx = 0, cy = 0;
    cT.forEach(p => { cx += p.x; cy += p.y; });
    cx /= cT.length; cy /= cT.length;
    let meanR = 0;
    cT.forEach(p => { meanR += Math.hypot(p.x - cx, p.y - cy); });
    meanR /= cT.length;
    const p0 = cT[0], p1 = cT[1];
    const a0 = Math.atan2(p0.y - cy, p0.x - cx), a1 = Math.atan2(p1.y - cy, p1.x - cx);
    let dA0 = a1 - a0;
    while (dA0 > Math.PI) dA0 -= 2 * Math.PI;
    while (dA0 < -Math.PI) dA0 += 2 * Math.PI;
    const forwardSign = dA0 >= 0 ? 1 : -1;
    const startHeading = Math.atan2(p1.y - p0.y, p1.x - p0.x);
    return { outer, inner, centerline: cT, center: { x: cx, y: cy }, meanR, forwardSign, startPos: { x: p0.x, y: p0.y }, startHeading };
  }
  let outer, inner, centerline, center, meanR, forwardSign, startPos, startHeading;
  let trackBounds = { minx: -1, miny: -1, maxx: 1, maxy: 1 };
  let trackLength = 1;
  let trackNum = 1;
  let episodesOnTrack = 0;
  function regenTrack() {
    const t = buildTrack();
    outer = t.outer; inner = t.inner; centerline = t.centerline; center = t.center;
    meanR = t.meanR; forwardSign = t.forwardSign; startPos = t.startPos; startHeading = t.startHeading;
    let mnx = 1e9, mny = 1e9, mxx = -1e9, mxy = -1e9;
    outer.forEach(p => { mnx = Math.min(mnx, p.x); mny = Math.min(mny, p.y); mxx = Math.max(mxx, p.x); mxy = Math.max(mxy, p.y); });
    trackBounds = { minx: mnx, miny: mny, maxx: mxx, maxy: mxy };
    let L = 0;
    for (let i = 0; i < centerline.length; i++) {
      const a = centerline[i], b = centerline[(i + 1) % centerline.length];
      L += Math.hypot(b.x - a.x, b.y - a.y);
    }
    trackLength = L;
  }
  regenTrack();

  let car = { x: startPos.x, y: startPos.y, heading: startHeading, speed: MAX_SPEED * 0.6 };
  let camX = car.x, camY = car.y, camZ = VIEW_ZOOM, camShake = 0, crashFlash = 0;
  const sparks = [];
  const trail = [];
  const TRAIL_MAX = 140;
  let distance = 0, bestDistance = 0, steps = 0, episodeReward = 0, episodeCount = 0;
  let currentState = null;
  let lastLidar = [1, 1, 1, 1, 1];
  let lastHits = [null, null, null, null, null];
  let lastActionMean = [0, 0];
  let lastActionSample = [0, 0];
  let lastNotableEvent = '';
  let fastMode = false, mainStarted = false;
  let episodeData = [];

  function gaussian() {
    let u = 0, v = 0;
    while (!u) u = Math.random();
    while (!v) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /* ============================================================
  ГЕОМЕТРИЯ
  ============================================================ */
  function pointInPoly(p, poly) {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const xi = poly[i].x, yi = poly[i].y, xj = poly[j].x, yj = poly[j].y;
      const inter = ((yi > p.y) !== (yj > p.y)) && (p.x < (xj - xi) * (p.y - yi) / (yj - yi) + xi);
      if (inter) inside = !inside;
    }
    return inside;
  }
  function carOnTrack() {
    const cH = Math.cos(car.heading), sH = Math.sin(car.heading);
    const pts = [
      [0, 0], [CAR_HALF_L, 0], [CAR_HALF_L, CAR_HALF_W], [CAR_HALF_L, -CAR_HALF_W],
      [-CAR_HALF_L, CAR_HALF_W], [-CAR_HALF_L, -CAR_HALF_W]
    ];
    for (let i = 0; i < pts.length; i++) {
      const wx = car.x + pts[i][0] * cH - pts[i][1] * sH;
      const wy = car.y + pts[i][0] * sH + pts[i][1] * cH;
      if (!pointInPoly({ x: wx, y: wy }, outer)) return false;
      if (pointInPoly({ x: wx, y: wy }, inner)) return false;
    }
    return true;
  }
  function raySeg(px, py, dx, dy, ax, ay, bx, by) {
    const rx = bx - ax, ry = by - ay;
    const den = dx * ry - dy * rx;
    if (Math.abs(den) < 1e-9) return Infinity;
    const t = ((ax - px) * ry - (ay - py) * rx) / den;
    const u = ((ax - px) * dy - (ay - py) * dx) / den;
    if (t >= 0 && u >= 0 && u <= 1) return t;
    return Infinity;
  }
  function rayPoly(px, py, dx, dy, poly) {
    let best = Infinity;
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i], b = poly[(i + 1) % poly.length];
      const t = raySeg(px, py, dx, dy, a.x, a.y, b.x, b.y);
      if (t < best) best = t;
    }
    return best;
  }
  function readLidars() {
    const res = [];
    for (let i = 0; i < LIDAR_ANGLES.length; i++) {
      const ang = car.heading + LIDAR_ANGLES[i];
      const dx = Math.cos(ang), dy = Math.sin(ang);
      let d = Math.min(rayPoly(car.x, car.y, dx, dy, outer), rayPoly(car.x, car.y, dx, dy, inner));
      if (d > LIDAR_RANGE) { d = LIDAR_RANGE; lastHits[i] = null; }
      else { lastHits[i] = { x: car.x + dx * d, y: car.y + dy * d }; }
      res.push(d / LIDAR_RANGE);
    }
    return res;
  }

  /* ============================================================
  СЕТЬ
  ============================================================ */
  function randomNet() {
    const s1 = Math.sqrt(2.0 / IN), s2 = Math.sqrt(2.0 / H1), s3 = Math.sqrt(2.0 / H2);
    return {
      W1: Array.from({ length: H1 }, () => Array.from({ length: IN }, () => (Math.random() - 0.5) * s1 * 2)),
      b1: Array.from({ length: H1 }, () => (Math.random() - 0.5) * 0.2),
      W2: Array.from({ length: H2 }, () => Array.from({ length: H1 }, () => (Math.random() - 0.5) * s2 * 2)),
      b2: Array.from({ length: H2 }, () => (Math.random() - 0.5) * 0.2),
      W3: Array.from({ length: OUT }, () => Array.from({ length: H2 }, () => (Math.random() - 0.5) * s3 * 2)),
      b3: Array.from({ length: OUT }, () => (Math.random() - 0.5) * 0.2)
    };
  }
  function forward(s) {
    const h1 = new Array(H1);
    for (let j = 0; j < H1; j++) {
      let sum = b1[j];
      for (let i = 0; i < IN; i++) sum += W1[j][i] * s[i];
      h1[j] = Math.tanh(sum);
    }
    const h2 = new Array(H2);
    for (let k = 0; k < H2; k++) {
      let sum = b2[k];
      for (let j = 0; j < H1; j++) sum += W2[k][j] * h1[j];
      h2[k] = Math.tanh(sum);
    }
    const out = new Array(OUT);
    for (let m = 0; m < OUT; m++) {
      let sum = b3[m];
      for (let k = 0; k < H2; k++) sum += W3[m][k] * h2[k];
      out[m] = Math.tanh(sum);
    }
    return { h1, h2, out };
  }

  /* ============================================================
  REINFORCE + SGD
  ============================================================ */
  function endEpisodeLearn() {
    const n = episodeData.length;
    if (!n) { episodeData.length = 0; return; }
    const ret = new Array(n);
    let G = 0;
    for (let i = n - 1; i >= 0; i--) { G = episodeData[i].r + GAMMA * G; ret[i] = G; }
    let mean = 0;
    for (let i = 0; i < n; i++) mean += ret[i];
    mean /= n;
    let vr = 0;
    for (let i = 0; i < n; i++) vr += (ret[i] - mean) * (ret[i] - mean);
    const std = Math.sqrt(vr / n) + 1e-6;
    const gW1 = Array.from({ length: H1 }, () => new Array(IN).fill(0));
    const gb1 = new Array(H1).fill(0);
    const gW2 = Array.from({ length: H2 }, () => new Array(H1).fill(0));
    const gb2 = new Array(H2).fill(0);
    const gW3 = Array.from({ length: OUT }, () => new Array(H2).fill(0));
    const gb3 = new Array(OUT).fill(0);
    for (let i = 0; i < n; i++) {
      const rec = episodeData[i];
      let adv = (ret[i] - mean) / std;
      adv = clamp(adv, -ADV_CLIP, ADV_CLIP);
      const sig = rec.sig || sigma;
      const sig2 = sig * sig + 1e-8;
      const dOut = new Array(OUT);
      for (let mm = 0; mm < OUT; mm++) {
        dOut[mm] = adv * (rec.action[mm] - rec.mean[mm]) / sig2 * (1 - rec.mean[mm] * rec.mean[mm]);
      }
      for (let mm = 0; mm < OUT; mm++) {
        gb3[mm] += dOut[mm];
        for (let k = 0; k < H2; k++) gW3[mm][k] += dOut[mm] * rec.h2[k];
      }
      const dH2 = new Array(H2).fill(0);
      for (let k = 0; k < H2; k++) {
        let back = 0;
        for (let mm = 0; mm < OUT; mm++) back += W3[mm][k] * dOut[mm];
        dH2[k] = back * (1 - rec.h2[k] * rec.h2[k]);
      }
      for (let k = 0; k < H2; k++) {
        gb2[k] += dH2[k];
        for (let j = 0; j < H1; j++) gW2[k][j] += dH2[k] * rec.h1[j];
      }
      const dH1 = new Array(H1).fill(0);
      for (let j = 0; j < H1; j++) {
        let back = 0;
        for (let k = 0; k < H2; k++) back += W2[k][j] * dH2[k];
        dH1[j] = back * (1 - rec.h1[j] * rec.h1[j]);
      }
      for (let j = 0; j < H1; j++) {
        gb1[j] += dH1[j];
        for (let ii = 0; ii < IN; ii++) gW1[j][ii] += dH1[j] * rec.s[ii];
      }
    }
    const inv = 1 / n;
    for (let mm = 0; mm < OUT; mm++) {
      b3[mm] += LR * clamp(gb3[mm] * inv, -1, 1);
      b3[mm] = clamp(b3[mm], -4, 4);
      for (let k = 0; k < H2; k++) { W3[mm][k] += LR * clamp(gW3[mm][k] * inv, -1, 1); W3[mm][k] = clamp(W3[mm][k], -6, 6); }
    }
    for (let k = 0; k < H2; k++) {
      b2[k] += LR * clamp(gb2[k] * inv, -1, 1);
      b2[k] = clamp(b2[k], -4, 4);
      for (let j = 0; j < H1; j++) { W2[k][j] += LR * clamp(gW2[k][j] * inv, -1, 1); W2[k][j] = clamp(W2[k][j], -5, 5); }
    }
    for (let j = 0; j < H1; j++) {
      b1[j] += LR * clamp(gb1[j] * inv, -1, 1);
      b1[j] = clamp(b1[j], -4, 4);
      for (let ii = 0; ii < IN; ii++) { W1[j][ii] += LR * clamp(gW1[j][ii] * inv, -1, 1); W1[j][ii] = clamp(W1[j][ii], -5, 5); }
    }
    episodeData.length = 0;
  }

  /* ============================================================
  ЛОГИКА ЭПИЗОДА
  ============================================================ */
  function resetEpisode() {
    car.x = startPos.x; car.y = startPos.y; car.heading = startHeading; car.speed = MAX_SPEED * 0.6;
    distance = 0; steps = 0; episodeReward = 0; trail.length = 0;
    currentState = readLidars();
    lastLidar = currentState.slice();
    lastActionMean = [0, 0]; lastActionSample = [0, 0];
    /* камеру и эффекты НЕ сбрасываем: после аварии камера эффектно долетает до новой стартовой точки */
  }
  function stepGame(action) {
    car.heading += action[0] * TURN_RATE;
    car.speed = clamp(car.speed + action[1] * 0.15, MIN_SPEED, MAX_SPEED);
    const px = car.x, py = car.y;
    car.x += Math.cos(car.heading) * car.speed;
    car.y += Math.sin(car.heading) * car.speed;
    lastLidar = readLidars();
    const a0 = Math.atan2(py - center.y, px - center.x);
    const a1 = Math.atan2(car.y - center.y, car.x - center.x);
    let dA = a1 - a0;
    while (dA > Math.PI) dA -= 2 * Math.PI;
    while (dA < -Math.PI) dA += 2 * Math.PI;
    const prog = dA * forwardSign * meanR;
    const speedFactor = 0.6 + 0.8 * (car.speed / MAX_SPEED);
    const reward = prog * speedFactor;
    distance += prog;
    if (distance > bestDistance) bestDistance = distance;
    let done = false, result = 'движение';
    const onTrack = carOnTrack();
    if (!onTrack) { done = true; result = 'авария'; }
    steps++;
    if (!done && steps >= MAX_STEPS) { done = true; result = 'время'; }
    if (result !== 'движение') lastNotableEvent = result;
    if (done && result === 'авария') return { reward: reward + CRASH_PENALTY, done, result };
    return { reward, done, result };
  }
  /* смена трассы как «стадия»: круг пройден или трасса уверенно освоена */
  function maybeChangeTrack(res) {
    const lap = distance >= trackLength * 0.95;
    const mastered = res.result === 'время' && distance >= trackLength * 0.5;
    const stale = episodesOnTrack >= TRACK_MAX_EPISODES;
    if (!lap && !mastered && !stale) return;
    trackNum++;
    episodesOnTrack = 0;
    bestDistance = 0;
    regenTrack();
    sigma = Math.max(sigma, SIGMA_NEWTRACK);   // всплеск исследования, как стадии у lunar
    lastNotableEvent = 'трасса';
  }
  function stepOnce() {
    const s = currentState;
    const fwd = forward(s);
    const mean = fwd.out;
    lastActionMean = mean.slice();
    const sigmaNow = sigma;
    const action = [clamp(mean[0] + sigmaNow * gaussian(), -1, 1), clamp(mean[1] + sigmaNow * gaussian(), -1, 1)];
    lastActionSample = action.slice();
    const res = stepGame(action);
    trail.push({ x: car.x, y: car.y, v: car.speed });
    if (trail.length > TRAIL_MAX) trail.shift();
    episodeReward += res.reward;
    episodeData.push({ s, h1: fwd.h1, h2: fwd.h2, mean, action, r: res.reward, sig: sigmaNow });
    if (res.done) {
      if (res.result === 'авария') spawnCrash();
      episodeCount++;
      episodesOnTrack++;
      sigma = Math.max(SIGMA_MIN, sigma * SIGMA_DECAY);
      endEpisodeLearn();
      maybeChangeTrack(res);
      resetEpisode();
    } else {
      currentState = lastLidar.slice();
    }
  }

  /* ============================================================
  КАМЕРА И ЭФФЕКТЫ
  ============================================================ */
  function updateCamera() {
    const lead = 18 + 26 * (car.speed / MAX_SPEED);
    const tx = car.x + Math.cos(car.heading) * lead, ty = car.y + Math.sin(car.heading) * lead;
    camX += (tx - camX) * 0.08;
    camY += (ty - camY) * 0.08;
    const zt = VIEW_ZOOM * (1.15 - 0.35 * (car.speed / MAX_SPEED));
    camZ += (zt - camZ) * 0.05;
    if (camShake > 0.05) camShake *= 0.86; else camShake = 0;
    if (crashFlash > 0) crashFlash = Math.max(0, crashFlash - 0.06);
    for (let i = sparks.length - 1; i >= 0; i--) {
      const s = sparks[i];
      s.x += s.vx; s.y += s.vy; s.vx *= 0.92; s.vy *= 0.92; s.life -= 0.05;
      if (s.life <= 0) sparks.splice(i, 1);
    }
  }
  function spawnCrash() {
    camShake = 7; crashFlash = 1; sparks.length = 0;
    for (let i = 0; i < 8; i++) {
      const a = Math.random() * Math.PI * 2, sp = 1.5 + Math.random() * 2.5;
      sparks.push({ x: car.x, y: car.y, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp, life: 1 });
    }
  }

  /* ============================================================
  РЕНДЕР: ИГРА (камера, сетка, след, мини-карта, эффекты)
  ============================================================ */
  function drawPoly(c, poly) {
    c.beginPath();
    poly.forEach((p, i) => { if (i) c.lineTo(p.x, p.y); else c.moveTo(p.x, p.y); });
    c.closePath();
  }
  function drawCross(c, x, y, s) {
    c.beginPath();
    c.moveTo(x - s, y - s); c.lineTo(x + s, y + s);
    c.moveTo(x - s, y + s); c.lineTo(x + s, y - s);
    c.stroke();
  }
  function drawCarRect(c, x, y, h) {
    c.save();
    c.translate(x, y); c.rotate(h);
    c.fillStyle = CAR_COLOR;
    c.fillRect(-CAR_HALF_L, -CAR_HALF_W, CAR_HALF_L * 2, CAR_HALF_W * 2);
    c.fillStyle = DOT_COLOR;
    for (let i = 0; i < 5; i++) {
      const a = LIDAR_ANGLES[i];
      const dx = Math.cos(a) * (CAR_HALF_L * 0.6), dy = Math.sin(a) * (CAR_HALF_L * 0.6);
      c.fillRect(dx - 0.9, dy - 0.9, 1.8, 1.8);
    }
    c.restore();
  }
  function drawGrid() {
    const step = 60;
    const hw = GW / (2 * camZ), hh = GH / (2 * camZ);
    const x0 = Math.floor((camX - hw) / step) * step, y0 = Math.floor((camY - hh) / step) * step;
    g.strokeStyle = 'rgba(120,130,145,0.13)'; g.lineWidth = 1 / camZ;
    g.beginPath();
    for (let x = x0; x <= camX + hw; x += step) { g.moveTo(x, camY - hh); g.lineTo(x, camY + hh); }
    for (let y = y0; y <= camY + hh; y += step) { g.moveTo(camX - hw, y); g.lineTo(camX + hw, y); }
    g.stroke();
  }
  function drawTrail() {
    if (trail.length < 2) return;
    g.lineWidth = 1.4;
    for (let i = 1; i < trail.length; i++) {
      const a = i / trail.length, sp = trail[i].v / MAX_SPEED;
      const r = Math.round(150 - 97 * sp), gg = Math.round(155 + 12 * sp), b = Math.round(165 + 90 * sp);
      g.strokeStyle = `rgba(${r},${gg},${b},${(0.06 + 0.42 * a).toFixed(3)})`;
      g.beginPath(); g.moveTo(trail[i - 1].x, trail[i - 1].y); g.lineTo(trail[i].x, trail[i].y); g.stroke();
    }
  }
  function drawSparks() {
    if (!sparks.length) return;
    g.strokeStyle = SPARK_COLOR; g.lineWidth = 0.9;
    for (let i = 0; i < sparks.length; i++) {
      const s = sparks[i];
      g.globalAlpha = Math.max(0, s.life);
      g.beginPath(); g.moveTo(s.x, s.y); g.lineTo(s.x - s.vx * 2.2, s.y - s.vy * 2.2); g.stroke();
    }
    g.globalAlpha = 1;
  }
  function drawMinimap() {
    const S = 104, P = 10;
    const x0 = GW - S - P, y0 = P;
    g.save();
    g.fillStyle = 'rgba(255,255,255,0.88)'; g.strokeStyle = 'rgba(130,134,139,0.45)'; g.lineWidth = 1;
    g.fillRect(x0, y0, S, S); g.strokeRect(x0, y0, S, S);
    const b = trackBounds;
    const sc = Math.min((S - 14) / ((b.maxx - b.minx) || 1), (S - 14) / ((b.maxy - b.miny) || 1));
    const ox = x0 + (S - (b.maxx - b.minx) * sc) / 2 - b.minx * sc;
    const oy = y0 + (S - (b.maxy - b.miny) * sc) / 2 - b.miny * sc;
    g.strokeStyle = 'rgba(130,134,139,0.9)'; g.lineWidth = 1;
    g.beginPath();
    for (let i = 0; i < centerline.length; i++) {
      const X = centerline[i].x * sc + ox, Y = centerline[i].y * sc + oy;
      if (i) g.lineTo(X, Y); else g.moveTo(X, Y);
    }
    g.closePath(); g.stroke();
    g.fillStyle = MAP_START_COLOR;
    g.beginPath(); g.arc(startPos.x * sc + ox, startPos.y * sc + oy, 3, 0, Math.PI * 2); g.fill();
    g.fillStyle = ACC;
    g.beginPath(); g.arc(car.x * sc + ox, car.y * sc + oy, 2.5, 0, Math.PI * 2); g.fill();
    g.restore();
  }
  function drawGame() {
    updateCamera();
    g.clearRect(0, 0, GW, GH);
    g.fillStyle = '#f7f9fb'; g.fillRect(0, 0, GW, GH);
    const shx = camShake > 0.2 ? (Math.random() - 0.5) * camShake : 0;
    const shy = camShake > 0.2 ? (Math.random() - 0.5) * camShake : 0;
    g.save();
    g.translate(GW / 2 + shx, GH / 2 + shy);
    g.scale(camZ, camZ);
    g.translate(-camX, -camY);
    g.lineJoin = 'round'; g.lineCap = 'round';
    drawGrid();
    drawTrail();
    g.strokeStyle = WALL_COLOR; g.lineWidth = 2.5;
    drawPoly(g, outer); g.stroke();
    drawPoly(g, inner); g.stroke();
    g.strokeStyle = HIT_COLOR; g.lineWidth = 0.9;
    for (let i = 0; i < 5; i++) { const h = lastHits[i]; if (h) drawCross(g, h.x, h.y, 3); }
    drawSparks();
    drawCarRect(g, car.x, car.y, car.heading);
    g.restore();
    drawMinimap();
    if (crashFlash > 0) {
      g.strokeStyle = `rgba(${FLASH_COLOR},${(0.45 * crashFlash).toFixed(3)})`;
      g.lineWidth = 6;
      g.strokeRect(3, 3, GW - 6, GH - 6);
    }
  }

  /* ============================================================
  РЕНДЕР: СЕТЬ
  ============================================================ */
  // максимум веса после clamp (см. endEpisodeLearn: W1/W2 → ±5, W3 → ±6) —
  // используется, чтобы толщина/прозрачность линии покрывали весь диапазон
  // реально достижимых весов, а не упирались в потолок задолго до него
  const NET_WMAX = 6;
  function drawEdge(c, x1, y1, x2, y2, w) {
    const aw = Math.abs(w);
    if (aw < 0.03) return;
    const t = Math.sqrt(Math.min(1, aw / NET_WMAX));
    c.save();
    c.strokeStyle = '#000';
    c.globalAlpha = 0.12 + 0.83 * t;
    c.lineWidth = 0.4 + 2.1 * t;
    if (w < 0) c.setLineDash([2, 2]);
    c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2, y2); c.stroke();
    c.restore();
  }
  function drawNet() {
    nctx.clearRect(0, 0, GRID, GRID);
    const inputX = 95, h1X = 185, h2X = 265, outputX = 345;
    const inputY = i => 60 + i * 70, h1Y = j => 90 + j * 80, h2Y = k => 110 + k * 100, outputY = m => 160 + m * 100;
    const f = forward(currentState);
    for (let j = 0; j < H1; j++) for (let i = 0; i < IN; i++) drawEdge(nctx, inputX + 7, inputY(i), h1X - 7, h1Y(j), W1[j][i]);
    for (let k = 0; k < H2; k++) for (let j = 0; j < H1; j++) drawEdge(nctx, h1X + 7, h1Y(j), h2X - 7, h2Y(k), W2[k][j]);
    for (let m = 0; m < OUT; m++) for (let k = 0; k < H2; k++) drawEdge(nctx, h2X + 7, h2Y(k), outputX - 7, outputY(m), W3[m][k]);

    for (let i = 0; i < IN; i++) {
      const x = inputX, y = inputY(i);
      const v = clamp(Math.abs(currentState[i] || 0), 0, 1);
      nctx.beginPath(); nctx.arc(x, y, 6, 0, Math.PI * 2);
      nctx.fillStyle = `rgba(0,0,0,${0.08 + 0.85 * v})`; nctx.fill();
      nctx.strokeStyle = '#000'; nctx.lineWidth = 1; nctx.stroke();
      nctx.fillStyle = '#000'; nctx.font = '10px Inter, sans-serif'; nctx.textAlign = 'right';
      nctx.fillText(INPUT_LABELS[i][0], inputX - 11, y - 2);
      nctx.fillText(INPUT_LABELS[i][1], inputX - 11, y + 10);
    }
    for (let j = 0; j < H1; j++) {
      const x = h1X, y = h1Y(j);
      const v = clamp(Math.abs(f.h1[j] || 0), 0, 1);
      nctx.beginPath(); nctx.arc(x, y, 6, 0, Math.PI * 2);
      nctx.fillStyle = `rgba(0,0,0,${0.05 + 0.8 * v})`; nctx.fill();
      nctx.strokeStyle = '#000'; nctx.lineWidth = 1; nctx.stroke();
    }
    for (let k = 0; k < H2; k++) {
      const x = h2X, y = h2Y(k);
      const v = clamp(Math.abs(f.h2[k] || 0), 0, 1);
      nctx.beginPath(); nctx.arc(x, y, 6, 0, Math.PI * 2);
      nctx.fillStyle = `rgba(0,0,0,${0.05 + 0.8 * v})`; nctx.fill();
      nctx.strokeStyle = '#000'; nctx.lineWidth = 1; nctx.stroke();
    }
    for (let m = 0; m < OUT; m++) {
      const x = outputX, y = outputY(m);
      const v = clamp(Math.abs(f.out[m] || 0), 0, 1);
      nctx.beginPath(); nctx.arc(x, y, 6, 0, Math.PI * 2);
      nctx.fillStyle = `rgba(0,0,0,${0.08 + 0.85 * v})`; nctx.fill();
      nctx.strokeStyle = Engine.ACCENT_HEX; nctx.lineWidth = 2.5; nctx.stroke();
      nctx.textAlign = 'left';
      nctx.fillStyle = '#000'; nctx.font = '10px Inter, sans-serif';
      nctx.fillText(OUTPUT_LABELS[m], outputX + 12, y - 6);
      nctx.font = '11px ' + MONO;
      nctx.fillText(f.out[m].toFixed(2), outputX + 12, y + 8);
    }
  }

  /* ============================================================
  РЕНДЕР: СТАТИСТИКА (фиксированная ширина чисел — требование №2)
  ============================================================ */
  function drawStats() {
    statsEl.innerHTML =
      '<span>Эпизод: <span class="v">' + Engine.fmtFixed(episodeCount, 5, 0) + '</span></span>' +
      '<span>Трасса: <span class="v">' + Engine.fmtFixed(trackNum, 3, 0) + '</span></span>' +
      '<span>Дистанция: <span class="v">' + Engine.fmtFixed(Math.max(0, Math.round(distance)), 5, 0) + '</span></span>' +
      '<span>Рекорд: <span class="v">' + Engine.fmtFixed(Math.round(bestDistance), 5, 0) + '</span></span>' +
      '<span>Награда: <span class="v">' + Engine.fmtFixed(episodeReward, 5, 1) + '</span></span>';
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
  speedBtn.addEventListener('click', () => {
    fastMode = !fastMode;
    speedBtn.textContent = fastMode ? 'Обучение: ускоренное' : 'Обучение: обычное'; speedBtn.classList.toggle('eng-btn-accent', fastMode);
  });
  infoBtn.addEventListener('click', () => { modalOverlay.classList.add('is-open'); });
  modalClose.addEventListener('click', () => { modalOverlay.classList.remove('is-open'); });
  modalOk.addEventListener('click', () => { modalOverlay.classList.remove('is-open'); });
  modalOverlay.addEventListener('click', e => { if (e.target === modalOverlay) modalOverlay.classList.remove('is-open'); });
  resetBtn.addEventListener('click', () => { resetAll(); });
  weightsBtn.addEventListener('click', () => {
    Engine.openWeightsModal({
      key: DEMO_KEY,
      exportWeights: exportWeights,
      importWeights: importWeights,
      presetWeights: (window.DemoWeights && window.DemoWeights[DEMO_KEY]) || null
    });
  });

  function resetAll() {
    const fresh = randomNet();
    W1 = fresh.W1; b1 = fresh.b1; W2 = fresh.W2; b2 = fresh.b2; W3 = fresh.W3; b3 = fresh.b3;
    bestDistance = 0; episodeCount = 0; sigma = SIGMA_START;
    episodeData.length = 0;
    lastNotableEvent = '';
    trackNum = 1; episodesOnTrack = 0;
    regenTrack();
    resetEpisode();
    curve.reset();
    drawGame(); drawNet(); drawStats();
  }

  /* ============================================================
  ЭКСПОРТ / ИМПОРТ ВЕСОВ
  ============================================================ */
  function exportWeights() { return { W1, b1, W2, b2, W3, b3 }; }
  function importWeights(W) {
    try {
      if (!W || !W.W1 || !W.b1 || !W.W2 || !W.b2 || !W.W3 || !W.b3) return false;
      if (W.W1.length !== H1 || W.b1.length !== H1) return false;
      if (W.W2.length !== H2 || W.b2.length !== H2) return false;
      if (W.W3.length !== OUT || W.b3.length !== OUT) return false;
      W1 = W.W1; b1 = W.b1; W2 = W.W2; b2 = W.b2; W3 = W.W3; b3 = W.b3;
      return true;
    } catch (e) { return false; }
  }

  /* ============================================================
  ЗАСТАВКА (~31 c; технический язык уместен — в презентацию не
  идёт, показывается только при автономном открытии страницы)
  ============================================================ */
  const introOverlay = document.getElementById('racing-introOverlay');
  const introCanvas = document.getElementById('racing-introCanvas');
  const ictx = introCanvas.getContext('2d');
  let introFinished = false, introStart = null;

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

  const INTRO_LIDARS = [
    { label: 'лев-лев', value: 0.60, ang: -0.9, appear: 5.2 },
    { label: 'левее', value: 0.55, ang: -0.45, appear: 7.6 },
    { label: 'прямо', value: 0.30, ang: 0, appear: 10.0 },
    { label: 'правее', value: 0.70, ang: 0.45, appear: 12.4 },
    { label: 'прав-прав', value: 0.90, ang: 0.9, appear: 14.8 }
  ];
  const CAPS = [
    [0.0, 2.0, 'Случайная трасса-змейка. Цель — ехать быстро и далеко, не врезаться.'],
    [2.0, 5.0, '5 лидаров измеряют расстояние до стен.'],
    [5.0, 7.4, 'Вход 1: лидар левее-левее.'],
    [7.4, 9.8, 'Вход 2: лидар левее.'],
    [9.8, 12.2, 'Вход 3: лидар прямо.'],
    [12.2, 14.6, 'Вход 4: лидар правее.'],
    [14.6, 17.0, 'Вход 5: лидар правее-правее.'],
    [17.0, 21.0, 'Сеть 5 → 4 → 3 → 2: поворот и газ/тормоз.'],
    [21.0, 25.0, 'Награды: прогресс × скорость, столкновение −10.'],
    [25.0, 31.5, 'Обучение с подкреплением начинается…']
  ];
  function drawIntroCar(c, x, y, h) {
    c.save();
    c.translate(x, y); c.rotate(h);
    c.fillStyle = CAR_COLOR;
    c.fillRect(-8, -5, 16, 10);
    c.fillStyle = DOT_COLOR;
    for (let i = 0; i < 5; i++) {
      const a = LIDAR_ANGLES[i];
      c.fillRect(Math.cos(a) * 5 - 1.1, Math.sin(a) * 5 - 1.1, 2.2, 2.2);
    }
    c.restore();
  }
  function drawIntroScene(c, t, alpha) {
    if (alpha <= 0) return;
    c.save(); c.globalAlpha = alpha;
    c.strokeStyle = '#9aa0a6'; c.lineWidth = 3; c.lineJoin = 'round';
    c.beginPath(); c.moveTo(150, 120); c.lineTo(150, 450); c.stroke();
    c.beginPath(); c.moveTo(150, 210); c.lineTo(430, 165); c.stroke();
    c.beginPath(); c.moveTo(330, 450); c.quadraticCurveTo(420, 330, 430, 165); c.stroke();
    drawIntroCar(c, 230, 260, -Math.PI / 2);
    c.restore();
  }
  function drawIntroLidarRays(c, t) {
    if (t < 2.0) return;
    const cx = 230, cy = 260, heading = -Math.PI / 2;
    const appear = prog(t, 2.0, 2.6);
    INTRO_LIDARS.forEach(L => {
      const ang = heading + L.ang;
      const full = L.value * 150, len = full * appear;
      const ex = cx + Math.cos(ang) * len, ey = cy + Math.sin(ang) * len;
      const active = t >= L.appear && t < L.appear + 2.4;
      if (appear >= 1) {
        c.strokeStyle = HIT_COLOR; c.lineWidth = 1.4;
        c.beginPath();
        c.moveTo(ex - 5, ey - 5); c.lineTo(ex + 5, ey + 5);
        c.moveTo(ex - 5, ey + 5); c.lineTo(ex + 5, ey - 5);
        c.stroke();
      }
      if (active) {
        const lx = cx + Math.cos(ang) * (full + 16), ly = cy + Math.sin(ang) * (full + 16);
        c.fillStyle = '#000'; c.font = '12px ' + MONO;
        c.textAlign = Math.cos(ang) < -0.2 ? 'right' : (Math.cos(ang) > 0.2 ? 'left' : 'center');
        c.fillText(L.value.toFixed(2), lx, ly + 4);
      }
    });
  }
  function drawIntroMiniNet(c, alpha) {
    c.save(); c.globalAlpha = alpha;
    const x1 = 460, x2 = 520, x3 = 580, x4 = 640;
    const yIn = i => 292 + i * 14, yH1 = j => 296 + j * 18, yH2 = k => 302 + k * 22, yOut = m => 310 + m * 26;
    c.strokeStyle = 'rgba(0,0,0,0.12)'; c.lineWidth = 1;
    for (let i = 0; i < 5; i++) for (let j = 0; j < 4; j++) { c.beginPath(); c.moveTo(x1, yIn(i)); c.lineTo(x2, yH1(j)); c.stroke(); }
    for (let j = 0; j < 4; j++) for (let k = 0; k < 3; k++) { c.beginPath(); c.moveTo(x2, yH1(j)); c.lineTo(x3, yH2(k)); c.stroke(); }
    for (let k = 0; k < 3; k++) for (let m = 0; m < 2; m++) { c.beginPath(); c.moveTo(x3, yH2(k)); c.lineTo(x4, yOut(m)); c.stroke(); }
    c.fillStyle = '#000';
    for (let i = 0; i < 5; i++) { c.beginPath(); c.arc(x1, yIn(i), 3, 0, Math.PI * 2); c.fill(); }
    for (let j = 0; j < 4; j++) { c.beginPath(); c.arc(x2, yH1(j), 3, 0, Math.PI * 2); c.fill(); }
    for (let k = 0; k < 3; k++) { c.beginPath(); c.arc(x3, yH2(k), 3, 0, Math.PI * 2); c.fill(); }
    for (let m = 0; m < 2; m++) { c.beginPath(); c.arc(x4, yOut(m), 4, 0, Math.PI * 2); c.fill(); }
    c.font = '11px Inter, sans-serif'; c.textAlign = 'center';
    c.fillText('поворот', x4, yOut(0) + 16);
    c.fillText('газ/тормоз', x4, yOut(1) + 16);
    c.font = '11px ' + MONO;
    c.fillText('5 → 4 → 3 → 2', (x1 + x4) / 2, 372);
    c.restore();
  }
  function drawIntroPanel(c, t) {
    const x0 = 440;
    c.fillStyle = '#000'; c.font = '12px Inter, sans-serif'; c.textAlign = 'left';
    c.fillText('Вход нейросети (5 лидаров):', x0, 84);
    INTRO_LIDARS.forEach((L, i) => {
      if (t < L.appear) return;
      const alpha = prog(t, L.appear, L.appear + 0.4);
      const y = 110 + i * 26;
      c.save(); c.globalAlpha = alpha;
      c.fillStyle = '#000'; c.font = '12px Inter, sans-serif'; c.textAlign = 'left';
      c.fillText('лидар ' + L.label, x0, y);
      c.font = '12px ' + MONO; c.textAlign = 'right';
      c.fillText(L.value.toFixed(2), x0 + 260, y);
      c.restore();
    });
    const na = prog(t, 17.0, 18.0);
    if (na > 0) drawIntroMiniNet(c, na);
    const ra = prog(t, 21.0, 21.6) * (1 - prog(t, 24.8, 25.4));
    if (ra > 0) {
      c.save(); c.globalAlpha = ra;
      c.fillStyle = '#000'; c.font = '12px Inter, sans-serif'; c.textAlign = 'left';
      c.fillText('награда = прогресс × скорость', x0, 405);
      c.fillText('столкновение: −10', x0, 423);
      c.fillText('γ = 0.99 · SGD · REINFORCE', x0, 441);
      c.restore();
    }
  }
  function drawIntro(t) {
    const c = ictx;
    c.clearRect(0, 0, 720, 480);
    c.fillStyle = '#fff'; c.fillRect(0, 0, 720, 480);
    const cap = CAPS.find(x => t >= x[0] && t < x[1]);
    if (cap) { c.fillStyle = '#000'; c.font = '14px Inter, sans-serif'; c.textAlign = 'left'; c.fillText(cap[2], 14, 24); }
    const sceneA = ease(prog(t, 0.2, 1.2));
    drawIntroScene(c, t, sceneA);
    drawIntroLidarRays(c, t);
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

  /* ============================================================
  АДАПТИВНЫЕ КАНВАСЫ (требование №3)
  ============================================================ */
  let gameScale = 1, netScale = 1, gameOX = 0, gameOY = 0, netOX = 0, netOY = 0;
  Engine.resizeCanvas(gameCanvas, { onResize: (w, h) => { gameScale = Math.min(w, h) / GRID; gameOX = (w - GRID * gameScale) / 2; gameOY = (h - GRID * gameScale) / 2; } });
  Engine.resizeCanvas(netCanvas, { onResize: (w, h) => { netScale = Math.min(w, h) / GRID; netOX = (w - GRID * netScale) / 2; netOY = (h - GRID * netScale) / 2; } });
  const rawDrawGame = drawGame, rawDrawNet = drawNet;
  drawGame = function () { g.save(); g.translate(gameOX, gameOY); g.scale(gameScale, gameScale); g.beginPath(); g.rect(0, 0, GRID, GRID); g.clip(); rawDrawGame(); g.lineWidth = 3 / gameScale; g.strokeStyle = '#1a1c20'; g.strokeRect(0, 0, GRID, GRID); g.restore(); };
  drawNet = function () { nctx.save(); nctx.translate(netOX, netOY); nctx.scale(netScale, netScale); rawDrawNet(); nctx.restore(); };

  const curve = Engine.makeCurve(document.getElementById('racing-curve'), {
    primaryLabel: 'рекорд дистанции', secondaryLabel: 'награда эп.'
  });
  Engine.resizeCanvas(document.getElementById('racing-curve'), {});

  /* ============================================================
  СТАРТ
  ============================================================ */
  (function initAll() {
    const fresh = randomNet();
    W1 = fresh.W1; b1 = fresh.b1; W2 = fresh.W2; b2 = fresh.b2; W3 = fresh.W3; b3 = fresh.b3;
    resetEpisode();
    drawGame(); drawNet(); drawStats();
  })();
  parker.raf(mainLoop);
  parker.raf(introFrame);

  /* ============================================================
  КРИВАЯ ОБУЧЕНИЯ: сэмплирование раз в 250мс, пока демо активно
  ============================================================ */
  let lastEpiSample = -1;
  setInterval(() => {
    if (!parker.isRunning()) return;
    if (episodeCount === lastEpiSample) return;
    lastEpiSample = episodeCount;
    curve.sample(episodeCount, Math.round(bestDistance), Math.round(episodeReward * 100) / 100);
  }, 250);

  /* ============================================================
  РЕГИСТРАЦИЯ В КОЛОДЕ
  ============================================================ */
  Engine.register(DEMO_KEY, {
    key: DEMO_KEY,
    title: 'Racing',
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
