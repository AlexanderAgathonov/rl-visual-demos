/* Автор: А. А. Агафонов (КФУ). Лицензия: MIT. Часть материалов к статье об изобразительных средствах обучения основам RL. */
/* ==== js/demos/dino.js (demo-kit/dino_ppo.html) ==== */
(function () {
  'use strict';

  /* ============================================================
  КОНФИГ
  ============================================================ */
  const DEMO_KEY = 'dino';
  const ACC = Engine.ACCENT_HEX;
  const MONO = Engine.FONT_MONO;
  const GRID = 420;
  const METRIC_LABEL = 'Рекорд';

  /* DOM */
  const gameCanvas = document.getElementById('dino-game');
  const g = gameCanvas.getContext('2d');
  const netCanvas  = document.getElementById('dino-net');
  const nctx       = netCanvas.getContext('2d');
  const statsEl    = document.getElementById('dino-stats');
  const speedBtn   = document.getElementById('dino-speedBtn');
  const infoBtn    = document.getElementById('dino-infoBtn');
  const resetBtn   = document.getElementById('dino-resetBtn');
  const weightsBtn = document.getElementById('dino-weightsBtn');
  const modalOverlay = document.getElementById('dino-modalOverlay');
  const modalClose   = document.getElementById('dino-modalClose');
  const modalOk       = document.getElementById('dino-modalOk');

  /* ============================================================
  СПРАЙТЫ
  ============================================================ */
  const RUN1 = [
    "...........................","...........................","..............#########....",
    ".............###########...",".............##.########...",".............###########...",
    ".............###########...",".............###########...",".............###########...",
    ".............######........",".............#########.....","..#.........######.........",
    "..#.......########.........","..##....##########.........","..###..#############.......",
    "..################.#.......","..################.........","...###############.........",
    "....#############..........",".....############..........","......##########...........",
    ".......########............","........###.##.............","........##...##............",
    "........#..................","........#..................","........##................."
  ];
  const RUN2 = [
    "..........................","..........................","...............#########..",
    "..............###########.","..............##.########.","..............###########.",
    "..............###########.","..............###########.","..............###########.",
    "..............######......","..............#########...","...#.........######.......",
    "...#.......########.......","...##....##########.......","...###..#############.....",
    "...################.#.....","...################.......","....###############.......",
    ".....#############........","......############........",".......##########.........",
    "........########..........",".........###.##...........",".........#....#...........",
    ".........##...#...........","..............#...........","..............##.........."
  ];
  const DUCK1 = [
    "..................................","..................................","..................................",
    "..................................","..................................","..................................",
    "..................................","..................................","..................................",
    "..................................","..................................","..................................",
    ".###......#########....#########..",".#######################.########.","..###############################.",
    "...##############################.","....#############################.",".....############################.",
    "......#####################.......",".......##############..########...","........############..............",
    "........#..####...#...............","........#..###....##..............","........##.##.....................",
    "...........#......................","...........##....................."
  ];
  const DUCK2 = [
    "..................................","..................................","..................................",
    "..................................","..................................","..................................",
    "..................................","..................................","..................................",
    "..................................","..................................","..................................",
    ".#................................",".####.....#########....#########..",".#######################.########.",
    "..######################.########.","...##############################.","....#############################.",
    ".....############################.","......#####################.......",".......##############..########...",
    "........#############.............",".........##...###.#...............","........##........###.............",
    "........##........................","........#.........................","........##........................"
  ];
  const CACTUS = [
    "...........................###.......................","...........................###.......................",
    ".........................#######.....................",".........................#######.....................",
    ".........................#######.....................",".........................#######.....................",
    ".........................#######.....................","..................###....#######.....................",
    "..................###....#######.....................","................#####....#######.....................",
    "................#####....#######.....................","................#####....#######.....................",
    "................#####....#######.....................","................#####....#######......###............",
    "................#####....#######......###............","................#####....#######....#####............",
    "................#####....#######....#####............","................#####....#######....#####............",
    "................#####....#######....#####............","................#####....#######....#####............",
    "................#####....#######....#####............","................#####....#######....#####............",
    ".................###############....#####............","..................##############....#####............",
    "...................#############....#####............","...................#############....#####............",
    ".........................################............",".........................################............",
    ".........................###############.............",".........................########....................",
    ".........................#######.....................",".........................#######.....................",
    ".........................#######.....................",".........................#######.....................",
    "........................########.....................",".........................#######.....................",
    ".........................#######.....................",".........................#######.....................",
    ".........................#######.....................",".........................#######.....................",
    ".........................#######.....................",".........................#######....................."
  ];
  const BIRD = [
    ".........................#...........................",".........................##..........................",
    ".........................###.........................",".........................#####.......................",
    ".........................#####.......................",".........................#######.....................",
    ".........................#######.....................",".........................#######.....................",
    ".........................#######.....................","................###......#########...................",
    "...............####......#########...................","..............#######....###########.................",
    ".............########....###########.................","............###########..#############...............",
    "..........#############..##############..............","..........#############..##############..............",
    "........#######################################......","........#######################################......",
    "...................######################............","...................######################............",
    ".......................######################........",".......................######################........",
    "...........................############..............","...........................############.............."
  ];
  function compilePattern(rows) {
    let minR = 1e9, maxR = -1, minC = 1e9, maxC = -1;
    rows.forEach((r, ri) => {
      for (let ci = 0; ci < r.length; ci++) if (r[ci] === '#') {
        if (ri < minR) minR = ri; if (ri > maxR) maxR = ri;
        if (ci < minC) minC = ci; if (ci > maxC) maxC = ci;
      }
    });
    const px = [];
    rows.forEach((r, ri) => {
      for (let ci = 0; ci < r.length; ci++) if (r[ci] === '#') px.push([ci - minC, ri - minR]);
    });
    return { px, w: maxC - minC + 1, h: maxR - minR + 1 };
  }
  const SPR_RUN1   = compilePattern(RUN1);
  const SPR_RUN2   = compilePattern(RUN2);
  const SPR_DUCK1  = compilePattern(DUCK1);
  const SPR_DUCK2  = compilePattern(DUCK2);
  const SPR_CACTUS = compilePattern(CACTUS);
  const SPR_BIRD   = compilePattern(BIRD);
  const DINO_PX = 2;
  // На адаптивном канвасе ширина в CSS-пикселях редко кратна внутренней
  // сетке GRID, поэтому масштаб спрайта (scale игры × dpr) обычно дробный.
  // Если рисовать спрайт как россыпь fillRect(), их края попадают на
  // не целые пиксели устройства и браузер их сглаживает — пиксель-арт
  // становится смазанным. Фикс: один раз растеризуем паттерн в offscreen-
  // канвас 1 пиксель спрайта = 1 пиксель канваса, затем растягиваем его
  // drawImage()'ом с imageSmoothingEnabled=false — это всегда «ближайший
  // сосед», чёткие блоки при любом дробном масштабе.
  const SPRITE_BITMAP_CACHE = new Map();
  function spriteBitmap(spr, color) {
    let byColor = SPRITE_BITMAP_CACHE.get(spr);
    if (!byColor) { byColor = new Map(); SPRITE_BITMAP_CACHE.set(spr, byColor); }
    let bmp = byColor.get(color);
    if (!bmp) {
      bmp = document.createElement('canvas');
      bmp.width = spr.w; bmp.height = spr.h;
      const bctx = bmp.getContext('2d');
      bctx.fillStyle = color;
      for (const [cx, cy] of spr.px) bctx.fillRect(cx, cy, 1, 1);
      byColor.set(color, bmp);
    }
    return bmp;
  }
  function drawSprite(c, spr, x, yBottom, scale, color) {
    const bmp = spriteBitmap(spr, color);
    const w = spr.w * scale, h = spr.h * scale;
    c.save();
    c.imageSmoothingEnabled = false;
    c.drawImage(bmp, x, yBottom - h, w, h);
    c.restore();
  }

  /* ============================================================
  МИР
  ============================================================ */
  const GW = GRID, GH = GRID, GROUND = 360, DINO_X = 60;
  const GRAVITY = 0.7, JUMP_VY = -13, FAST_FALL = 1.0;
  const MAX_STEPS = 3000;
  const SPAWN_TRIGGER_GAP = 260;

  const STAGE_SPEEDS = [5, 9, 12, 12];
  const STAGE_UP_RATE = 0.8;
  const CROUCH_STAGE = 3;
  const HARD_PATTERNS_STAGE = 4;
  const BIRD_LOW = 36, BIRD_MID = 14, BIRD_HIGH = 70;
  const BIRD_CHANCE_MIN = 0.10, BIRD_CHANCE_MAX = 0.40;
  const BIRD_RAMP_EPISODES = 450, HARD_RAMP_EPISODES = 400;
  const JUMP_MASK_CLEAR = 0.5;
  const BLOCK_MERGE_GAP = 80;
  const ENABLE_CLUSTERS = true, ENABLE_COMBOS = false;
  const SPEED_JITTER = 0.5;
  const CLUSTER_GAP_MIN = 40, CLUSTER_GAP_RAND = 30, CLUSTER_CHANCE = 0.25;
  const COMBO_GAP_MIN = 560, COMBO_GAP_RAND = 160, COMBO_CHANCE = 0.12;

  /* ============================================================
  PPO ГИПЕРПАРАМЕТРЫ
  ============================================================ */
  const GAMMA = 0.99;
  const GAE_LAMBDA = 0.95;
  const PPO_ROLLOUT = 512;
  const PPO_EPOCHS = 4;
  const PPO_MINIBATCH = 64;
  const PPO_CLIP = 0.2;
  const VF_COEF = 0.5;
  const ENT_COEF = 0.01;
  const ADAM_B1 = 0.9, ADAM_B2 = 0.999, ADAM_EPS = 1e-8;
  const MAX_GRAD_NORM = 0.5;
  const BOOST_LR_BY_STAGE = [1.5e-3, 1.5e-3, 8e-4, 4e-4];
  const POST_LR_BY_STAGE  = [5e-4, 5e-4, 3e-4, 2e-4];
  const BOOST_EP_BY_STAGE = [150, 250, 400, 400];
  const BOOST_ON_FAIL_EP = 300;
  const ROLLBACK_BOOST_EP = 150;
  const ROLLBACK_LIMIT = 5;
  const ROLLBACK_AVG_FRAC = 0.35;
  const PROMOTE_MIN_EVALS = 5;
  const PROMOTE_AVG = 0.8;
  const EVAL_EVERY = 15;
  const FAST_STEPS = 60;
  const NORMAL_STEP_DELAY = 13;
  const TIMING_SHAPING = true;

  /* ============================================================
  СЕТЬ
  ============================================================ */
  const IN = 8, H1 = 16, OUT = 3;
  const INPUT_LABELS = ['до блока','шир. блока','высота','скорость','на земле','просвет','высота д.','скорость д.'];
  const OUTPUT_LABELS = ['ничего','прыжок','присед'];
  const ENC_TAG = 'ppo-dino-1.0';

  /* ============================================================
  СОСТОЯНИЕ
  ============================================================ */
  let W1, b1;
  let Wpi, bpi;
  let Wv, bv;
  let adamOpt = null;
  let bestNet = null;
  let rollbacksInStage = 0;
  let mastered = false;

  let dino = { x: DINO_X, y: GROUND, vy: 0, onGround: true, ducking: false };
  let obstacles = [];
  let worldSpeed = 5;
  let groundOffset = 0;

  let curStage = 1, stageStartEp = 0;
  let episodeSteps = 0, episodeReward = 0, episodePassed = 0, episodeCount = 0;
  let lastEpisodeReward = 0, bestReward = 0;
  let evalMode = false, lastEval = null, evalWindow = [];
  let boostUntilEp = 0;
  let lastEvent = '', lastAction = 0;
  let fastMode = false, mainStarted = false;
  let envSteps = 0, trainCalls = 0;
  let currentState = null;

  let rollState = [], rollAct = [], rollRew = [], rollDone = [];
  let rollLogp = [], rollVal = [], rollMask = [], rollVNext = [];

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  /* ============================================================
  ДИНОЗАВРИК
  ============================================================ */
  function dinoSprite() {
    if (!dino.onGround) return SPR_RUN1;
    if (dino.ducking) return (Math.floor(episodeSteps / 8) % 2) ? SPR_DUCK1 : SPR_DUCK2;
    return (Math.floor(episodeSteps / 8) % 2) ? SPR_RUN1 : SPR_RUN2;
  }
  function dinoBox() {
    const spr = dinoSprite();
    const w = spr.w * DINO_PX, h = spr.h * DINO_PX;
    return { x: dino.x + w * 0.12, y: dino.y - h * 0.92, w: w * 0.76, h: h * 0.9 };
  }

  /* ============================================================
  КУРИКУЛУМ
  ============================================================ */
  function stageSpeed(st) { return STAGE_SPEEDS[st - 1]; }
  function stageCeiling(st) {
    const sp = STAGE_SPEEDS[st - 1];
    const gapSteps = (450 + 15 * sp) / sp;
    return 0.02 * MAX_STEPS + MAX_STEPS / gapSteps;
  }
  function birdChance() {
    const since = episodeCount - stageStartEp;
    const t = clamp(since / BIRD_RAMP_EPISODES, 0, 1);
    return BIRD_CHANCE_MIN + (BIRD_CHANCE_MAX - BIRD_CHANCE_MIN) * t;
  }
  function hardChances() {
    const since = episodeCount - stageStartEp;
    const t = clamp(since / HARD_RAMP_EPISODES, 0, 1);
    return { cluster: CLUSTER_CHANCE * t, combo: COMBO_CHANCE * t };
  }
  function evalSuccessThreshold() { return 0.9 * stageCeiling(curStage); }
  function evalSuccessCount() {
    const th = evalSuccessThreshold(); let c = 0;
    for (let i = 0; i < evalWindow.length; i++) if (evalWindow[i] >= th) c++;
    return c;
  }
  function evalSuccessRate() {
    if (!evalWindow.length) return null;
    return evalSuccessCount() / evalWindow.length;
  }
  function evalAvg() {
    if (!evalWindow.length) return null;
    let s = 0; for (let i = 0; i < evalWindow.length; i++) s += evalWindow[i];
    return s / evalWindow.length;
  }

  /* ============================================================
  СПАВН
  ============================================================ */
  function spawnObstacle(x, forceType) {
    let type = 0, w, h, yTop, spr, bottom = 0;
    if (forceType !== undefined) type = forceType;
    else if (curStage >= CROUCH_STAGE && Math.random() < birdChance()) type = 1;
    if (type === 1) {
      w = SPR_BIRD.w; h = SPR_BIRD.h; spr = SPR_BIRD;
      const r = Math.random();
      bottom = r < 0.35 ? BIRD_LOW : (r < 0.7 ? BIRD_MID : BIRD_HIGH);
      yTop = GROUND - bottom - h;
    } else {
      w = SPR_CACTUS.w; h = SPR_CACTUS.h; spr = SPR_CACTUS;
      bottom = 0; yTop = GROUND - h;
    }
    const o = { x, w, h, yTop, type, spr, bottom, passed: false, shaped: false,
                clearance: (GROUND - (yTop + h)) / 100 };
    obstacles.push(o);
    return o;
  }
  function spawnGroup(x) {
    const o = spawnObstacle(x);
    const hard = curStage >= HARD_PATTERNS_STAGE;
    if (!hard) return;
    const ch = hardChances();
    if (ENABLE_CLUSTERS && o.type === 0 && Math.random() < ch.cluster) {
      const cx = x + o.w + CLUSTER_GAP_MIN + Math.random() * CLUSTER_GAP_RAND;
      spawnObstacle(cx, 0);
      return;
    }
    if (ENABLE_COMBOS && Math.random() < ch.combo) {
      spawnObstacle(x + o.w + COMBO_GAP_MIN + Math.random() * COMBO_GAP_RAND, o.type === 0 ? 1 : 0);
    }
  }
  function spawnIfNeeded() {
    if (obstacles.length === 0) { spawnGroup(GW + 100 + worldSpeed * 15); return; }
    const last = obstacles[obstacles.length - 1];
    if (last.x < GW - SPAWN_TRIGGER_GAP) {
      spawnGroup(GW + 100 + Math.random() * 180 + worldSpeed * 15);
    }
  }

  /* ============================================================
  ЭПИЗОД
  ============================================================ */
  function resetEpisode() {
    dino = { x: DINO_X, y: GROUND, vy: 0, onGround: true, ducking: false };
    obstacles.length = 0;
    worldSpeed = clamp(stageSpeed(curStage) + (Math.random() * 2 - 1) * SPEED_JITTER, 3, 12);
    spawnObstacle(GW + 120 + worldSpeed * 15);
    spawnObstacle(GW + 460 + worldSpeed * 15);
    episodeSteps = 0; episodeReward = 0; episodePassed = 0;
    currentState = readState();
  }

  /* ============================================================
  НАБЛЮДЕНИЕ
  ============================================================ */
  function nearestObstacle() {
    const b = dinoBox();
    let bestO = null, bd = 1e9;
    for (const o of obstacles) {
      if (o.x + o.w > b.x - 2) {
        const d = o.x - b.x;
        if (d < bd) { bd = d; bestO = o; }
      }
    }
    return bestO;
  }
  function blockEnd(nearest) {
    let end = nearest.x + nearest.w, changed = true;
    while (changed) {
      changed = false;
      for (const o of obstacles) {
        if (o.x >= nearest.x - 1 && o.x <= end + BLOCK_MERGE_GAP) {
          if (o.x + o.w > end) { end = o.x + o.w; changed = true; }
        }
      }
    }
    return end;
  }
  function readState() {
    const o = nearestObstacle();
    const speedT = clamp(worldSpeed / 12, 0, 1);
    const onGround = dino.onGround ? 1 : 0;
    const b = dinoBox();
    const dinoH = clamp((GROUND - b.y) / 120, 0, 1);
    const dinoV = clamp(dino.vy / 15, -1, 1);
    if (!o) return [1, 0, 0, speedT, onGround, 1, dinoH, dinoV];
    const d = Math.max(0, o.x - b.x);
    const timeTo = clamp(d / (worldSpeed * 60), 0, 1);
    const blockW = clamp((blockEnd(o) - o.x) / 240, 0, 1);
    const height = clamp(o.h / 60, 0, 1);
    const clearance = clamp(o.clearance || 0, 0, 1);
    return [timeTo, blockW, height, speedT, onGround, clearance, dinoH, dinoV];
  }

  /* ============================================================
  SHAPING
  ============================================================ */
  function timingShaping(action) {
    if (!TIMING_SHAPING) return 0;
    if (!dino.onGround) return 0;
    const o0 = nearestObstacle(); if (!o0) return 0;
    if (o0.shaped) return 0;
    const b = dinoBox();
    const stFrames = Math.max(0, o0.x - b.x) / worldSpeed;
    const stSec = stFrames / 60;
    const clear = o0.clearance;
    const needsJump = clear < 0.3;
    const needsDuck = clear >= 0.3 && clear < 0.5;
    const highBird = clear >= 0.5;
    let bonus = 0;
    if (needsJump) {
      const blockLen = blockEnd(o0) - o0.x;
      const L = blockLen / (worldSpeed * 60);
      const hi = clamp(0.56 - L, 0.15, 0.51);
      if (action === 1 && stSec > 0.10 && stSec < hi) bonus = 0.5;
      else if (stSec < 0.17 && action !== 1) bonus = -0.2;
    } else if (needsDuck) {
      if (action === 2 && stSec > 0.03 && stSec < 0.43) bonus = 0.5;
      else if (stSec < 0.17 && action !== 2) bonus = -0.2;
    } else if (highBird) {
      if (action === 0 && stSec < 0.30) bonus = 0.1;
    }
    if (bonus !== 0) o0.shaped = true;
    return bonus;
  }

  /* ============================================================
  ШАГ СРЕДЫ
  ============================================================ */
  function stepGame(action) {
    let reward = 0.02 + timingShaping(action);
    dino.ducking = (action === 2) && dino.onGround;
    if (action === 1 && dino.onGround) { dino.vy = JUMP_VY; dino.onGround = false; }
    if (!dino.onGround) {
      dino.vy += (action === 2 ? FAST_FALL : GRAVITY);
      dino.y += dino.vy;
      if (dino.y >= GROUND) { dino.y = GROUND; dino.vy = 0; dino.onGround = true; }
    }
    for (let i = obstacles.length - 1; i >= 0; i--) {
      const o = obstacles[i];
      o.x -= worldSpeed;
      const b = dinoBox();
      if (!o.passed && o.x + o.w < b.x) { o.passed = true; episodePassed++; reward += 1.0; }
      if (o.x + o.w < -60) obstacles.splice(i, 1);
    }
    groundOffset += worldSpeed;
    spawnIfNeeded();
    const b = dinoBox();
    let hit = false;
    for (const o of obstacles) {
      const ol = o.x + 2, orr = o.x + o.w - 2, ot = o.yTop + 2, ob = o.yTop + o.h - 2;
      if (b.x + b.w > ol && b.x < orr && b.y + b.h > ot && b.y < ob) { hit = true; break; }
    }
    episodeSteps++;
    let doneEnv = false, terminal = false;
    if (hit) { reward += -10.0; doneEnv = true; terminal = true; lastEvent = 'удар'; }
    else if (episodeSteps >= MAX_STEPS) { doneEnv = true; terminal = false; lastEvent = 'время'; }
    return { reward, doneEnv, terminal };
  }

  /* ============================================================
  СЕТЬ
  ============================================================ */
  function randomNet() {
    const s1 = Math.sqrt(2.0 / IN), spi = Math.sqrt(2.0 / H1);
    return {
      W1: Array.from({ length: H1 }, () =>
        Array.from({ length: IN }, () => (Math.random() - 0.5) * s1 * 2)),
      b1: new Array(H1).fill(0),
      Wpi: Array.from({ length: OUT }, () =>
        Array.from({ length: H1 }, () => (Math.random() - 0.5) * spi * 2)),
      bpi: new Array(OUT).fill(0),
      Wv: Array.from({ length: 1 }, () =>
        Array.from({ length: H1 }, () => (Math.random() - 0.5) * spi * 2)),
      bv: [0]
    };
  }
  function copyNet(n) {
    return {
      W1: n.W1.map(r => r.slice()), b1: n.b1.slice(),
      Wpi: n.Wpi.map(r => r.slice()), bpi: n.bpi.slice(),
      Wv: n.Wv.map(r => r.slice()), bv: n.bv.slice()
    };
  }
  function forward(s) {
    const h1 = new Array(H1);
    for (let j = 0; j < H1; j++) {
      let sum = b1[j];
      for (let i = 0; i < IN; i++) sum += W1[j][i] * s[i];
      h1[j] = Math.tanh(sum);
    }
    const logits = new Array(OUT);
    for (let m = 0; m < OUT; m++) {
      let sum = bpi[m];
      for (let j = 0; j < H1; j++) sum += Wpi[m][j] * h1[j];
      logits[m] = sum;
    }
    let v = bv[0];
    for (let j = 0; j < H1; j++) v += Wv[0][j] * h1[j];
    return { h1, logits, v };
  }
  function clipWeights() {
    for (let j = 0; j < H1; j++) {
      b1[j] = clamp(b1[j], -4, 4);
      for (let i = 0; i < IN; i++) W1[j][i] = clamp(W1[j][i], -5, 5);
    }
    for (let m = 0; m < OUT; m++) {
      bpi[m] = clamp(bpi[m], -4, 4);
      for (let j = 0; j < H1; j++) Wpi[m][j] = clamp(Wpi[m][j], -6, 6);
    }
    bv[0] = clamp(bv[0], -20, 20);
    for (let j = 0; j < H1; j++) Wv[0][j] = clamp(Wv[0][j], -6, 6);
  }
  function actionMask(s) {
    const onGround = s[4] > 0.5, clear = s[5];
    const m = [1, 1, 1];
    if (!onGround || clear >= JUMP_MASK_CLEAR) m[1] = 0;
    if (curStage < CROUCH_STAGE) m[2] = 0;
    return m;
  }
  function maskedSoftmax(logits, mask) {
    let maxL = -Infinity;
    for (let m = 0; m < OUT; m++) if (mask[m] && logits[m] > maxL) maxL = logits[m];
    let sum = 0;
    for (let m = 0; m < OUT; m++) if (mask[m]) sum += Math.exp(logits[m] - maxL);
    const logZ = Math.log(sum + 1e-12) + maxL;
    const probs = new Array(OUT).fill(0);
    const logProbs = new Array(OUT).fill(-Infinity);
    for (let m = 0; m < OUT; m++) if (mask[m]) {
      logProbs[m] = logits[m] - logZ;
      probs[m] = Math.exp(logProbs[m]);
    }
    return { probs, logProbs };
  }
  function sampleAction(probs) {
    const r = Math.random(); let cum = 0;
    for (let m = 0; m < OUT; m++) { cum += probs[m]; if (r < cum) return m; }
    return 0;
  }
  function greedyActionFromLogits(logits, mask) {
    let best = -1, bestV = -Infinity;
    for (let m = 0; m < OUT; m++) if (mask[m] && logits[m] > bestV) { bestV = logits[m]; best = m; }
    return best < 0 ? 0 : best;
  }

  /* ============================================================
  ADAM
  ============================================================ */
  function makeAdam() {
    return {
      mW1: Array.from({ length: H1 }, () => new Array(IN).fill(0)),
      vW1: Array.from({ length: H1 }, () => new Array(IN).fill(0)),
      mb1: new Array(H1).fill(0), vb1: new Array(H1).fill(0),
      mWpi: Array.from({ length: OUT }, () => new Array(H1).fill(0)),
      vWpi: Array.from({ length: OUT }, () => new Array(H1).fill(0)),
      mbpi: new Array(OUT).fill(0), vbpi: new Array(OUT).fill(0),
      mWv: new Array(H1).fill(0), vWv: new Array(H1).fill(0),
      mbv: [0], vbv: [0], t: 0
    };
  }
  function adamApply(lr, gW1, gb1, gWpi, gbpi, gWv, gbv) {
    const o = adamOpt; o.t++;
    const bc1 = 1 - Math.pow(ADAM_B1, o.t);
    const bc2 = 1 - Math.pow(ADAM_B2, o.t);
    const step = (gr, m, v) => {
      const mN = ADAM_B1 * m + (1 - ADAM_B1) * gr;
      const vN = ADAM_B2 * v + (1 - ADAM_B2) * gr * gr;
      return [mN, vN, lr * (mN / bc1) / (Math.sqrt(vN / bc2) + ADAM_EPS)];
    };
    for (let j = 0; j < H1; j++) {
      for (let i = 0; i < IN; i++) {
        const r = step(gW1[j][i], o.mW1[j][i], o.vW1[j][i]);
        o.mW1[j][i] = r[0]; o.vW1[j][i] = r[1];
        W1[j][i] -= r[2];
      }
      const rb = step(gb1[j], o.mb1[j], o.vb1[j]);
      o.mb1[j] = rb[0]; o.vb1[j] = rb[1]; b1[j] -= rb[2];
    }
    for (let m = 0; m < OUT; m++) {
      for (let j = 0; j < H1; j++) {
        const r = step(gWpi[m][j], o.mWpi[m][j], o.vWpi[m][j]);
        o.mWpi[m][j] = r[0]; o.vWpi[m][j] = r[1];
        Wpi[m][j] -= r[2];
      }
      const rb = step(gbpi[m], o.mbpi[m], o.vbpi[m]);
      o.mbpi[m] = rb[0]; o.vbpi[m] = rb[1]; bpi[m] -= rb[2];
    }
    for (let j = 0; j < H1; j++) {
      const r = step(gWv[j], o.mWv[j], o.vWv[j]);
      o.mWv[j] = r[0]; o.vWv[j] = r[1]; Wv[0][j] -= r[2];
    }
    const rbv = step(gbv[0], o.mbv[0], o.vbv[0]);
    o.mbv[0] = rbv[0]; o.vbv[0] = rbv[1]; bv[0] -= rbv[2];
  }

  /* ============================================================
  PPO
  ============================================================ */
  function currentLR() {
    const si = curStage - 1;
    const inBoost = episodeCount < boostUntilEp;
    let lr = inBoost ? BOOST_LR_BY_STAGE[si] : POST_LR_BY_STAGE[si];
    if (mastered) lr *= 0.25;
    return lr;
  }
  function ppoUpdate() {
    const N = rollState.length;
    if (N < 2) return clearRollout();

    const adv = new Float32Array(N);
    const ret = new Float32Array(N);
    let lastGae = 0;
    for (let t = N - 1; t >= 0; t--) {
      const nonTerm = rollDone[t] ? 0 : 1;
      const delta = rollRew[t] + GAMMA * rollVNext[t] * nonTerm - rollVal[t];
      lastGae = delta + GAMMA * GAE_LAMBDA * nonTerm * lastGae;
      adv[t] = lastGae;
      ret[t] = lastGae + rollVal[t];
    }
    let mean = 0;
    for (let i = 0; i < N; i++) mean += adv[i];
    mean /= N;
    let sq = 0;
    for (let i = 0; i < N; i++) { const d = adv[i] - mean; sq += d * d; }
    const std = Math.sqrt(sq / N) + 1e-8;
    for (let i = 0; i < N; i++) adv[i] = (adv[i] - mean) / std;

    const lr = currentLR();
    let lossSum = 0, entSum = 0, klSum = 0;

    for (let epoch = 0; epoch < PPO_EPOCHS; epoch++) {
      const idx = new Uint32Array(N);
      for (let i = 0; i < N; i++) idx[i] = i;
      for (let i = N - 1; i > 0; i--) {
        const j = (Math.random() * (i + 1)) | 0;
        const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
      }
      for (let mb = 0; mb < N; mb += PPO_MINIBATCH) {
        const end = Math.min(mb + PPO_MINIBATCH, N);
        const mbSize = end - mb;

        const gW1 = Array.from({ length: H1 }, () => new Array(IN).fill(0));
        const gb1 = new Array(H1).fill(0);
        const gWpi = Array.from({ length: OUT }, () => new Array(H1).fill(0));
        const gbpi = new Array(OUT).fill(0);
        const gWv = new Array(H1).fill(0);
        const gbv = [0];

        for (let k = mb; k < end; k++) {
          const i = idx[k];
          const s = rollState[i];
          const a = rollAct[i];
          const mask = rollMask[i];
          const oldLogp = rollLogp[i];
          const A = adv[i];
          const R = ret[i];

          const h1 = new Array(H1);
          for (let j = 0; j < H1; j++) {
            let sum = b1[j];
            for (let ii = 0; ii < IN; ii++) sum += W1[j][ii] * s[ii];
            h1[j] = Math.tanh(sum);
          }
          const logits = new Array(OUT);
          for (let m = 0; m < OUT; m++) {
            let sum = bpi[m];
            for (let j = 0; j < H1; j++) sum += Wpi[m][j] * h1[j];
            logits[m] = sum;
          }
          let v = bv[0];
          for (let j = 0; j < H1; j++) v += Wv[0][j] * h1[j];

          const { probs, logProbs } = maskedSoftmax(logits, mask);
          const newLogp = logProbs[a];
          const ratio = Math.exp(newLogp - oldLogp);
          const unclipped = ratio * A;
          const clipped = clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * A;
          const useUnclipped = unclipped <= clipped;
          const gLogp = useUnclipped ? -A * ratio : 0;

          let H = 0;
          for (let m = 0; m < OUT; m++) if (probs[m] > 1e-12) H -= probs[m] * Math.log(probs[m]);
          entSum += H;
          klSum += oldLogp - newLogp;

          const dLogits = new Array(OUT).fill(0);
          for (let m = 0; m < OUT; m++) {
            if (!mask[m]) continue;
            const delta = (m === a) ? 1 : 0;
            const gp = gLogp * (delta - probs[m]);
            const ge = ENT_COEF * probs[m] * (Math.log(probs[m] + 1e-12) + H);
            dLogits[m] = gp + ge;
          }

          const vErr = v - R;
          lossSum += 0.5 * vErr * vErr;

          gbv[0] += VF_COEF * vErr;
          for (let j = 0; j < H1; j++) gWv[j] += VF_COEF * vErr * h1[j];

          for (let m = 0; m < OUT; m++) {
            gbpi[m] += dLogits[m];
            for (let j = 0; j < H1; j++) gWpi[m][j] += dLogits[m] * h1[j];
          }

          const dH1 = new Array(H1).fill(0);
          for (let j = 0; j < H1; j++) {
            let sum = 0;
            for (let m = 0; m < OUT; m++) sum += Wpi[m][j] * dLogits[m];
            sum += VF_COEF * vErr * Wv[0][j];
            dH1[j] = sum;
          }
          for (let j = 0; j < H1; j++) {
            const dz = dH1[j] * (1 - h1[j] * h1[j]);
            gb1[j] += dz;
            for (let ii = 0; ii < IN; ii++) gW1[j][ii] += dz * s[ii];
          }
        }

        const inv = 1 / mbSize;
        for (let j = 0; j < H1; j++) {
          gb1[j] *= inv; for (let i = 0; i < IN; i++) gW1[j][i] *= inv;
        }
        for (let m = 0; m < OUT; m++) {
          gbpi[m] *= inv; for (let j = 0; j < H1; j++) gWpi[m][j] *= inv;
        }
        for (let j = 0; j < H1; j++) gWv[j] *= inv;
        gbv[0] *= inv;

        let gnorm = 0;
        for (let j = 0; j < H1; j++) {
          gnorm += gb1[j] * gb1[j];
          for (let i = 0; i < IN; i++) gnorm += gW1[j][i] * gW1[j][i];
        }
        for (let m = 0; m < OUT; m++) {
          gnorm += gbpi[m] * gbpi[m];
          for (let j = 0; j < H1; j++) gnorm += gWpi[m][j] * gWpi[m][j];
        }
        for (let j = 0; j < H1; j++) gnorm += gWv[j] * gWv[j];
        gnorm += gbv[0] * gbv[0];
        gnorm = Math.sqrt(gnorm);
        if (gnorm > MAX_GRAD_NORM) {
          const sc = MAX_GRAD_NORM / (gnorm + 1e-8);
          for (let j = 0; j < H1; j++) {
            gb1[j] *= sc; for (let i = 0; i < IN; i++) gW1[j][i] *= sc;
          }
          for (let m = 0; m < OUT; m++) {
            gbpi[m] *= sc; for (let j = 0; j < H1; j++) gWpi[m][j] *= sc;
          }
          for (let j = 0; j < H1; j++) gWv[j] *= sc;
          gbv[0] *= sc;
        }
        adamApply(lr, gW1, gb1, gWpi, gbpi, gWv, gbv);
      }
    }

    trainCalls++;
    clipWeights();
    clearRollout();
  }
  function clearRollout() {
    rollState.length = 0; rollAct.length = 0; rollRew.length = 0;
    rollDone.length = 0; rollLogp.length = 0; rollVal.length = 0;
    rollMask.length = 0; rollVNext.length = 0;
  }

  /* ============================================================
  ШАГ ОБУЧЕНИЯ
  ============================================================ */
  function stepOnce() {
    const s = currentState;
    const mask = actionMask(s);
    const f = forward(s);
    const { probs, logProbs } = maskedSoftmax(f.logits, mask);

    let a;
    if (evalMode) a = greedyActionFromLogits(f.logits, mask);
    else a = sampleAction(probs);
    const logp = logProbs[a];
    lastAction = a;

    const res = stepGame(a);
    episodeReward += res.reward;
    envSteps++;

    const ns = readState();

    let vNext = 0;
    if (!res.doneEnv) vNext = forward(ns).v;

    if (!evalMode) {
      rollState.push(s); rollAct.push(a); rollRew.push(res.reward);
      rollDone.push(res.doneEnv ? 1 : 0);
      rollLogp.push(logp); rollVal.push(f.v); rollMask.push(mask);
      rollVNext.push(vNext);
    }

    if (res.doneEnv) {
      episodeCount++;
      lastEpisodeReward = episodeReward;
      if (episodeReward > bestReward) bestReward = episodeReward;

      if (evalMode) {
        lastEval = episodeReward;
        evalWindow.push(episodeReward);
        if (evalWindow.length > 10) evalWindow.shift();

        if (evalWindow.length >= 10 && evalSuccessCount() === 0 &&
            episodeCount >= boostUntilEp) {
          boostUntilEp = episodeCount + BOOST_ON_FAIL_EP;
        }

        if (bestNet && rollbacksInStage < ROLLBACK_LIMIT && bestReward > 50 &&
            episodeCount >= boostUntilEp && evalWindow.length >= 10 &&
            evalAvg() < ROLLBACK_AVG_FRAC * bestReward) {
          rollbacksInStage++;
          W1 = bestNet.W1.map(r => r.slice()); b1 = bestNet.b1.slice();
          Wpi = bestNet.Wpi.map(r => r.slice()); bpi = bestNet.bpi.slice();
          Wv = bestNet.Wv.map(r => r.slice()); bv = bestNet.bv.slice();
          adamOpt = makeAdam();
          clearRollout();
          boostUntilEp = episodeCount + ROLLBACK_BOOST_EP;
          evalWindow.length = 0;
          mastered = false;
        }

        mastered = evalWindow.length >= PROMOTE_MIN_EVALS &&
                   evalSuccessRate() >= STAGE_UP_RATE;

        if (curStage < 4 && evalWindow.length >= PROMOTE_MIN_EVALS) {
          const rate = evalSuccessRate();
          const avg = evalAvg();
          if (rate >= STAGE_UP_RATE || avg >= PROMOTE_AVG * stageCeiling(curStage)) {
            curStage++;
            stageStartEp = episodeCount;
            boostUntilEp = episodeCount + BOOST_EP_BY_STAGE[curStage - 1];
            evalWindow.length = 0;
            mastered = false;
            rollbacksInStage = 0;
          }
        }

        if (!bestNet && bestReward > 0) bestNet = copyNet({ W1, b1, Wpi, bpi, Wv, bv });
        else if (bestNet && episodeReward >= bestReward) {
          bestNet = copyNet({ W1, b1, Wpi, bpi, Wv, bv });
        }
      }

      evalMode = (episodeCount % EVAL_EVERY === 0);
      resetEpisode();
    } else {
      currentState = ns;
    }

    if (rollState.length >= PPO_ROLLOUT) ppoUpdate();
  }

  /* ============================================================
  РЕНДЕР ИГРЫ
  ============================================================ */
  function drawGame() {
    g.clearRect(0, 0, GW, GH);
    g.fillStyle = '#fff';
    g.fillRect(0, 0, GW, GH);
    g.strokeStyle = '#000';
    g.lineWidth = 2;
    g.beginPath();
    g.moveTo(0, GROUND + 1);
    g.lineTo(GW, GROUND + 1);
    g.stroke();
    for (const o of obstacles) drawSprite(g, o.spr, o.x, o.yTop + o.h, 1, '#000');
    drawSprite(g, dinoSprite(), dino.x, dino.y, DINO_PX, ACC);
  }

  /* ============================================================
  РЕНДЕР СЕТИ
  ============================================================ */
  // максимум веса после clamp (см. train(): W1 → ±5, Wpi/Wv → ±6) —
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
    const inputX = 80, h1X = 200, outX = 340;
    const cy = GRID / 2;
    // каждая колонка центрируется по вертикали относительно cy, чтобы
    // сеть выглядела симметрично несмотря на разное число узлов в колонках
    const inSpan = (IN - 1) * 42, inStart = cy - inSpan / 2;
    const inputY = i => inStart + i * 42;
    const h1Span = (H1 - 1) * 22, h1Start = cy - h1Span / 2;
    const h1Y = j => h1Start + j * 22;
    const polSpacing = 80, valGap = 110;
    const outSpan = (OUT - 1) * polSpacing + valGap, outStart = cy - outSpan / 2;
    const polY = m => outStart + m * polSpacing;
    const valY = outStart + (OUT - 1) * polSpacing + valGap;

    const f = forward(currentState);
    const mask = actionMask(currentState);
    const { probs } = maskedSoftmax(f.logits, mask);

    for (let j = 0; j < H1; j++) {
      for (let i = 0; i < IN; i++)
        drawEdge(nctx, inputX + 7, inputY(i), h1X - 7, h1Y(j), W1[j][i]);
    }
    for (let m = 0; m < OUT; m++) {
      for (let j = 0; j < H1; j++)
        drawEdge(nctx, h1X + 7, h1Y(j), outX - 8, polY(m), Wpi[m][j]);
    }
    for (let j = 0; j < H1; j++)
      drawEdge(nctx, h1X + 7, h1Y(j), outX - 8, valY, Wv[0][j]);

    for (let i = 0; i < IN; i++) {
      const x = inputX, y = inputY(i);
      const v = clamp(Math.abs(currentState[i] || 0), 0, 1);
      nctx.beginPath(); nctx.arc(x, y, 5, 0, Math.PI * 2);
      nctx.fillStyle = `rgba(0,0,0,${0.08 + 0.85 * v})`; nctx.fill();
      nctx.strokeStyle = '#000'; nctx.lineWidth = 1; nctx.stroke();
      nctx.fillStyle = '#000'; nctx.font = '10px ' + MONO; nctx.textAlign = 'right';
      nctx.fillText(INPUT_LABELS[i], inputX - 8, y + 3);
    }
    for (let j = 0; j < H1; j++) {
      const x = h1X, y = h1Y(j);
      const v = clamp(Math.abs(f.h1[j] || 0), 0, 1);
      nctx.beginPath(); nctx.arc(x, y, 4, 0, Math.PI * 2);
      nctx.fillStyle = `rgba(0,0,0,${0.05 + 0.8 * v})`; nctx.fill();
      nctx.strokeStyle = '#000'; nctx.lineWidth = 1; nctx.stroke();
    }
    for (let m = 0; m < OUT; m++) {
      const x = outX, y = polY(m);
      const v = clamp(probs[m], 0, 1);
      const isMasked = !mask[m];
      nctx.beginPath(); nctx.arc(x, y, 8, 0, Math.PI * 2);
      nctx.fillStyle = `rgba(0,0,0,${0.08 + 0.85 * v})`; nctx.fill();
      nctx.strokeStyle = m === lastAction ? ACC : '#000';
      nctx.lineWidth = m === lastAction ? 2.5 : 1.5;
      nctx.stroke();
      nctx.fillStyle = isMasked ? '#aaa' : '#000';
      nctx.font = '10px ' + MONO; nctx.textAlign = 'left';
      const p = isMasked ? '—' : (probs[m] * 100).toFixed(0) + '%';
      nctx.fillText(`${OUTPUT_LABELS[m]} ${p}`, outX + 12, y + 3);
    }
    nctx.beginPath(); nctx.arc(outX, valY, 8, 0, Math.PI * 2);
    nctx.fillStyle = `rgba(0,0,0,${0.05 + 0.6 * clamp(Math.abs(f.v) / 20, 0, 1)})`; nctx.fill();
    nctx.strokeStyle = '#000'; nctx.lineWidth = 1.5; nctx.stroke();
    nctx.fillStyle = '#000'; nctx.font = '10px ' + MONO; nctx.textAlign = 'left';
    nctx.fillText('V ' + f.v.toFixed(2), outX + 12, valY + 3);
  }

  /* ============================================================
  СТАТИСТИКА
  ============================================================ */
  function drawStats() {
    statsEl.innerHTML =
      '<span>Эпизод: <span class="v">' + Engine.fmtFixed(episodeCount, 5, 0) + '</span></span>' +
      '<span>Стадия: <span class="v">' + curStage + '/4</span></span>' +
      '<span>Успехов: <span class="v">' + (evalWindow.length ? evalSuccessCount() + '/' + evalWindow.length : '—') + '</span></span>' +
      '<span>Награда: <span class="v">' + Engine.fmtFixed(episodeReward, 3, 1) + '</span></span>' +
      '<span>Лучшая награда: <span class="v">' + Engine.fmtFixed(bestReward, 3, 1) + '</span></span>';
  }

  /* ============================================================
  ГЛАВНЫЙ ЦИКЛ + ПАРКОВЩИК rAF
  ============================================================ */
  let lastNormalStep = 0;
  function mainLoop(ts) {
    if (mainStarted) {
      if (fastMode) { for (let i = 0; i < FAST_STEPS; i++) stepOnce(); }
      else if (ts - lastNormalStep >= NORMAL_STEP_DELAY) {
        stepOnce(); lastNormalStep = ts;
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
    if (t >= 10.0) { finishIntro(); return; }
    parker.raf(introFrame);
  }
  const parker = Engine.makeRafParker({
    getMain: function () { return mainLoop; },
    getIntro: function () { return introFrame; }
  });

  /* ============================================================
  АДАПТИВНЫЕ КАНВАСЫ
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
  const curve = Engine.makeCurve(document.getElementById('dino-curve'), {
    primaryLabel: 'рекорд', secondaryLabel: 'награда эп.'
  });
  Engine.resizeCanvas(document.getElementById('dino-curve'), {});
  let lastEpiSample = -1;
  setInterval(() => {
    if (!parker.isRunning()) return;
    if (episodeCount === lastEpiSample) return;
    lastEpiSample = episodeCount;
    curve.sample(episodeCount, Math.round(bestReward * 100) / 100, Math.round(lastEpisodeReward * 100) / 100);
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
    const fresh = randomNet();
    W1 = fresh.W1; b1 = fresh.b1;
    Wpi = fresh.Wpi; bpi = fresh.bpi;
    Wv = fresh.Wv; bv = fresh.bv;
    adamOpt = makeAdam();
    bestNet = null;
    rollbacksInStage = 0;
    mastered = false;
    clearRollout();
    episodeCount = 0; envSteps = 0; trainCalls = 0;
    bestReward = 0; lastEpisodeReward = 0;
    lastEval = null; evalMode = false; evalWindow.length = 0;
    curStage = 1; stageStartEp = 0;
    boostUntilEp = BOOST_EP_BY_STAGE[0];
    worldSpeed = STAGE_SPEEDS[0]; groundOffset = 0;
    lastEvent = '';
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
  ЭКСПОРТ / ИМПОРТ
  ============================================================ */
  function exportWeights() {
    return {
      enc: ENC_TAG,
      net: { W1, b1, Wpi, bpi, Wv, bv },
      best: { reward: bestReward, success: evalSuccessRate(), stage: curStage }
    };
  }
  function importWeights(obj) {
    try {
      if (!obj || obj.enc !== ENC_TAG) return false;
      const W = obj.net ? obj.net : obj;
      if (!W || !W.W1 || !W.b1 || !W.Wpi || !W.bpi || !W.Wv || !W.bv) return false;
      if (W.W1.length !== H1 || W.b1.length !== H1) return false;
      if (W.Wpi.length !== OUT || W.bpi.length !== OUT) return false;
      W1 = W.W1.map(r => r.slice()); b1 = W.b1.slice();
      Wpi = W.Wpi.map(r => r.slice()); bpi = W.bpi.slice();
      Wv = W.Wv.map(r => r.slice()); bv = W.bv.slice();
      adamOpt = makeAdam();
      bestNet = null; rollbacksInStage = 0;
      if (obj.best && Number.isFinite(obj.best.reward)) bestReward = obj.best.reward;
      if (obj.best && Number.isFinite(obj.best.stage)) {
        curStage = clamp(Math.round(obj.best.stage), 1, 4);
        stageStartEp = episodeCount - 1000;
        boostUntilEp = 0; mastered = false;
        evalWindow.length = 0;
      }
      return true;
    } catch (e) { return false; }
  }

  /* ============================================================
  ЗАСТАВКА
  ============================================================ */
  const introOverlay = document.getElementById('dino-introOverlay');
  const introCanvas  = document.getElementById('dino-introCanvas');
  const ictx = introCanvas.getContext('2d');
  let introFinished = false, introStart = null;

  function finishIntro() { if (introFinished) return; introFinished = true; introOverlay.style.display = 'none'; mainStarted = true; }
  introOverlay.addEventListener('click', finishIntro);

  function drawIntro(t) {
    const c = ictx;
    c.clearRect(0, 0, 720, 480);
    c.fillStyle = '#fff'; c.fillRect(0, 0, 720, 480);
    c.fillStyle = '#000'; c.font = '18px Inter, sans-serif'; c.textAlign = 'left';
    c.fillText('Динозаврик · PPO (актёр-критик)', 30, 60);
    c.font = '14px Inter, sans-serif';
    if (t >= 1) {
      c.fillText('8 чисел → ствол 16 → политика 3 + ценность 1.', 30, 120);
    }
    if (t >= 3) {
      c.fillText('Rollout 512 шагов → GAE(λ=0.95) → 4 эпохи по 64.', 30, 155);
      c.fillText('Клиппированная цель ε = 0.2, энтропия 0.01, value 0.5.', 30, 180);
    }
    if (t >= 6) {
      c.fillText('Маски действий: прыжок только на земле и не под птицей.', 30, 220);
      c.fillText('Присед доступен со стадии 3.', 30, 245);
    }
    if (t >= 8) {
      c.fillText('Награды: шаг +0.02, проход +1, удар −10.', 30, 285);
      c.fillText('Откаты к лучшей политике: до 5 раз на стадию.', 30, 310);
    }
    c.fillStyle = '#999'; c.font = '11px Inter, sans-serif'; c.textAlign = 'right';
    c.fillText('клик — пропустить', 712, 474);
  }

  /* ============================================================
  СТАРТ
  ============================================================ */
  (function initAll() {
    const fresh = randomNet();
    W1 = fresh.W1; b1 = fresh.b1;
    Wpi = fresh.Wpi; bpi = fresh.bpi;
    Wv = fresh.Wv; bv = fresh.bv;
    adamOpt = makeAdam();
    boostUntilEp = BOOST_EP_BY_STAGE[0];
    resetEpisode();
    drawGame(); drawNet(); drawStats();
  })();
  parker.raf(mainLoop);
  parker.raf(introFrame);

  /* ============================================================
  РЕГИСТРАЦИЯ В КОЛОДЕ
  ============================================================ */
  Engine.register(DEMO_KEY, {
    key: DEMO_KEY,
    title: 'Динозаврик',
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
