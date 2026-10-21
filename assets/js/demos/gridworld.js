/* Автор: А. А. Агафонов (КФУ). Лицензия: MIT. Часть материалов к статье об изобразительных средствах обучения основам RL. */
/* ==== js/demos/gridworld.js ====
   Гридворлд 3×4 — живое Q/TD-обучение + «выученная политика».
   Один экземпляр демо, переезжает между двумя слайдами колоды:
   012 (data-demo-mode="train" — живое обучение с нуля) и
   013 (data-demo-mode="policy" — застывший снимок выученной политики
   + прочерченный жадный маршрут старт→цель). Адаптировано из
   demo-kit/value.html: убрана половина Value Iteration (в лекции её
   не было — только TD/Q-learning) и интро-ролик (в колоду интро не
   идёт вовсе, см. embed.css). Зависит от общего window.Engine. */
(function () {
  'use strict';

  var DEMO_KEY = 'gridworld';
  var ACC = Engine.ACCENT_HEX;
  var MONO = Engine.FONT_MONO;
  var GRID = 420;

  var gameCanvas = document.getElementById('gw-game');
  var g = gameCanvas.getContext('2d');
  var netCanvas = document.getElementById('gw-net');
  var nctx = netCanvas.getContext('2d');
  var statsEl = document.getElementById('gw-stats');
  var modeBtn = document.getElementById('gw-modeBtn');
  var resetBtn = document.getElementById('gw-resetBtn');
  var weightsBtn = document.getElementById('gw-weightsBtn');
  var infoBtn = document.getElementById('gw-infoBtn');
  var modalOverlay = document.getElementById('gw-modalOverlay');

  var MAP = [
    [0, 0, 0, 2],
    [0, 3, 0, 0],
    [-1, 0, 0, 0]
  ];
  var R = 3, C = 4, CELL = 105, OY = 52;
  var START = { y: 2, x: 0 }, GOAL = { y: 0, x: 3 }, TRAP = { y: 1, x: 1 };
  var STEP_R = -0.1, GOAL_R = 1, TRAP_R = -1, GAMMA = 0.9, ALPHA_TD = 0.5, EPS_TD = 0.2;
  var DIRS = [{ dy: -1, dx: 0, n: '↑' }, { dy: 0, dx: 1, n: '→' }, { dy: 1, dx: 0, n: '↓' }, { dy: 0, dx: -1, n: '←' }];
  var ACTION_SHORT = ['↑', '→', '↓', '←'];

  function isTerm(y, x) { return (y === GOAL.y && x === GOAL.x) || (y === TRAP.y && x === TRAP.x); }
  function termValue(y, x) { return (y === GOAL.y && x === GOAL.x) ? GOAL_R : TRAP_R; }
  function idx(y, x) { return y * C + x; }
  function nextOf(y, x, a) {
    var ny = y + DIRS[a].dy, nx = x + DIRS[a].dx;
    if (ny < 0 || ny >= R || nx < 0 || nx >= C) return { y: y, x: x, bump: true };
    return { y: ny, x: nx, bump: false };
  }
  function rewardOf(y, x, a, n) {
    if (n.y === GOAL.y && n.x === GOAL.x) return GOAL_R;
    if (n.y === TRAP.y && n.x === TRAP.x) return TRAP_R;
    return STEP_R;
  }

  /* ---- TD / Q-LEARNING (единственный метод — VI в этом демо убран) ---- */
  var Qq = Array.from({ length: R * C }, function () { return [0, 0, 0, 0]; });
  var tdAgent = { y: START.y, x: START.x };
  var tdBreakdown = null, tdEp = 1, tdEpR = 0, tdLastA = 0;
  function tdStep() {
    var s = idx(tdAgent.y, tdAgent.x);
    var a;
    if (Math.random() < EPS_TD) a = Math.floor(Math.random() * 4);
    else { var q = Qq[s]; a = 0; for (var i = 1; i < 4; i++) if (q[i] > q[a]) a = i; }
    tdLastA = a;
    var n = nextOf(tdAgent.y, tdAgent.x, a);
    var r = rewardOf(tdAgent.y, tdAgent.x, a, n);
    var s2 = idx(n.y, n.x);
    var done = isTerm(n.y, n.x);
    var boot = done ? 0 : Math.max.apply(null, Qq[s2]);
    var old = Qq[s][a];
    var target = r + GAMMA * boot;
    Qq[s][a] = old + ALPHA_TD * (target - old);
    tdBreakdown = { s: s, a: a, r: r, boot: boot, old: old, target: target, neu: Qq[s][a], next: n, done: done };
    tdEpR += r;
    if (done) { tdEp++; tdEpR = 0; tdAgent = { y: START.y, x: START.x }; }
    else tdAgent = { y: n.y, x: n.x };
  }

  /* ---- «выученная политика»: досрочно доучиваем и прокладываем жадный маршрут ---- */
  var POLICY_MODE = false, policyPath = null;
  function fastForward(steps) { for (var i = 0; i < steps; i++) tdStep(); }
  function computeGreedyPath() {
    var path = [{ y: START.y, x: START.x }];
    var cy = START.y, cx = START.x, seen = {};
    for (var i = 0; i < 16; i++) {
      if (isTerm(cy, cx)) break;
      var s = idx(cy, cx);
      if (seen[s]) break; /* защита от цикла у неоптимальной политики */
      seen[s] = true;
      var q = Qq[s], a = 0; for (var k = 1; k < 4; k++) if (q[k] > q[a]) a = k;
      var n = nextOf(cy, cx, a);
      if (n.bump) break;
      cy = n.y; cx = n.x;
      path.push({ y: cy, x: cx });
    }
    return path;
  }
  function showPolicy() {
    fastForward(400); /* на 3×4 c этими α/ε — с большим запасом сходится за десятки эпизодов */
    policyPath = computeGreedyPath();
    POLICY_MODE = true;
    /* застывший режим рисует один кадр и больше не перерисовывается — нельзя
       полагаться на ResizeObserver (асинхронный) после переезда d.root в
       новый .demo-box: форсируем пересчёт размера канвасов синхронно перед
       единственной отрисовкой, иначе первый кадр может уйти в 0×0 */
    gameResize.apply(); netResize.apply();
    drawGame(); drawRight(); drawStats();
  }

  var paused = false, mainStarted = false;
  var TD_DELAY = 900;

  var drawSkull = Engine.drawSkullAndBones;
  var drawTrapCell = Engine.drawTrapCell;
  var drawArrow = Engine.drawArrow;
  var shadeFor = Engine.shadeFor;
  var fmt = Engine.fmt;

  var ROBOT_IMG = new Image();
  ROBOT_IMG.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"><g fill="none" stroke="#1a1c20" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M24 9 V4.5"/><path d="M12 24 L6 29.5 M36 24 L42 29.5"/><path d="M19 37.5 V43 M29 37.5 V43 M16 43 H22 M26 43 H32"/><rect x="12" y="9" width="24" height="29" rx="11" fill="#1a1c20" fill-opacity=".12"/><rect x="16" y="15" width="16" height="10" rx="5"/><circle cx="20.5" cy="20" r="2" fill="' + ACC + '" stroke="none"/><circle cx="27.5" cy="20" r="2" fill="' + ACC + '" stroke="none"/><path d="M20 31 H28"/></g></svg>');
  /* ---------- ЛЕВАЯ ПАНЕЛЬ: поле ---------- */
  function drawGame() {
    g.clearRect(0, 0, GRID, GRID); g.fillStyle = '#fff'; g.fillRect(0, 0, GRID, GRID);
    var blink = Math.floor(performance.now() / 600) % 2 === 0;
    g.strokeStyle = '#e5e5e5'; g.lineWidth = 1;
    for (var i = 1; i < C; i++) { g.beginPath(); g.moveTo(i * CELL + 0.5, OY); g.lineTo(i * CELL + 0.5, OY + R * CELL); g.stroke(); }
    for (var j = 1; j < R; j++) { g.beginPath(); g.moveTo(0, OY + j * CELL + 0.5); g.lineTo(C * CELL, OY + j * CELL + 0.5); g.stroke(); }
    g.strokeStyle = '#000'; g.lineWidth = 1;
    g.strokeRect(0.5, OY + 0.5, C * CELL - 1, R * CELL - 1);
    for (var y = 0; y < R; y++) for (var x = 0; x < C; x++) {
      var px = x * CELL, py = OY + y * CELL, code = MAP[y][x];
      if (code === 2) {
        var sx0 = px + CELL / 2, sy0 = py + CELL / 2 - 4, sp = [];
        for (var si = 0; si < 10; si++) { var sr = si % 2 === 0 ? CELL * 0.36 : CELL * 0.15, sa = -Math.PI / 2 + si * Math.PI / 5; sp.push([sx0 + sr * Math.cos(sa), sy0 + sr * Math.sin(sa)]); }
        g.beginPath(); sp.forEach(function (q, qi) { if (qi) g.lineTo(q[0], q[1]); else g.moveTo(q[0], q[1]); }); g.closePath();
        g.fillStyle = ACC; g.fill(); g.lineJoin = 'round'; g.lineWidth = 2; g.strokeStyle = ACC; g.stroke();
        g.fillStyle = ACC; g.font = '700 13px Inter, sans-serif'; g.textAlign = 'center'; g.fillText('цель', sx0, py + CELL - 8);
      }
      if (code === 3) {
        g.fillStyle = 'rgba(26,28,32,.92)'; g.fillRect(px + 3, py + 3, CELL - 6, CELL - 6);
        g.strokeStyle = '#fff'; g.lineWidth = 6; g.lineCap = 'round';
        g.beginPath(); g.moveTo(px + CELL / 2 - 17, py + CELL / 2 - 24); g.lineTo(px + CELL / 2 + 17, py + CELL / 2 + 10);
        g.moveTo(px + CELL / 2 + 17, py + CELL / 2 - 24); g.lineTo(px + CELL / 2 - 17, py + CELL / 2 + 10); g.stroke();
        g.fillStyle = '#fff'; g.font = '700 13px Inter, sans-serif'; g.textAlign = 'center'; g.fillText('ловушка', px + CELL / 2, py + CELL - 12);
      }
      if (code === -1) {
        g.fillStyle = '#454b55'; g.font = '600 13px Inter, sans-serif'; g.textAlign = 'center'; g.fillText('старт', px + CELL / 2, py + CELL - 8);
      }
    }
    /* застывший жадный маршрут старт → цель (только в режиме «политика») */
    if (POLICY_MODE && policyPath && policyPath.length > 1) {
      g.save();
      g.strokeStyle = ACC; g.lineWidth = 3; g.setLineDash([2, 8]); g.lineCap = 'round';
      g.beginPath();
      policyPath.forEach(function (p, i) {
        var cx2 = p.x * CELL + CELL / 2, cy2 = OY + p.y * CELL + CELL / 2;
        if (i === 0) g.moveTo(cx2, cy2); else g.lineTo(cx2, cy2);
      });
      g.stroke();
      g.restore();
    }
    /* живой агент — только пока идёт обучение, не в застывшей политике */
    var ag = POLICY_MODE ? START : tdAgent;
    if (ROBOT_IMG.complete && ROBOT_IMG.naturalWidth) {
      var rs = CELL * 0.78;
      g.drawImage(ROBOT_IMG, ag.x * CELL + CELL / 2 - rs / 2, OY + ag.y * CELL + CELL / 2 - rs / 2, rs, rs);
    } else {
      g.beginPath(); g.arc(ag.x * CELL + CELL / 2, OY + ag.y * CELL + CELL / 2, CELL / 3.1, 0, Math.PI * 2);
      g.fillStyle = ACC; g.fill(); g.strokeStyle = '#000'; g.lineWidth = 2; g.stroke();
    }
  }

  /* ---------- ПРАВАЯ ПАНЕЛЬ: сетка с числами + бары + разбор ---------- */
  function drawRight() {
    nctx.clearRect(0, 0, GRID, GRID);
    nctx.fillStyle = '#000'; nctx.font = '13px Inter, sans-serif'; nctx.textAlign = 'center';
    nctx.fillText(POLICY_MODE ? 'Выученная политика: Q после ' + Engine.fmtFixed(tdEp, 4, 0).trim() + ' эпизодов'
      : 'TD: Q ← Q + α[r + γmaxQ′ − Q]', 210, 16);

    var cw = 96, chh = 70, x0 = 18, y0 = 28;
    var maxAbs = 0.01;
    for (var s = 0; s < R * C; s++) {
      var yy = Math.floor(s / C), xx = s % C;
      var v = isTerm(yy, xx) ? termValue(yy, xx) : Math.max.apply(null, Qq[s]);
      if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
    }
    for (var y = 0; y < R; y++) for (var x = 0; x < C; x++) {
      var s3 = idx(y, x), px = x0 + x * cw, py = y0 + y * chh, w = cw - 4, h = chh - 4;
      var isCur = !POLICY_MODE && s3 === idx(tdAgent.y, tdAgent.x);
      nctx.fillStyle = '#fff'; nctx.fillRect(px, py, w, h);
      if (isTerm(y, x)) {
        var vt = termValue(y, x);
        nctx.fillStyle = shadeFor(vt, maxAbs); nctx.fillRect(px + 1, py + 1, w - 2, h - 2);
        nctx.fillStyle = Math.abs(vt) / maxAbs > 0.55 ? '#fff' : '#000';
        nctx.font = '14px ' + MONO; nctx.textAlign = 'center';
        nctx.fillText(fmt(vt), px + w / 2, py + h / 2 + 5);
        if (isCur) { nctx.strokeStyle = '#000'; nctx.lineWidth = 2.5; nctx.strokeRect(px + 1.5, py + 1.5, w - 1, h - 1); }
        continue;
      }
      var v2 = Math.max.apply(null, Qq[s3]);
      var ba = Qq[s3].indexOf(v2);
      nctx.fillStyle = shadeFor(v2, maxAbs); nctx.fillRect(px + 1, py + 1, w - 2, h - 2);
      var light = Math.abs(v2) / maxAbs > 0.55;
      nctx.fillStyle = light ? '#fff' : '#000';
      nctx.font = '14px ' + MONO; nctx.textAlign = 'center';
      nctx.fillText(fmt(v2), px + w / 2, py + 30);
      drawArrow(nctx, px + w / 2, py + 50, ba, 11, light ? '#fff' : '#000');
      if (isCur) { nctx.strokeStyle = '#000'; nctx.lineWidth = 2.5; nctx.strokeRect(px + 1.5, py + 1.5, w - 1, h - 1); }
    }
    nctx.strokeStyle = '#ccc'; nctx.lineWidth = 1;
    for (var ci = 0; ci <= C; ci++) { nctx.beginPath(); nctx.moveTo(x0 + ci * cw - 2 + 0.5, y0); nctx.lineTo(x0 + ci * cw - 2 + 0.5, y0 + R * chh - 4); nctx.stroke(); }
    for (var cj = 0; cj <= R; cj++) { nctx.beginPath(); nctx.moveTo(x0, y0 + cj * chh - 2 + 0.5); nctx.lineTo(x0 + C * cw - 4, y0 + cj * chh - 2 + 0.5); nctx.stroke(); }

    if (!POLICY_MODE) {
      /* бары переоценки четырёх действий текущего состояния — только пока идёт обучение */
      var barY0 = y0 + R * chh + 16;
      var s4 = idx(tdAgent.y, tdAgent.x);
      var barVals = Qq[s4].slice(), barHot = tdLastA;
      var barTitle = 'Q(s,a) для s=(' + tdAgent.y + ',' + tdAgent.x + ')';
      var barMax = 0.01;
      for (var bi = 0; bi < barVals.length; bi++) if (Math.abs(barVals[bi]) > barMax) barMax = Math.abs(barVals[bi]);
      nctx.fillStyle = '#000'; nctx.font = '11px Inter, sans-serif'; nctx.textAlign = 'left';
      nctx.fillText(barTitle, 20, barY0 - 6);
      for (var a2 = 0; a2 < 4; a2++) {
        var yb = barY0 + a2 * 18;
        var v3 = barVals[a2];
        nctx.fillStyle = '#000'; nctx.font = '12px ' + MONO; nctx.textAlign = 'left';
        nctx.fillText(ACTION_SHORT[a2] + ' ' + fmt(v3), 20, yb + 12);
        var bx = 92, bw = 200;
        nctx.fillStyle = '#f0f2f5'; nctx.fillRect(bx, yb, bw, 14);
        if (Math.abs(v3) > 1e-6) {
          nctx.fillStyle = v3 >= 0 ? 'rgba(59,111,224,0.85)' : 'rgba(70,70,75,0.85)';
          nctx.fillRect(bx, yb, Math.max(2, Math.abs(v3) / barMax * bw), 14);
        }
        if (a2 === barHot) { nctx.strokeStyle = '#000'; nctx.lineWidth = 2; nctx.strokeRect(bx - 1, yb - 1, bw + 2, 16); }
        else { nctx.strokeStyle = '#ddd'; nctx.lineWidth = 1; nctx.strokeRect(bx + 0.5, yb + 0.5, bw - 1, 13); }
      }
      var ly = barY0 + 4 * 18 + 16;
      nctx.font = '11px ' + MONO; nctx.textAlign = 'left';
      if (tdBreakdown) {
        var t = tdBreakdown;
        nctx.fillStyle = '#000';
        nctx.fillText('s=(' + Math.floor(t.s / C) + ',' + (t.s % C) + ') a=' + ACTION_SHORT[t.a] + '  r=' + fmt(t.r) + '  maxQ(s′)=' + fmt(t.boot), 20, ly); ly += 15;
        nctx.fillText('цель=r+γ·maxQ′=' + fmt(t.target) + '   Q старое=' + fmt(t.old), 20, ly); ly += 15;
        nctx.fillStyle = ACC;
        nctx.fillText('Q новое=' + fmt(t.old) + '+' + ALPHA_TD + '·(' + fmt(t.target) + '−' + fmt(t.old) + ')=' + fmt(t.neu), 20, ly);
      }
    } else {
      nctx.fillStyle = '#454b55'; nctx.font = '12px Inter, sans-serif'; nctx.textAlign = 'center';
      nctx.fillText('пунктир слева — жадный маршрут по выученным Q', 210, y0 + R * chh + 34);
    }
    nctx.fillStyle = '#999'; nctx.font = '10px Inter, sans-serif'; nctx.textAlign = 'center';
    nctx.fillText('синий = выгодно · серый = невыгодно · рамка = текущее состояние', 210, 414);
  }

  function drawStats() {
    statsEl.innerHTML =
      '<span>Эпизод: <span class="v">' + Engine.fmtFixed(tdEp, 4, 0) + '</span></span>' +
      '<span>ε: <span class="v">0.2</span></span>';
  }

  function mainLoop(ts) {
    if (mainStarted && !paused && !POLICY_MODE) {
      if (ts - lastTs >= TD_DELAY) { tdStep(); lastTs = ts; }
      drawGame(); drawRight(); drawStats();
    }
    parker.raf(mainLoop);
  }
  var lastTs = 0;
  var parker = Engine.makeRafParker({ getMain: function () { return mainLoop; } });

  modeBtn.addEventListener('click', function () { paused = !paused; modeBtn.textContent = paused ? 'Продолжить' : 'Пауза'; });
  resetBtn.addEventListener('click', function () {
    Qq = Array.from({ length: R * C }, function () { return [0, 0, 0, 0]; });
    tdAgent = { y: START.y, x: START.x }; tdEp = 1; tdEpR = 0; tdBreakdown = null;
    POLICY_MODE = false; policyPath = null;
    drawGame(); drawRight(); drawStats();
  });
  infoBtn.addEventListener('click', function () { modalOverlay.classList.add('is-open'); });
  document.getElementById('gw-modalClose').addEventListener('click', function () { modalOverlay.classList.remove('is-open'); });
  document.getElementById('gw-modalOk').addEventListener('click', function () { modalOverlay.classList.remove('is-open'); });
  modalOverlay.addEventListener('click', function (e) { if (e.target === modalOverlay) modalOverlay.classList.remove('is-open'); });
  weightsBtn.addEventListener('click', function () {
    Engine.openWeightsModal({
      key: DEMO_KEY,
      exportWeights: exportWeights,
      importWeights: importWeights,
      presetWeights: (window.DemoWeights && window.DemoWeights[DEMO_KEY]) || null
    });
  });

  /* ---------- адаптивные канвасы ---------- */
  var gameScale = 1, netScale = 1, gameOX = 0, gameOY = 0, netOX = 0, netOY = 0;
  var gameResize = Engine.resizeCanvas(gameCanvas, { onResize: function (w, h) { gameScale = Math.min(w, h) / GRID; gameOX = (w - GRID * gameScale) / 2; gameOY = (h - GRID * gameScale) / 2; } });
  var netResize = Engine.resizeCanvas(netCanvas, { onResize: function (w, h) { netScale = Math.min(w, h) / GRID; netOX = (w - GRID * netScale) / 2; netOY = (h - GRID * netScale) / 2; } });
  var rawDrawGame = drawGame, rawDrawRight = drawRight;
  drawGame = function () { g.save(); g.translate(gameOX, gameOY); g.scale(gameScale, gameScale); g.beginPath(); g.rect(0, 0, GRID, GRID); g.clip(); rawDrawGame(); g.lineWidth = 3 / gameScale; g.strokeStyle = '#1a1c20'; g.strokeRect(0, 0, GRID, GRID); g.restore(); };
  drawRight = function () { nctx.save(); nctx.translate(netOX, netOY); nctx.scale(netScale, netScale); rawDrawRight(); nctx.restore(); };

  /* ---------- веса (только Q — VI в этом демо нет) ---------- */
  function exportWeights() { return { Q: Qq }; }
  function importWeights(W) {
    try {
      if (!W || !Array.isArray(W.Q) || W.Q.length !== R * C) return false;
      if (!W.Q.every(function (r) { return Array.isArray(r) && r.length === 4; })) return false;
      Qq = W.Q.map(function (r) { return r.slice(); });
      return true;
    } catch (e) { return false; }
  }

  /* ---------- регистрация в колоде ---------- */
  drawGame(); drawRight(); drawStats();

  Engine.register(DEMO_KEY, {
    key: DEMO_KEY,
    title: 'Гридворлд 3×4: TD/Q-обучение',
    root: document.getElementById('demo-' + DEMO_KEY),
    resume: parker.resume,
    pause: parker.pause,
    kick: function () { parker.kick(true); },
    startMain: function () {
      POLICY_MODE = false;
      parker.startMain(function () { mainStarted = true; });
    },
    showPolicy: showPolicy,
    exportWeights: function () { try { return JSON.parse(JSON.stringify(exportWeights())); } catch (e) { return null; } },
    importWeights: function (W) { return importWeights(W); },
    isFast: function () { return false; },
    setFast: function () { /* у этого демо нет ускоренного режима */ },
    reset: function () { resetBtn.click(); }
  });
})();
