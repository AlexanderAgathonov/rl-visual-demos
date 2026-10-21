/* Автор: А. А. Агафонов (КФУ). Лицензия: MIT. Часть материалов к статье об изобразительных средствах обучения основам RL. */
/* ==== js/demos/sama.js (demo-kit/sama.html) ==== */
(function () {
  'use strict';

  /* ============================================================
  КОНФИГУРАЦИЯ ДЕМО
  ============================================================ */
  const DEMO_KEY = 'sama';
  const ACC = Engine.ACCENT_HEX;
  const MONO = Engine.FONT_MONO;
  const GRID = 420; // внутренняя система координат обоих квадратных канвасов
  const METRIC_LABEL = 'Побед за 100';

  /* ============================================================
  DOM
  ============================================================ */
  const gameCanvas   = document.getElementById('sama-game');
  const g            = gameCanvas.getContext('2d');
  const memoryCanvas = document.getElementById('sama-memory');
  const mctx         = memoryCanvas.getContext('2d');
  const statsEl       = document.getElementById('sama-stats');
  const modeBtn        = document.getElementById('sama-modeBtn');
  const infoBtn         = document.getElementById('sama-infoBtn');
  const resetBtn        = document.getElementById('sama-resetBtn');
  const weightsBtn       = document.getElementById('sama-weightsBtn');
  const modalOverlay = document.getElementById('sama-modalOverlay');
  const modalClose   = document.getElementById('sama-modalClose');
  const modalOk       = document.getElementById('sama-modalOk');

  /* ============================================================
  МОДЕЛЬ ИГРЫ «ШЕСТЬ ПЕШЕК»
  Белые = автоматический соперник, ходят первыми, идут вверх.
  Чёрные = агент САМА, ходит второй, идёт вниз.
  ============================================================ */
  const EMPTY = 0, WHITE = 1, BLACK = 2;
  const ROBOT = BLACK, OPPONENT = WHITE;

  const INITIAL_BOARD = [
    [BLACK, BLACK, BLACK],
    [EMPTY, EMPTY, EMPTY],
    [WHITE, WHITE, WHITE]
  ];

  const MAX_STEPS = 60;
  const INITIAL_BEADS = 3;
  const MAX_BEADS = 5;
  const ADD_ON_WIN = false;
  const CURVE_WINDOW = 100;

  const NORMAL_PHASE_DELAY = {
    episodeEnd: 1100, whiteMove: 750, robotChoose: 850, robotMove: 750
  };
  const FAST_EPISODES_PER_FRAME = 3;

  // у каждого хода в коробке — свой цвет (для наглядности возврата к
  // цветной схеме); сила хода (число бусинок) дополнительно показывается
  // прозрачностью/толщиной, как и вес связи в остальных демо колоды
  const MOVE_COLORS = [
    '#e63946', '#f4a261', '#2a9d8f', '#6a4c93',
    '#118ab2', '#06d6a0', '#ff7096', '#8d99ae'
  ];
  function beadWeight(cnt) { return Math.max(0, Math.min(1, cnt / MAX_BEADS)); }

  /* ============================================================
  БАЗОВЫЕ УТИЛИТЫ
  ============================================================ */
  function cloneBoard(b) { return b.map(row => row.slice()); }
  function mirrorBoard(b) { return b.map(row => [row[2], row[1], row[0]]); }
  function boardKey(b) { return b.map(row => row.join('')).join('|'); }
  function makeMoveId(fr, fc, tr, tc) { return `${fr}${fc}-${tr}${tc}`; }
  function makeMove(fr, fc, tr, tc) {
    return { id: makeMoveId(fr, fc, tr, tc), from: [fr, fc], to: [tr, tc] };
  }
  function canonicalInfo(b) {
    const m = mirrorBoard(b);
    const k1 = boardKey(b), k2 = boardKey(m);
    if (k1 <= k2) return { board: b, key: k1, mirrored: false };
    return { board: m, key: k2, mirrored: true };
  }
  function countPieces(b, player) {
    let n = 0;
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) if (b[r][c] === player) n++;
    return n;
  }
  function hasPieceOnRow(b, player, row) {
    for (let c = 0; c < 3; c++) if (b[row][c] === player) return true;
    return false;
  }
  function legalMoves(b, player) {
    const opp = player === WHITE ? BLACK : WHITE;
    const dir = player === WHITE ? -1 : 1;
    const moves = [];
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        if (b[r][c] !== player) continue;
        const nr = r + dir;
        if (nr < 0 || nr > 2) continue;
        if (b[nr][c] === EMPTY) moves.push(makeMove(r, c, nr, c));
        if (c - 1 >= 0 && b[nr][c - 1] === opp) moves.push(makeMove(r, c, nr, c - 1));
        if (c + 1 < 3 && b[nr][c + 1] === opp) moves.push(makeMove(r, c, nr, c + 1));
      }
    }
    moves.sort((a, b) => (a.id < b.id ? -1 : 1));
    return moves;
  }
  function applyMove(b, mv, player) {
    const nb = cloneBoard(b);
    nb[mv.to[0]][mv.to[1]] = player;
    nb[mv.from[0]][mv.from[1]] = EMPTY;
    return nb;
  }
  function gameStatus(b, nextPlayer) {
    const whiteCount = countPieces(b, WHITE), blackCount = countPieces(b, BLACK);
    if (whiteCount === 0) return 'black-win';
    if (blackCount === 0) return 'white-win';
    if (hasPieceOnRow(b, WHITE, 0)) return 'white-win';
    if (hasPieceOnRow(b, BLACK, 2)) return 'black-win';
    if (nextPlayer) {
      if (legalMoves(b, nextPlayer).length === 0) return nextPlayer === WHITE ? 'black-win' : 'white-win';
    }
    return 'play';
  }
  function randomMove(b, player) {
    const moves = legalMoves(b, player);
    if (!moves.length) return null;
    return moves[(Math.random() * moves.length) | 0];
  }

  /* ============================================================
  ПОСТРОЕНИЕ КОРОБКОВ ДЛЯ МАШИНЫ
  ============================================================ */
  function buildSamaBoxes() {
    const states = [];
    const keyToId = Object.create(null);
    const queue = [];
    function addState(board, key) {
      const id = states.length;
      states.push({ id, board: cloneBoard(board), key, moves: legalMoves(board, ROBOT) });
      keyToId[key] = id;
      return id;
    }
    const firstMoves = legalMoves(INITIAL_BOARD, OPPONENT);
    for (const om of firstMoves) {
      const afterWhite = applyMove(INITIAL_BOARD, om, OPPONENT);
      if (gameStatus(afterWhite, ROBOT) !== 'play') continue;
      const info = canonicalInfo(afterWhite);
      if (keyToId[info.key] === undefined) { addState(info.board, info.key); queue.push(info.board); }
    }
    while (queue.length) {
      const cur = queue.shift();
      const robotMoves = legalMoves(cur, ROBOT);
      for (const rm of robotMoves) {
        const afterBlack = applyMove(cur, rm, ROBOT);
        if (gameStatus(afterBlack, OPPONENT) !== 'play') continue;
        const whiteMoves = legalMoves(afterBlack, OPPONENT);
        for (const wm of whiteMoves) {
          const afterWhite = applyMove(afterBlack, wm, OPPONENT);
          if (gameStatus(afterWhite, ROBOT) !== 'play') continue;
          const info = canonicalInfo(afterWhite);
          if (keyToId[info.key] === undefined) { addState(info.board, info.key); queue.push(info.board); }
        }
      }
    }
    return { states, keyToId };
  }
  const BOX_DATA = buildSamaBoxes();
  const BOX_STATES = BOX_DATA.states;
  const BOX_KEY_TO_ID = BOX_DATA.keyToId;
  const BOX_COUNT = BOX_STATES.length;

  function getBoxMatch(board) {
    const info = canonicalInfo(board);
    const id = BOX_KEY_TO_ID[info.key];
    if (id === undefined) return { id: -1, mirrored: info.mirrored, canonicalBoard: info.board, moves: [] };
    const st = BOX_STATES[id];
    return { id, mirrored: info.mirrored, canonicalBoard: st.board, moves: st.moves };
  }
  function mirrorMove(mv) {
    const fr = mv.from[0], fc = 2 - mv.from[1], tr = mv.to[0], tc = 2 - mv.to[1];
    return { id: makeMoveId(fr, fc, tr, tc), from: [fr, fc], to: [tr, tc] };
  }
  function getActualMove(board, canonicalMove) {
    if (!canonicalMove) return null;
    const match = getBoxMatch(board);
    if (match.id < 0) return null;
    if (!match.mirrored) return canonicalMove;
    return mirrorMove(canonicalMove);
  }

  /* ============================================================
  ПАМЯТЬ АГЕНТА: БУСИНКИ
  ============================================================ */
  const boxMeta = BOX_STATES.map(st => ({
    moves: st.moves.map((mv, j) => ({ id: mv.id, from: mv.from, to: mv.to, color: MOVE_COLORS[j % MOVE_COLORS.length] }))
  }));

  let memory = [];
  let totalBeads = 0;

  function initMemory() {
    memory = BOX_STATES.map(st => {
      const beads = Object.create(null);
      st.moves.forEach(mv => { beads[mv.id] = INITIAL_BEADS; });
      return { beads };
    });
    recalcTotalBeads();
  }
  function recalcTotalBeads() {
    totalBeads = 0;
    for (const box of memory) for (const k in box.beads) totalBeads += box.beads[k];
  }
  function restoreBox(boxId) {
    const meta = boxMeta[boxId];
    if (!meta) return;
    meta.moves.forEach(mv => { memory[boxId].beads[mv.id] = INITIAL_BEADS; });
  }
  function chooseBead(boxId) {
    const meta = boxMeta[boxId];
    if (!meta || !meta.moves.length) return null;
    const pool = [];
    for (const mv of meta.moves) {
      const cnt = memory[boxId].beads[mv.id] || 0;
      for (let i = 0; i < cnt; i++) pool.push(mv.id);
    }
    if (!pool.length) {
      restoreBox(boxId);
      for (const mv of meta.moves) {
        const cnt = memory[boxId].beads[mv.id] || 0;
        for (let i = 0; i < cnt; i++) pool.push(mv.id);
      }
    }
    if (!pool.length) return null;
    return pool[(Math.random() * pool.length) | 0];
  }

  /* ============================================================
  СОСТОЯНИЕ ДЕМО
  ============================================================ */
  let board = cloneBoard(INITIAL_BOARD);
  let episodeHistory = [];
  let episodeCount = 0, robotWins = 0, robotLosses = 0, draws = 0;
  let winHistory = [];
  let lastEvent = '—';       // подпись фазы на канвасе игры (может быть длинной)
  let lastEventShort = '—';  // короткая подпись для строки статуса (фикс. ширина)
  let currentBoxId = -1, currentMirrored = false;
  let selectedCanonical = null, selectedActual = null, lastMove = null;
  let phase = 'episodeEnd', phaseStart = 0, episodeSteps = 0;
  let fastMode = false, mainStarted = false, introFinished = false;
  let flashEvents = [];
  let resultFlash = 0, resultFlashKind = null; // обводка поля по итогу партии

  /* ============================================================
  ВСПЫШКИ ПАМЯТИ
  ============================================================ */
  function pushFlash(boxId, moveId, type) {
    if (fastMode && Math.random() > 0.12) return;
    if (flashEvents.length > 220) flashEvents.shift();
    flashEvents.push({ boxId, moveId, type, t: performance.now() });
  }
  function makeFlashMap(now) {
    flashEvents = flashEvents.filter(f => now - f.t < 700);
    const map = new Map();
    for (const f of flashEvents) map.set(f.boxId + '|' + f.moveId, f.type);
    return map;
  }

  /* ============================================================
  ОБНОВЛЕНИЕ ПАМЯТИ
  ============================================================ */
  function updateMemory(outcome) {
    if (outcome === 'loss') {
      for (const step of episodeHistory) {
        const mem = memory[step.boxId];
        if (!mem) continue;
        if (mem.beads[step.moveId] > 0) { mem.beads[step.moveId]--; pushFlash(step.boxId, step.moveId, 'remove'); }
      }
    } else if (outcome === 'win') {
      if (ADD_ON_WIN) {
        for (const step of episodeHistory) {
          const mem = memory[step.boxId];
          if (!mem) continue;
          if (mem.beads[step.moveId] < MAX_BEADS) { mem.beads[step.moveId]++; pushFlash(step.boxId, step.moveId, 'add'); }
        }
      } else {
        for (const step of episodeHistory) pushFlash(step.boxId, step.moveId, 'good');
      }
    }
    recalcTotalBeads();
  }

  /* ============================================================
  ФАЗЫ ПАРТИИ
  ============================================================ */
  function setPhase(p, ts) { phase = p; phaseStart = ts || performance.now(); }
  function startNewEpisode() {
    board = cloneBoard(INITIAL_BOARD);
    episodeHistory = [];
    currentBoxId = -1; currentMirrored = false;
    selectedCanonical = null; selectedActual = null; lastMove = null;
    episodeSteps = 0;
    setPhase('whiteMove');
  }
  function finishCommon(status) {
    episodeCount++;
    let outcome = 'draw';
    if (status === 'black-win') {
      robotWins++; lastEvent = 'агент победил'; lastEventShort = 'победа';
      winHistory.push(1); outcome = 'win';
    } else if (status === 'white-win') {
      robotLosses++; lastEvent = 'агент проиграл'; lastEventShort = 'поражение';
      winHistory.push(-1); outcome = 'loss';
    } else {
      draws++; lastEvent = 'ничья / лимит'; lastEventShort = 'ничья';
      winHistory.push(0);
    }
    if (winHistory.length > CURVE_WINDOW) winHistory.shift();
    updateMemory(outcome);
    currentBoxId = -1; selectedCanonical = null; selectedActual = null;
    // явная вспышка обводки поля по итогу партии — как жирная обводка при
    // аварии в racing.html, но без ярких цветов: победа — толстая рамка
    // акцентным цветом, поражение — тёмная рамка, ничья — тонкая серая
    resultFlash = 1; resultFlashKind = outcome;
    setPhase('episodeEnd');
  }
  function doWhiteMove() {
    const mv = randomMove(board, WHITE);
    if (!mv) { finishCommon('black-win'); return; }
    board = applyMove(board, mv, WHITE);
    lastMove = { player: WHITE, mv };
    episodeSteps++;
    const status = gameStatus(board, BLACK);
    if (status !== 'play') { finishCommon(status); return; }
    const match = getBoxMatch(board);
    currentBoxId = match.id; currentMirrored = match.mirrored;
    if (currentBoxId < 0) {
      const rm = randomMove(board, BLACK);
      if (!rm) { finishCommon('white-win'); return; }
      selectedCanonical = null; selectedActual = rm;
      setPhase('robotMove');
      return;
    }
    setPhase('robotChoose');
  }
  function chooseRobotBead() {
    if (currentBoxId < 0) { setPhase('whiteMove'); return; }
    const canonicalMoveId = chooseBead(currentBoxId);
    if (!canonicalMoveId) {
      const rm = randomMove(board, BLACK);
      if (!rm) { finishCommon('white-win'); return; }
      selectedCanonical = null; selectedActual = rm;
      setPhase('robotMove');
      return;
    }
    const canonicalMove = BOX_STATES[currentBoxId].moves.find(m => m.id === canonicalMoveId);
    selectedCanonical = canonicalMove;
    let actual = getActualMove(board, canonicalMove);
    const legal = legalMoves(board, BLACK);
    const ok = actual && legal.some(m =>
      m.from[0] === actual.from[0] && m.from[1] === actual.from[1] &&
      m.to[0] === actual.to[0] && m.to[1] === actual.to[1]
    );
    if (!ok) actual = randomMove(board, BLACK);
    selectedActual = actual;
    episodeHistory.push({ boxId: currentBoxId, moveId: canonicalMoveId });
    setPhase('robotMove');
  }
  function doRobotMove() {
    if (!selectedActual) {
      const rm = randomMove(board, BLACK);
      if (!rm) { finishCommon('white-win'); return; }
      selectedActual = rm;
    }
    board = applyMove(board, selectedActual, BLACK);
    lastMove = { player: BLACK, mv: selectedActual };
    episodeSteps++;
    const status = gameStatus(board, WHITE);
    if (status !== 'play') { finishCommon(status); return; }
    if (episodeSteps >= MAX_STEPS) { finishCommon('draw'); return; }
    setPhase('whiteMove');
  }
  function advancePhase(ts) {
    if (phase === 'episodeEnd') startNewEpisode();
    else if (phase === 'whiteMove') doWhiteMove();
    else if (phase === 'robotChoose') chooseRobotBead();
    else if (phase === 'robotMove') doRobotMove();
    else setPhase('episodeEnd', ts);
  }

  /* ============================================================
  УСКОРЕННЫЙ РЕЖИМ
  ============================================================ */
  function stepFastEpisode() {
    startNewEpisode();
    let status = 'play', guard = 0;
    while (status === 'play' && guard < MAX_STEPS) {
      const wm = randomMove(board, WHITE);
      if (!wm) { status = 'black-win'; break; }
      board = applyMove(board, wm, WHITE);
      guard++;
      status = gameStatus(board, BLACK);
      if (status !== 'play') break;
      const match = getBoxMatch(board);
      currentBoxId = match.id;
      let actual = null;
      if (match.id >= 0) {
        const canonicalMoveId = chooseBead(match.id);
        const canonicalMove = BOX_STATES[match.id].moves.find(m => m.id === canonicalMoveId);
        if (canonicalMove) {
          episodeHistory.push({ boxId: match.id, moveId: canonicalMoveId });
          actual = getActualMove(board, canonicalMove);
        }
      }
      if (!actual) actual = randomMove(board, BLACK);
      if (!actual) { status = 'white-win'; break; }
      board = applyMove(board, actual, BLACK);
      lastMove = { player: BLACK, mv: actual };
      guard++;
      status = gameStatus(board, WHITE);
    }
    if (status === 'play') status = 'draw';
    finishCommon(status);
  }

  /* ============================================================
  РЕНДЕР: ОБЩИЕ ФУНКЦИИ
  ============================================================ */
  function cellCenter(x0, y0, cell, r, c) { return [x0 + (c + 0.5) * cell, y0 + (r + 0.5) * cell]; }
  function drawBoardArrow(c, x1, y1, x2, y2, color, width, alpha, dashed) {
    c.save();
    c.globalAlpha = alpha;
    c.strokeStyle = color; c.fillStyle = color; c.lineWidth = width;
    if (dashed) c.setLineDash([4, 3]);
    const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
    const ux = dx / len, uy = dy / len;
    const head = Math.min(10, len * 0.35);
    const ex = x2 - ux * head * 0.6, ey = y2 - uy * head * 0.6;
    c.beginPath(); c.moveTo(x1, y1); c.lineTo(ex, ey); c.stroke();
    const ang = Math.atan2(uy, ux);
    c.beginPath();
    c.moveTo(x2, y2);
    c.lineTo(x2 - head * Math.cos(ang - 0.45), y2 - head * Math.sin(ang - 0.45));
    c.lineTo(x2 - head * Math.cos(ang + 0.45), y2 - head * Math.sin(ang + 0.45));
    c.closePath(); c.fill();
    c.restore();
  }

  /* ============================================================
  РЕНДЕР: ПЕШКА
  ============================================================ */
  function drawPawn(c, cx, cy, r, fill, stroke, lw) {
    // Та же форма, что у иконки i-chess (viewBox 48): основание-плашка, тело с кривыми боками,
    // перекладина-воротник, круглая голова. Высота фигуры = 1.8*r, центр — в (cx, cy).
    const s = (1.8 * r) / 37;               // в иконке пешка занимает по высоте 37 ед. (y 6.5 … 43.5)
    c.save();
    c.translate(cx, cy);
    c.scale(s, s);
    c.translate(-24, -25);                  // центр иконки по высоте (6.5+43.5)/2 = 25
    c.lineJoin = 'round'; c.lineCap = 'round';
    c.fillStyle = fill; c.strokeStyle = stroke; c.lineWidth = lw / s;
    // основание: скруглённая плашка x11 y38.5 w26 h5 rx2.5
    c.beginPath();
    c.moveTo(13.5, 38.5); c.lineTo(34.5, 38.5);
    c.arc(34.5, 41, 2.5, -Math.PI / 2, Math.PI / 2);
    c.lineTo(13.5, 43.5);
    c.arc(13.5, 41, 2.5, Math.PI / 2, Math.PI * 1.5);
    c.closePath(); c.fill(); c.stroke();
    // тело: трапеция с кривыми боками
    c.beginPath();
    c.moveTo(20, 22.5);
    c.bezierCurveTo(20, 29.5, 17, 33, 15.5, 38.5);
    c.lineTo(32.5, 38.5);
    c.bezierCurveTo(31, 33, 28, 29.5, 28, 22.5);
    c.closePath(); c.fill(); c.stroke();
    // воротник-перекладина
    c.beginPath(); c.moveTo(17, 22.5); c.lineTo(31, 22.5); c.stroke();
    // голова
    c.beginPath(); c.arc(24, 12.5, 6, 0, Math.PI * 2); c.fill(); c.stroke();
    c.restore();
  }

  /* ============================================================
  РЕНДЕР: ИГРА
  ============================================================ */
  function drawGame() {
    const W = GRID, H = GRID;
    g.clearRect(0, 0, W, H);
    g.fillStyle = '#fff'; g.fillRect(0, 0, W, H);

    const cell = 112, x0 = (W - cell * 3) / 2, y0 = 40;

    let phaseText = '';
    if (phase === 'whiteMove') phaseText = 'ход белых';
    else if (phase === 'robotChoose') phaseText = 'агент выбирает бусинку';
    else if (phase === 'robotMove') phaseText = 'ход агента';
    else phaseText = lastEvent;

    g.fillStyle = '#000'; g.font = '15px Inter, sans-serif'; g.textAlign = 'center';
    g.fillText(phaseText, W / 2, 22);

    g.strokeStyle = '#ccc'; g.lineWidth = 2;
    for (let i = 0; i <= 3; i++) {
      g.beginPath(); g.moveTo(x0 + i * cell, y0); g.lineTo(x0 + i * cell, y0 + 3 * cell); g.stroke();
      g.beginPath(); g.moveTo(x0, y0 + i * cell); g.lineTo(x0 + 3 * cell, y0 + i * cell); g.stroke();
    }

    if (lastMove) {
      const mv = lastMove.mv;
      const [x1, y1] = cellCenter(x0, y0, cell, mv.from[0], mv.from[1]);
      const [x2, y2] = cellCenter(x0, y0, cell, mv.to[0], mv.to[1]);
      const col = lastMove.player === WHITE ? '#999' : ACC;
      drawBoardArrow(g, x1, y1, x2, y2, col, 4, 0.35);
    }

    if (selectedActual && (phase === 'robotChoose' || phase === 'robotMove')) {
      const [x1, y1] = cellCenter(x0, y0, cell, selectedActual.from[0], selectedActual.from[1]);
      const [x2, y2] = cellCenter(x0, y0, cell, selectedActual.to[0], selectedActual.to[1]);
      drawBoardArrow(g, x1, y1, x2, y2, ACC, 6, 0.9);
    }

    for (let r = 0; r < 3; r++) {
      for (let cc = 0; cc < 3; cc++) {
        if (board[r][cc] === EMPTY) continue;
        const [cx, cy] = cellCenter(x0, y0, cell, r, cc);
        if (board[r][cc] === WHITE) drawPawn(g, cx, cy, cell * 0.36, '#fff', '#000', 2.5);
        else drawPawn(g, cx, cy, cell * 0.36, '#111', ACC, 3);
      }
    }

    g.font = '11px Inter, sans-serif'; g.fillStyle = '#888'; g.textAlign = 'center';
    g.fillText('белые — случайный соперник · чёрные — агент', W / 2, y0 + cell * 3 + 24);

    // вспышка-обводка поля по итогу партии (см. crashFlash в racing.html):
    // победа — толстая акцентная рамка, поражение — тёмная, ничья — тонкая серая
    if (resultFlash > 0) {
      resultFlash = Math.max(0, resultFlash - 0.03);
      let strokeCol = `rgba(120,120,120,${(0.4 * resultFlash).toFixed(3)})`, lw = 4;
      if (resultFlashKind === 'win') { strokeCol = `rgba(${Engine.ACCENT_RGB},${(0.55 * resultFlash).toFixed(3)})`; lw = 7; }
      else if (resultFlashKind === 'loss') { strokeCol = `rgba(17,19,24,${(0.5 * resultFlash).toFixed(3)})`; lw = 7; }
      g.strokeStyle = strokeCol; g.lineWidth = lw;
      g.strokeRect(lw / 2, lw / 2, W - lw, H - lw);
    }
  }

  /* ============================================================
  РЕНДЕР: ПАМЯТЬ МАШИНЫ (19 коробков в сетке 4×5, вписаны в
  квадратный канвас GRID×GRID — общий для всех демо формат
  адаптивных канвасов, требование №3)
  ============================================================ */
  function drawMiniBoard(c, st, boxId, x, y, size, flashMap) {
    const cell = size / 3;
    c.fillStyle = '#fafafa'; c.fillRect(x, y, size, size);
    c.strokeStyle = '#ddd'; c.lineWidth = 1; c.strokeRect(x, y, size, size);
    for (let r = 0; r < 3; r++) {
      for (let cc = 0; cc < 3; cc++) {
        if (st.board[r][cc] === EMPTY) continue;
        const [cx, cy] = cellCenter(x, y, cell, r, cc);
        const pr = Math.max(3, cell * 0.36);
        if (st.board[r][cc] === WHITE) drawPawn(c, cx, cy, pr, '#fff', '#444', 1.1);
        else drawPawn(c, cx, cy, pr, '#111', ACC, 1.1);
      }
    }
    const meta = boxMeta[boxId];
    if (!meta) return;
    for (const mvMeta of meta.moves) {
      const cnt = memory[boxId].beads[mvMeta.id] || 0;
      const w = beadWeight(cnt);
      const [x1, y1] = cellCenter(x, y, cell, mvMeta.from[0], mvMeta.from[1]);
      const [x2, y2] = cellCenter(x, y, cell, mvMeta.to[0], mvMeta.to[1]);
      const flashType = flashMap.get(boxId + '|' + mvMeta.id);
      // цвет хода — свой у каждой бусинки; сила хода (число бусинок)
      // дополнительно — прозрачностью/толщиной, как вес связи в остальных демо
      let color = mvMeta.color, alpha = cnt > 0 ? 0.55 + 0.4 * w : 0.20, width = 1 + 1.3 * w;
      if (flashType === 'remove') { color = '#e63946'; alpha = 0.95; width = 2.5; }
      else if (flashType === 'add' || flashType === 'good') { color = '#2a9d8f'; alpha = 0.95; width = 2.5; }
      if (cnt === 0 && !flashType) color = '#bbb';
      drawBoardArrow(c, x1, y1, x2, y2, color, width, alpha);
    }
  }

  const BOX_COLS = 4, BOX_ROWS = 5, BOX_GAP = 8;
  function boxCellRect(i) {
    const col = i % BOX_COLS, row = Math.floor(i / BOX_COLS);
    const cellW = (GRID - BOX_GAP * (BOX_COLS + 1)) / BOX_COLS;
    const cellH = (GRID - BOX_GAP * (BOX_ROWS + 1)) / BOX_ROWS;
    const x = BOX_GAP + col * (cellW + BOX_GAP);
    const y = BOX_GAP + row * (cellH + BOX_GAP);
    return { x, y, w: cellW, h: cellH };
  }

  function drawMemory() {
    const W = GRID, H = GRID;
    mctx.clearRect(0, 0, W, H);
    mctx.fillStyle = '#fff'; mctx.fillRect(0, 0, W, H);

    const now = performance.now();
    const flashMap = makeFlashMap(now);

    for (let i = 0; i < BOX_COUNT; i++) {
      const { x, y, w: cellW, h: cellH } = boxCellRect(i);
      const active = i === currentBoxId;

      mctx.fillStyle = active ? 'rgba(59,111,224,0.07)' : '#fff';
      mctx.fillRect(x, y, cellW, cellH);
      mctx.strokeStyle = active ? ACC : '#ccc';
      mctx.lineWidth = active ? 2.5 : 1;
      mctx.strokeRect(x, y, cellW, cellH);

      const boardSize = Math.min(cellW - 31, cellH - 10);
      const bx = x + 3, by = y + (cellH - boardSize) / 2;
      drawMiniBoard(mctx, BOX_STATES[i], i, bx, by, boardSize, flashMap);

      const beadX = bx + boardSize + 5;
      mctx.fillStyle = '#888'; mctx.font = 'bold 10px ' + MONO; mctx.textAlign = 'left';
      mctx.fillText(String(i + 1), beadX, y + 11);
      let yy = y + 25;
      const meta = boxMeta[i];
      for (const mvMeta of meta.moves) {
        const cnt = memory[i].beads[mvMeta.id] || 0;
        const flashType = flashMap.get(i + '|' + mvMeta.id);
        let chipColor = mvMeta.color;
        if (cnt === 0) chipColor = '#ddd';
        if (flashType === 'remove') chipColor = '#e63946';
        if (flashType === 'add' || flashType === 'good') chipColor = '#2a9d8f';

        mctx.fillStyle = chipColor;
        mctx.fillRect(beadX, yy - 5, 7, 7);

        mctx.fillStyle = cnt > 0 ? '#000' : '#bbb';
        mctx.font = '9px ' + MONO; mctx.textAlign = 'left';
        mctx.fillText('×' + cnt, beadX + 9, yy + 2);

        if (selectedCanonical && i === currentBoxId && selectedCanonical.id === mvMeta.id) {
          mctx.strokeStyle = ACC; mctx.lineWidth = 1.5;
          mctx.strokeRect(beadX - 2, yy - 7, 26, 11);
        }
        yy += 12;
      }
    }
  }

  /* ============================================================
  СТАТИСТИКА (фиксированная ширина чисел — требование №2)
  ============================================================ */
  function winRate100() {
    if (!winHistory.length) return 0;
    const wins = winHistory.filter(x => x === 1).length;
    return Math.round(wins / winHistory.length * 100);
  }
  function drawStats() {
    statsEl.innerHTML =
      '<span>Партия: <span class="v">' + Engine.fmtFixed(episodeCount, 5, 0) + '</span></span>' +
      '<span>' + METRIC_LABEL + ': <span class="v">' + Engine.fmtFixed(winRate100(), 3, 0) + '%</span></span>' +
      '<span>Бусинок: <span class="v">' + Engine.fmtFixed(totalBeads, 3, 0) + '</span></span>';
  }

  /* ============================================================
  ГЛАВНЫЙ ЦИКЛ + ПАРКОВЩИК rAF
  ============================================================ */
  function mainLoop(ts) {
    if (mainStarted) {
      if (fastMode) {
        for (let i = 0; i < FAST_EPISODES_PER_FRAME; i++) stepFastEpisode();
      } else {
        const delay = NORMAL_PHASE_DELAY[phase] || 700;
        if (ts - phaseStart >= delay) advancePhase(ts);
      }
      drawGame(); drawMemory(); drawStats();
    }
    parker.raf(mainLoop);
  }
  function introFrame(ts) {
    if (introFinished) return;
    if (introStart === null) introStart = ts;
    const t = (ts - introStart) / 1000;
    drawIntro(t);
    if (t >= 24.0) { finishIntro(); return; }
    parker.raf(introFrame);
  }
  const parker = Engine.makeRafParker({
    getMain: function () { return mainLoop; },
    getIntro: function () { return introFrame; }
  });

  /* ============================================================
  КНОПКИ
  ============================================================ */
  modeBtn.addEventListener('click', () => {
    fastMode = !fastMode;
    modeBtn.textContent = fastMode ? 'Обучение: ускоренное' : 'Обучение: обычное'; modeBtn.classList.toggle('eng-btn-accent', fastMode);
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
    initMemory();
    episodeCount = 0; robotWins = 0; robotLosses = 0; draws = 0;
    winHistory.length = 0;
    lastEvent = '—'; lastEventShort = '—';
    flashEvents.length = 0;
    curve.reset();
    startNewEpisode();
    drawGame(); drawMemory(); drawStats();
  }

  /* ============================================================
  ЭКСПОРТ / ИМПОРТ ПАМЯТИ
  ============================================================ */
  function exportWeights() {
    return {
      enc: 'sama-memory-1.0',
      boxes: memory.map(m => { const out = {}; for (const k in m.beads) out[k] = m.beads[k]; return out; }),
      stats: { episodeCount, robotWins, robotLosses, draws, totalBeads }
    };
  }
  function importWeights(obj) {
    try {
      if (!obj || obj.enc !== 'sama-memory-1.0') return false;
      if (!Array.isArray(obj.boxes) || obj.boxes.length !== memory.length) return false;
      obj.boxes.forEach((b, i) => {
        const beads = {};
        BOX_STATES[i].moves.forEach(mv => {
          let val = 0;
          if (b && Number.isFinite(b[mv.id])) val = Math.max(0, Math.floor(b[mv.id]));
          beads[mv.id] = val;
        });
        memory[i].beads = beads;
      });
      if (obj.stats) {
        if (Number.isFinite(obj.stats.episodeCount)) episodeCount = obj.stats.episodeCount;
        if (Number.isFinite(obj.stats.robotWins)) robotWins = obj.stats.robotWins;
        if (Number.isFinite(obj.stats.robotLosses)) robotLosses = obj.stats.robotLosses;
        if (Number.isFinite(obj.stats.draws)) draws = obj.stats.draws;
      }
      recalcTotalBeads();
      return true;
    } catch (e) { return false; }
  }

  /* ============================================================
  ЗАСТАВКА (интро, ~24 c; технический язык уместен — в презентацию
  не идёт, показывается только при автономном открытии страницы)
  ============================================================ */
  const introOverlay = document.getElementById('sama-introOverlay');
  const introCanvas = document.getElementById('sama-introCanvas');
  const ictx = introCanvas.getContext('2d');
  let introStart = null;

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

  const CAPS = [
    [0.0, 2.5, 'Доска 3×3, шесть пешек: белые внизу, чёрные наверху.'],
    [2.5, 5.0, 'Белые — случайный соперник, ходят первыми и идут вверх.'],
    [5.0, 8.0, 'Чёрные — агент САМА, ходит вторым и идёт вниз.'],
    [8.0, 12.5, 'Память агента — 19 коробков. В каждом — цветные бусинки.'],
    [12.5, 17.0, 'Цвет бусинки = ход. Агент тянет бусинку вслепую.'],
    [17.0, 21.0, 'Проиграл партию — бусинки его ходов исчезают из коробка.'],
    [21.0, 24.0, 'Обучение начинается…']
  ];

  const I_CELL = 84, I_X0 = 40, I_Y0 = 74;
  function drawIntroBoard(c, t) {
    const alpha = ease(prog(t, 0.2, 1.2));
    if (alpha <= 0) return;
    c.save(); c.globalAlpha = alpha;
    c.strokeStyle = '#ccc'; c.lineWidth = 2;
    for (let i = 0; i <= 3; i++) {
      c.beginPath(); c.moveTo(I_X0 + i * I_CELL, I_Y0); c.lineTo(I_X0 + i * I_CELL, I_Y0 + 3 * I_CELL); c.stroke();
      c.beginPath(); c.moveTo(I_X0, I_Y0 + i * I_CELL); c.lineTo(I_X0 + 3 * I_CELL, I_Y0 + i * I_CELL); c.stroke();
    }
    for (let r = 0; r < 3; r++) {
      for (let cc = 0; cc < 3; cc++) {
        if (INITIAL_BOARD[r][cc] === EMPTY) continue;
        const [cx, cy] = cellCenter(I_X0, I_Y0, I_CELL, r, cc);
        if (INITIAL_BOARD[r][cc] === WHITE) drawPawn(c, cx, cy, I_CELL * 0.34, '#fff', '#000', 2.5);
        else drawPawn(c, cx, cy, I_CELL * 0.34, '#111', ACC, 3);
      }
    }
    c.restore();
  }
  function drawIntroTurnOrder(c, t) {
    // белые ходят вверх
    const aw = Math.min(prog(t, 2.5, 3.1), 1 - prog(t, 4.6, 5.2));
    if (aw > 0) {
      c.save(); c.globalAlpha = aw;
      const [x1, y1] = cellCenter(I_X0, I_Y0, I_CELL, 2, 1);
      const [x2, y2] = cellCenter(I_X0, I_Y0, I_CELL, 1, 1);
      drawBoardArrow(c, x1, y1, x2, y2, '#666', 3, 1);
      c.fillStyle = '#000'; c.font = '13px Inter, sans-serif'; c.textAlign = 'left';
      c.fillText('ходят первыми, идут вверх', I_X0 + 3 * I_CELL + 20, I_Y0 + 2 * I_CELL);
      c.restore();
    }
    // чёрные ходят вниз
    const ab = Math.min(prog(t, 5.0, 5.6), 1 - prog(t, 7.4, 8.0));
    if (ab > 0) {
      c.save(); c.globalAlpha = ab;
      const [x1, y1] = cellCenter(I_X0, I_Y0, I_CELL, 0, 1);
      const [x2, y2] = cellCenter(I_X0, I_Y0, I_CELL, 1, 1);
      drawBoardArrow(c, x1, y1, x2, y2, ACC, 3, 1);
      c.fillStyle = '#000'; c.font = '13px Inter, sans-serif'; c.textAlign = 'left';
      c.fillText('ходят вторыми, идут вниз — это агент', I_X0 + 3 * I_CELL + 20, I_Y0 + I_CELL);
      c.restore();
    }
  }

  const I_BOX = { x: 440, y: 90, w: 160, h: 140 };
  const I_SAMPLE = BOX_STATES[0];
  const I_SAMPLE_META = boxMeta[0];
  function drawIntroBoxPanel(t) {
    const c = ictx;
    const alpha = ease(prog(t, 8.0, 9.0));
    if (alpha <= 0) return;
    c.save(); c.globalAlpha = alpha;

    c.fillStyle = '#fafafa'; c.fillRect(I_BOX.x, I_BOX.y, I_BOX.w, I_BOX.h);
    c.strokeStyle = ACC; c.lineWidth = 2; c.strokeRect(I_BOX.x, I_BOX.y, I_BOX.w, I_BOX.h);
    c.fillStyle = '#000'; c.font = '12px Inter, sans-serif'; c.textAlign = 'left';
    c.fillText('коробок №1', I_BOX.x, I_BOX.y - 8);

    const bs = 64, bx = I_BOX.x + 14, by = I_BOX.y + 16;
    drawMiniBoard(c, I_SAMPLE, 0, bx, by, bs, new Map());

    let yy = by + 6;
    const beadX = bx + bs + 14;
    I_SAMPLE_META.moves.forEach((mv, idx) => {
      const rowAlpha = prog(t, 9.0 + idx * 0.3, 9.5 + idx * 0.3);
      c.save(); c.globalAlpha = alpha * rowAlpha;
      c.fillStyle = mv.color; c.fillRect(beadX, yy - 6, 10, 10);
      c.fillStyle = '#000'; c.font = '11px ' + MONO; c.textAlign = 'left';
      c.fillText('×' + INITIAL_BEADS, beadX + 15, yy + 3);
      c.restore();
      yy += 18;
    });

    // цвет = ход: стрелка от первой бусинки к соответствующему ходу на доске
    const linkA = Math.min(prog(t, 12.5, 13.3), 1 - prog(t, 16.4, 17.0));
    if (linkA > 0 && I_SAMPLE_META.moves.length) {
      c.save(); c.globalAlpha = alpha * linkA;
      const mv0 = I_SAMPLE_META.moves[0];
      const [mx1, my1] = cellCenter(bx, by, bs / 3, mv0.from[0], mv0.from[1]);
      const [mx2, my2] = cellCenter(bx, by, bs / 3, mv0.to[0], mv0.to[1]);
      c.strokeStyle = mv0.color; c.lineWidth = 3;
      c.beginPath(); c.moveTo(mx1, my1); c.lineTo(mx2, my2); c.stroke();
      c.fillStyle = '#000'; c.font = '13px Inter, sans-serif'; c.textAlign = 'left';
      c.fillText('цвет бусинки = ход', I_BOX.x, I_BOX.y + I_BOX.h + 26);
      c.fillStyle = '#555'; c.font = '12px Inter, sans-serif';
      c.fillText('вытянута вслепую — какой цвет, такой ход', I_BOX.x, I_BOX.y + I_BOX.h + 44);
      c.restore();
    }

    // проигрыш: бусинка исчезает
    const dieA = Math.min(prog(t, 17.0, 17.8), 1 - prog(t, 20.4, 21.0));
    if (dieA > 0 && I_SAMPLE_META.moves.length) {
      c.save(); c.globalAlpha = alpha;
      const mv0 = I_SAMPLE_META.moves[0];
      const chipY = by + 6;
      // затираем старую бусинку белым и рисуем угасающую версию поверх, затем крестик
      c.globalAlpha = alpha * (1 - dieA);
      c.fillStyle = mv0.color; c.fillRect(beadX, chipY - 6, 10, 10);
      c.globalAlpha = alpha * dieA;
      c.strokeStyle = '#e63946'; c.lineWidth = 2;
      c.beginPath(); c.moveTo(beadX - 2, chipY - 8); c.lineTo(beadX + 12, chipY + 6); c.stroke();
      c.beginPath(); c.moveTo(beadX + 12, chipY - 8); c.lineTo(beadX - 2, chipY + 6); c.stroke();
      c.globalAlpha = alpha * dieA;
      c.fillStyle = '#e63946'; c.font = '13px Inter, sans-serif'; c.textAlign = 'left';
      c.fillText('проигрыш → бусинка хода исчезает', I_BOX.x, I_BOX.y + I_BOX.h + 26);
      c.fillStyle = '#555'; c.font = '12px Inter, sans-serif';
      c.fillText('победа — бусинки остаются как есть', I_BOX.x, I_BOX.y + I_BOX.h + 44);
      c.restore();
    }

    c.restore();
  }

  function drawIntro(t) {
    const c = ictx;
    c.clearRect(0, 0, 720, 480);
    c.fillStyle = '#fff'; c.fillRect(0, 0, 720, 480);
    const cap = CAPS.find(x => t >= x[0] && t < x[1]);
    if (cap) { c.fillStyle = '#000'; c.font = '14px Inter, sans-serif'; c.textAlign = 'left'; c.fillText(cap[2], 14, 24); }
    drawIntroBoard(c, t);
    drawIntroTurnOrder(c, t);
    drawIntroBoxPanel(t);
    c.fillStyle = '#999'; c.font = '11px Inter, sans-serif'; c.textAlign = 'right';
    c.fillText('клик — пропустить', 712, 474);
    const fo = prog(t, 22.2, 23.5);
    if (fo > 0) { c.fillStyle = 'rgba(255,255,255,' + fo + ')'; c.fillRect(0, 0, 720, 480); }
    const ta = prog(t, 22.4, 23.0);
    if (ta > 0 && t < 23.8) {
      c.fillStyle = 'rgba(0,0,0,' + ta + ')'; c.font = '20px Inter, sans-serif'; c.textAlign = 'center';
      c.fillText('Обучение начинается…', 360, 240);
    }
  }

  /* ============================================================
  АДАПТИВНЫЕ КАНВАСЫ (требование №3)
  ============================================================ */
  let gameScale = 1, memScale = 1, gameOX = 0, gameOY = 0, memOX = 0, memOY = 0;
  Engine.resizeCanvas(gameCanvas, { onResize: (w, h) => { gameScale = Math.min(w, h) / GRID; gameOX = (w - GRID * gameScale) / 2; gameOY = (h - GRID * gameScale) / 2; } });
  Engine.resizeCanvas(memoryCanvas, { onResize: (w, h) => { memScale = Math.min(w, h) / GRID; memOX = (w - GRID * memScale) / 2; memOY = (h - GRID * memScale) / 2; } });
  const rawDrawGame = drawGame, rawDrawMemory = drawMemory;
  drawGame = function () { g.save(); g.translate(gameOX, gameOY); g.scale(gameScale, gameScale); g.beginPath(); g.rect(0, 0, GRID, GRID); g.clip(); rawDrawGame(); g.lineWidth = 3 / gameScale; g.strokeStyle = '#1a1c20'; g.strokeRect(0, 0, GRID, GRID); g.restore(); };
  drawMemory = function () { mctx.save(); mctx.translate(memOX, memOY); mctx.scale(memScale, memScale); rawDrawMemory(); mctx.restore(); };

  const curve = Engine.makeCurve(document.getElementById('sama-curve'), {
    primaryLabel: 'побед за 100', secondaryLabel: 'побед всего'
  });
  Engine.resizeCanvas(document.getElementById('sama-curve'), {});

  /* ============================================================
  СТАРТ
  ============================================================ */
  initMemory();
  startNewEpisode();
  drawGame(); drawMemory(); drawStats();
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
    curve.sample(episodeCount, winRate100(), robotWins);
  }, 250);

  /* ============================================================
  РЕГИСТРАЦИЯ В КОЛОДЕ
  ============================================================ */
  Engine.register(DEMO_KEY, {
    key: DEMO_KEY,
    title: 'САМА',
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
    setFast: function (on) { if (fastMode !== !!on) modeBtn.click(); },
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
