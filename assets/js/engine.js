/* Автор: А. А. Агафонов (КФУ). Лицензия: MIT. Часть материалов к статье об изобразительных средствах обучения основам RL. */
/* ============================================================
engine.js — общий движок для всех демо «Физмат прорыва».
Один файл на все 8 демо. HTML/CSS-обёртки разные
(standalone.css — самостоятельный файл, embed.css — слайд),
а логика игр и рендера канвасов — общая.
Подключение: тегом script src="engine.js" ДО скрипта
самого демо. Всё висит на window.Engine.
============================================================ */
(function (global) {
'use strict';
function hexRgb(h){ h = String(h).trim().replace('#',''); if (h.length === 3) h = h[0]+h[0]+h[1]+h[1]+h[2]+h[2];
var n = parseInt(h,16); return ((n>>16)&255)+','+((n>>8)&255)+','+(n&255); }
/* Единственный источник акцента — CSS-токен --accent (ДЕК-КОД v1.1).
Канвасы демо и SVG-сцены перекрашиваются вместе с темой. */
var ACCENT_HEX = (getComputedStyle(document.documentElement).getPropertyValue('--accent') || '#3b6fe0').trim() || '#3b6fe0';
var ACCENT_RGB = hexRgb(ACCENT_HEX);
var DARK_RGB   = '70,70,75';
/* моноширинный стек для чисел и формул на канвасах: они не «гуляют» по ширине
   и отличаются от обычного текста демо */
var FONT_MONO = "ui-monospace,'SF Mono',Menlo,Consolas,monospace";
/* ============================================================
1. ПАРКОВЩИК rAF
Раньше копировался в каждый файл под глобальные имена
mainLoop/introFrame. Теперь — фабрика: на каждый демо своя
независимая копия (важно, когда на одной странице/слайде
живут сразу несколько демо).
hooks:
getMain()  -> ссылка на текущую функцию главного цикла
getIntro() -> ссылка на текущую функцию заставки (может
отсутствовать, если у демо нет интро)
============================================================ */
function makeRafParker(hooks) {
hooks = hooks || {};
var d_native = global.requestAnimationFrame.bind(global);
var d_cancelNative = global.cancelAnimationFrame.bind(global);
var d_on = false, d_wait = [], d_live = 0, d_mainRefs = 0, d_only = null;
var d_open = {};
function isMainFn(cb) { return !!hooks.getMain && cb === hooks.getMain(); }
function d_wrap(cb, box) {
var isMain = isMainFn(cb);
return function (ts) {
delete d_open[box.id];
d_live--;
if (isMain) d_mainRefs--;
cb(ts);
};
}
function d_go(cb) {
var box = {};
d_live++;
box.id = d_native(d_wrap(cb, box));
d_open[box.id] = cb;
return box.id;
}
function d_pass(cb) { return !d_only || cb === d_only; }
function raf(cb) {
if (isMainFn(cb)) d_mainRefs++;
if (d_on && d_pass(cb)) d_go(cb);
else d_wait.push(cb);
return 0;
}
function d_flush() {
var q = d_wait; d_wait = [];
q.forEach(function (cb) {
if (d_pass(cb)) d_go(cb); else d_wait.push(cb);
});
}
function resume() {
if (d_on) return;
d_on = true;
d_flush();
}
function pause() {
d_on = false;
for (var id in d_open) {
d_cancelNative(+id);
var cb = d_open[id];
d_live--;
if (isMainFn(cb)) d_mainRefs--;
d_wait.push(cb);
}
d_open = {};
}
/* завершить интро и запустить главный цикл */
function startMain(onBeforeStart) {
try {
d_only = null;
if (onBeforeStart) onBeforeStart();
if (hooks.getMain && d_mainRefs === 0) raf(hooks.getMain());
} catch (e) {}
try { if (d_on) d_flush(); } catch (e) {}
}
/* «пнуть»: если ничего не запланировано — поставить в очередь
интро (пока не кончилось) либо главный цикл */
function kick(introFinished) {
try {
if (!introFinished && hooks.getIntro) {
if (d_live === 0 && d_wait.length === 0) raf(hooks.getIntro());
} else if (hooks.getMain && d_mainRefs === 0) {
raf(hooks.getMain());
}
} catch (e) {}
}
/* показать заставку заново, временно ограничив планировщик
только её кадрами */
function showOnly(cb) {
d_wait = d_wait.filter(function (c) { return c !== cb; });
d_only = cb;
raf(cb);
}
function isRunning() { return d_on; }
return {
raf: raf, resume: resume, pause: pause,
startMain: startMain, kick: kick, showOnly: showOnly,
isRunning: isRunning
};
}
/* ============================================================
2. ФОРМАТИРОВАНИЕ ЧИСЕЛ
Задача требования №2: строка статуса не должна «прыгать» —
фиксированное число разрядов и знак всегда на своём месте.
============================================================ */
function fmtFixed(v, intDigits, decDigits) {
intDigits = intDigits == null ? 1 : intDigits;
decDigits = decDigits == null ? 2 : decDigits;
var neg = v < 0;
var s = Math.abs(v).toFixed(decDigits);
var dot = s.indexOf('.');
var ip = dot === -1 ? s : s.slice(0, dot);
var fp = dot === -1 ? '' : s.slice(dot);
while (ip.length < intDigits) ip = '0' + ip;
return (neg ? '−' : ' ') + ip + fp; //   — цифровой пробел вместо знака
}
function fmt(v)  { return (v < 0 ? '−' : '') + Math.abs(v).toFixed(2); }
function fmt1(v) { return (v < 0 ? '−' : '') + Math.abs(v).toFixed(1); }
/* ============================================================
3. РИСОВАЛЬНЫЕ ПРИМИТИВЫ («эталон «Сокровищ»»)
Раньше дублировались в value/explore/treasures/snake-q.
Всегда используют акцент презентации, если явно не передан
другой цвет.
============================================================ */
function drawArrow(c, x, y, action, len, color, dirs) {
dirs = dirs || [[0, -1], [1, 0], [0, 1], [-1, 0]];
var d = dirs[action], x2 = x + d[0] * len, y2 = y + d[1] * len;
color = color || ACCENT_HEX;
c.strokeStyle = color; c.fillStyle = color; c.lineWidth = 1.5;
c.beginPath(); c.moveTo(x, y); c.lineTo(x2, y2); c.stroke();
var ang = Math.atan2(y2 - y, x2 - x), s = 4;
c.beginPath();
c.moveTo(x2, y2);
c.lineTo(x2 - s * Math.cos(ang - 0.5), y2 - s * Math.sin(ang - 0.5));
c.lineTo(x2 - s * Math.cos(ang + 0.5), y2 - s * Math.sin(ang + 0.5));
c.closePath(); c.fill();
}
function drawSmallArrow(c, x1, y1, x2, y2, color) {
c.strokeStyle = color; c.fillStyle = color; c.lineWidth = 2;
c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2, y2); c.stroke();
var ang = Math.atan2(y2 - y1, x2 - x1), s = 5;
c.beginPath();
c.moveTo(x2, y2);
c.lineTo(x2 - s * Math.cos(ang - 0.5), y2 - s * Math.sin(ang - 0.5));
c.lineTo(x2 - s * Math.cos(ang + 0.5), y2 - s * Math.sin(ang + 0.5));
c.closePath(); c.fill();
}
function shadeFor(v, maxAbs, positiveRgb) {
positiveRgb = positiveRgb || ACCENT_RGB;
var norm = Math.min(1, Math.abs(v) / (maxAbs || 1e-6));
return v >= 0
? 'rgba(' + positiveRgb + ',' + (0.08 + 0.72 * norm).toFixed(3) + ')'
: 'rgba(' + DARK_RGB + ',' + (0.08 + 0.72 * norm).toFixed(3) + ')';
}
function drawSkullAndBones(c, cx, cy, size) {
var s = size;
c.save();
c.translate(cx, cy);
c.lineCap = 'round'; c.lineJoin = 'round';
var boneLen = s * 0.44, boneY = s * 0.18;
c.strokeStyle = '#000'; c.fillStyle = '#000';
c.lineWidth = Math.max(1.1, s * 0.075);
c.beginPath(); c.moveTo(-boneLen, boneY); c.lineTo(boneLen, -boneY); c.stroke();
c.beginPath(); c.moveTo(-boneLen, -boneY); c.lineTo(boneLen, boneY); c.stroke();
var knob = Math.max(0.7, s * 0.042);
var ends = [[-boneLen, boneY], [boneLen, -boneY], [-boneLen, -boneY], [boneLen, boneY]];
ends.forEach(function (e) {
c.beginPath(); c.arc(e[0], e[1], knob, 0, Math.PI * 2); c.fill();
});
var r = s * 0.22, skullY = -s * 0.10;
c.beginPath(); c.arc(0, skullY, r, 0, Math.PI * 2); c.fill();
c.fillRect(-r * 0.44, skullY + r * 0.72, r * 0.88, r * 0.52);
c.fillStyle = '#fff';
var eyeR = Math.max(0.6, r * 0.18);
c.beginPath(); c.arc(-r * 0.38, skullY - r * 0.08, eyeR, 0, Math.PI * 2); c.fill();
c.beginPath(); c.arc(r * 0.38, skullY - r * 0.08, eyeR, 0, Math.PI * 2); c.fill();
c.restore();
}
function drawTrapCell(c, px, py, cell) {
c.fillStyle = '#fff';
c.fillRect(px + 2, py + 2, cell - 4, cell - 4);
c.strokeStyle = '#000'; c.lineWidth = 1;
c.strokeRect(px + 2.5, py + 2.5, cell - 5, cell - 5);
drawSkullAndBones(c, px + cell / 2, py + cell / 2, cell * 0.72);
}
function drawDynamicWall(c, px, py, size) {
c.fillStyle = '#efefef';
c.fillRect(px + 1, py + 1, size - 2, size - 2);
c.save();
c.beginPath(); c.rect(px + 1, py + 1, size - 2, size - 2); c.clip();
c.strokeStyle = '#bbb'; c.lineWidth = Math.max(1, size * 0.06);
for (var d = -size; d < size; d += Math.max(5, size / 4)) {
c.beginPath(); c.moveTo(px + d, py + size); c.lineTo(px + d + size, py); c.stroke();
}
c.restore();
c.strokeStyle = '#777'; c.lineWidth = 1;
c.strokeRect(px + 1.5, py + 1.5, size - 3, size - 3);
}
/* ============================================================
4. АДАПТИВНЫЙ КАНВАС (требование №3)
По умолчанию размер полностью отдан CSS: обёртка задаёт
ширину/высоту (100% + aspect-ratio, либо фиксированная
высота — см. tokens.css/standalone.css/embed.css), а этот
код только читает уже посчитанный браузером размер и
подгоняет под него backing store канвасов с поправкой на
devicePixelRatio. Так исключён цикл обратной связи «JS меняет
стиль → это меняет размер контейнера → ResizeObserver снова
срабатывает → JS опять меняет стиль» — сюда, где раньше был
бы риск «расползания» канваса на маленьких экранах.
Если у контейнера нет CSS-соотношения сторон (например,
нестандартный канвас без класса .eng-canvas-wrap), можно явно
передать opts.ratio (ширина/высота) или opts.fixedHeight —
тогда JS сам вычислит и выставит инлайн-размер.
============================================================ */
function resizeCanvas(canvas, opts) {
opts = opts || {};
var jsOwnsSize = !!(opts.ratio || opts.fixedHeight);
function apply() {
var cssW, cssH;
if (jsOwnsSize) {
var host = opts.host || canvas.parentElement;
var rect = host.getBoundingClientRect();
cssW = Math.max(1, Math.floor(rect.width));
cssH = opts.fixedHeight
? Math.max(1, Math.floor(opts.fixedHeight))
: Math.max(1, Math.floor(cssW / opts.ratio));
if (opts.maxHeight) cssH = Math.min(cssH, opts.maxHeight);
canvas.style.width = cssW + 'px';
canvas.style.height = cssH + 'px';
} else {
var rectC = canvas.getBoundingClientRect();
cssW = Math.max(1, Math.floor(rectC.width));
cssH = Math.max(1, Math.floor(rectC.height));
}
var dpr = global.devicePixelRatio || 1;
var w = Math.round(cssW * dpr), h = Math.round(cssH * dpr);
if (canvas.width !== w || canvas.height !== h) {
canvas.width = w;
canvas.height = h;
}
var ctx = canvas.getContext('2d');
ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
if (opts.onResize) opts.onResize(cssW, cssH);
}
apply();
var ro = null;
if (global.ResizeObserver) {
ro = new ResizeObserver(function () { apply(); });
ro.observe(opts.host || canvas.parentElement);
} else {
global.addEventListener('resize', apply);
}
return { apply: apply, disconnect: function () { if (ro) ro.disconnect(); } };
}
/* ============================================================
5. КРИВАЯ ОБУЧЕНИЯ
Общий рендер + прореживание точек, как было в каждом файле,
но параметризовано подписями.
============================================================ */
function makeCurve(canvas, opts) {
opts = opts || {};
var primaryLabel = opts.primaryLabel || 'результат';
var secondaryLabel = opts.secondaryLabel || 'награда эп.';
var pts = [], lastEpi = -1;
function sample(episode, primaryVal, secondaryVal) {
if (episode === lastEpi) return;
lastEpi = episode;
pts.push([episode, primaryVal, secondaryVal]);
if (pts.length > 700) {
var win = Math.floor(pts.length / 300) + 1;
var out = [];
for (var i = 0; i < pts.length; i += win) {
var chunk = pts.slice(i, i + win);
var mn = chunk[0], mx = chunk[0];
chunk.forEach(function (p) { if (p[1] < mn[1]) mn = p; if (p[1] > mx[1]) mx = p; });
out.push(mn); if (mx !== mn) out.push(mx);
}
pts = out;
}
draw();
}
function reset() { pts = []; lastEpi = -1; draw(); }
function draw() {
var ctx = canvas.getContext('2d');
var W0 = canvas.width, H0 = canvas.height, P = 22;
var dpr = global.devicePixelRatio || 1;
var cssW = W0 / dpr, cssH = H0 / dpr;
ctx.save();
ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
ctx.clearRect(0, 0, cssW, cssH);
if (pts.length < 2) { ctx.restore(); return; }
var xs = pts.map(function (p) { return p[0]; });
var yMet = pts.map(function (p) { return p[1]; });
var yRew = pts.map(function (p) { return p[2]; });
var x0 = xs[0], x1 = Math.max(xs[xs.length - 1], x0 + 1);
var y1 = Math.max.apply(null, yMet) || 1;
var yLo = Math.min.apply(null, yRew.concat([0]));
var yRange = (y1 - yLo) || 1;
function X(v) { return P + (v - x0) / (x1 - x0) * (cssW - P - 8); }
function Y(v) { return cssH - P - (v - yLo) / yRange * (cssH - P - 10); }
ctx.strokeStyle = '#e2e5ea'; ctx.lineWidth = 1;
ctx.beginPath(); ctx.moveTo(P, cssH - P); ctx.lineTo(cssW - 6, cssH - P); ctx.stroke();
ctx.strokeStyle = '#b9bdc4'; ctx.lineWidth = 1; ctx.beginPath();
pts.forEach(function (p, i) { var XX = X(p[0]), YY = Y(p[2]); i ? ctx.lineTo(XX, YY) : ctx.moveTo(XX, YY); });
ctx.stroke();
ctx.strokeStyle = ACCENT_HEX; ctx.lineWidth = 2.2; ctx.beginPath();
pts.forEach(function (p, i) { var XX = X(p[0]), YY = Y(p[1]); i ? ctx.lineTo(XX, YY) : ctx.moveTo(XX, YY); });
ctx.stroke();
ctx.fillStyle = '#454b55'; ctx.font = '12px Inter, sans-serif'; ctx.textAlign = 'left';
ctx.fillText(primaryLabel + ': ' + yMet[yMet.length - 1], P, 14);
ctx.fillStyle = '#b9bdc4';
ctx.fillText(secondaryLabel + ': ' + (typeof yRew[yRew.length - 1] === 'number' ? yRew[yRew.length - 1].toFixed(1) : yRew[yRew.length - 1]), P, 30);
ctx.fillStyle = '#454b55'; ctx.textAlign = 'right';
ctx.fillText('эпизод ' + xs[xs.length - 1], cssW - 8, cssH - 6);
ctx.restore();
}
return { sample: sample, reset: reset, draw: draw };
}
/* ============================================================
6. МОДАЛКА ВЕСОВ (замена prompt()/alert())
Один <textarea>-диалог на страницу, переиспользуется всеми
демо. Показывает текущий JSON, позволяет вставить свой.
============================================================ */
var weightsModalEl = null, weightsToastEl = null, weightsToastT = 0, lastClickShift = false;
document.addEventListener('click', function (e) { lastClickShift = !!e.shiftKey; }, true);
function metaLines(meta) {
if (!meta) return [];
var L = [];
if (meta.title) L.push(['', meta.title]);
if (meta.trained) L.push(['Обучение', meta.trained]);
if (meta.score) L.push(['Результат', meta.score]);
if (meta.net) L.push(['Модель', meta.net]);
if (meta.date) L.push(['Дата', meta.date]);
if (meta.note) L.push(['', meta.note]);
return L;
}
function metaHtml(meta) {
return metaLines(meta).map(function (p) {
var t = String(p[1]).replace(/&/g, '&amp;').replace(/</g, '&lt;');
return p[0] ? '<div class="eng-wt-row"><span>' + p[0] + '</span><b>' + t + '</b></div>' : '<div class="eng-wt-line">' + t + '</div>';
}).join('');
}
function ensureWeightsModal() {
if (weightsModalEl) return weightsModalEl;
var wrap = document.createElement('div');
wrap.className = 'eng-weights-overlay';
wrap.innerHTML =
'<div class="eng-weights-box">' +
'<button class="eng-weights-close" type="button" aria-label="Закрыть">×</button>' +
'<h2 class="eng-weights-title">Веса модели</h2>' +
'<div class="eng-weights-meta"></div>' +
'<p class="eng-weights-hint">Текущие веса — ниже в формате JSON. Свои можно вставить сюда или загрузить из файла.</p>' +
'<textarea class="eng-weights-textarea" spellcheck="false"></textarea>' +
'<div class="eng-weights-actions">' +
'<button class="eng-weights-preset" type="button">Загрузить обученную модель</button>' +
'<button class="eng-weights-import" type="button">Импортировать</button>' +
'<button class="eng-weights-file" type="button">Из файла…</button>' +
'<button class="eng-weights-save" type="button">Скачать</button>' +
'</div>' +
'<input class="eng-weights-fin" type="file" accept=".json,application/json" hidden>' +
'<div class="eng-weights-status" aria-live="polite"></div>' +
'</div>';
document.body.appendChild(wrap);
weightsModalEl = wrap;
wrap.addEventListener('click', function (e) { if (e.target === wrap) closeWeightsModal(); });
wrap.querySelector('.eng-weights-close').addEventListener('click', closeWeightsModal);
return wrap;
}
function closeWeightsModal() {
if (weightsModalEl) weightsModalEl.classList.remove('is-open');
}
function stripMeta(data) {
if (data && typeof data === 'object' && !Array.isArray(data) && data._meta) { var c = {}; for (var k in data) if (k !== '_meta') c[k] = data[k]; return c; }
return data;
}
function showWeightsToast(cfg, html, title) {
if (!weightsToastEl) {
weightsToastEl = document.createElement('div');
weightsToastEl.className = 'eng-wt-toast';
document.body.appendChild(weightsToastEl);
}
var t = weightsToastEl;
t.innerHTML = '<div class="eng-wt-h">' + title + '</div>' + html +
'<div class="eng-wt-act"><button type="button" data-a="zero">С нуля</button><button type="button" data-a="more">Подробнее…</button><button type="button" data-a="x" aria-label="Закрыть">×</button></div>';
t.classList.add('is-open');
clearTimeout(weightsToastT);
weightsToastT = setTimeout(function () { t.classList.remove('is-open'); }, 9000);
t.onclick = function (e) {
var b = e.target.closest('button'); if (!b) return;
var a = b.getAttribute('data-a');
if (a === 'x') { t.classList.remove('is-open'); }
if (a === 'zero') {
t.classList.remove('is-open');
var d = global.Demos && cfg.key && global.Demos[cfg.key];
if (d && d.reset) d.reset();
}
if (a === 'more') { t.classList.remove('is-open'); openWeightsModal(cfg, true); }
};
}
function openWeightsModal(cfg, forceModal) {
cfg = cfg || {};
var key = cfg.key, preset = cfg.presetWeights, meta = key && global.DemoMeta && global.DemoMeta[key];
/* один клик: вшитая обученная модель загружается сразу */
if (preset && !forceModal && !lastClickShift) {
var ok = false;
try { ok = cfg.importWeights ? cfg.importWeights(stripMeta(JSON.parse(JSON.stringify(preset)))) : false; } catch (e) { ok = false; }
if (ok) { showWeightsToast(cfg, metaHtml(meta) || '', 'Загружена обученная модель'); return; }
}
var el = ensureWeightsModal();
var ta = el.querySelector('.eng-weights-textarea');
var status = el.querySelector('.eng-weights-status');
var presetBtn = el.querySelector('.eng-weights-preset');
var importBtn = el.querySelector('.eng-weights-import');
var fileBtn = el.querySelector('.eng-weights-file');
var saveBtn = el.querySelector('.eng-weights-save');
var fin = el.querySelector('.eng-weights-fin');
el.querySelector('.eng-weights-meta').innerHTML = preset ? metaHtml(meta) : '<div class="eng-wt-line">Вшитой обученной модели для этого демо нет — можно загрузить свою.</div>';
status.textContent = '';
try { ta.value = JSON.stringify(cfg.exportWeights ? cfg.exportWeights() : {}, null, 0); }
catch (e) { ta.value = ''; }
presetBtn.style.display = preset ? '' : 'none';
function doImport(data, what) {
try {
var ok2 = cfg.importWeights ? cfg.importWeights(stripMeta(data)) : false;
status.textContent = ok2 ? (what || 'Импортировано.') : 'Не удалось импортировать — формат не подходит.';
} catch (e) { status.textContent = 'Ошибка: ' + e.message; }
}
presetBtn.onclick = function () { doImport(JSON.parse(JSON.stringify(preset)), 'Обученная модель загружена.'); };
importBtn.onclick = function () {
try { doImport(JSON.parse(ta.value)); } catch (e) { status.textContent = 'Ошибка JSON: ' + e.message; }
};
fileBtn.onclick = function () { fin.value = ''; fin.click(); };
fin.onchange = function () {
var f = fin.files && fin.files[0]; if (!f) return;
var rd = new FileReader();
rd.onload = function () { try { var d = JSON.parse(rd.result); ta.value = JSON.stringify(stripMeta(d)); doImport(d, 'Файл «' + f.name + '» загружен.'); } catch (e) { status.textContent = 'Ошибка JSON: ' + e.message; } };
rd.readAsText(f);
};
saveBtn.onclick = function () {
try {
var w = cfg.exportWeights ? cfg.exportWeights() : {};
var blob = new Blob([JSON.stringify(w)], { type: 'application/json' });
var a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'weights-' + (key || 'demo') + '.json'; a.click();
setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
} catch (e) { status.textContent = 'Не удалось сохранить.'; }
};
el.classList.add('is-open');
}
/* ============================================================
7. РЕГИСТРАЦИЯ ДЕМО
============================================================ */
function register(key, api) {
global.Demos = global.Demos || {};
global.Demos[key] = api;
return api;
}
global.Engine = {
ACCENT_HEX: ACCENT_HEX,
ACCENT_RGB: ACCENT_RGB,
FONT_MONO: FONT_MONO,
makeRafParker: makeRafParker,
fmtFixed: fmtFixed,
fmt: fmt,
fmt1: fmt1,
drawArrow: drawArrow,
drawSmallArrow: drawSmallArrow,
shadeFor: shadeFor,
drawSkullAndBones: drawSkullAndBones,
drawTrapCell: drawTrapCell,
drawDynamicWall: drawDynamicWall,
resizeCanvas: resizeCanvas,
makeCurve: makeCurve,
openWeightsModal: openWeightsModal,
closeWeightsModal: closeWeightsModal,
register: register
};
})(window);
