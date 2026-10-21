/* Хаб демонстраций: показывает по одному демо, остальные стоят на паузе.
   Тот же приём, что в презентации: один экземпляр на игру переезжает в видимый контейнер. */
(function () {
  'use strict';
  var DEMOS = window.HUB_DEMOS || [];
  var hold = document.getElementById('demo-hold');
  var stage = document.getElementById('stage');
  var tabs = document.getElementById('tabs');
  var tTitle = document.getElementById('d-title'), tSub = document.getElementById('d-sub'), tNote = document.getElementById('d-note');
  var current = null;

  function api(key) { return window.Demos && window.Demos[key]; }

  // вкладки по группам
  var groups = [];
  DEMOS.forEach(function (d) { if (groups.indexOf(d.group) < 0) groups.push(d.group); });
  groups.forEach(function (g) {
    var box = document.createElement('div');
    box.className = 'tab-group';
    box.setAttribute('role', 'tablist');
    var gl = document.createElement('span'); gl.className = 'gl'; gl.textContent = g; box.appendChild(gl);
    DEMOS.filter(function (d) { return d.group === g; }).forEach(function (d) {
      var b = document.createElement('button');
      b.type = 'button'; b.className = 'tab'; b.textContent = d.tab;
      b.setAttribute('role', 'tab'); b.dataset.key = d.key; b.setAttribute('aria-selected', 'false');
      b.addEventListener('click', function () { select(d.key, true); });
      box.appendChild(b);
    });
    tabs.appendChild(box);
  });

  function unmount() {
    if (!current) return;
    var d = api(current);
    if (d) { try { d.pause(); } catch (e) {} hold.appendChild(d.root); }
    current = null;
  }

  function select(key, push) {
    var meta = DEMOS.filter(function (d) { return d.key === key; })[0];
    var d = api(key);
    if (!meta || !d) return;
    if (current === key) return;
    unmount();
    stage.appendChild(d.root);
    d.setFast(false);
    d.resume();
    d.startMain();
    d.kick();
    current = key;
    tTitle.textContent = meta.title; tSub.textContent = meta.sub; tNote.textContent = meta.note;
    document.title = meta.title + ' · Обучение с подкреплением: демонстрации';
    Array.prototype.forEach.call(tabs.querySelectorAll('.tab'), function (b) {
      b.setAttribute('aria-selected', b.dataset.key === key ? 'true' : 'false');
    });
    if (push) { try { history.replaceState(null, '', '#' + key); } catch (e) { location.hash = key; } }
  }

  // все демо в начале на паузе (как в презентации)
  Object.keys(window.Demos || {}).forEach(function (k) { try { window.Demos[k].pause(); } catch (e) {} });

  function fromHash() {
    var k = (location.hash || '').replace('#', '');
    return DEMOS.some(function (d) { return d.key === k; }) ? k : DEMOS[0].key;
  }
  window.addEventListener('hashchange', function () { select(fromHash(), false); });
  window.addEventListener('keydown', function (e) {
    if (e.target && /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
    var i = DEMOS.findIndex(function (d) { return d.key === current; });
    if (e.key === 'ArrowRight' && !e.altKey && !e.ctrlKey && !e.metaKey) select(DEMOS[Math.min(DEMOS.length - 1, i + 1)].key, true);
    if (e.key === 'ArrowLeft' && !e.altKey && !e.ctrlKey && !e.metaKey) select(DEMOS[Math.max(0, i - 1)].key, true);
  });
  // в фоновой вкладке браузер сам останавливает анимацию; при возврате оживляем текущее демо
  document.addEventListener('visibilitychange', function () {
    var d = current && api(current);
    if (d && !document.hidden) { try { d.resume(); d.kick(); } catch (e) {} }
  });

  select(fromHash(), false);
})();
