/* ViDecide project page: method animation controls, comparison players (tabs, auto-advance), tile groups that replay together,
   data bars in the tables and the random-sample gallery. Data: assets/data.js (window.VD). */
(function () {
  'use strict';
  const VD = window.VD || {};
  const REDUCE = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const HOLD = 1400;            // ms on the last frame before the next example (or the same one again)
  const EPS = 1e-3;

  function el(tag, attrs, kids) {
    const n = document.createElement(tag);
    for (const k in attrs || {}) {
      const v = attrs[k];
      if (v === null || v === undefined || v === false) continue;
      if (k === 'class') n.className = v;
      else if (k === 'text') n.textContent = v;
      else if (k === 'html') n.innerHTML = v;
      else if (k === 'style') n.style.cssText = v;
      else n.setAttribute(k, v === true ? '' : v);
    }
    for (const c of [].concat(kids || [])) if (c !== null && c !== undefined) n.append(c);
    return n;
  }
  const pct = (a, b) => (100 * a / b).toFixed(3) + '%';
  const sec = t => t.toFixed(1) + ' s';
  const NAME = { 'Snake': 'Snake Routing', 'Reward Collection': 'TSP Reward Collection', 'Tower of Hanoi (robot)': 'Tower of Hanoi' };
  const nameOf = it => NAME[it.title] || it.title;
  const RULES = {
    'VideoRLVR|FlowFree': 'Link the endpoints of each color; paths never cross and every cell is filled.',
    'VideoRLVR|Sokoban': 'Push every box onto a target; boxes cannot be pulled.',
    'VBVR-MultiStep|Tower of Hanoi': 'Move one top block at a time until the left stacks match the target on the right, in the fewest moves.',
    'VBVR-MultiStep|Snake Routing': 'Eat every pellet; each pellet adds exactly one cell.',
    'VBVR-MultiStep|TSP Reward Collection': 'Collect every reward before the end point, one cell at a time.',
    'VBVR-MultiStep|Ordinal Number Sequence': 'Visit the numbered cells in increasing order, then the end point.',
    'RoboTwin-Plan|Blocks World': 'Move one top block at a time until the stacks match the small model at the front.',
    'RoboTwin-Plan|Tower of Hanoi': 'Move all blocks onto the green pad, never a larger block on a smaller one.',
    'RoboTwin-Plan|Color Sorting': 'Move one top block at a time until each block stands on its own color.'
  };
  const ruleOf = it => RULES[it.bench + '|' + nameOf(it).replace(' 8×8', '')] || '';
  const TAB = { 'para_v4_hanoi_triangle_2-5_1000006': 'Hanoi (robot)', rl_ordinal: 'Ordinal Number Sequence',
    rl_robotwin_1000016: 'Hanoi (robot)' };
  const CAPTION = { rl_ordinal: 'post-training from the same SFT model.' };
  const labelOf = p => {
    const k = /K = (\d)/.exec(p.note || '');
    return k && !/\(K/.test(p.label) ? `${p.label} (K = ${k[1]})` : p.label;
  };
  // the longest expert video in training, per task (Fig. supp_longhorizon caption)
  // OOD tiles: title, what makes the puzzle out of distribution, the longest expert video of the task in training (s)
  const LONG = {
    ood_2: ['Sokoban, 5 boxes', 'OOD: 5 boxes, training has at most 3', 15.8],
    ood_1: ['Sokoban, 4 boxes', 'OOD: 4 boxes, training has at most 3', 15.8],
    ood_0: ['FlowFree, 9×9 board', 'OOD: 9×9 board, training has at most 8×8', 10.8],
    ood_4: ['Blocks World, 10 moves', 'OOD: 10-move plan, training has at most 6', 11.6],
    ood_5: ['Blocks World, 2×2 layout', 'OOD: places in a 2×2 grid, unseen in training', null],
    ood_hanoi4: ['Tower of Hanoi, 2×2 layout', 'OOD: pads in a 2×2 grid, unseen in training', null] };
  const ICON_PLAY = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1.2v9.6L10.4 6z"/></svg>';
  const ICON_PAUSE = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1h3v10H2zM7 1h3v10H7z"/></svg>';

  // ------------------------------------------------------------------------------------------------ playback
  const near = new IntersectionObserver(es => {
    for (const e of es) if (e.isIntersecting && e.target._ctl) e.target._ctl.attach();
  }, { rootMargin: '300px 0px' });
  const seen = new IntersectionObserver(es => {
    for (const e of es) {
      const c = e.target._ctl; if (!c) continue;
      c.visible = e.isIntersecting;
      if (c.visible) c.autoplay(); else c.pause();
    }
  }, { threshold: 0.25 });

  function controller(video, box, hold) {
    const c = { visible: false, userPaused: REDUCE, src: null, timer: null, next: null, onstate: null };
    video.muted = true; video.defaultMuted = true; video.playsInline = true;
    video.setAttribute('muted', ''); video.setAttribute('playsinline', '');
    c.attach = () => { if (c.src && video.getAttribute('src') !== c.src) { video.src = c.src; video.load(); } };
    c.setSrc = s => { c.src = s; clearTimeout(c.timer); if (c.visible || video.getAttribute('src')) c.attach(); };
    c.autoplay = () => {
      c.attach();
      if (c.userPaused || !c.visible) return;
      const p = video.play(); if (p && p.catch) p.catch(() => {});
    };
    c.pause = () => { clearTimeout(c.timer); video.pause(); };
    c.toggle = () => {
      if (video.paused) { c.userPaused = false; c.visible = true; c.autoplay(); } else { c.userPaused = true; c.pause(); }
      if (c.onstate) c.onstate();
    };
    video.addEventListener('ended', () => {
      clearTimeout(c.timer);
      c.timer = setTimeout(() => {
        if (!c.visible || c.userPaused) return;
        if (c.next) c.next(); else { video.currentTime = 0; c.autoplay(); }
      }, hold === undefined ? HOLD : hold);
    });
    box._ctl = c; near.observe(box); seen.observe(box);
    box.addEventListener('click', c.toggle);
    return c;
  }

  function onClock(video, cb) {
    const fire = () => cb(video.ended ? video.duration : video.currentTime, video.ended);
    if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
      const step = (now, md) => { cb(md.mediaTime, false); video.requestVideoFrameCallback(step); };
      video.requestVideoFrameCallback(step);
    } else {
      let raf = 0;
      const loop = () => { fire(); raf = video.paused ? 0 : requestAnimationFrame(loop); };
      video.addEventListener('play', () => { if (!raf) raf = requestAnimationFrame(loop); });
    }
    for (const ev of ['seeked', 'loadeddata', 'ended', 'pause']) video.addEventListener(ev, fire);
  }

  // ------------------------------------------------------------------------------------------------ method animation
  function buildMethod() {
    const video = document.getElementById('method-video');
    if (!video) return;
    const c = controller(video, video, 2600);              // the video itself ends on a 5 s hold of the full figure
    c.src = 'videos/method_anim.mp4';
    const play = document.getElementById('m-play'), seek = document.getElementById('m-seek');
    const time = document.getElementById('m-time'), speed = document.getElementById('m-speed');
    const steps = [...document.querySelectorAll('#method-steps .step')];
    const fmt = t => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
    let dragging = false;
    const sync = () => { play.innerHTML = video.paused ? ICON_PLAY : ICON_PAUSE; play.setAttribute('aria-label', video.paused ? 'Play' : 'Pause'); };
    c.onstate = sync;
    video.addEventListener('play', sync); video.addEventListener('pause', sync); sync();
    play.addEventListener('click', () => c.toggle());
    const rates = [1, 0.75, 0.5];
    speed.addEventListener('click', () => {
      const r = rates[(rates.indexOf(video.playbackRate) + 1) % rates.length] || 1;
      video.playbackRate = video.defaultPlaybackRate = r;
      speed.textContent = `${r}×`;
    });
    seek.addEventListener('input', () => {
      dragging = true;
      if (video.duration) video.currentTime = seek.value / 1000 * video.duration;
    });
    seek.addEventListener('change', () => { dragging = false; });
    onClock(video, t => {
      const d = video.duration || 35.5;
      if (!dragging) seek.value = Math.round(1000 * t / d);
      seek.style.setProperty('--p', (100 * t / d).toFixed(2) + '%');
      time.textContent = `${fmt(t)} / ${fmt(d)}`;
      const k = t < 7.6 ? 0 : t < 14.0 ? 1 : 2;
      const all = t >= 31.0;                              // the complete figure: every part is on
      steps.forEach((s, i) => s.classList.toggle('on', all || i === k));
    });
    steps.forEach(s => s.addEventListener('click', () => {
      c.attach(); video.currentTime = +s.dataset.t; c.userPaused = false; c.visible = true; c.autoplay();
    }));
  }

  // ------------------------------------------------------------------------------------------------ players
  function buildPlayer(root) {
    const ids = (root.dataset.items || '').split(',').filter(id => VD[id]);
    if (!ids.length) return;
    const P = +(root.dataset.panel || 250);                  // displayed panel width in px
    const below = root.dataset.labels !== 'over';            // violation and verdict labels under the panels, not on the video
    const auto = root.dataset.auto === '1' && ids.length > 1; // move on to the next example after each video
    let tabs = null, row = null, c = null, cur = null;
    if (ids.length > 1) {
      tabs = el('div', { class: 'tabs', role: 'tablist', 'aria-label': 'Examples' });
      const prev = el('button', { class: 'nav-btn', type: 'button', 'aria-label': 'Previous example', text: '‹' });
      const nextB = el('button', { class: 'nav-btn', type: 'button', 'aria-label': 'Next example', text: '›' });
      prev.addEventListener('click', () => { c.userPaused = false; load(ids[(ids.indexOf(cur) - 1 + ids.length) % ids.length]); });
      nextB.addEventListener('click', () => { c.userPaused = false; load(ids[(ids.indexOf(cur) + 1) % ids.length]); });
      row = el('div', { class: 'tabs-row' }, [prev, tabs, nextB]);
    }
    const heads = el('div', { class: 'heads' });
    const video = el('video', { preload: 'metadata' });
    const stage = el('div', { class: 'stage', title: 'Click to pause or play' }, [video]);
    const cap = el('p', { class: 'p-caption' });
    const status = below ? el('div', { class: 'pstatus' }) : null;
    const body = el('div', { class: 'pbody' }, [heads, stage, status, cap].filter(Boolean));
    root.append(...[row, body].filter(Boolean));
    c = controller(video, stage);
    let item = null, ui = [];
    if (auto) c.next = () => load(ids[(ids.indexOf(cur) + 1) % ids.length]);

    let first = true;
    function load(id) {
      cur = id; item = VD[id];
      if (!first) {
        stage.classList.add('swap');
        video.addEventListener('loadeddata', () => requestAnimationFrame(() => stage.classList.remove('swap')), { once: true });
        setTimeout(() => stage.classList.remove('swap'), 900);
      }
      first = false;
      if (tabs) for (const b of tabs.children) b.setAttribute('aria-selected', b.dataset.id === id ? 'true' : 'false');
      const n = item.n, W = item.w, scale = P / item.size;
      body.style.maxWidth = Math.round(n * P + (n - 1) * item.gap * scale) + 'px';
      heads.innerHTML = '';
      heads.style.gridTemplateColumns = `repeat(${n}, minmax(0, 1fr))`;
      heads.style.columnGap = pct(item.gap, W);
      for (const o of stage.querySelectorAll('.ovl')) o.remove();
      if (status) { status.innerHTML = ''; status.style.gridTemplateColumns = heads.style.gridTemplateColumns; status.style.columnGap = heads.style.columnGap; }
      ui = item.panels.map((p, j) => {
        heads.append(el('span', { class: 'head-label' + (p.ours ? ' ours' : ''), text: labelOf(p), title: labelOf(p) }));
        const viol = el('div', { class: 'ovl-viol' });
        const eos = p.eos_overlay ? el('div', { class: 'ovl-eos' }) : null;
        const tag = below ? null : el('div', { class: 'ovl-tag' });
        const verdict = below ? null : el('div', { class: 'ovl-verdict', html: p.ok ? '<span class="ok">&#10003;</span> solved'
          : '<span class="no">&#10007;</span> ' + (p.verdict || 'fails') });
        const st = below ? status.appendChild(el('span', { class: 'pst' })) : null;
        stage.append(el('div', { class: 'ovl', style: `left:${pct(j * (item.size + item.gap), W)};width:${pct(item.size, W)}` },
          [eos, viol, tag, verdict]));
        return { p, viol, eos, tag, verdict, st, shown: null };
      });
      cap.innerHTML = '';
      cap.append(el('b', { text: nameOf(item) }), ` · ${item.bench} · ` + (CAPTION[id] || ruleOf(item)));
      c.setSrc(item.video);
      update(0, false);
      c.autoplay();
    }

    function update(t, ended) {
      for (const q of ui) {
        const p = q.p;
        const first = p.marks.length ? p.marks[0].t : null;
        q.viol.classList.toggle('on', first !== null && t >= first - EPS);
        let lab = null;
        for (const m of p.marks) if (t >= m.t - EPS) lab = m.label;
        const done = ended || t >= p.end - 0.07;
        if (q.eos) q.eos.classList.toggle('on', p.eos !== null && t >= p.eos - EPS);
        if (q.st) {                                          // the label line under the panel
          const text = p.ok ? (done ? 'solved' : '') : (lab || (done ? p.verdict || 'fails' : ''));
          if (text !== q.shown) {
            q.shown = text; q.st.textContent = '';
            q.st.classList.toggle('good', p.ok);
            if (text) q.st.append(el('span', { class: p.ok ? 'ok' : 'no', html: p.ok ? '&#10003;' : '&#10007;' }), ' ' + text);
          }
          continue;
        }
        if (lab) { q.tag.textContent = lab; q.tag.classList.add('on'); } else q.tag.classList.remove('on');
        q.verdict.classList.toggle('on', done);
      }
    }
    onClock(video, update);

    if (tabs) {
      for (const id of ids) {
        const b = el('button', { class: 'tab', type: 'button', role: 'tab', 'data-id': id, text: TAB[id] || nameOf(VD[id]) });
        b.addEventListener('click', () => { c.userPaused = false; load(id); });
        tabs.append(b);
      }
    }
    load(ids[0]);
  }

  // ------------------------------------------------------------------------------------------------ tiles
  function tile(it, title, meta, extra) {
    const p = it.panels[0];
    const video = el('video', { preload: 'metadata', 'aria-label': `${nameOf(it)}, ${it.bench}` });
    video.muted = true; video.defaultMuted = true; video.playsInline = true;
    video.setAttribute('muted', ''); video.setAttribute('playsinline', '');
    const frame = el('div', { class: 'tile-frame' }, [video]);
    const viol = p.marks.length ? el('div', { class: 'ovl-viol' }) : null;
    if (viol) {
      frame.append(el('div', { class: 'ovl', style: 'left:0;width:100%' }, [viol]));
      onClock(video, t => viol.classList.toggle('on', t >= p.marks[0].t - EPS));
    }
    const fig = el('figure', { class: 'tile' }, [frame, el('figcaption', {}, [
      el('span', { class: 'tile-title', text: title }),
      meta ? el('span', { class: 'tile-meta', text: meta }) : null, extra || null])]);
    return { fig, video, frame, src: it.video };
  }

  // a group of tiles plays together: each video stops on its last frame, and the group restarts when the last one has ended
  function group(box, tiles) {
    let started = false, visible = false, timer = null, paused = REDUCE;
    const done = new Set();
    const attach = () => { for (const t of tiles) if (!t.video.getAttribute('src')) { t.video.src = t.src; t.video.load(); } };
    const playAll = () => {
      if (!visible || paused) return;
      for (const t of tiles) { const p = t.video.play(); if (p && p.catch) p.catch(() => {}); }
    };
    const restart = () => { done.clear(); for (const t of tiles) t.video.currentTime = 0; playAll(); };
    tiles.forEach((t, i) => {
      t.video.addEventListener('ended', () => {
        done.add(i);
        if (done.size === tiles.length) { clearTimeout(timer); timer = setTimeout(restart, HOLD); }
      });
      t.frame.addEventListener('click', () => { paused = !paused; if (paused) tiles.forEach(x => x.video.pause()); else playAll(); });
    });
    new IntersectionObserver(es => {
      for (const e of es) {
        if (e.isIntersecting) attach();
      }
    }, { rootMargin: '300px 0px' }).observe(box);
    new IntersectionObserver(es => {
      for (const e of es) {
        visible = e.isIntersecting;
        if (visible) { attach(); if (!started) { started = true; restart(); } else for (const t of tiles) { if (!t.video.ended) t.video.play().catch(() => {}); } }
        else { tiles.forEach(t => t.video.pause()); }
      }
    }, { threshold: 0.2 }).observe(box);
  }

  function buildBench() {
    const box = document.getElementById('bench-grid');
    if (!box) return;
    const all = [];
    for (const row of box.querySelectorAll('.bench-tiles')) {
      for (const id of row.dataset.items.split(',')) {
        if (!VD[id]) continue;
        const t = tile(VD[id], nameOf(VD[id]), null);
        row.append(t.fig); all.push(t);
      }
    }
    group(box, all);
  }

  function buildLong() {
    const box = document.getElementById('long-grid');
    if (!box) return;
    const all = [];
    for (const id of box.dataset.items.split(',')) {
      const it = VD[id]; if (!it) continue;
      const p = it.panels[0];
      const [title, ood, max] = LONG[id] || [nameOf(it), null, null];
      const meta = `${it.bench} · ${p.note === '+ CFT' ? 'with CFT' : 'chunk-wise model'}`;
      const chips = el('div', { class: 'dur' }, [
        ood ? el('span', { class: 'chip ood', text: ood }) : null,
        el('span', { class: 'chip eos', text: `EOS at ${sec(p.eos)}` }),
        max ? el('span', { class: 'chip max', text: `longest training video ${sec(max)}` }) : null]);
      const t = tile(it, title, meta, chips);
      box.append(t.fig); all.push(t);
    }
    group(box, all);
  }

  function buildGallery() {
    const box = document.getElementById('gallery'), bar = document.getElementById('gal-filters');
    if (!box || !bar) return;
    const items = Object.values(VD).filter(it => it.section === 'gallery');
    const fam = it => it.bench === 'VideoRLVR' ? nameOf(it) : it.bench;
    const tiles = items.map(it => {
      const p = it.panels[0];
      const verdict = el('span', { class: 'tile-verdict', html: p.ok ? '<span class="ok">&#10003;</span> solved'
        : `<span class="no">&#10007;</span> ${p.verdict || 'fails'}` });
      const t = tile(it, nameOf(it), it.detail || '', verdict);
      const c = controller(t.video, t.frame);
      c.src = it.video;
      t.fig.dataset.fam = fam(it);
      box.append(t.fig);
      return t.fig;
    });
    const btns = [];
    const mk = (key, label) => {
      const sub = key === 'all' ? items : items.filter(it => fam(it) === key);
      const ok = sub.filter(it => it.panels[0].ok).length;
      const b = el('button', { class: 'filter', type: 'button', role: 'tab', 'aria-selected': key === 'all' ? 'true' : 'false' },
        [label, el('span', { class: 'cnt', text: `${ok}/${sub.length}` })]);
      b.addEventListener('click', () => {
        for (const x of btns) x.setAttribute('aria-selected', x === b ? 'true' : 'false');
        for (const t of tiles) t.hidden = !(key === 'all' || t.dataset.fam === key);
      });
      btns.push(b); bar.append(b);
    };
    mk('all', 'All');
    for (const f of ['FlowFree', 'Sokoban', 'VBVR-MultiStep', 'RoboTwin-Plan']) mk(f, f);
  }

  // ------------------------------------------------------------------------------------------------ tables and layout
  function alignTeaser() {                                 // the Fig. 1 chart takes the height of the two videos beside it
    const vids = document.querySelector('.teaser-videos'), img = document.querySelector('.teaser-chart img');
    if (!vids || !img) return;
    const fit = () => {
      const side = vids.getBoundingClientRect(), wrap = vids.parentElement.getBoundingClientRect();
      if (side.width < wrap.width * 0.75) { img.style.height = Math.round(side.height * 0.87) + 'px'; img.style.width = 'auto'; img.parentElement.style.width = 'auto'; }
      else { img.style.height = ''; img.style.width = ''; img.parentElement.style.width = ''; }
    };
    new ResizeObserver(fit).observe(vids);
    window.addEventListener('resize', fit);
    fit();
  }


  // ------------------------------------------------------------------------------------------------ charts (More results)
  const NS = 'http://www.w3.org/2000/svg';
  function sv(tag, a, text) { const n = document.createElementNS(NS, tag); for (const k in a) n.setAttribute(k, a[k]); if (text !== undefined) n.textContent = text; return n; }
  const tip = (node, text) => { node.append(sv('title', {}, text)); return node; };
  const NEUTRAL = '#8a919e';

  function legend(box, entries) {                          // [label, colour, style] with style 'line' | 'dash' | 'box' | 'dot'
    const row = el('div', { class: 'legend-row' });
    for (const [lab, col, st] of entries) {
      const i = el('i', { class: st === 'dash' ? 'dash' : st === 'box' ? 'box' : '' });
      if (st === 'box') i.style.background = col; else i.style.borderTopColor = col;
      if (st === 'dot') { i.className = 'box'; i.style.borderRadius = '50%'; i.style.background = col; }
      row.append(el('span', {}, [i, lab]));
    }
    box.append(row);
  }

  function dumbbell(box) {
    const panels = [
      { title: 'Sokoban', a: 'final frame only', b: 'every transition', rows: [['Bidirectional', NEUTRAL, 98.2, 50.8], ['Ours (K = 8)', '#5aa2ea', 84.5, 75.0], ['Ours (K = 1)', '#1f5fbf', 83.5, 83.0]] },
      { title: 'VBVR-MultiStep', a: 'strict', b: 'strict and shortest', rows: [['Bidirectional', NEUTRAL, 24.3, 22.3], ['Ours (K = 8)', '#5aa2ea', 65.2, 57.8], ['Ours (K = 1)', '#1f5fbf', 85.7, 74.8]] }];
    box.innerHTML = '';
    legend(box, [['first check (hollow)', NEUTRAL, 'line'], ['stricter check (filled)', '#1f5fbf', 'dot']]);
    const W = Math.max(320, box.clientWidth), two = W > 640, pw = two ? (W - 30) / 2 : W, ph = 190;
    const g = sv('svg', { viewBox: `0 0 ${W} ${two ? ph : 2 * ph + 20}` });
    panels.forEach((P, pi) => {
      const ox = two ? pi * (pw + 30) : 0, oy = two ? 0 : pi * (ph + 20), L = 104, R = 44, T = 34, rowH = 42;
      const X = v => ox + L + (pw - L - R) * v / 100;
      g.append(sv('text', { x: ox, y: oy + 16, 'font-size': 15, 'font-weight': 600, fill: '#141a29' }, P.title));
      g.append(sv('text', { x: ox + pw, y: oy + 16, 'font-size': 12.5, 'text-anchor': 'end' }, `${P.a} → ${P.b}`));
      for (const v of [0, 25, 50, 75, 100]) {
        g.append(sv('line', { x1: X(v), x2: X(v), y1: oy + T - 6, y2: oy + T + rowH * 3 - 10, class: 'grid' }));
        g.append(sv('text', { x: X(v), y: oy + T + rowH * 3 + 6, 'font-size': 12, 'text-anchor': 'middle' }, v));
      }
      P.rows.forEach(([name, col, a, b], r) => {
        const y = oy + T + 10 + r * rowH;
        g.append(sv('text', { x: ox, y: y + 5, 'font-size': 14 }, name));
        g.append(sv('line', { x1: X(a), x2: X(b), y1: y, y2: y, stroke: col, 'stroke-width': 2.5 }));
        g.append(tip(sv('circle', { cx: X(a), cy: y, r: 6, fill: '#fff', stroke: col, 'stroke-width': 2.2 }), `${name}, ${P.a}: ${a.toFixed(1)}%`));
        g.append(tip(sv('circle', { cx: X(b), cy: y, r: 6.5, fill: col, stroke: '#fff', 'stroke-width': 1.5 }), `${name}, ${P.b}: ${b.toFixed(1)}%`));
        if (Math.abs(a - b) < 9) {
          const right = ox + pw - X(Math.max(a, b)) - 12 >= 80;   // the pair's label goes left of the dots when the panel is narrow
          g.append(sv('text', { x: right ? X(Math.max(a, b)) + 12 : X(Math.min(a, b)) - 12, y: y + 5, 'font-size': 13, class: 'val',
                                'text-anchor': right ? 'start' : 'end' }, `${a.toFixed(1)} → ${b.toFixed(1)}`));
        } else {
          g.append(sv('text', { x: X(a) + (a > b ? 11 : -11), y: y + 5, 'font-size': 13, 'text-anchor': a > b ? 'start' : 'end' }, a.toFixed(1)));
          g.append(sv('text', { x: X(b) + (a > b ? -11 : 11), y: y + 5, 'font-size': 13, class: 'val', 'text-anchor': a > b ? 'end' : 'start' }, b.toFixed(1)));
        }
      });
    });
    box.append(g);
  }

  function passk(box) {
    const methods = [['SFT model', '#a3aab6', 'dash'], ['+ Continued SFT', '#727272', 'dash'], ['+ Sample-level RFT', '#9966dd', 'line'],
      ['+ Decision-level RFT', '#5aa2ea', 'line'], ['+ CFT (ours)', '#1f5fbf', 'line']];
    const sets = [
      ['FlowFree', [[77.9, 90.8, 96.1], [75.5, 89.6, 94.6], [83.2, 92.9, 96.6], [80.1, 91.7, 96.6], [84.4, 93.2, 97.4]]],
      ['FlowFree-Hard', [[44.8, 59.4, 72.2], [48.6, 60.4, 73.4], [50.2, 63.0, 76.6], [52.0, 67.2, 80.0], [52.0, 68.6, 79.8]]],
      ['Sokoban', [[83.0, 88.7, 92.0], [85.5, 90.3, 93.2], [90.1, 93.1, 94.8], [89.8, 94.0, 96.2], [90.8, 94.5, 96.8]]],
      ['Sokoban-Hard', [[50.0, 64.2, 73.8], [59.6, 73.0, 82.2], [61.6, 75.2, 83.4], [73.8, 83.4, 88.0], [78.2, 85.0, 88.6]]],
      ['VBVR-MultiStep', [[74.8, 82.7, 87.8], [74.2, 81.5, 87.2], [79.8, 86.0, 90.5], [78.8, 86.8, 90.7], [82.2, 87.5, 91.3]]],
      ['RoboTwin-Plan', [[73.9, 84.3, 89.7], [74.3, 83.6, 91.1], [77.9, 86.3, 90.9], [70.7, 80.9, 87.6], [78.0, 85.9, 90.7]]]];
    box.innerHTML = '';
    legend(box, methods.map(([n, c, st]) => [n, c, st]));
    const W = Math.max(320, box.clientWidth), cols = W > 760 ? 3 : 2, gap = 26, pw = (W - gap * (cols - 1)) / cols, ph = 170;
    const rows = Math.ceil(sets.length / cols);
    const g = sv('svg', { viewBox: `0 0 ${W} ${rows * (ph + 16)}` });
    sets.forEach(([title, data], si) => {
      const ox = (si % cols) * (pw + gap), oy = Math.floor(si / cols) * (ph + 16), L = 34, R = 34, T = 26, B = 22;
      const all = data.flat(), lo = Math.floor((Math.min(...all) - 2) / 5) * 5, hi = Math.min(100, Math.ceil((Math.max(...all) + 1) / 5) * 5);
      const X = k => ox + L + (pw - L - R) * k / 2, Y = v => oy + T + (ph - T - B) * (1 - (v - lo) / (hi - lo));
      g.append(sv('text', { x: ox + L, y: oy + 15, 'font-size': 14, 'font-weight': 600, fill: '#141a29' }, title));
      const step = (hi - lo) > 30 ? 10 : 5;
      for (let v = lo; v <= hi + 1e-6; v += step) {
        g.append(sv('line', { x1: X(0), x2: X(2), y1: Y(v), y2: Y(v), class: 'grid' }));
        g.append(sv('text', { x: X(0) - 7, y: Y(v) + 4, 'font-size': 11.5, 'text-anchor': 'end' }, v));
      }
      ['k=1', 'k=2', 'k=4'].forEach((lab, i) => g.append(sv('text', { x: X(i), y: oy + ph - 4, 'font-size': 12, 'text-anchor': 'middle' }, lab)));
      data.forEach((ys, mi) => {
        const [name, col, st] = methods[mi], ours = mi === 4;
        g.append(sv('polyline', { points: ys.map((v, i) => `${X(i)},${Y(v)}`).join(' '), fill: 'none', stroke: col,
          'stroke-width': ours ? 2.8 : 1.8, 'stroke-dasharray': st === 'dash' ? '5 4' : 'none', 'stroke-linejoin': 'round' }));
        ys.forEach((v, i) => g.append(tip(sv('circle', { cx: X(i), cy: Y(v), r: ours ? 4.6 : 3.4, fill: col, stroke: '#fff', 'stroke-width': 1.3 }), `${title}, ${name}, k=${[1, 2, 4][i]}: ${v.toFixed(1)}%`)));
        if (ours) g.append(sv('text', { x: X(2) + 8, y: Y(ys[2]) + 4, 'font-size': 12.5, class: 'val', fill: col }, ys[2].toFixed(1)));
      });
    });
    box.append(g);
  }

  function rounds(box) {
    const sets = [['FlowFree Test', 84.4, 84.5, 86.2], ['FlowFree Hard', 52.0, 53.0, 57.6], ['Sokoban Test', 90.8, 90.7, 92.4], ['Sokoban Hard', 78.2, 76.4, 80.2]];
    box.innerHTML = '';
    legend(box, [['same number of extra updates on the round-1 data', '#a3aab6', 'box'], ['round 2 on new corrections', '#1f5fbf', 'box']]);
    const W = Math.max(320, box.clientWidth), H = 250, L = 44, R = 10, T = 18, B = 48;
    const lo = -2, hi = 6, Y = v => T + (H - T - B) * (hi - v) / (hi - lo);
    const g = sv('svg', { viewBox: `0 0 ${W} ${H}` });
    for (let v = lo; v <= hi; v += 2) {
      g.append(sv('line', { x1: L, x2: W - R, y1: Y(v), y2: Y(v), class: v === 0 ? 'axis' : 'grid' }));
      g.append(sv('text', { x: L - 8, y: Y(v) + 4, 'font-size': 12, 'text-anchor': 'end' }, (v > 0 ? '+' : '') + v));
    }
    g.append(sv('text', { x: 0, y: 10, 'font-size': 12.5 }, 'points vs. round 1'));
    const gw = (W - L - R) / sets.length, bw = Math.min(46, gw / 3.2);
    sets.forEach(([name, r1, same, r2], i) => {
      const cx = L + gw * (i + 0.5);
      [[same - r1, '#a3aab6', 'same updates, round-1 data'], [r2 - r1, '#1f5fbf', 'round 2']].forEach(([d, col, lab], j) => {
        const x = cx + (j === 0 ? -bw - 1 : 1), y0 = Y(0), y1 = Y(d);
        g.append(tip(sv('rect', { x, y: Math.min(y0, y1), width: bw, height: Math.max(Math.abs(y1 - y0), 1.5), rx: 3, fill: col }), `${name}, ${lab}: ${(d > 0 ? '+' : '') + d.toFixed(1)} points`));
        g.append(sv('text', { x: x + bw / 2, y: d >= 0 ? y1 - 6 : y1 + 15, 'font-size': 13, 'text-anchor': 'middle', class: j ? 'val' : '' }, (d > 0 ? '+' : '') + d.toFixed(1)));
      });
      g.append(sv('text', { x: cx, y: H - 24, 'font-size': 13.5, 'text-anchor': 'middle', fill: '#141a29' }, name));
      g.append(sv('text', { x: cx, y: H - 8, 'font-size': 12, 'text-anchor': 'middle' }, `round 1: ${r1.toFixed(1)}%`));
    });
    box.append(g);
  }

  function charts() {
    const draw = [['chart-strict', dumbbell], ['chart-passk', passk], ['chart-rounds', rounds]];
    const run = () => { for (const [id, f] of draw) { const b = document.getElementById(id); if (b && b.clientWidth > 0) f(b); } };
    for (const d of document.querySelectorAll('details.more')) d.addEventListener('toggle', run);
    let last = window.innerWidth, timer = 0;
    window.addEventListener('resize', () => { clearTimeout(timer); timer = setTimeout(() => { if (window.innerWidth !== last) { last = window.innerWidth; run(); } }, 150); });
    run();
  }

  // ------------------------------------------------------------------------------------------------ scroll reveal
  function reveal() {
    if (REDUCE || !('IntersectionObserver' in window)) return;
    const sel = '.block > :not(h3):not(.design-tag):not(h4), .ana > .ana-body, .ana > .ana-head, #bench-grid, .ssr, #more > details';
    const items = [...document.querySelectorAll(sel)].filter(n => n.getBoundingClientRect().top > window.innerHeight * 0.92);
    const io = new IntersectionObserver(es => {
      for (const e of es) if (e.isIntersecting) { e.target.classList.remove('pre'); io.unobserve(e.target); }
    }, { threshold: 0.08, rootMargin: '0px 0px -6% 0px' });
    for (const n of items) { n.classList.add('pre', 'rise'); io.observe(n); }
  }

  // click a teaser chart or a paper figure to see it at full size
  function lightbox() {
    const box = el('div', { class: 'lightbox', role: 'dialog', 'aria-label': 'Enlarged figure', hidden: true });
    const big = el('img', { alt: '' });
    box.append(big);
    document.body.append(box);
    const close = () => { box.hidden = true; big.removeAttribute('src'); };
    box.addEventListener('click', close);
    document.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });
    for (const img of document.querySelectorAll('.teaser-chart img, .fig > img, .fig .scroller > img, .ana-fig img')) {
      img.classList.add('zoomable');
      img.addEventListener('click', () => { big.src = img.currentSrc || img.src; big.alt = img.alt; box.hidden = false; });
    }
  }

  // BibTeX copy button (public page only)
  function copyBib() {
    for (const btn of document.querySelectorAll('.copy-btn')) {
      const pre = btn.parentElement.querySelector('pre');
      btn.addEventListener('click', () => {
        const done = () => { btn.textContent = 'Copied'; setTimeout(() => { btn.textContent = 'Copy'; }, 1600); };
        const select = () => { const r = document.createRange(); r.selectNodeContents(pre); const s = getSelection(); s.removeAllRanges(); s.addRange(r); };
        try { navigator.clipboard.writeText(pre.textContent).then(done, select); } catch (e) { select(); }
      });
    }
  }

  function boot() {
    buildMethod();
    for (const p of document.querySelectorAll('.player')) buildPlayer(p);
    buildBench();
    buildLong();
    buildGallery();
    alignTeaser();
    lightbox();
    copyBib();
    charts();
    reveal();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
