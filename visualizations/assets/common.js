// ?embed=... : hide chrome and report the page height to the parent (used by blog.html)
const EMBED = new URLSearchParams(location.search).get("embed");
if (EMBED !== null) {
  document.documentElement.classList.add("embed");
  const post = () => parent.postMessage({ embedHeight: document.body.offsetHeight, src: location.pathname.split("/").pop() }, "*");
  addEventListener("load", () => { post(); if (window.ResizeObserver) new ResizeObserver(post).observe(document.body); });
}

// Shared helpers: JS ports of the Python schedulers + tiny chart / tooltip toolkit.
// Everything is float64 here; the Python code uses float32 inside the scheduler.

// ---------- RNG ----------
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------- alpha schedulers (dllm/core/schedulers/alpha.py) ----------
const ALPHA = {
  linear: { name: "LinearAlphaScheduler", a: (t) => 1 - t, d: () => -1 },
  cosine: {
    name: "CosineAlphaScheduler",
    a: (t) => 1 - Math.cos((Math.PI / 2) * (1 - t)),
    d: (t) => -(Math.PI / 2) * Math.sin((Math.PI / 2) * (1 - t)),
  },
};
const alphaWeight = (S, t) => -S.d(t) / (1 - S.a(t) + 1e-6);
// P(token is still masked at s | it is masked at t), s < t
const reverseMaskProb = (S, s, t) => (1 - S.a(s)) / (1 - S.a(t));
// Python computes this in float32 (alpha.py casts to torch.float32); emulate it so rounding ties match.
const f32 = Math.fround;
const reverseMaskProb32 = (S, s, t) => {
  const s32 = f32(s), t32 = f32(t);
  const as = f32(S.a(s32)), at = f32(S.a(t32));
  return f32(f32(1 - as) / f32(1 - at));
};

// ---------- kappa schedulers (dllm/core/schedulers/kappa.py) ----------
function kappaCubic(a, b) {
  return {
    k: (t) => (a + 1) * t ** 3 - (a + b + 1) * t ** 2 + (b + 1) * t,
    d: (t) => 3 * (a + 1) * t ** 2 - 2 * (a + b + 1) * t + (b + 1),
  };
}
const KAPPA_COS = {
  k: (t) => 1 - Math.cos(0.5 * Math.PI * t),
  d: (t) => 0.5 * Math.PI * Math.sin(0.5 * Math.PI * t),
};
const kappaWeight = (K, t) => K.d(t) / (1 - K.k(t) + 1e-6);

// torch.round is round-half-to-even
function roundHE(x) {
  const f = Math.floor(x), d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}
function binomial(n, p, rnd) {
  let k = 0;
  for (let i = 0; i < n; i++) if (rnd() < p) k++;
  return k;
}

// Port of dllm/core/samplers/utils.py::get_num_transfer_tokens
// counts: masked-token count per batch row. Returns raw [B][steps], packed [B][maxLen], probs[steps]
function numTransfer(counts, steps, S, stochastic, rnd, exact64) {
  const mask = counts.slice();
  const raw = counts.map(() => new Array(steps).fill(0));
  const probs = [], remainingBefore = [], expected = counts.map(() => []);
  for (let j = 0; j < steps; j++) {
    const s = (steps - 1 - j) / steps, t = (steps - j) / steps;
    const p = 1 - (exact64 ? reverseMaskProb(S, s, t) : reverseMaskProb32(S, s, t));
    probs.push(p);
    remainingBefore.push(mask.slice());
    for (let b = 0; b < mask.length; b++) {
      expected[b].push(mask[b] * p);
      let n = stochastic ? binomial(mask[b], p, rnd) : roundHE(mask[b] * p);
      n = Math.min(n, mask[b]);
      raw[b][j] = n;
      mask[b] -= n;
    }
  }
  let maxLen = 0;
  for (const r of raw) maxLen = Math.max(maxLen, r.filter((v) => v > 0).length);
  const packed = raw.map((r) => {
    const nz = r.filter((v) => v > 0);
    while (nz.length < maxLen) nz.push(0);
    return nz;
  });
  return { raw, packed, probs, maxLen, remainingBefore, expected };
}

// ---------- tooltip (event delegation on [data-tip]) ----------
(function () {
  const tip = document.createElement("div");
  tip.id = "tip";
  document.addEventListener("DOMContentLoaded", () => document.body.appendChild(tip));
  document.addEventListener("mousemove", (e) => {
    const el = e.target.closest && e.target.closest("[data-tip]");
    if (!el) { tip.style.display = "none"; return; }
    tip.textContent = el.getAttribute("data-tip");
    tip.style.display = "block";
    const w = tip.offsetWidth;
    tip.style.left = Math.min(e.clientX + 14, window.innerWidth - w - 8) + "px";
    tip.style.top = e.clientY + 16 + "px";
  });
})();

// ---------- line chart with crosshair tooltip ----------
// o: {xs? | x:[a,b], n, series:[{name,color,f? | ys?, dash?}], y:[a,b]?, xlabel, ylabel, fmtX, fmtY, markers?:[{x,y,color}]}
function lineChart(host, o) {
  const W = 640, H = 280, m = { l: 48, r: 16, t: 10, b: 36 };
  const xs = o.xs || Array.from({ length: o.n || 200 }, (_, i) => o.x[0] + ((o.x[1] - o.x[0]) * i) / ((o.n || 200) - 1));
  const data = o.series.map((s) => (s.ys ? s.ys : xs.map(s.f)));
  let ymin = Infinity, ymax = -Infinity;
  if (o.y) [ymin, ymax] = o.y;
  else {
    for (const ys of data) for (const v of ys) if (isFinite(v)) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); }
    const pad = (ymax - ymin) * 0.06 || 1; ymin -= pad; ymax += pad;
  }
  const x0 = xs[0], x1 = xs[xs.length - 1];
  const sx = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const sy = (v) => H - m.b - ((v - ymin) / (ymax - ymin || 1)) * (H - m.t - m.b);
  const fx = o.fmtX || ((v) => +v.toFixed(2)), fy = o.fmtY || ((v) => +v.toFixed(3));
  let g = "";
  for (let i = 0; i <= 4; i++) {
    const v = ymin + ((ymax - ymin) * i) / 4;
    g += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${sy(v)}" y2="${sy(v)}"/><text x="${m.l - 6}" y="${sy(v) + 4}" text-anchor="end">${fy(v)}</text>`;
  }
  const xt = o.xticks || 5;
  for (let i = 0; i < xt; i++) {
    const v = x0 + ((x1 - x0) * i) / (xt - 1);
    g += `<text x="${sx(v)}" y="${H - m.b + 16}" text-anchor="middle">${fx(v)}</text>`;
  }
  g += `<line class="axis" x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}"/>`;
  if (o.xlabel) g += `<text x="${(W + m.l) / 2}" y="${H - 4}" text-anchor="middle">${o.xlabel}</text>`;
  if (o.ylabel) g += `<text x="12" y="${(H - m.b) / 2}" transform="rotate(-90 12 ${(H - m.b) / 2})" text-anchor="middle">${o.ylabel}</text>`;
  let paths = "";
  data.forEach((ys, si) => {
    const s = o.series[si];
    const d = ys.map((v, i) => (isFinite(v) ? `${i ? "L" : "M"}${sx(xs[i]).toFixed(1)},${sy(v).toFixed(1)}` : "")).join("");
    paths += `<path d="${d}" fill="none" stroke="var(${s.color})" stroke-width="2" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""} stroke-linejoin="round" stroke-linecap="round"/>`;
    if (o.dots) ys.forEach((v, i) => { paths += `<circle cx="${sx(xs[i])}" cy="${sy(v)}" r="3" fill="var(${s.color})" stroke="var(--surface)" stroke-width="1.5"/>`; });
  });
  (o.markers || []).forEach((mk) => { paths += `<circle cx="${sx(mk.x)}" cy="${sy(mk.y)}" r="5" fill="var(${mk.color})" stroke="var(--surface)" stroke-width="2"/>`; });
  host.classList.add("chart");
  host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img">${g}${paths}<line id="cx" class="axis" y1="${m.t}" y2="${H - m.b}" style="display:none"/><g id="dots"></g><rect id="hit" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/></svg><div class="tt"></div>`;
  const svg = host.querySelector("svg"), tt = host.querySelector(".tt"), cx = host.querySelector("#cx"), dots = host.querySelector("#dots");
  const hit = host.querySelector("#hit");
  hit.addEventListener("mousemove", (e) => {
    const r = svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    let bi = 0, bd = Infinity;
    xs.forEach((v, i) => { const d = Math.abs(sx(v) - px); if (d < bd) { bd = d; bi = i; } });
    cx.setAttribute("x1", sx(xs[bi])); cx.setAttribute("x2", sx(xs[bi])); cx.style.display = "";
    dots.innerHTML = data.map((ys, si) => `<circle cx="${sx(xs[bi])}" cy="${sy(ys[bi])}" r="4.5" fill="var(${o.series[si].color})" stroke="var(--surface)" stroke-width="2"/>`).join("");
    tt.innerHTML = `<b>${o.xname || "x"} = ${fx(xs[bi])}</b><br>` + o.series.map((s, si) => `${s.name}: ${fy(data[si][bi])}`).join("<br>");
    tt.style.display = "block";
    const hr = host.getBoundingClientRect();
    tt.style.left = Math.min(e.clientX - hr.left + 14, hr.width - tt.offsetWidth - 4) + "px";
    tt.style.top = "10px";
  });
  hit.addEventListener("mouseleave", () => { tt.style.display = "none"; cx.style.display = "none"; dots.innerHTML = ""; });
}

// ---------- bar chart (+ optional line on same axis) ----------
// o: {vals, line?, labelX, tipFor(i)->string, highlight?: idx}
function barChart(host, o) {
  const W = 640, H = 230, m = { l: 44, r: 12, t: 10, b: 30 };
  const n = o.vals.length, ymax = Math.max(1, ...o.vals, ...(o.line || [0])) * 1.08;
  const bw = (W - m.l - m.r) / n;
  const sy = (v) => H - m.b - (v / ymax) * (H - m.t - m.b);
  let g = "";
  for (let i = 0; i <= 4; i++) {
    const v = (ymax * i) / 4;
    g += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${sy(v)}" y2="${sy(v)}"/><text x="${m.l - 6}" y="${sy(v) + 4}" text-anchor="end">${Math.round(v)}</text>`;
  }
  const every = Math.max(1, Math.ceil(n / 16));
  o.vals.forEach((v, i) => {
    const x = m.l + i * bw;
    const cls = v === 0 ? "var(--line2)" : "var(--c1)";
    g += `<rect data-tip="${(o.tipFor ? o.tipFor(i) : "").replace(/"/g, "&quot;")}" x="${x + 1}" y="${sy(v)}" width="${Math.max(1, bw - 2)}" height="${Math.max(v === 0 ? 2 : 0, H - m.b - sy(v))}" rx="2" fill="${cls}" ${o.hi === i ? 'stroke="var(--c2)" stroke-width="2"' : ""}/>`;
    if (i % every === 0) g += `<text x="${x + bw / 2}" y="${H - m.b + 14}" text-anchor="middle">${i}</text>`;
  });
  if (o.line) {
    const d = o.line.map((v, i) => `${i ? "L" : "M"}${(m.l + (i + 0.5) * bw).toFixed(1)},${sy(v).toFixed(1)}`).join("");
    g += `<path d="${d}" fill="none" stroke="var(--c2)" stroke-width="2" stroke-dasharray="5 4"/>`;
  }
  g += `<line class="axis" x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}"/><text x="${(W + m.l) / 2}" y="${H - 4}" text-anchor="middle">${o.labelX || "step"}</text>`;
  host.classList.add("chart");
  host.innerHTML = `<svg viewBox="0 0 ${W} ${H}">${g}</svg>`;
}

// ---------- misc ----------
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
function bindRange(id, cb) {
  const el = document.getElementById(id), out = document.getElementById(id + "_o");
  const upd = () => { if (out) out.textContent = el.value; cb(); };
  el.addEventListener("input", upd);
  if (out) out.textContent = el.value;
  return el;
}
function navHtml(active) {
  if (EMBED !== null) return "";
  const items = [["index.html", "Overview"], ["schedulers.html", "Schedulers"], ["utils.html", "samplers/utils"], ["mdlm.html", "MDLM sampler"], ["bd3lm.html", "BD3LM sampler"], ["training.html", "Training"]];
  return `<nav class="top">${items.map(([h, t]) => `<a href="${h}" class="${h === active ? "on" : ""}">${t}</a>`).join("")}</nav>`;
}

// ---------- code panel: lines tagged "@tag,tag2|code"; hl(['tag']) highlights them ----------
const _esc = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
function codePanel(host, title, code) {
  const rows = code.replace(/\n$/, "").split("\n").map((l) => {
    const m = /^@([\w,]+)\|(.*)$/.exec(l);
    return m ? { tags: m[1].split(","), text: m[2] } : { tags: [], text: l };
  });
  host.innerHTML = `<div class="code"><div class="ttl">${title}</div>` + rows.map((r, i) => { const ind = r.text.length - r.text.trimStart().length; return `<span class="ln ${r.tags.length ? "" : "plain"}" data-i="${i}" style="padding-left:calc(11px + ${ind + 4}ch);text-indent:-4ch">${_esc(r.text.trimStart()).replace(/(#.*)$/, '<span class="cm">$1</span>') || " "}</span>`; }).join("") + `</div>`;
  const box = host.querySelector(".code"), els = [...host.querySelectorAll(".ln")];
  return {
    hl(tags) {
      let first = -1;
      els.forEach((e, i) => { const on = rows[i].tags.some((t) => tags.includes(t)); e.classList.toggle("on", on); if (on && first < 0) first = i; });
      if (first >= 0) box.scrollTop = Math.max(0, els[first].offsetTop - box.clientHeight / 3);
    },
  };
}

// ---------- watch panel: rows of [name, value, note?] ----------
function watch(host, rows) {
  host.classList.add("watch");
  host.innerHTML = rows.map(([n, v, t]) => `<div><b>${n}</b><span>${_esc(String(v))}</span>${t ? `<span class="t">${_esc(t)}</span>` : ""}</div>`).join("");
}

// ---------- stepper (◀ play ▶ + slider, arrow keys) ----------
function stepper(host, onFrame, ms = 650) {
  host.innerHTML = `<div class="stepper"><button data-a="prev">◀</button><button data-a="play" class="primary">▶ play</button><button data-a="next">▶</button><input type="range" min="0" max="0" value="0"><span class="lab"></span></div>`;
  const rng = host.querySelector("input"), lab = host.querySelector(".lab"), play = host.querySelector('[data-a="play"]');
  let n = 1, cur = 0, timer = null;
  const stop = () => { if (timer) { clearInterval(timer); timer = null; } play.textContent = "▶ play"; };
  const set = (i) => { cur = Math.max(0, Math.min(n - 1, i)); rng.value = cur; lab.textContent = `${cur + 1} / ${n}`; onFrame(cur); };
  host.querySelector('[data-a="prev"]').onclick = () => { stop(); set(cur - 1); };
  host.querySelector('[data-a="next"]').onclick = () => { stop(); set(cur + 1); };
  play.onclick = () => {
    if (timer) return stop();
    if (cur >= n - 1) set(0);
    play.textContent = "❚❚ pause";
    timer = setInterval(() => { if (cur >= n - 1) return stop(); set(cur + 1); }, ms);
  };
  rng.addEventListener("input", () => { stop(); set(+rng.value); });
  document.addEventListener("keydown", (e) => {
    if (/INPUT|SELECT|TEXTAREA/.test((e.target.tagName || ""))) return;
    if (e.key === "ArrowRight") { stop(); set(cur + 1); } else if (e.key === "ArrowLeft") { stop(); set(cur - 1); }
  });
  const hashF = (/f=(\d+)/.exec(location.hash || "") || [])[1];
  let first = true;
  return { setN(m, keep) { n = m; rng.max = m - 1; const start = first && hashF !== undefined ? +hashF : keep ? Math.min(cur, m - 1) : 0; first = false; set(start); }, get: () => cur };
}
