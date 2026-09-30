#!/usr/bin/env node
// Smoke-test an interactive visualization page WITHOUT a browser:
// runs the page's inline scripts against a stub DOM and visits every frame of every stepper().
// usage: node check_frames.js path/to/page.html
// It catches JS errors in any frame. It does NOT check layout: use screenshot.sh for that.
const fs = require("fs"), path = require("path");
const page = path.resolve(process.argv[2]);
const html = fs.readFileSync(page, "utf8");
const dir = path.dirname(page);

const mk = () => ({
  value: "", checked: false, innerHTML: "", textContent: "", options: [], style: {}, dataset: {}, hidden: false,
  classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
  addEventListener() {}, setAttribute() {}, getAttribute() { return ""; },
  querySelector() { return mk(); }, querySelectorAll() { return []; },
  getBoundingClientRect() { return { left: 0, width: 1, top: 0 }; },
  offsetWidth: 0, clientWidth: 600, clientHeight: 400, scrollTop: 0, max: 0, min: 0, closest() { return null; },
});
const els = {};
global.document = {
  getElementById: (id) => (els[id] = els[id] || mk()),
  querySelector: () => mk(), querySelectorAll: () => [], addEventListener() {}, write() {},
  createElement: mk, body: mk(), documentElement: mk(),
};
global.window = { innerWidth: 1000 };
global.location = { search: "", hash: "", pathname: "/x.html" };
global.parent = { postMessage() {} };
global.addEventListener = () => {};

// give every <input>/<select> in the page its default value
for (const m of html.matchAll(/<(?:input|select)[^>]*id="([^"]+)"[^>]*>/g)) {
  const e = document.getElementById(m[1]), v = /value="([^"]*)"/.exec(m[0]);
  if (v) e.value = v[1];
  if (/checked/.test(m[0])) e.checked = true;
}
for (const m of html.matchAll(/<select[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) {
  const o = /<option value="([^"]*)"/.exec(m[2]);
  if (o) document.getElementById(m[1]).value = o[1];
}

// shared assets (relative to the page), with top-level const/let/function made visible to the page scripts
const common = fs.readFileSync(path.join(dir, "assets/common.js"), "utf8").replace(/\(function \(\) \{[\s\S]*?\}\)\(\);/, "");
(0, eval)(common.replace(/^const /gm, "var "));
// visit every frame instead of waiting for clicks
(0, eval)('var stepper = function (host, onFrame) { return { setN: function (m) { console.log("  stepper: visiting " + m + " frames"); for (var i = 0; i < m; i++) onFrame(i); onFrame(0); }, get: function () { return 0; } }; };');

let ok = true;
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]).filter((s) => !s.includes("document.write(navHtml"));
for (const s of scripts) {
  try { (0, eval)(s.replace(/^let /gm, "var ").replace(/^const /gm, "var ")); }
  catch (e) { ok = false; console.log(path.basename(page), "RUNTIME ERROR:", e.stack.split("\n").slice(0, 3).join(" | ")); }
}
if (ok) console.log(path.basename(page), "ran without errors");
process.exit(ok ? 0 : 1);
