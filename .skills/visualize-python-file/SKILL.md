---
name: visualize-python-file
description: How to write an interactive, line-by-line visualization of a Python file or function in this repo — step-through frames on the left, the real code with the executing lines highlighted plus a live-variable panel on the right — and how to verify it. Use when asked to "visualize", "explain with a visualization", or add a page under visualizations/.
---

# Visualizing a Python file

The output is one self-contained HTML page in `visualizations/`, built from the shared
`assets/common.js` + `assets/style.css`. No build step, no CDN, no external requests.
Good examples to copy from: `utils.html` (tabs, one function each), `mdlm.html`
(phases per step + an explorer tab), `bd3lm.html` (state strip + before/now counters),
`schedulers.html` (sliders instead of a stepper).

## The style (this is what the user likes — keep to it)

- **Line by line, side by side.** The code that is executing is on screen next to the picture,
  its lines highlighted, with a **watch panel** of the live variables/shapes underneath.
  The reader steps through with the buttons or ← →.
- **Frames are the code's real phases, in real execution order** (e.g. `fwd → rollback → select`,
  not the order that is easiest to explain). Each frame changes something visible.
- **One short caption per frame**: the phase name in bold, then what happens. No lectures.
- **Explain only the crux** — what this code does and why it is shaped that way. Do not add long
  prose, side notes, caveats about non-essentials, or anything the picture already shows.
  Longer explanation belongs in the blog post, and there only what the visualization cannot show.
- **Interactive inputs on top** (sliders, a text box, a checkbox) that rebuild the frames. Several
  functions in one file → one tab per function. A concept that needs exploring rather than
  stepping (a mask, a window) → its own tab with hover.
- When the code is about cost, show **before / now counters** (tokens through the model, bytes copied).
- The code shown must be the **real, current code**, condensed. Re-read the file first: it may have
  been refactored since you last saw it, and stale names in a snippet are a bug.

## Workflow

1. **Read the current file.** Identify the functions, the phases of the main loop, the tensors and
   shapes that carry the idea, and the options that change behaviour.
2. **Plan frames.** Write the phases in execution order, the code lines each one owns, and what
   changes on screen for each (a cell colours, a bar appears, a strip grows, a shape changes).
3. **Port the algorithm to a small JS engine that produces all frames up-front** (state snapshots plus
   whatever the frame draws). Keep the logic identical to the Python, including quirks
   (float32 inside a scheduler, `torch.round` is half-to-even, tie-breaking, `-inf` guards).
   Fake only the model: random confidences / token ids are fine, and say so in the UI if it matters.
4. **Verify the port against the real Python.** Run both on many inputs and require an exact match
   (write a throw-away compare script: Python dumps JSON, node compares). A visualization that
   disagrees with the code is worse than none.
5. **Build the page from `page-template.html`** (copy it into `visualizations/`).
6. **Check it, twice** (see Verification). Never report a page as done from the harness alone.
7. Wire it up: nav item (`navHtml` list in `assets/common.js`), card on `index.html`, and the
   `Makefile` targets if the pages are served together.

## Building blocks (`assets/common.js`)

| Helper | Use |
|---|---|
| `codePanel(host, title, code)` → `{hl(tags)}` | Code with `@tag,tag2\|line` markers. `hl(["fwd"])` lights every line carrying that tag and scrolls to it. Lines without a tag stay dimly visible. Long lines wrap with a hanging indent. |
| `watch(host, [[name, value, note?], …])` | The live-variable panel (shapes, values, "before → after"). |
| `stepper(host, onFrame, ms)` → `{setN(n, keepFrame), get()}` | ◀ play ▶ slider, ← → keys, `#f=N` deep link. Call `setN` whenever the frames are rebuilt. |
| `lineChart(host, {xs|x, series, y, markers, dots, …})`, `barChart(host, {vals, line})` | SVG charts with a hover crosshair / tooltip. |
| `bindRange(id, cb)` | Range input + its `<output id="id_o">` + callback. |
| `mulberry32(seed)` | Seeded RNG so frames are reproducible. |
| `numTransfer`, `ALPHA`, `reverseMaskProb32`, `kappaCubic` | JS ports of the repo's scheduler / `get_num_transfer_tokens`. Reuse instead of re-porting. |
| `navHtml(active)`, `EMBED` | Nav bar; `EMBED` is the `?embed=` value (`null` when not embedded). |

Layout classes (`assets/style.css`): `.panel`, `.split` (viz left, code right; stays two columns
inside embeds), `.sticky`, `.cap`, `.controls`, `.tabs`, `.legend`, `.stat` (big before/now numbers),
`.numgrid/.ncell` (small integer matrices), and the token grid: `.seq > .row > .cell` with states
`prompt mask done eos pad kv tmp sel dim proc`, `.row.bars` for confidence bars, `.row.win` for a
window bracket. Colours are CSS variables (`--c1..--c4`, `--bad`, …) and adapt to dark mode.

## Code-snippet rules

- Copy from the real file; drop branches that are irrelevant to this page (say so in a comment if
  something important is omitted), keep the real names, put shapes in comments (`# [B, T, V]`).
- Tag lines by the frame phase that executes them; a line can carry several tags (`@conf,topk|`).
  The panel title is the file path and function.
- Keep lines ≲ 80 columns; the panel wraps but very long lines read badly.

## Verification

- `node .skills/visualize-python-file/check_frames.js visualizations/<page>.html` — runs the page
  against a stub DOM and visits **every frame** of every stepper. Catches JS errors and bad state.
  It cannot see layout. For pages with tabs, call `activate("<tab>")` in a temporary copy to visit each.
- `.skills/visualize-python-file/screenshot.sh "visualizations/<page>.html#f=N" out.png [height] [width]`
  then **Read the PNG** and look at it. Check mid-run frames (`#f=N`), each tab, the edge configs
  (0 masks, `k = 0`, temperature 0, one row), and the embedded version (`?embed=…`).
- On shared machines cap CPU threads for any Python check you run (`OMP_NUM_THREADS=2`,
  `torch.set_num_threads(2)`) and keep test grids small.

## Pitfalls we already hit

- **Size cells from the container width**, not a constant: `clientWidth / T`, clamped. Constants
  clipped the canvas as soon as the column got narrower.
- **`hidden` loses to `display:flex`** unless `[hidden]{display:none!important}` is in the CSS (it is).
- **Grid children shrink to nothing** inside `display:grid; place-items:center` — give bars `width:100%`.
- **`vh` inside an iframe** is the iframe's own height: no `max-height: NNvh` in embed mode (handled by
  `.embed .code{max-height:none}`), and report height with `document.body.offsetHeight`, not
  `scrollHeight` (which can never shrink below the iframe's current height).
- **Negative bars**: leave room below for the value label and the category label.
- **Screenshots of long pages**: use one tall window and crop; `#anchor` jumps and lazy iframes
  outside the window produce blank images.
- **Stale snippets** after a refactor (renamed hooks, moved code) — re-read the source before finishing.

## Embedding in the blog post (`visualizations/scheduler_sampler.html`)

Add `<figure><iframe data-embed="page.html?embed=<tab>" height="320" loading="lazy"></iframe>
<figcaption>…<a href="page.html">Open full page</a></figcaption></figure>`. In `?embed=` mode the page
hides its nav/title, opens the named tab, and posts its height to the parent, which resizes the iframe.
Give each embed one sentence of "what to try"; put the explanation the picture can't carry in the
surrounding prose, and nothing else.
