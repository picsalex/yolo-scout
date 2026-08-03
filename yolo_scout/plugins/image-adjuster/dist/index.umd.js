/*!
 * @ultralytics/image-adjuster v1.0.0
 *
 * Floating brightness / contrast / overlay-opacity controller for the
 * FiftyOne sample viewer.  Pure vanilla JS — zero dependencies, no build step.
 *
 * The trigger button is fixed-positioned to sit visually inside FiftyOne's
 * bottom-right looker controls bar without touching React's DOM at all.
 * The control card opens just above it.
 */
(function () {
  "use strict";

  var _base = document.currentScript ? document.currentScript.src.replace(/\/dist\/[^/]+$/, '') : '';

  /* ══════════════════════════════════════════════════════════════════════════
   * 0.  SVG icon (loaded from images/slider.svg)
   * ══════════════════════════════════════════════════════════════════════════ */
  var ICON_SVG = '';
  fetch(_base + '/images/slider.svg').then(function(r){return r.text();}).then(function(t){ICON_SVG=t;});

  /* ══════════════════════════════════════════════════════════════════════════
   * 1.  Shared state
   * ══════════════════════════════════════════════════════════════════════════ */
  var adj  = { brightness: 100, contrast: 100, opacity: 100 };
  var card = null;
  var drag = null;

  /* ══════════════════════════════════════════════════════════════════════════
   * 2.  Finding the looker + canvases
   * ══════════════════════════════════════════════════════════════════════════ */
  function getLooker() {
    // _lookerControls_ only exists while a sample viewer is open.
    // Use it as the sole presence check — no canvas size measurement needed.
    return document.querySelector('[class*="_lookerControls_"]') || null;
  }

  /* ══════════════════════════════════════════════════════════════════════════
   * 2b. Recoil bridge — read/write colorScheme.opacity via React fiber
   *
   * Two complementary strategies:
   *
   * A) Direct looker update: find the FiftyOne looker object in the React
   *    fiber tree and call looker.updateOptions({alpha: value}) directly.
   *    This immediately re-renders the canvas overlay.
   *
   * B) Recoil state update: write colorScheme.opacity to the Recoil store
   *    and notify the batcher so Color Settings slider stays in sync.
   *
   * atomValues is a HashArrayMappedTrieMap (HAMT), not a JS Map — use its
   * native .get()/.set()/.delete()/.clone() methods.
   * ══════════════════════════════════════════════════════════════════════════ */

  // --- Strategy A: direct looker ---

  // Find the FiftyOne looker object stored in a useRef hook in the Modal fiber.
  // The looker exposes updateOptions(opts) and updateSample(sample) methods.
  function findLooker() {
    var rootEl = document.getElementById('root');
    if (!rootEl) return null;
    var rootKey = Object.keys(rootEl).find(function (k) { return k.startsWith('__reactContainer'); });
    if (!rootKey) return null;
    var fiber = rootEl[rootKey];
    var queue = [fiber];
    var checked = 0;
    while (queue.length > 0 && checked < 6000) {
      var f = queue.shift();
      checked++;
      try {
        var hook = f.memoizedState;
        while (hook) {
          var ms = hook.memoizedState;
          if (ms && typeof ms === 'object' && 'current' in ms && ms.current &&
              typeof ms.current.updateOptions === 'function' &&
              typeof ms.current.updateSample === 'function') {
            return ms.current;
          }
          hook = hook.next;
        }
      } catch (e) {}
      if (f.child) queue.push(f.child);
      if (f.sibling) queue.push(f.sibling);
    }
    return null;
  }

  // --- Strategy B: Recoil store ---

  // Walk up the fiber tree from a fiftyone-class element to find the Recoil store.
  // Returns { store, providerFiber } or null.
  function findRecoilStore() {
    var el = document.querySelector('[class*="fiftyone"]');
    if (!el) return null;
    var fiberKey = Object.keys(el).find(function (k) {
      return k.startsWith('__reactFiber') || k.startsWith('__reactInternalInstance');
    });
    if (!fiberKey) return null;
    var fiber = el[fiberKey];
    while (fiber) {
      var val = fiber.memoizedProps && fiber.memoizedProps.value;
      if (val && val.current && typeof val.current.getState === 'function') {
        return { store: val.current, providerFiber: fiber };
      }
      fiber = fiber.return;
    }
    return null;
  }

  // Read colorScheme.opacity from Recoil (0.0–1.0), or null if unavailable.
  function readColorScheme() {
    var ctx = findRecoilStore();
    if (!ctx) return null;
    try {
      var tree = ctx.store.getState().currentTree;
      var loadable = tree.atomValues.get('colorScheme') || tree.atomValues.get('__colorScheme_selector');
      if (loadable && loadable.contents && typeof loadable.contents.opacity === 'number') {
        return loadable.contents;
      }
      return null;
    } catch (e) { return null; }
  }

  // Update label opacity via both strategies.
  function writeOpacity(opacityFloat) {
    // Strategy A: direct looker update (immediate canvas re-render)
    try {
      var looker = findLooker();
      if (looker) looker.updateOptions({ alpha: opacityFloat });
    } catch (e) {}

    // Strategy B: Recoil state update (keeps Color Settings slider in sync)
    try {
      var ctx = findRecoilStore();
      if (ctx) {
        ctx.store.replaceState(function (prevTree) {
          var cs = prevTree.atomValues.get('colorScheme');
          if (!cs) return prevTree;
          var nc = Object.assign({}, cs.contents, { opacity: opacityFloat });
          var newAtomValues = prevTree.atomValues.clone();
          newAtomValues.set('colorScheme', { state: 'hasValue', contents: nc });
          newAtomValues.delete('__colorScheme_selector');
          var newDirty = new Set(prevTree.dirtyAtoms || []);
          newDirty.add('colorScheme');
          newDirty.add('__colorScheme_selector');
          return Object.assign({}, prevTree, { atomValues: newAtomValues, dirtyAtoms: newDirty });
        });
        // Notify Recoil's batcher to commit nextTree and schedule a React re-render.
        if (ctx.providerFiber && ctx.providerFiber.return) {
          var hook = ctx.providerFiber.return.memoizedState;
          while (hook) {
            var ms = hook.memoizedState;
            if (ms && typeof ms === 'object' && 'current' in ms && typeof ms.current === 'function') {
              try { ms.current(); } catch (e) {}
              break;
            }
            hook = hook.next;
          }
        }
      }
    } catch (e) {}
  }

  /* ══════════════════════════════════════════════════════════════════════════
   * 3.  Applying adjustments
   * ══════════════════════════════════════════════════════════════════════════ */
  function applyAdj() {
    if (!getLooker()) return;
    var flt = "brightness(" + adj.brightness + "%) contrast(" + adj.contrast + "%)";
    // Multiple _lookerCanvas_ elements exist (filmstrip + main viewer).
    // Target by largest area (w*h) — the main sample canvas is always tallest.
    var canvases = Array.from(document.querySelectorAll('[class*="_lookerCanvas_"]'));
    var canvas = canvases.reduce(function (best, c) {
      var area = c.offsetWidth * c.offsetHeight;
      var bestArea = best ? best.offsetWidth * best.offsetHeight : 0;
      return area > bestArea ? c : best;
    }, null);
    if (canvas) canvas.style.filter = flt;
    // Update FiftyOne's label opacity via the Recoil colorScheme atom (0.0–1.0).
    writeOpacity(adj.opacity / 100);
  }

  /* ══════════════════════════════════════════════════════════════════════════
   * 4.  CSS
   * ══════════════════════════════════════════════════════════════════════════ */
  var STYLES = (
    /* ── Trigger button — lives inside the toolbar as a normal child ── */
    "#fo-ia-btn{" +
      "width:28px;height:28px;border-radius:4px;" +
      "background:transparent;border:none;cursor:pointer;" +
      "display:inline-flex;align-items:center;justify-content:center;" +
      "color:rgba(255,255,255,0.6);" +
      "transition:color .15s,background .15s;padding:4px;" +
      "flex-shrink:0;" +
    "}" +
    "#fo-ia-btn:hover{color:#fff;background:rgba(255,255,255,.12);}" +
    "#fo-ia-btn.ia-active{color:#f97316;background:rgba(249,115,22,.2);}" +

    /* ── Control card — opens just above the trigger button ── */
    "#fo-ia{" +
      "position:fixed;bottom:60px;right:8px;width:224px;" +
      "background:#1c1c1c;border:1px solid #3d3d3d;border-radius:8px;" +
      "padding:10px 14px 12px;z-index:2147483647;" +
      "font:13px/1.4 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;" +
      "color:#e0e0e0;box-shadow:0 8px 28px rgba(0,0,0,.65);user-select:none;" +
    "}" +
    "#fo-ia .ia-hdr{" +
      "display:flex;justify-content:space-between;align-items:center;" +
      "margin-bottom:10px;cursor:move;font-weight:600;font-size:13px;color:#fff;" +
    "}" +
    "#fo-ia .ia-title{display:flex;align-items:center;gap:6px;}" +
    "#fo-ia .ia-x{cursor:pointer;opacity:.5;font-size:17px;line-height:1;padding:0 2px;}" +
    "#fo-ia .ia-x:hover{opacity:1;}" +
    "#fo-ia .ia-lbl{display:block;margin-bottom:9px;}" +
    "#fo-ia .ia-row{" +
      "display:flex;justify-content:space-between;margin-bottom:3px;" +
      "font-size:11px;color:#999;" +
    "}" +
    "#fo-ia input[type=range]{" +
      "display:block;width:100%;margin:0;height:4px;" +
      "accent-color:#f97316;cursor:pointer;" +
    "}" +
    "#fo-ia .ia-reset{" +
      "margin-top:6px;width:100%;padding:5px;" +
      "background:#2c2c2c;border:1px solid #4a4a4a;border-radius:4px;" +
      "color:#bbb;cursor:pointer;font-size:12px;" +
    "}" +
    "#fo-ia .ia-reset:hover{background:#383838;color:#fff;}"
  );

  function ensureStyles() {
    if (document.getElementById("fo-ia-css")) return;
    var el = document.createElement("style");
    el.id  = "fo-ia-css";
    el.textContent = STYLES;
    document.head.appendChild(el);
  }

  /* ══════════════════════════════════════════════════════════════════════════
   * 5.  Control card
   * ══════════════════════════════════════════════════════════════════════════ */
  function sliderRow(key, label, min, max) {
    return (
      '<label class="ia-lbl">' +
        '<div class="ia-row">' +
          '<span>' + label + '</span>' +
          '<span id="fo-ia-v-' + key + '">' + adj[key] + '%</span>' +
        '</div>' +
        '<input type="range" min="' + min + '" max="' + max + '"' +
               ' value="' + adj[key] + '" data-key="' + key + '">' +
      '</label>'
    );
  }

  function buildCard() {
    // Sync adj.opacity with FiftyOne's current label opacity before building UI.
    // Try looker first (most direct), fall back to Recoil atom.
    var looker = findLooker();
    if (looker && looker.state && looker.state.options && typeof looker.state.options.alpha === 'number') {
      adj.opacity = Math.round(looker.state.options.alpha * 100);
    } else {
      var currentScheme = readColorScheme();
      if (currentScheme !== null) {
        adj.opacity = Math.round(currentScheme.opacity * 100);
      }
    }

    var div = document.createElement("div");
    div.id  = "fo-ia";
    div.innerHTML = (
      '<div class="ia-hdr">' +
        '<span class="ia-title">' + ICON_SVG + '<span>Image Adjuster</span></span>' +
        '<span class="ia-x" title="Close">✕</span>' +
      '</div>' +
      sliderRow("brightness", "Brightness", 0, 200) +
      sliderRow("contrast",   "Contrast",   0, 200) +
      sliderRow("opacity",    "Opacity",     0, 100) +
      '<button type="button" class="ia-reset">Reset</button>'
    );

    div.querySelectorAll("input[type=range]").forEach(function (inp) {
      inp.addEventListener("input", function () {
        var k = this.dataset.key;
        adj[k] = +this.value;
        var vEl = document.getElementById("fo-ia-v-" + k);
        if (vEl) vEl.textContent = adj[k] + "%";
        applyAdj();
      });
    });

    div.querySelector(".ia-reset").addEventListener("click", function () {
      adj = { brightness: 100, contrast: 100, opacity: 100 };
      div.querySelectorAll("input[type=range]").forEach(function (inp) {
        inp.value = 100;
        var el = document.getElementById("fo-ia-v-" + inp.dataset.key);
        if (el) el.textContent = "100%";
      });
      applyAdj();
    });

    div.querySelector(".ia-x").addEventListener("click", function () {
      div.remove();
      card = null;
      var btn = document.getElementById("fo-ia-btn");
      if (btn) btn.classList.remove("ia-active");
    });

    div.querySelector(".ia-hdr").addEventListener("mousedown", function (e) {
      if (e.target.classList.contains("ia-x")) return;
      var r = div.getBoundingClientRect();
      drag = { ox: e.clientX - r.left, oy: e.clientY - r.top };
    });

    return div;
  }

  /* ══════════════════════════════════════════════════════════════════════════
   * 6.  Trigger button
   * ══════════════════════════════════════════════════════════════════════════ */
  function buildTrigger() {
    // Match FiftyOne's own toolbar item structure exactly:
    // <div class="_lookerClickable_..." style="padding:2px;display:flex;grid-area:...">
    var btn = document.createElement("div");
    btn.id    = "fo-ia-btn";
    btn.title = "Image Adjuster";
    btn.innerHTML = ICON_SVG;
    // Copy the inline style FiftyOne uses for every toolbar icon.
    // Borrow FiftyOne's own hover/active class + compute next grid column.
    var tb = getLooker();
    var nextCol = 17; // fallback
    if (tb) {
      var siblings = tb.querySelectorAll('[class*="_lookerClickable_"]');
      var lastSibling = siblings[siblings.length - 1];
      if (lastSibling) {
        btn.className = lastSibling.className;
        var cols = Array.from(siblings).map(function(el) {
          return parseInt((el.style.gridArea || "").split("/")[1]) || 99;
        });
        var minCol = Math.min.apply(null, cols);
        nextCol = minCol - 1;
      }
    }
    btn.style.cssText = "padding:2px;display:flex;cursor:pointer;grid-area:2 / " + nextCol + " / 2 / " + nextCol + ";";
    btn.addEventListener("click", function () {
      var existing = document.getElementById("fo-ia");
      if (existing) {
        existing.remove();
        card = null;
        btn.classList.remove("ia-active");
      } else {
        card = buildCard();
        // Position card above the trigger button.
        var br = btn.getBoundingClientRect();
        card.style.bottom = "auto";
        card.style.right  = "auto";
        card.style.top    = Math.max(8, br.top - 210) + "px";
        card.style.left   = Math.max(8, br.left - 180) + "px";
        document.body.appendChild(card);
        btn.classList.add("ia-active");
        setTimeout(applyAdj, 150);
      }
    });
    return btn;
  }

  var toolbarObs = null;

  function injectIntoToolbar() {
    var tb = getLooker();
    if (!tb) return false;
    // Already inside the toolbar — nothing to do.
    if (tb.contains(document.getElementById("fo-ia-btn"))) return true;
    // Build fresh and append inside the toolbar pill.
    var old = document.getElementById("fo-ia-btn");
    if (old) old.remove();
    tb.appendChild(buildTrigger());
    return true;
  }

  function showTrigger() {
    ensureStyles();
    if (!injectIntoToolbar()) return;

    // Guard: if React ever removes our button, put it back immediately.
    if (toolbarObs) toolbarObs.disconnect();
    var tb = getLooker();
    if (tb) {
      toolbarObs = new MutationObserver(function () {
        if (!tb.contains(document.getElementById("fo-ia-btn"))) {
          tb.appendChild(buildTrigger());
          if (card) document.getElementById("fo-ia-btn").classList.add("ia-active");
        }
      });
      toolbarObs.observe(tb, { childList: true });
    }
  }

  function hideTrigger() {
    if (toolbarObs) { toolbarObs.disconnect(); toolbarObs = null; }
    var btn = document.getElementById("fo-ia-btn");
    if (btn) btn.remove();
    var existingCard = document.getElementById("fo-ia");
    if (existingCard) existingCard.remove();
    card = null;
    adj = { brightness: 100, contrast: 100, opacity: 100 };
    // Clear filter and opacity from all canvases when leaving the viewer.
    document.querySelectorAll('[class*="_lookerCanvas_"]').forEach(function (c) {
      c.style.filter = "";
    });
    // Opacity persists in FiftyOne's Recoil state when the viewer closes —
    // consistent with FiftyOne's native Color Settings behaviour.
  }

  /* ══════════════════════════════════════════════════════════════════════════
   * 7.  Drag
   * ══════════════════════════════════════════════════════════════════════════ */
  document.addEventListener("mousemove", function (e) {
    if (!drag || !card) return;
    card.style.top    = (e.clientY - drag.oy) + "px";
    card.style.left   = (e.clientX - drag.ox) + "px";
    card.style.bottom = "auto";
    card.style.right  = "auto";
  });
  document.addEventListener("mouseup", function () { drag = null; });

  /* ══════════════════════════════════════════════════════════════════════════
   * 8.  Main observer
   * ══════════════════════════════════════════════════════════════════════════ */
  var debounceTimer = null;
  function onMutation() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(function () {
      if (getLooker()) {
        showTrigger();
        if (document.getElementById("fo-ia")) applyAdj();
      } else {
        hideTrigger();
      }
    }, 80); // short debounce so closing the modal hides the card quickly
  }

  new MutationObserver(onMutation).observe(document.body, {
    childList: true,
    subtree: true,
  });

  // Handle page-load case.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", onMutation);
  } else {
    onMutation();
  }

})();
