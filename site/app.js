/* BitCliff playground — static, vanilla JS, no build step, no frameworks.
 *
 * Loads site/fixtures/pilot-0a.json (exported by tools/export_fixtures.py)
 * and renders two views over it:
 *   - Ladder view: every rung's output stacked for one prompt, F16 on top.
 *   - A/B view: two rungs side by side with a snapping slider between them.
 *
 * Routing is hash-based so every view is a stable permalink:
 *   #/ladder/<item_id>
 *   #/ab/<item_id>/<rungA>/<rungB>
 */

(function () {
  "use strict";

  var FIXTURES_URL = "fixtures/pilot-0a.json";
  var DEFAULT_AB_RUNG_A = "F16";
  var DEFAULT_AB_RUNG_B = "IQ2_M";

  var GRADE_LABEL = {
    correct: "correct",
    partial: "partial",
    wrong: "wrong",
    unscored: "unscored",
  };

  var app = document.getElementById("app");
  var toastEl = document.getElementById("toast");
  var copyLinkBtn = document.getElementById("copy-link-btn");
  var viewTabs = Array.prototype.slice.call(document.querySelectorAll(".view-tab"));

  /** @type {{ladder:string[], rungs:Object, items:Array, itemsById:Object, bySuite:Object}} */
  var DATA = null;

  var state = {
    view: "ladder", // "ladder" | "ab"
    itemId: null,
    rungA: DEFAULT_AB_RUNG_A,
    rungB: DEFAULT_AB_RUNG_B,
  };

  // ---------------------------------------------------------------------
  // utilities
  // ---------------------------------------------------------------------

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function el(html) {
    var t = document.createElement("template");
    t.innerHTML = html.trim();
    return t.content.firstElementChild;
  }

  function showToast(msg) {
    toastEl.textContent = msg;
    toastEl.classList.add("is-visible");
    window.clearTimeout(showToast._t);
    showToast._t = window.setTimeout(function () {
      toastEl.classList.remove("is-visible");
    }, 1800);
  }

  function copyToClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(
        function () { showToast("Link copied"); },
        function () { fallbackCopy(text); }
      );
    } else {
      fallbackCopy(text);
    }
  }

  function fallbackCopy(text) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand("copy");
      showToast("Link copied");
    } catch (e) {
      showToast("Copy failed — copy from the address bar");
    }
    document.body.removeChild(ta);
  }

  /**
   * Render `text` as HTML, highlighting every word from `divergenceIndex`
   * (word-level, str.split()-equivalent) onward. Mirrors the fixture's
   * word-level divergence-vs-F16 index. Returns escaped, highlight-wrapped
   * HTML. divergenceIndex === null/undefined means "no divergence" (F16
   * itself, or a rung whose output matched F16 exactly) -> no highlight.
   */
  function renderOutputHtml(text, divergenceIndex) {
    if (divergenceIndex === null || divergenceIndex === undefined) {
      return escapeHtml(text);
    }
    var parts = text.split(/(\s+)/); // alternating word / whitespace chunks
    var wordCount = 0;
    var html = "";
    var inHighlight = false;
    for (var i = 0; i < parts.length; i++) {
      var part = parts[i];
      if (part === "") continue;
      var isWhitespace = /^\s+$/.test(part);
      if (!isWhitespace) {
        if (wordCount === divergenceIndex) {
          inHighlight = true;
        }
        wordCount++;
      }
      var esc = escapeHtml(part);
      html += inHighlight
        ? '<span class="divergence-hl" title="first word where this rung’s output diverges from full precision (word-level, pilot)">' + esc + "</span>"
        : esc;
    }
    return html;
  }

  function gradeBadge(grade) {
    var cls = "badge-" + (GRADE_LABEL[grade] ? grade : "unscored");
    var label = GRADE_LABEL[grade] || grade || "unscored";
    return '<span class="badge ' + cls + '">' + escapeHtml(label) + "</span>";
  }

  function cliffColorVar(rungLabel) {
    var idx = DATA.ladder.indexOf(rungLabel);
    if (idx < 0) idx = 0;
    return "var(--cliff-" + idx + ")";
  }

  function shortSuiteLabel(item) {
    return item.suite.charAt(0).toUpperCase() + item.suite.slice(1);
  }

  // ---------------------------------------------------------------------
  // routing
  // ---------------------------------------------------------------------

  function parseHash() {
    var raw = window.location.hash.replace(/^#\/?/, "");
    var parts = raw.split("/").filter(function (p) { return p.length > 0; });
    // ["ladder", "<item_id>"]  or  ["ab", "<item_id>", "<rungA>", "<rungB>"]
    if (parts[0] === "ladder" && parts[1]) {
      return { view: "ladder", itemId: decodeURIComponent(parts[1]) };
    }
    if (parts[0] === "ab" && parts[1]) {
      var rungA = parts[2] ? decodeURIComponent(parts[2]) : DEFAULT_AB_RUNG_A;
      var rungB = parts[3] ? decodeURIComponent(parts[3]) : DEFAULT_AB_RUNG_B;
      return { view: "ab", itemId: decodeURIComponent(parts[1]), rungA: rungA, rungB: rungB };
    }
    return null;
  }

  function buildHash(view, itemId, rungA, rungB) {
    if (view === "ab") {
      return "#/ab/" + encodeURIComponent(itemId) + "/" + encodeURIComponent(rungA) + "/" + encodeURIComponent(rungB);
    }
    return "#/ladder/" + encodeURIComponent(itemId);
  }

  function navigate(view, itemId, rungA, rungB, replace) {
    var hash = buildHash(view, itemId, rungA, rungB);
    if (replace) {
      var url = window.location.pathname + window.location.search + hash;
      window.history.replaceState(null, "", url);
      onRouteChange();
    } else {
      window.location.hash = hash;
    }
  }

  function onRouteChange() {
    var parsed = parseHash();
    if (!parsed || !DATA.itemsById[parsed.itemId]) {
      var fallbackId = DATA.items[0].id;
      navigate("ladder", fallbackId, DEFAULT_AB_RUNG_A, DEFAULT_AB_RUNG_B, true);
      return;
    }
    state.view = parsed.view;
    state.itemId = parsed.itemId;
    if (parsed.view === "ab") {
      state.rungA = DATA.rungs[parsed.rungA] ? parsed.rungA : DEFAULT_AB_RUNG_A;
      state.rungB = DATA.rungs[parsed.rungB] ? parsed.rungB : DEFAULT_AB_RUNG_B;
    }
    render();
  }

  // ---------------------------------------------------------------------
  // rendering: shared pieces
  // ---------------------------------------------------------------------

  function renderPicker() {
    var suites = DATA.suiteOrder;
    var groupsHtml = suites.map(function (suite) {
      var itemsInSuite = DATA.bySuite[suite];
      var chips = itemsInSuite.map(function (it, idx) {
        var active = it.id === state.itemId ? " is-active" : "";
        var preview = it.prompt.length > 70 ? it.prompt.slice(0, 70) + "…" : it.prompt;
        return (
          '<button class="picker-chip' + active + '" type="button" data-item-id="' +
          escapeHtml(it.id) + '" title="' + escapeHtml(it.id + ": " + preview) + '">' +
          "#" + (idx + 1) + "</button>"
        );
      }).join("");
      return (
        '<div class="picker-group">' +
        "<h3>" + escapeHtml(suite) + " (" + itemsInSuite.length + ")</h3>" +
        '<div class="picker-chips">' + chips + "</div>" +
        "</div>"
      );
    }).join("");

    return (
      '<div class="picker">' +
      groupsHtml +
      '<div class="picker-random">' +
      '<button class="random-btn" type="button" id="random-btn" ' +
      'title="Uniform draw over every fixture item, all suites — the honesty check on the curated picks above">' +
      '<span class="random-die" aria-hidden="true">&#127922;</span> Random</button>' +
      "</div>" +
      "</div>"
    );
  }

  function renderPromptBlock(item) {
    var expectedHtml = "";
    if (item.expected !== null && item.expected !== undefined) {
      var exp = Array.isArray(item.expected) ? item.expected.join(", ") : String(item.expected);
      expectedHtml = '<div class="prompt-expected">expected: ' + escapeHtml(exp) + "</div>";
    }
    return (
      '<div class="prompt-block">' +
      '<div class="prompt-meta">' +
      '<span class="suite-tag suite-' + escapeHtml(item.suite) + '">' + escapeHtml(item.suite) + "</span>" +
      "<span>" + escapeHtml(item.id) + "</span>" +
      "</div>" +
      '<div class="prompt-text">' + escapeHtml(item.prompt) + "</div>" +
      expectedHtml +
      "</div>"
    );
  }

  /**
   * Render one output card for `rungLabel` on `item`.
   * `compact` shrinks the max-height for the stacked ladder view.
   */
  function renderRungCard(item, rungLabel) {
    var meta = DATA.rungs[rungLabel];
    var rungData = item.rungs[rungLabel];
    if (!meta || !rungData) {
      return (
        '<div class="rung-card"><div class="rung-body"><em>missing data for ' +
        escapeHtml(rungLabel) + "</em></div></div>"
      );
    }

    var divergenceIndex = rungLabel === "F16" ? null : rungData.divergence;
    var outputHtml = renderOutputHtml(rungData.text, divergenceIndex);

    var flagBadges = "";
    if (rungData.truncated) flagBadges += '<span class="badge badge-flag">truncated</span>';
    if (rungData.loop) flagBadges += '<span class="badge badge-flag">loop</span>';

    var inhouseBadge = "";
    if (meta.spectacle_only) {
      inhouseBadge =
        '<span class="badge badge-inhouse" title="' + escapeHtml(DATA.inHouseTag) + '">' +
        escapeHtml(DATA.inHouseTag) +
        "</span>";
    }

    var divergenceNote = "";
    if (divergenceIndex !== null && divergenceIndex !== undefined) {
      divergenceNote =
        '<span class="rung-fact" title="first word where this rung’s output diverges from full precision (word-level, pilot)">' +
        "diverges @ word " + divergenceIndex + "</span>";
    }

    return (
      '<div class="rung-card" data-rung="' + escapeHtml(rungLabel) + '">' +
      '<div class="rung-rail" style="background:' + cliffColorVar(rungLabel) + '"></div>' +
      '<div class="rung-body">' +
      '<div class="rung-head">' +
      '<span class="rung-label">' + escapeHtml(rungLabel) + "</span>" +
      '<span class="rung-uploader">' + escapeHtml(meta.uploader || "—") +
      (meta.imatrix ? " · imatrix" : "") + "</span>" +
      inhouseBadge +
      gradeBadge(rungData.state) +
      flagBadges +
      '<span class="rung-spacer"></span>' +
      '<span class="rung-facts">' +
      "<span>" + escapeHtml(meta.size_human) + "</span>" +
      "<span title=\"sha256 (first 12 chars)\">" + escapeHtml(meta.sha256_short) + "</span>" +
      divergenceNote +
      "</span>" +
      "</div>" +
      '<div class="rung-output">' + outputHtml + "</div>" +
      "</div>" +
      "</div>"
    );
  }

  // ---------------------------------------------------------------------
  // ladder view
  // ---------------------------------------------------------------------

  function renderLadderView(item) {
    var cards = DATA.ladder.map(function (label) {
      return renderRungCard(item, label);
    }).join("");
    return '<div class="ladder-view">' + cards + "</div>";
  }

  // ---------------------------------------------------------------------
  // A/B view
  // ---------------------------------------------------------------------

  function renderAbSelects() {
    var options = DATA.ladder.map(function (label) {
      return '<option value="' + escapeHtml(label) + '">' + escapeHtml(label) + "</option>";
    }).join("");
    return (
      '<div class="ab-selects">' +
      '<label class="ab-select-group"><span class="ab-swatch ab-swatch-a"></span>A ' +
      '<select id="ab-select-a">' + options + "</select></label>" +
      '<label class="ab-select-group"><span class="ab-swatch ab-swatch-b"></span>B ' +
      '<select id="ab-select-b">' + options + "</select></label>" +
      "</div>"
    );
  }

  function renderAbTrack() {
    var n = DATA.ladder.length;
    var ticks = DATA.ladder.map(function (label, i) {
      var pct = (i / (n - 1)) * 100;
      return (
        '<div class="ab-tick" style="left:' + pct + '%" data-index="' + i + '"></div>' +
        '<div class="ab-tick-label" style="left:' + pct + '%">' + escapeHtml(label) + "</div>"
      );
    }).join("");
    return (
      '<div class="ab-slider-block">' +
      '<div class="ab-slider-legend"><span>F16 (stable)</span><span>IQ1_S (deranged)</span></div>' +
      '<div class="ab-track-wrap">' +
      '<div class="ab-track" id="ab-track">' +
      ticks +
      '<div class="ab-handle ab-handle-a" id="ab-handle-a" tabindex="0" role="slider" aria-label="Rung A">A</div>' +
      '<div class="ab-handle ab-handle-b" id="ab-handle-b" tabindex="0" role="slider" aria-label="Rung B">B</div>' +
      "</div>" +
      "</div>" +
      renderAbSelects() +
      "</div>"
    );
  }

  function renderAbView(item) {
    return (
      renderAbTrack() +
      '<div class="ab-panes">' +
      '<div class="ab-pane">' + renderRungCard(item, state.rungA) + "</div>" +
      '<div class="ab-pane">' + renderRungCard(item, state.rungB) + "</div>" +
      "</div>"
    );
  }

  function positionAbHandles() {
    var n = DATA.ladder.length;
    var idxA = DATA.ladder.indexOf(state.rungA);
    var idxB = DATA.ladder.indexOf(state.rungB);
    var handleA = document.getElementById("ab-handle-a");
    var handleB = document.getElementById("ab-handle-b");
    if (handleA) handleA.style.left = (idxA / (n - 1)) * 100 + "%";
    if (handleB) handleB.style.left = (idxB / (n - 1)) * 100 + "%";
    var selA = document.getElementById("ab-select-a");
    var selB = document.getElementById("ab-select-b");
    if (selA) selA.value = state.rungA;
    if (selB) selB.value = state.rungB;
  }

  function updateAbPanesOnly() {
    var item = DATA.itemsById[state.itemId];
    var panes = document.querySelectorAll(".ab-pane");
    if (panes.length === 2) {
      panes[0].innerHTML = renderRungCard(item, state.rungA);
      panes[1].innerHTML = renderRungCard(item, state.rungB);
    }
    positionAbHandles();
  }

  function setAbRung(which, rungLabel) {
    if (!DATA.rungs[rungLabel]) return;
    if (which === "a") state.rungA = rungLabel;
    else state.rungB = rungLabel;
    navigate("ab", state.itemId, state.rungA, state.rungB, true);
    updateAbPanesOnly();
  }

  function attachAbHandlers() {
    var track = document.getElementById("ab-track");
    if (!track) return;
    var n = DATA.ladder.length;

    function nearestIndex(clientX) {
      var rect = track.getBoundingClientRect();
      var ratio = (clientX - rect.left) / rect.width;
      ratio = Math.max(0, Math.min(1, ratio));
      return Math.round(ratio * (n - 1));
    }

    function startDrag(which, ev) {
      ev.preventDefault();
      function onMove(moveEv) {
        var clientX = moveEv.touches ? moveEv.touches[0].clientX : moveEv.clientX;
        var idx = nearestIndex(clientX);
        var label = DATA.ladder[idx];
        if (which === "a" && label !== state.rungA) setAbRung("a", label);
        if (which === "b" && label !== state.rungB) setAbRung("b", label);
      }
      function onUp() {
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
        document.removeEventListener("touchmove", onMove);
        document.removeEventListener("touchend", onUp);
      }
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
      document.addEventListener("touchmove", onMove, { passive: false });
      document.addEventListener("touchend", onUp);
    }

    var handleA = document.getElementById("ab-handle-a");
    var handleB = document.getElementById("ab-handle-b");
    handleA.addEventListener("mousedown", function (e) { startDrag("a", e); });
    handleA.addEventListener("touchstart", function (e) { startDrag("a", e); }, { passive: false });
    handleB.addEventListener("mousedown", function (e) { startDrag("b", e); });
    handleB.addEventListener("touchstart", function (e) { startDrag("b", e); }, { passive: false });

    handleA.addEventListener("keydown", function (e) { handleAbKey("a", e); });
    handleB.addEventListener("keydown", function (e) { handleAbKey("b", e); });

    track.addEventListener("click", function (e) {
      if (e.target !== track) return; // clicks on ticks/handles handled separately
      var idx = nearestIndex(e.clientX);
      var label = DATA.ladder[idx];
      // move whichever handle is currently closer to the click
      var idxA = DATA.ladder.indexOf(state.rungA);
      var idxB = DATA.ladder.indexOf(state.rungB);
      var which = Math.abs(idx - idxA) <= Math.abs(idx - idxB) ? "a" : "b";
      setAbRung(which, label);
    });

    Array.prototype.forEach.call(document.querySelectorAll(".ab-tick"), function (tick) {
      tick.addEventListener("click", function () {
        var idx = parseInt(tick.getAttribute("data-index"), 10);
        var label = DATA.ladder[idx];
        var idxA = DATA.ladder.indexOf(state.rungA);
        var idxB = DATA.ladder.indexOf(state.rungB);
        var which = Math.abs(idx - idxA) <= Math.abs(idx - idxB) ? "a" : "b";
        setAbRung(which, label);
      });
    });

    var selA = document.getElementById("ab-select-a");
    var selB = document.getElementById("ab-select-b");
    selA.addEventListener("change", function () { setAbRung("a", selA.value); });
    selB.addEventListener("change", function () { setAbRung("b", selB.value); });

    positionAbHandles();
  }

  function handleAbKey(which, e) {
    var n = DATA.ladder.length;
    var current = which === "a" ? state.rungA : state.rungB;
    var idx = DATA.ladder.indexOf(current);
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
      idx = Math.max(0, idx - 1);
    } else if (e.key === "ArrowRight" || e.key === "ArrowUp") {
      idx = Math.min(n - 1, idx + 1);
    } else {
      return;
    }
    e.preventDefault();
    setAbRung(which, DATA.ladder[idx]);
  }

  // ---------------------------------------------------------------------
  // top-level render
  // ---------------------------------------------------------------------

  function render() {
    var item = DATA.itemsById[state.itemId];
    if (!item) return;

    viewTabs.forEach(function (btn) {
      btn.classList.toggle("is-active", btn.getAttribute("data-view") === state.view);
    });

    var html = renderPicker() + renderPromptBlock(item);
    if (state.view === "ab") {
      html += renderAbView(item);
    } else {
      html += renderLadderView(item);
    }
    app.innerHTML = html;

    attachPickerHandlers();
    if (state.view === "ab") attachAbHandlers();
  }

  function attachPickerHandlers() {
    Array.prototype.forEach.call(document.querySelectorAll(".picker-chip"), function (chip) {
      chip.addEventListener("click", function () {
        var itemId = chip.getAttribute("data-item-id");
        navigate(state.view, itemId, state.rungA, state.rungB);
      });
    });
    var randomBtn = document.getElementById("random-btn");
    if (randomBtn) {
      randomBtn.addEventListener("click", function () {
        var pool = DATA.items;
        var pick = pool[Math.floor(Math.random() * pool.length)];
        navigate(state.view, pick.id, state.rungA, state.rungB);
      });
    }
  }

  viewTabs.forEach(function (btn) {
    btn.addEventListener("click", function () {
      var view = btn.getAttribute("data-view");
      if (view === state.view) return;
      navigate(view, state.itemId, state.rungA, state.rungB);
    });
  });

  copyLinkBtn.addEventListener("click", function () {
    copyToClipboard(window.location.href);
  });

  window.addEventListener("hashchange", onRouteChange);

  // ---------------------------------------------------------------------
  // boot
  // ---------------------------------------------------------------------

  fetch(FIXTURES_URL)
    .then(function (resp) {
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      return resp.json();
    })
    .then(function (fixture) {
      var itemsById = {};
      var bySuite = {};
      var suiteOrder = fixture.suites.slice().sort();
      suiteOrder.forEach(function (s) { bySuite[s] = []; });
      fixture.items.forEach(function (it) {
        itemsById[it.id] = it;
        if (!bySuite[it.suite]) bySuite[it.suite] = [];
        bySuite[it.suite].push(it);
      });
      DATA = {
        ladder: fixture.ladder,
        rungs: fixture.rungs,
        items: fixture.items,
        itemsById: itemsById,
        bySuite: bySuite,
        suiteOrder: suiteOrder,
        inHouseTag: fixture.in_house_tag,
        disclaimer: fixture.disclaimer,
      };
      onRouteChange();
    })
    .catch(function (err) {
      app.innerHTML =
        '<p class="error">Could not load fixtures (' + escapeHtml(err.message) +
        "). Serve this directory over http(s) — e.g. <code>python3 -m http.server</code> " +
        "from <code>site/</code> — rather than opening index.html directly via file://.</p>";
    });
})();
