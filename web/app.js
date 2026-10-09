/* Modullandschaft – Clusterung live im Browser aus vorab berechneter Zusammenlegungsfolge.
   Bei eigener Gewichtung werden Gesamtwert und Zusammenlegungsfolge im Browser neu berechnet. */
(async function () {
  "use strict";

  const data = await (await fetch("data/data.json")).json();
  const N = data.meta.n;
  const M = data.modules;
  let SIM = data.sim;     // Gesamtwert zur aktuellen Gewichtung
  const DIMS = data.dims;
  let Q = data.meta.quantiles;
  const SG = data.meta.studiengaenge;   // Kürzel → Name des Studiengangs

  const LEVELS = [
    { min: 99, label: "sehr hoch", var: "--lvl-4" },
    { min: 96, label: "hoch", var: "--lvl-3" },
    { min: 90, label: "mittel", var: "--lvl-2" },
    { min: -1, label: "gering", var: "--lvl-1" },
  ];

  // Dimensionen und Standardgewichte (in %); merges/orders/sim/quantiles in data.json gelten für diese
  const DIM_KEYS = ["inhalte", "kompetenzen", "literatur"];
  const DIM_LABEL = { inhalte: "Inhalte", kompetenzen: "Kompetenzen", literatur: "Literatur" };
  const DEFAULT_W = Object.fromEntries(DIM_KEYS.map((d) => [d, Math.round(data.meta.weights[d] * 100)]));

  // w: Reglerwerte 0–100 je Dimension, wirksam ist ihr Anteil an der Summe
  const state = { cap: "5", k: 90, view: "map", sel: null, hover: null, fixed: false, w: { ...DEFAULT_W } };
  let clusters = [];      // aktive Cluster
  let clusterOf = [];     // Modulindex → Cluster
  let transform = d3.zoomIdentity;

  const $ = (id) => document.getElementById(id);
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const fmt = new Intl.NumberFormat("de-DE", { maximumFractionDigits: 1 });
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // ---------------------------------------------------------------- Einordnung

  // Q: Quantile aller Paarwerte in gleichmäßigen Schritten (z. B. 1001 Werte = 0,1-%-Stufen)
  function percentile(z) {
    const last = Q.length - 1, step = 100 / last;
    // Bei Gleichstand (z. B. viele Paare ohne gemeinsame Literatur) zählt die Mitte des Bereichs
    const a = d3.bisectLeft(Q, z), b = d3.bisectRight(Q, z);
    if (b - a > 1) return ((a + b - 1) / 2) * step;
    if (z <= Q[0]) return 0;
    if (z >= Q[last]) return 100;
    let lo = 0, hi = last;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (Q[mid] <= z) lo = mid; else hi = mid; }
    return (lo + (z - Q[lo]) / (Q[hi] - Q[lo] || 1)) * step;
  }
  const levelOf = (pct) => LEVELS.find((l) => pct >= l.min);
  function topShare(pct) {
    const share = 100 - pct;
    if (share < 0.1) return "obere 0,1 %";
    return `obere ${fmt.format(share < 1 ? Math.round(share * 10) / 10 : Math.round(share))} %`;
  }
  const dimWord = (z) => (z == null ? "–" : z >= 2 ? "sehr ähnlich" : z >= 1 ? "ähnlich" : z >= 0 ? "etwas ähnlich" : "kaum ähnlich");
  const clip = (s, n = 200) => (s.length > n ? s.slice(0, s.lastIndexOf(" ", n - 2) > n * 0.6 ? s.lastIndexOf(" ", n - 2) : n - 2) + " …" : s);
  const pairKey = (a, b) => (a < b ? `${a}-${b}` : `${b}-${a}`);
  const sgName = (i) => SG[M[i].sg] || M[i].sg;
  const sgTag = (i) => `<span class="sg-tag" title="${esc(sgName(i))}">${esc(M[i].sg)}</span>`;
  // Studiengänge eines Clusters mit Anzahl, häufigster zuerst: „ACT (2), FIN“
  function sgMix(mem) {
    const n = d3.rollups(mem, (v) => v.length, (i) => M[i].sg).sort((a, b) => b[1] - a[1]);
    return n.map(([sg, k]) => (k > 1 ? `${sg} (${k})` : sg)).join(", ");
  }

  // ---------------------------------------------------------------- Gewichtung

  const share = (w, d) => w[d] / DIM_KEYS.reduce((s, k) => s + w[k], 0);
  const isDefaultW = (w) => DIM_KEYS.every((d) => Math.abs(share(w, d) - share(DEFAULT_W, d)) < 1e-9);
  const weightsText = (w) => DIM_KEYS.map((d) => `${DIM_LABEL[d]} ${Math.round(share(w, d) * 100)} %`).join(", ");

  // Gesamtwert, Quantile, Zusammenlegungsfolge und Matrixreihenfolge zur aktuellen Gewichtung
  let struct;
  function buildStructure() {
    if (isDefaultW(state.w)) {
      // Vorab berechnet – so bleiben Links und Exporte mit Standardgewichtung exakt reproduzierbar
      struct = { sim: data.sim, q: data.meta.quantiles, merges: (cap) => data.merges[cap], order: (cap) => data.orders[cap] };
    } else {
      const sim = combineDims(state.w);
      const cache = new Map();
      let base;
      const get = (cap) => {
        if (!cache.has(cap)) {
          const merges = cappedAverageLinkage(sim, +cap);
          base ??= leafOrder(cap === "0" ? merges : cappedAverageLinkage(sim, 0), d3.range(N));
          cache.set(cap, { merges, order: leafOrder(merges, base) });
        }
        return cache.get(cap);
      };
      struct = { sim, q: quantiles(sim), merges: (cap) => get(cap).merges, order: (cap) => get(cap).order };
    }
    SIM = struct.sim;
    Q = struct.q;
  }

  // Gewichtetes Mittel der Dimensionswerte wie similarity.combine: fehlt einem Modul die Literatur, zählen nur
  // die übrigen Dimensionen. Bleibt keine übrig (nur Literatur gewichtet), gilt 0 – wie „nichts geteilt“.
  function combineDims(w) {
    const out = Array.from({ length: N }, () => new Array(N).fill(null));
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
      let num = 0, den = 0;
      for (const d of DIM_KEYS) {
        const v = DIMS[d][i][j];
        if (v != null && w[d] > 0) { num += w[d] * v; den += w[d]; }
      }
      out[i][j] = out[j][i] = den > 0 ? num / den : 0;
    }
    return out;
  }

  function quantiles(sim) {
    const v = [];
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) v.push(sim[i][j]);
    v.sort((a, b) => a - b);
    return d3.range(1001).map((k) => d3.quantileSorted(v, k / 1000));
  }

  // Wie export_web.capped_average_linkage: Rückgabe [a, b, ähnlichkeit] mit Cluster-IDs im SciPy-Schema
  function cappedAverageLinkage(sim, cap) {
    const S = 2 * N;
    const total = new Float64Array(S * S);   // Summen der paarweisen Ähnlichkeiten zwischen Clustern
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) total[i * S + j] = i === j ? 0 : sim[i][j] ?? -Infinity;
    const sizes = new Int32Array(S).fill(1, 0, N);
    let active = d3.range(N), next = N;
    const merges = [];
    while (active.length > 1) {
      let best = -Infinity, ba = -1, bb = -1;
      for (let x = 0; x < active.length; x++) {
        const a = active[x];
        for (let y = x + 1; y < active.length; y++) {
          const b = active[y];
          if (cap && sizes[a] + sizes[b] > cap) continue;
          const avg = total[a * S + b] / (sizes[a] * sizes[b]);
          if (avg > best) { best = avg; ba = a; bb = b; }
        }
      }
      if (ba < 0) break;
      const c = next++;
      sizes[c] = sizes[ba] + sizes[bb];
      active = active.filter((x) => x !== ba && x !== bb);
      for (const x of active) total[x * S + c] = total[c * S + x] = total[ba * S + x] + total[bb * S + x];
      active.push(c);
      merges.push([ba, bb, Math.round(best * 1000) / 1000]);
    }
    return merges;
  }

  // Wie export_web.leaf_order: Blattreihenfolge des Merge-Walds, Wurzeln nach Position in `base`
  function leafOrder(merges, base) {
    const rank = new Map(base.map((leaf, r) => [leaf, r]));
    const merged = new Set(merges.flatMap(([a, b]) => [a, b]));
    const leaves = (c) => {
      if (c < N) return [c];
      const [a, b] = merges[c - N];
      const la = leaves(a), lb = leaves(b);
      return d3.min(la, (x) => rank.get(x)) <= d3.min(lb, (x) => rank.get(x)) ? la.concat(lb) : lb.concat(la);
    };
    const roots = d3.range(N + merges.length).filter((c) => !merged.has(c)).map(leaves);
    roots.sort((p, q) => rank.get(p[0]) - rank.get(q[0]));
    return roots.flat();
  }

  // ---------------------------------------------------------------- Clusterung

  function computeClusters() {
    const merges = struct.merges(state.cap);
    const steps = Math.min(N - state.k, merges.length);
    const members = new Map();
    for (let i = 0; i < N; i++) members.set(i, [i]);
    for (let t = 0; t < steps; t++) {
      const [a, b] = merges[t];
      members.set(N + t, members.get(a).concat(members.get(b)));
      members.delete(a); members.delete(b);
    }
    clusters = [];
    clusterOf = new Array(N);
    for (const [id, mem] of members) {
      const c = { id, members: mem, size: mem.length };
      if (c.size > 1) {
        let s = 0, cnt = 0;
        for (let i = 0; i < mem.length; i++) for (let j = i + 1; j < mem.length; j++) { s += SIM[mem[i]][mem[j]]; cnt++; }
        c.meanZ = s / cnt;
        c.pct = percentile(c.meanZ);
        c.level = levelOf(c.pct);
        c.terms = sharedTerms(mem);
        c.label = c.terms.slice(0, 3).map((t) => t[0]).join(" · ") || M[mem[0]].name;
        c.edges = mst(mem);
      } else {
        c.label = M[mem[0]].name;
        c.terms = [];
        c.edges = [];
      }
      clusters.push(c);
      for (const i of mem) clusterOf[i] = c;
    }
    clusters.sort((a, b) => (b.size > 1) - (a.size > 1) || (b.meanZ ?? 0) - (a.meanZ ?? 0));
    // Fortlaufende Nummer als Bezug für Arbeitsgruppen („Cluster 7“), gleich in Website und Excel-Export
    clusters.forEach((c, idx) => { c.nr = idx + 1; });
  }

  function sharedTerms(mem) {
    const score = new Map(), count = new Map(), shown = new Map();
    for (const i of mem) for (const [t, w, disp] of M[i].terms) {
      score.set(t, (score.get(t) || 0) + w);
      count.set(t, (count.get(t) || 0) + 1);
      if (!shown.has(t)) shown.set(t, disp);
    }
    const cand = [...score].filter(([t]) => count.get(t) >= 2).sort((a, b) => b[1] - a[1]);
    const out = []; // [Term, Anzeige, Anzahl]
    for (const [t] of cand) {
      // Enthaltene Begriffe und Beugungsformen („Geschäftsmodelle“/„Geschäftsmodellen“) überspringen
      if (out.some(([o]) => o.includes(t) || (o.length > 6 && t.length > 6 && o.slice(0, -2) === t.slice(0, o.length - 2) && Math.abs(o.length - t.length) <= 2))) continue;
      // Festes Wortpaar ersetzt ein schon gewähltes Einzelwort („Supply“ → „Supply Chain“)
      const parts = out.filter(([o]) => t.split(" ").includes(o));
      if (parts.length) {
        const k = out.indexOf(parts[0]);
        out[k] = [t, shown.get(t), count.get(t)];
        for (const p of parts.slice(1)) out.splice(out.indexOf(p), 1);
        continue;
      }
      out.push([t, shown.get(t), count.get(t)]);
      if (out.length === 8) break;
    }
    return out.map(([, disp, n]) => [disp, n]);
  }

  function mst(mem) {
    // Maximal aufspannender Baum über die Ähnlichkeit: zeigt die stärksten Verbindungen im Cluster
    const inTree = new Set([mem[0]]), edges = [];
    while (inTree.size < mem.length) {
      let best = -Infinity, pair = null;
      for (const a of inTree) for (const b of mem) {
        if (inTree.has(b)) continue;
        if (SIM[a][b] > best) { best = SIM[a][b]; pair = [a, b]; }
      }
      inTree.add(pair[1]);
      edges.push(pair);
    }
    return edges;
  }

  // ---------------------------------------------------------------- Karte

  const svg = d3.select("#map");
  const gEdges = svg.append("g");
  const gPts = svg.append("g");
  const gLabels = svg.append("g");
  let xS, yS;

  // Kräftemodell: Ausgangspunkt ist die UMAP-Karte; Module eines Clusters ziehen sich an,
  // damit sich Cluster beim Verschieben des Reglers sichtbar formieren.
  const SPACE = 1000;
  const nodes = M.map((m, i) => ({ i, bx: m.x * SPACE, by: (1 - m.y) * SPACE, x: m.x * SPACE, y: (1 - m.y) * SPACE }));
  const sim = d3.forceSimulation(nodes)
    .force("x", d3.forceX((d) => d.bx).strength(0.05))
    .force("y", d3.forceY((d) => d.by).strength(0.05))
    .force("collide", d3.forceCollide(11))
    .force("link", d3.forceLink([]).distance(22).strength(0.8))
    .alphaDecay(0.035)
    .stop();
  sim.on("tick", () => positionMap());
  let firstLayout = true;

  function relayout() {
    const links = clusters.flatMap((c) => c.edges.map(([a, b]) => ({ source: a, target: b })));
    sim.force("link").links(links);
    if (firstLayout) {
      // Erster Aufbau ohne Animation
      sim.alpha(1);
      for (let t = 0; t < 300; t++) sim.tick();
      firstLayout = false;
      positionMap();
    } else {
      sim.alpha(0.7).restart();
    }
  }

  function sizeMap() {
    const w = svg.node().clientWidth, h = svg.node().clientHeight, pad = 40;
    const ext = (k) => d3.extent(nodes, (d) => d[k]);
    const [x0, x1] = ext("bx"), [y0, y1] = ext("by");
    xS = d3.scaleLinear().domain([x0 - 30, x1 + 30]).range([pad, w - pad]);
    yS = d3.scaleLinear().domain([y0 - 30, y1 + 30]).range([pad, h - pad]);
  }
  const px = (i) => transform.applyX(xS(nodes[i].x));
  const py = (i) => transform.applyY(yS(nodes[i].y));

  const zoom = d3.zoom().scaleExtent([0.8, 10]).on("zoom", (ev) => { transform = ev.transform; positionMap(); });
  svg.call(zoom).on("dblclick.zoom", null);
  svg.on("click", (ev) => { if (ev.target === svg.node()) select(null); });

  function colorOf(c) { return c.size > 1 ? css(c.level.var) : css("--single"); }

  function drawMap() {
    const pts = gPts.selectAll("circle").data(M, (d) => d.id);
    pts.join(
      (enter) => enter.append("circle")
        .attr("r", 5)
        .attr("tabindex", -1)
        .on("mouseenter", (ev, d) => { const i = M.indexOf(d); state.hover = clusterOf[i]; highlight(); showTip(ev, tipModule(i)); })
        .on("mousemove", (ev) => moveTip(ev))
        .on("mouseleave", () => { state.hover = null; highlight(); hideTip(); })
        .on("click", (ev, d) => {
          ev.stopPropagation();
          const i = M.indexOf(d), c = clusterOf[i];
          select(c.size > 1 && !(state.sel?.type === "cluster" && state.sel.c === c) ? { type: "cluster", c } : { type: "module", i });
        })
    )
      .attr("class", (d, i) => "pt" + (clusterOf[i].size > 1 ? "" : " single"))
      .transition().duration(300)
      .attr("r", (d, i) => (clusterOf[i].size > 1 ? 6 : 4.5))
      .attr("fill", (d, i) => colorOf(clusterOf[i]));

    const edges = clusters.flatMap((c) => c.edges.map(([a, b]) => ({ a, b, c, key: pairKey(a, b) })));
    gEdges.selectAll("line").data(edges, (d) => d.key).join(
      (enter) => enter.append("line").attr("class", "edge").attr("opacity", 0)
        .call((s) => s.transition().duration(300).attr("opacity", 1)),
      (update) => update,
      (exit) => exit.transition().duration(200).attr("opacity", 0).remove()
    ).attr("stroke", (d) => colorOf(d.c));

    relayout();
    positionMap();
    highlight();
  }

  function positionMap() {
    gPts.selectAll("circle").attr("cx", (d, i) => px(i)).attr("cy", (d, i) => py(i));
    gEdges.selectAll("line").attr("x1", (d) => px(d.a)).attr("y1", (d) => py(d.a)).attr("x2", (d) => px(d.b)).attr("y2", (d) => py(d.b));
    placeLabels();
  }

  function placeLabels() {
    // Beschriftungen senkrecht auseinanderschieben, wenn sie sich überdecken würden
    const labels = gLabels.selectAll("text").nodes().map((el) => {
      const i = d3.select(el).datum();
      return { el, x: px(i) + 9, y: py(i) + 4, w: Math.min(260, el.getComputedTextLength?.() || 160) };
    }).sort((a, b) => a.y - b.y);
    for (let pass = 0; pass < 4; pass++) {
      for (let a = 0; a < labels.length; a++) for (let b = a + 1; b < labels.length; b++) {
        const A = labels[a], B = labels[b];
        const overlapX = A.x < B.x + B.w && B.x < A.x + A.w;
        if (overlapX && B.y - A.y < 15) B.y = A.y + 15;
      }
    }
    for (const l of labels) d3.select(l.el).attr("x", l.x).attr("y", l.y);
  }

  function focusSet() {
    const c = state.hover || (state.sel?.type === "cluster" ? state.sel.c : state.sel?.type === "module" ? clusterOf[state.sel.i] : null);
    return c ? new Set(c.members) : null;
  }

  function highlight() {
    const f = focusSet();
    gPts.selectAll("circle")
      .classed("dim", (d, i) => f && !f.has(i))
      .classed("selected", (d, i) => state.sel?.type === "module" && state.sel.i === i);
    gEdges.selectAll("line").classed("dim", (d) => f && !f.has(d.a));
    const labelled = f ? [...f] : [];
    gLabels.selectAll("text").data(labelled, (d) => d).join("text")
      .attr("class", "pt-label")
      .text((d) => (M[d].name.length > 38 ? M[d].name.slice(0, 36) + " …" : M[d].name));
    positionMap();
    if (state.view === "matrix") drawMatrix();
  }

  // ---------------------------------------------------------------- Matrix

  const canvas = $("matrix");
  const ctx = canvas.getContext("2d");
  let mGeom = null;

  function drawMatrix() {
    const order = struct.order(state.cap);
    const wrap = canvas.parentElement;
    const size = Math.floor(Math.min(wrap.clientWidth, wrap.clientHeight || wrap.clientWidth));
    const dpr = window.devicePixelRatio || 1;
    canvas.style.width = canvas.style.height = size + "px";
    canvas.width = canvas.height = Math.round(size * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const cell = size / N;
    mGeom = { order, cell, size };
    const lo = d3.color(css("--surface")), hi = d3.color(css("--heat-hi"));
    const ramp = d3.interpolateRgb(lo, hi);
    ctx.fillStyle = css("--surface");
    ctx.fillRect(0, 0, size, size);
    const f = focusSet();
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const a = order[r], b = order[c];
        if (a === b) { ctx.fillStyle = css("--surface-2"); }
        else {
          // Unterdurchschnittliche Paare bleiben hell, damit hohe Ähnlichkeit heraussticht
          const t = Math.max(0, Math.min(1, (SIM[a][b] - 0.5) / 2.5));
          ctx.fillStyle = ramp(Math.pow(t, 0.9));
        }
        ctx.fillRect(c * cell, r * cell, Math.ceil(cell), Math.ceil(cell));
      }
    }
    // Clusterblöcke auf der Diagonale (in der Reihenfolge zusammenhängend)
    const pos = new Map(order.map((m, r) => [m, r]));
    for (const cl of clusters) {
      if (cl.size < 2) continue;
      const rows = cl.members.map((m) => pos.get(m));
      const r0 = Math.min(...rows), r1 = Math.max(...rows) + 1;
      const active = f && cl.members.every((m) => f.has(m));
      ctx.strokeStyle = active ? css("--accent") : css("--ink-2");
      ctx.lineWidth = active ? 2.5 : 1;
      ctx.strokeRect(r0 * cell + 0.5, r0 * cell + 0.5, (r1 - r0) * cell - 1, (r1 - r0) * cell - 1);
    }
  }

  function matrixHit(ev) {
    if (!mGeom) return null;
    const rect = canvas.getBoundingClientRect();
    const c = Math.floor((ev.clientX - rect.left) / mGeom.cell), r = Math.floor((ev.clientY - rect.top) / mGeom.cell);
    if (r < 0 || c < 0 || r >= N || c >= N) return null;
    return [mGeom.order[r], mGeom.order[c]];
  }
  canvas.addEventListener("mousemove", (ev) => {
    const hit = matrixHit(ev);
    if (!hit) return hideTip();
    const [a, b] = hit;
    if (a === b) return showTip(ev, tipModule(a));
    const pct = percentile(SIM[a][b]);
    showTip(ev, `<strong>${esc(M[a].name)}</strong> ${sgTag(a)}<br><strong>${esc(M[b].name)}</strong> ${sgTag(b)}
      <div class="tt-sub">Ähnlichkeit: ${topShare(pct)} aller Paare${clusterOf[a] === clusterOf[b] ? " · gleiches Cluster" : ""}</div>`);
  });
  canvas.addEventListener("mouseleave", hideTip);
  canvas.addEventListener("click", (ev) => {
    const hit = matrixHit(ev);
    if (!hit) return;
    const [a, b] = hit;
    const c = clusterOf[a];
    select(c === clusterOf[b] && c.size > 1 ? { type: "cluster", c } : { type: "module", i: a });
  });

  // ---------------------------------------------------------------- Tooltip

  const tip = $("tooltip");
  function tipModule(i) {
    const c = clusterOf[i];
    const sub = c.size > 1 ? `Cluster ${c.nr} „${esc(c.label)}“ · ${c.size} Module · Ähnlichkeit ${c.level.label}` : "Einzelmodul – kein Cluster";
    return `<strong>${esc(M[i].name)}</strong><div class="tt-sub">${M[i].id} · ${esc(sgName(i))}<br>${sub}</div>`;
  }
  function showTip(ev, html) { tip.innerHTML = html; tip.hidden = false; moveTip(ev); }
  function moveTip(ev) {
    const r = tip.getBoundingClientRect();
    let x = ev.clientX + 14, y = ev.clientY + 14;
    if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - 14;
    if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - 14;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  function hideTip() { tip.hidden = true; }

  // ---------------------------------------------------------------- Seitenleiste

  const detail = $("detail");
  const dot = (c) => `<span class="swatch" style="background:${colorOf(c)}"></span>`;

  function renderDetail() {
    const s = state.sel;
    if (!s) return renderOverview();
    if (s.type === "cluster") return renderCluster(s.c);
    return renderModule(s.i);
  }

  const OVERVIEW_LIMIT = 15;
  let showAll = false;

  function renderOverview() {
    const multi = clusters.filter((c) => c.size > 1);
    const singles = clusters.length - multi.length;
    const shown = showAll ? multi : multi.slice(0, OVERVIEW_LIMIT);
    detail.innerHTML = `
      <div class="panel-head"><h2>${multi.length} Cluster mit mehreren Modulen</h2>
        <p>Sortiert nach Ähnlichkeit. ${singles} Module stehen für sich.</p></div>
      <div class="panel-body">${shown.map((c) => `
        <button class="cluster-card" data-cluster="${c.id}">
          <div class="cc-top"><span class="cc-nr">${c.nr}</span>${dot(c)}<span class="cc-title">${esc(c.label)}</span><span class="cc-count">${c.size} Module</span></div>
          <div class="cc-members" style="margin-left:calc(2.2em + 28px)">${c.members.map((m) => `${esc(M[m].name)} <span class="sg-inline">${esc(M[m].sg)}</span>`).join(" · ")}</div>
        </button>`).join("") || `<p class="muted">Bei dieser Einstellung bleiben alle Module einzeln.</p>`}
        ${multi.length > shown.length ? `<button class="back" id="show-all" style="margin:8px 12px">Alle ${multi.length} Cluster anzeigen</button>` : ""}
      </div>`;
    $("show-all")?.addEventListener("click", () => { showAll = true; renderOverview(); });
    detail.querySelectorAll("[data-cluster]").forEach((el) => {
      const c = clusters.find((x) => x.id === +el.dataset.cluster);
      el.addEventListener("click", () => select({ type: "cluster", c }));
      el.addEventListener("mouseenter", () => { state.hover = c; highlight(); });
      el.addEventListener("mouseleave", () => { state.hover = null; highlight(); });
    });
  }

  // Literatur, die mindestens zwei Module eines Clusters teilen
  function sharedLiterature(mem) {
    const lit = new Map();
    for (const i of mem) for (const [key, label] of new Map(M[i].lit)) {
      if (!lit.has(key)) lit.set(key, { label, n: 0 });
      lit.get(key).n++;
    }
    return [...lit.values()].filter((x) => x.n >= 2).sort((a, b) => b.n - a.n);
  }

  function renderCluster(c) {
    const mem = c.members;
    const pairs = [];
    for (let i = 0; i < mem.length; i++) for (let j = i + 1; j < mem.length; j++) pairs.push([mem[i], mem[j]]);
    pairs.sort((p, q) => SIM[q[0]][q[1]] - SIM[p[0]][p[1]]);

    // Stärkste Stichpunkt-Übereinstimmungen über alle Paare
    const matches = [];
    for (const [a, b] of pairs) {
      const ev = data.evidence[pairKey(a, b)];
      if (!ev) continue;
      const flip = a > b;
      for (const m of ev.m.slice(0, 2)) matches.push({ s: m.s, a: flip ? b : a, b: flip ? a : b, ta: m.a, tb: m.b });
    }
    matches.sort((p, q) => q.s - p.s);

    const sharedLit = sharedLiterature(mem);

    detail.innerHTML = `
      <div class="panel-head">
        <button class="back" id="back">← Alle Cluster</button>
        <h2><span class="nr-badge">Cluster ${c.nr}</span>${esc(c.label)}</h2>
        <p><span class="level-pill">${dot(c)}Ähnlichkeit ${c.level.label}</span>
        &nbsp;${c.size} Module · ähnlicher als ${fmt.format(Math.min(99.9, c.pct))} % aller Modulpaare</p>
        <p>Studiengänge: ${esc(sgMix(mem))}</p>
      </div>
      <div class="panel-body">
        ${c.terms.length ? `<div class="section"><h3>Gemeinsame Themen</h3><div class="chips">${c.terms.map(([t, n]) => `<span class="chip">${esc(t)}</span>`).join("")}</div></div>` : ""}
        <div class="section"><h3>Module</h3>
          ${mem.map((i) => `<div class="member"><span class="member-id">${M[i].id}</span>
            <button class="linkish" data-module="${i}">${esc(M[i].name)}</button>${sgTag(i)}</div>`).join("")}
        </div>
        ${matches.length ? `<div class="section"><h3>Was sie verbindet</h3>
          ${matches.slice(0, 5).map((m) => `<div class="match"><div class="match-pair">
            <div><div class="match-src">${esc(M[m.a].name)} · ${esc(M[m.a].sg)}</div>${esc(clip(m.ta))}</div>
            <div class="match-arrow">⟷</div>
            <div><div class="match-src">${esc(M[m.b].name)} · ${esc(M[m.b].sg)}</div>${esc(clip(m.tb))}</div></div></div>`).join("")}</div>` : ""}
        ${sharedLit.length ? `<div class="section"><h3>Gemeinsame Literatur</h3><ul class="lit-list">
          ${sharedLit.slice(0, 8).map((x) => `<li>${esc(x.label)} <span class="muted">(${x.n} Module)</span></li>`).join("")}</ul></div>` : ""}
        <div class="section"><h3>Paarweise Ähnlichkeit</h3>
          ${pairs.slice(0, 10).map(([a, b]) => simRow(a, b, `${esc(M[a].name)} ⟷ ${esc(M[b].name)}`)).join("")}
        </div>
      </div>`;
    $("back").addEventListener("click", () => select(null));
    bindModuleLinks();
  }

  function simRow(a, b, title, link) {
    const z = SIM[a][b], pct = percentile(z);
    const ev = data.evidence[pairKey(a, b)];
    const nLit = ev?.l ? ev.l.split("; ").length : 0;
    const litTxt = DIMS.literatur[a][b] == null ? "Literatur: keine Angaben" : nLit ? `${nLit} gemeinsame${nLit > 1 ? "" : "s"} Werk${nLit > 1 ? "e" : ""}` : "keine gemeinsame Literatur";
    return `<div class="simrow">
      <span>${link ? `<button class="linkish" data-module="${b}">${title}</button>` : title}</span>
      <span class="simval">${topShare(pct)}</span>
      <div class="simbar"><span style="width:${Math.max(2, pct)}%"></span></div>
      <div class="dims">Inhalte ${dimWord(DIMS.inhalte[a][b])} · Kompetenzen ${dimWord(DIMS.kompetenzen[a][b])} · ${litTxt}</div>
    </div>`;
  }

  function renderModule(i) {
    const m = M[i], c = clusterOf[i];
    const nearest = d3.range(N).filter((j) => j !== i).sort((a, b) => SIM[i][b] - SIM[i][a]).slice(0, 6);
    detail.innerHTML = `
      <div class="panel-head">
        <button class="back" id="back">← ${c.size > 1 ? "Zum Cluster" : "Alle Cluster"}</button>
        <h2>${esc(m.name)}</h2>
        <p class="sg-line">${sgTag(i)}${esc(sgName(i))}</p>
        <p>${m.id}${m.ects ? ` · ${m.ects} ECTS` : ""}${m.name_en && m.name_en !== m.name ? ` · <em>${esc(m.name_en)}</em>` : ""}</p>
      </div>
      <div class="panel-body">
        <div class="section"><h3>Cluster</h3>
          ${c.size > 1 ? `<button class="linkish" id="to-cluster"><span class="level-pill">${dot(c)}Cluster ${c.nr} · ${esc(c.label)} · ${c.size} Module</span></button>`
            : `<span class="muted">Bei ${state.k} Clustern steht dieses Modul für sich.</span>`}
        </div>
        <div class="section"><h3>Ähnlichste Module</h3>
          ${nearest.map((j) => simRow(i, j, `${esc(M[j].name)} <span class="sg-inline">${esc(M[j].sg)}</span>`, true)).join("")}
        </div>
        ${m.terms.length ? `<div class="section"><h3>Prägende Fachbegriffe</h3><div class="chips">${m.terms.slice(0, 12).map(([, , d]) => `<span class="chip">${esc(d)}</span>`).join("")}</div></div>` : ""}
        <div class="section"><h3>Inhalte</h3><ul class="points-list">${m.points.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>
          ${m.points_total > m.points.length ? `<p class="muted">… und ${m.points_total - m.points.length} weitere Stichpunkte</p>` : ""}</div>
      </div>`;
    $("back").addEventListener("click", () => select(c.size > 1 ? { type: "cluster", c } : null));
    $("to-cluster")?.addEventListener("click", () => select({ type: "cluster", c }));
    bindModuleLinks();
  }

  function bindModuleLinks() {
    detail.querySelectorAll("[data-module]").forEach((el) => el.addEventListener("click", () => select({ type: "module", i: +el.dataset.module })));
  }

  function select(sel) {
    state.sel = sel;
    state.hover = null;
    renderDetail();
    highlight();
    detail.parentElement.scrollTop = 0;
  }

  // ---------------------------------------------------------------- Kopfzahlen, Legende, Steuerung

  function renderStats() {
    const multi = clusters.filter((c) => c.size > 1);
    const inMulti = multi.reduce((s, c) => s + c.size, 0);
    const strong = multi.filter((c) => c.pct >= 96).length;
    $("stats").innerHTML = [
      [multi.length, "Cluster mit mehreren Modulen"],
      [inMulti, "Module in diesen Clustern"],
      [strong, "Cluster mit hoher oder sehr hoher Ähnlichkeit"],
    ].map(([v, l]) => `<div class="stat"><div class="stat-value">${v}</div><div class="stat-label">${l}</div></div>`).join("");
  }

  function renderLegend() {
    const el = $("legend");
    if (state.view === "map") {
      el.innerHTML = `<span class="legend-title">Ähnlichkeit im Cluster</span>` +
        [...LEVELS].reverse().map((l) => `<span class="legend-item"><span class="swatch" style="background:${css(l.var)}"></span>${l.label}</span>`).join("") +
        `<span class="legend-item"><span class="swatch" style="background:${css("--single")}"></span>Einzelmodul</span>`;
    } else {
      el.innerHTML = `<span class="legend-title">Ähnlichkeit des Paares</span>
        <span class="legend-item">gering <span style="display:inline-block;width:120px;height:10px;border-radius:5px;border:1px solid var(--border);background:linear-gradient(90deg,${css("--surface")},${css("--heat-hi")})"></span> hoch</span>
        <span class="legend-item"><span style="display:inline-block;width:14px;height:14px;border:1.5px solid ${css("--ink-2")};border-radius:2px"></span>Cluster</span>`;
    }
  }

  const slider = $("k-slider");
  function setupControls() {
    const caps = data.meta.caps;
    $("cap-group").innerHTML = caps.map((c) => `<button role="radio" data-cap="${c}" aria-checked="${String(c) === state.cap}">${c === 0 ? "∞" : c}</button>`).join("");
    $("cap-group").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      state.cap = b.dataset.cap;
      $("cap-group").querySelectorAll("button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
      updateSliderRange();
      update(true);
    }));
    slider.addEventListener("input", () => { state.k = +slider.value; update(false); });
    document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => setView(t.dataset.view)));
    $("module-list").innerHTML = M.map((m) => `<option value="${esc(m.name)} · ${m.id}" label="${esc(m.name)} · ${m.id} · ${esc(SG[m.sg] || m.sg)}">`).join("");
    $("w-group").innerHTML = DIM_KEYS.map((d) => `<label class="w-row"><span>${DIM_LABEL[d]}</span>
      <input type="range" min="0" max="100" step="5" data-dim="${d}" aria-label="Gewicht ${DIM_LABEL[d]}"><output id="w-${d}"></output></label>`).join("");
    $("w-group").querySelectorAll("input").forEach((el) => el.addEventListener("input", () => {
      state.w[el.dataset.dim] = +el.value;
      // Mindestens eine Dimension muss zählen
      if (DIM_KEYS.every((d) => state.w[d] === 0)) state.w[el.dataset.dim] = +el.step;
      applyWeights();
    }));
    $("w-reset").addEventListener("click", () => { state.w = { ...DEFAULT_W }; applyWeights(); });
    $("search").addEventListener("change", (ev) => {
      const v = ev.target.value.toLowerCase();
      const i = M.findIndex((m) => `${m.name} · ${m.id}`.toLowerCase() === v || m.id.toLowerCase() === v || m.name.toLowerCase() === v);
      const j = i >= 0 ? i : M.findIndex((m) => m.name.toLowerCase().includes(v) || m.id.toLowerCase().includes(v));
      if (j >= 0) { select({ type: "module", i: j }); ev.target.value = ""; ev.target.blur(); }
    });
  }

  function renderWeights() {
    for (const d of DIM_KEYS) {
      $("w-group").querySelector(`[data-dim="${d}"]`).value = state.w[d];
      $(`w-${d}`).textContent = `${Math.round(share(state.w, d) * 100)} %`;
    }
    const def = isDefaultW(state.w);
    $("w-reset").hidden = def;
    $("w-hint").textContent = def ? "Standardgewichtung" : "Eigene Gewichtung – Cluster neu berechnet.";
    $("w-fixed").textContent = weightsText(state.w);
    $("footer").textContent = `${N} Module ausgewertet · ${data.meta.excluded.length} Rahmenmodule ausgeblendet · Gewichtung: ${weightsText(state.w)}`;
  }

  function applyWeights() {
    buildStructure();
    renderWeights();
    updateSliderRange();
    update(false);
    if (state.view === "matrix") drawMatrix();
  }

  function updateSliderRange() {
    const minK = N - struct.merges(state.cap).length;
    slider.min = minK; slider.max = N;
    state.k = Math.max(minK, Math.min(N, state.k));
    slider.value = state.k;
    $("k-min").textContent = minK; $("k-max").textContent = N;
    $("cap-hint").textContent = state.cap === "0" ? "Ohne Grenze können sehr große Sammelcluster entstehen." : `Mit dieser Grenze sind mindestens ${minK} Cluster möglich.`;
  }

  function setView(v) {
    state.view = v;
    document.querySelectorAll(".tab").forEach((t) => t.setAttribute("aria-selected", String(t.dataset.view === v)));
    $("view-map").hidden = v !== "map";
    $("view-matrix").hidden = v !== "matrix";
    renderLegend();
    if (v === "matrix") drawMatrix();
  }

  function update(resetSel) {
    const prevSel = state.sel;
    computeClusters();
    $("k-value").textContent = state.k;
    $("k-of").textContent = `aus ${N} Modulen`;
    // Auswahl erhalten, wenn möglich
    if (resetSel || !prevSel) state.sel = null;
    else if (prevSel.type === "cluster") {
      const c = clusterOf[prevSel.c.members[0]];
      state.sel = c.size > 1 ? { type: "cluster", c } : null;
    }
    renderStats();
    renderDetail();
    drawMap();
    history.replaceState(null, "", "#" + hashFor(state.fixed));
  }

  function hashFor(fixed) {
    const w = isDefaultW(state.w) ? "" : `&gewichte=${DIM_KEYS.map((d) => state.w[d]).join("-")}`;
    return `k=${state.k}&max=${state.cap}${w}${fixed ? "&ansicht=fest" : ""}`;
  }

  function readHash() {
    const p = new URLSearchParams(location.hash.slice(1));
    state.fixed = p.get("ansicht") === "fest";
    if (p.has("max") && data.meta.caps.map(String).includes(p.get("max"))) state.cap = p.get("max");
    if (p.has("k")) state.k = +p.get("k") || state.k;
    // gewichte=Inhalte-Kompetenzen-Literatur, z. B. 60-20-20
    const w = (p.get("gewichte") || "").split("-").map(Number);
    if (w.length === 3 && w.every((v) => Number.isInteger(v) && v >= 0 && v <= 100) && w.some((v) => v > 0)) {
      DIM_KEYS.forEach((d, i) => { state.w[d] = w[i]; });
    }
  }

  // ---------------------------------------------------------------- Link teilen & Excel-Export

  function fixedLink() {
    return `${location.origin}${location.pathname}#${hashFor(true)}`;
  }

  function setupActions() {
    const share = $("share-btn");
    share.addEventListener("click", async () => {
      const url = fixedLink();
      const label = share.querySelector("span");
      try {
        await navigator.clipboard.writeText(url);
        label.textContent = "Link kopiert";
      } catch {
        prompt("Link zur festen Ansicht:", url);
      }
      setTimeout(() => { label.textContent = "Link zur festen Ansicht"; }, 2000);
    });
    $("export-btn").addEventListener("click", exportExcel);
  }

  let xlsxLoading;
  function loadXlsx() {
    // SheetJS (~0,9 MB) erst beim ersten Export laden
    xlsxLoading ??= new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = "vendor/xlsx.full.min.js";
      s.onload = () => resolve(window.XLSX);
      s.onerror = () => { xlsxLoading = null; reject(new Error("SheetJS konnte nicht geladen werden")); };
      document.head.appendChild(s);
    });
    return xlsxLoading;
  }

  function sheet(XLSX, rows, widths) {
    const ws = XLSX.utils.aoa_to_sheet(rows);
    ws["!cols"] = widths.map((wch) => ({ wch }));
    if (rows.length > 1) ws["!autofilter"] = { ref: XLSX.utils.encode_range({ s: { r: 0, c: 0 }, e: { r: rows.length - 1, c: rows[0].length - 1 } }) };
    return ws;
  }

  async function exportExcel() {
    const btn = $("export-btn");
    btn.disabled = true;
    try {
      const XLSX = await loadXlsx();
      const capTxt = state.cap === "0" ? "ohne Grenze" : state.cap;
      const levelTxt = (c) => (c.size > 1 ? c.level.label : "–");
      const nearestIn = (i, pool) => {
        const cand = pool.filter((j) => j !== i);
        if (!cand.length) return "";
        const j = cand.reduce((a, b) => (SIM[i][b] > SIM[i][a] ? b : a));
        return `${M[j].name} (${M[j].id}, ${M[j].sg}, ${topShare(percentile(SIM[i][j]))})`;
      };
      const all = d3.range(N);

      const multi = clusters.filter((c) => c.size > 1);
      const clusterRows = [["Cluster-Nr.", "Clustername", "Anzahl Module", "Ähnlichkeitsstufe", "Ähnlicher als … % aller Modulpaare",
        "Gemeinsame Themen", "Studiengänge", "Modulnummern", "Module", "Gemeinsame Literatur", "Bewertung", "Kommentar"]];
      for (const c of multi) {
        clusterRows.push([c.nr, c.label, c.size, levelTxt(c), Math.round(Math.min(99.9, c.pct) * 10) / 10,
          c.terms.map(([t]) => t).join(", "),
          sgMix(c.members),
          c.members.map((i) => M[i].id).join(", "),
          c.members.map((i) => `${M[i].name} (${M[i].sg})`).join("; "),
          sharedLiterature(c.members).map((x) => `${x.label} (${x.n})`).join("; "),
          "", ""]);
      }

      const moduleRows = [["Cluster-Nr.", "Clustername", "Module im Cluster", "Ähnlichkeitsstufe", "Modulnummer", "Modulname",
        "Modulname (englisch)", "Studiengang (Kürzel)", "Studiengang", "ECTS", "Ähnlichstes Modul im Cluster", "Ähnlichstes Modul insgesamt", "Bewertung", "Kommentar"]];
      for (const c of clusters) {
        for (const i of c.members) {
          moduleRows.push([c.nr, c.size > 1 ? c.label : "Einzelmodul", c.size, levelTxt(c), M[i].id, M[i].name, M[i].name_en || "",
            M[i].sg, sgName(i), M[i].ects ?? "", c.size > 1 ? nearestIn(i, c.members) : "", nearestIn(i, all), "", ""]);
        }
      }

      const today = new Date();
      const infoRows = [
        ["Modullandschaft – Export"],
        [],
        ["Stand", today.toLocaleDateString("de-DE")],
        ["Anzahl Cluster", state.k],
        ["Max. Module je Cluster", capTxt],
        ["Gewichtung", weightsText(state.w) + (isDefaultW(state.w) ? " (Standard)" : "")],
        ["Cluster mit mehreren Modulen", multi.length],
        ["Module ausgewertet", N],
        ["Link zu diesem Stand", fixedLink()],
        [],
        ["Hinweise"],
        ["Cluster-Nummern gelten nur für diese Einstellung. Bei anderer Clusteranzahl oder Größengrenze ändert sich die Clusterung."],
        ["Ähnlichkeitsstufe: mittlere Ähnlichkeit der Module eines Clusters, eingeordnet gegenüber allen Modulpaaren (sehr hoch = oberes 1 %, hoch = obere 4 %, mittel = obere 10 %)."],
        [`Standardgewichtung: ${weightsText(DEFAULT_W)}. Andere Gewichtungen ergeben eine andere Clusterung.`],
        [`Studiengänge: ${Object.entries(SG).map(([k, v]) => `${k} = ${v}`).join("; ")}`],
        [`Nicht enthalten (Rahmenmodule): ${data.meta.excluded.join(", ")}`],
        ["Die Spalten „Bewertung“ und „Kommentar“ sind frei auszufüllen."],
      ];

      const wb = XLSX.utils.book_new();
      XLSX.utils.book_append_sheet(wb, sheet(XLSX, clusterRows, [10, 40, 12, 16, 16, 40, 24, 30, 70, 50, 18, 40]), "Cluster");
      XLSX.utils.book_append_sheet(wb, sheet(XLSX, moduleRows, [10, 36, 12, 16, 12, 50, 44, 10, 36, 6, 50, 50, 18, 40]), "Module");
      const info = XLSX.utils.aoa_to_sheet(infoRows);
      info["!cols"] = [{ wch: 30 }, { wch: 90 }];
      XLSX.utils.book_append_sheet(wb, info, "Hinweise");

      const stamp = today.toISOString().slice(0, 10);
      const wTxt = isDefaultW(state.w) ? "" : `_Gewichte-${DIM_KEYS.map((d) => state.w[d]).join("-")}`;
      XLSX.writeFile(wb, `Modullandschaft_${state.k}-Cluster_max-${state.cap === "0" ? "ohne" : state.cap}${wTxt}_${stamp}.xlsx`);
    } catch (err) {
      alert("Export fehlgeschlagen: " + err.message);
    } finally {
      btn.disabled = false;
    }
  }

  // ---------------------------------------------------------------- Start

  readHash();
  document.body.classList.toggle("fixed", state.fixed);
  $("cap-value").textContent = state.cap === "0" ? "∞" : state.cap;
  setupControls();
  setupActions();
  buildStructure();
  renderWeights();
  updateSliderRange();
  sizeMap();
  update(true);
  renderLegend();

  let rT;
  addEventListener("resize", () => { clearTimeout(rT); rT = setTimeout(() => { sizeMap(); positionMap(); if (state.view === "matrix") drawMatrix(); }, 120); });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { drawMap(); renderLegend(); renderDetail(); if (state.view === "matrix") drawMatrix(); });
})();
