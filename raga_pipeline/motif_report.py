"""Standalone HTML report generator for motif mining index."""

import csv
import json
import os
from html import escape
from typing import Any, Dict, List, Optional


def _load_index(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_summary_csv(path: Optional[str]) -> List[Dict[str, str]]:
    if not path or not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _get_report_css() -> str:
    return """
:root {
  --bg-page: #e6d3b3;
  --bg-header: #dcb98a;
  --bg-panel: #f3e2c6;
  --bg-panel-soft: #ead2ad;
  --bg-input: #f8ecd9;
  --bg-accent-soft: #e9b878;
  --ink: #1a120a;
  --ink-muted: #5f4b35;
  --line: #c3a273;
  --line-strong: #8c5a2b;
  --accent: #c26a2b;
  --accent-strong: #8f4316;
}

@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: 'Space Grotesk', 'Segoe UI', system-ui, sans-serif;
  background: var(--bg-page);
  color: var(--ink);
  line-height: 1.5;
  padding: 0;
}

header {
  background: var(--bg-header);
  border-bottom: 2px solid var(--line-strong);
  padding: 1.5rem 2rem;
}

header h1 {
  font-size: 1.6rem;
  font-weight: 700;
  color: var(--accent-strong);
}

header .subtitle {
  color: var(--ink-muted);
  font-size: 0.85rem;
  margin-top: 0.25rem;
}

.container { max-width: 1200px; margin: 0 auto; padding: 1.5rem 2rem; }

/* Summary cards */
.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 0.75rem;
  margin-bottom: 1.5rem;
}

.stat-card {
  background: var(--bg-panel);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 0.75rem 1rem;
  text-align: center;
}

.stat-card .stat-value {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--accent-strong);
}

.stat-card .stat-label {
  font-size: 0.75rem;
  color: var(--ink-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

/* Params */
.params-bar {
  background: var(--bg-panel-soft);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 0.5rem 1rem;
  font-size: 0.8rem;
  color: var(--ink-muted);
  margin-bottom: 1.5rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem 1.5rem;
}

.params-bar span { white-space: nowrap; }
.params-bar strong { color: var(--ink); }

/* Tabs */
.tab-nav {
  display: flex;
  gap: 0;
  margin-bottom: 0.75rem;
  border-bottom: 2px solid var(--line);
}

.tab-btn {
  background: none;
  border: none;
  font-family: inherit;
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--ink-muted);
  padding: 0.6rem 1.25rem;
  cursor: pointer;
  border-bottom: 3px solid transparent;
  margin-bottom: -2px;
  transition: color 0.15s, border-color 0.15s;
}

.tab-btn:hover { color: var(--ink); }

.tab-btn.active {
  color: var(--accent-strong);
  border-bottom-color: var(--accent);
}

.tab-panel { display: none; }
.tab-panel.active { display: block; }

/* Search */
.search-bar {
  margin-bottom: 1rem;
}

.search-bar input {
  width: 100%;
  max-width: 400px;
  padding: 0.5rem 0.75rem;
  font-family: inherit;
  font-size: 0.9rem;
  background: var(--bg-input);
  border: 1px solid var(--line);
  border-radius: 4px;
  color: var(--ink);
  outline: none;
}

.search-bar input:focus { border-color: var(--accent); }

/* Accordions */
details {
  background: var(--bg-panel);
  border: 1px solid var(--line);
  border-radius: 6px;
  margin-bottom: 0.5rem;
  overflow: hidden;
}

details[open] { border-color: var(--line-strong); }

summary {
  padding: 0.6rem 1rem;
  cursor: pointer;
  font-weight: 600;
  font-size: 0.95rem;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  user-select: none;
  list-style: none;
}

summary::-webkit-details-marker { display: none; }

summary::before {
  content: '\\25B6';
  font-size: 0.65rem;
  color: var(--ink-muted);
  transition: transform 0.15s;
}

details[open] > summary::before { transform: rotate(90deg); }

summary .raga-name { color: var(--accent-strong); }

summary .badge {
  font-size: 0.7rem;
  font-weight: 500;
  background: var(--bg-panel-soft);
  border: 1px solid var(--line);
  border-radius: 3px;
  padding: 0.1rem 0.4rem;
  color: var(--ink-muted);
}

.details-body { padding: 0 1rem 0.75rem; }

/* Tables */
.motif-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82rem;
}

.motif-table th {
  background: var(--bg-panel-soft);
  border-bottom: 2px solid var(--line-strong);
  padding: 0.4rem 0.5rem;
  text-align: left;
  font-weight: 600;
  color: var(--ink-muted);
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
  position: sticky;
  top: 0;
}

.motif-table th:hover { color: var(--accent); }

.motif-table th .sort-arrow {
  font-size: 0.6rem;
  margin-left: 0.25rem;
  opacity: 0.4;
}

.motif-table th.sorted .sort-arrow { opacity: 1; color: var(--accent); }

.motif-table td {
  padding: 0.35rem 0.5rem;
  border-bottom: 1px solid var(--line);
  vertical-align: top;
}

.motif-table tr:hover td { background: var(--bg-panel-soft); }

.motif-table .motif-str {
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 0.78rem;
  word-break: break-all;
}

.motif-table .num { text-align: right; font-variant-numeric: tabular-nums; }

/* Show more button */
.show-more-btn {
  display: inline-block;
  margin: 0.5rem 0;
  padding: 0.35rem 0.75rem;
  font-family: inherit;
  font-size: 0.8rem;
  font-weight: 600;
  background: var(--bg-accent-soft);
  color: var(--accent-strong);
  border: 1px solid var(--line-strong);
  border-radius: 4px;
  cursor: pointer;
  transition: background 0.15s;
}

.show-more-btn:hover { background: var(--accent); color: #fff; }

/* Recording table */
.rec-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82rem;
}

.rec-table th {
  background: var(--bg-panel-soft);
  border-bottom: 2px solid var(--line-strong);
  padding: 0.4rem 0.5rem;
  text-align: left;
  font-weight: 600;
  color: var(--ink-muted);
}

.rec-table td {
  padding: 0.35rem 0.5rem;
  border-bottom: 1px solid var(--line);
}

.rec-table tr:hover td { background: var(--bg-panel-soft); }

.status-ok { color: #2f7a32; font-weight: 600; }
.status-warn { color: var(--accent); font-weight: 600; }
.status-err { color: #a6401b; font-weight: 600; }

/* No data message */
.no-data {
  text-align: center;
  color: var(--ink-muted);
  padding: 2rem;
  font-style: italic;
}

/* Legend / Info panel */
.info-toggle {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.35rem 0.75rem;
  font-family: inherit;
  font-size: 0.82rem;
  font-weight: 600;
  background: var(--bg-panel);
  color: var(--accent-strong);
  border: 1px solid var(--line-strong);
  border-radius: 4px;
  cursor: pointer;
  margin-bottom: 1rem;
}

.info-toggle:hover { background: var(--bg-accent-soft); }

.info-toggle .icon { font-size: 1rem; }

.legend-panel {
  background: var(--bg-panel);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 1rem 1.25rem;
  margin-bottom: 1.25rem;
  font-size: 0.82rem;
  line-height: 1.6;
}

.legend-panel h3 {
  font-size: 0.9rem;
  color: var(--accent-strong);
  margin-bottom: 0.5rem;
}

.legend-panel dl {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 0.2rem 1rem;
}

.legend-panel dt {
  font-weight: 600;
  color: var(--ink);
  white-space: nowrap;
}

.legend-panel dd {
  color: var(--ink-muted);
  margin: 0;
}

.legend-section { margin-bottom: 0.75rem; }
.legend-section:last-child { margin-bottom: 0; }

.sargam-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 0.15rem 1.5rem;
}

.sargam-grid .sargam-item {
  display: flex;
  gap: 0.5rem;
}

.sargam-grid .sargam-token {
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-weight: 700;
  min-width: 1.5rem;
}

/* Footer */
footer {
  border-top: 1px solid var(--line);
  padding: 1rem 2rem;
  text-align: center;
  font-size: 0.75rem;
  color: var(--ink-muted);
}

/* Responsive */
@media (max-width: 640px) {
  .container { padding: 1rem; }
  header { padding: 1rem; }
  .stat-grid { grid-template-columns: repeat(2, 1fr); }
}
"""


def _get_report_js() -> str:
    return """
// Tab switching
document.querySelectorAll('.tab-btn').forEach(function(btn) {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.tab-btn').forEach(function(b) { b.classList.remove('active'); });
    document.querySelectorAll('.tab-panel').forEach(function(p) { p.classList.remove('active'); });
    btn.classList.add('active');
    document.getElementById(btn.dataset.target).classList.add('active');
    // re-apply search filter for new tab
    filterItems();
  });
});

// Search/filter
var searchInput = document.getElementById('search-input');
searchInput.addEventListener('input', filterItems);

function filterItems() {
  var query = searchInput.value.toLowerCase().trim();
  var activePanel = document.querySelector('.tab-panel.active');
  if (!activePanel) return;
  activePanel.querySelectorAll('details').forEach(function(d) {
    var name = (d.dataset.name || '').toLowerCase();
    d.style.display = (!query || name.indexOf(query) !== -1) ? '' : 'none';
  });
}

// Column sorting — loads all motifs first so sort covers the full dataset
function sortTable(tableEl, colIdx, dataType) {
  // Expand all motifs before sorting
  var detailsEl = tableEl.closest('.details-body');
  if (detailsEl) {
    var btn = detailsEl.querySelector('.show-more-btn');
    if (btn && btn.style.display !== 'none') {
      btn.click();
    }
  }
  var tbody = tableEl.querySelector('tbody');
  var rows = Array.from(tbody.querySelectorAll('tr'));
  var th = tableEl.querySelectorAll('th')[colIdx];
  // Toggle direction
  var asc = th.dataset.sortDir !== 'asc';
  // Reset all th in this table
  tableEl.querySelectorAll('th').forEach(function(h) {
    h.classList.remove('sorted');
    h.dataset.sortDir = '';
  });
  th.dataset.sortDir = asc ? 'asc' : 'desc';
  th.classList.add('sorted');
  // Update arrow
  tableEl.querySelectorAll('th .sort-arrow').forEach(function(arrow, i) {
    arrow.textContent = i === colIdx ? (asc ? '\\u25B2' : '\\u25BC') : '\\u25B2';
  });
  rows.sort(function(a, b) {
    var va = a.cells[colIdx].getAttribute('data-val') || a.cells[colIdx].textContent;
    var vb = b.cells[colIdx].getAttribute('data-val') || b.cells[colIdx].textContent;
    if (dataType === 'num') { va = parseFloat(va) || 0; vb = parseFloat(vb) || 0; }
    if (va < vb) return asc ? -1 : 1;
    if (va > vb) return asc ? 1 : -1;
    return 0;
  });
  rows.forEach(function(r) { tbody.appendChild(r); });
}

// Lazy motif expansion
function showAllMotifs(ragaKey, btnEl) {
  var data = MOTIF_DATA[ragaKey];
  if (!data) return;
  var tableEl = btnEl.parentElement.querySelector('table');
  if (!tableEl) return;
  var tbody = tableEl.querySelector('tbody');
  var existing = tbody.querySelectorAll('tr').length;
  for (var i = existing; i < data.length; i++) {
    var m = data[i];
    var tr = document.createElement('tr');
    tr.innerHTML =
      '<td class="motif-str">' + escHtml(m.motif_str) + '</td>' +
      '<td class="num">' + m.length + '</td>' +
      '<td class="num" data-val="' + m.weight + '">' + m.weight.toFixed(4) + '</td>' +
      '<td class="num" data-val="' + m.specificity + '">' + m.specificity.toFixed(4) + '</td>' +
      '<td class="num" data-val="' + m.entropy + '">' + m.entropy.toFixed(4) + '</td>' +
      '<td class="num" data-val="' + m.coverage + '">' + m.coverage.toFixed(4) + '</td>' +
      '<td class="num">' + m.recording_support + '</td>' +
      '<td class="num">' + m.total_occurrences + '</td>';
    tbody.appendChild(tr);
  }
  btnEl.style.display = 'none';
}

function escHtml(s) {
  var d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// Legend toggle
function toggleLegend() {
  var panel = document.getElementById('legend-panel');
  panel.style.display = panel.style.display === 'none' ? '' : 'none';
}
"""


_LEGEND_HTML = """
<button class="info-toggle" onclick="toggleLegend()">
  <span class="icon">&#9432;</span> Legend &amp; Notation Guide
</button>
<div id="legend-panel" class="legend-panel" style="display:none">

<div class="legend-section">
<h3>Sargam Notation</h3>
<p style="color:var(--ink-muted);margin-bottom:0.4rem">
  Uppercase = shuddha (natural) or tivra; lowercase = komal (flat).
  Motif tokens are written as <code>sargam:pitch_class</code> (e.g., <code>R:2</code> = shuddha Re, pitch class 2).
</p>
<div class="sargam-grid">
  <div class="sargam-item"><span class="sargam-token">S</span> Sa (tonic)</div>
  <div class="sargam-item"><span class="sargam-token">r</span> Komal Re</div>
  <div class="sargam-item"><span class="sargam-token">R</span> Shuddha Re</div>
  <div class="sargam-item"><span class="sargam-token">g</span> Komal Ga</div>
  <div class="sargam-item"><span class="sargam-token">G</span> Shuddha Ga</div>
  <div class="sargam-item"><span class="sargam-token">m</span> Shuddha Ma</div>
  <div class="sargam-item"><span class="sargam-token">M</span> Tivra Ma</div>
  <div class="sargam-item"><span class="sargam-token">P</span> Pa</div>
  <div class="sargam-item"><span class="sargam-token">d</span> Komal Dha</div>
  <div class="sargam-item"><span class="sargam-token">D</span> Shuddha Dha</div>
  <div class="sargam-item"><span class="sargam-token">n</span> Komal Ni</div>
  <div class="sargam-item"><span class="sargam-token">N</span> Shuddha Ni</div>
</div>
</div>

<div class="legend-section">
<h3>Motif Statistics</h3>
<dl>
  <dt>Weight</dt>
  <dd>Primary ranking signal: <code>specificity / (1 + entropy)</code>. Higher = more distinctive to this raga.</dd>
  <dt>Specificity</dt>
  <dd>Fraction of this motif's appearances that belong to this raga: <code>raga_support / global_support</code>. 1.0 = unique to this raga.</dd>
  <dt>Entropy</dt>
  <dd>Shannon entropy of the motif's distribution across ragas. Lower = more concentrated in fewer ragas.</dd>
  <dt>Coverage</dt>
  <dd>Fraction of this raga's recordings that contain this motif: <code>recording_support / recording_count</code>.</dd>
  <dt>Rec. Support</dt>
  <dd>Number of recordings (within this raga) where this motif appears at least once.</dd>
  <dt>Occurrences</dt>
  <dd>Total number of times this motif appears across all recordings of this raga.</dd>
</dl>
</div>

</div>
"""


def _fmt(val: float, decimals: int = 4) -> str:
    return f"{val:.{decimals}f}"


def _motif_row_html(m: Dict[str, Any]) -> str:
    ms = escape(m.get("motif_str", ""))
    ln = m.get("length", 0)
    w = m.get("weight", 0.0)
    sp = m.get("specificity", 0.0)
    en = m.get("entropy", 0.0)
    co = m.get("coverage", 0.0)
    rs = m.get("recording_support", 0)
    to = m.get("total_occurrences", 0)
    return (
        f'<tr>'
        f'<td class="motif-str">{ms}</td>'
        f'<td class="num">{ln}</td>'
        f'<td class="num" data-val="{w}">{_fmt(w)}</td>'
        f'<td class="num" data-val="{sp}">{_fmt(sp)}</td>'
        f'<td class="num" data-val="{en}">{_fmt(en)}</td>'
        f'<td class="num" data-val="{co}">{_fmt(co)}</td>'
        f'<td class="num">{rs}</td>'
        f'<td class="num">{to}</td>'
        f'</tr>'
    )


_MOTIF_TABLE_HEADER = """<table class="motif-table" id="tbl-{table_id}">
<thead><tr>
  <th onclick="sortTable(this.closest('table'),0,'str')">Motif <span class="sort-arrow">&#9650;</span></th>
  <th onclick="sortTable(this.closest('table'),1,'num')">Len <span class="sort-arrow">&#9650;</span></th>
  <th onclick="sortTable(this.closest('table'),2,'num')">Weight <span class="sort-arrow">&#9650;</span></th>
  <th onclick="sortTable(this.closest('table'),3,'num')">Specificity <span class="sort-arrow">&#9650;</span></th>
  <th onclick="sortTable(this.closest('table'),4,'num')">Entropy <span class="sort-arrow">&#9650;</span></th>
  <th onclick="sortTable(this.closest('table'),5,'num')">Coverage <span class="sort-arrow">&#9650;</span></th>
  <th onclick="sortTable(this.closest('table'),6,'num')">Rec. Support <span class="sort-arrow">&#9650;</span></th>
  <th onclick="sortTable(this.closest('table'),7,'num')">Occurrences <span class="sort-arrow">&#9650;</span></th>
</tr></thead>
<tbody>
"""


def _build_summary_header(index: Dict[str, Any], summary_rows: List) -> str:
    totals = index.get("totals", {})
    params = index.get("params", {})
    created = index.get("created_at", "unknown")

    cards = [
        ("Ragas", totals.get("kept_ragas", 0)),
        ("Motifs", f'{totals.get("kept_motifs", 0):,}'),
        ("Recordings", totals.get("valid_recordings", 0)),
        ("GT Rows", totals.get("csv_rows", 0)),
        ("Missing", totals.get("missing_transcriptions", 0)),
    ]

    html = '<div class="stat-grid">\n'
    for label, val in cards:
        html += (
            f'<div class="stat-card">'
            f'<div class="stat-value">{val}</div>'
            f'<div class="stat-label">{label}</div>'
            f'</div>\n'
        )
    html += '</div>\n'

    html += '<div class="params-bar">\n'
    param_items = [
        ("N-gram range", f'{params.get("min_len", "?")} - {params.get("max_len", "?")}'),
        ("Min support", params.get("min_recording_support", "?")),
        ("Source", params.get("transcription_source", "?")),
        ("Generated", created[:19].replace("T", " ")),
    ]
    for label, val in param_items:
        html += f'<span><strong>{label}:</strong> {escape(str(val))}</span>\n'
    html += '</div>\n'

    return html


def _build_raga_tab(index: Dict[str, Any], top_n: int) -> str:
    ragas = index.get("ragas", {})
    html_parts = []

    for raga_name in sorted(ragas.keys()):
        raga_data = ragas[raga_name]
        rec_count = raga_data.get("recording_count", 0)
        motif_count = raga_data.get("kept_motif_count", 0)
        motifs = raga_data.get("motifs", [])
        safe_key = escape(raga_name)
        js_key = raga_name.replace("'", "\\'").replace("\\", "\\\\")
        table_id = raga_name.replace(" ", "_").replace("'", "")

        html_parts.append(
            f'<details data-name="{safe_key}">\n'
            f'<summary>'
            f'<span class="raga-name">{safe_key}</span>'
            f'<span class="badge">{rec_count} recordings</span>'
            f'<span class="badge">{motif_count:,} motifs</span>'
            f'</summary>\n'
            f'<div class="details-body">\n'
        )

        html_parts.append(_MOTIF_TABLE_HEADER.format(table_id=escape(table_id)))
        for m in motifs[:top_n]:
            html_parts.append(_motif_row_html(m))
        html_parts.append('</tbody></table>\n')

        if len(motifs) > top_n:
            remaining = len(motifs) - top_n
            html_parts.append(
                f'<button class="show-more-btn" '
                f"onclick=\"showAllMotifs('{js_key}', this)\">"
                f'Show all {len(motifs):,} motifs ({remaining:,} more)'
                f'</button>\n'
            )

        html_parts.append('</div>\n</details>\n')

    if not ragas:
        html_parts.append('<div class="no-data">No raga data found in the index.</div>')

    return "".join(html_parts)


def _build_recording_tab(summary_rows: List[Dict[str, str]]) -> str:
    if not summary_rows:
        return '<div class="no-data">No summary CSV provided. Re-run mining with --summary-out to enable the recording view.</div>'

    # Group by raga
    by_raga: Dict[str, List[Dict[str, str]]] = {}
    for row in summary_rows:
        raga = row.get("raga", "Unknown")
        by_raga.setdefault(raga, []).append(row)

    html_parts = []
    for raga_name in sorted(by_raga.keys()):
        rows = by_raga[raga_name]
        safe_name = escape(raga_name)
        html_parts.append(
            f'<details data-name="{safe_name}">\n'
            f'<summary>'
            f'<span class="raga-name">{safe_name}</span>'
            f'<span class="badge">{len(rows)} recordings</span>'
            f'</summary>\n'
            f'<div class="details-body">\n'
            f'<table class="rec-table"><thead><tr>'
            f'<th>Filename</th><th>Tonic</th><th>Gender</th>'
            f'<th>Instrument</th><th>Tokens</th><th>Source</th><th>Status</th>'
            f'</tr></thead><tbody>\n'
        )
        for r in rows:
            status = r.get("status", "")
            status_cls = "status-ok" if status == "ok" else ("status-warn" if status == "warn" else "status-err")
            html_parts.append(
                f'<tr>'
                f'<td>{escape(r.get("filename", ""))}</td>'
                f'<td>{escape(r.get("tonic", ""))}</td>'
                f'<td>{escape(r.get("gender", ""))}</td>'
                f'<td>{escape(r.get("instrument", ""))}</td>'
                f'<td class="num">{escape(r.get("token_count", ""))}</td>'
                f'<td>{escape(r.get("transcription_source", ""))}</td>'
                f'<td class="{status_cls}">{escape(status)}</td>'
                f'</tr>\n'
            )
        html_parts.append('</tbody></table>\n</div>\n</details>\n')

    return "".join(html_parts)


def _build_motif_data_json(index: Dict[str, Any]) -> str:
    """Build the MOTIF_DATA JS object with full motif arrays keyed by raga."""
    ragas = index.get("ragas", {})
    data: Dict[str, List[Dict[str, Any]]] = {}
    for raga_name, raga_data in ragas.items():
        motifs = raga_data.get("motifs", [])
        data[raga_name] = [
            {
                "motif_str": m.get("motif_str", ""),
                "length": m.get("length", 0),
                "weight": m.get("weight", 0.0),
                "specificity": m.get("specificity", 0.0),
                "entropy": m.get("entropy", 0.0),
                "coverage": m.get("coverage", 0.0),
                "recording_support": m.get("recording_support", 0),
                "total_occurrences": m.get("total_occurrences", 0),
            }
            for m in motifs
        ]
    return json.dumps(data, ensure_ascii=False)


def generate_motif_report(
    index_path: str,
    summary_csv_path: Optional[str] = None,
    output_html_path: str = "motif_report.html",
    top_n_per_raga: int = 50,
) -> str:
    """Generate a standalone HTML report from a motif index and optional summary CSV.

    Args:
        index_path: Path to motif_index.json.
        summary_csv_path: Optional path to summary CSV from --summary-out.
        output_html_path: Where to write the HTML report.
        top_n_per_raga: Motifs shown per raga before "show more".

    Returns:
        Absolute path to the generated HTML file.
    """
    index = _load_index(index_path)
    summary_rows = _load_summary_csv(summary_csv_path)

    summary_header = _build_summary_header(index, summary_rows)
    raga_tab = _build_raga_tab(index, top_n_per_raga)
    recording_tab = _build_recording_tab(summary_rows)
    motif_data_json = _build_motif_data_json(index)
    css = _get_report_css()
    js = _get_report_js()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Motif Mining Report</title>
<style>{css}</style>
</head>
<body>

<header>
  <h1>Motif Mining Report</h1>
  <p class="subtitle">Corpus-level motif index overview</p>
</header>

<div class="container">

{summary_header}

{_LEGEND_HTML}

<nav class="tab-nav">
  <button class="tab-btn active" data-target="by-raga">By Raga</button>
  <button class="tab-btn" data-target="by-recording">By Recording</button>
</nav>

<div class="search-bar">
  <input type="text" id="search-input" placeholder="Filter ragas or recordings...">
</div>

<div id="by-raga" class="tab-panel active">
{raga_tab}
</div>

<div id="by-recording" class="tab-panel">
{recording_tab}
</div>

</div>

<footer>Raga Detection Pipeline -- Motif Mining Report</footer>

<script>
var MOTIF_DATA = {motif_data_json};
{js}
</script>
</body>
</html>"""

    out_path = os.path.abspath(output_html_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    return out_path
