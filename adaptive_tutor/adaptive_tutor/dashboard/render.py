"""Server-rendered HTML for the teacher-facing dashboard (Phase 13)."""

from __future__ import annotations

import html
from collections.abc import Mapping
from typing import Any

_PAGE_STYLE = """
:root {
  --bg: #f6f8fb;
  --card: #ffffff;
  --text: #1a2332;
  --muted: #5c6b7f;
  --accent: #2563eb;
  --accent-soft: #dbeafe;
  --border: #e2e8f0;
  --good: #059669;
  --warn: #d97706;
}
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
  margin: 0;
  background: var(--bg);
  color: var(--text);
  line-height: 1.55;
}
header {
  background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
  color: #fff;
  padding: 1.75rem 2rem;
}
header h1 { margin: 0 0 0.35rem 0; font-size: 1.5rem; font-weight: 650; }
header p { margin: 0; opacity: 0.92; font-size: 0.95rem; }
nav {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  padding: 0.75rem 2rem;
  background: var(--card);
  border-bottom: 1px solid var(--border);
}
nav a {
  color: var(--accent);
  text-decoration: none;
  font-weight: 600;
  font-size: 0.9rem;
}
nav a:hover { text-decoration: underline; }
main { max-width: 56rem; margin: 0 auto; padding: 1.5rem 2rem 3rem; }
.alert {
  background: #fffbeb;
  border: 1px solid #fcd34d;
  border-radius: 10px;
  padding: 1rem 1.15rem;
  color: #78350f;
  margin-bottom: 1.25rem;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(11rem, 1fr));
  gap: 0.9rem;
  margin: 1rem 0 1.5rem;
}
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem 1.1rem;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.card .label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); margin: 0 0 0.25rem; }
.card .value { font-size: 1.35rem; font-weight: 700; margin: 0; color: var(--text); }
h2 { font-size: 1.15rem; margin: 2rem 0 0.75rem; border-bottom: 2px solid var(--accent-soft); padding-bottom: 0.35rem; }
h3 { font-size: 1rem; margin: 1.5rem 0 0.5rem; color: var(--muted); }
table {
  width: 100%;
  border-collapse: collapse;
  background: var(--card);
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid var(--border);
  font-size: 0.88rem;
}
th, td { text-align: left; padding: 0.55rem 0.75rem; border-bottom: 1px solid var(--border); }
th { background: var(--accent-soft); color: #1e3a8a; font-weight: 650; }
tr:last-child td { border-bottom: none; }
.teach-box {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.15rem 1.35rem;
  margin: 1rem 0;
}
.teach-box ul { margin: 0.5rem 0 0 1.1rem; padding: 0; }
.teach-box li { margin: 0.35rem 0; }
.badge { display: inline-block; background: var(--accent-soft); color: #1e40af; padding: 0.15rem 0.5rem; border-radius: 6px; font-size: 0.8rem; font-weight: 600; }
footer { font-size: 0.8rem; color: var(--muted); margin-top: 2.5rem; text-align: center; }
code { background: #eef2ff; padding: 0.12rem 0.35rem; border-radius: 4px; font-size: 0.86em; }
"""


def _esc(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return "yes" if x else "no"
    return html.escape(str(x))


def _fmt_float(x: Any, *, nd: int = 4) -> str:
    try:
        v = float(x)
        if nd == 0:
            return str(int(round(v)))
        s = f"{v:.{nd}f}"
        return s.rstrip("0").rstrip(".") if "." in s else s
    except (TypeError, ValueError):
        return _esc(x)


def _table_from_rows(headers: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def _teaching_model_section() -> str:
    return """
<section id="teaching-model" class="teach-box">
  <h2 style="margin-top:0;border:none;">Why use DQN and PPO together?</h2>
  <p><span class="badge">Two time scales</span> Your system splits the problem the way teachers already do:</p>
  <ul>
    <li><strong>Curriculum (macro)</strong> &mdash; <em>what</em> to teach next and <em>how long</em> to stay on a concept.
      <strong>PPO</strong> learns this from <strong>concept-level</strong> outcomes (mastery trend, engagement, frustration,
      number of steps). It answers: "Should we stay on fractions or move on?"</li>
    <li><strong>Tutoring moves (micro)</strong> &mdash; <em>how</em> to respond step by step (hint, easier problem, encouragement, ...).
      <strong>DQN</strong> learns this from <strong>each simulator step</strong> with reward from your <strong>RewardEngine</strong>,
      after <strong>Phase&nbsp;7</strong> safety filters. It answers: "Given this learner state right now, which tutor action helps?"</li>
  </ul>
  <p><strong>Why not only one?</strong> A single policy would have to pick both the concept <em>and</em> the fine-grained action in one shot.
    That mixes two very different decisions: long-term pacing (episodes across concepts) versus immediate pedagogy (next utterance/action).
    Training them separately keeps each part <strong>interpretable</strong> and stable for a thesis: PPO = long horizon planning;
    DQN = short horizon control inside the safe action set.</p>
  <p><strong>How they connect in your code</strong> PPO chooses a concept segment (and pacing knobs); inside that segment,
    DQN proposes ASSISTments actions; Phase&nbsp;7 may revise the action; the environment updates the learner; rewards train DQN;
    aggregated segment outcomes train PPO.</p>
</section>
"""


def render_dashboard_page(experiment: dict[str, Any] | None) -> str:
    body_experiment = ""
    if experiment is None:
        body_experiment = """
<div class="alert">
  <strong>No experiment loaded yet.</strong> Run a batch in another terminal (same Python environment), then refresh this page.
  <div style="margin-top:0.6rem"><code>python3 -c "from adaptive_tutor.experiments import run_experiment; run_experiment({'seed':0,'episodes':5,'policy':'heuristic','max_episode_steps':24})"</code></div>
</div>
"""
    else:
        cm = experiment.get("comparison_ready_metrics") or {}
        dm = experiment.get("dataset_metrics") or {}
        summaries = experiment.get("episode_summaries") or []

        cards = f"""
<div class="grid">
  <div class="card"><p class="label">Policy</p><p class="value">{_esc(experiment.get("policy"))}</p></div>
  <div class="card"><p class="label">Avg reward</p><p class="value">{_fmt_float(experiment.get("avg_reward"))}</p></div>
  <div class="card"><p class="label">Learning rate</p><p class="value">{_fmt_float(experiment.get("avg_learning_rate"))}</p></div>
  <div class="card"><p class="label">Frustration rate</p><p class="value">{_fmt_float(experiment.get("frustration_rate"))}</p></div>
  <div class="card"><p class="label">Mastery gain</p><p class="value">{_fmt_float(cm.get("mastery_gain"))}</p></div>
  <div class="card"><p class="label">Success rate</p><p class="value">{_fmt_float(cm.get("success_rate"))}</p></div>
</div>
"""
        ep_rows: list[list[str]] = []
        for s in summaries:
            if not isinstance(s, dict):
                continue
            ep_rows.append(
                [
                    _esc(s.get("episode")),
                    _fmt_float(s.get("total_reward")),
                    _esc(s.get("length")),
                    _fmt_float(s.get("mastery_gain")),
                    _fmt_float(s.get("learning_rate")),
                    _fmt_float(s.get("frustration_rate")),
                ]
            )
        ep_table = _table_from_rows(
            ["Episode", "Total reward", "Steps", "Mastery gain", "Learning rate", "Frustration"],
            ep_rows,
        )

        ds_rows: list[list[str]] = []
        per = dm.get("per_skill")
        if isinstance(per, Mapping):
            for skill in sorted(per.keys()):
                row = per.get(skill)
                if not isinstance(row, Mapping):
                    continue
                ds_rows.append(
                    [
                        _esc(skill),
                        _fmt_float(row.get("mean_correctness")),
                        _fmt_float(row.get("mean_hint_count")),
                        _fmt_float(row.get("n_records"), nd=1),
                    ]
                )
        ds_table = _table_from_rows(
            ["Skill (dataset)", "Avg correctness", "Avg hints", "Rows"],
            ds_rows,
        )

        cfg = experiment.get("config_resolved") or {}
        cfg_bits = ""
        if isinstance(cfg, Mapping):
            cfg_bits = (
                f"<p style='color:var(--muted);font-size:0.88rem;margin:0 0 1rem'>"
                f"Run id: <strong>{_esc(experiment.get('experiment_id'))}</strong> &middot; "
                f"Episodes: {_esc(cfg.get('episodes'))} &middot; "
                f"Max steps: {_esc(cfg.get('max_episode_steps'))}"
                f"</p>"
            )

        body_experiment = f"""
{cfg_bits}
{cards}
<h2>Episode overview</h2>
{ep_table}
<h2>Historical cohort (ASSISTments snapshot)</h2>
<p style="color:var(--muted);font-size:0.9rem;margin:0 0 0.5rem">
  Baseline correctness (all skills): <strong>{_fmt_float(dm.get("baseline_correctness"))}</strong> &middot;
  Rows: <strong>{_fmt_float(dm.get("n_rows"), nd=0)}</strong>
</p>
{ds_table}
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Adaptive Tutor &mdash; Dashboard</title>
  <style>{_PAGE_STYLE}</style>
</head>
<body>
  <header>
    <h1>Adaptive Tutor</h1>
    <p>Read-only overview for instructors and thesis demos &mdash; simulator runs and dataset context.</p>
  </header>
  <nav>
    <a href="#overview">Overview</a>
    <a href="#teaching-model">DQN + PPO</a>
    <a href="#api">JSON API</a>
    <a href="/explain/view">Explain an action</a>
  </nav>
  <main>
    <section id="overview">
      <h2>Latest experiment</h2>
      {body_experiment}
    </section>
    {_teaching_model_section()}
    <section id="api">
      <h2>JSON API (for tools / scripts)</h2>
      <p style="color:var(--muted);font-size:0.9rem">
        Same data as machine-readable endpoints:
        <code>/experiment/latest</code>, <code>/metrics/simulator</code>, <code>/metrics/dataset</code>, <code>/explain/action</code>.
      </p>
    </section>
    <footer>Adaptive Tutor &mdash; bachelor thesis scope. No login; no writes from this UI.</footer>
  </main>
</body>
</html>
"""


def render_explain_form(
    *,
    action: str,
    mastery: float,
    engaged: float,
    confused: float,
    frustrated: float,
    explanation: dict[str, Any] | None,
    error: str | None,
) -> str:
    err = f'<div class="alert">{_esc(error)}</div>' if error else ""
    result = ""
    if explanation is not None:
        drivers = explanation.get("top_drivers") or []
        drv_html = "<ul>"
        if isinstance(drivers, list):
            for d in drivers[:8]:
                if isinstance(d, dict):
                    drv_html += f"<li><strong>{_esc(d.get('feature'))}</strong>, weight {_fmt_float(d.get('weight'))}</li>"
        drv_html += "</ul>"
        p7 = explanation.get("phase7_checks") or {}
        p7_html = ""
        if isinstance(p7, dict):
            p7_html = "<ul>" + "".join(
                f"<li>{_esc(k)}: {_esc(v)}</li>" for k, v in p7.items()
            ) + "</ul>"
        text = _esc(explanation.get("explanation_text", ""))
        result = f"""
<h3>Explanation</h3>
<p>{text}</p>
<h3>Top drivers (rule-based)</h3>
{drv_html}
<h3>Phase 7 checks (demo defaults)</h3>
{p7_html}
<pre style="background:#f1f5f9;padding:1rem;border-radius:8px;overflow:auto;font-size:0.82rem">{_esc(str(explanation))}</pre>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Explain action</title>
  <style>{_PAGE_STYLE}</style>
</head>
<body>
  <header><h1>Explain a tutoring action</h1><p>Rule-based Phase 10 explainer (not neural attribution).</p></header>
  <nav><a href="/">&larr; Dashboard</a></nav>
  <main>
    {err}
    <form method="get" action="/explain/view" class="teach-box" style="margin-top:1rem">
      <h2 style="margin-top:0;border:none;">Learner snapshot</h2>
      <p style="display:grid;grid-template-columns:repeat(auto-fill,minmax(10rem,1fr));gap:0.75rem">
        <label>Action<br/><input name="action" value="{_esc(action)}" style="width:100%"/></label>
        <label>Mastery 0&ndash;1<br/><input name="mastery" type="number" step="0.05" min="0" max="1" value="{mastery}" style="width:100%"/></label>
        <label>Engaged<br/><input name="engaged" type="number" step="0.05" min="0" max="1" value="{engaged}" style="width:100%"/></label>
        <label>Confused<br/><input name="confused" type="number" step="0.05" min="0" max="1" value="{confused}" style="width:100%"/></label>
        <label>Frustrated<br/><input name="frustrated" type="number" step="0.05" min="0" max="1" value="{frustrated}" style="width:100%"/></label>
      </p>
      <p><button type="submit" style="background:var(--accent);color:#fff;border:none;padding:0.5rem 1.2rem;border-radius:8px;font-weight:600;cursor:pointer">Generate explanation</button></p>
    </form>
    {result}
  </main>
</body>
</html>
"""
