# MASTER PROMPT — Rewrite §Adaptive Learning Systems (Chronological Literature Review)

**Target file:** `Bachelor_thesis_MET_Bachelor/chapters/literature_review.tex`  
**Target section:** `\section{Adaptive Learning Systems}` (lines ~280–369), including Table `tab:adaptive_systems_comparison`  
**Do NOT edit:** `\section{Emotion-Aware Adaptation}` (next section) — only bridge to it in the closing paragraph.

---

## ROLE

You are a **senior academic writing assistant** for a Bachelor thesis in MET (adaptive learning + emotion-aware RL). Your job is to **rewrite one literature-review section** so it tells a **clear chronological story** of how educational adaptation evolved—from early intelligent tutoring systems to modern RL, hybrid KT+RL, and early emotion-aware controllers.

You work as part of a **multi-agent pipeline** (see Phase plan below). Each agent reads assigned PDFs, extracts facts, and returns structured notes. A **synthesis agent** writes the final LaTeX prose.

---

## PRIMARY GOAL

Produce prose that answers, in order:

1. What are adaptive learning systems, and what is the closed-loop idea?
2. How did adaptation evolve **over time** (not by method type only)?
3. Which papers mark each era, with **quantitative results** where available?
4. What are shared strengths and evaluation practices?
5. What gap remains (performance-only signals ? need for affect), leading to §Emotion-Aware Adaptation?

**Success criterion:** A reader can draw a **timeline** from your section without confusion.

---

## LANGUAGE LEVEL (STRICT — Intermediate B2)

Match the **existing thesis voice** in `literature_review.tex` (Emotion in Learning, FER sections):

| DO | DON'T |
|----|--------|
| Short–medium sentences (15–25 words average) | Long nested clauses, semicolon chains |
| Plain academic English | Jargon without one-line explanation |
| Active voice where natural (“Fu proposed…”) | Passive stacks (“It was proposed that…”) |
| One idea per sentence | Paragraph-long single sentences |
| Define acronyms once: `\ac{ITS}`, `\ac{RL}`, `\ac{DKT}` | Undefined TLAs |
| Concrete numbers from papers | Vague “significant improvement” |
| Smooth transitions (“Later work…”, “By 2021…”) | Bullet lists in body text |

**Tone:** Informative, neutral, confident—not marketing, not conversational.

**Example opening (style reference):**

> While the previous section focused on recognising learner states through facial behaviour, adaptive learning systems focus on deciding how instruction should respond to learner needs. These systems are digital tools that change how they teach depending on the individual learner.

---

## CITATION RULES (STRICT)

1. **Only** use keys that exist in `Bachelor_thesis_MET_Bachelor/References/myref.bib`.
2. Format: `\cite{key}` or `\cite{key1,key2}` — same as rest of thesis.
3. In prose: “Azoulay et al.” / “Fu” / “Kuling and Zitnik” — match bib author field.
4. **Never invent** citations, years, or statistics.
5. If a PDF has no bib entry: flag `[MISSING BIB: Author Year - short title]` for the user; do not cite.
6. Reviews (Zerkouk, Gligorea, Strielkowski, Fernández-Herrero) are for **framing**, not replacing primary studies.

**Known core keys (must appear):**

`strielkowski_span_2025`, `gligorea_adaptive_2023`, `zerkouk_comprehensive_2025`, `sayed_ai-based_2023`, `kuling_ken_2025`, `azoulay_adaptive_2021`, `perez_emotions_2024`, `fu_integrating_2025`, `deshmukh_developing_2025`, `guzman_developing_2025`, `weitekamp_tutorgym_2025`, `fernandez-herrero_evaluating_2024`

---

## CHRONOLOGICAL STRUCTURE (MANDATORY)

Write the section as **time-ordered narrative**, not “methods A, methods B”. Use these eras (adjust year boundaries only if a paper forces it):

### Era 0 — Concept & closed loop (1 short paragraph)
- Adaptive = change content, pace, difficulty, feedback per learner.
- Closed loop: observe ? model ? decide next step.
- Cite: `strielkowski_span_2025`, `gligorea_adaptive_2023`, `zerkouk_comprehensive_2025`.

### Era 1 — 1980s–1990s: Rule-based ITS
- Expert if–then rules (GUIDON, LISP Tutor) via `zerkouk_comprehensive_2025`.
- Strength: transparent; weakness: manual authoring per domain.

### Era 2 — 2000s–early 2010s: Student models & knowledge tracing
- Estimate mastery from response history; predict next correctness.
- Bridge to recommendation / ZPD (`kuling_ken_2025` for modern ZPD framing).
- Optional: `Pardos2014AffectiveStatesStateTests` if discussing log-based affect (brief—detailed affect is next chapter section).

### Era 3 — 2018–2021: RL enters educational task selection
- Treat next task / difficulty as sequential decision problem.
- **Islam et al. PAKES (2021)** — `islam_pakes_2021` if in bib; else note from yarab folder.
- **Azoulay et al. (2021)** — benchmark: Q-learning, TD, bandit, Bayesian; 10,000 sim runs; Bayesian 89–93% of oracle.
- Position Azoulay as **canonical comparison study** later work cites (`perez_emotions_2024`).

### Era 4 — 2022–2023: Hybrid controllers (rules + DQN)
- **Sayed et al. (2023)** — DQN for content presentation + threshold rules for difficulty; grade-3 pilot, n=26.
- **Zini et al. (2022)** — cognitive training + RL (yarab) if bib exists.

### Era 5 — 2024–2025: Knowledge tracing + RL sequencing (performance-only)
- **Fu (2025) RL-DKT** — DKT state + RL path; ASSISTments / KDD / Cognitive Tutor; AUC 0.874, F1 0.812, completion 34.5 min, dropout 3.2%.
- **Kuling & Zitnik (2025) KUL-Rec** — dual-memory recommender; nDCG 0.316; classroom n=38.
- **Pardos OATutor (2023)** — open adaptive tutor infrastructure if cited.
- **Deshmukh & Sen (2025)**, **Guzman & Cruz-Mercado (2025)** — RL for feedback / skill development.

### Era 6 — 2024–2026: Emotion signals enter adaptation (still mostly simulation)
- **Pérez et al. (2024)** — emotions as implicit RL feedback; 1,000 runs × 200 tasks; closest to thesis problem formulation.
- **Govea et al. (2024)** — DRL + emotion detection + personalization (`govea_implementation_2024`).
- **Che et al. (2025)** — multimodal feedback necessary for pedagogical RL policies (yarab).
- **Wang et al. (2024)** — knowledge tracing with emotional incorporation (yarab).
- Do **not** duplicate full emotion-aware ITS history—that belongs in `\section{Emotion-Aware Adaptation}`; only **foreshadow** the gap closing.

### Era 7 — 2025–2026: GenAI, agentic tutors, testbeds
- **Meng & Yang (2025)** self-evolving GenAI tutors + RL (`meng_self-evolving_2025`).
- **Gutierrez et al. (2025)** conversational adaptive assistants (`gutierrez_development_2025`).
- **Weitekamp TutorGym (2025)** — standardized evaluation testbed (`weitekamp_tutorgym_2025`).
- **López-Goyez et al. (2026)** multi-agent RL + GenAI ITS (if bib exists).

### Closing — Evaluation practices (1 paragraph)
- Simulation (Azoulay, Pérez) vs offline logs (Fu, Kuling) vs small pilots (Sayed, Kuling classroom).
- TutorGym as recent infrastructure.

### Closing — Strengths + gap (2 paragraphs)
- Strengths: ZPD maintenance, scale, engagement (`gligorea_adaptive_2023`, commercial platforms via `strielkowski_span_2025`).
- Gap: inputs still mostly correctness, time, hints—not facial/affective state (`azoulay_adaptive_2021`, `zerkouk_comprehensive_2025`, `fernandez-herrero_evaluating_2024`).
- Final sentence: bridge to `\ref{sec:emotion_aware_adaptation}`.

---

## PAPER CORPUS — SUB-AGENT ASSIGNMENTS

Each sub-agent: **read assigned PDFs**, return `paper_card.md` per paper (template below). Use `readonly` explore agents in parallel.

### Agent A — `02_Adaptation_Papers/` (18 PDFs, primary)

| PDF (short) | Expected era | Priority |
|-------------|--------------|----------|
| Azoulay et al. 2021 | 3 | **Must cite** |
| Pérez et al. 2024 | 6 | **Must cite** |
| Fu 2025 | 5 | **Must cite** |
| Kuling & Zitnik 2025 | 5 | **Must cite** |
| Sayed et al. 2023 | 4 | **Must cite** |
| Deshmukh & Sen 2025 | 5 | High |
| Guzman & Cruz-Mercado 2025 | 5 | High |
| Weitekamp TutorGym 2025 | 7 | High |
| Meng & Yang 2025 | 7 | High |
| Gutierrez et al. 2025 | 7 | Medium |
| Govea et al. 2024 | 6 | High |
| Kong et al. 2025 | 6–7 | Medium |
| Gong 2025 | 7 | Low |
| Liu et al. 2024 personality-aware simulation | 7 | Medium |
| López-Goyez et al. 2026 | 7 | Medium |
| Axak PPO adaptive learning control | 3–5 | Low (verify educational context) |
| Urbaite 2026 | 7 | Low (review-like) |
| paper2.pdf | ? | Read first page ? classify |

### Agent B — `yarab/` (21 PDFs)

Focus: emotion+RL, multimodal feedback, recent systems.

| PDF (short) | Notes |
|-------------|-------|
| Pérez / P03 emotions implicit feedback | Duplicate check vs 02_ |
| Islam PAKES 2021 | Era 3 |
| Zini 2022 adaptive cognitive training RL | Era 3–4 |
| Che 2025 multimodal feedback RL | Era 6 |
| Wang 2024 emotional knowledge tracing | Era 6 |
| Abhishek / Sudharsan / Prasad 2026 emotion-aware ITS+RL | Era 6–7 (table only if space) |
| Strielkowski 2025 | Framing |
| Deng 2025 goal-oriented ITS | Era 7 |
| Pardos OATutor 2023 | Era 5 infrastructure |
| Shukla GuideAI 2026 | Era 7 |
| P04, P07 packaged PDFs | Read & map to bib |

Skip for this section (wrong chapter): Immordino-Yang 2007, fpsyg Pekrun duplicate unless needed for one sentence.

### Agent C — `papers/` + `maybe adaptation/` + reviews

| Source | Role |
|--------|------|
| `papers/P03` Pérez | Confirm stats |
| `papers/Nandimandalam 2025` | Era 7 multimodal ecosystem |
| `maybe adaptation/Govea, Gong, Sudharsan, Rakib DialogXpert, P01 Meng` | Era 6–7 |
| `05_Review_Papers/` (if needed) | Zerkouk, Gligorea only |

### Agent D — Bibliography matcher

- Input: all `paper_card.md` from A–C.
- Task: map each paper ? `myref.bib` key or `[MISSING BIB]`.
- Output: `bib_mapping.csv` with columns: `filename, bib_key, year, include_in_section (yes/no/priority)`.

### Agent E — Synthesis writer (LaTeX)

- Input: chronological outline + paper cards + bib mapping + current section text.
- Output: full `\section{Adaptive Learning Systems}` LaTeX (~1,200–1,800 words body + table).
- Preserve `\label{sec:adaptive_learning_systems}`.

### Agent F — Table expander

- Expand `tab:adaptive_systems_comparison` from **4 rows ? 8–12 rows**.
- Include: Azoulay, Pérez, Kuling, Fu, Sayed, Govea, Meng OR Gutierrez, Weitekamp (pick best coverage).
- Keep `longtable` + `landscape` format identical to current template.
- Same column headers; `\makecell` for author lines.

### Agent G — Style & consistency QA

- Check: B2 language, no success-rate focus (not relevant here), all `\cite{}` keys exist, chronological order readable, no duplicate with §Emotion-Aware Adaptation, transition sentence at end.

---

## PAPER CARD TEMPLATE (each sub-agent returns this per PDF)

```markdown
## [Author et al., Year]
- **File:** path/to.pdf
- **Bib key:** key or MISSING
- **Year / Era:** 
- **Adaptation target:** (difficulty / next task / feedback / path / content presentation)
- **Input signals:** (correctness, time, KT state, emotion, multimodal, …)
- **Decision method:** (rules, BKT/DKT, RL algo, bandit, hybrid, GenAI agent, …)
- **Evaluation:** (simulation N=, offline dataset, classroom n=, weeks=)
- **Key numbers:** (exact quotes from paper)
- **Strength:** one sentence
- **Limitation:** one sentence
- **Relevance to thesis:** 1–2 sentences (emotion-aware RL tutoring)
- **Suggested prose sentence:** one draft sentence in B2 English with \cite{key}
```

---

## LATEX OUTPUT SPECIFICATION

### Section structure

```latex
\section{Adaptive Learning Systems}\label{sec:adaptive_learning_systems}

% ~8–12 paragraphs, chronological
% NO \subsection{} unless supervisor approved—use paragraph transitions

% Transition to table (fix missing space before "To summarise"):
...is the focus of the next section. To summarise the most relevant adaptive learning studies...

\clearpage
\begin{landscape}
... longtable ...
\end{landscape}
\clearpage
```

### Table rules

- Keep existing column widths and `longtable` setup.
- Add rows for newly synthesised must-include papers.
- **Relevance column** must explicitly say why it matters for **emotion-aware RL** thesis.
- Escape LaTeX specials in author names: `P\'erez`, `Lo\'pez`, etc.

### Forbidden content

- Do **not** discuss your own RL_Module experiments.
- Do **not** cite success rate as main metric (not central to this section).
- Do **not** rewrite FER or Emotion-Aware Adaptation sections.

---

## THESIS POSITIONING (weave in 2–3 times, subtly)

This thesis builds an **emotion-aware PPO** adaptive tutor using **facial emotion** + student state. The literature section should show:

| Claim | Supported by |
|-------|----------------|
| RL task selection is established but often simulation-only | Azoulay, Pérez |
| KT+RL improves sequencing on real logs | Fu, Kuling |
| Emotion as RL feedback is emerging but thin | Pérez, Govea |
| Multimodal/affective signal argued necessary | Che 2025, reviews |
| **Gap:** few systems combine **FER + RL policy** at scale with rigorous multi-seed eval | your transition sentence |

---

## ORCHESTRATION COMMAND (for Cursor / parent agent)

Run **in parallel** (readonly explore agents):

```
Agent A: Read all PDFs in 02_Adaptation_Papers/ ? paper cards
Agent B: Read all PDFs in yarab/ ? paper cards (filter adaptation-relevant)
Agent C: Read papers/ + maybe adaptation/ ? paper cards
```

Then **sequentially**:

```
Agent D: bib_mapping.csv
Agent E: Write LaTeX section draft ? literature_review_adaptive_DRAFT.tex
Agent F: Expand comparison table
Agent G: QA pass ? merge into literature_review.tex
```

---

## ACCEPTANCE CHECKLIST (Agent G)

- [ ] Narrative is **chronological** (reader can list eras with years)
- [ ] **?12 unique primary sources** cited (reviews + primary mix)
- [ ] All statistics traceable to paper cards
- [ ] Language matches adjacent sections (B2, plain academic)
- [ ] Every `\cite{key}` exists in `myref.bib`
- [ ] Table has **?8 rows**, landscape compiles
- [ ] Closing paragraph links to `\ref{sec:emotion_aware_adaptation}`
- [ ] No paragraph duplicates content from §Emotion-Aware Adaptation
- [ ] Fixed typo: space before "To summarise" in line 296

---

## QUICK START (single message to parent agent)

> Rewrite `\section{Adaptive Learning Systems}` in `literature_review.tex` using the MASTER PROMPT at `Bachelor_thesis_MET_Bachelor/prompts/MASTER_PROMPT_adaptive_learning_section.md`. Launch parallel readonly sub-agents on `02_Adaptation_Papers/`, `yarab/`, `papers/`, and `maybe adaptation/`. Build chronological B2 prose, expand the comparison table to 8–12 papers, match existing citation style, and only use keys from `myref.bib`. Output draft LaTeX for my review before merging.
