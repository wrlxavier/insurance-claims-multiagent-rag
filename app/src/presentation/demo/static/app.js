// M6-03 local demo UI. Vanilla ES module, no dependencies, no build.
// Drives the real /v1 assessment API; /demo/api adds the examples, the
// citation -> full-clause join, and the live pipeline-progress read.

const POLL_MS = 2500;
const PROGRESS_POLL_MS = 3500;
const MAX_WAIT_MS = 5 * 60 * 1000;
const AUTOSUBMIT_EXAMPLES = true;
const RECENTS_KEY = "demo.recents";

// ---------------------------------------------------------------------------
// tiny DOM helper — children are appended as text unless they are Nodes
// ---------------------------------------------------------------------------
function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v; // model text always arrives this way
    else if (k === "value") el.value = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else el.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}
const $ = (sel) => document.querySelector(sel);

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), 3200);
}

class ApiError extends Error {
  constructor(status, code, message, details) {
    super(message || `HTTP ${status}`);
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

async function apiFetch(path, opts) {
  let res;
  try {
    res = await fetch(path, {
      headers: { "content-type": "application/json" },
      ...opts,
    });
  } catch {
    throw new ApiError(0, "unreachable", "API unreachable — is the stack up?");
  }
  const bodyText = await res.text();
  let body = null;
  try {
    body = bodyText ? JSON.parse(bodyText) : null;
  } catch {
    /* leave body null */
  }
  if (!res.ok) {
    const e = body && body.error;
    throw new ApiError(
      res.status,
      e ? e.code : "http_error",
      e ? e.message : `HTTP ${res.status}`,
      e ? e.details : null,
    );
  }
  return body;
}

// ---------------------------------------------------------------------------
// recents (localStorage) — the /v1 list only returns awaiting_review/decided,
// so in-flight and failed runs are tracked here to stay reachable.
// ---------------------------------------------------------------------------
function recents() {
  try {
    return JSON.parse(localStorage.getItem(RECENTS_KEY) || "[]");
  } catch {
    return [];
  }
}
function addRecent(entry) {
  try {
    const list = recents().filter((r) => r.assessment_id !== entry.assessment_id);
    list.unshift({ ...entry, at: Date.now() });
    localStorage.setItem(RECENTS_KEY, JSON.stringify(list.slice(0, 25)));
  } catch {
    /* private mode — ignore */
  }
}

// ---------------------------------------------------------------------------
// labels
// ---------------------------------------------------------------------------
const NODE_LABEL = {
  intake: "Intake",
  clarification: "Clarification",
  clarification_exhausted: "Clarification exhausted",
  retrieval: "Retrieval",
  compatibility: "Compatibility",
  consistency: "Consistency",
  injection_scan: "Injection scan",
  recommendation: "Recommendation",
  human_review: "Human review",
};
const ACTION_LABEL = {
  extract_entities: "extracted entities",
  generate_questions: "generated questions",
  exhaust_clarification_budget: "clarification budget exhausted",
  retrieve_clauses: "retrieved clauses",
  assess: "assessed compatibility",
  deterministic_checks: "deterministic checks",
  semantic_judgement: "semantic judgement",
  flagged: "flagged a span",
  consolidate: "consolidated recommendation",
  persist_audit_trail_failed: "audit-trail write failed",
};
const MISSING_LABEL = {
  ambito_geografico: "Geographic scope",
  uso_do_veiculo: "Vehicle use",
  data_evento_vigencia: "Event date vs. policy period",
  valor_franquia_limite: "Amount / deductible / limit",
  tipo_evento_condicao: "Event type / condition",
};
// the stepper's collapsed view of the graph
const STAGES = [
  { key: "intake", label: "Intake", nodes: ["intake"] },
  {
    key: "clarification",
    label: "Clarification",
    nodes: ["clarification", "clarification_exhausted"],
    conditional: true,
  },
  { key: "retrieval", label: "Retrieval", nodes: ["retrieval"] },
  {
    key: "assessment",
    label: "Assessment",
    nodes: ["compatibility", "consistency", "injection_scan"],
  },
  { key: "recommendation", label: "Recommendation", nodes: ["recommendation"] },
  { key: "review", label: "Human review", nodes: ["human_review"] },
];

const verdictText = (v) =>
  v == null ? "—" : v.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
const pct = (x) => (x == null ? "—" : `${Math.round(x * 100)}%`);
const ts = (s) => {
  if (!s) return "—";
  const d = new Date(s);
  return isNaN(d) ? s : d.toLocaleString();
};

// ---------------------------------------------------------------------------
// router
// ---------------------------------------------------------------------------
const view = () => $("#view");
let teardown = null;

function render(node) {
  if (teardown) {
    teardown();
    teardown = null;
  }
  view().replaceChildren(node);
}

function banner(kind, msg, onRetry) {
  return h(
    "div",
    { class: `banner banner--${kind}` },
    msg,
    onRetry ? " " : null,
    onRetry
      ? h("a", { href: "#", onclick: (e) => (e.preventDefault(), onRetry()) }, "Retry")
      : null,
  );
}

async function route() {
  const hash = location.hash || "#/";
  const m = hash.match(/^#\/a\/([^/]+)/);
  try {
    if (m) await viewAssessment(decodeURIComponent(m[1]));
    else if (hash.startsWith("#/history")) await viewHistory();
    else await viewIntake();
  } catch (err) {
    render(
      h(
        "div",
        {},
        banner("error", errText(err), route),
      ),
    );
  }
}
const errText = (e) =>
  e instanceof ApiError ? `${e.message}${e.code ? ` (${e.code})` : ""}` : String(e);

window.addEventListener("hashchange", route);
window.addEventListener("DOMContentLoaded", route);

// ===========================================================================
// view: intake
// ===========================================================================
async function viewIntake() {
  const wrap = h("div", {});
  render(wrap);

  wrap.append(
    h("h1", {}, "Assess a claim"),
    h(
      "p",
      { class: "lead" },
      "Paste a claimant's narrative, optionally name the registered product it " +
        "is filed against, and run it through the assessment graph. Or start " +
        "from one of the prepared examples below.",
    ),
  );

  const form = h(
    "form",
    { class: "card", onsubmit: (e) => (e.preventDefault(), submit()) },
    h(
      "div",
      { class: "field" },
      h("label", { for: "raw" }, "Claim narrative"),
      h("textarea", {
        id: "raw",
        lang: "pt-BR",
        rows: "7",
        required: "required",
        placeholder: "Ao dar ré na garagem bati a traseira do carro no portão…",
      }),
    ),
    h(
      "div",
      { class: "row" },
      h(
        "div",
        { class: "field", style: "flex:1;min-width:220px" },
        h("label", { for: "policy" }, "SUSEP process (optional)"),
        h("input", {
          id: "policy",
          type: "text",
          placeholder: "15414.610650/2024-59",
        }),
        h(
          "div",
          { class: "field-hint" },
          "The registered product the claim is filed against.",
        ),
      ),
      h(
        "div",
        { class: "field", style: "flex:1;min-width:180px" },
        h("label", { for: "cid" }, "Claim id (optional)"),
        h("input", { id: "cid", type: "text", placeholder: "auto" }),
      ),
    ),
    h(
      "div",
      { class: "btn-row" },
      h("button", { type: "submit", class: "btn btn--primary" }, "Run assessment"),
    ),
    h("div", { id: "form-msg" }),
  );
  wrap.append(form);

  const exWrap = h("div", { class: "section" }, h("h2", {}, "Examples"));
  wrap.append(exWrap);
  try {
    const examples = await apiFetch("/demo/api/examples");
    const grid = h("div", { class: "examples" });
    for (const ex of examples) grid.append(exampleCard(ex));
    exWrap.append(
      h(
        "p",
        { class: "muted", style: "font-size:13px" },
        "Curated from the project's synthetic claim set. Verdicts are what each " +
          "run should reach — a live run is non-deterministic. A full run takes " +
          "about one to four minutes.",
      ),
      grid,
    );
  } catch (err) {
    exWrap.append(banner("warn", `Could not load examples — ${errText(err)}`));
  }

  function exampleCard(ex) {
    return h(
      "button",
      {
        type: "button",
        class: "example",
        onclick: () => {
          $("#raw").value = ex.raw_text;
          $("#policy").value = ex.policy_ref || "";
          $("#cid").value = ex.claim_id || "";
          if (AUTOSUBMIT_EXAMPLES) submit();
          else $("#raw").scrollIntoView({ behavior: "smooth", block: "center" });
        },
      },
      h(
        "span",
        { class: `pill pill--${ex.category}` },
        verdictText(ex.expected_verdict),
      ),
      h("h3", {}, ex.label),
      h("p", {}, ex.note),
    );
  }

  async function submit() {
    const raw = $("#raw").value.trim();
    const msg = $("#form-msg");
    msg.replaceChildren();
    if (!raw) {
      msg.append(banner("error", "The narrative is required."));
      return;
    }
    const btn = form.querySelector("button[type=submit]");
    btn.disabled = true;
    btn.textContent = "Submitting…";
    try {
      const body = { raw_text: raw };
      const policy = $("#policy").value.trim();
      const cid = $("#cid").value.trim();
      if (policy) body.policy_ref = policy;
      if (cid) body.claim_id = cid;
      const res = await apiFetch("/v1/assessments", {
        method: "POST",
        body: JSON.stringify(body),
      });
      addRecent({
        assessment_id: res.assessment_id,
        claim_id: cid || "(minted)",
        status: "pending",
      });
      location.hash = `#/a/${encodeURIComponent(res.assessment_id)}`;
    } catch (err) {
      btn.disabled = false;
      btn.textContent = "Run assessment";
      msg.append(banner("error", errText(err)));
    }
  }
}

// ===========================================================================
// view: assessment (progress / review / decided / failed)
// ===========================================================================
async function viewAssessment(id) {
  const root = h("div", {});
  render(root);
  root.append(h("div", { class: "row" }, h("span", { class: "spinner" }), " Loading…"));

  let stopped = false;
  let progressAvailable = true;
  const startedAt = Date.now();
  let progress = null;

  teardown = () => {
    stopped = true;
  };

  async function tick() {
    if (stopped) return;
    let a;
    try {
      a = await apiFetch(`/v1/assessments/${encodeURIComponent(id)}`);
    } catch (err) {
      if (err.status === 404) {
        root.replaceChildren(banner("error", "No assessment with that id."));
        return;
      }
      root.replaceChildren(banner("error", errText(err), () => route()));
      scheduleNext();
      return;
    }
    addRecent({
      assessment_id: id,
      claim_id: a.claim_id,
      status: a.status,
      verdict: a.verdict,
    });

    if (a.status === "pending" || a.status === "running") {
      if (progressAvailable) {
        try {
          const p = await apiFetch(
            `/demo/api/assessments/${encodeURIComponent(id)}/progress`,
          );
          progress = p;
          if (!p.available) progressAvailable = false;
        } catch {
          progressAvailable = false;
        }
      }
      renderProgress(root, id, a, progress, startedAt);
      scheduleNext();
    } else if (a.status === "failed") {
      root.replaceChildren(renderFailed(id, a));
    } else if (a.status === "awaiting_review") {
      await renderReview(root, id, a);
    } else if (a.status === "decided") {
      await renderDecided(root, id, a);
    }
  }

  function scheduleNext() {
    if (stopped) return;
    const wait =
      Date.now() - startedAt > MAX_WAIT_MS ? POLL_MS * 4 : POLL_MS;
    setTimeout(tick, wait);
  }

  tick();
}

function stepState(stage, seen, next, status) {
  const done = stage.nodes.some((n) => seen.includes(n));
  const active = stage.nodes.some((n) => next.includes(n));
  if (status === "awaiting_review" && stage.key !== "review") return "done";
  if (done && !active) return "done";
  if (active) return "active";
  if (stage.conditional && seen.length && !done) return "skipped";
  return "pending";
}

function renderProgress(root, id, a, progress, startedAt) {
  const seen = (progress && progress.nodes_seen) || [];
  const next = (progress && progress.next_nodes) || [];
  const elapsed = Math.round((Date.now() - startedAt) / 1000);
  const over = Date.now() - startedAt > MAX_WAIT_MS;

  const pipeline = h("div", { class: "pipeline" });
  for (const stage of STAGES) {
    const st = progress
      ? stepState(stage, seen, next, a.status)
      : a.status === "running"
        ? stage.key === "intake"
          ? "active"
          : "pending"
        : "pending";
    pipeline.append(
      h(
        "div",
        { class: `step step--${st}` },
        h("span", { class: "step-dot" }),
        stage.label,
      ),
    );
  }

  const clarNote =
    progress && seen.includes("clarification")
      ? h(
          "p",
          { class: "muted", style: "font-size:13px" },
          "The claim is missing information — the clarification loop is running.",
        )
      : null;

  root.replaceChildren(
    h(
      "div",
      { class: "card" },
      h(
        "div",
        { class: "card-title" },
        h("h2", {}, "Assessing…"),
        h(
          "span",
          { class: "elapsed" },
          h("span", { class: "spinner" }),
          `  ${elapsed}s elapsed`,
        ),
      ),
      h(
        "p",
        { class: "muted" },
        `Claim ${a.claim_id} · status: ${a.status}` +
          (progress && !progress.available
            ? " · live pipeline view unavailable — showing a timer"
            : ""),
      ),
      pipeline,
      clarNote,
      over
        ? banner(
            "warn",
            "This run has been going for over five minutes. It may still finish " +
              "— polling continues, less often.",
          )
        : null,
    ),
    backLink(),
  );
}

function renderFailed(id, a) {
  return h(
    "div",
    {},
    h(
      "div",
      { class: "card" },
      h("h2", {}, "The run failed"),
      h("p", { class: "muted" }, `Claim ${a.claim_id}`),
      banner("error", a.error || "No cause recorded."),
      h(
        "p",
        { class: "muted", style: "font-size:13px" },
        "A real failure is worth showing. About one claim in ten fails at intake " +
          "when the fast model returns empty structured output — re-running " +
          "usually succeeds.",
      ),
      h("a", { class: "btn btn--sm", href: "#/" }, "Start over"),
    ),
    backLink(),
  );
}

// --- review ---------------------------------------------------------------
async function renderReview(root, id, a) {
  const ctxById = await clauseContext(a.citations);
  const body = h("div", {});

  body.append(verdictBanner(a));

  if (a.missing_information && a.missing_information.length) {
    body.append(
      h(
        "div",
        { class: "card" },
        h("h3", {}, "Still missing"),
        h(
          "div",
          { class: "row" },
          ...a.missing_information.map((t) =>
            h("span", { class: "chip" }, MISSING_LABEL[t] || t),
          ),
        ),
        a.clarification_exhausted
          ? h(
              "p",
              { class: "muted", style: "margin:8px 0 0;font-size:13px" },
              "The clarification loop asked for these and the answers never came " +
                "back — the run terminated as insufficient_information.",
            )
          : null,
      ),
    );
  }

  body.append(
    h(
      "div",
      { class: "card" },
      h("h3", {}, "Reasoning"),
      h("p", { lang: "pt-BR" }, a.reasoning || "—"),
      h("div", { class: "hr" }),
      h("h3", {}, "Recommended action"),
      h("p", { lang: "pt-BR", style: "margin:0" }, a.recommended_action || "—"),
    ),
  );

  if (a.consistency_flags && a.consistency_flags.length) {
    body.append(
      h(
        "div",
        { class: "card" },
        h("h3", {}, `Consistency flags (${a.consistency_flags.length})`),
        h(
          "div",
          { class: "stack" },
          ...a.consistency_flags.map((f) =>
            h(
              "div",
              {},
              h(
                "div",
                { class: "row" },
                h("span", { class: `pill pill--${f.severity}` }, f.severity),
                h("b", { style: "font-size:13px" }, f.check),
                h("span", { class: "pill pill--ghost" }, f.source),
              ),
              h("p", { lang: "pt-BR", class: "muted", style: "margin:4px 0 0" }, f.detail),
            ),
          ),
        ),
      ),
    );
  }

  body.append(citationsCard(a.citations, ctxById));
  body.append(checkpointCard(id, a, ctxById));
  body.append(backLink());
  root.replaceChildren(body);
}

function verdictBanner(a) {
  return h(
    "div",
    { class: "verdict" },
    h(
      "span",
      { class: `pill pill--${a.verdict || "neutral"}`, style: "font-size:13px" },
      verdictText(a.verdict),
    ),
    h("span", { class: "verdict-label" }, verdictText(a.verdict)),
    h(
      "div",
      { class: "meter" },
      h(
        "div",
        { class: "meter-track" },
        h("div", {
          class: "meter-fill",
          style: `width:${Math.round((a.confidence || 0) * 100)}%`,
        }),
      ),
      h("div", { class: "meter-cap" }, `confidence ${pct(a.confidence)}`),
    ),
    h(
      "span",
      { class: `pill pill--${a.is_grounded ? "accent" : "ghost"}` },
      a.is_grounded ? "grounded" : "no citations",
    ),
  );
}

function citationsCard(citations, ctxById) {
  const card = h(
    "div",
    { class: "card" },
    h("h3", {}, `Citations (${(citations || []).length})`),
  );
  if (!citations || !citations.length) {
    card.append(
      h(
        "p",
        { class: "muted", style: "margin:0" },
        "This assessment cites no clause — it abstained.",
      ),
    );
    return card;
  }
  const indexMissing = citations.some((c) => !ctxById[c.clause_id]);
  for (const c of citations) {
    const ctx = ctxById[c.clause_id];
    const src = ctx
      ? h(
          "div",
          { class: "citation-src" },
          h("b", {}, ctx.insurer),
          ` · ${ctx.product_line} · SUSEP ${ctx.susep_process} · ${ctx.filing_year}`,
          h("br"),
          ctx.title
            ? h("span", { lang: "pt-BR" }, ctx.title)
            : `clause ${c.clause_id}`,
          `  ·  p. ${
            ctx.page_start === ctx.page_end
              ? ctx.page_start
              : `${ctx.page_start}–${ctx.page_end}`
          }`,
        )
      : h(
          "div",
          { class: "citation-src" },
          `SUSEP ${c.susep_process} · document ${c.document_id} · clause ${c.clause_id}`,
        );

    card.append(
      h(
        "div",
        { class: "citation" },
        h(
          "div",
          { class: "row", style: "justify-content:space-between" },
          h("span", { class: "pill pill--ghost" }, c.clause_type),
          h(
            "span",
            { class: "faint", style: "font-size:12px" },
            `relevance ${c.relevance_score != null ? c.relevance_score.toFixed(2) : "—"}`,
          ),
        ),
        src,
        h("blockquote", { class: "excerpt", lang: "pt-BR" }, c.excerpt || "—"),
        ctx
          ? h(
              "details",
              {},
              h("summary", {}, "Full clause text"),
              h("div", { class: "clause-full", lang: "pt-BR" }, ctx.text || "(empty)"),
            )
          : null,
      ),
    );
  }
  if (indexMissing) {
    card.append(
      h(
        "p",
        { class: "faint", style: "font-size:12px;margin:10px 0 0" },
        "Full clause text / page not shown for some citations — the parsed corpus " +
          "(build/parsed_clauses.jsonl) is not available to this container.",
      ),
    );
  }
  return card;
}

function checkpointCard(id, a, ctxById) {
  const msg = h("div", { style: "margin-top:10px" });
  const card = h(
    "div",
    { class: "card", style: "border-color:var(--accent)" },
    h("h3", {}, "Human checkpoint"),
    h(
      "p",
      { class: "muted", style: "margin-top:0" },
      "Nothing is recorded until you decide. Your decision is stored beside the " +
        "system's opinion, never over it.",
    ),
    h(
      "div",
      { class: "field" },
      h("label", { for: "notes" }, "Notes (optional)"),
      h("textarea", { id: "notes", rows: "2", lang: "pt-BR" }),
    ),
    h(
      "div",
      { class: "btn-row" },
      h(
        "button",
        { class: "btn btn--ok", onclick: () => decide("approve") },
        "Approve",
      ),
      h("button", { class: "btn", onclick: () => toggleEdit() }, "Edit"),
      h(
        "button",
        { class: "btn btn--bad", onclick: () => decide("reject") },
        "Reject",
      ),
    ),
    msg,
  );

  const editBox = h("div", { hidden: true, style: "margin-top:14px" });
  card.append(editBox);
  let editBuilt = false;

  function toggleEdit() {
    editBox.hidden = !editBox.hidden;
    if (!editBox.hidden && !editBuilt) {
      editBox.append(editForm(a));
      editBuilt = true;
    }
  }

  async function decide(decision, edited) {
    const btns = card.querySelectorAll("button");
    btns.forEach((b) => (b.disabled = true));
    msg.replaceChildren(h("span", { class: "spinner" }), " Submitting decision…");
    try {
      const body = { decision, notes: $("#notes").value };
      if (edited) body.edited = edited;
      await apiFetch(`/v1/assessments/${encodeURIComponent(id)}/decision`, {
        method: "POST",
        body: JSON.stringify(body),
      });
      route(); // re-poll -> decided view
    } catch (err) {
      btns.forEach((b) => (b.disabled = false));
      if (err.status === 409) {
        toast("Already decided.");
        route();
        return;
      }
      let extra = "";
      if (err.code === "unknown_clause" && err.details && err.details.clause_ids)
        extra = ` — unknown clause(s): ${err.details.clause_ids.join(", ")}`;
      msg.replaceChildren(banner("error", errText(err) + extra));
    }
  }

  function editForm(a) {
    const wrap = h("div", {});
    const vSel = h(
      "select",
      { id: "e-verdict" },
      ...["compatible", "incompatible", "insufficient_information"].map((v) =>
        h("option", { value: v, selected: v === a.verdict }, verdictText(v)),
      ),
    );
    const reasoning = h("textarea", { id: "e-reasoning", rows: "3", lang: "pt-BR" });
    reasoning.value = a.reasoning || "";
    const action = h("textarea", { id: "e-action", rows: "2", lang: "pt-BR" });
    action.value = a.recommended_action || "";
    const conf = h("input", {
      id: "e-conf",
      type: "text",
      value: String(a.confidence ?? 0.3),
    });

    const citeWrap = h("div", { class: "stack" });
    const rows = [];
    function addCiteRow(c) {
      const r = h(
        "div",
        { class: "row", style: "gap:6px" },
        mini("clause_id", c.clause_id),
        mini("document_id", c.document_id),
        mini("susep_process", c.susep_process),
        mini("clause_type", c.clause_type),
        h(
          "button",
          {
            type: "button",
            class: "btn btn--sm",
            onclick: () => {
              citeWrap.removeChild(r);
              rows.splice(rows.indexOf(entry), 1);
            },
          },
          "×",
        ),
      );
      const entry = { row: r };
      rows.push(entry);
      citeWrap.append(r);
    }
    function mini(name, val) {
      const i = h("input", {
        type: "text",
        placeholder: name,
        value: val || "",
        style: "flex:1;min-width:90px;font-size:12px",
      });
      i.dataset.field = name;
      return i;
    }
    (a.citations || []).forEach(addCiteRow);

    wrap.append(
      h("p", { class: "muted", style: "font-size:13px" },
        "Your revision is recorded in the decision — the system's own verdict, " +
          "prose and citations stay unchanged."),
      field("Verdict", vSel),
      field("Reasoning", reasoning),
      field("Recommended action", action),
      field("Confidence (0–1)", conf),
      h("label", {}, "Citations (at least one)"),
      citeWrap,
      h(
        "div",
        { class: "btn-row", style: "margin-top:8px" },
        h(
          "button",
          {
            type: "button",
            class: "btn btn--sm",
            onclick: () =>
              addCiteRow({ susep_process: a.citations?.[0]?.susep_process || "" }),
          },
          "+ citation",
        ),
        h(
          "button",
          { type: "button", class: "btn btn--sm btn--primary", onclick: submitEdit },
          "Submit edit",
        ),
      ),
    );
    return wrap;

    function submitEdit() {
      const citations = rows.map((entry) => {
        const inputs = entry.row.querySelectorAll("input[data-field]");
        const o = {};
        inputs.forEach((i) => (o[i.dataset.field] = i.value.trim()));
        const orig = (a.citations || []).find((c) => c.clause_id === o.clause_id);
        o.excerpt = orig ? orig.excerpt : o.clause_id;
        o.relevance_score = orig ? orig.relevance_score : 0;
        return o;
      });
      if (!citations.length) {
        msg.replaceChildren(banner("error", "An edit needs at least one citation."));
        return;
      }
      decide("edit", {
        verdict: vSel.value,
        reasoning: reasoning.value.trim(),
        recommended_action: action.value.trim(),
        confidence: Number(conf.value) || 0,
        citations,
      });
    }
  }

  function field(label, control) {
    return h("div", { class: "field" }, h("label", {}, label), control);
  }

  return card;
}

// --- decided + audit ----------------------------------------------------
async function renderDecided(root, id, a) {
  const ctxById = await clauseContext(a.citations);
  const d = a.decision || {};
  const body = h("div", {});

  body.append(
    h(
      "div",
      { class: `card` },
      h(
        "div",
        { class: "card-title" },
        h("h2", {}, "Decided"),
        h("span", { class: `pill pill--${decisionPill(d.decision)}` }, d.decision || "—"),
      ),
      h("div", { class: "kv" }, h("b", {}, "Decided at"), ts(d.decided_at)),
      d.notes
        ? h("div", { class: "kv" }, h("b", {}, "Notes"), h("span", { lang: "pt-BR" }, d.notes))
        : null,
    ),
  );

  const system = h(
    "div",
    { class: "card" },
    h("h3", {}, "System opinion"),
    h("p", {}, h("span", { class: `pill pill--${a.verdict}` }, verdictText(a.verdict)), ` · confidence ${pct(a.confidence)}`),
    h("p", { lang: "pt-BR", class: "muted" }, a.reasoning || "—"),
  );
  const analyst = h(
    "div",
    { class: "card" },
    h("h3", {}, "Analyst decision"),
    d.edited_assessment
      ? h(
          "div",
          {},
          h("p", {}, h("span", { class: `pill pill--${d.edited_assessment.verdict}` }, verdictText(d.edited_assessment.verdict)), ` · confidence ${pct(d.edited_assessment.confidence)}`),
          h("p", { lang: "pt-BR", class: "muted" }, d.edited_assessment.reasoning || "—"),
          h("p", { lang: "pt-BR", class: "muted", style: "margin:0" }, d.edited_assessment.recommended_action || ""),
        )
      : h(
          "p",
          { class: "muted", style: "margin:0" },
          d.decision === "approve"
            ? "Approved as recommended."
            : d.decision === "reject"
              ? "Rejected. The system's opinion stands on record, unchanged."
              : "—",
        ),
  );
  body.append(h("div", { class: "compare" }, system, analyst));

  body.append(citationsCard(a.citations, ctxById));

  // audit trail
  const auditCard = h(
    "div",
    { class: "card" },
    h("h3", {}, "Audit trail"),
    h(
      "p",
      { class: "muted", style: "margin-top:0;font-size:13px" },
      "Empty until a decision is submitted — the durable trail is written once, " +
        "at the checkpoint.",
    ),
  );
  body.append(auditCard);
  try {
    const trail = await apiFetch(`/v1/assessments/${encodeURIComponent(id)}/audit`);
    auditCard.append(auditTable(trail.entries || []));
  } catch (err) {
    auditCard.append(banner("warn", `Could not load the audit trail — ${errText(err)}`));
  }

  body.append(backLink());
  root.replaceChildren(body);
}

const decisionPill = (d) =>
  d === "approve" ? "compatible" : d === "reject" ? "incompatible" : "product_claim_mismatch";

function auditTable(entries) {
  const wrap = h("div", { class: "audit-wrap" });
  const table = h(
    "table",
    { class: "audit" },
    h(
      "thead",
      {},
      h(
        "tr",
        {},
        ...["#", "node", "action", "model", "tokens", "conf", "detail", "time"].map((t) =>
          h("th", {}, t),
        ),
      ),
    ),
  );
  const tbody = h("tbody", {});
  for (const e of entries) {
    const isDecision = e.action && e.action.startsWith("human_decision:");
    const tr = h(
      "tr",
      { class: isDecision ? "audit-row--decision" : "" },
      h("td", {}, String(e.sequence)),
      h("td", {}, NODE_LABEL[e.node] || e.node),
      h("td", {}, ACTION_LABEL[e.action] || e.action.replace(/^human_decision:/, "decision: ")),
      h("td", {}, e.model || "—"),
      h(
        "td",
        {},
        e.total_tokens != null
          ? `${e.input_tokens ?? "?"}/${e.output_tokens ?? "?"}/${e.total_tokens}`
          : "—",
      ),
      h("td", {}, e.confidence != null ? e.confidence.toFixed(2) : "—"),
      h("td", { class: "wrap" }, e.node_input || "—"),
      h("td", {}, ts(e.timestamp)),
    );
    tbody.append(tr);
    if (e.payload) {
      tbody.append(
        h(
          "tr",
          {},
          h(
            "td",
            { colspan: "8" },
            h(
              "details",
              { open: isDecision ? "open" : null },
              h("summary", {}, "payload"),
              h(
                "pre",
                { class: "audit-payload", lang: "pt-BR" },
                JSON.stringify(e.payload, null, 2),
              ),
            ),
          ),
        ),
      );
    }
  }
  table.append(tbody);
  wrap.append(table);
  return wrap;
}

// ===========================================================================
// view: history
// ===========================================================================
async function viewHistory() {
  const root = h("div", {}, h("h1", {}, "History"));
  render(root);

  const local = recents();
  if (local.length) {
    root.append(
      h("h2", { class: "section" }, "This session"),
      list(local, (r) => ({
        id: r.assessment_id,
        claim: r.claim_id,
        status: r.status,
        verdict: r.verdict,
        at: r.at,
      })),
      h(
        "p",
        { class: "faint", style: "font-size:12px" },
        "Kept in this browser. In-flight and failed runs never appear in the " +
          "server list below — only awaiting-review and decided ones do.",
      ),
    );
  }

  root.append(h("h2", { class: "section" }, "On the server"));
  try {
    const rows = await apiFetch("/v1/assessments?limit=50");
    if (!rows.length) root.append(h("p", { class: "muted" }, "Nothing decided yet."));
    else
      root.append(
        list(rows, (a) => ({
          id: a.assessment_id,
          claim: a.claim_id,
          status: a.status,
          verdict: a.verdict,
          at: Date.parse(a.created_at),
        })),
      );
  } catch (err) {
    root.append(banner("error", errText(err), viewHistory));
  }

  function list(items, pick) {
    const ul = h("ul", { class: "list-reset stack" });
    for (const it of items) {
      const v = pick(it);
      ul.append(
        h(
          "li",
          { class: "card", style: "padding:12px 16px" },
          h(
            "a",
            { href: `#/a/${encodeURIComponent(v.id)}`, style: "display:block" },
            h(
              "div",
              { class: "row", style: "justify-content:space-between" },
              h(
                "span",
                {},
                h("b", {}, v.claim || v.id),
                v.verdict
                  ? h(
                      "span",
                      { class: `pill pill--${v.verdict}`, style: "margin-left:8px" },
                      verdictText(v.verdict),
                    )
                  : null,
              ),
              h("span", { class: "pill pill--ghost" }, v.status || "—"),
            ),
            h(
              "div",
              { class: "faint", style: "font-size:12px;margin-top:4px" },
              `${v.id}${v.at ? ` · ${new Date(v.at).toLocaleString()}` : ""}`,
            ),
          ),
        ),
      );
    }
    return ul;
  }
}

// ===========================================================================
// shared
// ===========================================================================
async function clauseContext(citations) {
  const ids = [...new Set((citations || []).map((c) => c.clause_id))];
  if (!ids.length) return {};
  try {
    const q = ids.map((i) => `clause_id=${encodeURIComponent(i)}`).join("&");
    const res = await apiFetch(`/demo/api/clause-context?${q}`);
    return res.contexts || {};
  } catch {
    return {};
  }
}

function backLink() {
  return h(
    "p",
    { style: "margin-top:18px" },
    h("a", { href: "#/" }, "← New assessment"),
    "   ",
    h("a", { href: "#/history" }, "History"),
  );
}
