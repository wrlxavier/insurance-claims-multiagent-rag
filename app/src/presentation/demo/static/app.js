// M6-03 local demo UI. Vanilla ES module, no dependencies, no build.
// Drives the real /v1 assessment API; /demo/api adds the examples, the
// citation -> full-clause join, and the live pipeline-progress read.
// UI language: pt-BR. No changes to API contracts or endpoints below.

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
    else if (k === "text") el.textContent = v;
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
    throw new ApiError(0, "unreachable", "API inacessível — o serviço está no ar?");
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
// labels (pt-BR)
// ---------------------------------------------------------------------------
const NODE_LABEL = {
  intake: "Entrada",
  clarification: "Esclarecimento",
  clarification_exhausted: "Esclarecimento esgotado",
  retrieval: "Recuperação",
  compatibility: "Compatibilidade",
  consistency: "Consistência",
  injection_scan: "Verificação de injeção",
  recommendation: "Recomendação",
  human_review: "Revisão humana",
};
const ACTION_LABEL = {
  extract_entities: "entidades extraídas",
  generate_questions: "perguntas geradas",
  exhaust_clarification_budget: "orçamento de esclarecimento esgotado",
  retrieve_clauses: "cláusulas recuperadas",
  assess: "compatibilidade avaliada",
  deterministic_checks: "verificações determinísticas",
  semantic_judgement: "julgamento semântico",
  flagged: "trecho sinalizado",
  consolidate: "recomendação consolidada",
  persist_audit_trail_failed: "falha ao gravar trilha de auditoria",
};
const MISSING_LABEL = {
  ambito_geografico: "Abrangência geográfica",
  uso_do_veiculo: "Uso do veículo",
  data_evento_vigencia: "Data do evento × vigência da apólice",
  valor_franquia_limite: "Valor / franquia / limite",
  tipo_evento_condicao: "Tipo de evento / condição",
};
const VERDICT_LABEL = {
  compatible: "Compatível",
  incompatible: "Incompatível",
  insufficient_information: "Informação insuficiente",
};
const DECISION_LABEL = { approve: "aprovado", reject: "rejeitado", edit: "editado" };

// the stepper's collapsed view of the graph
const STAGES = [
  { key: "intake", label: "Entrada", nodes: ["intake"] },
  {
    key: "clarification",
    label: "Esclarecimento",
    nodes: ["clarification", "clarification_exhausted"],
    conditional: true,
  },
  { key: "retrieval", label: "Recuperação", nodes: ["retrieval"] },
  {
    key: "assessment",
    label: "Avaliação",
    nodes: ["compatibility", "consistency", "injection_scan"],
  },
  { key: "recommendation", label: "Recomendação", nodes: ["recommendation"] },
  { key: "review", label: "Revisão humana", nodes: ["human_review"] },
];

const verdictText = (v) => (v == null ? "—" : VERDICT_LABEL[v] || v);
const pct = (x) => (x == null ? "—" : `${Math.round(x * 100)}%`);
const ts = (s) => {
  if (!s) return "—";
  const d = new Date(s);
  return isNaN(d) ? s : d.toLocaleString();
};
const truncate = (text, n) => {
  if (!text) return text;
  return text.length > n ? text.slice(0, n).trim() + "…" : text;
};

// ---------------------------------------------------------------------------
// clause modal (full text pop-up, replaces inline expansion)
// ---------------------------------------------------------------------------
function openClauseModal(title, text) {
  $("#modal-title").textContent = title || "Cláusula completa";
  $("#modal-body").textContent = text || "(vazio)";
  $("#modal-backdrop").hidden = false;
}
function closeClauseModal() {
  $("#modal-backdrop").hidden = true;
}
function setupModal() {
  const backdrop = $("#modal-backdrop");
  $("#modal-close").addEventListener("click", closeClauseModal);
  backdrop.addEventListener("click", (e) => {
    if (e.target === backdrop) closeClauseModal();
  });
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeClauseModal();
  });
}
setupModal();

// ---------------------------------------------------------------------------
// relevance donut
// ---------------------------------------------------------------------------
function relevanceDonut(score) {
  const p = Math.round((score || 0) * 100);
  return h(
    "div",
    {
      class: "donut",
      style: `background:conic-gradient(var(--accent) ${p}%, var(--border) 0)`,
    },
    h("div", { class: "donut-hole" }, h("span", {}, `${p}%`)),
  );
}

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
      ? h("a", { href: "#", onclick: (e) => (e.preventDefault(), onRetry()) }, "Repetir")
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
    render(h("div", {}, banner("error", errText(err), route)));
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
    h("h1", {}, "Avaliar um sinistro"),
    h(
      "p",
      { class: "lead" },
      "Cole o relato do reclamante, informe o produto registrado ao qual o sinistro está " +
        "vinculado e execute a avaliação. Ou comece a partir de um dos exemplos prontos abaixo.",
    ),
  );

  const form = h(
    "form",
    { class: "card", onsubmit: (e) => (e.preventDefault(), submit()) },
    h(
      "div",
      { class: "field" },
      h("label", { for: "raw" }, "Relato do sinistro"),
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
        h("label", { for: "policy" }, "Processo SUSEP"),
        h("input", {
          id: "policy",
          type: "text",
          required: "required",
          placeholder: "15414.610650/2024-59",
        }),
        h(
          "div",
          { class: "field-hint" },
          "O produto registrado ao qual o sinistro está vinculado.",
        ),
      ),
      h(
        "div",
        { class: "field", style: "flex:1;min-width:180px" },
        h("label", { for: "cid" }, "ID do sinistro (opcional)"),
        h("input", { id: "cid", type: "text", placeholder: "automático" }),
      ),
    ),
    h(
      "div",
      { class: "btn-row" },
      h("button", { type: "submit", class: "btn btn--primary" }, "Executar avaliação"),
    ),
    h("div", { id: "form-msg" }),
  );
  wrap.append(form);

  const exWrap = h("div", { class: "section" }, h("h2", {}, "Exemplos"));
  wrap.append(exWrap);
  try {
    const examples = await apiFetch("/demo/api/examples");
    const grid = h("div", { class: "examples" });
    for (const ex of examples) grid.append(exampleCard(ex));
    exWrap.append(
      h(
        "p",
        { class: "muted", style: "font-size:13px" },
        "Selecionados do conjunto sintético de sinistros do projeto. Os veredictos indicam o " +
          "que cada execução deveria alcançar — uma execução real é não-determinística. Uma " +
          "execução completa leva de um a quatro minutos.",
      ),
      grid,
    );
  } catch (err) {
    exWrap.append(banner("warn", `Não foi possível carregar os exemplos — ${errText(err)}`));
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
          else $("#raw").focus();
        },
      },
      h("span", { class: `pill pill--${ex.category}` }, verdictText(ex.expected_verdict)),
      h("h3", {}, ex.label),
      h("p", {}, ex.note),
    );
  }

  async function submit() {
    const raw = $("#raw").value.trim();
    const policy = $("#policy").value.trim();
    const cid = $("#cid").value.trim();
    const msg = $("#form-msg");
    msg.replaceChildren();
    if (!raw) {
      msg.append(banner("error", "O relato é obrigatório."));
      return;
    }
    if (!policy) {
      msg.append(banner("error", "O processo SUSEP é obrigatório."));
      return;
    }
    const btn = form.querySelector("button[type=submit]");
    btn.disabled = true;
    btn.textContent = "Enviando…";
    try {
      const body = { raw_text: raw, policy_ref: policy };
      if (cid) body.claim_id = cid;
      const res = await apiFetch("/v1/assessments", {
        method: "POST",
        body: JSON.stringify(body),
      });
      addRecent({
        assessment_id: res.assessment_id,
        claim_id: cid || "(gerado)",
        status: "pending",
      });
      location.hash = `#/a/${encodeURIComponent(res.assessment_id)}`;
    } catch (err) {
      btn.disabled = false;
      btn.textContent = "Executar avaliação";
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
  root.append(h("div", { class: "row" }, h("span", { class: "spinner" }), " Carregando…"));

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
        root.replaceChildren(banner("error", "Nenhuma avaliação com esse id."));
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
    const wait = Date.now() - startedAt > MAX_WAIT_MS ? POLL_MS * 4 : POLL_MS;
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

  const dots = [];
  let activeIdx = -1;
  let lastResolvedIdx = -1;
  STAGES.forEach((stage, i) => {
    const st = progress
      ? stepState(stage, seen, next, a.status)
      : a.status === "running"
        ? stage.key === "intake"
          ? "active"
          : "pending"
        : "pending";
    if (st === "active" && activeIdx === -1) activeIdx = i;
    if (st === "done" || st === "skipped") lastResolvedIdx = i;
    dots.push(h("div", { class: `step step--${st}` }, h("span", { class: "step-dot" }), h("span", { class: "step-label" }, stage.label)));
  });

  let fraction;
  if (a.status === "awaiting_review") fraction = 1;
  else if (activeIdx >= 0) fraction = activeIdx / (STAGES.length - 1);
  else if (lastResolvedIdx >= 0) fraction = (lastResolvedIdx + 1) / (STAGES.length - 1);
  else fraction = 0;
  fraction = Math.max(0, Math.min(1, fraction));

  const pipeline = h(
    "div",
    { class: "pipeline" },
    h("div", { class: "pipeline-line-bg" }),
    h("div", {
      class: "pipeline-line-fill",
      style: `width:calc((100% - 76px) * ${fraction})`,
    }),
    h("div", { class: "pipeline-dots" }, ...dots),
  );

  const clarNote =
    progress && seen.includes("clarification")
      ? h(
          "p",
          { class: "muted", style: "font-size:13px" },
          "O sinistro está com informações pendentes — o ciclo de esclarecimento está em " +
            "andamento.",
        )
      : null;

  root.replaceChildren(
    h(
      "div",
      { class: "card" },
      h(
        "div",
        { class: "card-title" },
        h("h2", {}, "Avaliando…"),
        h(
          "span",
          { class: "elapsed" },
          h("span", { class: "spinner" }),
          `  ${elapsed}s decorridos`,
        ),
      ),
      h(
        "p",
        { class: "muted" },
        `Sinistro ${a.claim_id} · status: ${a.status}` +
          (progress && !progress.available
            ? " · visão ao vivo do pipeline indisponível — mostrando um cronômetro"
            : ""),
      ),
      pipeline,
      clarNote,
      over
        ? banner(
            "warn",
            "Esta execução já passa de cinco minutos. Ainda pode terminar — a consulta " +
              "continua, com menos frequência.",
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
      h("h2", {}, "A execução falhou"),
      h("p", { class: "muted" }, `Sinistro ${a.claim_id}`),
      banner("error", a.error || "Nenhuma causa registrada."),
      h(
        "p",
        { class: "muted", style: "font-size:13px" },
        "Uma falha real vale a pena mostrar. Cerca de um sinistro em dez falha na entrada " +
          "quando o modelo rápido retorna uma saída estruturada vazia — repetir a execução " +
          "geralmente funciona.",
      ),
      h("a", { class: "btn btn--sm", href: "#/" }, "Recomeçar"),
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
        h("h3", {}, "Informações pendentes"),
        h(
          "div",
          { class: "row" },
          ...a.missing_information.map((t) => h("span", { class: "chip" }, MISSING_LABEL[t] || t)),
        ),
        a.clarification_exhausted
          ? h(
              "p",
              { class: "muted", style: "margin:12px 0 0;font-size:13px" },
              "O ciclo de esclarecimento solicitou esta informação e a resposta nunca " +
                "chegou — a execução foi encerrada como informação insuficiente.",
            )
          : null,
      ),
    );
  }

  body.append(
    h(
      "div",
      { class: "card" },
      h("h3", {}, "Justificativa"),
      h("p", { lang: "pt-BR" }, a.reasoning || "—"),
      h("div", { class: "hr" }),
      h("h3", {}, "Ação recomendada"),
      h("p", { lang: "pt-BR", style: "margin:0" }, a.recommended_action || "—"),
    ),
  );

  if (a.consistency_flags && a.consistency_flags.length) {
    body.append(
      h(
        "div",
        { class: "card" },
        h("h3", {}, `Alertas de consistência (${a.consistency_flags.length})`),
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
                h("b", { style: "font-size:13.5px" }, f.check),
                h("span", { class: "pill pill--ghost" }, f.source),
              ),
              h("p", { lang: "pt-BR", class: "muted", style: "margin:6px 0 0;font-size:13.5px" }, f.detail),
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
  const confPct = Math.round((a.confidence || 0) * 100);
  return h(
    "div",
    { class: "verdict" },
    h(
      "span",
      { class: `pill pill--${a.verdict || "neutral"}`, style: "font-size:15px;padding:8px 18px" },
      verdictText(a.verdict),
    ),
    h(
      "div",
      { class: "meter" },
      h(
        "div",
        { class: "meter-head" },
        h("span", { class: "meter-label" }, "Confiança"),
        h("span", { class: "meter-value" }, `${confPct}%`),
      ),
      h("div", { class: "meter-track" }, h("div", { class: "meter-fill", style: `width:${confPct}%` })),
    ),
    h(
      "span",
      { class: `pill pill--${a.is_grounded ? "accent" : "ghost"}` },
      a.is_grounded ? "fundamentado" : "sem citações",
    ),
  );
}

function citationsCard(citations, ctxById) {
  const card = h(
    "div",
    { class: "card" },
    h("h3", {}, `Citações (${(citations || []).length})`),
  );
  if (!citations || !citations.length) {
    card.append(
      h(
        "p",
        { class: "muted", style: "margin:0" },
        "Esta avaliação não cita nenhuma cláusula — o sistema se absteve.",
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
          ctx.title ? h("span", { lang: "pt-BR" }, ctx.title) : `cláusula ${c.clause_id}`,
          `  ·  p. ${
            ctx.page_start === ctx.page_end ? ctx.page_start : `${ctx.page_start}–${ctx.page_end}`
          }`,
        )
      : h(
          "div",
          { class: "citation-src" },
          `SUSEP ${c.susep_process} · documento ${c.document_id} · cláusula ${c.clause_id}`,
        );

    const fullText = ctx ? ctx.text : c.excerpt;
    const shortExcerpt = truncate(c.excerpt, 200);
    const open = () => openClauseModal(ctx && ctx.title, fullText);

    card.append(
      h(
        "div",
        { class: "citation" },
        h(
          "div",
          { class: "row", style: "justify-content:space-between;align-items:center" },
          h("span", { class: "pill pill--ghost" }, c.clause_type),
          h(
            "div",
            { class: "row", style: "gap:8px" },
            h("span", { class: "donut-caption" }, "relevância"),
            relevanceDonut(c.relevance_score),
          ),
        ),
        src,
        h("blockquote", { class: "excerpt", lang: "pt-BR", onclick: open }, shortExcerpt || "—"),
        ctx || c.excerpt
          ? h("button", { type: "button", class: "link-btn", onclick: open }, "Ler cláusula completa →")
          : null,
      ),
    );
  }
  if (indexMissing) {
    card.append(
      h(
        "p",
        { class: "faint", style: "font-size:12px;margin:10px 0 0" },
        "Texto completo da cláusula / página não exibido para algumas citações — o corpus " +
          "processado (build/parsed_clauses.jsonl) não está disponível neste container.",
      ),
    );
  }
  return card;
}

function checkpointCard(id, a, ctxById) {
  const msg = h("div", { style: "margin-top:10px" });
  const card = h(
    "div",
    { class: "card", style: "border:2px solid var(--accent)" },
    h("h3", {}, "Checkpoint humano"),
    h(
      "p",
      { class: "muted", style: "margin-top:0" },
      "Nada é registrado até você decidir. Sua decisão é armazenada ao lado da opinião do " +
        "sistema, nunca por cima dela.",
    ),
    h(
      "div",
      { class: "field" },
      h("label", { for: "notes" }, "Observações (opcional)"),
      h("textarea", { id: "notes", rows: "2", lang: "pt-BR" }),
    ),
    h(
      "div",
      { class: "btn-row" },
      h("button", { class: "btn btn--ok", onclick: () => decide("approve") }, "Aprovar"),
      h("button", { class: "btn", onclick: () => toggleEdit() }, "Editar"),
      h("button", { class: "btn btn--bad", onclick: () => decide("reject") }, "Rejeitar"),
    ),
    msg,
  );

  const editBox = h("div", { hidden: true, style: "margin-top:20px;padding-top:20px;border-top:1px solid var(--border)" });
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
    msg.replaceChildren(h("span", { class: "spinner" }), " Enviando decisão…");
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
        toast("Já decidido.");
        route();
        return;
      }
      let extra = "";
      if (err.code === "unknown_clause" && err.details && err.details.clause_ids)
        extra = ` — cláusula(s) desconhecida(s): ${err.details.clause_ids.join(", ")}`;
      msg.replaceChildren(banner("error", errText(err) + extra));
    }
  }

  function editForm(a) {
    const wrap = h("div", { class: "stack" });
    const vSel = h(
      "select",
      { id: "e-verdict" },
      ...["compatible", "incompatible", "insufficient_information"].map((v) =>
        h("option", { value: v, selected: v === a.verdict }, VERDICT_LABEL[v]),
      ),
    );
    const reasoning = h("textarea", { id: "e-reasoning", rows: "3", lang: "pt-BR" });
    reasoning.value = a.reasoning || "";
    const action = h("textarea", { id: "e-action", rows: "2", lang: "pt-BR" });
    action.value = a.recommended_action || "";
    const conf = h("input", {
      id: "e-conf",
      type: "text",
      style: "font-family:var(--mono)",
      value: String(a.confidence ?? 0.3),
    });

    const citeWrap = h("div", { class: "stack" });
    const rows = [];
    function addCiteRow(c) {
      const r = h(
        "div",
        { class: "row", style: "gap:6px" },
        mini("clause_id", "ID da cláusula", c.clause_id),
        mini("document_id", "ID do documento", c.document_id),
        mini("susep_process", "processo SUSEP", c.susep_process),
        mini("clause_type", "tipo de cláusula", c.clause_type),
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
    function mini(fieldKey, placeholder, val) {
      const i = h("input", {
        type: "text",
        placeholder,
        value: val || "",
        style: "flex:1;min-width:110px;font-size:12px",
      });
      i.dataset.field = fieldKey;
      return i;
    }
    (a.citations || []).forEach(addCiteRow);

    wrap.append(
      h(
        "p",
        { class: "muted", style: "font-size:13px" },
        "Sua revisão é registrada na decisão — o veredito, o texto e as citações originais " +
          "do sistema permanecem inalterados.",
      ),
      field("Veredito", vSel),
      field("Justificativa", reasoning),
      field("Ação recomendada", action),
      field("Confiança (0–1)", conf),
      h("label", {}, "Citações (pelo menos uma)"),
      citeWrap,
      h(
        "div",
        { class: "btn-row", style: "margin-top:8px" },
        h(
          "button",
          {
            type: "button",
            class: "btn btn--sm",
            onclick: () => addCiteRow({ susep_process: a.citations?.[0]?.susep_process || "" }),
          },
          "+ citação",
        ),
        h(
          "button",
          { type: "button", class: "btn btn--sm btn--primary", onclick: submitEdit },
          "Enviar edição",
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
        msg.replaceChildren(banner("error", "Uma edição precisa de pelo menos uma citação."));
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
      { class: "card" },
      h(
        "div",
        { class: "card-title" },
        h("h2", {}, "Decidido"),
        h("span", { class: `pill pill--${decisionPill(d.decision)}` }, DECISION_LABEL[d.decision] || d.decision || "—"),
      ),
      h("div", { class: "kv" }, h("b", {}, "Decidido em"), ts(d.decided_at)),
      d.notes
        ? h("div", { class: "kv" }, h("b", {}, "Observações"), h("span", { lang: "pt-BR" }, d.notes))
        : null,
    ),
  );

  const system = h(
    "div",
    { class: "card" },
    h("h3", {}, "Opinião do sistema"),
    h(
      "p",
      {},
      h("span", { class: `pill pill--${a.verdict}` }, verdictText(a.verdict)),
      ` · confiança ${pct(a.confidence)}`,
    ),
    h("p", { lang: "pt-BR", class: "muted" }, a.reasoning || "—"),
  );
  const analyst = h(
    "div",
    { class: "card" },
    h("h3", {}, "Decisão do analista"),
    d.edited_assessment
      ? h(
          "div",
          {},
          h(
            "p",
            {},
            h("span", { class: `pill pill--${d.edited_assessment.verdict}` }, verdictText(d.edited_assessment.verdict)),
            ` · confiança ${pct(d.edited_assessment.confidence)}`,
          ),
          h("p", { lang: "pt-BR", class: "muted" }, d.edited_assessment.reasoning || "—"),
          h("p", { lang: "pt-BR", class: "muted", style: "margin:0" }, d.edited_assessment.recommended_action || ""),
        )
      : h(
          "p",
          { class: "muted", style: "margin:0" },
          d.decision === "approve"
            ? "Aprovado conforme recomendado."
            : d.decision === "reject"
              ? "Rejeitado. A opinião do sistema permanece registrada, sem alterações."
              : "—",
        ),
  );
  body.append(h("div", { class: "compare" }, system, analyst));

  body.append(citationsCard(a.citations, ctxById));

  // audit trail
  const auditCard = h(
    "div",
    { class: "card" },
    h("h3", {}, "Trilha de auditoria"),
    h(
      "p",
      { class: "muted", style: "margin-top:0;font-size:13px" },
      "Vazia até que uma decisão seja registrada — a trilha definitiva é gravada uma única " +
        "vez, no checkpoint.",
    ),
  );
  body.append(auditCard);
  try {
    const trail = await apiFetch(`/v1/assessments/${encodeURIComponent(id)}/audit`);
    auditCard.append(auditTable(trail.entries || []));
  } catch (err) {
    auditCard.append(banner("warn", `Não foi possível carregar a trilha de auditoria — ${errText(err)}`));
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
        ...["N°", "nó", "ação", "modelo", "tokens", "conf.", "detalhe", "hora"].map((t) => h("th", {}, t)),
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
      h("td", {}, ACTION_LABEL[e.action] || e.action.replace(/^human_decision:/, "decisão: ")),
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
              h("pre", { class: "audit-payload", lang: "pt-BR" }, JSON.stringify(e.payload, null, 2)),
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
  const root = h("div", {}, h("h1", {}, "Histórico"));
  render(root);

  const local = recents();
  if (local.length) {
    root.append(
      h("h2", { class: "section" }, "Nesta sessão"),
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
        "Mantido neste navegador. Execuções em andamento e falhas nunca aparecem na lista do " +
          "servidor abaixo — só as que aguardam revisão ou já foram decididas.",
      ),
    );
  }

  root.append(h("h2", { class: "section" }, "No servidor"));
  try {
    const rows = await apiFetch("/v1/assessments?limit=50");
    if (!rows.length) root.append(h("p", { class: "muted" }, "Nada decidido ainda."));
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
          { class: "card", style: "padding:14px 18px" },
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
                  ? h("span", { class: `pill pill--${v.verdict}`, style: "margin-left:8px" }, verdictText(v.verdict))
                  : null,
              ),
              h("span", { class: "pill pill--ghost" }, v.status || "—"),
            ),
            h(
              "div",
              { class: "faint", style: "font-size:12px;margin-top:6px;font-family:var(--mono)" },
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
    { style: "margin-top:20px" },
    h("a", { href: "#/" }, "← Nova avaliação"),
    "   ",
    h("a", { href: "#/history" }, "Histórico"),
  );
}
