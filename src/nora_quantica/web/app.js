const dom = {
  intro: document.querySelector("#introScreen"),
  enter: document.querySelector("#enterButton"),
  appShell: document.querySelector("#appShell"),
  composer: document.querySelector("#composer"),
  input: document.querySelector("#messageInput"),
  send: document.querySelector("#sendButton"),
  messages: document.querySelector("#messages"),
  empty: document.querySelector("#emptyState"),
  newChat: document.querySelector("#newChatButton"),
  labToggle: document.querySelector("#labToggle"),
  labClose: document.querySelector("#labClose"),
  labPanel: document.querySelector("#labPanel"),
  labEmpty: document.querySelector("#labEmpty"),
  labContent: document.querySelector("#labContent"),
  modeBadge: document.querySelector("#modeBadge"),
  demoNotice: document.querySelector("#demoNotice"),
  sourceStatus: document.querySelector("#sourceStatus"),
  sourceDetails: document.querySelector("#sourceDetails"),
  rawBytes: document.querySelector("#rawBytes"),
  stateMeters: document.querySelector("#stateMeters"),
  topicAffinityMeters: document.querySelector("#topicAffinityMeters"),
  impactGrid: document.querySelector("#impactGrid"),
  turnNumber: document.querySelector("#turnNumber"),
  deltaList: document.querySelector("#deltaList"),
  behaviorPlanSection: document.querySelector("#behaviorPlanSection"),
  behaviorControls: document.querySelector("#behaviorControls"),
  behaviorMove: document.querySelector("#behaviorMove"),
  instruction: document.querySelector("#behavioralInstruction"),
  modelDetails: document.querySelector("#modelDetails"),
  exportButton: document.querySelector("#exportButton"),
  toast: document.querySelector("#toast"),
};

let conversationId = null;
let sending = false;
let started = false;
let labDetailLevel = "full";
let sessionExportEnabled = true;

const labels = {
  relational_closeness: "Cercanía relacional",
  availability: "Disponibilidad",
  interest_bias: "Sesgo de interés",
  current_interest: "Interés actual",
  candor: "Franqueza",
  topic_orientation: "Orientación temática",
  topic_relevance: "Relevancia",
  novelty: "Novedad",
  continuity: "Continuidad",
  disagreement_strength: "Desacuerdo",
  social_signal: "Señal social",
  urgency: "Urgencia",
  event_intensity: "Intensidad",
  engagement_request: "Demanda",
  topic_shift: "Cambio de tema",
  primary_topic: "Tema principal",
  secondary_topic: "Tema secundario",
  social_distance: "Distancia social",
  response_budget: "Presupuesto de respuesta",
  topic_pull: "Atracción temática",
  conversational_autonomy: "Autonomía conversacional",
};

const topicLabels = {
  none: "Sin tema",
  everyday_life: "Vida cotidiana",
  personal_social: "Vida personal y social",
  arts_literature: "Arte y literatura",
  philosophy_ideas: "Filosofía e ideas",
  technology_science: "Tecnología y ciencia",
  politics_society: "Política y sociedad",
  work_study: "Trabajo y estudio",
  practical_hobbies: "Hobbies prácticos",
  travel_nature_weather: "Viajes, naturaleza y clima",
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    let message = `Error ${response.status}`;
    try {
      const payload = await response.json();
      message = Array.isArray(payload.detail)
        ? payload.detail.map((item) => item.msg).join("; ")
        : payload.detail || message;
    } catch (_) { /* La respuesta no era JSON. */ }
    throw new Error(message);
  }
  return response.json();
}

function showToast(message) {
  dom.toast.textContent = message;
  dom.toast.hidden = false;
  window.clearTimeout(showToast.timeout);
  showToast.timeout = window.setTimeout(() => { dom.toast.hidden = true; }, 4500);
}

function addMessage(role, content, thinking = false) {
  dom.empty.hidden = true;
  const article = document.createElement("article");
  article.className = `message ${role}${thinking ? " thinking" : ""}`;
  if (thinking) article.id = "thinkingMessage";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "assistant" ? "N" : "Vos";

  const body = document.createElement("div");
  body.className = "message-body";
  const meta = document.createElement("p");
  meta.className = "message-meta";
  meta.textContent = role === "assistant" ? "Nora" : "Vos";
  const text = document.createElement("div");
  text.className = "message-content";
  if (thinking) {
    text.className += " thinking-dots";
    text.innerHTML = "Pensando<span>.</span><span>.</span><span>.</span>";
  } else {
    text.textContent = content;
  }
  body.append(meta, text);
  article.append(avatar, body);
  dom.messages.append(article);
  dom.messages.scrollTo({ top: dom.messages.scrollHeight, behavior: "smooth" });
  return article;
}

function setSending(value) {
  sending = value;
  dom.send.disabled = value;
  dom.input.disabled = value;
}

async function createConversation() {
  setSending(true);
  try {
    const created = await api("/api/conversations", { method: "POST" });
    conversationId = created.conversation_id;
    sessionStorage.setItem("nora.conversation", conversationId);
    dom.messages.replaceChildren();
    dom.empty.hidden = false;
    dom.labEmpty.hidden = false;
    dom.labContent.hidden = true;
    if (sessionExportEnabled) {
      dom.exportButton.href = `/api/conversations/${conversationId}/export`;
    }
  } finally {
    setSending(false);
    dom.input.focus();
  }
}

async function restoreConversation(id) {
  try {
    const payload = await api(`/api/conversations/${id}/messages`);
    conversationId = id;
    dom.messages.replaceChildren();
    payload.messages.forEach((message) => addMessage(message.role, message.content));
    dom.empty.hidden = payload.messages.length > 0;
    if (sessionExportEnabled) {
      dom.exportButton.href = `/api/conversations/${id}/export`;
    }
    await refreshLab();
    return true;
  } catch (_) {
    sessionStorage.removeItem("nora.conversation");
    return false;
  }
}

async function sendMessage(message) {
  if (sending || !message.trim()) return;
  if (!conversationId) await createConversation();
  const clean = message.trim();
  dom.input.value = "";
  resizeInput();
  addMessage("user", clean);
  const thinking = addMessage("assistant", "", true);
  setSending(true);
  try {
    const result = await api(`/api/conversations/${conversationId}/messages`, {
      method: "POST",
      body: JSON.stringify({ message: clean }),
    });
    thinking.remove();
    addMessage("assistant", result.response);
    await refreshLab();
  } catch (error) {
    thinking.remove();
    showToast(error.message);
  } finally {
    setSending(false);
    dom.input.focus();
  }
}

function meter(name, value) {
  const row = document.createElement("div");
  row.innerHTML = `
    <div class="meter-head"><span>${labels[name] || name}</span><strong>${Math.round(value)}</strong></div>
    <div class="meter-track"><div class="meter-fill" style="width:${Math.max(0, Math.min(100, value))}%"></div></div>
  `;
  return row;
}

function details(items) {
  const fragment = document.createDocumentFragment();
  items.forEach(([key, value]) => {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = key;
    dd.textContent = value ?? "—";
    fragment.append(dt, dd);
  });
  return fragment;
}

async function refreshLab() {
  if (!conversationId) return;
  const lab = await api(`/api/conversations/${conversationId}/lab`);
  const source = lab.quantum_source;
  dom.sourceStatus.textContent = source.fallback_used ? "fallback" : "QRNG";
  dom.sourceStatus.classList.toggle("fallback", source.fallback_used);
  const sourceItems = labDetailLevel === "full"
    ? [
        ["Perfil", `${lab.prompt_profile.name} / ${lab.prompt_profile.version}`],
        ["Hash perfil", lab.prompt_profile.content_hash.slice(0, 12)],
        ["Proveedor", source.provider],
        ["Recuperado", new Date(source.retrieved_at).toLocaleString()],
        ["Hash SHA-256", source.raw_bytes_hash],
        ["Tipo de fallback", source.fallback_type],
        ["Replay auditable", lab.audit.passed ? `${lab.audit.checks} controles OK` : lab.audit.errors.join("; ")],
      ]
    : [
        ["Fuente", source.provider],
        ["Generado", new Date(source.retrieved_at).toLocaleString()],
      ];
  dom.sourceDetails.replaceChildren(details(sourceItems));
  if (labDetailLevel === "full") {
    dom.rawBytes.replaceChildren(...source.raw_bytes.map((value) => {
      const span = document.createElement("span");
      span.textContent = value.toString(16).padStart(2, "0");
      span.title = String(value);
      return span;
    }));
  }

  dom.stateMeters.replaceChildren();
  const values = {
    relational_closeness: lab.baseline.relational_closeness,
    availability: lab.current.availability,
    current_interest: lab.current.current_interest,
    candor: lab.current.candor,
    topic_orientation: lab.current.topic_orientation,
  };
  Object.entries(values).forEach(([name, value]) => dom.stateMeters.append(meter(name, value)));
  if (dom.topicAffinityMeters) {
    dom.topicAffinityMeters.replaceChildren();
    Object.entries(lab.baseline.topic_affinities || {}).forEach(([name, value]) => {
      dom.topicAffinityMeters.append(meter(topicLabels[name] || name, value));
    });
  }

  dom.labEmpty.hidden = Boolean(lab.last_turn);
  dom.labContent.hidden = false;
  if (!lab.last_turn) return;
  const turn = lab.last_turn;
  dom.turnNumber.textContent = `turno ${turn.turn_number}`;
  dom.impactGrid.replaceChildren(...Object.entries(turn.impact).map(([name, value]) => {
    const metric = document.createElement("div");
    metric.className = "metric";
    const label = document.createElement("span");
    label.textContent = labels[name] || name;
    label.title = labels[name] || name;
    const number = document.createElement("strong");
    const numeric = typeof value === "number";
    number.textContent = numeric ? Math.round(value) : (topicLabels[value] || value);
    number.classList.toggle("metric-text", !numeric);
    metric.append(label, number);
    return metric;
  }));
  dom.deltaList.replaceChildren(...Object.entries(turn.effective_deltas).map(([name, value]) => {
    const row = document.createElement("div");
    row.className = "delta-row";
    const label = document.createElement("span");
    label.textContent = labels[name] || name;
    const number = document.createElement("span");
    number.className = `delta-value ${value > 0 ? "positive" : value < 0 ? "negative" : ""}`;
    number.textContent = `${value > 0 ? "+" : ""}${value.toFixed(2)}`;
    row.append(label, number);
    return row;
  }));
  const plan = labDetailLevel === "full" ? turn.behavioral_plan : null;
  dom.behaviorPlanSection.hidden = !plan || labDetailLevel !== "full";
  if (plan && labDetailLevel === "full") {
    dom.behaviorMove.textContent = plan.move;
    dom.behaviorControls.replaceChildren(
      ...Object.entries(plan.controls).map(([name, value]) => meter(name, value)),
    );
  }
  if (labDetailLevel === "full") {
    dom.instruction.textContent = turn.behavioral_instruction;
    dom.modelDetails.replaceChildren(details([
      ["Evaluador", `${turn.evaluator_provider} / ${turn.evaluator_model}`],
      ["Latencia eval.", `${turn.evaluator_latency_ms} ms`],
      ["Generador", `${turn.generator_provider} / ${turn.generator_model}`],
      ["Latencia gen.", `${turn.generator_latency_ms} ms`],
      ["Delta máximo", turn.max_delta],
      ["Afinidad con el tema", lab.last_topic_affinity == null ? null : Math.round(lab.last_topic_affinity)],
      ["Política conductual", lab.behavior_policy],
    ]));
  }
}

function applyUiCapabilities() {
  document.querySelectorAll("[data-full-detail]").forEach((element) => {
    element.hidden = labDetailLevel !== "full";
  });
  document.querySelectorAll("[data-session-export]").forEach((element) => {
    element.hidden = !sessionExportEnabled;
    if (!sessionExportEnabled) element.removeAttribute("href");
  });
}

function toggleLab(force) {
  const open = force ?? !dom.labPanel.classList.contains("open");
  dom.labPanel.classList.toggle("open", open);
  dom.labPanel.setAttribute("aria-hidden", String(!open));
  dom.labToggle.setAttribute("aria-expanded", String(open));
  if (open) refreshLab().catch((error) => showToast(error.message));
}

function resizeInput() {
  dom.input.style.height = "auto";
  dom.input.style.height = `${Math.min(dom.input.scrollHeight, 180)}px`;
}

dom.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(dom.input.value);
});
dom.input.addEventListener("input", resizeInput);
dom.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    dom.composer.requestSubmit();
  }
});
dom.newChat.addEventListener("click", () => createConversation().catch((error) => showToast(error.message)));
dom.labToggle.addEventListener("click", () => toggleLab());
dom.labClose.addEventListener("click", () => toggleLab(false));

async function enterExperience() {
  if (started) return;
  started = true;
  dom.enter.disabled = true;
  try {
    const stored = sessionStorage.getItem("nora.conversation");
    if (!stored || !(await restoreConversation(stored))) await createConversation();
    dom.intro.hidden = true;
    dom.appShell.hidden = false;
    dom.input.focus();
  } catch (error) {
    started = false;
    dom.enter.disabled = false;
    showToast(`No se pudo iniciar Nora: ${error.message}`);
  }
}

dom.enter.addEventListener("click", enterExperience);

async function boot() {
  try {
    const health = await api("/api/health");
    dom.modeBadge.textContent = health.mode;
    dom.modeBadge.hidden = false;
    dom.demoNotice.hidden = health.mode !== "demo";
    labDetailLevel = health.lab_detail_level || "full";
    sessionExportEnabled = health.session_export_enabled !== false;
    applyUiCapabilities();
  } catch (error) {
    showToast(`No se pudo iniciar Nora: ${error.message}`);
  }
}

boot();
