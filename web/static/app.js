const state = {
  artifacts: {},
  questions: [],
  templateQuestions: [],
  risks: [],
  activeArtifact: "questions",
};

const $ = (id) => document.getElementById(id);

function fmtSize(bytes) {
  if (!bytes) return "-";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function pillClass(value) {
  const text = String(value || "").toUpperCase();
  if (text.includes("CRITICAL") || text.includes("HIGH")) return "pill danger";
  if (text.includes("MODERATE") || text.includes("MEDIUM") || text.includes("RUN")) return "pill warn";
  return "pill good";
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(body || `Request failed: ${res.status}`);
  }
  return res.json();
}

function switchView(name) {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.view === name);
  });
  document.querySelectorAll(".view").forEach((view) => {
    view.classList.toggle("active-view", view.id === name);
  });
}

function renderProject(data) {
  $("rootPath").textContent = data.root;
  const runtime = data.runtime || {};
  const packages = runtime.packages || {};
  const models = runtime.ollama_models || [];
  $("pythonRuntime").textContent = runtime.python || "-";
  $("spacyRuntime").textContent = packages.spacy ? "Installed" : "Fallback";
  $("embeddingRuntime").textContent = packages.sentence_transformers ? "Transformer ready" : "Hashing fallback";
  $("ollamaRuntime").textContent = runtime.ollama_reachable ? models.join(", ") : "Template fallback";

  const defaults = data.defaults || {};
  $("pitchInput").value = $("pitchInput").value || defaults.pitch || "";
  $("repoInput").value = $("repoInput").value || defaults.repo || "";
  $("patentsInput").value = $("patentsInput").value || defaults.patents || "";
  if (!$("modelInput").value && models.length) {
    $("modelInput").value = models.includes("phi3:latest") ? "phi3:latest" : models[0];
  }

  const summary = data.summary || {};
  const pitchStats = summary.pitch?.statistics || {};
  const graph = summary.graph || {};
  const risks = summary.risks || {};
  const questions = summary.questions || {};

  $("dealLine").textContent = summary.pitch?.source || "Current processed run";
  $("claimsMetric").textContent = pitchStats.technical_claims_extracted ?? "-";
  $("graphMetric").textContent = graph.nodes ? `${graph.nodes} / ${graph.edges}` : "-";
  $("risksMetric").textContent = risks.total_risks ?? "-";
  $("questionsMetric").textContent = questions.total_questions ?? "-";

  state.artifacts = data.artifacts || {};
  const entries = Object.entries(state.artifacts);
  $("artifactList").innerHTML = entries.map(([key, value]) => `
    <div class="artifact">
      <div>
        <strong>${value.file}</strong>
        <div class="meta">${key}</div>
      </div>
      <span class="${value.exists ? "pill good" : "pill"}">${value.exists ? fmtSize(value.size) : "Missing"}</span>
    </div>
  `).join("");

  $("artifactSelect").innerHTML = entries.map(([key, value]) => (
    `<option value="${key}">${value.file}</option>`
  )).join("");
  $("artifactSelect").value = state.activeArtifact;
}

function renderQuestions() {
  const sourceRows = state.questions.length ? state.questions : state.templateQuestions;
  const seen = new Set();
  const rows = sourceRows.filter((q) => {
    const key = q.chain_id || q.question || q.generated_question || JSON.stringify(q);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  if (!rows.length) {
    $("questionList").innerHTML = `<article class="item"><p>No generated questions found.</p></article>`;
    return;
  }
  $("questionList").innerHTML = rows.map((q, index) => {
    const text = q.question || q.generated_question || "";
    const category = q.question_category || q.risk_level || "question";
    const score = q.chain_score ?? q.risk_score ?? q.formal_confidence ?? "";
    return `
      <article class="item">
        <div class="item-head">
          <h3>Q${index + 1}. ${text}</h3>
          <span class="${pillClass(category)}">${category}</span>
        </div>
        <div class="meta">Score ${score || "-"} ${q.provider_used ? ` · ${q.provider_used}` : ""}</div>
      </article>
    `;
  }).join("");
}

function renderRisks() {
  if (!state.risks.length) {
    $("riskList").innerHTML = `<article class="item"><p>No risks found.</p></article>`;
    return;
  }
  $("riskList").innerHTML = state.risks.map((risk) => `
    <article class="item">
      <div class="item-head">
        <h3>${risk.category || "Risk"} · ${risk.target_entity || ""}</h3>
        <span class="${pillClass(risk.severity)}">${risk.severity || "UNKNOWN"}</span>
      </div>
      <p>${risk.recommended_action || "Review evidence."}</p>
      <div class="meta">Confidence ${risk.confidence_score ?? "-"} · ${risk.license_risk || "N/A"}</div>
    </article>
  `).join("");
}

async function loadQuestionsAndRisks() {
  const data = await api("/api/questions");
  state.questions = data.questions || [];
  state.templateQuestions = data.template_questions || [];
  state.risks = data.risks || [];
  renderQuestions();
  renderRisks();
}

async function loadArtifact(name = state.activeArtifact) {
  state.activeArtifact = name;
  const data = await api(`/api/artifact?name=${encodeURIComponent(name)}`);
  $("artifactPreview").textContent = JSON.stringify(data, null, 2);
}

async function refreshProject() {
  const data = await api("/api/project");
  renderProject(data);
  await loadQuestionsAndRisks();
  await loadArtifact(state.activeArtifact);
}

async function refreshJob() {
  const job = await api("/api/job");
  const running = job.running;
  $("jobState").textContent = running ? "Running" : "Idle";
  $("jobState").className = running ? "pill warn" : "pill";
  $("runButton").disabled = running;

  if (job.returncode === null) {
    $("exitCode").textContent = running ? "Running" : "No run";
    $("exitCode").className = running ? "pill warn" : "pill";
  } else {
    $("exitCode").textContent = `Exit ${job.returncode}`;
    $("exitCode").className = job.returncode === 0 ? "pill good" : "pill danger";
  }

  $("logOutput").textContent = (job.logs || []).join("\n");
  if (!running && job.finished_at) {
    await refreshProject();
  }
}

function payloadFromForm() {
  return {
    pitch: $("pitchInput").value.trim(),
    repo: $("repoInput").value.trim(),
    patents: $("patentsInput").value.trim(),
    start_stage: $("startStageInput").value,
    end_stage: $("endStageInput").value,
    max_questions: $("maxQuestionsInput").value,
    fusion_threshold: $("fusionInput").value,
    resolver_threshold: $("resolverInput").value,
    provider: $("providerInput").value,
    model: $("modelInput").value.trim(),
    dry_run: $("dryRunInput").checked,
  };
}

async function runPipeline() {
  $("runButton").disabled = true;
  await api("/api/run", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payloadFromForm()),
  });
  await refreshJob();
}

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => switchView(btn.dataset.view));
});

$("refreshButton").addEventListener("click", refreshProject);
$("runButton").addEventListener("click", runPipeline);
$("artifactSelect").addEventListener("change", (event) => loadArtifact(event.target.value));

refreshProject().catch((err) => {
  $("logOutput").textContent = err.message;
});
refreshJob();
setInterval(refreshJob, 1600);
