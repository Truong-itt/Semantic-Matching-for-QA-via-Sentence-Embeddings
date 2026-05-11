/* ================================================================
   main.js — Sentence Selection QA Web Demo
   ================================================================ */

"use strict";

// ── Tab navigation ────────────────────────────────────────────────
document.querySelectorAll(".nav-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.tab;
    document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(t => t.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${target}`).classList.add("active");
  });
});

// ── Example data (fetched once) ───────────────────────────────────
let EXAMPLES = [];
fetch("/api/examples")
  .then(r => r.json())
  .then(data => { EXAMPLES = data; });

// ── Demo tab: example chips ───────────────────────────────────────
document.querySelectorAll("#example-chips .chip").forEach(chip => {
  chip.addEventListener("click", () => {
    const idx = parseInt(chip.dataset.idx);
    document.querySelectorAll("#example-chips .chip").forEach(c => c.classList.remove("active"));
    chip.classList.add("active");
    document.getElementById("question").value = EXAMPLES[idx].question;
    document.getElementById("context").value  = EXAMPLES[idx].context;
  });
});

// ── Compare tab: example chips ────────────────────────────────────
document.querySelectorAll("#cmp-example-chips .chip").forEach(chip => {
  chip.addEventListener("click", () => {
    const idx = parseInt(chip.dataset.idx);
    document.querySelectorAll("#cmp-example-chips .chip").forEach(c => c.classList.remove("active"));
    chip.classList.add("active");
    document.getElementById("cmp-question").value = EXAMPLES[idx].question;
    document.getElementById("cmp-context").value  = EXAMPLES[idx].context;
  });
});

// ── Helper: toggle button loading state ──────────────────────────
function setLoading(btn, loading) {
  btn.disabled = loading;
  btn.querySelector(".btn-text").classList.toggle("hidden", loading);
  btn.querySelector(".btn-loader").classList.toggle("hidden", !loading);
}

// ── Helper: build a sentence row HTML ────────────────────────────
function buildSentenceRow(idx, text, normScore, rawScore, isPredicted, colorClass) {
  const widthPct = Math.max(4, Math.round(normScore * 100));
  const predTag  = isPredicted
    ? `<span class="predicted-tag">✅ Selected</span>`
    : "";
  const barColor = colorClass === "euclid" ? "euclid-score" : "";
  return `
    <div class="sentence-row ${isPredicted ? (colorClass === "euclid" ? "euclid-pred" : "is-predicted") : ""}">
      <div class="sentence-meta">
        <span class="sent-idx">${idx + 1}</span>
        <span class="sent-text">${escapeHtml(text)}</span>
        ${predTag}
      </div>
      <div class="score-bar-wrap">
        <div class="score-bar-bg">
          <div class="score-bar-fill ${barColor}" style="width: ${widthPct}%"></div>
        </div>
        <span class="score-num">${rawScore.toFixed(4)}</span>
      </div>
    </div>`;
}

function escapeHtml(str) {
  return str.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

// ── DEMO TAB — Predict ────────────────────────────────────────────
document.getElementById("btn-predict").addEventListener("click", async () => {
  const question = document.getElementById("question").value.trim();
  const context  = document.getElementById("context").value.trim();
  const metric   = document.querySelector('input[name="metric"]:checked').value;
  const btn = document.getElementById("btn-predict");

  if (!question) { alert("Please enter a question."); return; }
  if (!context)  { alert("Please enter a context paragraph."); return; }

  setLoading(btn, true);
  const placeholder = document.getElementById("result-placeholder");
  const content     = document.getElementById("result-content");
  const errDiv      = document.getElementById("result-error");

  placeholder.classList.add("hidden");
  content.classList.add("hidden");
  errDiv.classList.add("hidden");

  try {
    const res  = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, context, metric }),
    });
    const data = await res.json();

    if (data.error) {
      errDiv.textContent = "⚠️ " + data.error;
      errDiv.classList.remove("hidden");
      return;
    }

    // ── Populate result UI ────────────────────────────────────────
    const pred = data.predicted;
    document.getElementById("answer-text").textContent  = data.sentences[pred];
    document.getElementById("answer-score").textContent =
      `Score: ${data.raw_scores[pred].toFixed(6)}  (${metric})`;

    const badge = document.getElementById("result-metric-badge");
    badge.textContent = metric.charAt(0).toUpperCase() + metric.slice(1);
    badge.classList.remove("hidden");

    let html = "";
    data.sentences.forEach((s, i) => {
      html += buildSentenceRow(
        i, s,
        data.norm_scores[i],
        data.raw_scores[i],
        i === pred,
        "cosine"
      );
    });
    document.getElementById("sentences-list").innerHTML = html;

    content.classList.remove("hidden");
  } catch (e) {
    errDiv.textContent = "⚠️ Network error: " + e.message;
    errDiv.classList.remove("hidden");
  } finally {
    setLoading(btn, false);
  }
});

// ── Also allow Enter key in question input ────────────────────────
document.getElementById("question").addEventListener("keydown", e => {
  if (e.key === "Enter") document.getElementById("btn-predict").click();
});

// ── COMPARE TAB — Compare both metrics ───────────────────────────
document.getElementById("btn-compare").addEventListener("click", async () => {
  const question = document.getElementById("cmp-question").value.trim();
  const context  = document.getElementById("cmp-context").value.trim();
  const btn      = document.getElementById("btn-compare");

  if (!question) { alert("Please enter a question."); return; }
  if (!context)  { alert("Please enter a context paragraph."); return; }

  setLoading(btn, true);
  document.getElementById("compare-results").classList.add("hidden");

  try {
    const res  = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, context }),
    });
    const data = await res.json();

    if (data.error) { alert(data.error); return; }

    // ── Cosine results ────────────────────────────────────────────
    const cp = data.cosine.predicted;
    document.getElementById("cmp-cosine-text").textContent = data.cosine.sentences[cp];
    let cHtml = "";
    data.cosine.sentences.forEach((s, i) => {
      cHtml += buildSentenceRow(
        i, s,
        data.cosine.norm_scores[i],
        data.cosine.raw_scores[i],
        i === cp, "cosine"
      );
    });
    document.getElementById("cmp-cosine-list").innerHTML = cHtml;

    // ── Euclidean results ──────────────────────────────────────────
    const ep = data.euclidean.predicted;
    document.getElementById("cmp-euclidean-text").textContent = data.euclidean.sentences[ep];
    let eHtml = "";
    data.euclidean.sentences.forEach((s, i) => {
      eHtml += buildSentenceRow(
        i, s,
        data.euclidean.norm_scores[i],
        data.euclidean.raw_scores[i],
        i === ep, "euclid"
      );
    });
    document.getElementById("cmp-euclidean-list").innerHTML = eHtml;

    // ── Agreement indicator ────────────────────────────────────────
    const agreeBox = document.getElementById("agreement-box");
    if (cp === ep) {
      agreeBox.className = "agreement-box agree";
      agreeBox.textContent = `✅ Both metrics agree — Selected sentence [${cp + 1}]: "${data.cosine.sentences[cp].substring(0, 80)}…"`;
    } else {
      agreeBox.className = "agreement-box disagree";
      agreeBox.textContent =
        `⚠️ Metrics disagree — Cosine selects [${cp + 1}], Euclidean selects [${ep + 1}]. ` +
        `Cosine is generally more reliable for variable-length texts.`;
    }

    document.getElementById("compare-results").classList.remove("hidden");
  } catch (e) {
    alert("Network error: " + e.message);
  } finally {
    setLoading(btn, false);
  }
});

document.getElementById("cmp-question").addEventListener("keydown", e => {
  if (e.key === "Enter") document.getElementById("btn-compare").click();
});

// ── Auto-load first example on page load ─────────────────────────
window.addEventListener("load", () => {
  // Small delay to allow EXAMPLES to be fetched
  setTimeout(() => {
    if (EXAMPLES.length > 0) {
      document.getElementById("question").value = EXAMPLES[0].question;
      document.getElementById("context").value  = EXAMPLES[0].context;
      document.querySelectorAll("#example-chips .chip")[0]?.classList.add("active");
    }
  }, 400);
});
