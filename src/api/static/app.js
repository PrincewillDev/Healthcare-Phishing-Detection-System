(function () {
  "use strict";

  let currentMode = "raw_text";

  const tabButtons = document.querySelectorAll(".tab-btn");
  const panels = document.querySelectorAll(".tab-panel");
  const emailText = document.getElementById("email-text");
  const emailFileInput = document.getElementById("email-file");
  const fileNameLabel = document.getElementById("file-name");
  const thresholdSlider = document.getElementById("threshold-slider");
  const thresholdValueLabel = document.getElementById("threshold-value");
  const classifyBtn = document.getElementById("classify-btn");
  const errorBox = document.getElementById("error-box");
  const resultPanel = document.getElementById("result-panel");
  const resultBanner = document.getElementById("result-banner");
  const resultLabel = document.getElementById("result-label");
  const ensembleScoreEl = document.getElementById("ensemble-score");
  const thresholdUsedEl = document.getElementById("threshold-used");
  const baseScoresEl = document.getElementById("base-scores");
  const responseTimeEl = document.getElementById("response-time");
  const shapBarsEl = document.getElementById("shap-bars");

  function setMode(mode) {
    currentMode = mode;
    tabButtons.forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.mode === mode);
    });
    panels.forEach((panel) => {
      panel.classList.toggle("hidden", panel.dataset.panel !== mode);
    });
    hideError();
  }

  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => setMode(btn.dataset.mode));
  });

  thresholdSlider.addEventListener("input", () => {
    thresholdValueLabel.textContent = Number(thresholdSlider.value).toFixed(2);
  });

  emailFileInput.addEventListener("change", () => {
    const file = emailFileInput.files[0];
    fileNameLabel.textContent = file ? file.name : "";
  });

  // Each click gets a token; if a newer "Load example" click starts before
  // an older one's fetch resolves, the older one's result is discarded when
  // it finally comes in instead of overwriting the file input with stale
  // content (the two fetches race independently, so completion order is not
  // guaranteed to match click order).
  let exampleLoadToken = 0;

  document.querySelectorAll(".example-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      hideError();
      const filename = btn.dataset.file;
      const myToken = ++exampleLoadToken;
      try {
        const resp = await fetch(`/samples/${filename}`, { cache: "no-store" });
        if (!resp.ok) {
          throw new Error("sample fetch failed");
        }
        const blob = await resp.blob();
        if (myToken !== exampleLoadToken) {
          return; // a newer example click superseded this one
        }
        const file = new File([blob], filename, { type: "message/rfc822" });
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        emailFileInput.files = dataTransfer.files;
        fileNameLabel.textContent = filename;
        setMode("eml_upload");
      } catch (err) {
        if (myToken === exampleLoadToken) {
          showError("Could not load the example file.");
        }
      }
    });
  });

  function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
  }

  function hideError() {
    errorBox.classList.add("hidden");
    errorBox.textContent = "";
  }

  function friendlyErrorFromDetail(status, detail) {
    if (status === 400) {
      if (typeof detail === "string" && detail.toLowerCase().includes("threshold")) {
        return "The threshold value is invalid. Please pick a value between 0 and 1.";
      }
      if (typeof detail === "string" && detail.toLowerCase().includes(".eml")) {
        return "Only .eml files are supported for upload.";
      }
      return "The request could not be processed. Please check your input and try again.";
    }
    return "Something went wrong while classifying this email. Please try again.";
  }

  function renderResult(data, elapsedMs) {
    resultPanel.classList.remove("hidden");

    const isPhishing = data.classification === "phishing";
    resultBanner.classList.remove("phishing", "legitimate");
    resultBanner.classList.add(isPhishing ? "phishing" : "legitimate");
    resultLabel.textContent = isPhishing ? "PHISHING" : "LEGITIMATE";

    ensembleScoreEl.textContent = data.ensemble_probability.toFixed(4);
    thresholdUsedEl.textContent = data.threshold_used.toFixed(2);

    baseScoresEl.innerHTML = "";
    const scores = data.base_model_scores;
    [
      ["Random Forest", scores.random_forest],
      ["XGBoost", scores.xgboost],
      ["LightGBM", scores.lightgbm],
    ].forEach(([label, value]) => {
      const li = document.createElement("li");
      const nameSpan = document.createElement("span");
      nameSpan.textContent = label;
      const valSpan = document.createElement("span");
      valSpan.textContent = value.toFixed(4);
      li.appendChild(nameSpan);
      li.appendChild(valSpan);
      baseScoresEl.appendChild(li);
    });

    responseTimeEl.textContent = `${elapsedMs.toFixed(0)} ms`;

    shapBarsEl.innerHTML = "";
    const rfFeatures = (data.explanation && data.explanation.random_forest) || [];
    const maxAbs = rfFeatures.reduce(
      (max, f) => Math.max(max, Math.abs(f.shap_value)),
      0
    ) || 1;

    rfFeatures.forEach((f) => {
      const row = document.createElement("div");
      row.className = "shap-row";

      const nameEl = document.createElement("div");
      nameEl.className = "shap-feature-name";
      nameEl.textContent = f.feature;
      nameEl.title = f.feature;

      const track = document.createElement("div");
      track.className = "shap-bar-track";
      const fill = document.createElement("div");
      fill.className = `shap-bar-fill ${f.direction}`;
      const widthPct = (Math.abs(f.shap_value) / maxAbs) * 100;
      fill.style.width = `${widthPct}%`;
      track.appendChild(fill);

      const valEl = document.createElement("div");
      valEl.className = "shap-value";
      valEl.textContent = f.shap_value.toFixed(4);

      row.appendChild(nameEl);
      row.appendChild(track);
      row.appendChild(valEl);
      shapBarsEl.appendChild(row);
    });
  }

  async function classify() {
    hideError();
    resultPanel.classList.add("hidden");

    const threshold = Number(thresholdSlider.value);
    classifyBtn.disabled = true;
    classifyBtn.textContent = "Classifying...";

    const start = performance.now();
    try {
      let response;
      if (currentMode === "raw_text") {
        const text = emailText.value.trim();
        if (!text) {
          showError("Please paste some email text first.");
          return;
        }
        response = await fetch("/classify", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email_text: text, threshold: threshold }),
        });
      } else {
        const file = emailFileInput.files[0];
        if (!file) {
          showError("Please choose a .eml file first.");
          return;
        }
        const formData = new FormData();
        formData.append("file", file);
        formData.append("threshold", String(threshold));
        response = await fetch("/classify", {
          method: "POST",
          body: formData,
        });
      }

      const elapsedMs = performance.now() - start;

      if (!response.ok) {
        let detail = "";
        try {
          const errJson = await response.json();
          detail = errJson.detail;
        } catch (e) {
          detail = "";
        }
        showError(friendlyErrorFromDetail(response.status, detail));
        return;
      }

      const data = await response.json();
      renderResult(data, elapsedMs);
    } catch (err) {
      showError("Could not reach the classification service. Is the API running?");
    } finally {
      classifyBtn.disabled = false;
      classifyBtn.textContent = "Classify";
    }
  }

  classifyBtn.addEventListener("click", classify);
})();
