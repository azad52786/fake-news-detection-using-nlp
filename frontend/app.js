const API_BASE = "http://127.0.0.1:8000/api/v1";
const checkBtn = document.getElementById("checkBtn");
const newsTitle = document.getElementById("newsTitle");
const newsText = document.getElementById("newsText");
const resultDiv = document.getElementById("result");
const historyDiv = document.getElementById("history");
const statusSpan = document.getElementById("status");
const detailModal = document.getElementById("detailModal");
const modalOverlay = document.getElementById("modalOverlay");
const closeModalBtn = document.getElementById("closeModalBtn");

checkBtn.addEventListener("click", handlePredict);
window.addEventListener("DOMContentLoaded", loadHistory);
window.addEventListener("DOMContentLoaded", restoreLastResult);
closeModalBtn.addEventListener("click", closeModal);
modalOverlay.addEventListener("click", closeModal);

async function handlePredict() {
    const title = newsTitle.value.trim();
    const content = newsText.value.trim();
    if (!content) {
        renderError("Please enter some news content.");
        return;
    }

    setLoading(true);
    try {
        const res = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title, content })
        });

        if (!res.ok) {
            const msg = (await res.json())?.detail || "Server error";
            throw new Error(msg);
        }

        const data = await res.json();
        renderResult(data);
        prependHistory(data, content, title);
        persistLastResult(data, content, title);
    } catch (err) {
        renderError(err.message || "Something went wrong.");
    } finally {
        setLoading(false);
    }
}

async function loadHistory() {
    historyDiv.innerHTML = "<span class='muted'>Loading history...</span>";
    try {
        const res = await fetch(`${API_BASE}/history?limit=20`);
        if (!res.ok) throw new Error("Failed to load history");
        const data = await res.json();
        renderHistory(data.items || []);
    } catch (err) {
        historyDiv.innerHTML = `<span class='error'>${err.message}</span>`;
    }
}

function setLoading(isLoading) {
    checkBtn.disabled = isLoading;
    checkBtn.textContent = isLoading ? "Running..." : "Run Prediction";
    statusSpan.textContent = isLoading ? "Calling model..." : "";
}

function renderResult(data) {
    const label = data.label;
    const probPct = (data.probability * 100).toFixed(2);
    const badgeClass = label === "REAL" ? "real" : "fake";
    const tokens = data.top_tokens || [];

    resultDiv.style.display = "block";
    resultDiv.innerHTML = `
        <div class="badge ${badgeClass}">${label}</div>
        <div class="prob">${probPct}%</div>
        <div class="muted">Model ${data.model_version} · ${new Date(data.created_at).toLocaleString()}</div>
        ${tokens.length ? `<div class="tokens">${tokens.map(t => `<span class="token">${t}</span>`).join("")}</div>` : ""}
    `;
}

function renderError(message) {
    resultDiv.style.display = "block";
    resultDiv.innerHTML = `<span class="error">${message}</span>`;
}

function renderHistory(items) {
    if (!items.length) {
        historyDiv.innerHTML = "<span class='muted'>No history yet.</span>";
        return;
    }
    historyDiv.innerHTML = items
        .map((item) => {
            const badgeClass = item.label === "REAL" ? "real" : "fake";
            const snippet = (item.content || "").slice(0, 140).trim();
            const title = item.title && item.title.trim() ? item.title : "Untitled";
            return `
                <div class="history-item" onclick='openModal(${JSON.stringify(item)})' style="cursor: pointer;">
                    <div class="history-header">
                        <div class="history-title">${title}</div>
                        <span class="badge ${badgeClass}">${item.label}</span>
                    </div>
                    <div class="history-snippet">${snippet}${snippet.length === 140 ? "..." : ""}</div>
                    <div class="history-meta">
                        <span>${(item.probability * 100).toFixed(1)}%</span>
                        <span>·</span>
                        <span>${new Date(item.created_at).toLocaleString()}</span>
                    </div>
                </div>
            `;
        })
        .join("");
}

function prependHistory(item, content, title) {
    const existing = Array.from(historyDiv.querySelectorAll('.history-item'));
    const newItem = document.createElement('div');
    const badgeClass = item.label === "REAL" ? "real" : "fake";
    const snippet = content.slice(0, 140).trim();
    newItem.className = 'history-item';
    newItem.innerHTML = `
        <div class="history-header">
            <div class="history-title">${title || 'Untitled'}</div>
            <span class="badge ${badgeClass}">${item.label}</span>
        </div>
        <div class="history-snippet">${snippet}${snippet.length === 140 ? '...' : ''}</div>
        <div class="history-meta">
            <span>${(item.probability * 100).toFixed(1)}%</span>
            <span>·</span>
            <span>${new Date(item.created_at).toLocaleString()}</span>
        </div>
    `;
    historyDiv.prepend(newItem);
    if (existing.length > 0) {
        while (historyDiv.children.length > 20) {
            historyDiv.removeChild(historyDiv.lastChild);
        }
    }
}

function persistLastResult(item, content, title) {
    const payload = {
        item,
        content,
        title,
    };
    try {
        localStorage.setItem("lastPrediction", JSON.stringify(payload));
    } catch (e) {
        console.warn("Could not persist last prediction", e);
    }
}

function restoreLastResult() {
    try {
        const raw = localStorage.getItem("lastPrediction");
        if (!raw) return;
        const parsed = JSON.parse(raw);
        if (parsed?.item) {
            renderResult(parsed.item);
        }
    } catch (e) {
        console.warn("Could not restore last prediction", e);
    }
}

function openModal(item) {
    const title = item.title && item.title.trim() ? item.title : "Untitled";
    const badgeClass = item.label === "REAL" ? "real" : "fake";
    const probPct = (item.probability * 100).toFixed(2);
    
    document.getElementById("modalItemTitle").textContent = title;
    document.getElementById("modalItemLabel").className = `badge ${badgeClass}`;
    document.getElementById("modalItemLabel").textContent = item.label;
    document.getElementById("modalItemProb").textContent = `${probPct}%`;
    document.getElementById("modalItemContent").textContent = item.content || "N/A";
    document.getElementById("modalModelVersion").textContent = item.model_version || "N/A";
    document.getElementById("modalCreatedAt").textContent = new Date(item.created_at).toLocaleString();
    
    detailModal.style.display = "flex";
}

function closeModal() {
    detailModal.style.display = "none";
}
