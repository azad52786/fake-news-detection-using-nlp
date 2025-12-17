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
const themeToggle = document.getElementById("themeToggle");

checkBtn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    handlePredict();
});
console.log("✓ Button listener attached");

window.addEventListener("DOMContentLoaded", loadHistory);
console.log("✓ History listener attached");
window.addEventListener("DOMContentLoaded", initTheme);
console.log("✓ Theme listener attached");
if (themeToggle) themeToggle.addEventListener("click", toggleTheme);
closeModalBtn.addEventListener("click", closeModal);
modalOverlay.addEventListener("click", closeModal);

function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    if (themeToggle) themeToggle.textContent = theme === "light" ? "Dark" : "Light";
}

function initTheme() {
    let theme = localStorage.getItem("theme");
    if (!theme) {
        theme = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    applyTheme(theme);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    const next = current === "dark" ? "light" : "dark";
    applyTheme(next);
    try { localStorage.setItem("theme", next); } catch {}
}

async function handlePredict() {
    const title = newsTitle.value.trim();
    const content = newsText.value.trim();
    console.log("handlePredict called", { title, content });
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
        
        console.log(res);
        // return;
        console.log("Response status:", res.status);
        if (!res.ok) {
            const msg = (await res.json())?.detail || "Server error";
            throw new Error(msg);
        }

        const data = await res.json();
        console.log("Prediction result:", data);
        renderResult(data);
        // await loadHistory();
        prependHistory(data, content, title);
    } catch (err) {
        console.error("Error in handlePredict:", err);
        renderError(err.message || "Something went wrong.");
    } finally {
        setLoading(false);
    }
}

async function loadHistory() {
    console.log("loadHistory called");
    historyDiv.innerHTML = "<span class='muted'>Loading history...</span>";
    try {
        console.log("Calling API:", `${API_BASE}/history?limit=20`);
        const res = await fetch(`${API_BASE}/history?limit=20`);
        console.log("History response status:", res.status);
        if (!res.ok) throw new Error("Failed to load history");
        const data = await res.json();
        console.log("History loaded:", data);
        renderHistory(data.items || []);
    } catch (err) {
        console.error("Error loading history:", err);
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
    const modalPayload = {
        ...item,
        content,
        title: title && title.trim() ? title : 'Untitled'
    };
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
    newItem.style.cursor = 'pointer';
    newItem.addEventListener('click', () => openModal(modalPayload));
    historyDiv.prepend(newItem);
    if (existing.length > 0) {
        while (historyDiv.children.length > 20) {
            historyDiv.removeChild(historyDiv.lastChild);
        }
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
