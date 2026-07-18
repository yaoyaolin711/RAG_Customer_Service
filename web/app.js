(() => {
  "use strict";

  const WELCOME =
    "亲你好，我是店铺智能客服，商品、发货、售后有问题直接问我就行，我会尽快回复你哈。";

  const QUICK_BY_SCENARIO = {
    "咨询类 · RAG": [
      "这款有什么规格？适合什么人群？",
      "有运费险吗？支持七天无理由吗？",
      "大概多久发货？包邮吗？",
    ],
    "交易类": ["我的订单什么时候发货？", "怎么退款？"],
    "投诉类": ["客服态度太差了我要投诉", "这个东西质量有问题"],
    "其他类 · 闲聊": ["今天天气怎么样？", "你吃饭了吗？"],
  };

  const REPLY_MODE_LABEL = {
    rag: "RAG 知识库",
    cache: "缓存命中",
    no_hit: "未命中兜底",
    casual: "闲聊",
    transaction: "交易查询",
    handoff: "转人工",
  };

  const els = {
    messages: document.getElementById("messages"),
    chatForm: document.getElementById("chatForm"),
    messageInput: document.getElementById("messageInput"),
    sendBtn: document.getElementById("sendBtn"),
    userId: document.getElementById("userId"),
    buyerName: document.getElementById("buyerName"),
    sessionKey: document.getElementById("sessionKey"),
    clearBtn: document.getElementById("clearBtn"),
    quickList: document.getElementById("quickList"),
    healthPill: document.getElementById("healthPill"),
    healthDot: document.getElementById("healthDot"),
    healthLabel: document.getElementById("healthLabel"),
    metaChip: document.getElementById("metaChip"),
    chatSub: document.getElementById("chatSub"),
    toast: document.getElementById("toast"),
    pipelineEmpty: document.getElementById("pipelineEmpty"),
    pipelineBody: document.getElementById("pipelineBody"),
    rightRail: document.getElementById("rightRail"),
    togglePanelBtn: document.getElementById("togglePanelBtn"),
    closePanelBtn: document.getElementById("closePanelBtn"),
    drawerMask: document.getElementById("drawerMask"),
  };

  let busy = false;
  let typingEl = null;

  function resolveApiBase() {
    const params = new URLSearchParams(window.location.search);
    const fromQuery = params.get("api");
    if (fromQuery) {
      localStorage.setItem("rag_api_base", fromQuery.replace(/\/$/, ""));
    }
    const stored = localStorage.getItem("rag_api_base");
    if (stored) return stored.replace(/\/$/, "");
    return "";
  }

  const API_BASE = resolveApiBase();

  function apiUrl(path) {
    return `${API_BASE}${path}`;
  }

  function newSessionKey() {
    const stamp = Date.now().toString(36);
    return `dy_session_${stamp}`;
  }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function showToast(msg) {
    els.toast.hidden = !msg;
    els.toast.textContent = msg || "";
  }

  function autoGrow() {
    const ta = els.messageInput;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 120)}px`;
  }

  function scrollToBottom() {
    els.messages.scrollTop = els.messages.scrollHeight;
  }

  function appendMessage(role, content, meta) {
    const row = document.createElement("div");
    row.className = `msg ${role === "user" ? "user" : "bot"}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "买" : "客";

    const wrap = document.createElement("div");
    wrap.className = "bubble-wrap";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = escapeHtml(content).replace(/\n/g, "<br>");
    wrap.appendChild(bubble);

    if (meta && role === "assistant") {
      const tags = document.createElement("div");
      tags.className = "tags";
      const mode = meta.reply_mode || "";
      if (mode) {
        const t = document.createElement("span");
        t.className = `tag ${mode}`;
        t.textContent = REPLY_MODE_LABEL[mode] || mode;
        tags.appendChild(t);
      }
      if (typeof meta.rag_hit === "boolean") {
        const t = document.createElement("span");
        t.className = `tag ${meta.rag_hit ? "hit" : "miss"}`;
        t.textContent = meta.rag_hit ? "知识库命中" : "未命中";
        tags.appendChild(t);
      }
      if (meta.route) {
        const t = document.createElement("span");
        t.className = "tag";
        t.textContent = meta.route;
        tags.appendChild(t);
      }
      if (tags.childNodes.length) wrap.appendChild(tags);
    }

    row.appendChild(avatar);
    row.appendChild(wrap);
    els.messages.appendChild(row);
    scrollToBottom();
    return row;
  }

  function showTyping() {
    hideTyping();
    const row = document.createElement("div");
    row.className = "msg bot";
    row.id = "typingRow";
    row.innerHTML =
      '<div class="avatar">客</div><div class="bubble-wrap"><div class="bubble typing"><i></i><i></i><i></i></div></div>';
    els.messages.appendChild(row);
    typingEl = row;
    scrollToBottom();
  }

  function hideTyping() {
    if (typingEl) {
      typingEl.remove();
      typingEl = null;
    }
  }

  function setBusy(on) {
    busy = on;
    els.sendBtn.disabled = on;
    els.messageInput.disabled = on;
    document.querySelectorAll(".quick-btn").forEach((b) => {
      b.disabled = on;
    });
  }

  function renderQuick() {
    els.quickList.innerHTML = "";
    Object.entries(QUICK_BY_SCENARIO).forEach(([title, questions]) => {
      const group = document.createElement("div");
      group.className = "quick-group";
      const h = document.createElement("h3");
      h.textContent = title;
      group.appendChild(h);
      questions.forEach((q) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "quick-btn";
        btn.textContent = q;
        btn.addEventListener("click", () => sendMessage(q));
        group.appendChild(btn);
      });
      els.quickList.appendChild(group);
    });
  }

  function renderPipeline(data) {
    if (!data) {
      els.pipelineEmpty.hidden = false;
      els.pipelineBody.hidden = true;
      els.pipelineBody.innerHTML = "";
      return;
    }

    els.pipelineEmpty.hidden = true;
    els.pipelineBody.hidden = false;

    const sources = Array.isArray(data.sources) ? data.sources : [];
    const modeLabel = REPLY_MODE_LABEL[data.reply_mode] || data.reply_mode || "—";

    const sourceHtml = sources.length
      ? sources
          .map((s, i) => {
            const score = typeof s.score === "number" ? s.score : null;
            const low = score != null && score < 0.45;
            const title = [
              `片段 ${i + 1}`,
              s.source || "",
              s.section || "",
              score != null ? `score=${score.toFixed(2)}` : "",
            ]
              .filter(Boolean)
              .join(" · ");
            return `<div class="source-card ${low ? "low" : ""}">
              <div class="source-title">${escapeHtml(title)}</div>
              <div>${escapeHtml(s.content || "").replace(/\n/g, "<br>")}</div>
            </div>`;
          })
          .join("")
      : `<div class="empty-state">${
          data.reply_mode === "no_hit" || data.rag_hit === false
            ? "知识库未命中：不会编造产品/政策细节。"
            : "本轮无召回片段。"
        }</div>`;

    els.pipelineBody.innerHTML = `
      <div class="stat-grid">
        <div class="stat-card"><div class="label">路由</div><div class="value">${escapeHtml(data.route || "—")}</div></div>
        <div class="stat-card"><div class="label">回复模式</div><div class="value">${escapeHtml(modeLabel)}</div></div>
        <div class="stat-card"><div class="label">知识库</div><div class="value">${data.rag_hit ? "命中" : "未命中"}</div></div>
        <div class="stat-card"><div class="label">历史条数</div><div class="value">${data.history_count ?? 0}</div></div>
      </div>
      <div>
        <div class="panel-title">召回片段（${sources.length}）</div>
        ${sourceHtml}
      </div>
    `;
  }

  async function refreshHealth() {
    try {
      const res = await fetch(apiUrl("/api/v1/health"));
      const json = await res.json();
      const ok = json?.state?.code === 0 && json?.data?.status === "ok";
      const degraded = json?.state?.code === 0 && json?.data?.status === "degraded";
      els.healthPill.classList.toggle("ok", ok || degraded);
      els.healthPill.classList.toggle("bad", !ok && !degraded);
      if (ok) {
        els.healthLabel.textContent = "服务正常";
      } else if (degraded) {
        els.healthLabel.textContent = "服务降级";
      } else {
        els.healthLabel.textContent = json?.state?.message || "服务异常";
      }
    } catch {
      els.healthPill.classList.remove("ok");
      els.healthPill.classList.add("bad");
      els.healthLabel.textContent = "无法连接 API";
    }
  }

  async function refreshMeta() {
    try {
      const res = await fetch(apiUrl("/api/v1/meta"));
      const json = await res.json();
      if (json?.state?.code !== 0) throw new Error(json?.state?.message || "meta failed");
      const d = json.data || {};
      const kb = d.kb_doc || "知识库";
      const col = d.collection || "";
      els.metaChip.textContent = `${kb}${col ? " · " + col : ""} · 阈值 ${d.relevance_threshold ?? "—"}`;
      els.chatSub.textContent = `在线 · ${d.llm_model || "LLM"} · 目标约 15 秒内回复`;
    } catch {
      els.metaChip.textContent = "元信息不可用";
    }
  }

  async function sendMessage(raw) {
    const message = (raw ?? els.messageInput.value ?? "").trim();
    if (!message) {
      showToast("消息不能为空，请输入内容后再发送。");
      return;
    }
    if (busy) return;

    showToast("");
    appendMessage("user", message);
    els.messageInput.value = "";
    autoGrow();
    setBusy(true);
    showTyping();

    const payload = {
      message,
      user_id: els.userId.value.trim() || "buyer_demo_001",
      buyer_name: els.buyerName.value.trim() || undefined,
      session_key: els.sessionKey.value.trim() || undefined,
    };

    try {
      const res = await fetch(apiUrl("/api/v1/chat"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const json = await res.json();
      hideTyping();

      if (json?.state?.code !== 0) {
        const err = json?.state?.message || "请求失败";
        showToast(err);
        appendMessage("assistant", `抱歉，处理失败：${err}`, null);
        renderPipeline(null);
        return;
      }

      const data = json.data || {};
      const answer = data.answer || data.reply?.message || "（空回复）";
      appendMessage("assistant", answer, {
        reply_mode: data.reply_mode,
        rag_hit: data.rag_hit,
        route: data.route,
      });
      renderPipeline(data);
    } catch (e) {
      hideTyping();
      const msg = e?.message || String(e);
      showToast(`连接失败：${msg}`);
      appendMessage("assistant", `抱歉，网络或服务异常：${msg}`, null);
    } finally {
      setBusy(false);
      els.messageInput.focus();
    }
  }

  function clearChat() {
    els.messages.innerHTML = "";
    appendMessage("assistant", WELCOME, null);
    renderPipeline(null);
    showToast("");
    els.sessionKey.value = newSessionKey();
  }

  function openPanel() {
    els.rightRail.classList.add("open");
    els.drawerMask.hidden = false;
  }

  function closePanel() {
    els.rightRail.classList.remove("open");
    els.drawerMask.hidden = true;
  }

  function init() {
    renderQuick();
    clearChat();

    els.chatForm.addEventListener("submit", (e) => {
      e.preventDefault();
      sendMessage();
    });

    els.messageInput.addEventListener("input", autoGrow);
    els.messageInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    els.clearBtn.addEventListener("click", clearChat);
    els.togglePanelBtn.addEventListener("click", openPanel);
    els.closePanelBtn.addEventListener("click", closePanel);
    els.drawerMask.addEventListener("click", closePanel);

    refreshHealth();
    refreshMeta();
    setInterval(refreshHealth, 30000);
  }

  init();
})();
