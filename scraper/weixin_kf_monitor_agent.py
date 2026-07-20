"""
微信小店客服消息自动化监控与 AI 回复 - Playwright Agent

设计思路（API 拦截驱动）：
  1. 启动浏览器 → 加载登录态 → 进入 微信小店客服 页面
  2. 通过 page.on('response') 监听消息 API 的返回
  3. 自适应解析响应 JSON 中消息结构
  4. 发现新消息 → LLM 生成回复 → 填入 #im-input-box → 发送
  5. 通过 ReplyHistory 去重，避免反复回复

使用流程:
  # 第 1 步：分析 API（登录后等消息到达，自动推荐最优 endpoint）
  python -m scraper.weixin_kf_monitor_agent --store zhihuai --analyze

  # 第 2 步：启动监控
   python -m scraper.weixin_kf_monitor_agent --store zhihuai --endpoint "/shop/commkf/msg"
"""

import json
import os
import re
import time
import random
import logging
import threading
import concurrent.futures
from datetime import datetime
from typing import Optional, Any

from dotenv import load_dotenv

from scraper.reply_history import ReplyHistory

logger = logging.getLogger(__name__)

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# ============================================================
# URL 常量
# ============================================================

KF_URL = "https://store.weixin.qq.com/shop/kf"
HOME_URL = "https://store.weixin.qq.com"
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ============================================================
# API 关键字评分 — 用于 --analyze 自动发现消息 API
# ============================================================

API_URL_KEYWORDS = {
    "kf": 3, "commkf": 3, "msg": 3, "message": 3,
    "chat": 3, "conversation": 3, "session": 3,
    "poll": 2, "sync": 2, "push": 2, "notice": 2,
    "list": 1, "new": 1, "unread": 3,
    "send": 2, "receive": 2,
}

API_JSON_KEYS = {
    "content": 2, "msg_kf_content": 3, "msg": 2, "message": 2,
    "from": 2, "role": 2, "sender": 2,
    "timestamp": 1, "create_time": 1, "send_time": 1,
    "kf": 2, "customer": 2, "visitor": 2, "user": 1,
    "room_id": 3, "session_id": 2,
    "nickname": 1, "name": 1, "extra_info": 2,
    "msg_direction": 3, "msg_type": 1,
}

# 消息相关 API 路径特征（正则，用于拦截匹配 + 默认 endpoint 回退）
API_MESSAGE_PATTERNS = [
    r"/shop/commkf/msg",
    r"/api/.*(?:message|msg|chat|conversation|session|kf)",
    r"/kf/?(?:poll|sync|list|send|receive)",
]

# ============================================================
# JS 脚本 — MutationObserver：监听左侧列表新会话
# ============================================================

INJECT_SIDEBAR_OBSERVER_SCRIPT = """
() => {
    if (window.__kfObserverActive) return;
    window.__kfObserverActive = true;
    window.__kfPendingConversations = [];
    window.__kfObserverState = {};

    function pushPending(name, roomId, preview) {
        var found = window.__kfPendingConversations.find(function(p) { return p.name === name && p.roomId === roomId; });
        if (!found) {
            window.__kfPendingConversations.push({ name: name, roomId: roomId, preview: preview });
        }
    }

    setInterval(function() {
        var items = document.querySelectorAll('li.session-item-container');
        items.forEach(function(item) {
            var nickEl = item.querySelector('.user-nickname');
            if (!nickEl) return;
            var name = nickEl.textContent.trim();
            if (!name) return;

            var roomId = item.getAttribute('data-room-id') || '';
            var key = name + '|' + roomId;

            var badge = item.querySelector('.unread-badge');
            var hasUnread = badge && badge.offsetParent !== null;
            var badgeText = hasUnread ? (badge.textContent || '').trim() : '';

            var previewEl = item.querySelector('.text-content-wrap');
            var preview = previewEl ? (previewEl.innerText || previewEl.textContent || '') : '';
            preview = preview.trim();

            var prev = window.__kfObserverState[key];

            if (!prev) {
                window.__kfObserverState[key] = { name: name, roomId: roomId, badge: badgeText };
                if (hasUnread) {
                    console.log('[KF_OBSERVER] first load unread:', name);
                    pushPending(name, roomId, preview);
                }
                return;
            }

            // Only trigger on unread badge changes (new customer message)
            if (hasUnread && prev.badge !== badgeText) {
                console.log('[KF_OBSERVER] new msg via badge:', name);
                pushPending(name, roomId, preview);
            }

            prev.badge = badgeText;
        });
    }, 300);
}
"""

# ============================================================
# JS 脚本 — 点击指定会话
# ============================================================

CLICK_CONTACT_SCRIPT = """
(name) => {
    var items = document.querySelectorAll('li.session-item-container');
    for (var i = 0; i < items.length; i++) {
        var nickEl = items[i].querySelector('.user-nickname');
        if (nickEl && nickEl.textContent.trim() === name) {
            items[i].querySelector('.session-list-card').click();
            return true;
        }
    }
    var el = document.querySelector('[title="' + name + '"], [aria-label="' + name + '"]');
    if (el) { el.click(); return true; }
    return false;
}
"""

# ============================================================
# JS 脚本 — 提取聊天消息（仅用于兜底验证）
# ============================================================

EXTRACT_MESSAGES_SCRIPT = """
() => {
    var results = [];
    var items = document.querySelectorAll('.chat_content_has_room .message-item');
    if (items.length === 0) return results;

    for (var i = 0; i < items.length; i++) {
        var item = items[i];
        var cls = item.className || '';

        var isMe = cls.indexOf('justify-end') !== -1;
        var isOther = cls.indexOf('justify-start') !== -1;
        if (!isMe && !isOther) {
            if (item.querySelector('.message-time.right')) isMe = true;
            else isOther = true;
        }

        var text = '';
        var msgType = 'text';

        // 1. 语音消息优先检测
        var voiceParent = item.closest('[data-type="10"]') || item;
        var voiceEl = voiceParent.querySelector('[data-type="voice"]');
        if (voiceEl) {
            var transcribeEl = voiceEl.querySelector('[class*="whitespace-pre-wrap"]');
            var transcribeText = transcribeEl ? transcribeEl.textContent.trim() : '';
            if (transcribeText) {
                text = '[语音]' + transcribeText;
                msgType = 'voice';
            } else {
                text = '[语音]';
                msgType = 'voice';
            }
        }

        // 2. 纯文本消息
        if (!text) {
            var textEl = item.querySelector('.text-msg span');
            if (textEl) text = textEl.textContent.trim();
        }

        // 3. 视频/图片（跳过语音消息的图标）
        if (!text) {
            var hasVideo = item.querySelector('[data-type="video"]') && !voiceParent.querySelector('[data-type="voice"]');
            if (hasVideo) {
                text = '[视频]';
            } else if (item.querySelector('.item-img img, img[alt]')) {
                text = '[图片]';
            } else {
                continue;
            }
        }

        results.push({
            sender: isMe ? 'me' : 'other',
            content: text.substring(0, 500),
            timestamp: '',
            msg_type: msgType,
        });
    }
    return results;
}
"""

# ============================================================
# JS 脚本 — 发送消息（#im-input-box）
# ============================================================

FILL_INPUT_SCRIPT = """
(text) => {
    // WeChat store KF input: textarea#input-textarea.text-area
    var input = document.querySelector('#input-textarea');
    if (!input) input = document.querySelector('textarea.text-area');
    if (!input) input = document.querySelector('textarea');
    if (!input) return 'no_input';

    var nativeSetter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype, 'value'
    ).set;
    nativeSetter.call(input, text);
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
    return 'textarea';
}
"""

CLICK_SEND_SCRIPT = """
() => {
    // WeChat store KF: press Enter on textarea
    var input = document.querySelector('#input-textarea');
    if (!input) input = document.querySelector('textarea.text-area');
    if (!input) input = document.querySelector('textarea');
    if (input) {
        input.dispatchEvent(new KeyboardEvent('keydown', {
            key: 'Enter', code: 'Enter', keyCode: 13, which: 13,
            bubbles: true, cancelable: true,
        }));
        return 'enter';
    }
    return 'no_enter';
}
"""

GET_SELECTED_CONTACT_SCRIPT = """
() => {
    const selected = document.querySelector(
        '[class*="contactCard"][class*="selected"], ' +
        '[class*="contactCard"][class*="active"], ' +
        '[class*="selected"], [class*="active"], [aria-selected="true"]'
    );
    if (selected) {
        const nameEl = selected.querySelector('[class*="uname"], [class*="name"], [class*="title"]');
        const img = selected.querySelector('img');
        return {
            name: nameEl ? (nameEl.textContent || '').trim() : '',
            avatar: img ? (img.getAttribute('src') || '') : '',
        };
    }
    return null;
}
"""

ANALYZE_DOM_SCRIPT = """
() => {
    const info = {
        url: window.location.href,
        title: document.title,
        textLength: document.body.innerText.length,
        totalElements: document.querySelectorAll('*').length,
        classes: {},
    };

    const counts = {};
    document.querySelectorAll('[class]').forEach(el => {
        let cn = el.className;
        if (typeof cn === 'string') {
            cn.split(' ').forEach(cls => {
                const prefix = cls.substring(0, 40);
                if (!prefix) return;
                counts[prefix] = (counts[prefix] || 0) + 1;
            });
        }
    });
    info.classes = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 50);

    info.textareas = Array.from(document.querySelectorAll('textarea')).map(t => ({
        id: t.id,
        class: (t.className || '').substring(0, 60),
        placeholder: (t.placeholder || '').substring(0, 40),
        visible: t.offsetParent !== null,
    }));

    info.contenteditables = Array.from(document.querySelectorAll('[contenteditable="true"]')).map(t => ({
        id: t.id,
        class: (t.className || '').substring(0, 60),
        text: (t.textContent || '').substring(0, 40),
        visible: t.offsetParent !== null,
    }));

    info.sendButtons = Array.from(document.querySelectorAll('button, [role="button"], a, .weui-desktop-btn')).map(t => ({
        text: (t.textContent || '').trim().substring(0, 30),
        class: (t.className || '').substring(0, 60),
        visible: t.offsetParent !== null,
    }));

    info.contactElements = Array.from(document.querySelectorAll(
        '[class*="contact"], [class*="session"], [class*="chat-list"], [class*="conversation"], [class*="kf-list"], [class*="visitor"], [class*="customer"], li, [role="listitem"]'
    )).slice(0, 20).map(t => ({
        text: (t.textContent || '').trim().substring(0, 60),
        class: (t.className || '').substring(0, 60),
        tag: t.tagName,
        visible: t.offsetParent !== null,
    }));

    info.frames = Array.from(document.querySelectorAll('iframe, micro-app, webview')).map(f => ({
        tag: f.tagName,
        src: (f.src || '').substring(0, 120),
        name: f.name || '',
        id: f.id || '',
    }));

    info.messageContainers = Array.from(document.querySelectorAll(
        '[class*="message"], [class*="msg-list"], [class*="chat-content"], [class*="chat-body"], [class*="scroll"]'
    )).slice(0, 10).map(t => ({
        class: (t.className || '').substring(0, 60),
        tag: t.tagName,
        childCount: t.children.length,
        text: (t.textContent || '').trim().substring(0, 100),
        visible: t.offsetParent !== null,
    }));

    return info;
}
"""

ANALYZE_WXA_SCRIPT = """
() => {
    const app = document.querySelector('#app, #root, .app, .shop-app, .kf-app, micro-app, [data-wxa]');
    if (!app) return { noApp: true };

    const result = {
        appTag: app.tagName,
        appId: app.id,
        appClass: (app.className || '').substring(0, 80),
        shadowDOM: null,
        vueInfo: null,
    };

    if (app.shadowRoot) {
        const sr = app.shadowRoot;
        result.shadowDOM = {
            childCount: sr.children.length,
            htmlLength: sr.innerHTML.length,
            classes: {},
        };
        const counts = {};
        sr.querySelectorAll('[class]').forEach(el => {
            let cn = el.className;
            if (typeof cn === 'string') {
                cn.split(' ').forEach(cls => {
                    const p = cls.substring(0, 40);
                    if (p) counts[p] = (counts[p] || 0) + 1;
                });
            }
        });
        result.shadowDOM.classes = Object.entries(counts).sort((a,b) => b[1]-a[1]).slice(0, 30);
    }

    const vueApp = document.querySelector('#app, #root');
    if (vueApp) {
        const attrs = {};
        for (const attr of vueApp.attributes) {
            attrs[attr.name] = attr.value;
        }
        result.vueInfo = { attrs };
    }

    return result;
}
"""

FILTERED_MESSAGES = [
    "商品不太适合我，暂时不用了",
    "商品不太适合我",
    "暂时不需要",
    "已读",
]


class WeixinKFMonitorAgent:
    """微信小店客服消息自动化监控与 AI 回复（API 拦截驱动）"""

    name = "weixin_kf_monitor"
    display_name = "微信小店客服自动监控回复"

    def __init__(self, store: str = "", headless: bool = False,
                 dry_run: bool = False, max_replies_per_round: int = 10,
                 endpoint: str = "", fixed_reply: str = "",
                 min_delay: int = 2, max_delay: int = 6,
                 reply_cooldown: int = 60, tool_loop: bool = False,
                 sse_mode: bool = False):
        self.store = store
        self.headless = headless
        self.dry_run = dry_run
        self.max_replies_per_round = max_replies_per_round
        self.endpoint = endpoint
        self.fixed_reply = fixed_reply
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.reply_cooldown = reply_cooldown
        self.tool_loop = tool_loop
        self.sse_mode = sse_mode

        self.browser = None
        self.context = None
        self.page = None
        self._pw = None
        self.running = False

        base_dir = os.path.join(DATA_DIR, store or "weixin")
        self.storage_file = os.path.join(base_dir, "storage_state.json")
        self.cookie_file = os.path.join(base_dir, "cookies.json")
        self._crm_api_base = "http://localhost:7120/api/v1"

        _kf_dir = os.path.join(DATA_DIR, "weixin_kf_monitor")
        os.makedirs(_kf_dir, exist_ok=True)
        self.reply_history = ReplyHistory(db_path=os.path.join(_kf_dir, "reply_history.db"))

        # ---- API 拦截相关 ----
        self._message_queue: list[dict] = []
        self._queue_lock = threading.Lock()
        self._capturing = False             # --analyze 模式开启时置 True
        self._network_log: list[dict] = []   # 仅用于 --analyze
        self._captured_responses: list[dict] = []  # 仅用于 --analyze

        self._seen_msg_ids: set[str] = set()
        self._click_cooldown: dict[str, float] = {}
        self._last_msg_count: dict[str, int] = {}

    # -----------------------------------------------------------
    # DOM 消息提取（面板）
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    # 浏览器生命周期
    # -----------------------------------------------------------

    def _resolve_storage(self) -> Optional[str]:
        if os.path.exists(self.storage_file):
            logger.info(f"使用 storage state: {self.storage_file}")
            return self.storage_file
        logger.warning(f"未找到 storage state 文件: {self.storage_file}")
        logger.warning("请先运行: python -m scraper.weixin_login")
        return None

    def _load_cookies_direct(self) -> bool:
        if os.path.exists(self.cookie_file):
            try:
                with open(self.cookie_file, "r", encoding="utf-8") as f:
                    cookies = json.load(f)
                self.context.add_cookies(cookies)
                logger.info(f"已加载 cookie: {self.cookie_file} ({len(cookies)} 条)")
                return True
            except Exception as e:
                logger.warning(f"加载 cookie 失败: {e}")
        return False

    def _save_storage_state(self):
        base_dir = os.path.dirname(self.storage_file)
        os.makedirs(base_dir, exist_ok=True)
        try:
            self.context.storage_state(path=self.storage_file)
            cookies = self.context.cookies()
            with open(self.cookie_file, "w", encoding="utf-8") as f:
                json.dump(cookies, f, ensure_ascii=False, indent=2)
            logger.info(f"storage state 已保存 ({len(cookies)} 条 cookie)")
        except Exception as e:
            logger.warning(f"保存 storage state 失败: {e}")

    def start(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        state_file = self._resolve_storage()
        self.browser = self._pw.chromium.launch(
            headless=self.headless,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        )
        self.context = self.browser.new_context(
            storage_state=state_file,
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="zh-CN",
        )
        self.page = self.context.new_page()

        # ★ 立即注册网络监听，不放过从导航开始的所有网络活动
        self._network_log = []
        self._captured_responses = []
        self._capturing = False  # 仅 --analyze 模式开启
        self.context.on("request", self._on_network_event)
        self.context.on("response", self._on_network_event)
        self.context.on("websocket", self._on_websocket)

        if not state_file:
            self._load_cookies_direct()

    def _event_info(self, event) -> dict:
        """从 request / response / websocket 对象中安全提取信息"""
        import urllib.parse as up
        info = {"url": "", "method": "", "status": 0, "type": ""}
        try:
            if hasattr(event, "url"):
                info["url"] = str(event.url)[:300]
            if hasattr(event, "method"):
                info["method"] = str(event.method)
            elif hasattr(event, "request") and hasattr(event.request, "method"):
                info["method"] = str(event.request.method)
            if hasattr(event, "status"):
                info["status"] = event.status
            if hasattr(event, "request"):
                req = event.request
                if hasattr(req, "resource_type"):
                    info["type"] = str(req.resource_type)
                elif hasattr(req, "url") and req.url:
                    parsed = up.urlparse(req.url)
                    if parsed.scheme in ("ws", "wss"):
                        info["type"] = "websocket"
            return info
        except Exception as e:
            logger.debug(f"_event_info 异常: {e}")
            return info

    def close(self):
        try:
            if self.context:
                self.context.close()
            if self.browser:
                self.browser.close()
            if self._pw:
                self._pw.stop()
        except Exception as e:
            logger.warning(f"关闭浏览器异常: {e}")
        self.reply_history.close()

    def screenshot(self, name: str = "debug"):
        try:
            out_dir = os.path.join(DATA_DIR, "weixin_kf_monitor", "screenshots")
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"{name}_{datetime.now().strftime('%H%M%S')}.png")
            self.page.screenshot(path=path)
            logger.info(f"截图: {path}")
        except Exception as e:
            logger.warning(f"截图失败: {e}")

    def _wait_stable(self, timeout: int = 5):
        try:
            self.page.wait_for_load_state("networkidle", timeout=timeout * 1000)
        except Exception:
            pass
        time.sleep(2)

    # -----------------------------------------------------------
    # 登录
    # -----------------------------------------------------------

    def login(self) -> bool:
        logger.info(f"导航到客服工作台: {KF_URL}")
        try:
            self.page.goto(KF_URL, wait_until="load", timeout=120000)
        except Exception as e:
            logger.warning(f"导航超时: {e}")
        time.sleep(8)
        current_url = self.page.url
        logger.info(f"当前 URL: {current_url}")

        if "login" in current_url or "passport" in current_url:
            logger.error("登录态已过期，请先运行:")
            logger.error(f"  python -m scraper.weixin_login")
            self.screenshot("login_redirect")
            return False

        if "store.weixin.qq.com" in current_url:
            text_len = self.page.evaluate("document.body.innerText.length")
            logger.info(f"客服工作台已加载，文本长度: {text_len}")
            if text_len > 50:
                logger.info("登录成功")
                self._save_storage_state()
                return True
            logger.info("页面内容较少，继续等待...")
            for _ in range(10):
                time.sleep(3)
                text_len = self.page.evaluate("document.body.innerText.length")
                if text_len > 100:
                    logger.info(f"页面加载完成: {text_len} 字符")
                    self._save_storage_state()
                    return True
            logger.warning("页面加载超时")

        logger.warning("无法进入客服工作台，当前 URL: {current_url}")
        self.screenshot("login_failed")
        return False


    # -----------------------------------------------------------
    # DOM 交互（点击会话 + 发送消息，API 拦截不需要这些做轮询）
    # -----------------------------------------------------------

    def click_contact(self, name: str) -> bool:
        logger.info(f"点击会话: {name}")
        clicked = self.page.evaluate(CLICK_CONTACT_SCRIPT, name)
        if clicked:
            time.sleep(0.2)
            return True
        logger.info(f"滚动查找会话: {name}")
        try:
            grid = self.page.locator('[class*="ReactVirtualized__Grid"], [class*="virtual-grid"]').first
            if grid.count() > 0:
                for _ in range(100):
                    before = grid.evaluate("el => el.scrollTop")
                    max_pos = grid.evaluate("el => el.scrollHeight - el.clientHeight")
                    if before >= max_pos - 1: break
                    grid.evaluate(f"el => el.scrollTop = Math.min(el.scrollTop + el.clientHeight, max_pos)")
                    time.sleep(0.3)
                    clicked = self.page.evaluate(CLICK_CONTACT_SCRIPT, name)
                    if clicked:
                        time.sleep(0.5)
                        return True
                    if before == grid.evaluate("el => el.scrollTop"): break
        except Exception:
            pass
        logger.warning(f"未找到会话: {name}")
        return False

    def extract_messages(self, max_messages: int = 3) -> list[dict]:
        """提取当前选中会话的最新消息（滚动到底部取最新）"""
        try:
            self.page.evaluate("""() => {
                const containers = document.querySelectorAll(
                    '[class*="message-list"], [class*="chat-content"], [class*="chat-body"], ' +
                    '[class*="im-body"], [class*="scroll"]'
                );
                for (const c of containers) {
                    c.scrollTop = c.scrollHeight;
                }
            }""")
            time.sleep(0.3)
        except Exception:
            pass
        try:
            messages = self.page.evaluate(EXTRACT_MESSAGES_SCRIPT) or []
        except Exception as e:
            logger.warning(f"提取消息失败: {e}")
            return []
        messages = messages[-max_messages:]
        logger.debug(f"消息: {len(messages)} 条")
        for m in messages:
            preview = m.get("content", "")[:60] or f"[{m.get('msg_type','?')}]"
            logger.debug(f"  [{m['sender']}] {preview}")
        return messages

    def send_message(self, text: str) -> bool:
        if not text:
            logger.warning("发送内容为空")
            return False
        # textarea[data-qa-id="qa-send-message-textarea"]
        fill_result = self.page.evaluate(FILL_INPUT_SCRIPT, text)
        if fill_result == "no_input":
            # fallback: 找 textarea
            try:
                self.page.fill('textarea[data-qa-id="qa-send-message-textarea"]', text)
                logger.debug("输入框填充 (page.fill on textarea)")
            except Exception as e:
                logger.warning(f"未找到输入框: {e}")
                self.screenshot("no_input_box")
                return False
        else:
            logger.debug(f"输入框填充: {fill_result}")
        time.sleep(0.3)
        send_result = self.page.evaluate(CLICK_SEND_SCRIPT)
        logger.info(f"发送结果: {send_result}")
        if send_result == "no_button":
            self.screenshot("no_send_button")
            return False
        time.sleep(0.5)
        return True

    def click_transfer(self) -> bool:
        try:
            btn = self.page.locator('#session-transfer')
            if btn.count() > 0 and btn.is_visible():
                btn.click()
                logger.info("点击转接人工按钮")
                return True
            logger.warning("未找到转接按钮")
            return False
        except Exception as e:
            logger.warning(f"点击转接按钮失败: {e}")
            return False

    # -----------------------------------------------------------
    # AI 回复生成
    # -----------------------------------------------------------

    def _api_chat(self, contact_name: str, message_content: str) -> dict:
        if self.sse_mode:
            return self._api_chat_sse(contact_name, message_content)
        try:
            import requests
            resp = requests.post(
                f"{self._crm_api_base}/chat",
                json={
                    "message": message_content,
                    "user_id": self.store or "weixin",
                    "buyer_name": contact_name,
                    "session_key": contact_name,
                    "tool_loop": self.tool_loop,
                },
                timeout=(10, 30),
            )
            if resp.ok:
                body = resp.json()
                if body.get("state", {}).get("code") == 0:
                    data = body.get("data", {})
                    return {
                        "answer": data.get("answer", ""),
                        "needs_handoff": data.get("needs_handoff", False),
                        "route": data.get("route", ""),
                    }
                logger.warning(f"[API_CHAT] state.code != 0: {body.get('state')}")
            else:
                logger.warning(f"[API_CHAT] HTTP {resp.status_code}")
        except requests.exceptions.ConnectTimeout:
            logger.warning(f"[API_CHAT] CRM 连接超时: {self._crm_api_base}")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"[API_CHAT] CRM 连接失败: {e}")
        except requests.exceptions.Timeout:
            logger.warning("[API_CHAT] CRM 响应超时")
        except Exception as e:
            logger.warning(f"[API_CHAT] 异常: {e}")
        return {"answer": "", "needs_handoff": False, "route": ""}

    def _api_chat_sse(self, contact_name: str, message_content: str) -> dict:
        try:
            import requests
            resp = requests.post(
                f"{self._crm_api_base}/chat/stream",
                json={
                    "message": message_content,
                    "user_id": self.store or "weixin",
                    "buyer_name": contact_name,
                    "session_key": contact_name,
                    "tool_loop": self.tool_loop,
                },
                stream=True,
                timeout=(10, 120),
            )
            if not resp.ok:
                logger.warning(f"[API_CHAT_SSE] HTTP {resp.status_code}")
                return {"answer": "", "needs_handoff": False, "route": ""}
            answer = ""
            needs_handoff = False
            route = ""
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                typ = event.get("type", "")
                if typ == "result":
                    answer = event.get("text", "")
                    needs_handoff = event.get("needs_handoff", False)
                    route = event.get("route", "")
                    logger.info(f"[API_CHAT_SSE] result: {answer[:80]} handoff={needs_handoff}")
                elif typ == "placeholder":
                    logger.debug(f"[API_CHAT_SSE] placeholder: {event.get('text', '')[:40]}")
                elif typ == "error":
                    logger.warning(f"[API_CHAT_SSE] error: {event.get('message', '')}")
                    break
            return {"answer": answer, "needs_handoff": needs_handoff, "route": route}
        except requests.exceptions.ConnectTimeout:
            logger.warning(f"[API_CHAT_SSE] CRM 连接超时: {self._crm_api_base}")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"[API_CHAT_SSE] CRM 连接失败: {e}")
        except requests.exceptions.Timeout:
            logger.warning("[API_CHAT_SSE] CRM 响应超时")
        except Exception as e:
            logger.warning(f"[API_CHAT_SSE] 异常: {e}")
        return {"answer": "", "needs_handoff": False, "route": ""}

    def generate_ai_reply(self, customer_message: str, customer_name: str = "") -> str:
        return ""

    def _get_product_info(self, talent_name: str) -> str:
        try:
            from app.storage.talent_db import TalentDB
            talent = TalentDB().get_talent_by_name(talent_name)
            if talent and talent.get("shop"):
                return f"我们的店铺: {talent['shop']}"
        except Exception:
            pass
        return "我们的产品"

    def _should_filter(self, message: str) -> bool:
        return any(p in message for p in FILTERED_MESSAGES)

    # ============================================================
    # API 拦截核心逻辑
    # ============================================================

    def _find_json_message_structures(self, data: Any, depth: int = 0,
                                       max_depth: int = 12) -> list[dict]:
        """递归在 JSON 树中寻找消息结构

        通用规则:
          - dict 同时含 content/text + sender/from/role 等字段
        微信小店特定规则:
          - dict 含 messageBody 子字段，且 messageBody 有 content
          - dict 含 msgList 列表（只取列表元素，不递归进 messageBody）
        """
        if depth > max_depth:
            return []
        results: list[dict] = []

        if isinstance(data, dict):
            sender_keys = {"sender", "from", "from_user", "user", "role", "speaker", "send_user"}
            content_keys = {"content", "text", "message", "msg", "body"}

            has_sender = any(k in data for k in sender_keys)
            has_content = any(k in data for k in content_keys)
            has_time = any(k in data for k in {"time", "timestamp", "send_time", "create_time", "created_at"})

            if has_content and isinstance(data.get("content") or data.get("text") or data.get("message"), str):
                if has_sender or has_time:
                    results.append(data)

            # 微信小店特定: messageBody + msgList 已在此层处理，跳过子递归避免重复
            skip_keys = set()

            mb = data.get("messageBody")
            if isinstance(mb, dict) and mb.get("content"):
                results.append(data)
                skip_keys.add("messageBody")

            ml = data.get("msgList")
            if isinstance(ml, list):
                for item in ml:
                    if isinstance(item, dict):
                        results.append(item)
                skip_keys.add("msgList")

            for k, v in data.items():
                if k in skip_keys:
                    continue
                if isinstance(v, (dict, list)):
                    results.extend(self._find_json_message_structures(v, depth + 1, max_depth))

        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    results.extend(self._find_json_message_structures(item, depth + 1, max_depth))

        return results

    def _extract_contact_hint(self, data: Any, depth: int = 0, max_depth: int = 8) -> str:
        """从 JSON 响应中提取联系人名称（支持嵌套 ext 结构）"""
        if depth > max_depth:
            return ""
        if isinstance(data, dict):
            # 优先从 ext 子对象中提取
            ext = data.get("ext")
            if isinstance(ext, dict):
                for k in ("nickname", "uname", "user_name", "name"):
                    v = ext.get(k)
                    if v and isinstance(v, str) and len(v) > 1:
                        return v
            for k in ("nickname", "uname", "contact_name", "user_name", "from_user", "name", "friend_name"):
                if k in data and isinstance(data[k], str) and len(data[k]) > 1:
                    return data[k]
            for v in data.values():
                if isinstance(v, (dict, list)):
                    name = self._extract_contact_hint(v, depth + 1, max_depth)
                    if name:
                        return name
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    name = self._extract_contact_hint(item, depth + 1, max_depth)
                    if name:
                        return name
        return ""

    def _normalize_message(self, raw: dict) -> Optional[dict]:
        """将原始 JSON 消息对象标准化为统一格式

        支持通用结构和飞鸽 微信小店客服 嵌套结构:
          - 通用: {content, sender, timestamp, ...}
          - 飞鸽: {messageBody: {content, ext: {nickname, sender_role}}, serverMessageId}
        返回: {sender, content, contact_name, timestamp, msg_id} 或 None
        """
        # -------------------------------------------------------
        # 第一步: 尝试从 messageBody 提取（飞鸽 微信小店客服 嵌套结构）
        # -------------------------------------------------------
        mb = raw.get("messageBody")
        if isinstance(mb, dict):
            ext = mb.get("ext") or {}
            content = mb.get("content", "") or ""
            sender_role = str(ext.get("sender_role", ""))
            nickname = ext.get("nickname") or ext.get("uname", "")
            biz_role = ext.get("s:sender_biz_role", "")
            msg_id = str(raw.get("serverMessageId", "") or mb.get("serverMessageId", ""))
            timestamp = str(mb.get("createTime", "") or mb.get("createTimestamp", ""))
            direction = ext.get("flow_extra", "")

            # 发送方判断: sender_role="2"=我方, "1"=达人; 或 biz_role 判断
            is_staff = (sender_role == "2") or ("CurrentServer" in str(biz_role))
            is_buyer = (sender_role == "1") or ("Buyer" in str(biz_role))

            if content and (is_staff or is_buyer or sender_role):
                return {
                    "sender": "me" if is_staff else "other",
                    "content": content.strip(),
                    "contact_name": (nickname or "").strip(),
                    "timestamp": timestamp,
                    "msg_id": msg_id or f"msg:{content[:40]}:{timestamp}",
                }

        # -------------------------------------------------------
        # 第二步: 通用提取（平铺结构）
        # -------------------------------------------------------
        sender = ""
        for k in ("sender", "from", "from_user", "user", "role", "speaker", "send_user"):
            v = raw.get(k)
            if v and isinstance(v, str) and v.strip():
                sender = v
                break

        content = ""
        for k in ("content", "text", "message", "msg", "body"):
            v = raw.get(k)
            if v and isinstance(v, str):
                content = v
                break

        if not content:
            return None

        timestamp = ""
        for k in ("time", "timestamp", "send_time", "create_time", "created_at"):
            v = raw.get(k)
            if v and isinstance(v, str):
                timestamp = v
                break

        contact_name = ""
        for k in ("contact_name", "nickname", "user_name", "from_user", "name", "visitor_name", "customer_name", "extra_info"):
            v = raw.get(k)
            if v and isinstance(v, str) and len(v) > 1:
                contact_name = v
                break

        msg_id = ""
        for k in ("msg_id", "message_id", "id", "uuid", "mid", "msgid"):
            v = raw.get(k)
            if v and isinstance(v, (str, int)):
                msg_id = str(v)
                break
        if not msg_id:
            msg_id = f"{sender}:{content[:50]}:{timestamp}"

        is_me = any(s.lower() in ("me", "self", "mine", "staff", "kf", "我") for s in (sender, raw.get("role", "")))

        return {
            "sender": "me" if is_me else "other",
            "content": content.strip() if content else "",
            "contact_name": contact_name,
            "timestamp": timestamp,
            "msg_id": msg_id,
        }

    def _parse_messages_from_api(self, data: Any) -> list[dict]:
        """自适应解析 API 响应中的消息列表

        1. 优先处理微信小店 KF 格式 (msg_kf_content JSON 字符串)
        2. 回退到通用 JSON 树搜索
        3. 去重 + 返回
        """
        # -------------------------------------------------------
        # 微信小店 KF 专用解析: data.list[].msg_kf_content
        # -------------------------------------------------------
        if isinstance(data, dict):
            msg_list = data.get("list")
            if isinstance(msg_list, list):
                kf_messages = []
                for item in msg_list:
                    if not isinstance(item, dict):
                        continue
                    msg_direction = item.get("msg_direction", 0)
                    if msg_direction != 1:
                        continue
                    raw_content = item.get("msg_kf_content", "{}")
                    try:
                        parsed = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
                    except Exception:
                        parsed = {}
                    content = parsed.get("content", "")
                    if not content:
                        continue
                    extra_raw = item.get("extra_info", "{}")
                    try:
                        extra = json.loads(extra_raw) if isinstance(extra_raw, str) else extra_raw
                    except Exception:
                        extra = {}
                    contact_name = extra.get("nickname", "") or item.get("send_openid", "")
                    msg_id = str(item.get("msg_id", ""))
                    kf_messages.append({
                        "sender": "other",
                        "content": content.strip(),
                        "contact_name": contact_name,
                        "timestamp": str(item.get("create_time", "")),
                        "msg_id": msg_id or f"kf:{contact_name}:{content[:40]}",
                    })
                if kf_messages:
                    logger.debug(f"KF 格式解析: {len(kf_messages)} 条消息")
                    return kf_messages

        # -------------------------------------------------------
        # 通用解析（回退）
        # -------------------------------------------------------
        raw_messages = self._find_json_message_structures(data)
        logger.debug(f"通用格式中找到 {len(raw_messages)} 个候选消息结构")

        if not raw_messages:
            return []

        contact_name_hint = ""
        if raw_messages and not raw_messages[0].get("contact_name"):
            contact_name_hint = self._extract_contact_hint(data)

        normalized = []
        for raw in raw_messages:
            msg = self._normalize_message(raw)
            if msg is None:
                continue
            if not msg["contact_name"] and contact_name_hint:
                msg["contact_name"] = contact_name_hint
            if not msg["content"]:
                continue
            normalized.append(msg)

        seen = set()
        unique = []
        for msg in normalized:
            if msg["msg_id"] not in seen:
                seen.add(msg["msg_id"])
                unique.append(msg)

        return unique

    def _on_network_event(self, event):
        """统一网络入口 — 接收 context.on('request') 和 context.on('response')"""
        info = self._event_info(event)
        url = info["url"]
        method = info["method"]
        status = info["status"]
        rtype = info["type"]
        is_response = hasattr(event, "status")

        if status and rtype in ("xhr", "fetch"):
            logger.debug(f"[NET] {method} {status} {rtype} {url[:120]}")

        # -------------------------------------------------------
        # --analyze 模式：记录全部
        # -------------------------------------------------------
        if self._capturing:
            self._network_log.append({
                "url": url, "method": method, "status": status,
                "type": rtype, "time": datetime.now().isoformat(),
            })
            if is_response and status == 200 and rtype in ("xhr", "fetch", ""):
                try:
                    body = event.json()
                    self._captured_responses.append({
                        "url": url, "status": status,
                        "type": rtype, "body": body,
                        "time": datetime.now().isoformat(),
                    })
                except Exception:
                    pass
            return

        # -------------------------------------------------------
        # 监控模式：仅处理匹配 endpoint 的 HTTP 响应
        # -------------------------------------------------------
        if not self.endpoint:
            return
        if not is_response:
            return
        if status != 200:
            return
        if not re.search(self.endpoint, url):
            return

        try:
            data = event.json()
            logger.debug(f"原始响应 ({url[:80]}): {json.dumps(data, ensure_ascii=False)[:800]}")
        except Exception:
            return

        messages = self._parse_messages_from_api(data)
        if messages:
            logger.info(f"拦截到 {len(messages)} 条消息: {url}")
            with self._queue_lock:
                self._message_queue.extend(messages)

    def _on_websocket(self, ws):
        """WebSocket 连接事件 — 记录连接 + 注册帧监听"""
        url = ws.url
        logger.info(f"[WS] 连接: {url[:150]}")
        self._network_log.append({
            "url": url, "method": "WEBSOCKET", "status": 101,
            "type": "websocket", "time": datetime.now().isoformat(),
        })

        ws.on("framereceived", lambda f: self._on_ws_frame(url, f))
        ws.on("framesend", lambda f: self._on_ws_frame(url, f))

    def _on_ws_frame(self, ws_url: str, frame):
        """WebSocket 帧回调 — 记录 + 尝试解析 JSON 并入队"""
        text = frame.text if hasattr(frame, "text") else None
        if not text:
            return

        if frame.type == "close":
            logger.debug(f"[WS] 关闭: {ws_url[:80]}")
            return

        preview = text[:300]
        try:
            json.loads(text)
            logger.debug(f"[WS] JSON帧 ({ws_url[:80]}): {preview}")
        except (json.JSONDecodeError, TypeError):
            logger.debug(f"[WS] 非JSON帧 ({ws_url[:80]}): {preview}")
            return

        self._network_log.append({
            "url": f"{ws_url}#frame",
            "method": "WS_FRAME",
            "status": 0,
            "type": "websocket_frame",
            "body_preview": text[:500],
            "time": datetime.now().isoformat(),
        })

        # 监控模式：解析 JSON 并入队
        if self.running and self.endpoint and re.search(self.endpoint, ws_url):
            try:
                data = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return
            messages = self._parse_messages_from_api(data)
            if messages:
                logger.info(f"WebSocket 拦截到 {len(messages)} 条消息")
                with self._queue_lock:
                    self._message_queue.extend(messages)

    def _handle_incoming_message(self, msg: dict):
        """处理一条来自 API 的消息"""
        contact_name = msg.get("contact_name", "")
        content = msg.get("content", "").strip()
        sender = msg.get("sender", "other")
        msg_id = msg.get("msg_id", "")

        if not content or not contact_name:
            return

        if sender == "me":
            return

        if msg_id and msg_id in self._seen_msg_ids:
            return
        if msg_id:
            self._seen_msg_ids.add(msg_id)

        if self.reply_history.is_replied(contact_name, content):
            logger.debug(f"已回复过: {contact_name} - {content[:40]}")
            return

        if self._should_filter(content):
            logger.info(f"跳过过滤消息: {contact_name} - {content[:40]}")
            return

        logger.info(f"新消息: [{contact_name}] {content[:80]}")

        if not self.click_contact(contact_name):
            logger.warning(f"无法点击会话 {contact_name}")
            return

        time.sleep(1)

        if self.dry_run:
            logger.info(f"[DRY RUN] 将回复 {contact_name}: {content[:60]}")
            self.reply_history.record(contact_name, content, status="dry_run")
            return

        if self.fixed_reply:
            reply = self.fixed_reply
            needs_handoff = False
            _sent_placeholder = False
        else:
            _sent_placeholder = False
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._api_chat, contact_name, content)
                try:
                    result = future.result(timeout=15)
                    reply = result.get("answer", "")
                    needs_handoff = result.get("needs_handoff", False)
                except concurrent.futures.TimeoutError:
                    _placeholder = random.choice([
                        "好的，我看到您消息了，正在查询中稍等~",
                        "嗯嗯，我在呢，我帮您确认下~",
                        "收到，我查一下后台数据哈~",
                        "稍等，我核实一下马上回复您~",
                    ])
                    self.send_message(_placeholder)
                    logger.info(f"兜底 [{contact_name}]: {_placeholder}")
                    _sent_placeholder = True
                    try:
                        result = future.result(timeout=60)
                    except concurrent.futures.TimeoutError:
                        result = {"answer": "", "needs_handoff": False}
                    reply = result.get("answer", "")
                    needs_handoff = result.get("needs_handoff", False)
        if not reply and not _sent_placeholder:
            logger.warning(f"AI 回复生成失败: {contact_name}")
            return
        if not reply:
            return

        logger.info(f"AI 回复 [{contact_name}]: {reply[:100]}")

        delay = random.uniform(self.min_delay, self.max_delay)
        time.sleep(delay)
        sent = self.send_message(reply)
        if sent:
            logger.info(f"回复成功: {contact_name}")
            self.reply_history.record(contact_name, content, reply_content=reply, status="success")
        else:
            logger.error(f"发送失败: {contact_name}")
            self.reply_history.record(contact_name, content, reply_content=reply, status="send_failed")
            return

        if needs_handoff:
            time.sleep(random.uniform(1, 3))
            logger.info(f"接口要求转人工: {contact_name}")
            self.click_transfer()

    # -----------------------------------------------------------
    # 监控生命周期
    # -----------------------------------------------------------

    def start_monitor(self):
        """MutationObserver 监测左侧会话预览变化 → click → extract → AI reply → DOM 发送"""
        self.running = True
        logger.info("=" * 60)
        logger.info("微信小店客服监控启动 (Observer 模式)")
        logger.info(f"  店铺: {self.store or 'weixin'}")
        logger.info(f"  无头模式: {self.headless}")
        logger.info(f"  Dry Run: {self.dry_run}")
        logger.info("=" * 60)

        self.page.evaluate(INJECT_SIDEBAR_OBSERVER_SCRIPT)
        logger.info("会话监测已注入")

        last_stats_time = time.time()
        processing = set()

        try:
            while self.running:
                pending = self.page.evaluate("""() => {
                    var arr = window.__kfPendingConversations || [];
                    window.__kfPendingConversations = [];
                    return arr;
                }""") or []

                for item in pending:
                    if not self.running:
                        break
                    name = item.get("name", "") if isinstance(item, dict) else str(item)
                    if not name:
                        continue
                    if name in processing:
                        logger.debug(f"会话加锁中，跳过: {name}")
                        continue
                    if re.match(r'^\d+$', name):
                        continue

                    logger.info(f"新会话活动: {name}")
                    processing.add(name)

                    try:
                        if not self.click_contact(name):
                            logger.warning(f"无法点击会话: {name}")
                            continue
                        time.sleep(0.8)

                        messages = self.extract_messages(max_messages=50)

                        known = self._last_msg_count.get(name)
                        if known is None:
                            new_msgs = []
                            for msg in reversed(messages):
                                if msg["sender"] == "other":
                                    new_msgs.append(msg)
                                else:
                                    break
                            new_msgs.reverse()
                        else:
                            new_msgs = [m for m in messages[known:] if m["sender"] == "other"]

                        if not new_msgs:
                            logger.debug(f"{name} 无新消息 (已知 {known or 0} / 当前 {len(messages)})")
                            continue

                        self._last_msg_count[name] = len(messages)
                        logger.info(f"{name} 有 {len(new_msgs)} 条新消息: {[m['content'][:20] for m in new_msgs]}")

                        for msg in new_msgs:
                            if not self.running:
                                break
                            content = msg.get("content", "").strip()
                            if not content:
                                continue
                            if self._should_filter(content):
                                continue
                            if any(kw in content for kw in ("未回复", "请及时处理", "已超过")):
                                continue

                            if self.dry_run:
                                logger.info(f"[DRY RUN] 将回复 {name}: {content[:60]}")
                                self.reply_history.record(name, content, status="dry_run")
                                continue

                            if msg.get("msg_type") == "voice":
                                # 语音已转写，去掉前缀走 CRM 回复
                                content = content.replace("[语音]", "", 1).strip()
                                if not content:
                                    continue
                            elif "[视频]" in content:
                                reply = "我看看您的视频，稍后回复您"
                                self.send_message(reply)
                                self.reply_history.record(name, content, reply_content=reply, status="success")
                                time.sleep(random.uniform(1, 3))
                                logger.info(f"视频消息，回复后转接人工: {name}")
                                self.click_transfer()
                                continue
                            elif "[图片]" in content:
                                reply = "请稍等，我先看看您的消息"
                                self.send_message(reply)
                                self.reply_history.record(name, content, reply_content=reply, status="success")
                                time.sleep(random.uniform(1, 3))
                                logger.info(f"图片消息，回复后转接人工: {name}")
                                self.click_transfer()
                                continue

                            if self.fixed_reply:
                                reply = self.fixed_reply
                                needs_handoff = False
                                _sent_placeholder = False
                            else:
                                _sent_placeholder = False
                                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                                    future = executor.submit(self._api_chat, name, content)
                                    try:
                                        result = future.result(timeout=15)
                                        reply = result.get("answer", "")
                                        needs_handoff = result.get("needs_handoff", False)
                                    except concurrent.futures.TimeoutError:
                                        _placeholder = random.choice([
                                            "好的，我看到您消息了，正在查询中稍等~",
                                            "嗯嗯，我在呢，我帮您确认下~",
                                            "收到，我查一下后台数据哈~",
                                            "稍等，我核实一下马上回复您~",
                                        ])
                                        self.send_message(_placeholder)
                                        logger.info(f"兜底 [{name}]: {_placeholder}")
                                        _sent_placeholder = True
                                        try:
                                            result = future.result(timeout=60)
                                        except concurrent.futures.TimeoutError:
                                            result = {"answer": "", "needs_handoff": False}
                                        reply = result.get("answer", "")
                                        needs_handoff = result.get("needs_handoff", False)
                            if not reply and not _sent_placeholder:
                                continue
                            if not reply:
                                continue

                            delay = random.uniform(self.min_delay, self.max_delay)
                            time.sleep(delay)
                            sent = self.send_message(reply)
                            if sent:
                                logger.info(f"回复成功 {name}: {content[:30]} → {reply[:30]}")
                                self.reply_history.record(name, content, reply_content=reply, status="success")
                            else:
                                logger.error(f"发送失败 {name}: {content[:30]}")
                                self.reply_history.record(name, content, reply_content=reply, status="send_failed")
                                continue

                            time.sleep(1)

                            if needs_handoff:
                                logger.info(f"接口要求转人工: {name}")
                                self.click_transfer()
                    finally:
                        processing.discard(name)

                if time.time() - last_stats_time > 60:
                    try:
                        stats = self.reply_history.get_stats()
                        logger.info(f"统计: 共回复 {stats['total_replies']} 条 | "
                                   f"今日 {stats['today_replies']} 条 | "
                                   f"{stats['unique_contacts']} 个客户")
                    except Exception:
                        pass
                    last_stats_time = time.time()

                time.sleep(0.5)

        except KeyboardInterrupt:
            logger.info("用户中断")
        finally:
            logger.info("监控已停止")

    def stop(self):
        self.running = False

    # -----------------------------------------------------------
    # DOM 结构分析
    # -----------------------------------------------------------

    def analyze_dom(self) -> dict:
        """全面分析页面 DOM 结构"""
        logger.info("分析页面 DOM 结构...")
        self.screenshot("dom_analysis")
        self._save_html_snapshot("page")
        analysis = self.page.evaluate(ANALYZE_DOM_SCRIPT)
        logger.info(f"页面 URL: {analysis.get('url', '?')}")
        logger.info(f"页面标题: {analysis.get('title', '?')}")
        logger.info(f"文本长度: {analysis.get('textLength', '?')}")
        logger.info(f"总元素数: {analysis.get('totalElements', '?')}")
        for key in ("textareas", "contenteditables", "sendButtons", "contactElements", "frames", "messageContainers"):
            items = analysis.get(key, [])
            label = {"textareas": "输入框(textarea)", "contenteditables": "可编辑 div", "sendButtons": "发送按钮", "contactElements": "联系人元素", "frames": "iframe/micro-app", "messageContainers": "消息容器"}.get(key, key)
            logger.info(f"{label}: {len(items)} 个")
            for t in items[:5]:
                logger.info(f"  {t}")
        wxa = self.page.evaluate(ANALYZE_WXA_SCRIPT)
        analysis["wxa"] = wxa
        if isinstance(wxa, dict) and wxa.get("shadowDOM"):
            logger.info(f"Shadow DOM: {len(wxa['shadowDOM']['classes'])} 个类")
        return analysis

    def _save_html_snapshot(self, name: str = "page"):
        try:
            out_dir = os.path.join(DATA_DIR, "weixin_kf_monitor")
            os.makedirs(out_dir, exist_ok=True)
            html = self.page.content()
            path = os.path.join(out_dir, f"{name}.html")
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            logger.info(f"HTML 快照已保存 ({len(html)} chars): {path}")
        except Exception as e:
            logger.warning(f"保存 HTML 失败: {e}")

    # ============================================================
    # 网络分析模式（--analyze）
    # ============================================================

    def analyze_network(self, duration: int = 120):
        """分析客服工作台的网络请求，自动推荐消息 API endpoint（监听已在 start() 中注册）"""
        total_so_far = len(self._network_log)
        json_so_far = len(self._captured_responses)

        logger.info("=" * 60)
        logger.info(f"网络分析模式 — 已自动记录 {total_so_far} 个请求 / {json_so_far} 个 JSON 响应")
        logger.info(f"将在 {duration} 秒后自动分析并推荐最优 endpoint")
        logger.info("请在浏览器中操作：点击会话、刷新页面、或等待消息到达")
        logger.info("=" * 60)

        logger.info("已开启捕获，当前日志保留，持续记录中...")
        logger.info(f"（按 Enter 提前分析，或等待 {duration} 秒自动分析）")

        start_time = time.time()
        import msvcrt
        logger.info("按 Enter 提前结束 capture")
        while time.time() - start_time < duration:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key in (b'\r', b'\n'):
                    logger.info("用户提前结束 capture")
                    break
            time.sleep(0.5)
            elapsed = int(time.time() - start_time)
            if elapsed > 0 and elapsed % 30 == 0:
                logger.info(f"  已记录 {len(self._network_log)} 个请求, {len(self._captured_responses)} 个 JSON 响应 ({elapsed}s)")

        logger.info(f"记录完成: {len(self._network_log)} 请求 / {len(self._captured_responses)} JSON 响应")
        self._capturing = False

        # 评分
        scored = self._score_all_endpoints()
        self._print_scored_endpoints(scored)

        # 保存日志
        out_dir = os.path.join(DATA_DIR, "weixin_kf_monitor")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "network_analysis.json")
        output = {
            "analysis_time": datetime.now().isoformat(),
            "store": self.store,
            "total_requests": len(self._network_log),
            "total_json_responses": len(self._captured_responses),
            "scored_endpoints": scored[:20],
            "top_recommendation": scored[0] if scored else None,
            "network_log": self._network_log,
            "captured_responses": self._captured_responses,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"完整分析报告已保存: {out_path}")

        if scored:
            best = scored[0]
            logger.info("")
            logger.info("=" * 60)
            logger.info("推荐使用以下命令启动监控:")
            logger.info(f"  python -m scraper.weixin_kf_monitor_agent --store {self.store or 'pigeon'} --endpoint \"{best['url_pattern']}\"")
            logger.info("=" * 60)

        return scored

    def _score_api_endpoint(self, url: str, json_data: Any) -> float:
        """对单个 API endpoint 进行消息相关度评分"""
        score = 0.0

        # URL 关键字加分
        for kw, points in API_URL_KEYWORDS.items():
            if kw in url.lower():
                score += points

        # JSON 结构分析
        if isinstance(json_data, dict):
            data_str = json.dumps(json_data, ensure_ascii=False)
            # 检查 JSON key 名称
            for kw, points in API_JSON_KEYS.items():
                if kw in data_str.lower():
                    score += points

            # 查找消息结构
            msgs = self._find_json_message_structures(json_data)
            if msgs:
                score += len(msgs) * 5  # 每个消息结构 +5 分
                # 检查消息中是否有中文对话内容
                for m in msgs:
                    for k in ("content", "text", "message"):
                        v = m.get(k)
                        if v and isinstance(v, str) and len(v) > 5:
                            if any('\u4e00' <= c <= '\u9fff' for c in v):
                                score += 8  # 包含中文对话内容，高相关性

        return score

    def _score_all_endpoints(self) -> list[dict]:
        """对所有捕获的响应进行评分排序"""
        url_scores: dict[str, dict] = {}

        for entry in self._captured_responses:
            url = entry["url"]
            body = entry.get("body", {})
            if not body:
                continue

            # 聚合相同 URL 的评分
            if url not in url_scores:
                url_scores[url] = {
                    "url": url,
                    "score": 0.0,
                    "count": 0,
                    "sample_body": None,
                    "status": entry.get("status", 0),
                }

            score = self._score_api_endpoint(url, body)
            url_scores[url]["score"] += score
            url_scores[url]["count"] += 1
            # 保留第一条含消息结构的 body 作为示例
            if url_scores[url]["sample_body"] is None and score > 0:
                url_scores[url]["sample_body"] = body

        # 按总分排序
        scored = sorted(url_scores.values(), key=lambda x: x["score"], reverse=True)

        # 生成 URL 模式（通用化）
        for item in scored:
            url = item["url"]
            # 取路径部分作为 pattern
            from urllib.parse import urlparse
            try:
                parsed = urlparse(url)
                path = parsed.path
                if parsed.query:
                    path += "?" + parsed.query.split("&")[0]  # 只保留第一个参数
                item["url_pattern"] = re.escape(path).replace(r"\?", "?").replace(r"\&", "&")
            except Exception:
                item["url_pattern"] = url

        return scored

    def _print_scored_endpoints(self, scored: list[dict]):
        if not scored:
            logger.info("未捕获到任何可评分的 API 响应")
            return

        logger.info("")
        logger.info("=" * 60)
        logger.info("API Endpoint 评分排行 (TOP 10):")
        logger.info("=" * 60)
        for i, item in enumerate(scored[:10], 1):
            score = item["score"]
            url = item["url"]
            count = item["count"]
            has_msg = "✓" if item.get("sample_body") else " "
            logger.info(f"  {i}. [{score:4.0f}分] [{has_msg}] {url[:120]}")
            logger.info(f"      请求次数: {count}  |  推荐 pattern: {item.get('url_pattern', url)[:80]}")

        # 显示最佳候选的消息示例
        best = scored[0]
        if best.get("sample_body"):
            logger.info("")
            logger.info("最佳候选响应示例 (截取前 1000 字符):")
            sample = json.dumps(best["sample_body"], ensure_ascii=False, indent=2)[:1000]
            for line in sample.split("\n"):
                logger.info(f"  {line}")
        logger.info("=" * 60)


# ============================================================
# CLI
# ============================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="微信小店客服消息自动化监控与 AI 回复",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用流程:
  第1步 — 先登录:
    python -m scraper.weixin_login

  第2步 — 分析页面结构 + API:
    python -m scraper.weixin_kf_monitor_agent --store zhihuai --analyze

  第3步 — 交互式探查（可选）:
    python -m scraper.weixin_kf_monitor_agent --store zhihuai --interactive

  第4步 — 启动监控:
    python -m scraper.weixin_kf_monitor_agent --store zhihuai
        """,
    )
    parser.add_argument("--store", type=str, default="",
                        help="店铺目录名，对应 data/{store}/ 下的 storage_state (默认 data/weixin/)")
    parser.add_argument("--headless", action="store_true",
                        help="无头模式（默认 False）")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅检测预览，不实际回复")
    parser.add_argument("--max-replies", type=int, default=10,
                        help="单轮最大回复数（默认 10）")
    parser.add_argument("--endpoint", type=str, default="",
                        help="消息 API URL 模式（正则），如 '/api/im/message/list'")
    parser.add_argument("--fixed-reply", type=str, default="",
                        help="固定回复文本，设置后不调用 AI 回复，直接用此文本回复")
    parser.add_argument("--min-delay", type=int, default=2,
                        help="发送前最小随机延迟秒数（默认 2）")
    parser.add_argument("--max-delay", type=int, default=6,
                        help="发送前最大随机延迟秒数（默认 6）")
    parser.add_argument("--reply-cooldown", type=int, default=60,
                        help="同一条消息的重复回复冷却秒数（默认 60）")
    parser.add_argument("--analyze", action="store_true",
                        help="网络分析模式：捕获 API 请求，自动推荐消息 endpoint")
    parser.add_argument("--duration", type=int, default=120,
                        help="--analyze 模式持续时间（秒，默认 120）")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="交互模式：打开浏览器后可手动操作")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="详细日志")
    parser.add_argument("--tool-loop", action="store_true",
                        help="启用灵活工具调用模式（LLM 自主选择工具）")
    parser.add_argument("--sse-mode", action="store_true",
                        help="启用 SSE 流式模式（使用 /chat/stream 接口）")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    agent = WeixinKFMonitorAgent(
        store=args.store,
        headless=args.headless,
        dry_run=args.dry_run,
        max_replies_per_round=args.max_replies,
        endpoint=args.endpoint,
        fixed_reply=args.fixed_reply,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
        reply_cooldown=args.reply_cooldown,
        tool_loop=args.tool_loop,
        sse_mode=args.sse_mode,
    )

    try:
        logger.info("启动浏览器...")
        agent.start()

        if args.analyze or args.interactive:
            agent._capturing = True

        ok = agent.login()
        if not ok:
            logger.error("登录失败")
            return

        agent._wait_stable()

        if args.analyze:
            agent.analyze_dom()
            agent.screenshot("before_analysis")
            agent.analyze_network(duration=args.duration)
        elif args.interactive:
            agent.analyze_dom()
            agent.screenshot("interactive_mode")
            logger.info("=" * 60)
            logger.info("交互模式 — 浏览器已打开，可手动操作")
            logger.info("按 Ctrl+C 退出")
            logger.info("=" * 60)
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("用户中断")
        else:
            agent.start_monitor()

    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"运行失败: {e}", exc_info=True)
    finally:
        agent.close()


if __name__ == "__main__":
    main()
