"""
通用登录工具 — 打开任意网站手动登录并保存 Cookie + Storage State

不需要为每个平台写登录脚本，一个命令搞定所有网站。

用法:
  python -m scraper.generic_login --url https://example.com/login
  python -m scraper.generic_login --url https://example.com/login --name mysite --visible
  python -m scraper.generic_login --name mysite --validate
  python -m scraper.generic_login --name mysite --refresh --url https://example.com
   python -m scraper.generic_login --name jinritemai          # 从 sites.json 自动获取 URL
   python -m scraper.generic_login --list-sites               # 列出所有快捷站点
   python -m scraper.generic_login --name jinritemai --auto-save  # 关闭浏览器时自动保存
"""

import json
import os
import sys
import time
import shutil
import logging
import argparse
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ============================================================
# 工具函数
# ============================================================

def _resolve_name(url: str, name: str = "") -> str:
    """从 URL 中提取站点名，或使用用户指定的 --name"""
    if name:
        return name
    domain = urlparse(url).netloc.lower()
    domain = re.sub(r'^www\.', '', domain)
    parts = domain.split('.')
    if len(parts) >= 3:
        return parts[-2]
    return parts[-2] if len(parts) >= 2 else parts[0]


_SITES_PATH = os.path.join(os.path.dirname(__file__), "sites.json")


def _load_sites() -> dict[str, str]:
    """加载 sites.json，返回 {name: url} 映射（扁平化）"""
    if not os.path.exists(_SITES_PATH):
        return {}
    with open(_SITES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    sites = {}
    for key, value in data.items():
        if isinstance(value, str):
            sites[key] = value
        elif isinstance(value, dict):
            sites[key] = value.get("url", "")
    return sites


def _resolve_url_from_sites(name: str) -> str:
    """从 sites.json 中查找 --name 对应的 URL"""
    return _load_sites().get(name, "")


def _ensure_data_dir(data_dir: str):
    os.makedirs(data_dir, exist_ok=True)


def _resolve_paths(data_dir: str) -> tuple[str, str]:
    storage_file = os.path.join(data_dir, "storage_state.json")
    cookie_file = os.path.join(data_dir, "cookies.json")
    return storage_file, cookie_file


def _backup_state(data_dir: str):
    for fname in ["storage_state.json", "cookies.json"]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            bak = path.replace(".json", "_bak.json")
            try:
                shutil.copy2(path, bak)
                logger.info(f"已备份: {bak}")
            except Exception as e:
                logger.warning(f"备份失败 {fname}: {e}")


def _save_state(context, data_dir: str):
    _ensure_data_dir(data_dir)
    storage_file, cookie_file = _resolve_paths(data_dir)
    try:
        context.storage_state(path=storage_file)
        cookies = context.cookies()
        with open(cookie_file, "w", encoding="utf-8") as f:
            json.dump(cookies, f, ensure_ascii=False, indent=2)
        logger.info(f"状态已保存 → {storage_file} ({len(cookies)} 条 cookie)")
    except Exception as e:
        logger.warning(f"保存状态失败: {e}")


def _state_exists(data_dir: str) -> bool:
    storage_file, cookie_file = _resolve_paths(data_dir)
    return os.path.exists(storage_file) or os.path.exists(cookie_file)


def _get_state_file(data_dir: str) -> str | None:
    """返回可用的状态文件路径，优先 storage_state.json"""
    storage_file, cookie_file = _resolve_paths(data_dir)
    if os.path.exists(storage_file):
        return storage_file
    if os.path.exists(cookie_file):
        return cookie_file
    return None


def _launch_browser(headless: bool = False, storage_state: str | None = None):
    """启动 Chromium 浏览器（统一配置），可选加载 storage state"""
    from playwright.sync_api import sync_playwright
    pw = sync_playwright().start()
    browser = pw.chromium.launch(
        headless=headless,
        args=["--disable-blink-features=AutomationControlled"],
    )
    context = browser.new_context(
        storage_state=storage_state,
        viewport={"width": 1920, "height": 1080},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        locale="zh-CN",
    )
    page = context.new_page()
    return pw, browser, context, page

def _close_browser(pw, browser, context, page):
    try:
        if context:
            context.close()
        if browser:
            browser.close()
    except Exception:
        pass
    try:
        if pw:
            pw.stop()
    except Exception:
        pass


def _wait_for_browser_close(page, browser) -> bool:
    """等待用户关闭浏览器（页面关闭或浏览器断开连接）"""
    print("\n请在浏览器中完成操作，关闭浏览器窗口后将自动保存...", flush=True)
    try:
        page.wait_for_event("close", timeout=3600000)
    except Exception:
        pass
    time.sleep(0.3)
    return True


# ============================================================
# 四种模式
# ============================================================

def fresh_login(url: str, data_dir: str, auto_save: bool = False) -> bool:
    """模式1: 打开浏览器 → 手动登录 → 关闭浏览器自动保存（auto_save）或按 Enter 保存"""
    _ensure_data_dir(data_dir)
    if _state_exists(data_dir):
        _backup_state(data_dir)

    pw = browser = context = page = None
    try:
        pw, browser, context, page = _launch_browser(headless=False)

        logger.info(f"导航到: {url}")
        page.goto(url, wait_until="load", timeout=60000)
        logger.info(f"当前页面: {page.url}")

        if auto_save:
            _wait_for_browser_close(page, browser)
        else:
            print("\n请在浏览器中完成登录，然后按 Enter 保存状态...", flush=True)
            sys.stdin.readline()

        _save_state(context, data_dir)
        return True

    except KeyboardInterrupt:
        logger.info("用户中断")
        return False
    except Exception as e:
        logger.error(f"登录过程异常: {e}", exc_info=True)
        return False
    finally:
        _close_browser(pw, browser, context, page)


def validate(data_dir: str, url: str = "", visible: bool = False) -> bool:
    """模式2: 加载已有状态 → 导航到 URL → 截图 → 给你看结果"""
    state_file = _get_state_file(data_dir)
    if not state_file:
        logger.error(f"未找到状态文件 ({data_dir})")
        return False

    is_storage = state_file.endswith("storage_state.json")
    pw = browser = context = page = None
    try:
        pw, browser, context, page = _launch_browser(
            headless=not visible,
            storage_state=state_file if is_storage else None,
        )

        if not is_storage:
            with open(state_file, "r", encoding="utf-8") as f:
                context.add_cookies(json.load(f))
            logger.info(f"已加载 cookie 文件: {state_file}")
        else:
            logger.info(f"已加载 storage state: {state_file}")

        target = url or "about:blank"
        logger.info(f"导航到: {target}")
        page.goto(target, wait_until="load", timeout=60000)
        time.sleep(3)

        screenshot_path = os.path.join(data_dir, f"validate_{int(time.time())}.png")
        try:
            page.screenshot(path=screenshot_path)
            logger.info(f"截图已保存: {screenshot_path}")
        except Exception as e:
            logger.warning(f"截图失败: {e}")

        title = page.title()
        logger.info(f"页面标题: {title}")
        logger.info(f"最终 URL: {page.url}")

        if visible:
            print(f"\n当前页面: {page.url}", flush=True)
            print("请检查浏览器页面，关闭浏览器窗口将自动关闭...", flush=True)
            _wait_for_browser_close(page, browser)

        return True

    except Exception as e:
        logger.warning(f"验证异常: {e}")
        return False
    finally:
        _close_browser(pw, browser, context, page)


def refresh_session(url: str, data_dir: str, auto_save: bool = False) -> bool:
    """模式3: 加载旧状态 → 打开浏览器 → 手动操作 → 关闭浏览器自动保存（auto_save）或按 Enter 覆盖保存"""
    state_file = _get_state_file(data_dir)
    if not state_file:
        logger.error(f"未找到状态文件，请先使用 --url 登录 ({data_dir})")
        return False

    _backup_state(data_dir)

    is_storage = state_file.endswith("storage_state.json")
    pw = browser = context = page = None
    try:
        pw, browser, context, page = _launch_browser(
            headless=False,
            storage_state=state_file if is_storage else None,
        )

        if not is_storage:
            with open(state_file, "r", encoding="utf-8") as f:
                context.add_cookies(json.load(f))
            logger.info(f"已加载 cookie 文件: {state_file}")
        else:
            logger.info(f"已加载 storage state: {state_file}")

        logger.info(f"导航到: {url}")
        page.goto(url, wait_until="load", timeout=60000)
        logger.info(f"当前页面: {page.url}")

        if auto_save:
            _wait_for_browser_close(page, browser)
        else:
            print("\n请在浏览器中完成操作，然后按 Enter 保存状态...", flush=True)
            sys.stdin.readline()

        _save_state(context, data_dir)
        return True

    except Exception as e:
        logger.error(f"刷新会话异常: {e}", exc_info=True)
        return False
    finally:
        _close_browser(pw, browser, context, page)


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="通用登录工具 — 手动登录任意网站并保存 Cookie / Storage State",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 登录任意网站
  python -m scraper.generic_login --url https://example.com/login

  # 显示浏览器窗口 + 指定名称
  python -m scraper.generic_login --url https://example.com --name mysite --visible

  # 验证已有登录状态
  python -m scraper.generic_login --name mysite --validate

  # 刷新会话（加载旧状态，手动操作后覆盖保存）
  python -m scraper.generic_login --name mysite --refresh --url https://example.com

  # 自定义数据目录
  python -m scraper.generic_login --url https://example.com --data-dir ./my_data

  # 使用 sites.json 快捷名称（无需 --url）
  python -m scraper.generic_login --name jinritemai
  python -m scraper.generic_login --name weixin --validate
  python -m scraper.generic_login --name weixin --refresh
  python -m scraper.generic_login --list-sites

  # 关闭浏览器时自动保存（适用于 Web UI / 非交互调用）
  python -m scraper.generic_login --name jinritemai --auto-save
  python -m scraper.generic_login --name weixin --refresh --url https://example.com --auto-save

说明:
  - 状态文件保存在 data/{name}/ 下（storage_state.json + cookies.json）
  - --name 默认从 URL 域名自动提取，可手动指定
  - 支持配合现有 agent 使用（如 --name jinritemai 兼容巨量百应）
  - sites.json 中预置快捷名称，--name 即可自动匹配 URL，无需 --url
  - --auto-save: 关闭浏览器窗口时自动保存，无需按 Enter
        """,
    )
    parser.add_argument("--url", type=str, default="",
                        help="要打开的页面 URL（登录页或目标页）")
    parser.add_argument("--name", type=str, default="",
                        help="站点标识，用于区分不同站点的状态文件（默认从 URL 域名提取）")
    parser.add_argument("--visible", action="store_true",
                        help="--validate 模式下显示浏览器窗口（默认无头）")
    parser.add_argument("--validate", action="store_true",
                        help="验证已有登录状态是否有效")
    parser.add_argument("--refresh", action="store_true",
                        help="加载已有状态，手动操作后覆盖保存")
    parser.add_argument("--data-dir", type=str, default="",
                        help="自定义数据目录（默认 data/{name}/）")
    parser.add_argument("--list-sites", action="store_true",
                        help="列出 sites.json 中所有可用的快捷名称和 URL")
    parser.add_argument("--auto-save", action="store_true",
                        help="关闭浏览器时自动保存状态（无需按 Enter）")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # --list-sites: 列出所有快捷名称
    if args.list_sites:
        sites = _load_sites()
        if not sites:
            print("sites.json 为空或未找到，没有可用的快捷名称。")
        else:
            print(f"{'名称':<20} URL")
            print("-" * 80)
            for name, url in sorted(sites.items()):
                print(f"{name:<20} {url}")
        sys.exit(0)

    # 如果给了 --name 但没给 --url，尝试从 sites.json 查找
    if not args.url and args.name:
        resolved = _resolve_url_from_sites(args.name)
        if resolved:
            logger.info(f"从 sites.json 解析 URL: {args.name} → {resolved}")
            args.url = resolved

    # 判断模式
    if args.validate:
        if not args.name and not args.url:
            logger.error("--validate 需要 --name 或 --url 来定位状态文件")
            sys.exit(1)
        name = args.name or _resolve_name(args.url, "")
        data_dir = args.data_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", name
        )
        ok = validate(data_dir, url=args.url, visible=args.visible)

    elif args.refresh:
        if not args.url:
            logger.error("--refresh 需要 --url 来指定导航目标")
            sys.exit(1)
        name = args.name or _resolve_name(args.url, "")
        data_dir = args.data_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", name
        )
        ok = refresh_session(url=args.url, data_dir=data_dir, auto_save=args.auto_save)

    else:
        if not args.url:
            logger.error("请指定 --url")
            sys.exit(1)
        name = args.name or _resolve_name(args.url, "")
        data_dir = args.data_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", name
        )
        logger.info(f"站点: {name}  数据目录: {data_dir}")
        ok = fresh_login(url=args.url, data_dir=data_dir, auto_save=args.auto_save)

    if ok:
        logger.info("完成")
        sys.exit(0)
    else:
        logger.error("失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
