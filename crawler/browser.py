"""
Reusable headless Firefox driver for loading pages through Tor SOCKS proxy.

The driver is created once and reused across visits (huge speedup vs creating
a new Firefox per visit). Call make_driver() at the start, visit_page() for
each URL, and driver.quit() when done. If visits start failing, call
restart_driver() to get a fresh instance.
"""

import os
import time

try:
    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.firefox.service import Service
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.common.exceptions import TimeoutException, WebDriverException
except ImportError:
    webdriver = None

SNAP_FIREFOX_BIN = "/snap/firefox/current/usr/lib/firefox/firefox"


def _find_firefox_binary():
    """Find real Firefox ELF binary (snap puts a shell wrapper at /usr/bin/firefox)."""
    if os.path.isfile(SNAP_FIREFOX_BIN):
        return SNAP_FIREFOX_BIN
    return None


def make_driver(socks_host="127.0.0.1", socks_port=9050, page_timeout=60):
    """
    Create a headless Firefox driver routed through Tor SOCKS proxy.
    Reuse this driver for many visits -- much faster than creating per visit.
    """
    if webdriver is None:
        raise ImportError("selenium is required: pip install selenium")

    opts = Options()
    opts.add_argument("--headless")

    ff_bin = _find_firefox_binary()
    if ff_bin:
        opts.binary_location = ff_bin

    opts.set_preference("network.proxy.type", 1)
    opts.set_preference("network.proxy.socks", socks_host)
    opts.set_preference("network.proxy.socks_port", socks_port)
    opts.set_preference("network.proxy.socks_version", 5)
    opts.set_preference("network.proxy.socks_remote_dns", True)

    # Reduce background noise (telemetry, updates, safe browsing)
    opts.set_preference("datareporting.healthreport.uploadEnabled", False)
    opts.set_preference("app.update.enabled", False)
    opts.set_preference("browser.safebrowsing.enabled", False)
    opts.set_preference("browser.safebrowsing.malware.enabled", False)
    opts.set_preference("privacy.trackingprotection.enabled", False)

    service = Service(log_output="/dev/null")
    driver = webdriver.Firefox(options=opts, service=service)
    driver.set_page_load_timeout(page_timeout)
    driver.set_script_timeout(page_timeout)
    return driver


def visit_page(driver, url, post_load_wait=5):
    """
    Navigate to url and wait for full page load (readyState == complete).
    Returns True on success, False on timeout/error.
    The driver is NOT quit on failure -- caller decides whether to restart.
    """
    try:
        driver.get(url)
        WebDriverWait(driver, driver.timeouts.page_load / 1000).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        time.sleep(post_load_wait)
        return True
    except TimeoutException:
        return False
    except WebDriverException:
        return False
    except Exception:
        return False


def quit_driver(driver):
    """Safely quit driver."""
    if driver:
        try:
            driver.quit()
        except Exception:
            pass
