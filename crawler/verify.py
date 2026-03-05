"""
Verify crawler setup: Tor control, guard IP, xvfb, Firefox+Selenium, tcpdump.
Run from repo root: python -m crawler.verify
"""

import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    print("=== Verifying crawler setup ===\n")
    ok_all = True

    # 1. Python deps
    print("1. Python packages...")
    for name in ("stem", "selenium", "dpkt", "numpy"):
        try:
            __import__(name)
            print(f"   OK  {name}")
        except ImportError:
            print(f"   MISSING  {name}  -> pip install -r requirements-crawler.txt")
            ok_all = False

    # 2. Tor control port
    print("\n2. Tor control port (9051)...")
    guard_ip = None
    socks_port = 9050
    try:
        from crawler.tor_client import get_controller, get_socks_port, get_entry_guard_ip
        c = get_controller(9051)
        c.connect()
        c.authenticate()
        socks_port = get_socks_port(c)
        print(f"   OK  connected, SOCKS={socks_port}")
        guard_ip = get_entry_guard_ip(c)
        if guard_ip:
            print(f"   OK  guard IP = {guard_ip}")
        else:
            print("   WARN  no guard IP yet (no circuits built)")
        c.close()
    except Exception as e:
        print(f"   FAIL  {e}")
        print("   Add to /etc/tor/torrc: ControlPort 9051 + CookieAuthentication 0")
        print("   Then: sudo systemctl restart tor@default")
        ok_all = False

    # 3. tcpdump
    print("\n3. tcpdump...")
    td = shutil.which("tcpdump")
    if td:
        print(f"   OK  {td}")
    else:
        print("   MISSING  sudo apt install tcpdump")
        ok_all = False

    # 4. geckodriver
    print("\n4. geckodriver...")
    gd = shutil.which("geckodriver")
    if gd:
        print(f"   OK  {gd}")
    else:
        print("   MISSING  download from https://github.com/mozilla/geckodriver/releases")
        ok_all = False

    # 5. Firefox
    print("\n5. Firefox...")
    ff = shutil.which("firefox") or shutil.which("firefox-esr")
    if ff:
        print(f"   OK  {ff}")
    else:
        print("   MISSING  sudo apt install firefox-esr")
        ok_all = False

    # 6. xvfb (needed on WSL / headless)
    print("\n6. xvfb-run (virtual display for WSL)...")
    xvfb = shutil.which("xvfb-run")
    if xvfb:
        print(f"   OK  {xvfb}")
    else:
        print("   MISSING  sudo apt install xvfb")
        ok_all = False

    if not ok_all:
        print("\n=== Some checks failed. Fix them and re-run. ===")
        return 1

    print("\n=== All checks passed! ===")
    print("\nTo run a test crawl:")
    print("  sudo python -m crawler.crawl --sites 2 --visits 2 --new_circuit_per_visit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
