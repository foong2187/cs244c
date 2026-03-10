# Cursor SSH / Tailscale connection drops – root cause and fix

**TL;DR:** Drops were caused by **severe memory pressure** (swap thrashing) on the VM, not by VPN/network instability. Many Firefox processes from the crawler consumed 8–12 GB RAM; when RAM + swap were exhausted, SSH stalled and Cursor tunnels closed.

---

## Issue: Cursor SSH / Tailscale connection drops

The connection drops were **not** caused by VPN instability (OpenVPN / Tailscale / WireGuard).  
They were caused by **severe memory pressure** on the VM.

---

## Evidence

**System state during failures:**

- `load average`: 117, 127, 90
- **RAM:** 18 / 19 GiB used
- **Swap:** 4 / 4 GiB used

**vmstat** showed active swapping:

- `si`/`so` > 0 → memory being swapped constantly

**Process list** showed many Firefox processes (~250–470 MB each):

- firefox processes ≈ 20+
- each ≈ 250–470 MB  
- **Total Firefox usage ≈ 8–12 GB RAM.**

---

## What happens under memory pressure

When RAM is exhausted and swap is heavily used:

1. Kernel spends time reclaiming memory
2. Processes block waiting for pages from disk
3. SSH packets are delayed
4. TCP keepalives fail
5. Cursor Remote-SSH tunnels close  
6. **Appears like VPN/network drops**

So the issue was **swap thrashing**, not networking.

---

## Immediate fix

Kill browser processes:

```bash
pkill firefox
```

System stabilizes once memory pressure drops.

(If the crawler is running, also stop it so Firefox doesn’t respawn: `sudo ./crawler/stop_crawl.sh`.)

**Diagnose RAM after stopping:** run `./scripts/diagnose_ram.sh` to see total RSS by process type (firefox+content, tor, python, geckodriver) and top consumers. That shows how much the crawler’s Firefox fleet was using.

---

## Long-term fixes

### 1. Do not run browsers on compute VMs

Browsers spawn many processes and consume large memory.

**Use the VM for:**

- coding
- training
- builds
- services

**Run browsers locally** instead.

### 2. Increase swap

Current swap: 4 GB (too small).

**Recommended:** 12–16 GB swap.

Example:

```bash
sudo swapoff -a
sudo fallocate -l 12G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 3. Install earlyoom

Prevents system lockups by killing memory hogs early.

```bash
sudo apt install earlyoom
sudo systemctl enable --now earlyoom
```

### 4. (Optional) Limit Firefox processes

If running Firefox on the VM is unavoidable (e.g. crawler):

- `about:config`
- `dom.ipc.processCount` = 4

---

## Crawler-specific note

The **parallel crawler** (`crawl_parallel.py`) can run many workers; the **restart script now defaults to 5 workers** to reduce memory use. If you see `ReadTimeoutError: HTTPConnectionPool(host='localhost', port=...): Read timed out` in crawl.log, that’s usually Firefox/Marionette not responding (e.g. under memory pressure). The crawler now catches that, restarts the browser for that worker, and retries instead of killing the worker. You can still override workers, e.g. `CRAWLER_WORKERS=10 sudo ./crawler/restart_crawl.sh` (only if the VM has enough RAM).

Each worker has its own **Tor + Firefox**. That’s 10+ Firefox processes; under load they can spawn many more (tabs, content processes). To reduce memory pressure:

- Use fewer workers, e.g. `--workers 4` or `--workers 5`.
- Ensure `earlyoom` is installed so the system kills heavy processes before SSH dies.
- Consider increasing swap and/or RAM for the VM when running long crawls.

---

## Key takeaway

| Cause of connection failures | Not the cause |
|-----------------------------|----------------|
| **RAM exhaustion → swap thrashing → SSH timeouts** | Tailscale, OpenVPN, WireGuard, network instability |

---

*If you have a homelab trick that guarantees SSH survives even when RAM is exhausted, add it here.*
