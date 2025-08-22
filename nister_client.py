# nister_client.py
import os, hmac, json, time, logging, random, uuid
from typing import Dict, Any, Optional
from urllib.parse import urljoin
import requests

# --- Tunables (env overrides supported) ---------------------------------------
READ_TIMEOUT      = float(os.getenv("NISTER_HTTP_TIMEOUT",      "6.0"))   # seconds
CONNECT_TIMEOUT   = float(os.getenv("NISTER_CONNECT_TIMEOUT",   "3.05"))  # seconds
MAX_RETRIES       = int(  os.getenv("NISTER_MAX_RETRIES",       "4"))     # total attempts
BACKOFF_BASE      = float(os.getenv("NISTER_BACKOFF_BASE",      "0.5"))   # initial backoff
BACKOFF_CAP       = float(os.getenv("NISTER_BACKOFF_CAP",       "8.0"))   # max backoff
BACKOFF_JITTER    = float(os.getenv("NISTER_BACKOFF_JITTER",    "0.15"))  # +/- jitter (sec)

BASE_URL   = os.getenv("NISTER_TRADE_SERVER_URL", "https://trade.nister.org").rstrip("/") + "/"
API_TOKEN  = os.getenv("NISTER_API_TOKEN", "").strip()
SIG_SECRET = os.getenv("NISTER_SIGNING_SECRET", "").encode("utf-8") if os.getenv("NISTER_SIGNING_SECRET") else None

SIGNALS_PATH = os.getenv("NISTER_SIGNALS_PATH", "/signals").lstrip("/")
TRADES_PATH  = os.getenv("NISTER_TRADES_PATH",  "/trades").lstrip("/")

# Single shared session for connection pooling
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Nister-TradeAgent/1.0",
    "Accept": "application/json",
    "Content-Type": "application/json",
})

TIMEOUT_TUPLE = (CONNECT_TIMEOUT, READ_TIMEOUT)

# --- Helpers ------------------------------------------------------------------
def _auth_headers() -> Dict[str, str]:
    # If token is missing, don't crash the whole process—log once and proceed.
    if not API_TOKEN:
        logging.error("[nister_client] NISTER_API_TOKEN is missing; requests will likely be rejected (401).")
        return {}
    return {"Authorization": f"Bearer {API_TOKEN}"}

def _signed_headers(payload: Dict[str, Any]) -> Dict[str, str]:
    if not SIG_SECRET:
        return {}
    ts = str(int(time.time()))
    # Compact JSON to ensure signature stability
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    mac = hmac.new(SIG_SECRET, f"{ts}.{body}".encode("utf-8"), digestmod="sha256").hexdigest()
    return {"X-Ts": ts, "X-Signature": mac}

def _retryable_status(code: int) -> bool:
    # Includes Too Many Requests + common transient 5xx + Request Timeout
    return code in (408, 425, 429, 500, 502, 503, 504)

def _sleep_with_backoff(attempt_idx: int) -> None:
    # attempt_idx: 0-based (0,1,2,3,...) for exponential backoff
    delay = min(BACKOFF_CAP, BACKOFF_BASE * (2 ** attempt_idx))
    # Add +/- jitter to avoid herding
    jitter = random.uniform(-BACKOFF_JITTER, BACKOFF_JITTER)
    time.sleep(max(0.0, delay + jitter))

# --- Core HTTP sender with retries -------------------------------------------
def _post_json(path: str, payload: Dict[str, Any], *, idempotency_key: Optional[str] = None) -> Optional[requests.Response]:
    """
    Performs a POST with retries on network errors and retryable HTTP status codes.
    Returns a requests.Response on success, or None on total failure.
    """
    url = urljoin(BASE_URL, path)

    # Build headers once; if auth is missing, continue but never raise.
    try:
        headers = {**_auth_headers(), **_signed_headers(payload)}
    except Exception as e:
        logging.error(f"[nister_client] failed to build headers: {e}", exc_info=True)
        headers = {}

    if not idempotency_key:
        idempotency_key = f"auto-{uuid.uuid4()}"
    headers["Idempotency-Key"] = idempotency_key

    last_exception: Optional[Exception] = None
    last_response: Optional[requests.Response] = None

    for attempt in range(MAX_RETRIES):
        try:
            resp = SESSION.post(url, headers=headers, json=payload, timeout=TIMEOUT_TUPLE)
            status = resp.status_code

            # Retry on transient status codes
            if _retryable_status(status):
                snippet = (resp.text or "")[:200]
                logging.warning(
                    f"[nister_client] HTTP {status} on {url} (attempt {attempt+1}/{MAX_RETRIES}) "
                    f"— will retry. Body preview: {snippet!r}"
                )
                last_response = resp
                if attempt < MAX_RETRIES - 1:
                    _sleep_with_backoff(attempt)
                    continue
                # No more retries left; return the last response for caller to handle
                return last_response

            # Non-retryable: return immediately (caller will check resp.ok)
            return resp

        except requests.RequestException as e:
            last_exception = e
            logging.warning(
                f"[nister_client] network error POST {url}: {e} "
                f"(attempt {attempt+1}/{MAX_RETRIES}) — will retry."
            )
            if attempt < MAX_RETRIES - 1:
                _sleep_with_backoff(attempt)
                continue
            # Out of retries
            break

    # Total failure path: return the last HTTP response if we had one; else None
    if last_response is not None:
        return last_response

    logging.error(
        "[nister_client] giving up after retries; no response received."
        + (f" Last exception: {last_exception}" if last_exception else "")
    )
    return None

# --- Public API ---------------------------------------------------------------
def send_signal(payload: Dict[str, Any], *, idem: Optional[str] = None) -> bool:
    """
    Send a trade signal. Returns True on 2xx, False otherwise.
    Never raises on network/timeout errors.
    """
    resp = _post_json(SIGNALS_PATH, payload, idempotency_key=idem)
    if resp is None:
        logging.error("[nister_client] send_signal failed: network/timeout (no response)")
        return False
    if resp.ok:
        return True
    logging.error(f"[nister_client] send_signal failed {resp.status_code}: {(resp.text or '')[:400]}")
    return False

def send_trade(payload: Dict[str, Any], *, idem: Optional[str] = None) -> bool:
    """
    Send an executed trade. Returns True on 2xx, False otherwise.
    Never raises on network/timeout errors.
    """
    resp = _post_json(TRADES_PATH, payload, idempotency_key=idem)
    if resp is None:
        logging.error("[nister_client] send_trade failed: network/timeout (no response)")
        return False
    if resp.ok:
        return True
    logging.error(f"[nister_client] send_trade failed {resp.status_code}: {(resp.text or '')[:400]}")
    return False
