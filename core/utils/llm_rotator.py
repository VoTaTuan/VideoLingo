"""LLM Key & Account Rotator with Automatic Failover for VideoLingo.

Supports:
- Multi-Key Pool (round-robin / failover when token/rate limit or error occurs).
- Multiple Gemini / Google Accounts management.
- Cooldown tracking for rate-limited (HTTP 429 / Quota) credentials.
- Thread-safe rotation across parallel translation workers.
"""

from __future__ import annotations

import re
import time
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from rich import print as rprint

from core.utils.config_utils import load_key, load_key_or, update_key


class LLMRotator:
    """Manages pool of API keys and accounts with automatic rotation on failures."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._active_index: int = 0
        # Maps credential string -> unix timestamp when cooldown expires
        self._cooldowns: Dict[str, float] = {}
        # Track consecutive errors per credential
        self._error_counts: Dict[str, int] = {}

    def get_auth_mode(self) -> str:
        """Returns 'key_pool' or 'gemini_accounts'."""
        return str(load_key_or("api.auth_mode", "key_pool"))

    def set_auth_mode(self, mode: str) -> None:
        update_key("api.auth_mode", mode, add_missing=True)

    def is_auto_rotate_enabled(self) -> bool:
        return bool(load_key_or("api.auto_rotate", True))

    def set_auto_rotate(self, enabled: bool) -> None:
        update_key("api.auto_rotate", enabled, add_missing=True)

    def get_pool_keys(self) -> List[str]:
        """Extract clean list of API keys from config."""
        raw_pool = load_key_or("api.key_pool", None)
        keys: List[str] = []
        if isinstance(raw_pool, list):
            for k in raw_pool:
                cleaned = str(k).strip()
                if cleaned and cleaned != "YOUR_API_KEY":
                    keys.append(cleaned)
        elif isinstance(raw_pool, str):
            for line in raw_pool.replace(",", "\n").split("\n"):
                cleaned = line.strip()
                if cleaned and cleaned != "YOUR_API_KEY":
                    keys.append(cleaned)

        # Fallback to single api.key if key_pool is empty
        if not keys:
            single = str(load_key_or("api.key", "")).strip()
            if single and single != "YOUR_API_KEY":
                keys.append(single)
        return keys

    def set_pool_keys(self, keys: List[str]) -> None:
        cleaned = [k.strip() for k in keys if k.strip() and k.strip() != "YOUR_API_KEY"]
        update_key("api.key_pool", cleaned, add_missing=True)
        if cleaned:
            update_key("api.key", cleaned[0], add_missing=True)

    def get_gemini_accounts(self) -> List[Dict[str, str]]:
        """List of accounts: [{'email': ..., 'key': ..., 'label': ...}]."""
        raw = load_key_or("api.gemini_accounts", [])
        if isinstance(raw, list):
            return [acc for acc in raw if isinstance(acc, dict) and acc.get("key")]
        return []

    def set_gemini_accounts(self, accounts: List[Dict[str, str]]) -> None:
        cleaned = [
            {"email": str(a.get("email", "")).strip(), "key": str(a.get("key", "")).strip(), "label": str(a.get("label", "")).strip()}
            for a in accounts if isinstance(a, dict) and str(a.get("key", "")).strip()
        ]
        update_key("api.gemini_accounts", cleaned, add_missing=True)

    def get_all_credentials(self) -> List[Dict[str, Any]]:
        """Returns standard list of credentials depending on active auth_mode."""
        mode = self.get_auth_mode()
        now = time.time()
        creds: List[Dict[str, Any]] = []

        if mode == "gemini_accounts":
            accounts = self.get_gemini_accounts()
            for idx, acc in enumerate(accounts):
                k = acc.get("key", "")
                cd = self._cooldowns.get(k, 0)
                is_cooldown = cd > now
                label = acc.get("label") or acc.get("email") or f"Tài khoản {idx + 1}"
                creds.append({
                    "id": idx,
                    "label": label,
                    "key": k,
                    "masked": self.mask_key(k),
                    "is_cooldown": is_cooldown,
                    "cooldown_remaining": max(0, int(cd - now)),
                    "type": "account"
                })
        else:
            keys = self.get_pool_keys()
            for idx, k in enumerate(keys):
                cd = self._cooldowns.get(k, 0)
                is_cooldown = cd > now
                creds.append({
                    "id": idx,
                    "label": f"Key {idx + 1}",
                    "key": k,
                    "masked": self.mask_key(k),
                    "is_cooldown": is_cooldown,
                    "cooldown_remaining": max(0, int(cd - now)),
                    "type": "api_key"
                })

        return creds

    @staticmethod
    def mask_key(key: str) -> str:
        """Hide sensitive key characters (show first 6 and last 4)."""
        k = key.strip()
        if len(k) <= 10:
            return "******"
        return f"{k[:6]}...{k[-4:]}"

    def get_active_credential(self) -> Tuple[str, str, int]:
        """Thread-safe acquisition of current active credential.

        Returns: (api_key, label, active_index)
        """
        with self._lock:
            creds = self.get_all_credentials()
            if not creds:
                # Fallback to config key
                single = str(load_key_or("api.key", "")).strip()
                return single, "Mặc định", 0

            now = time.time()
            total = len(creds)

            # Find next non-cooldown credential starting from current index
            for offset in range(total):
                idx = (self._active_index + offset) % total
                c = creds[idx]
                if self._cooldowns.get(c["key"], 0) <= now:
                    self._active_index = idx
                    return c["key"], c["label"], idx

            # If all are in cooldown, find the one that recovers earliest
            earliest_idx = min(range(total), key=lambda i: self._cooldowns.get(creds[i]["key"], 0))
            earliest_cd = self._cooldowns.get(creds[earliest_idx]["key"], 0)
            wait_sec = max(0.0, earliest_cd - now)

            if wait_sec > 0:
                rprint(f"[yellow]⏳ Tất cả {total} Key/Tài khoản đang tạm ngưng do giới hạn token. Chờ {wait_sec:.1f}s hồi phục...[/yellow]")
                time.sleep(min(wait_sec, 60.0))

            self._active_index = earliest_idx
            return creds[earliest_idx]["key"], creds[earliest_idx]["label"], earliest_idx

    def mark_failed_and_rotate(self, error: Exception, failed_key: str, failed_label: str) -> Tuple[str, str, int]:
        """Analyze error and rotate to the next credential in the pool.

        Returns: (new_key, new_label, new_index)
        """
        with self._lock:
            now = time.time()
            err_str = str(error).lower()

            # Determine cooldown duration based on error category
            # 429 / Rate Limit / ResourceExhausted -> 65s cooldown (Google resets RPM after 60s)
            # Quota exceeded / 403 / 401 -> 300s cooldown
            # 500 / 503 / Server Error -> 25s cooldown
            if "429" in err_str or "rate" in err_str or "exhausted" in err_str or "resource" in err_str:
                cd_duration = 65.0
                reason = "Hết Token / Giới hạn tốc độ (Rate Limit 429)"
            elif "quota" in err_str or "insufficient" in err_str or "exceeded" in err_str:
                cd_duration = 300.0
                reason = "Hết hạn ngạch (Quota Exceeded)"
            elif "401" in err_str or "403" in err_str or "unauthorized" in err_str or "invalid" in err_str:
                cd_duration = 600.0
                reason = "Khóa không hợp lệ hoặc hết hạn (401/403)"
            else:
                cd_duration = 30.0
                reason = f"Lỗi kết nối ({type(error).__name__})"

            self._cooldowns[failed_key] = now + cd_duration
            self._error_counts[failed_key] = self._error_counts.get(failed_key, 0) + 1

            creds = self.get_all_credentials()
            total = len(creds)

            if total <= 1:
                rprint(f"[red]⚠️ [{failed_label}] gặp sự cố: {reason}. Không có Key dự phòng nào khác trong danh sách.[/red]")
                return failed_key, failed_label, 0

            # Advance to next index
            self._active_index = (self._active_index + 1) % total
            new_cred = creds[self._active_index]

            rprint(
                f"[bold yellow]🔄 [{failed_label}] {reason}! "
                f"Tự động chuyển sang [{new_cred['label']}] (Key {self._active_index + 1}/{total})...[/bold yellow]"
            )
            return new_cred["key"], new_cred["label"], self._active_index

    def reset_cooldown(self, key: str) -> None:
        with self._lock:
            self._cooldowns.pop(key, None)
            self._error_counts.pop(key, None)

    def test_single_key(self, api_key: str, base_url: str, model: str) -> Tuple[bool, str]:
        """Test a specific API key without modifying active pool state."""
        try:
            from core.utils.ask_gpt import normalize_base_url
            client = OpenAI(api_key=api_key, base_url=normalize_base_url(base_url), timeout=15)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
            )
            if resp.choices:
                return True, "Hoạt động tốt (200 OK)"
            return False, "Không có phản hồi từ mô hình"
        except Exception as e:
            err = str(e)
            if "429" in err:
                return False, "Bị giới hạn tốc độ / Hết lượt (429 Rate Limit)"
            if "401" in err or "403" in err:
                return False, "Khóa API không hợp lệ hoặc bị từ chối (401/403)"
            return False, f"Lỗi: {err[:120]}"


# Global singleton instance
rotator = LLMRotator()

