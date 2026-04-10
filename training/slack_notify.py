from __future__ import annotations

import json
import urllib.error
import urllib.request

from .runtime import RunPaths
from .runtime import append_event


def send_notification(
    paths: RunPaths,
    *,
    text: str,
    webhook_url: str | None = None,
    **fields: object,
) -> None:
    append_event(paths, "notification", text=text, webhook_enabled=bool(webhook_url), **fields)

    if not webhook_url:
        return

    payload = json.dumps({"text": text}).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            append_event(
                paths,
                "notification_delivered",
                text=text,
                http_status=getattr(response, "status", None),
                **fields,
            )
    except urllib.error.URLError as exc:
        append_event(paths, "notification_failed", text=text, error=str(exc), **fields)
