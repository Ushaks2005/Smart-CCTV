import os
from flask import request, abort

VALID_TOKENS = {
    token.strip()
    for token in os.getenv("SMART_CCTV_AUTH_TOKENS", "").split(",")
    if token.strip()
}


def require_auth():
    token = request.headers.get("Authorization")

    if token not in VALID_TOKENS:
        abort(401, description="Unauthorized")