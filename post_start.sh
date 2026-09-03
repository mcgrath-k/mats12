#!/usr/bin/env bash

# RunPod's base image calls this after starting SSH and Jupyter. Tailscale is
# best-effort so a networking problem never prevents the pod itself starting.

set -uo pipefail

readonly TS_STATE_DIR="${TS_STATE_DIR:-/workspace/.tailscale}"
readonly TS_SOCKET_DIR="/var/run/tailscale"
readonly TS_SOCKET="${TS_SOCKET_DIR}/tailscaled.sock"
readonly TS_LOG="/var/log/tailscaled.log"
readonly TS_HOSTNAME="${TS_HOSTNAME:-mats12}"

start_tailscale() {
    mkdir -p "${TS_STATE_DIR}" "${TS_SOCKET_DIR}"
    chmod 0700 "${TS_STATE_DIR}"

    if ! pgrep -x tailscaled >/dev/null 2>&1; then
        nohup tailscaled \
            --tun=userspace-networking \
            --statedir="${TS_STATE_DIR}" \
            --socket="${TS_SOCKET}" \
            >"${TS_LOG}" 2>&1 &
    fi

    for _ in $(seq 1 30); do
        [[ -S "${TS_SOCKET}" ]] && break
        sleep 1
    done

    if [[ ! -S "${TS_SOCKET}" ]]; then
        echo "Tailscale warning: daemon did not start; see ${TS_LOG}"
        return 1
    fi

    local backend_state
    backend_state="$({ tailscale --socket="${TS_SOCKET}" status --json 2>/dev/null || true; } \
        | python -c 'import json, sys; print(json.load(sys.stdin).get("BackendState", ""))' 2>/dev/null \
        || true)"

    if [[ "${backend_state}" == "Running" ]]; then
        if ! tailscale --socket="${TS_SOCKET}" set \
            --hostname="${TS_HOSTNAME}" \
            --accept-dns=false \
            --ssh >/dev/null; then
            return 1
        fi
    elif [[ -n "${TS_AUTHKEY:-}" ]]; then
        if ! tailscale --socket="${TS_SOCKET}" up \
            --auth-key="${TS_AUTHKEY}" \
            --hostname="${TS_HOSTNAME}" \
            --accept-dns=false \
            --ssh >/dev/null; then
            return 1
        fi
    else
        # An authorized node reconnects from persistent state without keeping
        # an auth key in the RunPod template.
        if ! tailscale --socket="${TS_SOCKET}" up \
            --hostname="${TS_HOSTNAME}" \
            --accept-dns=false \
            --ssh >/dev/null; then
            return 1
        fi
    fi

    local address
    address="$(tailscale --socket="${TS_SOCKET}" ip -4 2>/dev/null || true)"
    echo "Tailscale ready: ssh root@${TS_HOSTNAME} (${address:-address pending})"
}

if ! start_tailscale; then
    echo "Tailscale warning: setup failed. RunPod SSH and Jupyter are still available."
fi

exit 0
