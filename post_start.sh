#!/usr/bin/env bash

# RunPod's base image calls this after starting SSH and Jupyter. Tailscale is
# best-effort so a networking problem never prevents the pod itself starting.

set -uo pipefail

readonly TS_STATE_DIR="${TS_STATE_DIR:-/workspace/.tailscale}"
readonly TS_SOCKET_DIR="/var/run/tailscale"
readonly TS_SOCKET="${TS_SOCKET_DIR}/tailscaled.sock"
readonly TS_LOG="/var/log/tailscaled.log"
readonly TS_HOSTNAME="${TS_HOSTNAME:-mats12}"
readonly GITHUB_SSH_DIR="${HOME:-/root}/.ssh"
readonly GITHUB_SSH_KEY="${GITHUB_SSH_DIR}/runpod_github"
readonly GITHUB_KNOWN_HOSTS="${GITHUB_SSH_DIR}/known_hosts"

configure_github_ssh() {
    if [[ -z "${GITHUB_SSH_KEY_B64:-}" ]]; then
        echo "GitHub SSH warning: GITHUB_SSH_KEY_B64 is unset; skipping setup."
        return 0
    fi

    install -d -m 0700 "${GITHUB_SSH_DIR}"
    umask 077

    local key_tmp="${GITHUB_SSH_KEY}.tmp"
    if ! printf '%s' "${GITHUB_SSH_KEY_B64}" | base64 --decode >"${key_tmp}"; then
        rm -f "${key_tmp}"
        echo "GitHub SSH warning: GITHUB_SSH_KEY_B64 is not valid base64."
        return 1
    fi

    if ! ssh-keygen -y -P '' -f "${key_tmp}" >/dev/null 2>&1; then
        rm -f "${key_tmp}"
        echo "GitHub SSH warning: decoded secret is not an unencrypted private key."
        return 1
    fi

    mv "${key_tmp}" "${GITHUB_SSH_KEY}"
    chmod 0600 "${GITHUB_SSH_KEY}"

    local github_host_key='github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl'
    touch "${GITHUB_KNOWN_HOSTS}"
    if ! grep -qxF "${github_host_key}" "${GITHUB_KNOWN_HOSTS}"; then
        printf '%s\n' "${github_host_key}" >>"${GITHUB_KNOWN_HOSTS}"
    fi
    chmod 0600 "${GITHUB_KNOWN_HOSTS}"

    cat >"${GITHUB_SSH_DIR}/config" <<EOF
Host github.com
    HostName github.com
    User git
    IdentityFile ${GITHUB_SSH_KEY}
    IdentitiesOnly yes
    StrictHostKeyChecking yes
EOF
    chmod 0600 "${GITHUB_SSH_DIR}/config"

    # Existing HTTPS remotes use SSH automatically, so old workspaces need no
    # one-off `git remote set-url` migration.
    git config --global url."git@github.com:".insteadOf "https://github.com/"
    echo "GitHub SSH ready."
}

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

if ! configure_github_ssh; then
    echo "GitHub SSH warning: setup failed. Git remains available without credentials."
fi

if ! start_tailscale; then
    echo "Tailscale warning: setup failed. RunPod SSH and Jupyter are still available."
fi

exit 0
