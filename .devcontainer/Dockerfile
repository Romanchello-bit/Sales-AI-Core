# Use official Python 3.14 image
FROM mcr.microsoft.com/devcontainers/python:1-3.14-bookworm

# Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Copy packages.txt (if present)
COPY ../packages.txt /tmp/packages.txt

# Install system packages (including any from packages.txt), clean cache, and prepare workspace
RUN apt-get update && \
    pkg_list="$(if [ -s \"/tmp/packages.txt\" ]; then xargs -a \"/tmp/packages.txt\"; fi) build-essential git graphviz libpq-dev" && \
    apt-get install -y --no-install-recommends "$pkg_list" && \
    rm -rf "/var/lib/apt/lists/*" "/tmp/packages.txt"

# Create non-root 'vscode' user as devcontainers expects (quoted vars)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN if ! id -u "$USERNAME" > /dev/null 2>&1; then \
      groupadd --gid "$USER_GID" "$USERNAME" || true && \
      useradd -s /bin/bash --uid "$USER_UID" --gid "$USER_GID" -m "$USERNAME" || true; \
    fi

# Set working directory
WORKDIR /workspace

# Ensure scripts folder exists, set PATH and upgrade pip (single RUN for caching)
RUN mkdir -p /workspace/scripts \
    && chown -R "$USERNAME":"$USERNAME" /workspace \
    && printf '%s\n' 'export PATH=/workspace/scripts:$PATH' > /etc/profile.d/workspace_path.sh \
    && python -m pip install --upgrade pip setuptools wheel

# Default to non-root user
USER $USERNAME

# Keep container lightweight; venv and Python deps are installed in postCreateCommand
