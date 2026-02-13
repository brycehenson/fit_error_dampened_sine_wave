ARG VARIANT=3.13-bookworm
FROM mcr.microsoft.com/devcontainers/python:1-${VARIANT}


# Base setup and dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    echo "Checking internet" && \
    curl --silent --head --fail --connect-timeout 5 https://deb.debian.org/debian/ >/dev/null && \
    echo "Disabling stale Yarn apt repo from base image" && \
    rm -f /etc/apt/sources.list.d/yarn.list /etc/apt/sources.list.d/yarn.list.save && \
    apt-get update && \
    echo "Removing the debian system python since we don't need two pythons" && \
    apt-get -y remove python3 && sudo apt-get -y autoremove && \
    echo "get npm" && \
    apt-get install -y --no-install-recommends nodejs npm ca-certificates git && \
    echo "Adding ping and curl!" && \
    apt-get update && apt-get -y install iputils-ping curl && \
    echo "Adding lnav for log viewing" && \
    apt-get -y install lnav && \
    echo "Adding bc command line calculator" && \
    apt-get -y install bc && \
    echo "Adding usbutils!" && \
    apt-get -y install usbutils && \
    echo "adding latex fonts" && \
    apt-get -y install fonts-cmu texlive-fonts-recommended --no-install-recommends && \
    echo "Adding the git command line completion" && \
    echo "source /usr/share/bash-completion/completions/git" >> ~/.bashrc && \
    echo "Silence git safe dir stuff" && \
    git config --system --add safe.directory '*' && \
    echo "Installing pre-commit and notebook tools for git" && \
    pipx install --global nbstripout-fast nbdime pre-commit && \
    echo "install requirements for kaleido" && \
    apt-get install -y libnss3 libatk-bridge2.0-0 libcups2 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 libcairo2 libasound2 && \
    echo "Cleanup apt" && \
    apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* && \
    mkdir /root/setup_scripts && \
    echo "Done base setup"


# Copy the setup scripts
COPY .devcontainer/*.sh /setup_scripts/

# install codex
RUN npm install -g @openai/codex \
 && codex --version


RUN echo "Run setup setup_scripts" && \
    bash /setup_scripts/install_uv.sh && \
    echo "Fixing permissions" && \
    usermod -aG dialout vscode && \
    /setup_scripts/fix_permissions.sh && \
    chmod -R 777 /setup_scripts && \
    echo "Done"


USER vscode
