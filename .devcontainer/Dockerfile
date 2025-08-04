ARG VARIANT=3.13-bookworm
FROM mcr.microsoft.com/devcontainers/python:1-${VARIANT}


# Base setup and dependenciess
RUN echo "Removing the debian system python since we don't need two pythons" && \
    apt-get -y remove python3 && sudo apt-get -y autoremove && \
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
    echo "Cleanup apt" && \
    apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* && \
    mkdir /root/setup_scripts && \
    echo "Done base setup"

# Install Chrome for Kaleido
RUN apt-get update && apt-get install -y wget gnupg && \
    wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor > /usr/share/keyrings/google.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google.list && \
    apt-get update && \
    apt-get install -y google-chrome-stable
ENV CHROME_PATH="/usr/bin/google-chrome"

# Copy the script to install poetry
COPY .devcontainer/*.sh /setup_scripts/


RUN echo "Run setup setup_scripts" && \
    /setup_scripts/install_poetry.sh && \
    echo "Fixing permissions" && \
    usermod -aG dialout vscode && \
    /setup_scripts/fix_permissions.sh && \
    chmod -R 777 /setup_scripts && \
    echo "Done"


USER vscode
RUN echo "Make it turn off venvs because it's a container" && \
    poetry config virtualenvs.create false && \
    poetry config virtualenvs.in-project true && \
    echo "Done"
