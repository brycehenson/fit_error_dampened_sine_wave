ARG VARIANT=3.13-bookworm
FROM mcr.microsoft.com/devcontainers/python:1-${VARIANT}


RUN echo "Removing the debian system python since we don't need two pythons" && \
    apt-get -y remove python3 && sudo apt-get -y autoremove && \
    echo "Adding ping and curl!" && \
    apt-get update && apt-get -y install iputils-ping curl && \
    echo "Adding lnav for log viewing" &&\
    apt-get -y install lnav && \
    echo "Adding bc command line calculator" &&\
    apt-get -y install bc && \
    echo "Adding usbutils!" && \
    apt-get -y install usbutils && \
    echo "adding latex fonts" && \
    apt-get -y install fonts-cmu texlive-fonts-recommended --no-install-recommends && \
    echo "Adding the git command line completion" && \
    echo "source /usr/share/bash-completion/completions/git" >> ~/.bashrc && \
    echo "Silence git safe dir stuff" && \
    git config --system --add safe.directory '*' && \
    echo "installing pre-commit and notebook tools for git" && \
    pipx install --global nbstripout-fast nbdime pre-commit && \
    echo "Cleanup apt" && \
    apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* && \
    mkdir /root/setup_scripts && \
    echo "Done"



USER vscode
