#!/bin/bash
set -e


# set the launch directory as the project path env variable
export PROJECTPATH=$PWD
echo "PROJECTPATH environment variable set to ${QCSTACKPATH}"

export BROWSER_PATH="/usr/local/bin/chrome-wrapper"
echo "BROWSER_PATH environment variable set to ${BROWSER_PATH}"

# Add bash completions
echo "source /usr/share/bash-completion/completions/git" >> ~/.bashrc && \

# Setup some git settings to make it work out of the box
git config --global --add safe.directory ${PROJECTPATH}

# Merge by default
git config pull.rebase false
# Install stripping of outputs for ipynb
git config --local include.path "../.devcontainer/clear_ipynb_output.gitconfig" || true

# setup the git pre-commit hooks
pre-commit install

# Make sure everything is owned by us (we used to use the root user in the container)
sudo chown -R vscode:vscode $PROJECTPATH

# Install the packages
./install.sh

# fix jupyter behaviour
.devcontainer/fix_jupyter.sh

# get the embeded chrome for kaleido
# This is needed for static image export in Plotly
python3 -c "import kaleido; kaleido.get_chrome_sync()"
