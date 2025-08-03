#!/bin/bash
set -e


# set the launch directory as the project path env variable
export PROJECTPATH=$PWD
echo "PROJECTPATH environment variable set to ${QCSTACKPATH}"

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

# Merge by default
git config pull.rebase false
# Get latest
git -C . status
git pull || echo "Warning: 'git pull' failed, but continuing..."
git push || echo "Warning: 'git push' failed, but continuing..."
popd

# Install the packages
pip install -e .

# fix jupyter behaviour
.devcontainer/fix_jupyter.sh
