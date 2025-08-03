#! /bin/bash
set -e

POETRY_VERSION=2.1.0

# Ensure pipx is in PATH or install Poetry directly if pipx is unavailable
if ! command -v pipx &> /dev/null
then
    echo "pipx not found, installing Poetry directly..."
    curl -sSL https://install.python-poetry.org | POETRY_HOME=/usr/share/poetry python3 - --version $POETRY_VERSION
    echo "Adding poetry to the path via symlink"
    ( ([ -e "/usr/bin/poetry" ] && echo "symlink already exists") || ln -s /usr/share/poetry/bin/poetry /usr/bin/poetry )

else
    pipx ensurepath
    pipx install --global poetry==$POETRY_VERSION
fi


echo "Adding dynamic versioning plugin"
poetry self add poetry-dynamic-versioning[plugin]

echo "Adding poetry monorepo support"
poetry self add poetry-monorepo-dependency-plugin

echo "Setup poetry bash completion"
poetry completions bash > /etc/bash_completion.d/poetry

