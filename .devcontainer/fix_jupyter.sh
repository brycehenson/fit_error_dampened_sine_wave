#!/bin/bash

# Stop jupyter emitting the following warning:
# DeprecationWarning: Jupyter is migrating its paths to use standard platformdirs
# given by the platformdirs library.  To remove this warning and
# see the appropriate new directories, set the environment variable
# `JUPYTER_PLATFORM_DIRS=1` and then run `jupyter --paths`.
# The use of platformdirs will be the default in `jupyter_core` v6
echo "Supressing jupyter warning"
export JUPYTER_PLATFORM_DIRS=1
jupyter --paths
