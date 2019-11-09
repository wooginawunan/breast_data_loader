#!/usr/bin/env bash
export PNAME="breast_data_loader"
export ROOT="/gpfs/data/geraslab/Nan"
# "$( cd "$(dirname "$0")" ; pwd -P )"
echo "Welcome to $PNAME rooted at $ROOT"

# Configures paths. Adapt to your needs!
PYTHONPATH=$ROOT/public_repo/breast_data_loader

export PYTHONPATH
export RESULTS_DIR=$ROOT/public_repo/breast_data_loader/results

echo "PYTHONPATH set as $PYTHONPATH"

# Switches off importing out of environment packages
export PYTHONNOUSERSITE=1

# if [ ! -d "${DATA_DIR}" ]; then
#   echo "Creating ${DATA_DIR}"
#   mkdir -p ${DATA_DIR}
# fi

if [ ! -d "${RESULTS_DIR}" ]; then
  echo "Creating ${RESULTS_DIR}"
  mkdir -p ${RESULTS_DIR}
fi
