#!/bin/bash

# Download the nuimages v1.0 mini dataset for test

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DATA_DIR="${SCRIPT_DIR}/data/sets/nuimages"

if [ -d "${BASE_DATA_DIR}" ]; then
  echo "${BASE_DATA_DIR} already exists. Skipping data download."
else
  cd "${SCRIPT_DIR}"
  mkdir data
  cd data
  mkdir sets
  cd sets
  mkdir nuimage
  echo "Downloading NuImage v1.0-mini test data"
  wget -N https://www.nuscenes.org/data/nuimages-v1.0-mini.tgz
  echo "Unpacking data"
  tar -xzf nuimages-v1.0-mini.tgz -C nuimage
  rm nuimages-v1.0-mini.tgz
fi
