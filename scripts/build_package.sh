#!/bin/bash

set -e

rm -rf ./build/
rm -rf ./dist/
rm -rf ./src/*.egg-info/
python3 -m build