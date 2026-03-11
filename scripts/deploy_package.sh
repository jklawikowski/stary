#!/bin/bash

set -e

# Authentication via TWINE_USERNAME and TWINE_PASSWORD environment variables
if [ -z "$TWINE_USERNAME" ] || [ -z "$TWINE_PASSWORD" ]; then
    echo "Error: TWINE_USERNAME and TWINE_PASSWORD environment variables must be set."
    exit 1
fi

twine upload --repository-url https://af01p-igk.devtools.intel.com/artifactory/api/pypi/qaplatform-igk-local/ dist/* --verbose