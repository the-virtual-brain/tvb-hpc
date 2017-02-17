#!/bin/bash

set -eu 
set -o pipefail

if [[ ! -d docs ]]
then
    echo "please run from root of tvb-hpc repo";
    exit 1
fi

pushd docs
    sphinx-apidoc -f -o api-docs -e ../tvb_hpc
    make html
    pushd _build/html
        git init
        git remote add origin git@github.com:the-virtual-brain/tvb-hpc
        git checkout -b gh-pages
        git add .
        git commit -m 'add gh-pages sources'
        git push -f origin gh-pages
        rm -rf .git
    popd
popd

