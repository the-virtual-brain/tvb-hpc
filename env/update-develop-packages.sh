#!/usr/bin/env bash

# bootstrap recent Python, assuming sane dev env & ssl headers

if [[ ! -f tvb_hpc/__init__.py ]]
then
    echo "Please run this script from the root of the tvb_hpc repo."
    exit 1
fi

# may develop these
extra_pkgs="cgen genpy islpy pymbolic loopy"
for pkg in $extra_pkgs
do
    echo "setting up $pkg for dev"
    pushd env/$pkg
        if [[ "$pkg" == "islpy" ]]; then $PREFIX/bin/python3 setup.py build; fi
        $PREFIX/bin/python3 setup.py develop
    popd # env/$pkg
done

set -eu
set -o pipefail
