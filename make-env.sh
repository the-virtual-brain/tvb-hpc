#!/usr/bin/env bash

# bootstrap recent Python, assuming sane dev env & ssl headers

# if on macos check for ssl header, suggest install
if [[ "$(uname)" == "Darwin" ]]
then
    echo "[make-env.sh] uname is Darwin, checking for openssl headers.."
    if [[ -f /usr/local/opt/openssl/include/openssl/ssl.h ]]
    then
        echo "[make-env.sh] openssl headers present, good to go."
        export LDFLAGS="-L/usr/local/opt/openssl/lib $LDFLAGS"
        export CFLAGS="-I/usr/local/opt/openssl/include $CFLAGS"
        echo "[make-env.sh] LDFLAGS=$LDFLAGS"
        echo "[make-env.sh] CFLAGS=$CFLAGS"
    else
        echo "[make-env.sh] openssl headers not found."
        echo "[make-env.sh] consider \`brew install openssl\`."
    fi
fi

set -eu
set -o pipefail

export PREFIX=${PREFIX:-"$(pwd)/env-tvb-hpc"}
mkdir -p $PREFIX/src
pushd $PREFIX/src

    # this is awkward. but, it avoids having to know where the script is
    cat > soname-bzip.patch <<EOF
38c38
< 	\$(CC) -shared -Wl,-soname -Wl,libbz2.so.1.0 -o libbz2.so.1.0.6 \$(OBJS)
---
> 	\$(CC) -shared -o libbz2.so.1.0.6 \$(OBJS)
EOF

    j=${j:-"6"}

    zver=1.2.11
    zlib_url=http://zlib.net/zlib-$zver.tar.gz

    bzver=1.0.6
    bzip_url=http://www.bzip.org/$bzver/bzip2-$bzver.tar.gz

    sql_url=https://sqlite.org/2017/sqlite-autoconf-3160200.tar.gz

    pyver=3.6.0
    Py_url=https://www.python.org/ftp/python/$pyver/Python-$pyver.tgz

    for pkg in zlib bzip sql Py
    do
        if [[ -z "$(which curl)" ]]
        then
            wget $(eval echo \$${pkg}_url)
        else
            curl -O $(eval echo \$${pkg}_url)
        fi
        tar xzf ${pkg}*
        pushd ${pkg}*
        if [[ $pkg == "bzip" ]]
        then
            patch Makefile-libbz2_so ../soname-bzip.patch
            make -j$j -f Makefile-libbz2_so
            make -j$j install PREFIX=$PREFIX
            cp libbz2.so* $PREFIX/lib
        else
            ./configure --prefix=$PREFIX
            make -j$j
            make install
        fi
        popd
    done


    py_pkgs="numpy scipy sympy ipython six"

    for pkg in $py_pkgs
    do
        echo "pip installing $pkg"
        $PREFIX/bin/pip3 install $pkg
    done

    # latest loopy and its latest deps
    git clone https://github.com/inducer/loopy
    pushd loopy
    $PREFIX/bin/pip3 install -r requirements.txt
    $PREFIX/bin/python3 setup.py install
    popd # loopy

    cat > $PREFIX/activate <<EOF
export PATH=$PREFIX/bin:$PATH
EOF

    echo "done! use by issuing 'source $PREFIX/activate'"

popd # $PREFIX/src
