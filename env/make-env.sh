#!/usr/bin/env bash

# bootstrap recent Python, assuming sane dev env & ssl headers

if [[ ! -f tvb_hpc/__init__.py ]]
then
    echo "Please run this script from the root of the tvb_hpc repo."
    exit 1
fi

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

echo "will build environment in '$PREFIX'. 5 seconds to cancel.."
sleep 5

# ok go
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

    libffi_url=ftp://sourceware.org/pub/libffi/libffi-3.2.1.tar.gz

    pyver=3.6.0
    Py_url=https://www.python.org/ftp/python/$pyver/Python-$pyver.tgz

    for pkg in libffi zlib bzip sql Py
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

popd #  $PREFIX/src

cat > $PREFIX/activate <<EOF
export PATH=$PREFIX/bin:$PATH
EOF

# cffi needs to find libffi..
CFLAGS="-I$PREFIX/lib/libffi-3.2.1/include" \
LDFLAGS="-L$PREFIX/lib64 -L$PREFIX/lib -lpython3.6m" \
$PREFIX/bin/pip3 install cffi

py_pkgs="$(cat requirements.txt)"

for pkg in $py_pkgs
do
    echo "pip installing $pkg"
    $PREFIX/bin/pip3 install $pkg
done

# unlikely to change, install
extra_pkgs="pytools f2py"
for pkg in $extra_pkgs
do
    echo "setting up $pkg for dev"
    pushd env/$pkg
        $PREFIX/bin/python3 setup.py install
    popd # env/$pkg
done

# may develop these
extra_pkgs="cgen genpy islpy pymbolic pyopencl loopy"
for pkg in $extra_pkgs
do
    echo "setting up $pkg for dev"
    pushd env/$pkg
        $PREFIX/bin/python3 setup.py develop
    popd # env/$pkg
done

echo "done! use by issuing 'source $PREFIX/activate'"
