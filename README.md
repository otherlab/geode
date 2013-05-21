Otherlab core library
=====================

`other/core` is an open source utility library of arrays, vectors, matrices, and other mathematical code together
with a fast, lightweight python binding layer.  The license is standard three-clause BSD (see the included `LICENSE`
file or [LICENSE](https://github.com/otherlab/core/blob/master/LICENSE)).

### Acknowledgements

Much of core is based on parts of the [PhysBAM simulation library](http://physbam.stanford.edu) developed by
Ron Fedkiw et al. at Stanford University.

For random numbers, we use the [Random123 library](http://www.deshawresearch.com/resources_random123.html) of
John Salmon et al. at D. E. Shaw Research.  Random123 code is included inline in `core/random/random123`.

The interval arithmetic in `exact/Interval` is based on code by [Robert Bridson](http://www.cs.ubc.ca/~rbridson)
and [Tyson Brochu](http://www.cs.ubc.ca/~tbrochu).

### Dependencies

Required dependencies:

* [boost >= 1.46](http://www.boost.org): Various C++ utility libraries (Boost Software License)
* [scons >= 2.0](http://www.scons.org): A build system (MIT license)
* [cblas](http://www.netlib.org/blas/blast-forum/cblas.tgz): C wrappers for BLAS (BSD license)

Optional dependencies (see below for how to disable these):

* [python >= 2.6](http://python.org): A scripting language (Python Software Foundation (PSF) license)
* [numpy >= 1.5](http://numpy.scipy.org): Efficient multidimensional arrays for Python (BSD license)
* [scipy](http://www.scipy.org): Scientific computation for Python (BSD license)
* [py.test >= 2.1](http://pytest.org): Simple python testing (MIT license)
* [openexr](http://www.openexr.com): High dynamic range floating point image format (BSD license)
* [libpng](http://www.libpng.org): Lossless image format (Custom noncopyleft license)
* [libjpeg](http://www.ijg.org): Lossy image format (Custom noncopyleft license)
* [openmesh](http://www.openmesh.org): Halfedge triangle mesh data structure (LGPL)

`core` makes extensive use of C++11 features, so a relatively recent C++ compiler is necessary.  So far the code has been tested on

* [gcc 4.6](http://gcc.gnu.org)
* [gcc 4.7](http://gcc.gnu.org)
* [clang 3.1](http://clang.llvm.org)

### Setup

`core` is a mixed Python/C++ codebase, so for now the setup requires setting several environment variables.
Ideally, we will improve this part so that everything works more automatically.

1. Install dependencies:

        # MacPorts
        sudo port -v install python26 py26-numpy boost scons \
          py26-scipy py26-py libpng libjpeg openexr # optional

        # Debian/Ubuntu
        sudo apt-get install python python-numpy libboost-1.48 scons \
          python-scipy python-py libpng-dev libjpeg-dev libopenexr-dev # optional

2. Unpack `core` inside a root directory (we use `other`), and set $OTHER to that root directory.

        mkdir other 
        export OTHER=`pwd`/other
        cd $OTHER
        git clone https://github.com/otherlab/core.git

3. Set up build:

        cd $OTHER
        ./core/build/setup

4. Configure build: If desired, edit `config.py` and set any desired options.  For example:

        # config.py
        CXX = 'clang++'
        cache = '/tmp/scons-cache'

   The following flags can be used to disable optional components:

        has_python = 0
        has_openexr = 0
        has_libpng = 0
        has_libjpeg = 0
        has_openmesh = 0

5. Build:

        cd $OTHER
        scons -j5 # optimized build
        scons -j5 type=debug # debug build

   Run `scons -h` to get a complete list of available options.

   Libraries and directories will be installed into `$OTHER/install/<variant>`, where <variant> is debug or release.

6. Make a flavor symlink to your desired active variant.

        cd $OTHER/install
        ln -s <variant> flavor # <variant> can be debug or release

   Alternatively, you can add this to your .bashrc:

        other-flavor () {
          if test -n "$1" -a -e "$OTHER/install/$1"; then
            rm -v -f $OTHER/install/flavor
            ln -sf $1 $OTHER/install/flavor
          elif [ "$1" ]; then
            echo flavor \"$1\" not found.
          else
            ls -GFC -l $OTHER/install
          fi
        }

   and do

        other-flavor <variant>

7. Set up environment variables:

        export OTHER=$HOME/otherlab/other
        export PATH=$PATH:$OTHER/install/flavor/bin
        export PYTHONPATH=$PYTHONPATH:$OTHER/..:$OTHER/install/flavor/lib

8. Test:

        cd $OTHER/core
        py.test --version # Need at least 2.1
        py.test
