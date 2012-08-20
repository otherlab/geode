Otherlab core library
=====================

`other/core` is a utility library of vectors, matrices, arrays, and other mathematical code together
with a fast, lightweight python binding layer.  The code is originally based on parts of the
[PhysBAM simulation library](http://physbam.stanford.edu) but has since been heavily modified.

The code is released under a standard three-clause BSD license (see
[LICENSE](https://github.com/otherlab/core/blob/master/LICENSE)).

### Dependencies

Required dependencies:

* [python >= 2.6](http://python.org): A scripting language (Python Software Foundation (PSF) license)
* [numpy >= 1.5](http://numpy.scipy.org): Efficient multidimensional arrays for Python (BSD license)
* [boost >= 1.46](http://www.boost.org): Various C++ utility libraries (Boost Software License)
* [scons >= 2.0](http://www.scons.org): A build system (MIT license)

Optional dependencies:

* [scipy](http://www.scipy.org): Scientific computation for Python (BSD license)
* [py.test >= 2.1](http://pytest.org): Simple python testing (MIT license)
* [openexr](http://www.openexr.com): High dynamic range floating point image format (BSD license)
* [libpng](http://www.libpng.org): Lossless image format (Custom non-viral license)
* [libjpeg](http://www.ijg.org): Lossy image format (Custom non-viral license)

`core` makes extensive use of C++11 features, so a relatively recent C++ compile is necessary.  So far the code has been tested on

* [gcc 4.6](http://gcc.gnu.org)
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

2. Unpack `core` inside a root directory (we use `other`):

        mkdir other 
        export OTHER=<path-to-other>
        cd $OTHER
        git clone https://github.com/otherlab/core.git

3. Set up build:

        cd $OTHER
        ./core/build/setup

3. Configure build: Edit `config.py` and set any desired options.  For example

        # config.py
        CXX = 'clang++'
        cache = '/tmp/scons-cache'

   Run `scons -h` to get a complete list of available options.

4. Build:

        cd $OTHER
        scons -j5 # optimized build
        scons -j5 type=debug # debug build

   Libraries and directories will be installed into `$OTHER/install/<variant>`, where <variant> is debug or release.

5. Make a flavor symlink to your desired active variant.

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

6. Set up environment variables:

        export OTHER=$HOME/otherlab/other
        export PATH=$PATH:$OTHER/install/flavor/bin
        export PYTHONPATH=$PYTHONPATH:$OTHER/..:$OTHER/install/flavor/lib

7. Test:

        cd $OTHER/core
        py.test --version # Need at least 2.1
        py.test
