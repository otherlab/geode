Geode: The Otherlab computational geometry library
==================================================

Geode is an open source library of computational geometry and associated mathematical utilities
together with a fast, lightweight python binding layer.  The license is standard three-clause BSD
(see the included `LICENSE` file or [LICENSE](https://github.com/otherlab/core/blob/master/LICENSE)).

For questions or discussion, email geode-dev@googlegroups.com.

### Dependencies

For C++:

* [cmake >= 3.3](https://cmake.org): A build system (GPL)
* [gmp >= 4.0](http://gmplib.org): Arbitrary precision arithmetic (LGPL)
* [cblas](http://www.netlib.org/blas/blast-forum/cblas.tgz): C wrappers for BLAS (BSD license)

For Python:

* [python >= 2.7](http://python.org): A scripting language (Python Software Foundation (PSF) license)
* [numpy >= 1.5](http://numpy.scipy.org): Efficient multidimensional arrays for Python (BSD license)
* [setuptools >= 0.6](http://pythonhosted.org/setuptools): A packaging system for Python (PSF license)

Optional dependencies (see below for how to disable these):

* [py.test >= 2.1](http://pytest.org): Simple python testing (MIT license)
* [scipy](http://www.scipy.org): Scientific computation for Python (BSD license)
* [openexr](http://www.openexr.com): High dynamic range floating point image format (BSD license)
* [libpng](http://www.libpng.org): Lossless image format (Custom noncopyleft license)
* [libjpeg](http://www.ijg.org): Lossy image format (Custom noncopyleft license)

Geode makes extensive use of C++11 features, so a relatively recent C++ compiler is necessary.

### Windows

  This has been built and run successfully with the Visual Studio 2015 preview. We are working to make this more streamlined and robust (as well as fixing the hundreds of compiler warnings), but the following worked for me:

  Install the following dependencies:
   * WinPython 64bit version 2.7.9.2 (This includes numpy and scipy)
   * CMake
   * MPIR (commit 3a9dd527a2f87e6eff8cab8b54b0b1f31e0826fa but tweaked for VS2015)
   ** This will require installing vsyasm
   ** Remove definition of snprintf from /build.vc12/cfg.h
   ** You may need to set Tools->Options->Projects and Solutions->Build and Run->"Maximum number of parallel project builds" to 1 to avoid parallel builds clobbering config headers

   Create a config.py and point gmp to installation of mpir:
     gmp_libpath = '#/../mpir/build.vc12/x64/Debug'
     gmp_include='#/../mpir'
     gmp_publiclibs='mpir.lib'

   Setup Command Prompt environment:
     Python from:
       ...\WinPython-64bit-2.7.9.2\scripts\env.bat
     VS dev tools using:
       "C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\Tools\VsDevCmd.bat"
     Select x64 tools using:
       "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64

  Create a build directory separate from the sources (optional, but recommended):
    mkdir build

   Build with:
     cmake ../
     make

   If you wish to install geode to a different location (such as a python
   virtualenv), add -DCMAKE_BUILD_PREFIX=/path/to/wherever to the cmake command.

   Install with:
     make install

   To use geode outside of project directory or for test_worker to pass you must add geode to your PYTHONPATH:
     set PYTHONPATH=<path_to_dir_outside_repo>\geode;%PYTHONPATH%

### Installation

If necessary, dependencies can be installed via one of

    # Debian/Ubuntu
    sudo apt install cmake g++ python-dev python-pip python-numpy pkg-config libjpeg-dev libpng-dev zlib1g-dev libgmp-dev
    sudo apt-get install python-scipy python-pytest libpng-dev libjpeg-dev libopenexr-dev # optional
    pip install numpy

    # Homebrew (recommended)
    brew install cmake openexr gfortran python #Note: gfortran brew is now part of gcc, so although previous versions can still be accessed, brew install gcc is the preferred method
    sudo pip install --upgrade pip setuptools numpy scipy pytest #numpy and scipy can be

    # MacPorts (not recommended).  If you have python 2.7, replace py26 with py27.
    sudo port -v install python26 py26-numpy cmake
    sudo port -v install py26-scipy py26-py libpng jpeg openexr # optional
    sudo port -v install gcc47 # If clang is unavailable

Geode can then be installed from source via

    git clone https://github.com/otherlab/geode.git
    cd geode

    # Install c++ headers and libraries to /usr/local
    cmake . && make && make install

At this point, you have a choice of either developer mode or install mode

### Install mode

    sudo make install

    This will also install python bindings if enabled.

### Developer mode

The libraries are built into `geode` if you want to use them without installing.  To point python imports to your development tree, run one of

    sudo python setup.py develop
    python setup.py develop --prefix=$HOME

To link against this built version of geode, add this to your project's own
CMakeLists.txt:

    include(path/to/geode/CMakeLists.txt)
    target_link_libraries(foo geode)

If you've installed geode, you can instead use:

    find_package(Geode REQUIRED)
    target_link_libraries(foo geode)


which will allow you to develop with geode in C++ as if it was installed.

### Post install

On linux you may have to update the shared library cache via ldconfig if this is the first time you've installed. Make sure /usr/local/lib (or wherever you installed libgeode.so) is included in the cache's search path.

### Testing

Unit tests can be run via

    cd geode
    py.test

### Extra configuration

If additional build configuration is necessary, run ccmake instead of cmake.
CMake includes documentation generated from the geode build system.

For developers wishing to use without installing, see more options in Developer mode section below

These options can also be passed via command line to cmake.  Run `ccmake` for a complete list.
Use `CMAKE_BUILD_TYPE=Debug` for a much slower build with many more assertions:

    cmake -DCMAKE_BUILD_TYPE=Debug

### Acknowledgements

Parts of geode come from the [PhysBAM simulation library](http://physbam.stanford.edu) developed by
Ron Fedkiw et al. at Stanford University.

For random numbers, we use the [Random123 library](http://www.deshawresearch.com/resources_random123.html) of
John Salmon et al. at D. E. Shaw Research.  Random123 is included inline in `core/random/random123`.

The interval arithmetic in `exact/Interval` is based on code by [Robert Bridson](http://www.cs.ubc.ca/~rbridson)
and [Tyson Brochu](http://www.cs.ubc.ca/~tbrochu).

