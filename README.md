Geode: The Otherlab computational geometry library
==================================================

Geode is an open source library of computational geometry and associated mathematical utilities
together with a fast, lightweight python binding layer.  The license is standard three-clause BSD
(see the included `LICENSE` file or [LICENSE](https://github.com/otherlab/core/blob/master/LICENSE)).

For questions or discussion, email geode-dev@googlegroups.com.

### Dependencies

For C++:

* [scons >= 2.0](http://www.scons.org): A build system (MIT license)
* [gmp >= 4.0](http://gmplib.org): Arbitrary precision arithmetic (LGPL)
* [cblas](http://www.netlib.org/blas/blast-forum/cblas.tgz): C wrappers for BLAS (BSD license)
* [boost >= 1.46](http://www.boost.org): Not needed if a C++11 standard library exists (Boost Software License)

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
* [openmesh](http://www.openmesh.org): Halfedge triangle mesh data structure (LGPL)

Geode makes extensive use of C++11 features, so a relatively recent C++ compiler is necessary.
So far the code has been tested on

* [gcc 4.7](http://gcc.gnu.org)
* [gcc 4.8](http://gcc.gnu.org)
* [clang 3.1](http://clang.llvm.org)
* [clang 3.2](http://clang.llvm.org)
* [clang 3.3](http://clang.llvm.org)
* [clang 3.4](http://clang.llvm.org)
* [clang 3.5](http://clang.llvm.org)

### Windows
  This has been built and run successfully with the Visual Studio 2015 preview. We are working to make this more streamlined and robust (as well as fixing the hundreds of compiler warnings), but the following worked for me:

  Install the following dependencies:
   * WinPython 64bit version 2.7.9.2 (This includes numpy and scipy)
   * SCons using scons-local-2.3.4
   * MPIR (commit 3a9dd527a2f87e6eff8cab8b54b0b1f31e0826fa but tweaked for VS2015)
   ** This will require installing vsyasm
   ** Remove definition of snprintf from /build.vc12/cfg.h
   ** You may need to set Tools->Options->Projects and Solutions->Build and Run->"Maximum number of parallel project builds" to 1 to avoid parallel builds clobbering config headers

   Create a config.py and point gmp to installation of mpir:
     gmp_libpath = '#/../mpir/build.vc12/x64/Debug'
     gmp_include='#/../mpir'
     gmp_publiclibs='mpir.lib'

   Create a 'scons.bat' script and place in PATH (or otherwise make scons available):
     python %~dp0..\scons-local-2.3.4\scons.py %*

   Setup Command Prompt environment:
     Python from:
       ...\WinPython-64bit-2.7.9.2\scripts\env.bat
     VS dev tools using:
       "C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\Tools\VsDevCmd.bat"
     Select x64 tools using:
       "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64

   Build with:
     scons -j7 type=debug shared=0 Werror=0

   SCons install targets weren't tested but copying geode_all.pyd into geode will make it usable:
     cp build/native/optdebug/lib/build/native/optdebug/geode_all.pyd geode/geode_all.pyd

   To use geode outside of project directory or for test_worker to pass you must add geode to your PYTHONPATH:
     set PYTHONPATH=<path_to_dir_outside_repo>\geode;%PYTHONPATH%

### Installation

If necessary, dependencies can be installed via one of

    # Debian/Ubuntu
    sudo apt-get install python python-numpy scons libgmp-dev
    sudo apt-get install python-scipy python-pytest libpng-dev libjpeg-dev libopenexr-dev # optional

    # Homebrew (recommended)
    brew install scons openexr gfortran python #Note: gfortran brew is now part of gcc, so although previous versions can still be accessed, brew install gcc is the preferred method
    brew install boost # Not needed for 10.9 or later
    sudo pip install --upgrade pip setuptools numpy scipy pytest #numpy and scipy can be

    # MacPorts (not recommended).  If you have python 2.7, replace py26 with py27.
    sudo port -v install python26 py26-numpy scons boost
    sudo port -v install py26-scipy py26-py libpng jpeg openexr # optional
    sudo port -v install gcc47 # If clang is unavailable

Geode can then be installed from source via

    git clone https://github.com/otherlab/geode.git
    cd geode

    # Install c++ headers and libraries to /usr/local
    scons -j 5

At this point, you have a choice of either developer mode or install mode
### Install mode
    sudo scons install

    # Install python bindings
    sudo python setup.py install

### Developer mode

The libraries are built into `build/$arch/$type` (`build/native/release` by default) if you want to use them without installing.  To point python imports to your development tree, run one of

    sudo python setup.py develop
    python setup.py develop --prefix=$HOME

To create symlinks in /usr/local/{include,lib} pointing into the development tree, run

    sudo scons -j5 develop

or

    sudo scons -j5 develop type=debug

which will allow you to develop with geode in C++ as if it was installed.



### Testing

Unit tests can be run via

    cd geode
    py.test

### Extra configuration

If additional build configuration is necessary, create a `config.py` file and set any desired options.  For example,

    # config.py
    cxx = 'clang++'
    cache = '/tmp/scons-cache'

For developers wishing to use without installing, see more options in Developer mode section below

These options can also be passed via command line to scons.  Run `scons -h` for a complete list.
Use `type=debug` for a much slower build with many more assertions:

    scons -j 5 type=debug

The following flags can be used to disable optional components:

    # Command line
    scons use_python=0 use_openexr=0 use_libpng=0 use_libjpeg=0 use_openmesh=0

    # In config.py
    use_python = 0
    use_openexr = 0
    use_libpng = 0
    use_libjpeg = 0
    use_openmesh = 0

### Common Issues
On recent Linux machines, the boost libraries are already multithread-capable, and will not include the 'mt' suffix. As this is the default in the geode SConstruct, the following should be added in config.py:
    boost_lib_suffix = ''



### Acknowledgements

Parts of geode come from the [PhysBAM simulation library](http://physbam.stanford.edu) developed by
Ron Fedkiw et al. at Stanford University.

For random numbers, we use the [Random123 library](http://www.deshawresearch.com/resources_random123.html) of
John Salmon et al. at D. E. Shaw Research.  Random123 is included inline in `core/random/random123`.

The interval arithmetic in `exact/Interval` is based on code by [Robert Bridson](http://www.cs.ubc.ca/~rbridson)
and [Tyson Brochu](http://www.cs.ubc.ca/~tbrochu).
