
chek_dir()
{
if [ ! -f "GeodeSupport.cmake" ]
then
   echo "run from top level directory"
   exit
fi
return $(pwd) 
}

clean_build_dir()
{
if [ -a "build" ]
then
    echo "removing previous build directory"
    rm -rf build
fi
echo "creating new build directory"
mkdir build
}

build_cmake()
{
    cmake ../
}

build_make()
{

    NPROCS=1
    OS=$(uname -s)
    if [[ $OS == *"Linux"* ]]
    then
	NPROCS=$(grep -c ^processor /proc/cpuinfo)
    fi

    if [[ $OS == *"Darwin"* ]] # Assume Mac OS X
    then
	NPROCS=$(system_profiler | awk '/Number Of CPUs/{print $4}{next;}')
    fi

    make -j $NPROCS
}
    
build_phase()
{
    cd build
    build_cmake
    build_make
    cd ..
}
    
