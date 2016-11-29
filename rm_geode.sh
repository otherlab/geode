sudo rm -rf /usr/local/include/geode/
sudo rm -f /usr/local/lib/libgeode.so
sudo updatedb
sudo locate "/usr/local/lib/python2.7/dist-packages/geode" | xargs rm -rf
sudo updatedb
