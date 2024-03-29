#! /bin/bash
#brew install openssl libuv cmake zlib
git clone https://github.com/uWebSockets/uWebSockets 
cd uWebSockets
git checkout e94b6e1
patch CMakeLists.txt < ../cmakepatch.txt
mkdir build
# export CPPFLAGS=-I/usr/local/opt/openssl/include
# export LDFLAGS=-L/usr/local/opt/openssl/lib
export PKG_CONFIG_PATH=/usr/local/opt/openssl/lib/pkgconfig
#export OPENSSL_ROOT_DIR=/usr/local/opt/openssl
cd build
cmake -DOPENSSL_ROOT_DIR=/usr/local/Cellar/openssl/1.0.2l -DOPENSSL_LIBRARIES=/usr/local/Cellar/openssl/1.0.2l/lib ..
make 
sudo make install
cd ..
cd ..
sudo rm -r uWebSockets
