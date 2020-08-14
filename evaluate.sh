#/bin/bash

cd src/eval
rm -r build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
cd ../../../

for i in data/predictions/*.txt ; do 
	src/eval/build/eval data data/predictions/${i##*/} data/errors/${i##*/}
done