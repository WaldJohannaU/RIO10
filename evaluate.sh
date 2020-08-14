#/bin/bash

cd eval/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
cd ../../

for i in data/predictions/*.txt ; do 
	eval/build/eval data data/predictions/${i##*/} data/errors/${i##*/}
done