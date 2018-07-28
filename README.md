Android native implement(ncnn&&cblas) of mtcnn face detect&&alignment
Pure c++ implement, and you dont have to compile cblas/ncnn framework before compiling exe file.

USAGE:
cd cblas/ncnn/Android folder;ndk-build;run.bat

NOTE:
cblas version implement within this repo deponds on https://github.com/AlphaQi/MTCNN-light, i compile it in early 2017, and then port it to android using openblas(this is a default .so for android>=6.0)
ncnn version implement, i made it in early 2017 too, when ncnn is announced.
At that time, i made test script for win user, guess it will be easy to use and modify.

Tips:
The auther comes to be a GPU parallel computing programmer since 2009, familiar with cuda/opencl/neon programming.
ncnn is a good work, but it is not friendly enough for low-end mobile devices. e.g. unstable performance, heat problem for always-on features...this is not so acceptable so far as i know.
I do have implement mtcnn with opencl for certain mobile devices in early 2017, however, will not open-source recently.
