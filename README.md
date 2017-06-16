# C++ KCF_face
based on KCF and seetaface.

### KCF Algorithms (in this folder) ###

"KCFC++", command: ./KCF   
Description: KCF on HOG features, ported to C++ OpenCV. The original Matlab tracker placed 3rd in VOT 2014.

"KCFLabC++", command: ./KCF lab   
Description: KCF on HOG and Lab features, ported to C++ OpenCV. The Lab features are computed by quantizing CIE-Lab colors into 15 centroids, obtained from natural images by k-means.   

The CSK tracker [2] is also implemented as a bonus, simply by using raw grayscale as features (the filter becomes single-channel).   

### Compilation instructions ###
There are no external dependencies other than OpenCV 3.0.0. Tested on a freshly installed Ubuntu 14.04.   

1) cmake CMakeLists.txt   
2) make   

### Running instructions ###

The runtracker.cpp is prepared to be used with the VOT toolkit. The executable "KCF" should be called as:   

./KCF [OPTION_1] [OPTION_2] [...]

