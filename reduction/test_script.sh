nvcc -O2 -std=c++14 -lcurand -arch=sm_89 gpu_reduction.cu cpu_reduction_ref.cu test_reduction.cu -o test_reduction
./test_reduction

### Only use this if you have permission to open the program counter for profiling.
# nsys profile -w true -t cuda,nvtx,osrt --force-overwrite=true --stats=true --gpu-metrics-device=0 -x true -o test_reduction_nsys_profile ./test_reduction
# ncu -f -o test_reduction_ncu_profile --kernel-name gpu_reduction --launch-count 1 --set full --cache-control none --import-source true --target-processes all ./test_reduction
