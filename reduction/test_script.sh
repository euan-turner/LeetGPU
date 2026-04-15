nvcc -O2 -std=c++17 -lcurand -arch=sm_89 -lineinfo gpu_reduction.cu cpu_reduction_ref.cu test_reduction.cu -o test_reduction
./test_reduction

### Only use this if you have permission to open the program counter for profiling.
# nsys profile -w true -t cuda,nvtx,osrt --force-overwrite=true --stats=true --gpu-metrics-device=0 -x true -o test_reduction_nsys_profile ./test_reduction

# Capture one report containing all configurations.
# Only kernels inside the steady_state NVTX range are profiled.
ncu -f \
	-o test_reduction_ncu_profile \
	--nvtx \
	--nvtx-include "steady_state/" \
	--kernel-name "regex:gpu_reduction(lv1)?" \
	--set full \
	--cache-control none \
	--import-source true \
	--target-processes all \
	./test_reduction
