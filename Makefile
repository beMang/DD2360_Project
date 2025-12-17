CUDA_PATH     ?= /usr/local/cuda
HOST_COMPILER  = g++
NVCC           = nvcc -ccbin $(HOST_COMPILER)

NVCC_DBG       =

# GPU architecture for GTX 1080 (SM 61 for Adrien, might adapt for your GPU)
ARCH_FLAGS = \
    -arch=sm_61 \
    -gencode arch=compute_61,code=sm_61 \
    -gencode arch=compute_61,code=compute_61

NVCCFLAGS  = $(NVCC_DBG) -m64 $(ARCH_FLAGS) -O3 -use_fast_math

# --------------------------------------------------------------
# Directories
# --------------------------------------------------------------
SRC_DIR = src
REF_SRC_DIR = ref_src
BIN_DIR = bin

# Sources, headers, objects
SRCS = $(wildcard $(SRC_DIR)/*.cu)
INCS = $(wildcard $(SRC_DIR)/*.h)
OBJS = $(patsubst $(SRC_DIR)/%.cu,$(BIN_DIR)/%.o,$(SRCS))

TARGET = $(BIN_DIR)/cudart
REF = $(BIN_DIR)/cudart_ref
REF_B = $(BIN_DIR)/cudart_ref2

# --------------------------------------------------------------
# Build executable
# --------------------------------------------------------------
all: $(BIN_DIR) $(TARGET) $(REF) $(REF_B)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Ref should compile main.cu in ref_src/
$(REF): $(REF_SRC_DIR)/main.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Ref2 should compile main_parallel_gen.cu in ref_src/
$(REF_B): $(REF_SRC_DIR)/main_parallel_gen.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# --------------------------------------------------------------
# Compile .cu into .o in bin/
# --------------------------------------------------------------
$(BIN_DIR)/%.o: $(SRC_DIR)/%.cu $(INCS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# --------------------------------------------------------------
# Create bin directory if it doesn't exist
# --------------------------------------------------------------
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# --------------------------------------------------------------
# Output image
# --------------------------------------------------------------
test: $(TARGET)
	rm -f image.ppm
	$(TARGET)

# --------------------------------------------------------------
# Profiling
# --------------------------------------------------------------
profile_basic: $(TARGET)
	nvprof $(TARGET)

profile_metrics: $(TARGET)
	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer $(TARGET)

# --------------------------------------------------------------
# Clean
# --------------------------------------------------------------
clean:
	rm -rf $(BIN_DIR) image.ppm
