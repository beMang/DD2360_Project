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
BIN_DIR = bin

# Sources, headers, objects
SRCS = $(wildcard $(SRC_DIR)/*.cu)
INCS = $(wildcard $(SRC_DIR)/*.h)
OBJS = $(patsubst $(SRC_DIR)/%.cu,$(BIN_DIR)/%.o,$(SRCS))

TARGET = $(BIN_DIR)/cudart

# --------------------------------------------------------------
# Build executable
# --------------------------------------------------------------
all: $(BIN_DIR) $(TARGET)

$(TARGET): $(OBJS)
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
out.ppm: $(TARGET)
	rm -f out.ppm
	$(TARGET) > out.ppm

out.jpg: out.ppm
	rm -f out.jpg
	ppmtojpeg out.ppm > out.jpg

# --------------------------------------------------------------
# Profiling
# --------------------------------------------------------------
profile_basic: $(TARGET)
	nvprof $(TARGET) > out.ppm

profile_metrics: $(TARGET)
	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer $(TARGET) > out.ppm

# --------------------------------------------------------------
# Clean
# --------------------------------------------------------------
clean:
	rm -rf $(BIN_DIR) out.ppm out.jpg
