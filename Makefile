CUDA_PATH     ?= /usr/local/cuda
HOST_COMPILER  = g++
NVCC           = nvcc -ccbin $(HOST_COMPILER)

# GPU architecture for GTX 1080 (SM 61 for Adrien, adapt it for your GPU)
ARCH_FLAGS = \
    -arch=sm_61 \
    -gencode arch=compute_61,code=sm_61 \
    -gencode arch=compute_61,code=compute_61

NVCCFLAGS  = -m64 $(ARCH_FLAGS) -O3 -use_fast_math

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

CUDART_OBJS = $(BIN_DIR)/main.o
CUDART_SD_OBJS = $(BIN_DIR)/main_sd.o $(BIN_DIR)/util.o

TARGET = $(BIN_DIR)/cudart
TARGET_SD = $(BIN_DIR)/cudart_sd
REF = $(BIN_DIR)/cudart_ref
REF_PARAL = $(BIN_DIR)/cudart_ref_parallel

# --------------------------------------------------------------
# Build executable
# --------------------------------------------------------------
all: $(BIN_DIR) $(TARGET) $(REF) $(REF_PARAL) $(TARGET_SD)

$(TARGET): $(CUDART_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Ref should compile main.cu in ref_src/
$(REF): $(REF_SRC_DIR)/main.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Ref2 should compile main_parallel_gen.cu in ref_src/
$(REF_PARAL): $(REF_SRC_DIR)/main_parallel_gen.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(TARGET_SD): $(BIN_DIR)/main_sd.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^


# --------------------------------------------------------------
# Compile .cu into .o in bin/
# --------------------------------------------------------------
$(BIN_DIR)/%.o: $(SRC_DIR)/%.cu $(INCS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# --------------------------------------------------------------
# Create bin (and tmp) directory if it doesn't exist
# --------------------------------------------------------------
$(BIN_DIR):
	mkdir -p $(BIN_DIR)
	mkdir -p tmp

# --------------------------------------------------------------
# Output image
# --------------------------------------------------------------
test: all
	rm -f image.ppm
	$(REF_PARAL)
	$(TARGET)
	$(TARGET_SD)

# --------------------------------------------------------------
# Profiling
# --------------------------------------------------------------
profile_basic: $(TARGET_SD)
	nvprof $(TARGET_SD)

profile_metrics: $(TARGET_SD)
	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer $(TARGET_SD)

# --------------------------------------------------------------
# Clean
# --------------------------------------------------------------
clean:
	rm -rf $(BIN_DIR) image.ppm tmp
