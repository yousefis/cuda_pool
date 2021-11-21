# */
# * Developer: Sahar Yousefi
# * more info: https://github.com/yousefis/cuda_pool
# */


# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda


# CC compiler options:
CC=g++
CC_FLAGS=
CC_LIBS=




# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS= `pkg-config --cflags opencv`
NVCC_LIBS= -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio  -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc
NVCC_LIBS+= `pkg-config --libs opencv`

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

# Opencv libraries:
OPENCV_INC_DIR= -I/usr/include/opencv2/


# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = include


# Target executable name:
EXE = main

# Object files:
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/kernels.o


# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) $(NVCC_LIBS) $(OPENCV_INC_DIR)

# Compile main.cu file to object files:
$(OBJ_DIR)/%.o : %.cu
	$(NVCC)  $(NVCC_FLAGS) $(NVCC_LIBS) $(OPENCV_INC_DIR) -c $< -o $@ 



# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp include/%.h 
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)



