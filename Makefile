CC      = nvcc #(clang) activar compilador AMD (!)  , nvcc compilador de CUDA
CFLAGS   := -O3 -ffast-math -march=native -fopenmp
CPPFLAGS := -DN=$(N)
WFLAGS   := -std=c11 -Wall -Wextra -g
LDFLAGS  := -lm            # matemática
CUDA_LIB := -lcudart       # runtime CUDA
N?=256

# Ejecutable final
TARGET   := tiny_md

# Fuentes C y CUDA
SRC_C    := $(filter-out forces.cu, $(wildcard *.c))
SRC_CU   := forces.cu
OBJ_C    := $(SRC_C:.c=.o)
OBJ_CU   := $(SRC_CU:.cu=.o)
OBJECTS  := $(OBJ_C) $(OBJ_CU)

###############################################################################
# Reglas
###############################################################################

.PHONY: all clean

all: $(TARGET)

# Enlace final: enlaza objetos C y CUDA
$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(CUDA_LIB)

# Regla genérica para .c → .o
%.o: %.c
	$(CC) $(WFLAGS) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# Regla para .cu → .o
%.o: %.cu
	$(CC) $(CPPFLAGS) -Xcompiler="$(CFLAGS)" -c $< -o $@

###############################################################################
# Limpieza
###############################################################################
clean:
	rm -f $(TARGET) $(OBJECTS) *.xyz *.log .depend

###############################################################################
# Dependencias automáticas (opcional)
###############################################################################
.depend: $(SRC_C) $(SRC_CU)
	$(CC) -MM $^ > $@

-include .depend