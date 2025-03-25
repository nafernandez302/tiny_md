#!/bin/bash

# Verificar si se han pasado ambos parámetros N y name
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Debe proporcionar dos parámetros: N (número) y path (ruta del archivo)."
  exit 1
fi

# Asignar valores a las variables N y name
N=$1
path=$2

# Limpia los archivos y compila nuevamente
make clean
make N="$N"

# Ejecutar el programa tiny_md y redirigir la salida al archivo output/name
perf stat ./tiny_md > "output/${path}/${N}.out" 2>&1
