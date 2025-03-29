#!/bin/bash

# Verificar si se han pasado ambos parámetros N y path
if [ -z "$1" ]; then
  echo "Debe proporcionar path (ruta del archivo)."
  exit 1
fi

# Asignar valores a las variables N y path
path=$1
numeros=(256 500 864 1372 2048 2916)

# Limpia los archivos y compila nuevamente
for num in "${numeros[@]}"; do
  make clean
  make N=${num}

  # Ejecutar el programa tiny_md y redirigir la salida al archivo output/path
  perf stat ./tiny_md > "output/${path}/${num}.out" 2>&1

done

