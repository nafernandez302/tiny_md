#!/bin/bash

# Lista de números
numeros=(256 500 864 1372 2048 2916)

# Nombre del archivo de salida


# Recorremos todos los subdirectorios dentro de 'output/'
for dir in output/lab3/*/; do
    output_file="${dir}resultados.txt"
    touch ${output_file}
    # Recorremos cada número en la lista
    for num in "${numeros[@]}"; do
        # Definir el archivo de entrada
        input_file="${dir}${num}.out"
        
        # Verificar si el archivo de entrada existe
        if [[ -f $input_file ]]; then
            # Procesar el archivo con awk
            awk -v output_file="$output_file" '
            /# Número de partículas:/ {particulas = $5}
            /# Tiempo total de simulación/ {tiempo = $7}
            /# cells\/s/ {cells = $4}
            END {
                # Formato N elapsed cells/s
                print particulas, tiempo, cells >> output_file
            }
            ' "$input_file"
        else
            echo "El archivo $input_file no existe." >> $output_file
        fi
    done
done


