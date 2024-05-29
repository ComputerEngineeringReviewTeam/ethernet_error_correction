#!/bin/bash

# Initialize an empty array to hold the -args
args=()

# Loop over all arguments
for arg in "$@"
do
    # If the argument starts with -, it's an -arg
    if [[ $arg == -* ]]; then
        # Remove the - and split the rest into individual args
        arg=${arg:1}
        for ((i=0; i<${#arg}; i++)); do
            args+=("${arg:$i:1}")
        done
    # Otherwise, it's the string argument
    else
        string="$arg"
    fi
done

# Print the -args and string
echo "Args: ${args[@]}"
echo "String: $string"

# Recompile the converter if -c option is used
if [[ " ${args[@]} " =~ "c" ]]; then
    echo "Recompiling converter..."
    g++ src/prep/src/crc32.cpp src/prep/src/encoding.cpp src/prep/src/frame.cpp src/prep/src/transerrors.cpp src/prep/convert/conv.cpp -o bin/conv
    args=(${args[@]/c})
fi

# Run the converter with the remaining -args and filename string
echo "Running converter..."
cd bin
./conv "${args[@]}" "$string"
