#!/bin/bash

# 1: Compile the Cython extensions
echo "Compiling Cython extensions..."
python setup.py build_ext --inplace

# Check if the build was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

# 2: Move and rename the generated .so files to the parent directory
echo "Moving and renaming .so files to the parent directory..."
for so_file in *.so; do
    if [ -f "$so_file" ]; then
        # Extract the base name without the .cpython-39-x86_64-linux-gnu.so part
        base_name=$(echo $so_file | sed -e 's/\.[^.]*\.so$/.so/')
        mv "$so_file" "$base_name"
        echo "Moved and renamed $so_file to ../../$base_name"
    fi
done

# 3: Remove the build directory
echo "Removing some intermediate results..."
rm -rf build
if [ $? -ne 0 ]; then
    echo "Failed to remove build directory."
    exit 1
fi

# 4: Remove intermediate .c files
for c_file in *.c; do
    if [ -f "$c_file" ]; then
        rm "$c_file"
        echo "Removed $c_file"
    fi
done

echo "Build Successfully :)"
