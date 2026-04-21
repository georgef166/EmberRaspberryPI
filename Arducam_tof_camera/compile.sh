#!/bin/sh
# Ember build script — compiles the tactical_rescue heads-up display
# and the reference Arducam ToF demos.
workpath=$(cd "$(dirname "$0")" && pwd)

echo "workpath: $workpath"

if ! cmake -B "$workpath/build" -S "$workpath"; then
    echo "== CMake failed"
    exit 1
fi

build_dir="$workpath/build"

if cmake --build "$build_dir" --config Release --target tactical_rescue -j 4 &&
    cmake --build "$build_dir" --config Release --target preview_depth -j 4 &&
    cmake --build "$build_dir" --config Release --target capture_raw -j 4; then
    echo "== Build success"
    echo "== Run $build_dir/example/cpp/tactical_rescue"
else
    echo "== Retry build without -j 4"
    if cmake --build "$build_dir" --config Release --target tactical_rescue &&
        cmake --build "$build_dir" --config Release --target preview_depth &&
        cmake --build "$build_dir" --config Release --target capture_raw; then
        echo "== Build success"
        echo "== Run $build_dir/example/cpp/tactical_rescue"
    else
        echo "== Build failed"
    fi
fi
