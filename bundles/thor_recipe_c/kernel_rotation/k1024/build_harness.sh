#!/bin/bash
# Compila il banco di verifica contro le librerie del fork isolato k1024.
set -eu
T=@@ROOT:models@@/build/llamacpp-tq1-k1024
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -I"$T/ggml/include" harness_ggml_1024.cpp -o harness_ggml_1024 \
    -L"$T/build/bin" -lggml -lggml-base -lggml-cpu -Wl,-rpath,"$T/build/bin"
echo "compilato: $(ls -la harness_ggml_1024 | awk '{print $5}') byte"
