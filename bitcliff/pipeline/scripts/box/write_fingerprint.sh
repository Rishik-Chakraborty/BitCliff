#!/bin/bash
# usage: write_fingerprint.sh OUTFILE  -- RUN_0B_PRIME §4 per-run fingerprint, key order of runs-cloud/fingerprint.txt
set -e
cd ~/bitcliff/pipeline
T=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
md(){ curl -s -H "X-aws-ec2-metadata-token: $T" http://169.254.169.254/latest/meta-data/$1; }
LIB=.venv/lib/python3.11/site-packages/llama_cpp/lib/libggml-base.so.0.20.0
SHORT=$(strings $LIB | grep -m1 -E "^[0-9a-f]{7}-dirty$")
case "$SHORT" in 4df29be*) FULL=4df29be4f4c3673f428170fda944a5b19f743bb8;; *) echo "UNREADABLE_OR_UNEXPECTED_COMMIT $SHORT" >&2; exit 3;; esac
{
echo "instance=$(md instance-id)"
echo "instance_type=$(md instance-type)"
echo "ami=$(md ami-id)"
echo "az=$(md placement/availability-zone)"
echo "gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
echo "cuda=$(/usr/local/cuda/bin/nvcc --version | grep -o 'release [0-9.]*')"
echo "kernel=$(uname -r)"
echo "llama_cpp_commit=$FULL"
echo "llama_cpp_python=$(.venv/bin/python -c 'import importlib.metadata as m, llama_cpp; print(m.version("llama-cpp-python"), "CUDA build, gpu_offload_supported="+str(llama_cpp.llama_supports_gpu_offload()))' 2>/dev/null | tail -1)"
echo "python=$(.venv/bin/python --version)"
} > "$1"
echo "# llama_cpp_commit read from the installed build: GGML_COMMIT=\"$SHORT\" in $(basename $LIB); full hash = llama-cpp-python v0.3.35 vendor/llama.cpp submodule (OPEN_QUESTIONS §17)" > "$1.note"
