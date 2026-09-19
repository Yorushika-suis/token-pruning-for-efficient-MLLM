# Resolve CUDA_HOME to a directory that actually has nvcc.
# Parent shells often export CUDA_HOME=/usr/local/cuda-13.1; this machine has 13.0.
_repro_cuda_home() {
  local d
  for d in "${CUDA_HOME:-}" /usr/local/cuda-13.0 /usr/local/cuda /usr/local/cuda-13.1; do
    [[ -z "$d" ]] && continue
    if [[ -x "$d/bin/nvcc" ]]; then
      export CUDA_HOME="$d"
      export PATH="$CUDA_HOME/bin:$PATH"
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      unset CUDA_PATH || true
      return 0
    fi
  done
  echo "FAIL: no nvcc (CUDA_HOME=${CUDA_HOME:-unset}); expected /usr/local/cuda-13.0" >&2
  return 1
}
_repro_cuda_home
