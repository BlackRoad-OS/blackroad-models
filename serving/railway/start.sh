#!/bin/bash
# BlackRoad Model Server Startup Script

set -e

echo "🚀 Starting BlackRoad Model Server"
echo "Model: ${MODEL_ID}"
echo "Base: ${BASE_MODEL}"

# Download LoRA weights from S3 if not already present
if [ ! -d "${LORA_MODEL_PATH}" ] || [ -z "$(ls -A "${LORA_MODEL_PATH}" 2>/dev/null)" ]; then
  echo "📦 Downloading LoRA weights from S3..."

  if [ -z "${LORA_S3_URI}" ]; then
    echo "❌ LORA_S3_URI is not set. Cannot download LoRA weights." >&2
    exit 1
  fi

  mkdir -p "${LORA_MODEL_PATH}"

  aws s3 sync "${LORA_S3_URI}" "${LORA_MODEL_PATH}" \
    --no-progress \
    --exact-timestamps

  echo "✅ LoRA weights downloaded to ${LORA_MODEL_PATH}"
else
  echo "✅ LoRA weights already present at ${LORA_MODEL_PATH}"
fi

# Start vLLM server with LoRA
echo "🔥 Starting vLLM server..."

python -m vllm.entrypoints.openai.api_server \
  --model "${BASE_MODEL}" \
  --enable-lora \
  --lora-modules "blackroad-coder=${LORA_MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}" \
  --quantization "${VLLM_QUANTIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --enable-prefix-caching \
  --served-model-name "${MODEL_NAME}" \
  --api-key "${API_KEY}"
