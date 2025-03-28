#!/bin/bash

# Requirement: 2x GPUs.


# Model: meta-llama/Meta-Llama-3.1-8B-Instruct
# Query: 1024 input tokens, 6 output tokens, QPS 2/4/6/8, 100 requests
# Resource: 2x GPU
# Approaches:
# 2. Chunked prefill: 2 vllm instance with tp=4, equivalent to 1 tp=4 instance with QPS 4
# 3. Disaggregated prefill: 1 prefilling instance and 1 decoding instance
# Prefilling instance: max_output_token=1
# Decoding instance: force the input tokens be the same across requests to bypass prefilling

set -ex
PROXY_SERVER_PORT=9100
PREFILL_SERVER_PORT=9200
DECODING_SERVER_PORT=9300
MODEL_PATH="/host_model/meta-llama/Meta-Llama-3.1-8B-Instruct"
# MODEL_PATH="/host_model/meta-llama/Llama-3.3-70B-Instruct"

MODEL_NAME=$(echo "$MODEL_PATH" | awk -F'/' '{print $NF}')
TP=1
PP=1
SERVER1_GPU=6
SERVER2_GPU=7
export VLLM_LOGGING_LEVEL=INFO
export VLLM_USE_V1=1

inputlen=256
outputlen=1024
qps=10

NUM_PROMTS=100
MAX_TOKEN_LENGTH=8192
OUTPUT_LENGTH_LIMITATION="--sharegpt-output-len 4096"
OUTPUT_LENGTH_LIMITATION=""

Option="1p1d"
LOG_PREFIX="logs_$Option-$inputlen-$outputlen-qps$qps-num$NUM_PROMTS"
SAVE_RESULT=0

CONFIG_COLOCATED="
    --model $MODEL_PATH \
    --max-model-len $MAX_TOKEN_LENGTH \
    --enable-chunked-prefill true \
    --tensor-parallel-size $TP \
    --pipeline-parallel-size $PP \
    --gpu-memory-utilization 0.7 \
    --collect-detailed-traces all 
"
CONFIG_DISAGG="
  --no-enable-prefix-caching \
  --enforce-eager \
  --model $MODEL_PATH \
  --max-model-len $MAX_TOKEN_LENGTH \
  --enable-chunked-prefill false \
  --tensor-parallel-size $TP \
  --pipeline-parallel-size $PP \
  --collect-detailed-traces all
"
CONFIG_PREFILL="
  --max-num-batched-tokens 2048 \
  --max-num-seqs 256 \
"

kill_gpu_processes() {
  # kill all processes on GPU.
  pgrep pt_main_thread | xargs -r kill -9
  pgrep python3 | xargs -r kill -9
  for port in $PROXY_SERVER_PORT $PREFILL_SERVER_PORT $DECODING_SERVER_PORT; do lsof -t -i:$port | xargs -r kill -9; done
  sleep 1
}

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/health > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}


launch_chunked_prefill() {
  # chunked prefill
  CUDA_VISIBLE_DEVICES=$SERVER1_GPU \
    python3 \
    -m vllm.entrypoints.openai.api_server \
    --port $PREFILL_SERVER_PORT \
    $CONFIG_COLOCATED \
    2>&1 | tee $LOG_PREFIX/server1.log &
    # --distributed-executor-backend ray \
  
  CUDA_VISIBLE_DEVICES=$SERVER2_GPU \
    python3 \
    -m vllm.entrypoints.openai.api_server \
    --port $DECODING_SERVER_PORT \
    $CONFIG_COLOCATED \
    2>&1 | tee $LOG_PREFIX/server2.log &
    # --distributed-executor-backend ray \

  wait_for_server $PREFILL_SERVER_PORT
  wait_for_server $DECODING_SERVER_PORT
  python3 round_robin_proxy.py \
    --prefill_url "localhost:${PREFILL_SERVER_PORT}" \
    --decoding_url "localhost:${DECODING_SERVER_PORT}" \
    --proxy_port $PROXY_SERVER_PORT \
    2>&1 | tee $LOG_PREFIX/proxy.log &
  sleep 1
}


launch_disagg_prefill() {
  # disagg prefill
  CUDA_VISIBLE_DEVICES=$SERVER1_GPU \
  nsys profile \
  -o $LOG_PREFIX/$LOG_PREFIX-1.nsys-rep \
  --trace-fork-before-exec=true \
  --trace=cuda,nvtx \
  --gpu-metrics-devices=cuda-visible \
  --stats=true \
  --force-overwrite true \
  --delay 0 \
  --duration 60 \
  --kill none \
  python3 \
    -m vllm.entrypoints.openai.api_server \
    --port $PREFILL_SERVER_PORT \
    $CONFIG_DISAGG \
    $CONFIG_PREFILL \
    --gpu-memory-utilization 0.7 \
    --kv-transfer-config '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2, "kv_producer_size":1, "kv_buffer_size":15e9}' \
    2>&1 | tee $LOG_PREFIX/server1.log &
    # --distributed-executor-backend ray \
  
  CUDA_VISIBLE_DEVICES=$SERVER2_GPU \
  nsys profile \
  -o $LOG_PREFIX/$LOG_PREFIX-2.nsys-rep \
  --trace-fork-before-exec=true \
  --trace=cuda,nvtx \
  --gpu-metrics-devices=cuda-visible \
  --stats=true \
  --force-overwrite true \
  --delay 0 \
  --duration 90 \
  python3 \
    -m vllm.entrypoints.openai.api_server \
    --port $DECODING_SERVER_PORT \
    $CONFIG_DISAGG \
    --gpu-memory-utilization 0.7 \
    --kv-transfer-config '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_producer_size":1,"kv_buffer_size":10e9}' \
    2>&1 | tee $LOG_PREFIX/server2.log &
    # --distributed-executor-backend ray \

  wait_for_server $PREFILL_SERVER_PORT
  wait_for_server $DECODING_SERVER_PORT

  python3 disagg_prefill_proxy_server.py \
    --prefill_url "localhost:${PREFILL_SERVER_PORT}" \
    --decoding_url "localhost:${DECODING_SERVER_PORT}" \
    --proxy_port $PROXY_SERVER_PORT \
    2>&1 | tee $LOG_PREFIX/proxy.log &

  sleep 1
}


benchmark() {
  results_folder="./tmp_results"
  model="${MODEL_PATH}"
  dataset_name="random"
  random_input_len=$inputlen
  random_output_len=$outputlen
  dataset_path="/workspace/dataset/vllm/sharegpt/fixed_data.json"
  num_prompts=$NUM_PROMTS
  qps=$1
  tag=$2
  VLLM_USE_MODELSCOPE=True 
  python3 ../benchmark_serving.py \
          --backend vllm \
          --model $model \
          --dataset-name $dataset_name \
          --random-input-len $random_input_len \
          --random-output-len $random_output_len \
          --dataset-path $dataset_path \
          --num-prompts $num_prompts \
          $OUTPUT_LENGTH_LIMITATION \
          --port $PROXY_SERVER_PORT \
          --save-result \
          --result-dir $results_folder \
          --result-filename client.json \
          --request-rate "$qps" \
          --ignore-eos \
          2>&1 | tee $LOG_PREFIX/benchmark.log
  sleep 2

  if [ "$SAVE_RESULT" -ne 0 ]; then
      collection_dir="/workspace/logs_stored/$LOG_PREFIX"
      rm -rf ${collection_dir}/*
      mkdir -p $collection_dir
      mv tmp_results/* $collection_dir
      mv $LOG_PREFIX/* $collection_dir
      rm -rf $LOG_PREFIX
  else
      echo "[WARNING]: not saving results or logs!"
  fi

  sleep 2
}


main() {

  #(which wget && which curl) || (apt-get update && apt-get install -y wget curl)
  #(which jq) || (apt-get -y install jq)
  #(which socat) || (apt-get -y install socat)

  # pip install quart httpx matplotlib aiohttp

  cd "$(dirname "$0")"

  rm -rf tmp_results/*
  mkdir -p tmp_results
  rm -rf $LOG_PREFIX/*
  mkdir -p $LOG_PREFIX

  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

  if [[ "$Option" == "1p1d" ]]; then
    launch_disagg_prefill
    benchmark $qps disagg-prefill
    sleep 60
    kill_gpu_processes
  else
    launch_chunked_prefill
    benchmark $qps chunked-prefill
    kill_gpu_processes
  fi
  # python3 visualize_benchmark_results.py
}


main "$@"
