#!/bin/bash

prefix="logs_chunk-256-1024-qps10-num100"
P_start=24.3
P_end=30.2
D_start=20.7
D_end=22.7

P_start=$(echo "$P_start * 1000000000" | bc)
P_end=$(echo "$P_end * 1000000000" | bc)
D_start=$(echo "$D_start * 1000000000" | bc)
D_end=$(echo "$D_end * 1000000000" | bc)

path="/workspace/logs_stored/$prefix"

python3 tflops.py "$path/benchmark.log" 256
python3 kv_usage.py "$path/server1.log"
python3 kv_usage.py "$path/server2.log"


# 一次性查询多个 metricId 的平均值
query=$(sqlite3 "$path/$prefix-1.sqlite" "
SELECT 
    AVG(CASE WHEN metricId = 10 THEN value END),
    AVG(CASE WHEN metricId = 25 THEN value END),
    AVG(CASE WHEN metricId = 26 THEN value END)
FROM GPU_METRICS
WHERE timestamp > $P_start AND timestamp < $P_end;
")

# 解析查询结果，SQLite 默认用 `|` 作为分隔符
IFS='|' read -r avg_10 avg_25 avg_26 <<< "$query"

# 输出结果
echo "prefill SM-occupancy (metricId = 10): $(printf "%.2f" "$avg_10")"
echo "prefill read HBM-bandwidth-utilization (metricId = 25): $(printf "%.2f" "$avg_25")"
echo "prefill write HBM-bandwidth-utilization (metricId = 26): $(printf "%.2f" "$avg_26")"

#server2

query=$(sqlite3 "$path/$prefix-2.sqlite" "
SELECT 
    AVG(CASE WHEN metricId = 10 THEN value END),
    AVG(CASE WHEN metricId = 25 THEN value END),
    AVG(CASE WHEN metricId = 26 THEN value END)
FROM GPU_METRICS
WHERE timestamp > $P_start AND timestamp < $P_end;
")
IFS='|' read -r avg_10 avg_25 avg_26 <<< "$query"
echo "decode SM-occupancy (metricId = 10): $(printf "%.2f" "$avg_10")"
echo "decode read HBM-bandwidth-utilization (metricId = 25): $(printf "%.2f" "$avg_25")"
echo "decode write HBM-bandwidth-utilization (metricId = 26): $(printf "%.2f" "$avg_26")"


