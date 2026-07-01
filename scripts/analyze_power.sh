#!/bin/bash
# Analyze tegrastats log file for board power consumption.
#
# Usage:
#   bash scripts/analyze_power.sh [path-to-log]
#
# Default log path: diagnostic_frames/power_log.txt

LOG="${1:-diagnostic_frames/power_log.txt}"

if [ ! -f "$LOG" ]; then
    echo "ERROR: log file not found: $LOG"
    echo "Did you run tegrastats with --logfile option?"
    exit 1
fi

if [ ! -s "$LOG" ]; then
    echo "ERROR: log file is empty: $LOG"
    exit 1
fi

echo "==========================================="
echo "Power Analysis — $LOG"
echo "==========================================="
echo "Total samples: $(wc -l < "$LOG")"
echo ""

echo "VDD_IN (total board power)"
echo "-------------------------------------------"
grep -oP 'VDD_IN \K[0-9]+' "$LOG" | awk '{
  if (NR==1 || $1<min) min=$1
  if ($1>max) max=$1
  sum+=$1; n++
} END {
  if (n==0) { print "  No VDD_IN samples found"; exit }
  printf "  Min     = %6.2f W\n", min/1000
  printf "  Avg     = %6.2f W\n", sum/n/1000
  printf "  Peak    = %6.2f W\n", max/1000
  printf "  Samples = %d\n", n
}'
echo ""

echo "VDD_CPU_GPU_CV (compute-only power)"
echo "-------------------------------------------"
grep -oP 'VDD_CPU_GPU_CV \K[0-9]+' "$LOG" | awk '{
  if (NR==1 || $1<min) min=$1
  if ($1>max) max=$1
  sum+=$1; n++
} END {
  if (n==0) { print "  No VDD_CPU_GPU_CV samples found"; exit }
  printf "  Min     = %6.2f W\n", min/1000
  printf "  Avg     = %6.2f W\n", sum/n/1000
  printf "  Peak    = %6.2f W\n", max/1000
}'
echo ""

echo "VDD_SOC (SoC fixed power)"
echo "-------------------------------------------"
grep -oP 'VDD_SOC \K[0-9]+' "$LOG" | awk '{
  if (NR==1 || $1<min) min=$1
  if ($1>max) max=$1
  sum+=$1; n++
} END {
  if (n==0) { print "  No VDD_SOC samples found"; exit }
  printf "  Min     = %6.2f W\n", min/1000
  printf "  Avg     = %6.2f W\n", sum/n/1000
  printf "  Peak    = %6.2f W\n", max/1000
}'
echo ""

echo "==========================================="
echo "For paper / REVIEWER_RESPONSE.md:"
echo "==========================================="
grep -oP 'VDD_IN \K[0-9]+' "$LOG" | awk '{
  if (NR==1 || $1<min) min=$1
  if ($1>max) max=$1
  sum+=$1; n++
} END {
  if (n==0) exit
  printf "  Idle     ~ %.2f W (Min)\n",  min/1000
  printf "  Avg      ~ %.2f W (during inference)\n", sum/n/1000
  printf "  Peak     ~ %.2f W\n", max/1000
  printf "  Envelope: 25 W (MAXN_SUPER hardware ceiling)\n"
}'
