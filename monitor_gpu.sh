#!/bin/bash
# Monitor GPU usage in real-time while training
# Run this in a separate terminal: bash monitor_gpu.sh

watch -n 0.5 'nvidia-smi --query-gpu=timestamp,name,pci.bus_id,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv'
