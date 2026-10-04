#!/bin/bash
set -e; set -o pipefail
cd /home/xiedexia/project/LangSplat
export SRC=/home/xiedexia/project/LangSplat/dataset/lerf_ovs
bash run_master_table.sh waldo_kitchen 0.40 5570 > master_waldo.log 2>&1
echo "WALDO_DONE"
bash run_master_table.sh ramen 0.55 5590 > master_ramen.log 2>&1
echo "RAMEN_DONE"
echo "MASTER_ALL_DONE"
