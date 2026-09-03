#!/bin/bash
#
# LangSplat 完整训练脚本 (包含官方建议的3个Level训练)
#

set -e

# ==================== 配置区域 ====================

PROJECT_ROOT="$HOME/project/LangSplat"
GS_ROOT="$HOME/project/gaussian-splatting"
DATA_ROOT="$PROJECT_ROOT/dataset/lerf_ovs"
LABEL_ROOT="$DATA_ROOT/label"
OUTPUT_ROOT="$PROJECT_ROOT/output"
AE_CKPT_ROOT="$PROJECT_ROOT/autoencoder/ckpt"
EVAL_OUTPUT_ROOT="$PROJECT_ROOT/eval_result"

GS_ITERATIONS=30000
GS_TEST_ITERATIONS=7000
GS_CHECKPOINT_ITERATIONS="7000 15000 30000"

LANGSPLAT_ITERATIONS=20000
LANGSPLAT_CHECKPOINT_ITERATIONS="7000 15000 20000"

AE_ENCODER_DIMS="256 128 64 32 3"
AE_DECODER_DIMS="16 32 64 128 256 256 512"

LEVELS=(1 2 3)
MASK_THRESH=0.4
ALL_SCENES=("figurines" "ramen" "teatime" "waldo_kitchen")

# ==================== 辅助函数 ====================

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

log_section() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
    echo ""
}

check_conda_env() {
    local env_name=$1
    if ! conda env list | grep -q "^$env_name "; then
        log "错误: Conda环境 '$env_name' 不存在"
        exit 1
    fi
}

activate_env() {
    local env_name=$1
    log "激活Conda环境: $env_name"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate $env_name
}

# ==================== 步骤函数 ====================

train_3dgs() {
    local scene=$1
    local data_path="$DATA_ROOT/$scene"
    local output_path="$OUTPUT_ROOT/lerf_$scene"
    
    log_section "步骤1: 3DGS训练 - 场景: $scene"
    
    if [ -d "$output_path/point_cloud/iteration_30000" ]; then
        log "3DGS模型已存在，跳过训练: $output_path"
        return 0
    fi
    
    activate_env 3dgs
    cd "$GS_ROOT"
    
    log "数据路径: $data_path"
    log "输出路径: $output_path"
    
    python train.py \
        -s "$data_path" \
        -m "$output_path" \
        --iterations $GS_ITERATIONS \
        --test_iterations $GS_TEST_ITERATIONS \
        --checkpoint_iterations $GS_CHECKPOINT_ITERATIONS \
        --eval
    
    log "3DGS训练完成: $output_path"
}

extract_language_features() {
    local scene=$1
    local data_path="$DATA_ROOT/$scene"
    
    log_section "步骤2: 语言特征提取 - 场景: $scene"
    
    activate_env langsplat
    cd "$PROJECT_ROOT"
    
    if [ -d "$data_path/feature" ]; then
        local feature_count=$(find "$data_path/feature" -name "*.npy" | wc -l)
        if [ $feature_count -gt 0 ]; then
            log "语言特征已存在 ($feature_count 个文件)，跳过提取"
            return 0
        fi
    fi
    
    python preprocess.py --dataset_path "$data_path"
    
    log "语言特征提取完成"
}

train_autoencoder() {
    local scene=$1
    local data_path="$DATA_ROOT/$scene"
    local dataset_name="lerf_$scene"
    
    log_section "步骤3: Autoencoder训练 - 场景: $scene"
    
    activate_env langsplat
    cd "$PROJECT_ROOT/autoencoder"
    
    if [ -f "ckpt/$dataset_name/best_ckpt.pth" ]; then
        log "Autoencoder checkpoint已存在，跳过训练"
        return 0
    fi
    
    python train.py \
        --dataset_path "$data_path" \
        --dataset_name "$dataset_name" \
        --encoder_dims $AE_ENCODER_DIMS \
        --decoder_dims $AE_DECODER_DIMS
    
    log "Autoencoder训练完成: ckpt/$dataset_name/best_ckpt.pth"
}

generate_3d_features() {
    local scene=$1
    local data_path="$DATA_ROOT/$scene"
    local dataset_name="lerf_$scene"
    
    log_section "步骤4: 生成3维特征 - 场景: $scene"
    
    activate_env langsplat
    cd "$PROJECT_ROOT/autoencoder"
    
    if [ -d "$data_path/language_features_dim3" ]; then
        local feature_count=$(find "$data_path/language_features_dim3" -name "*.npy" | wc -l)
        if [ $feature_count -gt 0 ]; then
            log "3维特征已存在 ($feature_count 个文件)，跳过生成"
            return 0
        fi
    fi
    
    python test.py \
        --dataset_path "$data_path" \
        --dataset_name "$dataset_name" \
        --encoder_dims $AE_ENCODER_DIMS \
        --decoder_dims $AE_DECODER_DIMS
    
    log "3维特征生成完成"
}

train_langsplat() {
    local scene=$1
    local level=$2
    local data_path="$DATA_ROOT/$scene"
    local gs_model_path="$OUTPUT_ROOT/lerf_$scene"
    local output_path="$OUTPUT_ROOT/lerf_${scene}_$level"
    local checkpoint_path="$gs_model_path/chkpnt30000.pth"
    
    log "步骤5.$level: LangSplat训练 - Level $level"
    log "输入模型路径: $gs_model_path"
    log "Checkpoint: $checkpoint_path"
    log "输出路径: $output_path"
    
    activate_env langsplat
    cd "$PROJECT_ROOT"
    
    if [ -d "$output_path/point_cloud/iteration_20000" ]; then
        log "LangSplat模型已存在，跳过训练: $output_path"
        return 0
    fi
    
    python train.py \
        -s "$data_path" \
        -m "$gs_model_path" \
        --include_feature \
        --iterations $LANGSPLAT_ITERATIONS \
        --checkpoint_iterations $LANGSPLAT_CHECKPOINT_ITERATIONS \
        --feature_level $level \
        --start_checkpoint "$checkpoint_path" \
        --eval
    
    log "LangSplat训练完成: Level $level -> $output_path"
}

render_langsplat() {
    local scene=$1
    local level=$2
    local output_path="$OUTPUT_ROOT/lerf_${scene}_$level"
    
    log "步骤6.$level: 渲染 - Level $level"
    
    activate_env langsplat
    cd "$PROJECT_ROOT"
    
    python render.py \
        -m "$output_path" \
        --include_feature \
        --feature_level $level \
        --iteration $LANGSPLAT_ITERATIONS
    
    log "渲染完成: Level $level"
}

create_eval_symlinks() {
    local scene=$1
    
    log_section "步骤7: 创建评估符号链接 - 场景: $scene"
    
    # 创建GT标签符号链接
    local label_src="$LABEL_ROOT/$scene"
    local label_dst="$LABEL_ROOT/lerf_$scene"
    
    if [ -d "$label_src" ] && [ ! -L "$label_dst" ]; then
        [ -e "$label_dst" ] && mv "$label_dst" "${label_dst}.bak"
        ln -s "$scene" "$label_dst"
        log "创建GT标签符号链接: $label_dst -> $label_src"
    fi
    
    # 创建AE checkpoint的ae_ckpt子目录
    local ae_ckpt_dir="$AE_CKPT_ROOT/lerf_$scene"
    if [ -f "$ae_ckpt_dir/best_ckpt.pth" ] && [ ! -f "$ae_ckpt_dir/ae_ckpt/best_ckpt.pth" ]; then
        mkdir -p "$ae_ckpt_dir/ae_ckpt"
        cp "$ae_ckpt_dir/best_ckpt.pth" "$ae_ckpt_dir/ae_ckpt/"
        log "复制AE checkpoint"
    fi
}

evaluate_scene() {
    local scene=$1
    local eval_output="$EVAL_OUTPUT_ROOT/$scene"
    
    log_section "步骤8: 评估 - 场景: $scene (综合Level 1,2,3)"
    
    activate_env langsplat
    cd "$PROJECT_ROOT/eval"
    
    python evaluate_iou_loc.py \
        --dataset_name "lerf_$scene" \
        --feat_dir "$OUTPUT_ROOT" \
        --ae_ckpt_dir "$AE_CKPT_ROOT" \
        --output_dir "$eval_output" \
        --json_folder "$LABEL_ROOT" \
        --mask_thresh $MASK_THRESH
    
    log "评估完成"
}

save_metrics() {
    local scene=$1
    local eval_output="$EVAL_OUTPUT_ROOT/$scene/lerf_$scene"
    
    log_section "保存评估结果 - 场景: $scene"
    
    local log_file=$(ls -t "$eval_output"/*.log 2>/dev/null | head -1)
    
    if [ -n "$log_file" ] && [ -f "$log_file" ]; then
        python3 << PYEOF
import re
import json

log_file = "$log_file"
output_file = "$EVAL_OUTPUT_ROOT/$scene/metrics.json"

with open(log_file, 'r') as f:
    content = f.read()

iou_match = re.search(r'iou chosen:\s*(\d+\.\d+)', content)
miou = float(iou_match.group(1)) if iou_match else 0.0

loc_match = re.search(r'Localization accuracy:\s*(\d+\.\d+)', content)
loc_acc = float(loc_match.group(1)) if loc_match else 0.0

result = {
    "scene": "$scene",
    "mIoU": miou,
    "localization_accuracy": loc_acc
}

with open(output_file, 'w') as f:
    json.dump(result, f, indent=2)

print(f"mIoU: {miou:.4f}")
print(f"Localization Accuracy: {loc_acc:.4f}")
PYEOF
        
        log "结果已保存到: $EVAL_OUTPUT_ROOT/$scene/metrics.json"
    else
        log "警告: 未找到评估日志文件"
    fi
}

summarize_all_results() {
    log_section "所有场景评估结果汇总"
    
    echo ""
    echo "| 场景 | mIoU | Localization Accuracy |"
    echo "|------|------|----------------------|"
    
    for scene in "${ALL_SCENES[@]}"; do
        local metrics_file="$EVAL_OUTPUT_ROOT/$scene/metrics.json"
        if [ -f "$metrics_file" ]; then
            python3 -c "import json; d=json.load(open('$metrics_file')); print(f'| {d[\"scene\"]} | {d[\"mIoU\"]:.4f} | {d[\"localization_accuracy\"]:.4f} |')"
        else
            echo "| $scene | N/A | N/A |"
        fi
    done
    echo ""
}

# ==================== 主训练流程 ====================

train_scene() {
    local scene=$1
    
    log_section "开始训练场景: $scene"
    
    [ ! -d "$DATA_ROOT/$scene" ] && { log "错误: 数据目录不存在: $DATA_ROOT/$scene"; return 1; }
    
    train_3dgs $scene
    extract_language_features $scene
    train_autoencoder $scene
    generate_3d_features $scene
    
    for level in "${LEVELS[@]}"; do
        log "==================== 处理 Level $level ===================="
        train_langsplat $scene $level
        render_langsplat $scene $level
    done
    
    create_eval_symlinks $scene
    evaluate_scene $scene
    save_metrics $scene
}

main() {
    log_section "LangSplat 完整训练脚本 (3-Level训练)"
    
    check_conda_env "langsplat"
    check_conda_env "3dgs"
    
    mkdir -p "$OUTPUT_ROOT" "$AE_CKPT_ROOT" "$EVAL_OUTPUT_ROOT"
    
    local scenes=()
    [ -n "$1" ] && scenes=("$1") || scenes=("${ALL_SCENES[@]}")
    
    log "将训练以下场景: ${scenes[*]}"
    
    for scene in "${scenes[@]}"; do
        train_scene $scene
    done
    
    summarize_all_results
    
    log_section "所有训练任务完成!"
}

main "$@"
