#!/bin/bash
#
# LangSplat 快速测试脚本 (与train_all_lerf.sh保持一致，只用2张图片)
#

set -e

# ==================== 配置区域 (与train_all_lerf.sh一致) ====================

PROJECT_ROOT="$HOME/project/LangSplat"
GS_ROOT="$HOME/project/gaussian-splatting"
DATA_ROOT="$PROJECT_ROOT/dataset/lerf_ovs"
LABEL_ROOT="$DATA_ROOT/label"
OUTPUT_ROOT="$PROJECT_ROOT/output"
AE_CKPT_ROOT="$PROJECT_ROOT/autoencoder/ckpt"
EVAL_OUTPUT_ROOT="$PROJECT_ROOT/eval_result"

# 缩减迭代次数（快速测试）
GS_ITERATIONS=1000
GS_TEST_ITERATIONS=500
GS_CHECKPOINT_ITERATIONS="500 1000"

LANGSPLAT_ITERATIONS=500
LANGSPLAT_CHECKPOINT_ITERATIONS="500"

AE_ENCODER_DIMS="256 128 64 32 3"
AE_DECODER_DIMS="16 32 64 128 256 256 512"

# 保持3个Level
LEVELS=(1 2 3)
MASK_THRESH=0.4

# 测试场景
TEST_SCENE="figurines"

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

# 备份原始数据，只保留2张图片
backup_and_reduce_images() {
    local scene=$1
    local data_path="$DATA_ROOT/$scene"
    local backup_path="$DATA_ROOT/${scene}_backup"
    
    log_section "准备测试数据 (保留2张图片)"
    
    # 如果已有备份目录，说明已处理过
    if [ -d "$backup_path" ]; then
        log "备份已存在，跳过准备步骤"
        return 0
    fi
    
    # 备份原始images目录
    mv "$data_path/images" "$backup_path"
    log "备份原始图片到: $backup_path"
    
    # 创建新的images目录，只复制前2张图片
    mkdir -p "$data_path/images"
    local img_count=0
    for img in "$backup_path"/*; do
        if [ -f "$img" ]; then
            if [ $img_count -lt 2 ]; then
                cp "$img" "$data_path/images/"
                log "复制图片: $(basename $img)"
                img_count=$((img_count + 1))
            else
                break
            fi
        fi
    done
    
    log "测试数据准备完成: $img_count 张图片"
}

# 恢复原始数据
restore_original_data() {
    local scene=$1
    local data_path="$DATA_ROOT/$scene"
    local backup_path="$DATA_ROOT/${scene}_backup"
    
    log_section "恢复原始数据"
    
    if [ -d "$backup_path" ]; then
        rm -rf "$data_path/images"
        mv "$backup_path" "$data_path/images"
        log "已恢复原始图片"
    else
        log "备份目录不存在，无需恢复"
    fi
    
    # 清理生成的特征文件
    rm -rf "$data_path/feature"
    rm -rf "$data_path/language_features"
    rm -rf "$data_path/language_features_dim3"
    log "已清理特征文件"
}

# 清理测试输出
clean_test_output() {
    local scene=$1
    log_section "清理测试输出"
    
    rm -rf "$OUTPUT_ROOT/lerf_$scene"
    rm -rf "$OUTPUT_ROOT/lerf_${scene}_langsplat_"*
    rm -rf "$OUTPUT_ROOT/lerf_${scene}_"*
    rm -rf "$AE_CKPT_ROOT/lerf_$scene"
    rm -rf "$EVAL_OUTPUT_ROOT/${scene}_level"*
    rm -rf "$PROJECT_ROOT/autoencoder/ckpt/lerf_$scene"
    
    log "清理完成"
}

# ==================== 步骤函数 (与train_all_lerf.sh一致) ====================

train_3dgs() {
    local scene=$1
    local data_path="$DATA_ROOT/$scene"
    local output_path="$OUTPUT_ROOT/lerf_$scene"
    
    log_section "步骤1: 3DGS训练 - 场景: $scene"
    
    if [ -d "$output_path/point_cloud/iteration_1000" ]; then
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
    local ae_ckpt_dir="$AE_CKPT_ROOT/lerf_$scene"
    
    log_section "步骤3: Autoencoder训练 - 场景: $scene"
    
    activate_env langsplat
    cd "$PROJECT_ROOT/autoencoder"
    
    if [ -f "$ae_ckpt_dir/ae_ckpt/best_ckpt.pth" ]; then
        log "Autoencoder checkpoint已存在，跳过训练"
        return 0
    fi
    
    python train.py \
        --dataset_path "$data_path" \
        --dataset_name "lerf_$scene" \
        --encoder_dims $AE_ENCODER_DIMS \
        --decoder_dims $AE_DECODER_DIMS \
        --output_dir "$ae_ckpt_dir"
    
    log "Autoencoder训练完成: $ae_ckpt_dir"
}

setup_ae_ckpt_link() {
    local scene=$1
    
    cd "$PROJECT_ROOT/autoencoder"
    
    local expected_dir="ckpt/lerf_$scene"
    local expected_file="$expected_dir/best_ckpt.pth"
    local actual_file="$expected_dir/ae_ckpt/best_ckpt.pth"
    
    if [ -f "$actual_file" ]; then
        mkdir -p "$expected_dir"
        if [ ! -f "$expected_file" ]; then
            cp "$actual_file" "$expected_file"
            log "复制checkpoint: $actual_file -> $expected_file"
        fi
    fi
}

generate_3d_features() {
    local scene=$1
    local data_path="$DATA_ROOT/$scene"
    
    log_section "步骤4: 生成3维特征 - 场景: $scene"
    
    activate_env langsplat
    cd "$PROJECT_ROOT/autoencoder"
    
    setup_ae_ckpt_link $scene
    
    if [ -d "$data_path/language_features_dim3" ]; then
        local feature_count=$(find "$data_path/language_features_dim3" -name "*.npy" | wc -l)
        if [ $feature_count -gt 0 ]; then
            log "3维特征已存在 ($feature_count 个文件)，跳过生成"
            return 0
        fi
    fi
    
    python test.py \
        --dataset_path "$data_path" \
        --dataset_name "lerf_$scene" \
        --encoder_dims $AE_ENCODER_DIMS \
        --decoder_dims $AE_DECODER_DIMS
    
    log "3维特征生成完成"
}

train_langsplat() {
    local scene=$1
    local level=$2
    local data_path="$DATA_ROOT/$scene"
    local gs_model_path="$OUTPUT_ROOT/lerf_$scene"
    local output_path="$OUTPUT_ROOT/lerf_${scene}_langsplat_$level"
    
    log "步骤5.$level: LangSplat训练 - Level $level"
    
    activate_env langsplat
    cd "$PROJECT_ROOT"
    
    if [ -d "$output_path/point_cloud/iteration_500" ]; then
        log "LangSplat模型已存在，跳过训练: $output_path"
        return 0
    fi
    
    python train.py \
        -s "$data_path" \
        -m "$output_path" \
        --model_path "$gs_model_path" \
        --include_feature \
        --iterations $LANGSPLAT_ITERATIONS \
        --checkpoint_iterations $LANGSPLAT_CHECKPOINT_ITERATIONS \
        --feature_level $level \
        --eval
    
    log "LangSplat训练完成: Level $level -> $output_path"
}

render_langsplat() {
    local scene=$1
    local level=$2
    local output_path="$OUTPUT_ROOT/lerf_${scene}_langsplat_$level"
    
    log "步骤6.$level: 渲染 - Level $level"
    
    activate_env langsplat
    cd "$PROJECT_ROOT"
    
    python render.py -m "$output_path" --include_feature --feature_level $level
    
    log "渲染完成: Level $level"
}

create_eval_symlinks() {
    local scene=$1
    
    log_section "步骤7: 创建评估符号链接 - 场景: $scene"
    
    cd "$OUTPUT_ROOT"
    
    for level in "${LEVELS[@]}"; do
        local src="lerf_${scene}_langsplat_$level"
        local dst="lerf_${scene}_$level"
        
        if [ -d "$src" ]; then
            [ -L "$dst" ] && rm "$dst"
            [ -e "$dst" ] && mv "$dst" "${dst}.bak"
            ln -s "$src" "$dst"
            log "创建符号链接: $dst -> $src"
        fi
    done
    
    local label_src="$LABEL_ROOT/$scene"
    local label_dst="$LABEL_ROOT/lerf_$scene"
    
    if [ -d "$label_src" ] && [ ! -L "$label_dst" ]; then
        [ -e "$label_dst" ] && mv "$label_dst" "${label_dst}.bak"
        ln -s "$scene" "$label_dst"
        log "创建GT标签符号链接: $label_dst -> $label_src"
    fi
}

# ==================== 主测试流程 ====================

run_test() {
    local scene=$TEST_SCENE
    
    log_section "LangSplat 快速测试 (场景: $scene, 2张图片, 3个Level)"
    
    check_conda_env "langsplat"
    check_conda_env "3dgs"
    
    mkdir -p "$OUTPUT_ROOT" "$AE_CKPT_ROOT" "$EVAL_OUTPUT_ROOT"
    
    # 备份并减少图片
    backup_and_reduce_images $scene
    
    # 步骤1: 3DGS训练
    train_3dgs $scene
    
    # 步骤2: 语言特征提取
    extract_language_features $scene
    
    # 步骤3: Autoencoder训练
    train_autoencoder $scene
    
    # 步骤4: 生成3维特征
    generate_3d_features $scene
    
    # 步骤5-6: LangSplat训练和渲染 (3个Level)
    for level in "${LEVELS[@]}"; do
        log "==================== 处理 Level $level ===================="
        train_langsplat $scene $level
        render_langsplat $scene $level
    done
    
    # 步骤7: 创建符号链接
    create_eval_symlinks $scene
    
    log_section "测试完成!"
    log "输出目录: $OUTPUT_ROOT"
    log "使用 './test_pipeline.sh restore' 恢复原始数据"
}

# 主入口
case "${1:-run}" in
    run)
        run_test
        ;;
    restore)
        restore_original_data $TEST_SCENE
        ;;
    clean)
        clean_test_output $TEST_SCENE
        restore_original_data $TEST_SCENE
        ;;
    *)
        echo "用法: $0 [run|restore|clean]"
        echo "  run     - 运行测试"
        echo "  restore - 恢复原始图片数据"
        echo "  clean   - 清理测试输出并恢复原始数据"
        exit 1
        ;;
esac
