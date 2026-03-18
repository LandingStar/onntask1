#!/bin/bash
#SBATCH --job-name=onn_task1
#SBATCH --output=logs/task1_%j.out
#SBATCH --error=logs/task1_%j.err
#SBATCH --partition=normal        # 请替换为您的DCU队列名称 (例如: dcu)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8         # 根据需要分配CPU核心数
#SBATCH --gres=dcu:1              # 请求1张DCU卡
#SBATCH --time=24:00:00           # 最大运行时间

# 1. 环境变量设置
export PYTHONUNBUFFERED=1

# 2. 加载必要的模块 (根据集群情况修改)
# module load compiler/rocm/5.x.x
# module load anaconda3

# 3. 激活环境
# source activate onn_env

# 4. 路径设置
# 远程根目录: /public/home/landingstar/onn_training/onntask1
# task1 目录应为: /public/home/landingstar/onn_training/onntask1
# 数据集路径: ./continue5 (相对于 task1 目录)

PROJECT_ROOT="/public/home/landingstar/onn_training/onntask1"
cd "$PROJECT_ROOT"

# 5. 检查并创建日志目录
mkdir -p logs

echo "Job Start Time: $(date)"
echo "Current Directory: $(pwd)"
echo "Node List: $SLURM_JOB_NODELIST"

# 6. 运行训练脚本
# 注意：train.py 在 ai_refined_5_detectors 子目录下
# 且 config.json 中的 dataset_path 已配置为 "task1/continue5" 或 "./continue5"
# 如果当前在 task1 根目录，需要调整调用方式

python ai_refined_5_detectors/train.py

echo "Job End Time: $(date)"
