#!/bin/bash
# =====================================================================
# LLM图谱构建脚本 - 48GB显存版（一次性构建高质量图谱）
# =====================================================================
# 配置:
#   - 使用训练集新闻（默认上限 100,000 条，可调）
#   - Qwen2.5-14B语义提取
#   - 6类关系类型标注
#   - 预计时间: 取决于batch/推理参数，通常为数小时量级
# =====================================================================

set -e  # 遇到错误立即停止
cd /root/autodl-tmp/paper

# =====================================================================
# 检查是否在screen/tmux中运行
# =====================================================================
check_terminal() {
    if [ -z "$STY" ] && [ -z "$TMUX" ]; then
        echo "❌ 错误：未检测到screen或tmux会话！"
        echo ""
        echo "由于LLM图谱构建可能持续数小时，必须在screen/tmux中运行以防止SSH断连。"
        echo ""
        echo "请使用以下命令之一："
        echo "  方案1 (screen): screen -S llm_graph"
        echo "  方案2 (tmux):   tmux new -s llm_graph"
        echo ""
        echo "然后重新运行此脚本："
        echo "  ./run_llm_graph.sh"
        echo ""
        echo "分离会话:"
        echo "  screen: Ctrl+A, D"
        echo "  tmux:   Ctrl+B, D"
        echo ""
        echo "重新连接:"
        echo "  screen: screen -r llm_graph"
        echo "  tmux:   tmux attach -t llm_graph"
        echo ""
        exit 1
    fi
    
    # 显示当前会话信息
    if [ -n "$STY" ]; then
        echo "✅ 检测到screen会话: $STY"
    elif [ -n "$TMUX" ]; then
        echo "✅ 检测到tmux会话: $TMUX"
    fi
}

# =====================================================================
# 硬件检测
# =====================================================================
echo "====================================================================="
echo "🖥️  硬件配置检测"
echo "====================================================================="
echo ""
echo "GPU信息:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""
echo "内存信息:"
free -h | grep Mem
echo ""
echo "CPU核心数: $(nproc)"
echo ""
echo "磁盘空间:"
df -h | grep -E '(Filesystem|/root|autodl-tmp)' | head -3
echo ""

# 检查GPU显存
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
if [ "$GPU_MEMORY" -lt 40000 ]; then
    echo "⚠️  警告：检测到显存 ${GPU_MEMORY}MB < 48GB"
    echo "   LLM模式可能会遇到显存不足问题"
    read -p "是否继续？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查磁盘空间
FREE_SPACE=$(df -BG /root/autodl-tmp | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$FREE_SPACE" -lt 30 ]; then
    echo "⚠️  警告：磁盘剩余空间 ${FREE_SPACE}GB < 30GB"
    echo "   可能空间不足"
    read -p "是否继续？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# =====================================================================
# 配置确认
# =====================================================================
echo "====================================================================="
echo "🧠 LLM图谱构建配置"
echo "====================================================================="
echo ""
echo "📊 数据配置:"
echo "  - 股票范围: S&P 500 成分股 (~366只)"
echo "  - 新闻上限: 100,000条 (默认，可通过 --max_total 调整)"
echo "  - 采样策略: 分层采样 (每只股票最多200条；可通过 --max_per_ticker 调整)"
echo ""
echo "🤖 模型配置:"
echo "  - LLM模型: Qwen2.5-14B-Instruct"
echo "  - 关系类型: 6类 (supply/competition/cooperation/merger/lawsuit/investment)"
echo "  - 推理精度: FP16"
echo "  - 显存占用: ~26GB (模型) + ~15GB (数据) = ~41GB"
echo ""
echo "⏱️  时间预估:"
echo "  - 模型加载: 2-3分钟"
echo "  - 图谱构建: 取决于batch/推理参数与GPU吞吐（通常为数小时量级）"
echo ""
echo "💾 输出:"
echo "  - 邻接矩阵: data/processed/Graph_Adjacency.npy"
echo "  - Checkpoint: 每10000条自动保存"
echo ""
echo "⚙️  推理加速参数（可按需修改环境变量）:"
echo "  - LLM_BATCH_SIZE=48 (默认从48开始，OOM会自动降)"
echo "  - LLM_MAX_NEW_TOKENS=96 (关系抽取足够)"
echo "  - LLM_MAX_INPUT_TOKENS=1536"
echo "  - LLM_DO_SAMPLE=0 (确定性解码，结果更可复现)"
echo ""
echo "📈 预期质量:"
echo "  - 边数: 35,000+ (规则匹配只有17K)"
echo "  - 密度: 0.48+ (规则匹配只有0.24)"
echo "  - 准确率: 90%+ (规则匹配只有30%)"
echo "  - 关系类型: 带6类标签 (论文可单独分析)"
echo ""
echo "====================================================================="

# 检查terminal
check_terminal

echo ""
read -p "⚠️  确认开始可能持续数小时的LLM图谱构建？(输入YES确认) " confirm
if [ "$confirm" != "YES" ]; then
    echo "❌ 已取消"
    exit 0
fi

# =====================================================================
# 开始构建
# =====================================================================
echo ""
echo "====================================================================="
echo "🚀 开始LLM图谱构建"
echo "====================================================================="
echo ""
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
START_TIME=$(date +%s)
echo ""

# 记录日志
LOG_FILE="outputs/logs/llm_graph_$(date +%Y%m%d_%H%M%S).log"
mkdir -p outputs/logs

echo "日志文件: $LOG_FILE"
echo ""

# 运行构建（同时输出到终端和日志文件）
# 默认从 batch=48 开始，关系抽取用确定性解码以提高可复现性与速度
export LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-48}"
export LLM_MAX_NEW_TOKENS="${LLM_MAX_NEW_TOKENS:-96}"
export LLM_MAX_INPUT_TOKENS="${LLM_MAX_INPUT_TOKENS:-1536}"
export LLM_DO_SAMPLE="${LLM_DO_SAMPLE:-0}"

python -m dataProcessed.build_graph --use_llm 2>&1 | tee "$LOG_FILE"

# 捕获退出状态
EXIT_CODE=${PIPESTATUS[0]}

# =====================================================================
# 构建完成
# =====================================================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "====================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ LLM图谱构建成功完成！"
else
    echo "❌ 构建过程中出现错误 (退出码: $EXIT_CODE)"
fi
echo "====================================================================="
echo ""
echo "⏱️  总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
echo "📅 完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    # =====================================================================
    # 图谱统计
    # =====================================================================
    echo "====================================================================="
    echo "📊 图谱质量统计"
    echo "====================================================================="
    
    python -c "
import numpy as np
import os

graph_path = 'data/processed/Graph_Adjacency.npy'
if os.path.exists(graph_path):
    adj = np.load(graph_path)
    n = adj.shape[0]
    edges = (adj.sum() - n) / 2
    density = edges / (n * (n - 1) / 2)
    degrees = adj.sum(axis=1) - 1
    connected = (degrees > 0).sum()
    
    print(f'')
    print(f'  节点数 (股票数): {n}')
    print(f'  边数 (关系数): {int(edges):,}')
    print(f'  图密度: {density:.4f}')
    print(f'  连接率: {connected}/{n} ({connected/n*100:.1f}%)')
    print(f'  平均度: {degrees.mean():.1f}')
    print(f'  最大度: {int(degrees.max())}')
    print(f'')
    
    # 质量评估
    print('  质量评估:')
    if edges > 30000:
        print('    ✅ 边数优秀 (>30K)')
    elif edges > 25000:
        print('    ✅ 边数良好 (>25K)')
    else:
        print(f'    ⚠️  边数偏少 ({int(edges)})')
    
    if density > 0.4:
        print('    ✅ 密度优秀 (>0.4)')
    elif density > 0.3:
        print('    ✅ 密度良好 (>0.3)')
    else:
        print(f'    ⚠️  密度偏低 ({density:.4f})')
    
    if connected/n > 0.95:
        print('    ✅ 连接性优秀 (>95%)')
    else:
        print(f'    ⚠️  有 {n-connected} 个孤立节点')
    
    print('')
else:
    print('❌ 未找到图谱文件')
" || echo "统计脚本执行失败"
    
    echo "====================================================================="
    echo ""
    echo "📁 输出文件:"
    echo "  - 图谱: data/processed/Graph_Adjacency.npy"
    echo "  - 日志: $LOG_FILE"
    echo ""
    echo "🎯 下一步:"
    echo "  1. 检查图谱质量是否符合预期"
    echo "  2. 如果满意，开始训练模型:"
    echo "     python -m training.train_full"
    echo ""
    echo "  3. 如果需要重新构建:"
    echo "     rm data/processed/Graph_Adjacency.npy"
    echo "     ./run_llm_graph.sh"
    echo ""
else
    echo "❌ 构建失败，请检查日志:"
    echo "   tail -100 $LOG_FILE"
    echo ""
    echo "常见问题:"
    echo "  1. 显存不足 (OOM)"
    echo "     - 检查: watch -n 1 nvidia-smi"
    echo "     - 解决: 减少 max_total 参数"
    echo ""
    echo "  2. 模型加载失败"
    echo "     - 检查模型路径: /root/autodl-tmp/models/qwen/Qwen2.5-14B-Instruct"
    echo "     - 重新下载模型"
    echo ""
    echo "  3. 磁盘空间不足"
    echo "     - 检查: df -h"
    echo "     - 清理: 删除不必要的文件"
    echo ""
fi

echo "====================================================================="

exit $EXIT_CODE
