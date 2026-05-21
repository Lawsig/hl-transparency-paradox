#!/bin/bash
# DBA 论文端到端复现脚本 (Linux/Mac/WSL)
# 用法: bash 复现命令.sh
# 预计运行时间: 2-4 小时
# 磁盘需求: ~30 GB

set -e
export PYTHONIOENCODING=utf-8

echo "═══════════════════════════════════════════════════════════════"
echo "  DBA 论文端到端复现脚本"
echo "═══════════════════════════════════════════════════════════════"

# Phase 0: 环境检查
echo -e "\n[Phase 0] 环境检查..."
python --version
python -c "import pandas, numpy, linearmodels, matplotlib, docx, lz4, boto3; print('All dependencies OK')"

# Phase 1: 数据下载（默认跳过）
RUN_DOWNLOAD=false
if [ "$RUN_DOWNLOAD" = "true" ]; then
    echo -e "\n[Phase 1] 下载 HL + Binance 数据..."
    python scripts/hl_node_fills_downloader.py --dates 20251007-20251015
    python scripts/hl_node_fills_downloader.py --dates 20251117-20251125
    python scripts/hl_node_fills_downloader.py --dates 20260126-20260203
    python scripts/binance_vision_downloader.py
else
    echo -e "\n[Phase 1] 跳过数据下载（设 RUN_DOWNLOAD=true 启用）"
fi

# Phase 2: 第四章
echo -e "\n[Phase 2] 第四章事件研究..."
python scripts/chapter4_l2book_v2.py
python scripts/ch4_volume_analysis.py

# Phase 3: 第五章
echo -e "\n[Phase 3] 第五章面板回归..."
python scripts/ch5_Step1to6_run_analysis.py
python scripts/ch5_h31_volume_spread.py
python scripts/vpin_extract_takers.py
python scripts/vpin_calculation.py
python scripts/placebo_test.py

# Phase 4: 附录
echo -e "\n[Phase 4] 附录数据生成..."
python scripts/run_robustness_grid.py
python scripts/recompute_w0_from_ch4.py
python scripts/model_simulation.py
python scripts/binance_panel_build.py
python scripts/cross_exchange_ddd.py
python scripts/vpin_binance.py
python scripts/hl_replication_panel_build.py
python scripts/replication_twfe_compare.py
python scripts/hl_all_coin_hourly_liq.py

# Phase 5: 图表
echo -e "\n[Phase 5] 图表生成..."
python scripts/fig5_1_generate.py
python scripts/fig5_2_generate.py
python scripts/fig5_3_spread_vshape.py

echo -e "\n═══════════════════════════════════════════════════════════════"
echo "  复现完成! 后续手动步骤:"
echo "═══════════════════════════════════════════════════════════════"
echo "  1. 在 Word 中打开 完整版论文.docx"
echo "  2. 右键目录 → '更新域 → 更新整个目录'"
echo "  3. 文件 → 另存为 PDF（用于提交）"
echo "═══════════════════════════════════════════════════════════════"
