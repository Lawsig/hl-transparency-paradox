# DBA 论文端到端复现脚本 (Windows PowerShell)
# 用法: 在 PowerShell 中执行 .\复现命令.ps1
# 预计运行时间: 2-4 小时（含 VPIN 提取与回归）
# 磁盘需求: ~30 GB（含 HL raw_jsonl + Binance Vision）

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

# ════════════════════════════════════════════════════════════════
# Phase 0: 环境检查
# ════════════════════════════════════════════════════════════════
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  DBA 论文端到端复现脚本" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan

Write-Host "`n[Phase 0] 环境检查..." -ForegroundColor Yellow
python --version
python -c "import pandas, numpy, linearmodels, matplotlib, docx, lz4, boto3; print('All dependencies OK')"

# ════════════════════════════════════════════════════════════════
# Phase 1: 数据下载 (~25 GB)
# ════════════════════════════════════════════════════════════════
$RUN_DOWNLOAD = $false  # 默认跳过下载（假设数据已就位）
if ($RUN_DOWNLOAD) {
    Write-Host "`n[Phase 1] 下载 HL + Binance 数据..." -ForegroundColor Yellow
    # HL 主事件 Oct 7-15
    python scripts\hl_node_fills_downloader.py --dates 20251007-20251015
    # HL 复制事件
    python scripts\hl_node_fills_downloader.py --dates 20251117-20251125
    python scripts\hl_node_fills_downloader.py --dates 20260126-20260203
    # Binance Vision (R1.3)
    python scripts\binance_vision_downloader.py
} else {
    Write-Host "`n[Phase 1] 跳过数据下载（设 \$RUN_DOWNLOAD=`$true 启用）" -ForegroundColor Gray
}

# ════════════════════════════════════════════════════════════════
# Phase 2: 第四章事件研究
# ════════════════════════════════════════════════════════════════
Write-Host "`n[Phase 2] 第四章事件研究..." -ForegroundColor Yellow
python scripts\chapter4_l2book_v2.py
python scripts\ch4_volume_analysis.py

# ════════════════════════════════════════════════════════════════
# Phase 3: 第五章面板回归
# ════════════════════════════════════════════════════════════════
Write-Host "`n[Phase 3] 第五章面板回归..." -ForegroundColor Yellow
python scripts\ch5_Step1to6_run_analysis.py
python scripts\ch5_h31_volume_spread.py
python scripts\vpin_extract_takers.py
python scripts\vpin_calculation.py
python scripts\placebo_test.py     # R1.2 安慰剂

# ════════════════════════════════════════════════════════════════
# Phase 4: 附录数据生成
# ════════════════════════════════════════════════════════════════
Write-Host "`n[Phase 4] 附录数据生成..." -ForegroundColor Yellow
# Appendix A
python scripts\run_robustness_grid.py
python scripts\recompute_w0_from_ch4.py
# Appendix D
python scripts\model_simulation.py
# Appendix E (跨交易所)
python scripts\binance_panel_build.py
python scripts\cross_exchange_ddd.py
python scripts\vpin_binance.py
# Appendix F (多事件)
python scripts\hl_replication_panel_build.py
python scripts\replication_twfe_compare.py
python scripts\hl_all_coin_hourly_liq.py

# ════════════════════════════════════════════════════════════════
# Phase 5: 图表生成
# ════════════════════════════════════════════════════════════════
Write-Host "`n[Phase 5] 图表生成..." -ForegroundColor Yellow
python scripts\fig5_1_generate.py
python scripts\fig5_2_generate.py
python scripts\fig5_3_spread_vshape.py

# ════════════════════════════════════════════════════════════════
# Phase 6: 提示更新文档
# ════════════════════════════════════════════════════════════════
Write-Host "`n═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  复现完成! 后续手动步骤:" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  1. 在 Word 中打开 完整版论文.docx"
Write-Host "  2. 右键目录 → '更新域 → 更新整个目录'"
Write-Host "  3. 文件 → 另存为 PDF（用于提交）"
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
