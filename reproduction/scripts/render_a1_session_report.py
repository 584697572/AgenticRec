"""Generate this A1 stage report from verified, non-private summary artifacts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    summary_path = ROOT / "reproduction/native_runs/20261005_session/live_summary.json"
    diagnostics_path = ROOT / "reproduction/native_runs/20261005_session/failure_diagnostics_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert summary["status"] == "VERIFIED" and summary["usage_is_fixture"] is False
    usage = summary["usage"]
    table = "\n".join("| %s | %s | %s | %.6f |" % (
        "original weighted popularity" if method == "popularity" else method,
        row["samples"], row["hits"], row["upstream_text_hit"])
        for method, row in summary["evaluation"].items())
    text = f"""# 原 App 真实多轮与原评测链路（2026-10-05）

本报告由公开验证摘要和失败诊断摘要自动生成；不读取或公开真实对话、target、Key、原始 API 响应。执行源码 commit 为 `09853c04b52c1f9cc6b5ea56ae78f1a8e52ebd85`。继续 A1，不声称论文 A2。

## Completed

- 原 App 两轮真实运行通过：先 3 部 1990 年及之后的喜剧，再改为 3 部 1980 年之前的喜剧；目录、数量、年份、原 Filter/Ranking/Map 和回答标题检查通过。
- 第二轮原 prompt 实际包含第一轮输入和回答，原 memory 保留 4 条。
- 未修改的 one_turn_eval.py main、RecBotWrapper、hit_judge、one_turn_conversation_eval 已执行：recbot、random、原加权 popularity 分支各处理全部 45 条 A1 派生输入。
- 原 conversations 与逐样本命中重新核验，分母未删除失败或未命中样本；重复执行门禁通过，{summary['prior_evidence_files_unchanged']} 个旧连接/单轮证据文件字节不变。

## Files Created / Modified

新增原 App/评测 worker、独立预算 HTTP 边界、执行/验证/诊断/报告脚本及回归测试。公开摘要在 `reproduction/native_runs/20261005_session/`，原始请求、响应、模型日志和对话仅在忽略目录 `reproduction/logs/live_reproduction_20261005/`。更新 STATUS、TASKS、DECISIONS 和 README；规范和 upstream 未修改。

## Commands / Tests

```powershell
& .\\reproduction\\.venv-legacy\\Scripts\\python.exe -m unittest discover -s reproduction/tests -v
& .\\reproduction\\.venv-legacy\\Scripts\\python.exe -m unittest discover -s reproduction/tests -p test_upstream_text_hit.py -v
& .\\reproduction\\.venv-legacy\\Scripts\\python.exe reproduction/scripts/live_reproduction.py --offline
& .\\reproduction\\.venv-legacy\\Scripts\\python.exe reproduction/scripts/live_reproduction.py
& .\\reproduction\\.venv-legacy\\Scripts\\python.exe reproduction/scripts/verify_a1_session.py --directory reproduction/logs/live_reproduction_20261005 --scan-local-key
& .\\reproduction\\.venv-legacy\\Scripts\\python.exe reproduction/scripts/audit_a1_eval_failures.py
```

完整回归在新增指标测试前运行，40 项通过；新增原指标测试另运行 3 项通过，共 43 个不同测试已验证。最初缺新实现的失败回归、原始退出码和 native stdout/stderr 均保留。未知 CLI 参数退出 2，0 请求。源码及文档的 Git whitespace 检查通过；原生 warning 日志保留上游尾随空格与 CRLF，未改写原证据。

## Verification Results

| 分支 | 样本数 | 原模糊命中数 | upstream_text_hit |
| --- | ---: | ---: | ---: |
{table}

以上是正式保存的真实结果；零命中不改写为成功率或算法提升。随机和 popularity 均为原抽样分支、seed 42，popularity 按 visited_num 加权，不是确定性热门 Top-K。num_rec=5、zero-demo、reflection=0、shortening=0；原作者 canonical 50 条及模型/提示条件等价性 NOT VERIFIED。

真实 HTTP 尝试/响应 {summary['http_attempts']}/{summary['remote_api_requests']}，均 HTTP200；实际 usage 为输入 {usage['prompt_tokens']}、输出 {usage['completion_tokens']}、总计 {usage['total_tokens']} token。最多 120 次 / 1024 输出 token/次 / 3 CNY，零重试。按核验的 [DeepSeek 官方高峰价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)估算 {summary['estimated_cost_cny_peak_ceiling']:.6f} CNY；actual_cost_cny=null，账单未查询，客户端不是供应商账单硬限额。

总墙钟（含模型加载与目录编码）{summary['total_elapsed_including_load_seconds']:.6f} 秒。单次运行及各输入延迟只是本机功能证据，未开展完整 benchmark。offline 联调为 94 次 mock、真实 0 请求，不能混入真实 usage 或指标。私有证据扫描未发现当前 Key；Key 和 Key 哈希不进入 Git/证据索引。

## Findings

- {diagnostics['targets_unmapped_by_original_notebook']} / 45 个最终 target 无法由原 movie_map 映射当前目录；原筛选条件只要求对话至少有两个可映射正反馈，不保证最终 target 可映射。
- {diagnostics['mapped_targets_with_different_catalog_year']} 个映射 target 的年份与目录年份不同；保留原 signed-date 行为，不把该映射视为权威 ID ground truth。
- {diagnostics['targets_with_year_suffix']} 个 target 带年份后缀，而 Map 输出通常仅含标题。离线去掉年份的诊断产生 {diagnostics['year_stripped_text_hit_diagnostic_count']} 个模糊命中，但改变了评分目标表示，不能替代正式零结果，也不能称为模型提升；这些模糊命中也未获得记录 Map ID 的对应证明。
- 样本索引 {diagnostics['truncated_response_sample_indices']} 的输出 finish_reason=length，索引 {diagnostics['no_map_sample_indices']} 没有 Map 输出；它仍在 45 个分母中。HTTP200 和 evaluator 完成不代表每条推荐成功。
- 原 partial_ratio 指标能将 Aliens 对 Alien 判为命中，因此不等同 ID Hit@K/NDCG。checkpoint label、原 notebook CSV 格式与缓存输出不能独立恢复训练映射，见 `reproduction/source_alignment_review_20261005.json`。

## Blockers

真实 Key、原 App 多轮和原评测执行链不再阻塞。完整复现仍缺 checkpoint 权威标题映射、矩阵行语义、原包独立许可、canonical 作者子集及 demo/reflection 等原实验条件恢复。本轮 45 条兼容性评测具有目录可达性与表示限制，不支持论文质量结论。

## Deviations from Spec

无新增路线变更。保留既有 A1 差异：DeepSeek 非思考、CPU、zero-demo、关闭 reflection/shortening、API 边界上限及无重试、seed 42、原 notebook 的局部年份 dtype 适配。公开日志不含真实用户偏好或 target。未改原算法/prompt/指标、未删样本、未重置旧额度。

## Current Project State

单轮、多轮与原评测执行链已实证跑通；推荐质量结果如上。T00/T01/T02/T05 DONE；T03 IN_PROGRESS；T04 的多轮及 A1 派生输入评测已完成，整体仍因其余验收条件 BLOCKED；T06—T24 TODO。A2 和新算法收益尚未验证。

## Next Tasks

继续 T03 的权威来源/ID/许可核验和 T04 的原 demo/reflection 条件恢复；保留当前零结果与失败样本。当前 45 条不经事后筛选或改阈值充当更好的 test；后续模型对照按规范 T06 数据→T07 指标→T08 BPR-MF→T09 LightGCN 的前置推进。
"""
    output = ROOT / "reports/a1_multiturn_eval_20261005.md"
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(json.dumps({"report": output.relative_to(ROOT).as_posix(),
        "source_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "source_diagnostics_sha256": hashlib.sha256(diagnostics_path.read_bytes()).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
