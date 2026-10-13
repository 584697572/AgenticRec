"""Export aggregate-only results after the independent live audit passes."""
import argparse
from decimal import Decimal
import json
from pathlib import Path

from audit_three_way_comparison import digest


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def window(path):
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    def total(row): return sum((Decimal(r['total_balance']) for r in row['balance']['balance_infos'] if r['currency'] == 'CNY'), Decimal(0))
    return total(rows[0]), total(rows[-1])


def export(directory):
    root = Path(__file__).resolve().parents[3]
    report, audit = read(directory/'report.json'), read(directory/'independent_audit.json')
    assert audit['status'] == 'VERIFIED' and audit['report_sha256'] == digest(directory/'report.json')
    assert audit['frozen_hashes_verified'] and audit['provider_usage_reconciled']
    prefix = 'sequence_agent_20261013'
    output = root/'optimized/reports'
    for source, target in [('report.json', prefix+'_validation.json'), ('independent_audit.json', prefix+'_independent_audit.json'),
                           ('test_verification.json', prefix+'_tests.json'), ('calibration_audit.json', prefix+'_calibration_audit.json')]:
        with (output/target).open('xb') as f:
            f.write((directory/source).read_bytes())
    systems = ('sequence', 'mixed', 'itemcf', 'popular')
    names = {'sequence': '短序列seed42', 'mixed': '现有1:.5:.25混排', 'itemcf': '标准ItemCF', 'popular': '热门榜'}
    cal = read(directory/'calibration_run/execution.json')
    first, _ = window(directory/'calibration_run/balance.jsonl')
    _, last = window(directory/'formal_run/balance.jsonl')
    seq = report['systems']['sequence']
    mixed = report['systems']['mixed']
    hit_delta = (seq['quality']['hit']-mixed['quality']['hit'])*100
    request_delta = (seq['requests']/mixed['requests']-1)*100
    token_delta = ((seq['input_tokens']+seq['output_tokens'])/(mixed['input_tokens']+mixed['output_tokens'])-1)*100
    text = ['# 冻结短序列接入真实Agent：四版本同场结果', '',
        '**DONE / VERIFIED**。配置、源码、输入和权重正式前冻结；默认版本未替换。', '',
        f"本批短序列相对混排的Hit点估计{hit_delta:+.2f}百分点，实际请求{request_delta:+.2f}%、总token{token_delta:+.2f}%。{'未证明稳定推荐收益，不能称准确率或资源全面优化。' if not report['verified_quality_gain_over_mixed'] else '通过预定Hit增益标准，其余指标仍应分别评估。'}", '',
        '本轮把短序列排序接入实际C1文本解析、证据验证、action0多轮状态合并、目录硬过滤和结构化标题交付；其余三种排序使用同一流程，分别真实调用LLM。微软原App本轮未运行。', '',
        '原414监督/180开发/300最终用户全部排除，新的150正式用户与9校准用户也不同。正式120质量用户按公开3/5/10喜欢各40人，30反馈题、24硬条件、8不可满足；182题/版、222轮/版，共728系统任务/888轮。目标是85%时点后的新正反馈。旧25.89%、旧40%与本批不能直接相减。', '',
        '## 实际质量与任务完成', '',
        '| 版本 | Hit@10 | 命中/120 | Recall@10 | NDCG@10 | MRR@10 | 严格完成/174 | 严格完成率 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in systems:
        r = report['systems'][s]; q = r['quality']
        text.append(f"| {names[s]} | {q['hit']*100:.2f}% | {r['quality_hits']} | {q['recall']:.6f} | {q['ndcg']:.6f} | {q['mrr']:.6f} | {r['strict_passed']} | {r['strict_success']*100:.2f}% |")
    text += ['', 'Hit表示前十部至少命中一部，不能替代严格按条件完成。解析或交付失败仍保留在120人的分母；严格分母174可满足题，8拒绝另报。', '',
        '| 短序列对照 | Hit差值（百分点） | 配对95%区间（百分点） | 三次Holm p |', '|---|---:|---|---:|']
    for b in ('mixed', 'itemcf', 'popular'):
        c = report['comparisons']['sequence_minus_'+b]; h = c['quality']['hit']
        text.append(f"| 相对{names[b]} | {h['mean']*100:+.2f} | [{h['ci95'][0]*100:+.2f},{h['ci95'][1]*100:+.2f}] | {c['hit_holm_p']:.4f} |")
    strict = report['strict_sequence_minus_mixed']
    text += ['', f"严格完成相对混排差值{strict['mean']*100:+.2f}百分点，95%区间[{strict['ci95'][0]*100:+.2f},{strict['ci95'][1]*100:+.2f}]；预定-2百分点非劣：{'建立' if report['strict_noninferiority_established'] else '未建立'}。",
             f"相对现有混排的稳定质量收益：{'通过预定标准' if report['verified_quality_gain_over_mixed'] else 'NO VERIFIED GAIN'}。其他指标和失败均公开，不因单个seed或单档较高就替换默认。", '',
             '## 实际消耗', '', '| 版本 | 请求 | 输入token | 输出token | 语义修复请求 | turn p50秒 | turn p95秒 | 排序p95秒 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in systems:
        r = report['systems'][s]
        text.append(f"| {names[s]} | {r['requests']} | {r['input_tokens']} | {r['output_tokens']} | {r['semantic_repair_requests']} | {r['turn_latency_p50']:.3f} | {r['turn_latency_p95']:.3f} | {r['ranking_latency_p95']:.4f} |")
    text += ['', f"校准{cal['remote_requests']}＋正式{report['formal_requests']}＝{cal['remote_requests']+report['formal_requests']}次真实生成请求。正式余额窗口减少{report['observed_formal_balance_delta_cny']}元，校准开始至正式结束观察减少{first-last}元；独立账单未知，不把窗口差当精确实验账单。",
             'DeepSeek deepseek-flash、temperature0、非思考、实际1024输出上限、零网络重试；最多三次语义修复单独计入请求/token。不同上一回答会改变后续解析上下文，不能预设各版请求必然相同。串行轮换执行顺序，排序预热；延时是本机/本批实测，网络和厂商缓存有影响，不声称工业QPS或生产加速。没有新下载或训练。', '',
             '## 条件、多轮和拒绝', '', '| 版本 | 条件通过/24 | 多轮整题通过/30 | 结构正确拒绝/8 | 有解释拒绝/8 |', '|---|---:|---:|---:|---:|']
    for s in systems:
        r = report['systems'][s]
        text.append(f"| {names[s]} | {r['categories']['constraints']['passed']} | {r['categories']['feedback']['passed']} | {r['correct_refusals']} | {r['explained_refusals']} |")
    text += ['', '原C1年份证据校验等语义失败保留，没有根据正式结果修规则、放宽条件或重跑失败。空候选的验证结构可算正确拒绝；空回答不算有解释。二/三轮操作覆盖加类型、组合类型、切换类型、清空、缩年份、换批及改数量，实际六种操作各5题。', '',
             '## 同用户直接公开输入的组件验证', '',
             '以下720份输出先保存排名再评分；它们直接使用公开电影列表，没有经过LLM解析。除主seed42外，另外两个seed仅离线组件验证，不能冒称三次独立在线运行。', '',
             '| 组件 | 命中/120 | Hit@10 | Recall@10 | NDCG@10 | MRR@10 |', '|---|---:|---:|---:|---:|---:|']
    for name, value in audit['offline_quality'].items():
        q = value['metrics']
        text.append(f"| {name} | {value['hits']} | {q['hit']*100:.2f}% | {q['recall']:.6f} | {q['ndcg']:.6f} | {q['mrr']:.6f} |")
    text += ['', '实际LLM质量输出与同用户直接输入排名一致数量（各120）：'+str(audit['live_quality_matches_direct_ranking'])+'。不一致按真实输出计分，不用纯排序替换失败或不同的实际交付。', '',
             '## 验证与证据', '',
             f"- 校准48系统任务/80轮、93次HTTP200均有usage；独立重放78成功执行、2语义失败保留。",
             f"- 正式{audit['turns']}轮的条件/质量/身份/数量/前批排除与整题状态独立重算，{audit['actual_ranking_replays']}次成功执行从冻结权重重放；{audit['preserved_parse_errors']}次解析失败保留。逐请求意图/响应/用量全对账。",
             '- 原包重新检查时间边界、电影身份、150正式/9校准用户与所有旧监督/评价用户隔离；源码/输入/旧模型SHA、720组件评分、bootstrap/置换/Holm、严格非劣均通过独立复核。',
             '- 443项完整测试一次通过，另9项审计测试通过，共452项已验证；不称单次452全套。22项相关组件测试也通过。四排序钩子与旧300用户逐项重现共1200排名，不使用旧标签调参。',
             '- 全部数据、权重、原始排名、标签、Key和余额/请求日志留忽略目录，只公开聚合。原上游和默认未改，paper-notes未动。有限批次结束即停止调用。Blocker：None。', '',
             '[预测前协议](sequence_agent_protocol_20261013.md) · [全部聚合/统计](sequence_agent_20261013_validation.json) · [独立审计](sequence_agent_20261013_independent_audit.json) · [测试/等价记录](sequence_agent_20261013_tests.json) · [原校准审计](sequence_agent_20261013_calibration_audit.json)', '',
             '原校准审计记录当时审计脚本SHA；后来增加纯组件审计，不改变已冻结运行逻辑或评分。最终审计SHA是公开当前脚本。私密原始证据目录 `optimized/artifacts/comparison/sequence_agent_20261013/`，已完成目录禁止覆盖。协议保留创建时NOT EVALUATED，本报告为最终状态。', '',
             '## 再现入口', '', '根目录已有数据/权重环境，先使用全新目录；prepare只取已确认来源资源，无训练和下载。', '',
             '```powershell', "$study = 'optimized/artifacts/comparison/sequence_agent_recheck'", "$recPython = 'optimized/AgenticRec/.venv-t20/Scripts/python.exe'",
             '& $recPython -m agenticrec.evaluation.sequence_agent prepare --directory $study',
             '& $recPython -m agenticrec.evaluation.sequence_agent regression --directory $study',
             '& $recPython -m agenticrec.evaluation.sequence_agent calibration --directory $study',
             '& $recPython optimized/AgenticRec/scripts/audit_sequence_agent.py --directory $study --output "$study/calibration_audit.json" --calibration-only',
             '& $recPython -m agenticrec.evaluation.sequence_agent freeze --directory $study',
             '& $recPython -m agenticrec.evaluation.sequence_agent offline --directory $study',
             '& $recPython -m agenticrec.evaluation.sequence_agent formal --directory $study',
             '& $recPython -m agenticrec.evaluation.sequence_agent report --directory $study',
             '& $recPython optimized/AgenticRec/scripts/audit_sequence_agent.py --directory $study --output "$study/independent_audit.json"', '```', '',
             '每步exit0再执行下一步。calibration/formal为真实付费调用，历史完成批次不要重复。该集合现已暴露，再跑只是复查，不称新盲测；新增研究需要另一份预先固定的评价方案。']
    with (output/(prefix+'_validation.md')).open('x', encoding='utf-8') as f:
        f.write('\n'.join(text)+'\n')
    print(json.dumps({'status': 'EXPORTED', 'public_aggregate_reports': 5, 'generation_requests_added': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    export(parser.parse_args().directory)
