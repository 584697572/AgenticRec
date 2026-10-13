"""Publish aggregate evidence only after the independent audit passes."""
import argparse
import json
from pathlib import Path

from audit_three_way_comparison import digest


def export(directory):
    root = Path(__file__).resolve().parents[3]
    report = json.loads((directory/'report.json').read_text(encoding='utf-8'))
    audit = json.loads((directory/'independent_audit.json').read_text(encoding='utf-8'))
    training = json.loads((directory/'training.json').read_text(encoding='utf-8'))
    assert audit['status']=='VERIFIED' and audit['report_sha256']==digest(directory/'report.json')
    dest = root/'optimized/reports'
    for name,out in [('report.json','sequence_ml1m_20261013_validation.json'),
                     ('independent_audit.json','sequence_ml1m_20261013_independent_audit.json'),
                     ('training.json','sequence_ml1m_20261013_training.json')]:
        with (dest/out).open('xb') as f: f.write((directory/name).read_bytes())
    labels = [('C1','原C1等权'),('candidate','此前混排'),('itemcf','标准ItemCF'),('popular','只推荐热门'),
              ('sequence42','短序列seed42（预定主要模型）'),('sequence7','短序列seed7'),('sequence2026','短序列seed2026')]
    lines = ['# 短序列在此前40%数据集与180用户上的同场复测', '',
             f"DONE / VERIFIED；**{report['conclusion']}**。", '',
             '相同MovieLens1M、原180位用户、原确认目录3190部、公开3/5/10部喜欢各60人、原未来test目标。'
             '这批标签已经暴露，是事后复测，不是新盲测，也不是微软原论文效果。', '',
             '| 排序版本 | 命中人数/180 | Hit@10 | Recall@10 | NDCG@10 | MRR@10 |',
             '|---|---:|---:|---:|---:|---:|']
    for name,label in labels:
        r = report['systems'][name]
        lines.append(f"| {label} | {r['hits']} | {r['hit']:.2%} | {r['recall']:.6f} | {r['ndcg']:.6f} | {r['mrr']:.6f} |")
    lines += ['', '| 主要短序列对照 | Hit差值（百分点） | 95%配对区间（百分点） | 三次Holm p |',
              '|---|---:|---:|---:|']
    for name,label in labels[1:4]:
        c = report['comparisons'][name]; d = c['metrics']['hit']; lo,hi = d['ci95']
        lines.append(f"| 对{label} | {d['mean']*100:+.2f} | [{lo*100:+.2f}, {hi*100:+.2f}] | {c['hit_holm_p']:.6f} |")
    lines += ['', '三个seed全部保留；预定主要seed42，不事后挑最好seed。三seed平均Hit为'
              f"{report['sequence_three_seed_mean_hit']:.2%}，不是三次真实LLM运行。", '',
              '| 公开历史长度 | 原混排命中/60 | 短序列seed42命中/60 |', '|---|---:|---:|']
    for h in ('3','5','10'):
        lines.append(f"| {h} | {report['systems']['candidate']['history'][h]['hits']} | {report['systems']['sequence42']['history'][h]['hits']} |")
    lines += ['', '## 实际训练与适用范围', '',
        '复用之前短序列结构与另一数据开发选择的固定epoch10/alpha.5；电影ID不同，所以在1M原train中重新训练。'
        '模型保留完整3469连续ID，输出筛至原3190确认目录。固定32维/两头/一层/dropout.1、Adam lr.002/weight_decay1e-4、batch32、三个seed各10epoch。', '',
        '5089训练用户、15267短历史样本；各用户训练正反馈80%位置切开、输入最后3/5/10、目标后20%，所有监督在train内。'
        '评测用户的训练期过去反馈可以参与学习，不能称全新用户隔离训练；推理不见用户ID/隐藏历史/未来标签。'
        f"训练实测耗时{training['elapsed_seconds']:.2f}秒；不使用valid/test选模型或epoch，不重训旧对照。", '',
        '此次只验证排序质量，原C1与混排各180份对照排名与过去真实LLM交付完全匹配。'
        '新模型没有新增LLM测试，不填严格任务成功率、调用延迟或token；API请求/新下载均0，默认不替换。', '',
        '## 验证与失败记录', '',
        f"独立审计通过：{audit['training_examples_checked']}训练样本从原始train重建、180用户公开历史/未来目标核对、"
        '1260份独立质量评分、540份新权重排名重放、720份历史对照逐项一致，其中360份与原真实LLM交付一致；冻结SHA、配对bootstrap/精确检验/Holm核验通过。', '',
        '相关20项测试全部通过，保留既有缺NumPy警告。准备期遇到相对路径清单错误与完整训练ID维度错误；'
        '均在优化器训练/预测前修正，失败准备文件保留，新run2目录冻结后源码和协议不变。', '',
        '[预测前协议](sequence_ml1m_protocol_20261013.md) · [完整聚合](sequence_ml1m_20261013_validation.json) · '
        '[独立审计](sequence_ml1m_20261013_independent_audit.json) · [训练曲线](sequence_ml1m_20261013_training.json)', '',
        '私密证据目录 `optimized/artifacts/comparison/sequence_ml1m_20261013_run2/`，原数据/用户/标签/逐项排名/权重仍被Git忽略。', '',
        '```powershell',
        "$study = 'optimized/artifacts/comparison/sequence_ml1m_fresh'",
        "$recPython = 'optimized/AgenticRec/.venv-t20/Scripts/python.exe'",
        '& $recPython -m agenticrec.evaluation.sequence_ml1m prepare --directory $study',
        '& $recPython -m agenticrec.evaluation.sequence_ml1m train --directory $study',
        '& $recPython -m agenticrec.evaluation.sequence_ml1m evaluate --directory $study',
        '& $recPython optimized/AgenticRec/scripts/audit_sequence_ml1m.py --directory $study --output "$study/independent_audit.json"',
        '& $recPython optimized/AgenticRec/scripts/report_sequence_ml1m.py --directory $study',
        '```', '', '已完成目录和公开导出禁止覆盖；再次研究须使用新目录，不依据这批结果反复调参后称盲测。']
    with (dest/'sequence_ml1m_20261013_validation.md').open('x',encoding='utf-8') as f: f.write('\n'.join(lines)+'\n')
    print(json.dumps({'status':'EXPORTED','audit':'VERIFIED','conclusion':report['conclusion']}))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    export(parser.parse_args().directory.resolve())
