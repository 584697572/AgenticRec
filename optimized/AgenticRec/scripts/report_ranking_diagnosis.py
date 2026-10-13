"""Export only aggregate diagnostics after score, ranking and timeline audits."""
import argparse
import json
from pathlib import Path

from audit_three_way_comparison import digest


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def export(directory):
    root = Path(__file__).resolve().parents[3]
    report = read(directory/'report.json')
    audit = read(directory/'independent_audit.json')
    replay = read(directory/'ranking_replay_audit.json')
    timeline = read(directory/'timeline_audit.json')
    assert all(a['status']=='VERIFIED' for a in (audit,replay,timeline))
    assert audit['report_sha256']==digest(directory/'report.json')
    assert replay['rankings_sha256']==report['rankings_sha256']==digest(directory/'rankings.json')
    for name,sha in read(directory/'contract.json')['files'].items(): assert digest(root/name)==sha,name
    dest = root/'optimized/reports'
    for src,out in [('report.json','ranking_diagnosis_20261013.json'),
                    ('independent_audit.json','ranking_diagnosis_20261013_independent_audit.json'),
                    ('ranking_replay_audit.json','ranking_diagnosis_20261013_ranking_audit.json'),
                    ('timeline_audit.json','ranking_diagnosis_20261013_timeline_audit.json')]:
        with (dest/out).open('xb') as f: f.write((directory/src).read_bytes())
    main = report['systems']['sequence42']
    lines = ['# 原40%队列：为什么命中提升有限', '',
        '**DONE / VERIFIED：已定位可量化瓶颈，未训练或确认新模型收益。** 原180质量用户、既有冻结三seed权重、'
        '同一3190确认目录与未来目标；所有结论为标签已暴露的事后诊断。原正式指标和默认配置不变。', '',
        '## 1. 首要工程方向是前100到前10的排序', '',
        f"主短序列前10命中{int(main['hits10'])}/180（{main['quality']['hit']:.2%}），前100包含目标{main['hits_at_k']['100']}/180。"
        f"105位前10未命中用户中，{report['main_misses_target_in_11_to_100']}位的首个目标在11至100名，"
        f"其余{report['main_misses_no_target_top100']}位在100名之后。这是模型全目录排序的top100，不是外部召回服务。", '',
        '| 排序版本 | 前10命中人数 | 前20覆盖 | 前50覆盖 | 前100覆盖 | 前500覆盖 |',
        '|---|---:|---:|---:|---:|---:|']
    labels = [('sequence42','主短序列'),('mixed','此前混排'),('C1','原C1'),('itemcf','标准ItemCF'),('popular','热门')]
    for name,label in labels:
        r = report['systems'][name]
        lines.append('| '+label+' | '+' | '.join(str(r['hits_at_k'][str(k)]) for k in (10,20,50,100,500))+' |')
    lines += ['', '首个目标位置分布：1–10名75人、11–20名19人、21–50名33人、51–100名18人、101–500名27人、500以后8人。'
        '前100覆盖80.56%不能当Hit@10的已实现结果；重排能否把正确目标提上来必须另训练验证。', '',
        f"混排三个top100来源并集包含目标的用户{report['mixed_recall_union100_hit_users']}/180；"
        f"主短序列未命中的105人中{report['main_misses_recoverable_in_mixed_union']}人的目标在该并集。"
        '四方法前10并集oracle覆盖106/180（58.89%），但最多40个候选且需要知道正确答案才能选，不是实际top10提升，也不能据此保证突破50%。', '',
        '## 2. 公开信息和评测排除集合存在明显差距', '',
        f"180人完整train已评分历史中位{report['input_stats']['full_train_seen_count']['median']:.0f}部，"
        '模型只知道公开3/5/10部。评测目标排除了全部train已评分电影，而排名仅能排除公开已看。'
        f"主短序列前10平均{main['hidden_train_seen_top10_mean']:.2f}部（{main['hidden_train_seen_top10_slot_fraction']:.2%}位置）"
        '实际是隐藏train已看电影；它们不可能命中这次未来目标。混排平均5.54部、热门5.36部。', '',
        '| 版本 | 原Hit@10人数 | 用隐藏train已看列表过滤后的诊断人数 |', '|---|---:|---:|']
    for name,label in labels:
        r = report['systems'][name]
        lines.append(f"| {label} | {int(r['hits10'])} | {int(r['oracle_all_train_seen_filter_hits10'])} |")
    lines += ['', '主短序列此oracle诊断为93/180（51.67%）、热门104/180（57.78%）。'
        '**这是评测器额外提供隐藏过去信息，不是当前匿名系统的真实改善。** 不改变旧答案或评分规则；'
        '若产品能获得用户授权的完整已看列表，可另立有历史授权的场景与所有对照重新公平比较。', '',
        '## 3. 短序列监督和实际预测时间跨度相差很大', '',
        f"按同一个“最后输入至第一个目标”的定义：5089训练用户中位间隔"
        f"{timeline['training_last_input_to_first_target_days']['median']*86400:.0f}秒，"
        f"{timeline['training_first_target_within_one_hour_users']}/{timeline['training_distinct_users']}"
        f"（{timeline['training_first_target_within_one_hour_users']/timeline['training_distinct_users']:.2%}）在1小时内；"
        f"实际180用户中位间隔{timeline['test_last_input_to_first_target_days']['median']:.2f}天，最少30.01天。", '',
        f"训练用户中{timeline['training_first_target_same_timestamp_users']}/5089（41.36%）的输入/目标切点时间戳相同。"
        'MovieLens记录的是评分时间，同批评分不等于真实观看顺序；当前(timestamp,item_id)打破同时间戳次序是可复现规则，不能当真实序列事实。'
        '主模型反转历史结果从75/180降至65/180，说明顺序敏感，但未证明这种顺序对应真实观看意图。', '',
        '训练每用户只有一个80%位置切点、3个高度相关视图，15,267行不等于15,267独立用户。'
        'multi-positive softmax对所有未来train正反馈等权；并不直接优化top10至少命中一个。'
        f"平均每行还有{report['training']['earlier_positive_competing_count']['mean']:.2f}个更早正反馈未作为正目标且未被屏蔽，参与softmax竞争。"
        '这些是监督设计候选问题，不能单凭统计断言某种损失替换一定有效。', '',
        '## 4. 有个性化信号，尚未充分转成排名收益', '',
        f"主短序列每份top10与热门平均重合{main['popular_top10_overlap_mean']:.2f}部，top100重合{main['popular_top100_overlap_mean']:.2f}部；"
        f"180份top10共覆盖{main['catalog_top10_distinct_items']}个不同电影，不是简单复制热门榜。", '',
        '| 固定机制探针 | 命中人数/180 | Hit@10 |', '|---|---:|---:|']
    for name,label in [('sequence42','原主模型＋热门'),('sequence42_pure','只用模型分数'),
                       ('sequence42_reverse','反转同一公开历史'),('sequence42_rotated','同长度组换成别人的公开历史')]:
        r = report['systems'][name]
        lines.append(f"| {label} | {int(r['hits10'])} | {r['quality']['hit']:.2%} |")
    lines += ['', '去掉热门后61/180（33.89%）、换别人的输入也61/180；模型使用了公开喜欢信息，热门仍提供补充。'
        '反转/换人是人为机制探针，不是真实交付，也不以更高的某个探针选择生产配置；此处为描述性结果，未新增显著性筛选。', '',
        '另外，命中用户未来正目标数量中位43个，未命中用户中位8个。Hit受已观察正目标多少影响，'
        '不是统一难度的真实满意度概率；保持原指标同时应报告目标数量分层，不能换指标掩盖负结果。', '',
        '## 建议顺序与验证边界', '',
        '先修改训练样本设计：在train中采用真实时间边界、多切点，检查同时间戳批次与可用的较长预测间隔；'
        '在开发集验证监督和负项处理。再用现有top100/来源并集训练排序器，主目标仍是Hit@10。'
        '若用户愿意提供完整已看历史，单独验证该产品场景，不混入原匿名对照。'
        '此次不自动开始这些训练；已有180人仅作诊断，后续不能根据它反复选参再称盲测。', '',
        '## 证据与复核', '',
        '完整1800份目录排序、1260历史前10逐项重放、1800独立评分/位置/重合/隐藏历史oracle、'
        '15,267训练统计重建和冻结SHA均通过。另独立从固定权重重放1080完整排序、重建720完整基线排名/180候选并集，'
        '并从raw train与processed graph两路核验相同时间间隔定义。'
        '新增10项诊断测试、相关30项一次全部通过。保留既有PyTorch缺NumPy和mask警告。API0、训练0、下载0。', '',
        '[聚合诊断](ranking_diagnosis_20261013.json) · [独立评分审计](ranking_diagnosis_20261013_independent_audit.json) · '
        '[完整排序重放](ranking_diagnosis_20261013_ranking_audit.json) · [时间跨度审计](ranking_diagnosis_20261013_timeline_audit.json) · '
        '[预测前检查范围](ranking_diagnosis_protocol_20261013.md)', '',
        '私密证据目录 `optimized/artifacts/comparison/ranking_diagnosis_20261013/`。'
        '两个补充审计脚本在初版诊断结束后添加，仅扩大独立核验，未修改冻结运行源码/权重/排名/标签；各脚本SHA随对应审计公开。', '',
        '```powershell',
        "$recPython = 'optimized/AgenticRec/.venv-t20/Scripts/python.exe'",
        "$study = 'optimized/artifacts/comparison/ranking_diagnosis_fresh'",
        '& $recPython -m agenticrec.evaluation.ranking_diagnosis --directory $study',
        '& $recPython optimized/AgenticRec/scripts/audit_ranking_diagnosis.py --directory $study --output "$study/independent_audit.json"',
        '& $recPython optimized/AgenticRec/scripts/replay_ranking_diagnosis.py --directory $study --output "$study/ranking_replay_audit.json"',
        '& $recPython optimized/AgenticRec/scripts/audit_ranking_timeline.py --directory $study --output "$study/timeline_audit.json"',
        '```', '', '已完成证据禁止覆盖。公开导出文件固定命名且采用排他创建；不会分发数据、逐用户标签/排名或模型权重。']
    with (dest/'ranking_diagnosis_20261013.md').open('x',encoding='utf-8') as f: f.write('\n'.join(lines)+'\n')
    print(json.dumps({'status':'EXPORTED','diagnosis':'VERIFIED','api_requests':0}))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    export(parser.parse_args().directory.resolve())
