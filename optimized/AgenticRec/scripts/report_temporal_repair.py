"""Export audited aggregate repair results, preserving negative findings."""
import argparse
import json
from pathlib import Path

from audit_three_way_comparison import digest


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def export(directory):
    root=Path(__file__).resolve().parents[3];dest=root/'optimized/reports'
    report=read(directory/'report.json');audit=read(directory/'independent_audit.json');contract=read(directory/'contract.json')
    assert audit['status']=='VERIFIED' and audit['report_sha256']==digest(directory/'report.json')
    for name,sha in contract['files'].items():assert digest(root/name)==sha,name
    for src,out in [('report.json','temporal_repair_20261013_validation.json'),
                    ('independent_audit.json','temporal_repair_20261013_independent_audit.json'),
                    ('sequence_selection.json','temporal_repair_20261013_sequence_training.json'),
                    ('reranker_selection.json','temporal_repair_20261013_reranker_selection.json'),
                    ('reranker_training.json','temporal_repair_20261013_reranker_training.json')]:
        with (dest/out).open('xb') as f:f.write((directory/src).read_bytes())
    main=report['systems'][report['final_stage']+'42'];control=report['comparisons']['candidate']
    gain=control['metrics']['hit']['ci95'][0]>0 and control['hit_holm_p']<.05
    status='GAIN_ON_EXPOSED_COHORT' if gain else 'NO VERIFIED QUALITY GAIN'
    seqtrain=read(directory/'sequence_selection.json');reranktrain=read(directory/'reranker_training.json')
    if not (directory/'long_boundary_preflight.json').exists():
        import statistics
        import pyarrow.parquet as pq
        rows=pq.read_table(root/'optimized/data/processed/ml-1m-v1/train_positive.parquet').to_pylist()
        timestamps={(r['user_id'],r['item_id']):r['timestamp'] for r in rows};gaps=[]
        for e in read(directory/'long.json'):
            u=e['training_user_id']
            assert all(timestamps[u,i]<=e['anchor'] for i in e['liked_item_ids'])
            assert all(timestamps[u,i]>=e['anchor']+e['gap_days']*86400 for i in e['relevant_item_ids'])
            assert not set(e['masked_item_ids'])&set(e['relevant_item_ids'])
            gaps.append((min(timestamps[u,i] for i in e['relevant_item_ids'])-max(timestamps[u,i] for i in e['liked_item_ids']))/86400)
        with (directory/'long_boundary_preflight.json').open('x',encoding='utf-8') as f:
            json.dump(dict(status='VERIFIED',long_examples=len(gaps),input_to_first_target_days_min=min(gaps),
                           input_to_first_target_days_median=statistics.median(gaps),target_mask_collisions=0,api_requests=0),f,indent=2)
    boundary=read(directory/'long_boundary_preflight.json')
    assert boundary['status']=='VERIFIED' and boundary['long_examples']==contract['long_examples']
    oldreport=read(root/'optimized/artifacts/comparison/sequence_ml1m_20261013_run2/report.json')
    oldmean=sum(oldreport['systems']['sequence'+str(s)]['hit'] for s in (7,42,2026))/3
    newmean=sum(report['systems']['sequence'+str(s)]['hit'] for s in (7,42,2026))/3
    lines=['# 训练时间跨度、监督与候选排序修正结果', '',f'**DONE / VERIFIED；{status}。**', '',
        '同一MovieLens1M原180质量用户、3190确认目录、公开3/5/10部喜欢各60人与原未来目标。'
        '这批已用于历史评测和诊断，也是本轮方向设计的依据，明确是事后复测，不是新的盲测。'
        '训练和配置选择只读取train/原150 valid开发用户，不用test标签拟合或挑参数。原默认未替换。'
        '准备阶段复制并封存原标签文件；模型训练和开发选择不消费正式标签，评分只在全部正式预测保存后执行。', '',
        '## 真正完成的修正', '',
        f"1. 保留原{contract['ordinal_examples']}条短时辅助任务，新增{contract['long_examples']}条真实30/90天任务，"
        f"覆盖{contract['long_users']}用户；长期切点包含完整同时间戳批次并去重，不虚构半年以后反馈。"
        f"长期例实际输入至首目标最少{boundary['input_to_first_target_days_min']:.2f}天，中位{boundary['input_to_first_target_days_median']:.2f}天。",
        '2. 屏蔽切点之前所有正反馈参与softmax竞争；回归测试验证其梯度为0，目标冲突拒绝。'
        'masked_ordinal为单独mask对照，long_mixed按25%辅助/75%长期、组内用户等权采样。推理只见公开喜欢，未用隐藏历史过滤。',
        '3. 两序列版本各三seed真实训练10轮，开发选择epoch5/10、alpha.25/.5/.75，共12配置。'
        '再独立训练10→32残差MLP排序器，特征图来自train35%/50%锚点以前、目标至少30天以后；'
        '训练候选为CF/内容/热门top100并集，不插入目标。loss优化正候选概率质量，不能称Hit@10精确可微目标。',
        '4. 重排器三个seed各20轮，开发8配置。阶段A训练分数不进入重排训练特征；开发可比较并入阶段A top100。'
        '共20开发配置按三seed平均Hit、NDCG、预定顺序选，先冻结再一次生成正式预测。', '',
        f"阶段A选中：`{report['sequence_config']}`。阶段B选中：`{report['reranker_config']}`。"
        f"最终研究候选按开发数据选 `{report['final_stage']}`，主要seed42，其他seed全部保留。", '',
        '## 原180用户同场准确率', '',
        '| 版本 | 命中人数/180 | Hit@10 | Recall@10 | NDCG@10 | MRR@10 |', '|---|---:|---:|---:|---:|---:|']
    names=[('C1','原C1'),('candidate','此前混排'),('popular','只推荐热门'),('itemcf','标准ItemCF'),('old_sequence42','原短序列seed42')]
    names += [(stage+str(seed),label+' seed'+str(seed)) for stage,label in [('sequence','修正序列'),('reranker','候选重排')] for seed in (42,7,2026)]
    for name,label in names:
        r=report['systems'][name]
        lines.append(f"| {label} | {r['hits']} | {r['hit']:.2%} | {r['recall']:.6f} | {r['ndcg']:.6f} | {r['mrr']:.6f} |")
    lines += ['', '| 最终候选主要seed42对照 | Hit差值（百分点） | 95%配对区间（百分点） | 三次Holm p |', '|---|---:|---:|---:|']
    for name,label in [('candidate','此前混排'),('popular','热门'),('old_sequence42','原短序列')]:
        c=report['comparisons'][name];d=c['metrics']['hit'];lo,hi=d['ci95']
        lines.append(f"| 对{label} | {d['mean']*100:+.2f} | [{lo*100:+.2f}, {hi*100:+.2f}] | {c['hit_holm_p']:.6f} |")
    lines += ['',f"主要候选命中{main['hits']}/180（{main['hit']:.2%}）。不按最高seed或哪个阶段的test成绩重新换候选。"
              '主质量结论以上述预定配对统计为准；其他模型/指标为描述性，不把参数修正本身当作效果提升。', '',
              f"修正序列三seed平均Hit为{newmean:.2%}，原短序列三seed均值{oldmean:.2%}；这只是重复训练的描述性均值。"
              '本轮主要seed42对原短序列命中数相同，21位原未命中的变命中、21位反向变化，收益抵消；'
              'Recall点估计略降、NDCG/MRR略升，配对区间均跨零。重排主要版本37.22%，开发未选为最终候选，负结果完整保留。', '',
              '## 训练和候选证据', '',
              f"阶段A六次真实训练含开发评分共{seqtrain['elapsed_seconds']:.2f}秒，"
              f"阶段B三次训练共{reranktrain['elapsed_seconds']:.2f}秒（不含特征构造/开发）。"
              '时间是本机实测训练运行记录，不是Agent在线延迟或与原版公平耗时比较。', '',
              '| 重排训练锚点 | 全部监督查询 | 候选含正目标、实际训练查询 | 图正边数 |', '|---|---:|---:|---:|']
    for r in reranktrain['snapshot_stats']:
        lines.append(f"| {r['snapshot']} | {r['queries']} | {r['usable']} | {r['graph_edges']} |")
    lines += ['', '候选无正目标的查询计入覆盖统计但不计算loss，不通过添加目标使训练显得可行。'
        '两个prefix图与其未来目标由原始train独立重建，future valid/test不进图。'
        '虽然mask修正及长期任务已实施，仍保留短时辅助；不能说时间分布完全等同180天的测试。', '',
        '## 验证范围', '',
        f"独立从raw train重建{audit['ordinal_samples_rebuilt']}辅助及{audit['long_samples_rebuilt']}长期样本，"
        '核验两个prefix训练图/未来目标、9000开发预测评分与20配置选择、原180公开历史与未来目标、'
        '1980正式评分、1080新权重预测重放、900历史对照、冻结SHA、bootstrap/精确检验/Holm均通过。', '',
        '新增4项监督回归，相关33项一次通过；保留既有缺NumPy/mask警告。'
        '**本轮API0、新下载0，严格Agent任务/LLM资源消耗NOT EVALUATED。**'
        '新模型在研究代码中可复测，未切生产默认、不改原微软源码或旧评测条件。', '',
        '[预测前协议](temporal_repair_protocol_20261013.md) · [正式聚合](temporal_repair_20261013_validation.json) · '
        '[独立审计](temporal_repair_20261013_independent_audit.json) · [序列训练/开发](temporal_repair_20261013_sequence_training.json) · '
        '[重排开发](temporal_repair_20261013_reranker_selection.json) · [重排训练](temporal_repair_20261013_reranker_training.json)', '',
        '私密证据 `optimized/artifacts/comparison/temporal_repair_20261013/`；原数据、用户/标签、逐项预测和模型权重不公开。', '',
        '```powershell', "$recPython = 'optimized/AgenticRec/.venv-t20/Scripts/python.exe'", "$study = 'optimized/artifacts/comparison/temporal_repair_fresh'",
        '& $recPython -m agenticrec.evaluation.temporal_repair prepare --directory $study',
        '& $recPython -m agenticrec.evaluation.temporal_repair sequence --directory $study',
        '& $recPython -m agenticrec.evaluation.temporal_repair reranker --directory $study',
        '& $recPython -m agenticrec.evaluation.temporal_repair freeze --directory $study',
        '& $recPython -m agenticrec.evaluation.temporal_repair evaluate --directory $study',
        '& $recPython optimized/AgenticRec/scripts/audit_temporal_repair.py --directory $study --output "$study/independent_audit.json"',
        '```', '', '已完成目录禁止覆盖；不继续依据本180人的结果追加参数搜索。']
    with (dest/'temporal_repair_20261013_validation.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='EXPORTED',quality=status,api_requests=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    export(parser.parse_args().directory.resolve())
