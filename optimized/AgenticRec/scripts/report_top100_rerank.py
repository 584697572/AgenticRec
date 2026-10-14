"""Publish only audited aggregates from the fixed-top100 study."""
import argparse
import json
from pathlib import Path

from audit_three_way_comparison import digest


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def export(directory):
    root=Path(__file__).resolve().parents[3];dest=root/'optimized/reports'
    report=read(directory/'report.json');audit=read(directory/'independent_audit.json')
    assert audit['status']=='VERIFIED' and audit['report_sha256']==digest(directory/'report.json')
    contract=read(directory/'contract.json');frozen=read(directory/'frozen.json')
    for name,sha in contract['files'].items(): assert digest(root/name)==sha,name
    for name,sha in frozen['files'].items(): assert digest(directory/name)==sha,name
    for src,out in [('report.json','top100_rerank_20261014_validation.json'),
                    ('independent_audit.json','top100_rerank_20261014_independent_audit.json'),
                    ('selection.json','top100_rerank_20261014_selection.json'),
                    ('prefix_training.json','top100_rerank_20261014_prefix_training.json'),
                    ('residual_training.json','top100_rerank_20261014_residual_training.json'),
                    ('feature_training.json','top100_rerank_20261014_features.json')]:
        with (dest/out).open('xb') as f:f.write((directory/src).read_bytes())
    main=report['systems'][report['primary_system']];comparison=report['comparisons']['sequence42']
    supported=comparison['metrics']['hit']['ci95'][0]>0 and comparison['hit_holm_p']<.05
    conclusion='GAIN_ON_EXPOSED_COHORT' if supported else 'NO VERIFIED QUALITY GAIN'
    lines=['# 固定前100候选重排实验（2026-10-14）','',f'**DONE / VERIFIED；{conclusion}。**','',
        '同一MovieLens1M原180质量用户、3190确认目录与原未来目标，公开3/5/10部喜欢各60人。'
        '本轮只改变排序，不改变候选池、隐藏已看处理、数据跨度、原默认或微软源码。180人已用于历史诊断，明确是事后复测；'
        '不用本批答案训练或挑参数。参数只由原150 valid开发用户三seed平均Hit选择，先冻结再保存全部正式预测后评分。API0，下载0。','',
        '## 方法与真正执行内容','',
        '1. 基线为原冻结短序列epoch10/alpha.5，各seed原始前100。在完全相同池内对照线性边界残差、MLP边界残差、相同MLP概率质量目标。'
        '15特征明确加入原序列分数/排名，最终为原分数+有界修正；小幅修正上限由strength控制，不重新以热门分数作为基础。',
        '2. 两个已封存train前缀，各三seed实训原短序列10轮，合计60epoch。重排训练分数仅来自目标以前的图与模型；'
        'CF/热门也只用相同前缀。避免直接让完整train模型给它自己学过的未来答案生成堆叠训练特征。'
        '完整train原冻结模型只在开发/正式推理使用，未用valid/test标签训练。',
        '3. 三种残差各三seed20轮，共180epoch；边界目标比较最高正目标与第10高非正目标，辅助平方正则限制改动。'
        '它是Hit@10代理目标，并非精确指标；非正目标只是未观察到未来喜欢，不能称已证实不喜欢。候选无正目标统计但不造答案。',
        '4. 原排序零修正＋三路线×两个epoch×三个strength，共19开发配置，8550条预测。'
        '各路线的开发最佳均复测，最终选择包含原排序，主要seed42固定；没有按正式最高seed或路线换结论。','',
        f"开发选中：`{report['selected']}`；开发平均指标 `{frozen['selected_development_metrics']}`。",'',
        '## 原180人同场结果','',
        '| 排序版本 | 命中人数/180 | Hit@10 | Recall@10 | NDCG@10 | MRR@10 |',
        '|---|---:|---:|---:|---:|---:|']
    names=[('C1','原C1'),('candidate','此前混排'),('popular','热门')]
    for route,label in [('sequence','原短序列'),('linear_boundary','线性边界微调'),('mlp_boundary','MLP边界微调'),('mlp_mass','MLP概率质量微调')]:
        names += [(route+str(s),label+' seed'+str(s)) for s in (42,7,2026)]
    for name,label in names:
        r=report['systems'][name];lines.append(f"| {label} | {r['hits']} | {r['hit']:.2%} | {r['recall']:.6f} | {r['ndcg']:.6f} | {r['mrr']:.6f} |")
    lines+=['','| 开发最终候选seed42对照 | Hit差（百分点） | 95%配对区间（百分点） | 三次Holm p | 新命中/丢失 |',
            '|---|---:|---:|---:|---:|']
    for name,label in [('sequence42','原短序列42'),('candidate','此前混排'),('popular','热门')]:
        c=report['comparisons'][name];d=c['metrics']['hit'];lo,hi=d['ci95']
        lines.append(f"| 对{label} | {d['mean']*100:+.2f} | [{lo*100:+.2f}, {hi*100:+.2f}] | {c['hit_holm_p']:.6f} | {c['wins']}/{c['losses']} |")
    lines+=['',f"最终预定候选 `{report['primary_system']}` 命中{main['hits']}/180（{main['hit']:.2%}）。"
        f"相对原序列新增{comparison['wins']}人命中、丢失{comparison['losses']}人命中。"
        '主要统计以上述最终开发选择为准，未通过时不把点估计改善写成稳定增益。','',
        '各seed候选前100覆盖：'+', '.join(f"{s}: {v['hits']}/180（{v['hit100']:.2%}）" for s,v in report['top100_coverage'].items())+'。'
        '所有重排版本用同一池；前100覆盖不因重排变化，不是前10已实现成功率。','',
        '## 训练候选与验证','',
        '| 前缀 | seed | 全查询 | 可训练查询 | 候选正目标覆盖 |', '|---|---:|---:|---:|---:|']
    features=read(directory/'feature_training.json')
    for r in features['statistics']:
        lines.append(f"| {r['snapshot']} | {r['seed']} | {r['queries']} | {r['usable']} | {r['target_coverage']:.2%} |")
    prefix=read(directory/'prefix_training.json');residual=read(directory/'residual_training.json')
    lines+=['',f"本机实际前缀训练{prefix['elapsed_seconds']:.2f}秒，残差训练{residual['elapsed_seconds']:.2f}秒，"
            f"特征构造{features['elapsed_seconds']:.2f}秒。不是Agent线上延迟或跨环境性能提升。",'',
            f"独立审计从raw train重建两个图/监督及{audit['prefix_training_samples_rebuilt']}前缀训练样本，"
            f"核验{audit['feature_query_replays']}特征/池预测重放、8550开发评分和推理、19配置选择、"
            '原180输入/目标、2700正式独立评分、2160新排序重放、540原基线及540旧控制、固定100候选和SHA、配对统计全部通过。', '',
            '审计评分/原始时间边界/池排序/残差与概率质量公式及统计独立重算；神经网络结构/CF特征组件复用项目实现。'
            '不称模型算法的完全独立实现证明。**本轮严格Agent任务、LLM请求/token/线上延迟 NOT EVALUATED。**默认未替换。','',
            '[预注册协议](top100_rerank_protocol_20261014.md) · [正式聚合](top100_rerank_20261014_validation.json) · '
            '[独立审计](top100_rerank_20261014_independent_audit.json) · [开发选择](top100_rerank_20261014_selection.json) · '
            '[前缀训练](top100_rerank_20261014_prefix_training.json) · [残差训练](top100_rerank_20261014_residual_training.json) · '
            '[候选覆盖](top100_rerank_20261014_features.json)','',
            '原始用户/标签/候选/权重仅本地 `optimized/artifacts/comparison/top100_rerank_20261014/`。已完成目录禁止覆盖。','',
            '```powershell',"$recPython = 'optimized/AgenticRec/.venv-t20/Scripts/python.exe'",
            "$study = 'optimized/artifacts/comparison/top100_rerank_fresh'",
            *[f'& $recPython -m agenticrec.evaluation.top100_rerank {phase} --directory $study' for phase in ('prepare','prefix','cache_training','residual','freeze','evaluate')],
            '& $recPython optimized/AgenticRec/scripts/audit_top100_rerank.py --directory $study --output "$study/independent_audit.json"','```','',
            '本批结束后不按180人的结果追加参数搜索；后续继续须另设预定验证。']
    with (dest/'top100_rerank_20261014_validation.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='EXPORTED',quality=conclusion,api_requests=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    export(parser.parse_args().directory.resolve())
