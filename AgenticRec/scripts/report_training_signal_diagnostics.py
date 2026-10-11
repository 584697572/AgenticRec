"""Rebuild LG-D02 aggregates from private train-only diagnostic rows."""
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

from agenticrec.data import digest
from agenticrec.training_signal_diagnostics import aggregate_rows, temporal_summary


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'artifacts/runs/lightgcn_training_diagnostics_20261011'


def rebuild(directory):
    original = json.loads((directory/'result.json').read_text(encoding='utf-8'))
    for relative, sha in original['input_sha256'].items():
        if digest(ROOT/relative) != sha:
            raise ValueError('Diagnostic input changed: '+relative)
    for filename, sha in [('samples.json',original['samples_sha256']),
                          ('temporal_rows.json',original['temporal_rows_sha256'])]:
        if digest(directory/filename) != sha:
            raise ValueError('Diagnostic record SHA mismatch')
    samples = json.loads((directory/'samples.json').read_text(encoding='utf-8'))
    times = json.loads((directory/'temporal_rows.json').read_text(encoding='utf-8'))
    if temporal_summary(times) != original['temporal'] or (
            len(samples) != original['train_users'] or len(times) != len(samples) or
            sum(len(s['positive_ids']) for s in samples) != original['sample_pairs']):
        raise ValueError('Temporal/sample summaries cannot be reconstructed')
    checked = 0
    for entry in original['models'].values():
        path = directory/entry['rows_file']
        if digest(path) != entry['rows_sha256']:
            raise ValueError('Model row SHA mismatch')
        rows = json.loads(path.read_text(encoding='utf-8'))
        if aggregate_rows(rows) != entry['summary'] or len(rows) != len(samples):
            raise ValueError('Model aggregates cannot be reconstructed')
        checked += len(rows)
    return original, checked


def mean_std(values):
    return {'mean':statistics.mean(values), 'std':statistics.stdev(values), 'model_seeds':len(values)}


def main():
    original, checked = rebuild(BASE/'diagnosis')
    protected = json.loads((BASE/'preservation_before.json').read_text(encoding='utf-8'))
    for relative, sha in protected.items():
        if digest(ROOT/relative) != sha:
            raise ValueError('Historical evidence changed: '+relative)
    junit = ET.parse(BASE/'full_tests.xml').getroot()
    suites = list(junit) if junit.tag == 'testsuites' else [junit]
    test_counts = {k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    if test_counts['failures'] or test_counts['errors']:
        raise ValueError('Full regression has not passed')
    seed_summaries = {}
    for model in ('gcn_edge_uniform','bpr_edge'):
        values = [original['models'][f'{model}_seed_{seed}']['summary'] for seed in (7,42,2026)]
        seed_summaries[model] = {'users':values[0]['users'],'negative':{},'low':{}}
        for method in ('uniform','median_pool','hardest_pool'):
            seed_summaries[model]['negative'][method] = {
                key:mean_std([v['negative'][method][key] for v in values])
                for key in values[0]['negative'][method] if key != 'pairs'}
        for key in ('high_score_fraction','score_percentile'):
            seed_summaries[model]['low'][key] = mean_std([v['low'][key] for v in values])
        seed_summaries[model]['low']['pair'] = {
            key:mean_std([v['low']['pair'][key] for v in values]) for key in values[0]['low']['pair']}
    public = {**original,'seed_summaries':seed_summaries,
              'verification':{'raw_user_rows_rebuilt':checked,'historical_files_preserved':len(protected),
                              'full_tests':test_counts,'report_script_sha256':digest(Path(__file__))},
              'diagnostic_result_sha256':digest(BASE/'diagnosis/result.json')}
    lines=['# LG-D02：LightGCN训练信号检查（2026-10-11）','',
           '本轮只做训练期诊断；没有重训、没有读取valid/test标签，没有替换默认模型。准确率改进 **NOT EVALUATED / NO VERIFIED GAIN**。', '',
           f"固定MovieLens1M：{original['train_users']}暖用户、{original['candidate_items']}候选、{original['train_positive_edges']}训练正边；每模型{original['sample_pairs']}配对题目，固定样本，6个既有checkpoint。用户宏平均后再给三个模型seed均值/标准差，不是独立测试。",'',
           '## 1. 随机负样本难度','',
           'BPR数据loss不含正则；gradient是loss对分差的导数绝对值，不是参数梯度范数。容易题定义gradient<0.05。困难池中取最大分必然增加题目难度，不证明推荐质量提高。', '',
           '| 模型 | 负样本 | 容易题比例 | 负片超过正片比例 | 平均loss | 平均gradient |',
           '|---|---|---:|---:|---:|---:|']
    for name, summary in seed_summaries.items():
        for method, metrics in summary['negative'].items():
            lines.append(f"| {name} | {method} | {100*metrics['easy_fraction']['mean']:.2f}% | {100*metrics['negative_outranks_positive']['mean']:.2f}% | {metrics['loss']['mean']:.4f} | {metrics['gradient']['mean']:.4f} |")
    lines += ['', '全部seed/活跃度分层保留在JSON；未知负样本仍排除所有训练已评分，未来喜欢与否未被查看。', '',
              '## 2. 明确低分信号','',
              f"原train评分分布：{json.dumps(original['rating_counts_all_train'],sort_keys=True)}；暖用户×候选范围：{json.dumps(original['rating_counts_warm_candidates'],sort_keys=True)}。",'',
              f"有1/2星记录的暖用户{original['users_with_low_ratings']}/{original['train_users']}；至少5条低分的用户{original['users_with_at_least_five_low_ratings']}。3星单列，不按明确讨厌处理。",'',
              '| 模型 | 低分片相对未知片的平均分数百分位 | 低分片达到未知候选第10名分数阈值比例 | 正片未超过低分片比例（严格小于） |',
              '|---|---:|---:|---:|']
    for name,s in seed_summaries.items():
        lines.append(f"| {name} | {100*s['low']['score_percentile']['mean']:.2f}% | {100*s['low']['high_score_fraction']['mean']:.2f}% | {100*s['low']['pair']['negative_outranks_positive']['mean']:.2f}% |")
    t=original['temporal']
    lines += ['', '以上先按有低分的用户平均，各用户的全部候选低分用于分数百分位/阈值，最多8个固定低分用于配对。已评分电影实际会被过滤；阈值是假想插入诊断，不是实际违规推荐率。原规范要求已评分不作未知负采样，行为符合契约；另立显式反馈监督项才是潜在方法变更。', '',
              '## 3. 评分时间是否代表近期兴趣','',
              f"评分时间跨度中位数{t['median_span_days']:.3f}天；跨度≤1天用户{100*t['span_at_most_one_day_fraction']:.2f}%；跨度≥30天用户{100*t['span_at_least_30_days_fraction']:.2f}%；至少一半评分同一天用户{100*t['at_least_half_ratings_same_day_fraction']:.2f}%。",'',
              f"最早/最近四分位可严格分离{t['genre_users']}人（其中跨度≥30天{t['long_span_genre_users']}人）；状态计数{json.dumps(t['status_counts'],sort_keys=True)}。平均类型TV距离{t['mean_genre_tv']:.4f}，同人数大小的固定随机划分参照{t['mean_random_genre_tv']:.4f}，差值{t['mean_genre_tv_minus_random']:.4f}。",'',
              'TV为类型分布差异（0相同、1完全不重叠），每部多类型电影总权重仍为1。随机参照只一次、不是显著性检验。时间同值整组保留；评分日期不是观影日期，不能据此证明兴趣漂移或直接决定时间衰减参数。', '',
              '## 验证及贡献边界','',
              f"训练前提交{original['git_commit']}，Git dirty={original['git_dirty']}。来源/数据/权重SHA完整，私有{checked}条模型用户记录及时间/采样汇总重算一致；{len(protected)}历史文件SHA不变。完整测试{test_counts['tests']}项、失败{test_counts['failures']}、错误{test_counts['errors']}、跳过{test_counts['skipped']}。",'',
              '输入访问清单和身份在JSON。受控诊断使用Python/Arrow白名单读取，禁止全量交互与holdout，不是OS级安全隔离。完整工程测试会使用历史评测fixture，因此不声称全测试进程零holdout访问。API0、下载0、训练0。', '',
              '## 复现','', '```powershell',
              '& .\\AgenticRec\\.venv-t20\\Scripts\\python.exe -m agenticrec.training_signal_diagnostics --plan AgenticRec/configs/lightgcn_training_diagnostics_20261011.json --output artifacts/runs/lightgcn_training_diagnostics_20261011/diagnosis',
              '& .\\AgenticRec\\.venv-t20\\Scripts\\python.exe AgenticRec/scripts/report_training_signal_diagnostics.py',
              '```', '', '需要既有冻结数据和LG-O01六个checkpoint；诊断输出目录必须不存在。报告只接受首次生成或字节完全一致的重建，不覆盖旧报告。新质量确认需要另行登记的有界训练和未参与调参的评价。','']
    outputs = {'reports/rec_baselines/lightgcn_training_diagnostics_20261011.json':json.dumps(public,sort_keys=True,indent=2,allow_nan=False)+'\n',
               'reports/rec_baselines/lightgcn_training_diagnostics_20261011.md':'\n'.join(lines)}
    for relative,text in outputs.items():
        path=ROOT/relative; content=text.encode('utf-8')
        if path.exists():
            if path.read_bytes()!=content: raise FileExistsError('Refuse to replace previous report')
        else: path.write_bytes(content)
    print(json.dumps({'status':'VERIFIED','rows_rebuilt':checked,'preserved':len(protected),'full_tests':test_counts}))


if __name__=='__main__': main()
