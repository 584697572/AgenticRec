"""Bounded upgrade acceptance and real failure cases, derived from local evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from agenticrec.data import digest
from agenticrec.training import ROOT
from report_upgrade_research import stable_write


def read(path):
    data=Path(path).read_bytes()
    return json.loads(data.decode('utf-16' if data.startswith(b'\xff\xfe') else 'utf-8'))


def suite(path):
    entry=ET.parse(path).getroot().find('testsuite')
    return {key:int(entry.attrib[key]) for key in ('tests','failures','errors','skipped')}


def failure_cases():
    run=ROOT/'artifacts/runs/fusion_research/fusion_20261011'
    cases=[]
    for split in ('develop','external'):
        records=read(run/split/'predictions.json')
        if split=='develop':
            audit=set(read(ROOT/'data/eval_private/fusion-development-20261011/user_groups.json')['audit'])
            records=[r for r in records if r['user'] in audit]
        scored=[r for r in records if r['metrics']['rrf'] is not None]
        for category,a,b in [('rrf_gain','rrf','legacy_no_content'),
                             ('rrf_failure','rrf','legacy_no_content'),
                             ('gate_misses_content_benefit','fixed_best','learned_42_full')]:
            eligible=[r for r in scored if category!='gate_misses_content_benefit' or
                      r['gate_decisions']['learned_42_full']['weights'].get('content',0)==0]
            key=lambda r:r['metrics'][a]['ndcg']-r['metrics'][b]['ndcg']
            row=(min if category=='rrf_failure' else max)(eligible,key=key)
            identifier=hashlib.sha256(f"{split}:{row['user']}:{row['state']}".encode()).hexdigest()[:12]
            cases.append({'case_id':identifier,'split':split,'category':category,'state':row['state'],
                'source_predictions_sha256':digest(run/split/'predictions.json'),
                'public_liked_count':len(row['request']['liked_item_ids']),
                'public_disliked_count':len(row['request']['disliked_item_ids']),
                'evaluator_relevant_count':len(row['relevant']),
                'method_a':a,'method_b':b,'ndcg_a':row['metrics'][a]['ndcg'],
                'ndcg_b':row['metrics'][b]['ndcg'],'difference':key(row),
                'gate_content_weight':row['gate_decisions']['learned_42_full']['weights'].get('content',0),
                'selection':'Post-evaluation illustrative extreme; not representative frequency or an adoption decision'})
    return cases


def main():
    base=ROOT/'artifacts/runs/upgrade_20261011'
    fusion=read(ROOT/'reports/fusion_research_20261011.json')
    models=read(ROOT/'reports/rec_baselines/frozen_model_validation_20261011.json')
    final=suite(base/'full_final.xml'); clean=suite(base/'clean_tests.xml')
    if final!={'tests':292,'failures':0,'errors':0,'skipped':0} or clean!={'tests':35,'failures':0,'errors':0,'skipped':0}:
        raise ValueError('Acceptance tests incomplete or failed')
    if read(base/'known_demo.json')!=read(base/'clean_known_demo.json'):
        raise ValueError('Fresh environment inference differs')
    if read(base/'known_trace.json')!=read(base/'clean_known_trace.json'):
        raise ValueError('Fresh environment diagnostic differs')
    if read(base/'clean_fixture_v2.json')['status']!='PASS':
        raise ValueError('Offline fixture failed')
    for relative,sha in read(base/'preservation_before.json').items():
        if digest(ROOT/relative)!=sha: raise ValueError('Historical evidence modified')
    cases=failure_cases()
    stable_write(ROOT/'reports/fusion_failure_cases_20261011.json',{'cases':cases,
        'distinct_query_cases':len({c['case_id'] for c in cases}),
        'boundary':'Six comparisons from four distinct extreme queries, not six independent users; public report omits raw user IDs, targets, and catalog data.',
        'new_requests':0})
    case_lines=['# 融合失败与互补案例（2026-10-11）','',
        '从正式原始预测选出的6项对照、4个不同查询，仅作解释；不是代表性频率，不用于选参，不当6个独立用户。case_id可由私密行的split/user/state复算；不发布原用户、目标电影列表或数据包。', '',
        '| Case | 数据 | 类别 | 状态 | 方法A NDCG | 方法B NDCG | 门控内容权重 |',
        '|---|---|---|---|---:|---:|---:|']
    case_lines += [f"| {c['case_id']} | {c['split']} | {c['category']} | {c['state']} | {c['method_a']} {c['ndcg_a']:.6f} | {c['method_b']} {c['ndcg_b']:.6f} | {c['gate_content_weight']} |" for c in cases]
    case_lines += ['', 'RRF能找回内容互补，也有真实退步用户。门控禁用内容会错过固定融合能命中的目标；该例支持“有漏选”现象，不证明类别不平衡是唯一因果。来源分数、名次、重叠、截断及最终输出全部保留于本地source_records.jsonl。', '']
    stable_write(ROOT/'reports/fusion_failure_cases_20261011.md','\n'.join(case_lines))
    rrf=fusion['external']['rrf']; strong=fusion['external']['legacy_no_content']
    report={'date':'2026-10-11','scope_status':'DONE: bounded preregistered research, evaluation and opt-in integration',
        'algorithm_gate_conclusion':'NO VERIFIED GAIN','gcn_vs_strong_bpr_conclusion':'NO VERIFIED GAIN',
        'full_regression':final,'fresh_environment_tests':clean,'first_regression':suite(base/'full.xml'),
        'first_failure':'Legacy U1 worker exceeded 90s boundary; isolated bridge and final full run passed; root cause NOT VERIFIED',
        'fresh_environment':'New venv, exact lock, offline cache install; current checkout with existing frozen data/checkpoints',
        'fresh_inference_identical':True,'clean_checkout_full_retraining':'Prior T20 evidence retained; this round did not repeat a new full checkout training',
        'new_api_requests':0,'new_model_downloads':0,'official_external_zip_bytes':4924029,
        'new_training_runs':4,'new_training_actual_epochs':sum(models['runs'][f'{k}_{s}']['actual_epochs']
            for k in ('bpr','lightgcn') for s in (7,2026)),
        'checkpoints_validated':6,'preserved_historical_files':480,
        'raw_prediction_metric_checks':fusion['verification']['raw_metrics_rebuilt_exact']+models['verification']['raw_user_predictions_rebuilt_exact'],
        'secondary_external_rrf':{'ndcg':rrf['ndcg'],'baseline_ndcg':strong['ndcg'],
            'relative_observed_difference':rrf['ndcg']/strong['ndcg']-1,
            'hit':rrf['hit'],'baseline_hit':strong['hit'],
            'paired_ndcg':fusion['paired']['external']['rrf_minus_legacy_no_content']['ndcg']},
        'statistics_limit':'95% pairwise intervals unadjusted for multiple secondary comparisons; exploratory and domain-limited',
        'method_execution_provenance':{'development_commit':'f02b7c9','external_base_commit':'e1cb622',
            'primary_source_sha256':fusion['verification']['primary_method_source_sha256'],
            'post_evaluation_integration_commit':'a782793','ranking_math_changed_by_integration':False},
        'conditional_frontier_review':'DONE: defer extra SASRec/RL/industrial foundation models; current sparse outcome coverage and CPU scope do not support justified comparative claims',
        'stable_adoption':'Additional opt-in modules only; original CLI defaults unchanged; learned gate experimental',
        'file_sha256':{p:digest(ROOT/p) for p in ('reports/fusion_research_20261011.json',
             'reports/rec_baselines/frozen_model_validation_20261011.json','reports/t19_template_cluster_audit_20261011.json')},
        'test_log_sha256':{p:digest(base/p) for p in ('full.xml','full_final.xml','clean_tests.xml')},
        'source_commit_before_report':subprocess.check_output(['git','rev-parse','a782793'],cwd=ROOT,text=True).strip(),
        'generator_sha256':digest(Path(__file__))}
    stable_write(ROOT/'reports/technical_upgrade_20261011.json',report)
    lines=['# 深度技术升级验收（2026-10-11）','',
        '**完成范围：预登记的一条融合研究、强基线三seed、控制/消融/失败分析及可选推理接入。学习门控和GCN相对强BPR均为 NO VERIFIED GAIN。** 原论文T04阻塞不变，不声称整个项目或论文复现全部完成。', '',
        '## 问题、方法与结论','',
        '旧no_content同时改变召回和匿名排序，是组合消融。固定旧ranker移除内容、固定旧recall移除genre负惩罚后，单独负惩罚修复没有稳定收益。匹配RRF无内容控制另作后补诊断；新内容源有互补，但受到排序目标和数据域影响。100/路、200并集不变；实际CF前10丢失为0，不能将收益归因为候选保护。', '',
        '实现9个公开特征的logistic门控、标准化、四个正则×三seed、固定权重、条件规则、等权RRF、无内容强对照。门控训练标签105正/673负；三个seed开发审计全部291次禁用内容，外部153次中只有3次启用，但最终指标仍与无内容相同。类别不平衡与收益标签/阈值是可能解释，未被单独因果消融证实。去源质量与去保护无差异，保留负结果，不宣传每个组件有效。', '',
        '## 真实指标及统计范围','',
        f"外部结构化实验：旧无内容 NDCG@10 {strong['ndcg']:.6f} → RRF {rrf['ndcg']:.6f}，观测相对差异 {(rrf['ndcg']/strong['ndcg']-1)*100:.2f}%；Hit@10 {strong['hit']*100:.2f}% → {rrf['hit']*100:.2f}%。用户配对差值95%区间 {report['secondary_external_rrf']['paired_ndcg']['ci95']}。", '',
        '只有51个相关暖用户，694无相关用户另计；三个状态先在用户内平均。外部数据/协议及方法先冻结，未读external valid调参；不保证两个MovieLens发布的真实人群完全独立。开发审计RRF对旧无内容的差值区间跨0。以上是多个次要对照的未多重校正区间，不能写跨域稳定提升、线上喜欢概率、CTR或真实LLM任务提升。主学习门控差值0。', '',
        '固定模型三seed：BPR test NDCG 0.228919±0.003741，GCN 0.227695±0.000859；GCN−BPR用户区间跨0。valid分别0.145874/0.147595。test为 Previously Exposed Test，不能按结果改选模。参数282304/564608，维度32/64；训练/推理时长不是隔离速度比较。新四训练109实际epoch，六权重验证重载valid完全一致，原数据/图/采样数学不改。', '',
        '旧Agent真LLM结果不重跑、不重写：O/A同接口Strict Success均81.33%，请求190→99。模板聚类补审计实际69组，O−无内容仍负；F/U1解析混杂、replan0的边界保留。不把新组件分数归因到LLM策略。', '',
        '## 技术结构与发布边界','',
        '```mermaid','flowchart LR','  State[公开偏好状态] --> Router[既有路由与严格工具契约]',
        '  Router --> Gate[召回前硬约束与授权校验]','  Gate --> Sources[训练期CF / 热门 / 正向内容]',
        '  Sources --> Fusion[显式选择RRF / 无内容 / 条件 / 实验门控]',
        '  Fusion --> Rank[匿名融合排序 / 已授权真实模型重排]',
        '  Rank --> Final[最终硬校验与来源证据]','```','',
        'FusionPipeline兼容既有Agent依赖注入和多轮反馈。工厂接口修复先有失败回归，未改变已评测排序。新增validated profile严格匹配公开记录、checkpoint/训练ID/源码SHA及Python/PyArrow/Torch；padding0与未授权历史被拒绝。候选全链路诊断只写ignored artifacts，拒绝覆盖/越界。原CLI默认模型与排序保持原样，学习门控不自动上线。', '',
        '## 验证、失败与成本','',
        '完整292/292；新独立锁环境35/35，16依赖包离线安装，pip check通过，真实GCN推理及诊断与原环境完全一致；匿名BPR推理也通过。42024项融合原始预测指标与8610条模型用户预测重算，共50634项记录检查；480份历史证据SHA不变。报告重复重建字节一致。', '',
        '初轮完整281通过/1失败（U1约97秒），单独bridge14.26秒成功，最终完整292通过；90秒超时边界未放宽，根因NOT VERIFIED，日志全保留。数据首版空相关计数、模型报告相对路径错误均先复现再修，首版/失败产物不覆盖。集成测试夹具最初episode命名不合法后修正；CLI辅助演示误用未定义包入口、初次uv默认用户缓存无权限的失败也保留，改用既定cli入口和项目缓存。', '',
        '新增真实LLM请求0、权重下载0；仅官方100K ZIP 4,924,029bytes和README。正式融合开发203.57秒、外部9.69秒，均本机组件批次时间，不是在线延迟。Windows/CPU、Python3.11.14、Torch2.0.1+cpu、PyArrow21.0.0；GPU/Linux、非editable安装、新完整checkout重训本轮NOT VERIFIED。原T20干净源码重建训练证据保留。', '',
        '## 贡献、近期研究与剩余优先项','',
        '微软上游提供推荐工具/规划/记忆/反思；本轮新增贡献是候选诊断、公开可靠性特征和学习门控、目标一致融合、控制实验、三seed验证、外部冻结协议、用户/模板聚类和可重算证据。RRF与logistic方法已有先例，不称算法首创。', '',
        'TAIRA、TorchEasyRec、OpenOneRec、OneSearch及HSTU等一手核查见research_landscape_2026.md；借鉴源可靠性、训练服务一致、指令遵循/任务质量分离。没有本机复现其工业模型，也没有SOTA横比。P3条件审查决定本轮不继续SASRec/RL/多Agent/大型权重：稀疏目标、小外部样本和CPU预算无法支持合理比较，不为技术名称扩项目。', '',
        '下一优先项：先登记更大且尚未暴露的跨域评价与负反馈任务，才能验证RRF泛化或重新设计门控收益标签/校准。不得继续对本次external test调参。暖用户模型目前仍是静态训练画像，没有证明学习到会话兴趣；原ID/矩阵许可问题继续BLOCKED。', '',
        '## 可执行复现入口','',
        '沿用AgenticRec/docs/REPRODUCE.md的锁环境、官方数据重建及U1准备；先将官方100K ZIP与README保存在data/raw/ml-100k-upgrade。以下数据冻结命令仅用于私密目录尚不存在的全新checkout；已有冻结目录拒绝覆盖。正式实验要求源码提交且新目录；旧run拒绝覆盖。', '',
        '```powershell',
        '& AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.research_protocol development --output data/eval_private/fusion-development-20261011',
        '& AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.research_protocol external --output data/eval_private/fusion-external-100k-20261011-v2',
        '& AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.fusion_experiment develop --run-id <new_id>',
        '& AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.fusion_experiment external --run-id <same_id>',
        '& AgenticRec/.venv-t20/Scripts/python.exe -m agenticrec.fusion_demo --request AgenticRec/configs/fixed_request.example.json --model bpr --profile validated-20261011 --mode rrf',
        '& AgenticRec/.venv-t20/Scripts/python.exe AgenticRec/scripts/report_upgrade_research.py',
        '& AgenticRec/.venv-t20/Scripts/python.exe AgenticRec/scripts/report_frozen_validation.py',
        '& AgenticRec/.venv-t20/Scripts/python.exe AgenticRec/scripts/report_technical_upgrade.py',
        '```','',
        '固定模型训练/验证命令见rec_baselines/frozen_model_validation_20261011.md。报告生成器绑定本轮本地raw目录，非替代新run通用汇总；现有报告只允许字节一致重建。权重、raw目标和数据包不公开分发。原实现实验commit f02b7c9；外部base e1cb622核心源码SHA相同；后续工厂/公开权重守卫a782793，不改变已评测排序。模型验证路径修复执行时尚未提交，执行源码副本/SHA保留，后来8e76496提交相同修复。', '']
    stable_write(ROOT/'reports/technical_upgrade_20261011.md','\n'.join(lines))
    print(json.dumps({'status':'VERIFIED','full_tests':final['tests'],'fresh_tests':clean['tests'],
        'core_algorithm_conclusion':'NO VERIFIED GAIN','api_requests':0}))


if __name__=='__main__': main()
