"""Verify the committed export matches the actual measured Windows evidence bytes."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import xml.etree.ElementTree as ET

from agenticrec.data import digest
from agenticrec.training import ROOT
from report_upgrade_research import stable_write


PATHS = [
    'AgenticRec/configs/frozen_model_validation_20261011_seed_7.json',
    'AgenticRec/configs/frozen_model_validation_20261011_seed_2026.json',
    'reports/fusion_protocol_20261011.json',
    'reports/fusion_research_20261011.json','reports/fusion_research_20261011.md',
    'reports/fusion_failure_cases_20261011.json','reports/fusion_failure_cases_20261011.md',
    'reports/rec_baselines/frozen_model_validation_20261011.json',
    'reports/rec_baselines/frozen_model_validation_20261011.md',
    'reports/t19_template_cluster_audit_20261011.json',
    'reports/technical_upgrade_20261011.json','reports/technical_upgrade_20261011.md',
]


def main():
    commit=subprocess.check_output(['git','rev-parse','b0547b0'],cwd=ROOT,text=True).strip()
    output=ROOT/'artifacts/runs/upgrade_20261011'
    archive=output/'measured_evidence_export.tar'
    paths=PATHS+['.gitattributes','AgenticRec/tests/regression/test_research_distribution.py']
    if not archive.exists():
        subprocess.run(['git','archive','--format=tar','--output',str(archive),commit,*paths],
                       cwd=ROOT,check=True)
    sha={}
    with tarfile.open(archive) as tar:
        for path in paths:
            measured=(ROOT/path).read_bytes()
            committed=subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
            exported=tar.extractfile(path).read()
            if measured!=committed or measured!=exported:
                raise ValueError('Distribution changed evidence bytes: '+path)
            sha[path]=hashlib.sha256(measured).hexdigest()
    for seed in (7,2026):
        plan_sha=sha[f'AgenticRec/configs/frozen_model_validation_20261011_seed_{seed}.json']
        for model in ('bpr_l0_d32','lightgcn_l1_d64'):
            path=ROOT/f'artifacts/runs/recommendation_search/frozen_model_validation_20261011_seed_{seed}/{model}/result.json'
            result=json.loads(path.read_text(encoding='utf-8'))
            if result['identity']['plan_sha256']!=plan_sha: raise ValueError('Executed training plan differs')
    metadata=json.loads((ROOT/'reports/technical_upgrade_20261011.json').read_text(encoding='utf-8'))
    for path,expected in metadata['file_sha256'].items():
        if sha[path]!=expected: raise ValueError('Public evidence link SHA mismatch')
    before=json.loads((output/'preservation_before.json').read_text(encoding='utf-8'))
    if any(digest(ROOT/path)!=expected for path,expected in before.items()):
        raise ValueError('Historical evidence changed')
    suite=ET.parse(output/'full_distribution_final.xml').getroot().find('testsuite')
    tests={k:int(suite.attrib[k]) for k in ('tests','failures','errors','skipped')}
    if tests!={'tests':295,'failures':0,'errors':0,'skipped':0}:
        raise ValueError('Final full regression failed')
    report={'status':'VERIFIED','date':'2026-10-11','commit':commit,
        'exported_files':len(paths),'file_sha256':sha,'archive_sha256':digest(archive),
        'training_plan_identity_matches':4,'historical_files_unchanged':len(before),
        'final_full_tests':tests,'regression_red_failures':3,
        'final_test_xml_sha256':digest(output/'full_distribution_final.xml'),
        'generator_sha256':digest(Path(__file__)),
        'scope':'Fix publication byte normalization; no metric, model math or historical evidence changes',
        'provenance_limit':'Early Git snapshots normalized new CRLF plans/reports to LF; this commit preserves the actually executed/recorded bytes',
        'remote_api_requests':0}
    stable_write(ROOT/'reports/distribution_integrity_20261011.json',report)
    text='''# 发布字节完整性补充验收（2026-10-11）

最终完整回归 **295/295**，包含3项先失败的Git行尾回归；新锁环境35项结果仍有效。此报告补充technical_upgrade_20261011的292项时点记录，不改历史摘要或实验指标。

发布检查发现：Windows生成的新报告/训练计划是CRLF，Git默认eol=lf曾将新文件存为LF。早期Git工作树显示clean并不代表原始字节与Git完全一致；记录的config/report SHA在新clone会失配。

仅对20261011的新报告和两份seed训练计划使用-text保留测量时的实际字节，同时保留正常空白检查；不扩大到旧报告。旧Git提交不重写，补丁b0547b0恢复实际执行/记录的原字节，数值/方法不变。

已验证：12个证据/配置文件加属性和回归源码共14文件，当前磁盘、固定Git blob、git archive导出三者逐字节一致；4次训练的plan_sha256与导出配置一致；公开技术报告引用的三个报告SHA匹配；480份历史证据未改。完整SHA及导出tar SHA见机器报告，tar仅留本地，不包含数据、权重或密钥。

```powershell
& AgenticRec/.venv-t20/Scripts/python.exe AgenticRec/scripts/verify_research_distribution.py
```

生成器固定验证b0547b0及本轮本地日志，重复运行只允许字节一致结果。新的发布文件也沿用同日保字节规则。NO VERIFIED GAIN等算法结论与原实验保持一致。
'''
    stable_write(ROOT/'reports/distribution_integrity_20261011.md',text)
    print(json.dumps({'status':'VERIFIED','git_export_exact_files':len(paths),
                      'final_tests_passed':295,'old_evidence_unchanged':len(before)}))


if __name__=='__main__': main()
