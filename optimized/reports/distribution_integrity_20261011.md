# 发布字节完整性补充验收（2026-10-11）

最终完整回归 **295/295**，包含3项先失败的Git行尾回归；新锁环境35项结果仍有效。此报告补充technical_upgrade_20261011的292项时点记录，不改历史摘要或实验指标。

发布检查发现：Windows生成的新报告/训练计划是CRLF，Git默认eol=lf曾将新文件存为LF。早期Git工作树显示clean并不代表原始字节与Git完全一致；记录的config/report SHA在新clone会失配。

仅对20261011的新报告和两份seed训练计划使用-text保留测量时的实际字节，同时保留正常空白检查；不扩大到旧报告。旧Git提交不重写，补丁b0547b0恢复实际执行/记录的原字节，数值/方法不变。

已验证：12个证据/配置文件加属性和回归源码共14文件，当前磁盘、固定Git blob、git archive导出三者逐字节一致；4次训练的plan_sha256与导出配置一致；公开技术报告引用的三个报告SHA匹配；480份历史证据未改。完整SHA及导出tar SHA见机器报告，tar仅留本地，不包含数据、权重或密钥。

```powershell
& AgenticRec/.venv-t20/Scripts/python.exe AgenticRec/scripts/verify_research_distribution.py
```

生成器固定验证b0547b0及本轮本地日志，重复运行只允许字节一致结果。新的发布文件也沿用同日保字节规则。NO VERIFIED GAIN等算法结论与原实验保持一致。
