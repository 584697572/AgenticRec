# 执行规范勘误建议

复核更正：ADR-005 撤回依据维度不同冻结 B 的决定。规范不要求目录覆盖所有 embedding，较大模型可接受目录子集；上轮严格维度断言只证明行数不同。固定 notebook 保存输出与目录出现具体标题映射冲突，训练映射和原权重执行继续核查，T03 IN_PROGRESS。无需修改规范，修正的是本项目对证据的解释。

T03 新增事实（2026-10-01）：原包共享 ID 假设未获验证。完整电影目录 ID=1..9888，矩阵头部 (9889,9889)，checkpoint metadata 与物品 embedding 行数=36255；清单无 ID 映射文件。不是 Google 链接失效，也不能断言所有模型分数错误。要求提供权威共同映射、训练来源和资源许可；取得之前 A1 BLOCKED，采用规范 B 方法级重建，详见 ADR-004。以下为首轮历史建议，规范本体不修改。

规范文件未修改；SHA256仍为`d88f02c84855234673514519486e74e73c61ffd4e2ec75594020fdf9b8ebb677`。

1. §4结构示例写`LICENSE`，固定源码实际为`LICENSE.txt`。保留真实文件名，版本锁按实际文件取hash；不影响实验。
2. [S23]所指`eval/eval_single_turn.sh`存在，路径无需修正；源码shell含占位目录以及`--demo_dir`与Python的`--demo_dir_or_file`差异。建议把该具体差异补到规范§5.3，强化已有“实际参数以Python为准”的要求；Python使用parse_known_args，后续wrapper须报未知参数。
3. §16提到执行包README_START_HERE.md、AGENTS.md、TASKS.yaml及example config；本工作区起初只有完整规范，clone也无AGENTS.md。按附录生成TASKS.yaml用于调度，不伪称收到完整执行包；不创建推测性的AGENTS覆盖规则。
4. README资源链接在当前网络可读取HTML；Google下载入口返回474M文件确认页。Issue #110不是当前资源失效的充分证据，不自动选择B。需要完整包、授权、ID和模型兼容核验后完成T03。
5. 当前只有Python3.13.7可见，3.9/3.11未发现；后续环境阶段需要合法取得隔离解释器或记录替代方案，不能把规范建议版本当已建立环境。

所有运行/实验指标保持NOT EVALUATED。以上仅修正路径与事实记录，不修改任务依赖或研究路线。
