# 执行规范勘误建议

规范文件未修改；SHA256仍为`d88f02c84855234673514519486e74e73c61ffd4e2ec75594020fdf9b8ebb677`。

1. §4结构示例写`LICENSE`，固定源码实际为`LICENSE.txt`。保留真实文件名，版本锁按实际文件取hash；不影响实验。
2. [S23]所指`eval/eval_single_turn.sh`存在，路径无需修正；源码shell含占位目录以及`--demo_dir`与Python的`--demo_dir_or_file`差异。建议把该具体差异补到规范§5.3，强化已有“实际参数以Python为准”的要求；Python使用parse_known_args，后续wrapper须报未知参数。
3. §16提到执行包README_START_HERE.md、AGENTS.md、TASKS.yaml及example config；本工作区起初只有完整规范，clone也无AGENTS.md。按附录生成TASKS.yaml用于调度，不伪称收到完整执行包；不创建推测性的AGENTS覆盖规则。
4. README资源链接在当前网络可读取HTML；Google下载入口返回474M文件确认页。Issue #110不是当前资源失效的充分证据，不自动选择B。需要完整包、授权、ID和模型兼容核验后完成T03。
5. 当前只有Python3.13.7可见，3.9/3.11未发现；后续环境阶段需要合法取得隔离解释器或记录替代方案，不能把规范建议版本当已建立环境。

所有运行/实验指标保持NOT EVALUATED。以上仅修正路径与事实记录，不修改任务依赖或研究路线。
