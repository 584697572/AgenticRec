# 个人 public 仓库建立报告

日期：2026-10-05（Asia/Shanghai）。用户明确授权在自己的账号下新建 public 仓库。

## Completed

- GitHub 连接器与 Git Credential Manager 对应账号均核实为 `584697572`。
- 创建 [584697572/AgenticRec](https://github.com/584697572/AgenticRec)，HTTP 201，visibility 为 public。
- 为根项目配置 `origin`，首次推送全部 12 个已有提交，`master` 跟踪 `origin/master`。
- 匿名 API 验证 owner、public、默认分支及远程提交一致性。
- 新增根首页，列出已验证范围、上游归属、个人改动、证据、运行入口与当前限制。

## Files Created / Modified

- 新增：`README.md`、`reproduction/publication_20261005.json`、本报告。
- 更新：`STATUS.md`、`DECISIONS.md`。
- 本地 Git 配置：增加根项目 `origin` 和 `master` 跟踪关系。

## Commands / Tests

```powershell
git remote add origin https://github.com/584697572/AgenticRec.git
git push -u origin master
git diff --check
& .\AgenticRec\.venv-dev\Scripts\python.exe -m agenticrec.cli doctor --offline --config AgenticRec/configs/experiment_contract.example.yaml
& .\AgenticRec\.venv-dev\Scripts\python.exe -m agenticrec.cli fixture
& .\AgenticRec\.venv-dev\Scripts\python.exe -m pytest AgenticRec/tests/unit/test_config.py -q
```

另执行 Git 可达历史 blob 扫描、`.env` 忽略检查、首页相对链接检查、许可证字节比对，以及 GitHub 匿名 GET 核验。创建接口按 [GitHub 官方 REST 文档](https://docs.github.com/en/rest/repos/repos#create-a-repository-for-the-authenticated-user) 调用；凭据仅在本机进程内使用，没有写入仓库 URL 或日志。

## Verification Results

- 创建、首次推送及匿名访问：PASS。首次发布 commit 为 `9ab928d5c0fcfe58679d6c9147381a30f248911d`，与远程 `master` 完全一致。
- 发布前检查：12 个提交、251 个唯一 blob，共 1,244,294 bytes；最大 blob 为 90,216 bytes。未发现当前真实 Key、被忽略的数据/模型目录或用户主目录路径；仅有一处自造无效 URL 测试 fixture 命中凭据格式，已人工核对。
- `.env`：忽略且未跟踪。原数据、checkpoint、模型缓存、虚拟环境、私密日志未进入发布内容。
- README 离线命令：doctor PASS、fixture PASS、15 项测试通过；本轮 LLM 请求 0。
- README 14 个本地链接均存在；保留的 `LICENSE.txt` 与固定上游字节一致；执行规范 SHA256 不变。
- 历史检查是对当前 Key 及选定格式/路径规则的检查，不宣称能穷尽所有敏感信息。可公开核验摘要见 [publication_20261005.json](../reproduction/publication_20261005.json)，它锚定首次推送；本次首页和发布记录在随后单独提交。

## Findings

根工作区此前没有配置远程；Microsoft RecAI 是独立、被根项目忽略的 checkout。新的个人仓库保存项目贡献与复现记录，上游仍使用原 Microsoft URL 和固定 commit。GitHub 在首次推送后将默认分支设为现有的 `master`。

## Blockers

本次仓库创建与首次推送：None。原项目 T03/T04 的资源语义、许可与后续评测限制保持原状态。

## Deviations from Spec

None。公开推送基于用户本轮明确授权；没有修改执行规范、上游源码或实验协议。

## Current Project State

个人 public 仓库已建立并接收原有提交。T00/T01/T02/T05 DONE；T03 IN_PROGRESS；T04 整体 BLOCKED，但真实单轮 VERIFIED；T06—T24 TODO。此次公开首页不是 T20 最终 README/全项目复现验收。

## Next Tasks

按 `TASKS.yaml` 继续 T03 的原资源来源、权威 ID 对应和许可核验，以及 T04 的多轮与原评测；T06 在 T03/T05 依赖满足后推进，之后依次进行指标验证、推荐基线和算法改进。
