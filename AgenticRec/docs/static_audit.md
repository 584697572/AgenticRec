# 固定源码风险复核（T01）

commit：`0959ecb05b0794748426e73e6efc1b6b35ec433d`。2026-10-01。事实、静态推断、隔离测试与真实运行分开记录。

| 项目 | 本轮证据与结论 | 尚未验证 / 对应任务 |
|---|---|---|
| F01 | README70有两个资源入口；Issue #110公开报告可读取。**本机**Google预览200、下载入口200 HTML病毒扫描确认页（474M）；RecDrive200仅SPA。不能将Issue报告当本机测试，也不能把页面200当zip可得 | 原zip清单、授权、完整性 NOT VERIFIED；T03进行中 |
| F02 | requirements3/4/9/10/15/16/17分别固定旧LangChain/Gradio、OpenAI下界、旧sentence-transformers、torch范围、UniRec下界、pandas上界。当前仅可见Python3.13.7、无Conda、py launcher无已注册解释器 | 尚未安装，兼容失败未实测；T02/T05隔离环境，不能生成伪lock |
| F03 | utils/open_ai12/76使用新版客户端；one_turn_eval156仍ChatCompletion.create；user_simulator450仍openai.error.InvalidRequestError | 旧分支运行失败、最小SDK补丁 NOT VERIFIED；只对实际使用分支补丁，不全局降SDK |
| F04 | ToolBox76/80同名dict覆盖，测试直接执行原class，first-query丢失；保留要求失败退出1和观察通过退出0 | **已再现但未修复**；T14依赖T10/T13，本轮不提前实现有序executor |
| F05 | reco_model_tool150—156只传item_seq/item_seq_len/item_id；prefer来自标题，不是user_id | BPR-MF/LightGCN adapter待T10，不直接换checkpoint |
| F06 | preference164—177压分后topk；_rank_by_x215—225按原候选数量取N并压分，不从集合删除 | 静态可推断N覆盖全集时仍包含masked项；真实torch/numpy regression **NOT VERIFIED**，后续T11先fail再fix |
| F07 | app152/218共享buffer/bot；Gradio state只是对话列表，run清共享buffer | 共享可变状态事实确认；并发串扰未执行，T17；不声称生产就绪 |
| F08 | itemcf16/60/62使用npy与直接ID下标；排除第0列 | 真实映射、padding、矩阵维度NOT VERIFIED；T03/T06/T10 |
| F09 | one_turn_eval198—200热门加权抽样；205—211模糊文本hit；218给agent context、219才比较target | 新指标不混用；保留upstream_weighted_pop/upstream_text_hit名称；真实评测NOT EVALUATED |
| F10 | BaseGallery229使用names；46/63初始化gte-base；248只校验必需列存在 | 原表头、dtype、缓存、embedding加载NOT VERIFIED；T03/T02 |

## 补充静态发现

- ToolBox98/128用子串匹配工具名；_parse_llm_output535取action后不校验其名字。新executor需要严格注册表，但不能在本轮偷改基线。
- CandidateBuffer17/125用`range(1,len(gallery))`，可能遗漏真实目录末项，取决于资源是否包含padding行与连续ID。**风险不是已证实原数据bug**。
- CandidateBuffer.push不清similarity；SQL过滤后可能残留旧长度/顺序的相似度。用真实序列测试后再修，避免静态风险变成伪运行事实。
- MapTool62 clear不清tracker，原run431另行清tracks；该行为在隔离fixture实测通过。Map的docstring写随机选取，与实际前N不一致。
- OpenAICall._completion183仍调chat.completions却传prompt；静态接口不一致。当前未调用completion，不提前修。
- OpenAICall外层重试和SDK内层默认重试、usage未知记0、包装call只计一次，无法直接代表逐尝试成本。运行版本尚未锁定，先记风险。
- one_turn_eval未逐样本clear bot DialogueMemory；显式chat_history构造planner输入，但summary路径、其他agent分支和可选profile/critic递归的跨样本影响需专门测。
- 多轮simulator含隐藏target与元数据，共享Conversation只由prompt避免泄露标题；新evaluator应独立可见/隐藏对象，不能将模拟命中称用户满意度。

## 验收范围

四项审计检查通过（由`audit_result.json`生成结果）；重复调用保留要求仍失败（由`duplicate_requirement_result.json`生成结果）。初次审计脚本的prompt常量解析错误保留在未带v2的日志中；不将这些脚本错误当上游bug证据。

上游源码diff为空，无兼容补丁。后续最小改动候选：实际触发的SDK分支（T02）、模型评分adapter（T10）、硬过滤（T11）、有序计划（T14）；均受原依赖约束，当前未实施。
