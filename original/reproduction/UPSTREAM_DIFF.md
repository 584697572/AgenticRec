# 原版与兼容环境差异

2026-10-05 新增差异（不改原仓库或兼容副本 Python）：受限 App 测试层以 stdio 将真实 HTTP 与加载原资源的无真实 Key 进程分开；原 SDK/原方法继续执行，限制 retry/max_tokens/thinking；原 app launch 被抑制，仅调用其实际回调，明确 ui_served=false。ReDial 预处理单独把本地表副本的整数年份转成该年 1 月 1 日供原 movie_map 比较，保留带符号日期差与 set 选择逻辑；测试 seed=42，canonical 作者子集不明。先失败回归后适配，详见 DECISIONS ADR-008/009 和 reports/progress_20261005.md。现代 T05 包是新增实现，与 legacy 分开，尚无算法改进。

固定上游：Microsoft RecAI，commit `0959ecb05b0794748426e73e6efc1b6b35ec433d`。
`RecAI/` 保持 pristine；运行副本位于 `reproduction/.runtime/InteRecAgent-compat/`。
所有 Python 源文件与固定上游逐文件 hash 一致，未改变 planner、toolbox、buffer、prompt、推荐模型或评测指标。

## 依赖兼容 patch

准确 diff：`legacy_compat.patch`；输入清单：`requirements.legacy.in`；实际解析版本：`requirements.legacy.lock.txt`。
patch已通过固定upstream上的`git apply --check`；原requirements无末尾换行，导出时显式保留Git的no-newline标记并写LF，避免Windows换行导致不可应用。早先格式检查失败日志保留。

| 变化 | 实际触发原因 | 验证与限制 |
|---|---|---|
| 独立 Python 3.9.24；Torch 1.13.1+cpu | 系统只有3.13；原要求支持1.12.1至1.13.1 | 不修改系统Python/CUDA；CPU结果不能当GPU性能 |
| greenlet 2.0.2 | 自动解析的3.2.5要求MSVC，构建失败 | pip check及重建通过 |
| 不安装 chromadb 0.4.7 | chroma-hnswlib 0.7.2要求MSVC，构建失败 | 原生zero-demo路径通过导入；dynamic demo NOT VERIFIED |
| huggingface-hub 0.16.4、transformers 4.33.2、tokenizers 0.13.3 | 实际导入HfFolder/cached_download兼容失败 | 保留原SentenceTransformers 2.2.2和原gte-base模型 |
| accelerate 0.23.0 | 0.32.1调用Hub不存在的split_torch_state_dict_into_shards，Bert加载失败 | 保存native_tools_initial失败日志后固定 |
| UniRec固定0.0.1a4 | 原要求是>=0.0.1a4，固定允许版本以便重建 | 不宣称是checkpoint训练时的库版本 |

OpenAI SDK实际版本见environment_legacy.json；未做全局降级。原OpenAICall使用真实SDK和本地MockTransport的序列化/响应解析检查已通过，真实服务连接NOT VERIFIED。
参考核验使用[OpenAI Docs的Chat Completions接口说明](https://developers.openai.com/api/reference/cli/resources/chat/subresources/completions)，没有迁移到其他API或修改模型配置。

## 测试适配器与模型来源

资源只从固定README提供的公开ZIP读取movie成员，完整矩阵与checkpoint逐文件CRC/SHA核验；gte-base使用revision `c078288308d8dee004ab72c6191778064285ec0c`。
原始data/model与环境留在本地忽略目录，不入Git、不再分发；预制包独立再分发许可和checkpoint的权威标题映射仍NOT VERIFIED。

`native_tool_smoke.py`直接导入全部原模块；它不是AST fixture。
`--agent-replay`只替换LLM边界；`--app-startup`使用原app.py、原工具和真实SDK，本地HTTP响应明确标注fixture。若运行此模式，仅复用已经执行原构造器的同一Gallery对象，并将原Gradio启动限制为loopback/非阻塞以便自动验证和关闭。它不证明真实LLM任务成功。

早先工具测试中两个错误来自测试适配器：假设`Toy Stroy`一定匹配Toy Story，以及把零维NumPy数组当作可hash字符串。这些失败已保留日志，修的是测试契约，没有改原fuzzy_match。
原CandidateBuffer遗漏末尾ID、重复tool覆盖、训练模式dropout等问题按后续任务保留，未混入本轮兼容patch。
