# 原版材料请求草稿（2026-10-12）

**NOT SENT — 仅供审阅。** 本轮没有发邮件、Issue、评论或PR。建议渠道为论文公开作者联系方式；已有#75/#106可作为背景，不假定Steam材料适用于电影。不得附API Key、用户数据或未获许可的资源。

## Subject

InteRecAgent movie reproduction: item mappings, resource terms and evaluation artifacts

## Draft

Hello InteRecAgent authors,

We are reproducing the movie-domain implementation at RecAI commit `0959ecb05b0794748426e73e6efc1b6b35ec433d`. We can run the original components, and are keeping our independent MovieLens1M experiments separate from paper reproduction. Could you help clarify three remaining points?

1. **Movie identities and data provenance.** The provided movie catalog has 9,888 entries; its similarity matrix has shape 9,889 × 9,889, while the SASRec checkpoint has 36,255 item rows. Could you share the raw movie ID ↔ checkpoint row mapping, the catalog ID ↔ matrix row mapping, the exact source data release and preprocessing/training settings? A larger checkpoint vocabulary alone is not an error; we need to verify the correspondence. The checkpoint SHA256 is `814430acd5602e666b237c46c92ed48f56f2dce356da399af003a4b68ce1f678`. The data label/link refers to 10M, whereas arXiv v3 Table A1 reports 27,042,493 movie interactions and the notebook reads CSV; please confirm which release and catalog apply.

2. **External resource terms.** Could the authorized resource publisher clarify the terms for the catalog, similarity matrix and pretrained checkpoint in `all_resources.zip`, including local research use, redistribution and commercial use? We understand that the repository's MIT code license does not by itself establish terms for externally supplied resources, and that underlying data permissions may need separate confirmation. We currently distribute scripts and summaries rather than the data or old weights.

3. **Movie evaluation artifacts.** For a specified paper version, could you share the one-turn retrieval/ranking inputs and sample IDs (v3 Table A1 lists 496 MovieLens rows), the sampled 19-negative candidate lists, simulator inputs, generation/filtering logs and random seeds? Please also confirm prompts, demonstration resource, reflection settings, exact LLM/model version and training configuration, including the disabled SASRec positional embeddings. We found the Steam release in issue #75, but have not found the corresponding movie artifacts. The ReDial notebook's 50-row prefix is a different input path and we do not treat it as the paper's movie evaluation set.

We also noticed that the current movie generation notebook assigns `history` twice in cell 37, with the second assignment including the last item later used as `target`. We have not executed this segment or assumed that it was used for the published evaluation. Could you confirm the actual generation script/version used? If restricted files cannot be shared, hashes, sample ID manifests, mapping hashes and sufficient regeneration instructions would help.

Thank you for your work and any clarification you can provide.

## 收到回复后的核验

- 区分作者说明、第三方回复、可下载文件及真实执行证据；每份文件记来源、日期、SHA和适用版本。
- 映射分别校验checkpoint行和矩阵行的原始电影ID；包含padding、重复/缺失、目录子集，不用分数相似代替身份。
- 按准确原数据release核对条款，分别记录本地研究、再分发和商业范围；未获授权的内容不上传。
- 核对样本身份、负候选与所有实验条件；有差异时保留A1，不直接标A2。未经核验不执行提供的脚本或不受信checkpoint。
- 历史LLM是否可调用、重跑预算及指定实验范围需另行核实；回复本身不能作为已跑实验的证据。
