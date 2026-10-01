# A/B路线调查（T03尚未完成）

2026-10-01；当前保留规范默认的**A1优先调查**，最终路线**NOT VERIFIED**。本轮只建立资源访问及缺失证据；不是A1功能复现。

- GoogleDrive README原链接取得200预览页，title为all_resources.zip。标准公开下载入口转到drive.usercontent，200 HTML显示474M文件，要求病毒扫描确认；本轮未提交该确认或下载完整包。
- RecDrive原分享链接取得200前端SPA，需要JavaScript进一步确认分享状态。当前不声称“已停止分享服务”，那是Issue #110用户报告，非本机确认结果。
- Issue #110 API返回200，用户报告的两个链接问题与本机测量分开记录。证据正文仅留本地忽略目录。
- resources/movie/settings.json及四类原资源目前均缺失；以隔离模块执行实际再现FileNotFoundError。dtype、表头、映射、padding、相似度维度、checkpoint来源和UniRec兼容性未验证。
- 代码MIT已核验；预制资源包的数据许可未核验。MovieLens官方研究许可通过Web读取：须引用；再分发需另许可；商业用途需事先许可，不声称MIT适用数据。

## 后续最小核验

先取得公开原包清单及授权说明；必要时将包放本地忽略的资源目录。仅列zip内容/读取settings与许可，不在含密钥环境加载权重。校验ID、矩阵和原表后，再结合隔离legacy环境检查可信checkpoint。没有包或数据许可时如实记录阻塞；明确不可得则按规范转B，不无限等待。

**本轮不触发B训练或MovieLens准备。** 未获真实LLM预算，所有live/原评测not_run；后续无LLM工具检查可以独立推进。

## 实验可比性

当前没有改变模型、数据、prompt或指标，无法与论文数字比较。以后若走B，应命名InteRecAgent方法级重建，保留U1 upstream_rebuilt来源；U0缺失记not_run，不冒称A2论文实验复现。
