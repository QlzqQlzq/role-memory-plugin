# 基于知识库的动态人设

这是一个面向 MaiBot 的动态人设插件。它把同一角色的剧本、路线、周目、解包文本或设定笔记整理为本地知识库，并在每轮回复前只召回当前场景相关的原作证据、表达方式与关系状态。

当前版本：`0.3.0`

## 主要功能

- 从多个文本文件自动提取身份、背景、性格、价值、决策、关系、情绪、表达方式和代表台词。
- 支持同一角色的多剧本、多路线和多周目资料，保留来源路径并避免把互斥剧情强行拼成单一经历。
- 使用本地 SQLite 保存原作证据和关系状态，无需额外向量数据库。
- 根据用户消息与回复理由检索相关记忆，通过 `maisaka.replyer.before_request` 动态注入本轮演绎参考。
- 按 MaiBot 官方 `person_id` 区分聊天对象；无法取得时回退到平台与用户 ID。
- 由 MaiBot 决策模型按需调用 `role_memory_update_state`，渐进更新熟悉、信任、亲近、尊重、戒备、紧张和袒露意愿。
- 支持规则复审、风险场景智能复审和每轮智能复审，必要时有限重生成。

## 数据分工

- 固定人设：由 MaiBot 的 `personality` 与 `reply_style` 管理。
- 原作知识：由本插件从导入资料生成，只读，不被聊天内容改写。
- 聊天记忆：继续使用 MaiBot 自带的记忆系统。
- 关系状态：由本插件按聊天对象保存，跨会话延续，不在不同用户之间共享。

## 安装与初始化

插件要求 MaiBot `1.0.12` 及以上版本、MaiBot Plugin SDK `2.7.0` 及以上版本。

安装并加载插件后：

1. 在 WebUI 插件配置中填写 `plugin.character_name`。
2. 启用 `builder.admin_enabled`，并在 `builder.admin_qq_ids` 中填写允许管理角色库的 QQ 号；多个号码用逗号分隔。
3. 把资料放入 MaiBot 数据目录下的：

```text
data/plugins/github.QlzqQlzq.role-memory-plugin/imports/
```

4. 使用配置中的管理员 QQ 私聊机器人，发送 `/角色库初始化`。

支持 `txt`、`md`、`json`、`jsonl`、`csv`、`yaml`、`yml`、`ks`、`scn` 等文本文件，也可以继续按作品、路线或周目放进子目录。初始化在后台执行；处理中再次发送同一指令可查看进度。

初始化完成后，插件会生成并热安装：

```text
character_dossier.json
character_persona.txt
role_memory.sqlite3
```

重新初始化时会备份旧档案，并保留已有关系状态。

## 指令

| 指令 | 权限 | 作用 |
|---|---|---|
| `/角色记忆 <问题>` | 所有人 | 查询与剧情、关系或表达场景相关的原作记忆 |
| `/角色状态` | 所有人 | 查看当前聊天对象的关系状态 |
| `/人设状态` | 所有人 | `/角色状态` 的别名 |
| `/角色库初始化` | 配置的管理员 QQ 私聊 | 构建、备份并热安装当前角色知识库 |

每条指令都会返回可见回执。初始化指令只接受私聊，且管理员列表为空时默认禁用。

## 运行流程

```text
用户消息
  -> 识别会话与聊天对象
  -> 决策模型按需提交关系或情绪事件
  -> 检索当前场景相关的原作证据
  -> 注入动态演绎参考
  -> MaiBot 生成回复
  -> 按配置复审，必要时有限重试
```

## 配置

### `plugin`

- `enabled`：总开关。
- `character_name`：当前角色名称，初始化前必须填写。
- `character_aliases`：角色别名、译名或剧本中的 speaker ID。
- `query_command_enabled`：是否启用角色记忆查询。
- `state_command_enabled`：是否启用关系状态查询。

### `injection`

- `enabled`：是否启用动态提示词注入。
- `search_limit`：每轮最多召回的知识条数。
- `max_prompt_characters`：单轮动态提示的字符上限。
- `include_official_memory`：是否把 MaiBot 官方人物记忆摘要作为辅助背景。
- `state_weights`：各关系维度影响动态提示的权重。

### `state`

- `enabled`：是否维护关系与短期情绪状态。
- `update_from_messages`：是否允许根据聊天事件更新状态。
- `decision_mode`：`planner_tool` 让决策模型按需提交事件，`off` 停用语义状态更新。
- `confidence_threshold`：事件生效的最低置信度。
- `max_delta_per_turn`：单轮单项状态变化上限。
- `event_cooldown_seconds`：同类事件重复生效的冷却时间。
- `update_weights`：各关系维度的数值更新权重。

### `review`

- `enabled`：回复复审总开关。
- `mode = "off"`：关闭复审。
- `mode = "rules"`：只检查提示词泄露、模型身份和明显机械表达。
- `mode = "smart"`：规则检查后，仅在高紧张、长回复或敏感记忆场景调用模型复审。
- `mode = "always"`：规则检查后，每轮调用模型复审。
- `model`：MaiBot 中配置的模型任务名；留空时使用默认模型。
- `max_retries`：单轮最多要求重生成的次数。

### `builder`

- `enabled`、`admin_enabled`：初始化功能与管理员入口开关。
- `admin_qq_ids`：允许初始化的 QQ 号列表。
- `work_name`：作品名称，可留空。
- `model`：构建知识库使用的 MaiBot 模型任务名；留空时使用默认模型。
- `chunk_size`、`chunk_overlap`、`concurrency`：分块与并发参数。
- `max_source_files`、`max_source_characters`、`max_model_calls`：单次构建资源上限。

推荐先使用：

```toml
[state]
enabled = true
decision_mode = "planner_tool"

[review]
enabled = true
mode = "rules"
```

确认动态注入稳定后再尝试 `smart`。`always` 会增加每条回复的模型调用与延迟，不建议默认开启。

## 数据与隐私

- 原始资料、构建缓存、角色档案、SQLite 数据库和关系状态只保存在 MaiBot 分配的插件数据目录，不写入插件源码目录。
- 构建知识库时，原始资料分块会发送给你在 MaiBot 中配置的模型服务；启用智能复审时，本轮角色参考与待审回复也会发送给该模型服务。请根据所用模型服务的隐私政策决定是否导入敏感资料。
- 普通聊天不会改写原作知识；关系状态只保存有限的事件摘要与数值。
- 损坏或版本不兼容的 SQLite 会先改名备份，再创建新数据库，不会静默覆盖原文件。

## 限制

- 一次只启用一个角色，但一个角色可以导入多个剧本与来源分组。
- 检索采用本地关键词与标签评分，侧重部署简单和低资源占用，不等同于语义向量检索。

## 许可证

本项目使用 MIT License。
