# Simai 官方核心语言机器可读归档

## 1. 文件用途

本归档将 Celeca 在 simai Wiki 中定义的官方核心记法整理为适合程序使用的结构。它可作为：

- Simai/maidata 词法分析器与 Parser 的规范输入；
- `ChartIR` 的结构定义；
- 谱面生成器的输出约束；
- 无理检测器之前的语法与几何验证层；
- Agent Skill 的知识文件；
- Parser 的回归测试集。

归档没有把 Majdata、AstroDX 或其他模拟器的扩展语法冒充成官方 Simai 核心语法。

## 2. 文件列表

| 文件 | 用途 |
|---|---|
| `simai_core_spec.zh-CN.json` | 完整机器可读语义规范、端点规则、归一化规则和错误码 |
| `simai_core.ebnf` | 可用于实现 Parser 的规范化 EBNF |
| `simai_ast.schema.json` | 解析后标准 AST 的 JSON Schema |
| `simai_conformance_cases.json` | 合法、非法和兼容模式警告测试 |
| `README.md` | 中文实现说明和注意事项 |

## 3. 核心时间模型

Simai 谱面由“Note 信息 + 逗号”组成。一个逗号代表当前设定下的一段时间：

```text
每逗号秒数 = 240 / BPM / 分音值
```

例如 `(120){4}` 表示每个逗号推进一个四分音符，即 0.5 秒。

解析器应维护：

```text
current_bpm
current_slot_duration
current_cursor_seconds
first_seconds
```

每个 Slot 的处理顺序：

1. 应用 Slot 前出现的 BPM/分音指令；
2. 在当前 cursor 放置 Note；
3. 处理反引号产生的 +1ms 微偏移；
4. 遇到逗号后推进 current_slot_duration；
5. 遇到 `E` 结束。

## 4. Note 概要

```text
1                 TAP
1b                BREAK TAP
1x                EX TAP
1$ / 1$$          星形 / 旋转星形 TAP

5h[2:1]           HOLD
5bhx[2:1]         BREAK + EX HOLD
3h                疑似 TAP 的极短 HOLD

B1                TOUCH
B7f               带花火的 TOUCH
Ch[4:3]           TOUCH HOLD
Chf[1:2]          带花火的 TOUCH HOLD
Ch                疑似 TOUCH 的极短 TOUCH HOLD
```

`C1`、`C2` 都应在 AST 中归一成 `C`，但 Round-trip 模式应保留原始写法。

## 5. 同时押与微错位

```text
1/8h[2:1]         精确同时发生的 EACH
12                仅由普通、无修饰 TAP 构成的紧凑 EACH
1`2`3/4           1 在原时刻，2 晚 1ms，3/4 再晚 1ms
```

Parser 推荐优先级：

```text
反引号分组 > EACH 的斜杠分组 > 单 Note
```

也就是先按反引号分成多个微时间组，再在组内按 `/` 拆成同时 Note。

## 6. Slide

基本形式：

```text
始点 + 轨道 + 终点 + 时间
1-4[8:3]
```

常见轨道：

```text
-   直线
> < 外周顺/逆时针
^   短外周
v   过中心 V
p q 曲线
s z 闪电
pp qq 大曲线
V   指定中继点的大 V
w   三叉扇形
```

Slide 的起始星到达判定线后，默认等待当前 BPM 的一拍，再开始移动。轨道从开始到结束保持恒速。

### 同始点

```text
1-4[4:3]*-6[8:5]
```

只有第一条写始点，后续轨道省略始点。

### 连结

```text
1-4q7-2[1:2]                    整条共用时间
1-4[2:1]q7[2:1]-2[1:1]         每段分别指定时间
```

一旦选择逐段时间，所有段都必须指定。BREAK 只能写在最后一个 `]` 后，不能让连结中的某一小段单独成为 BREAK。

## 7. Slide 起点特殊写法

```text
1@-5[8:1]       轨道仍存在，但起点星显示为普通 TAP
1?-5[2:1]       无接近星，移动星淡入
1!-5[2:1]       无接近星，移动开始时突然出现
```

起点前的 `b/x` 作用于起始 TAP；最后 `]` 后的 `b` 作用于 Slide 轨道。这两个概念不能混为一谈。

## 8. 端点验证

JSON 规范已经把来源图片中的端点关系转写为机器规则：

- `-`：不能同点或相邻；
- `^`、`v`：不能同点或正对面；
- `s`、`z`、`w`：终点必须正对面；
- `> < p q pp qq`：所有始终点组合均被表格允许；
- `V`：始点到中继点必须相隔两个按钮，并继续使用完整条件矩阵验证中继点与终点。

Parser 可以负责语法，Validator 必须负责这些几何限制。

## 9. 严格、兼容与 Round-trip 模式

### strict_core

只接受网页明确描述的组合。适合数据库清洗、模型训练和生成器输出。

### compatibility

能分词但网页没有完全定义的组合可以进入 AST，同时产生 Warning。适合导入其他模拟器创建的谱面。

### round_trip

未知语法保存为 Extension/Unknown Node，保留源字符串和位置。适合 MajdataEdit 外挂工具，避免打开并保存后破坏原谱。

## 10. 与 Majdata 的关系

本归档是“官方 Simai 核心规范”，不代表 Majdata 的完整方言。项目中建议定义：

```text
official_simai_core
majdata_dialect
unknown_extension
```

先使用本规范解析共同核心，再为 Majdata 单独维护扩展层。不要直接把 Majdata 能读取的所有字符串写进核心规范。

## 11. 重要差异与不确定项

1. 英文页面最后更新时间显示为 2023-07-25。
2. 日文页面在 2026-03-24 补充：PRiSM PLUS 起官方出现中央 C 以外的 TOUCH HOLD；Simai 语法本身原本就允许其他传感器。
3. 网页没有发布正式 EBNF；归档中的 EBNF 是依据示例和描述构建的规范化解释。
4. `maidata.txt` 变量页面针对旧 3simai，不能未经验证就视作 Majdata 文件规范。
5. 同始点 Slide 若各轨道显式设置不同等待时间，网页描述存在语义张力；兼容模式应警告。
6. 未记录的模拟器扩展必须保持未知，不得靠 Agent 猜测语义。

## 12. 推荐项目接入方式

```text
Raw maidata / Simai
        ↓
Dialect detector
        ↓
Core tokenizer
        ↓
Core parser
        ↓
Semantic validator
        ↓
Endpoint validator
        ↓
Normalized ChartIR
        ↓
Majdata extension adapter
```

生成器反向输出时必须执行：

```text
ChartIR → Semantic validation → Simai serialization → Reparse → AST equality
```

只有 Round-trip 检查通过后，才允许生成候选文件。
