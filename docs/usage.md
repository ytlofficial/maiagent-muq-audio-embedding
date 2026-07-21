# Simai 读谱、分段与六维打分使用说明

本文档说明当前保留的脚本和输出文件如何使用。

## 保留的主要文件

### 脚本

```text
scripts/simai_note_counter.py
scripts/simai_measure_compiler.py
scripts/simai_measure_density.py
scripts/simai_density_segmenter.py
scripts/simai_global_six_dimension_table.py
scripts/simai_segment_scorer.py
scripts/simai_pattern_embedding.py
```

辅助脚本：

```text
scripts/build_chart_db.py
scripts/export_measure_segments.py
scripts/select_chart_dataset.py
scripts/count_large_dividers.py
```

### 输入数据

```text
outputs/simai_measures/index.json
outputs/simai_measures/charts/*.json
```

`outputs/simai_measures/index.json` 是谱面索引。

`outputs/simai_measures/charts/*.json` 是已经按小节导出的谱面 JSON。后续全曲六维、单曲分段、段落打分都从这里读取。

### 最新六维输出

```text
outputs/six_dimension/global_six_dimension_table.json
outputs/six_dimension/global_six_dimension_table.csv
outputs/six_dimension/global_six_dimension_intermediates.json
```

`global_six_dimension_table.json` 是完整全谱面六维表。

`global_six_dimension_table.csv` 是更方便表格查看的版本。

`global_six_dimension_intermediates.json` 包含全局 baseline、每张谱面的中间值、归一化值和分数细节。段落打分器默认读取它作为全局评分尺子。

### 文档

```text
docs/scoring_formulas.md
docs/usage.md
```

`scoring_formulas.md` 记录当前六维分数的具体公式。

`usage.md` 是本文件。

## 全谱面六维重算

运行：

```bash
python3 scripts/simai_global_six_dimension_table.py
```

默认读取：

```text
outputs/simai_measures/index.json
```

默认写出：

```text
outputs/six_dimension/global_six_dimension_table.json
outputs/six_dimension/global_six_dimension_table.csv
outputs/six_dimension/global_six_dimension_intermediates.json
```

如果要先删除旧结果再重算：

```bash
rm outputs/six_dimension/global_six_dimension_table.json \
   outputs/six_dimension/global_six_dimension_table.csv \
   outputs/six_dimension/global_six_dimension_intermediates.json

python3 scripts/simai_global_six_dimension_table.py
```

输出中的 `error_count` 表示无法解析的谱面数量。`errors` 数组会列出无法识别的谱面和错误原因。

## 单曲自动分段

按 `song_id` 读取一张谱：

```bash
python3 scripts/simai_density_segmenter.py \
  --song-id 1105 \
  --chart-kind DX \
  --difficulty-index 5 \
  --output outputs/example_segment.json
```

按标题模糊匹配：

```bash
python3 scripts/simai_density_segmenter.py \
  --title "初音天地開闢神話" \
  --chart-kind DX \
  --difficulty-index 5 \
  --output outputs/example_segment.json
```

强制段落数：

```bash
python3 scripts/simai_density_segmenter.py \
  --song-id 1105 \
  --chart-kind DX \
  --difficulty-index 5 \
  --segments 5 \
  --output outputs/example_segment.json
```

输出内容包括：

```text
segments
summary
density curve
note mix
special note ratios
six-dimension segment scores
segment labels
```

## 指定段落六维打分

手动指定小节范围：

```bash
python3 scripts/simai_segment_scorer.py \
  --song-id 1105 \
  --chart-kind DX \
  --difficulty-index 5 \
  --range 1:32 \
  --range 33:64 \
  --range 65:96 \
  --output outputs/example_segment_scores.json
```

`--range` 是 1-based 闭区间，可以重复传入。

也可以对接自动分段脚本输出的 JSON：

```bash
python3 scripts/simai_segment_scorer.py \
  --song-id 1105 \
  --chart-kind DX \
  --difficulty-index 5 \
  --segments-json outputs/example_segment.json \
  --output outputs/example_segment_scores.json
```

段落打分默认读取：

```text
outputs/six_dimension/global_six_dimension_intermediates.json
```

这保证段落分数和全谱面六维表使用同一套 baseline。

## 四小节规则 embedding

将指定 4 小节转成结构化规则向量：

```bash
python3 scripts/simai_pattern_embedding.py \
  --song-id 1259 \
  --chart-kind DX \
  --difficulty-index 5 \
  --range 4:9 \
  --output outputs/example_pattern_embedding.json
```

也可以直接传入一段 simai：

```bash
python3 scripts/simai_pattern_embedding.py \
  --simai '(120){4}2-4[8:1],,,,{1},{1},{1},{1},{1},E'
```

输出包括：

```text
embedding
feature_names
block_slices
block_weights
nonzero_features
```

当前向量保留 1-8 号键绝对信息，同时加入拓扑关系和 D8 旋转/镜像池化分支。星星/slide 会保留 shape、起点、终点、中继点、同起点多轨、@/?/!、BREAK/EX 等信息，并额外使用软几何轨迹特征，让 `2-4` 与 `2>4` 比 `2-4` 与 `2<4` 更接近。

## 四小节音频分段表

按 `export_segment_chunk_ranges.py` 的四小节 chunk 规则，从 `chartdata-rebuilt/<歌曲>/track.mp3` 切音频，并写入和谱面向量表同 key 的独立 LanceDB 表。如果歌曲目录里只有 `track.ogg`，导入器会先转成缓存 mp3，再进行切片。

```bash
scripts/import_audio_chunks.sh
```

一键脚本默认：

```text
SIMAI_AUDIO_CHARTDATA_ROOT=chartdata-rebuilt
SIMAI_AUDIO_DB_PATH=outputs/lancedb/simai_pattern_chunks
SIMAI_AUDIO_TABLE=simai_audio_chunks
SIMAI_AUDIO_INDEX_TABLE=simai_audio_chunk_index
SIMAI_AUDIO_OUT_DIR=outputs/audio_chunks/simai_audio_chunks
SIMAI_AUDIO_FALLBACK_FILENAME=track.ogg
SIMAI_AUDIO_CONVERTED_SOURCE_DIR=outputs/audio_chunks/converted_sources
SIMAI_AUDIO_MODE=overwrite
SIMAI_AUDIO_REUSE_EXISTING=1
SIMAI_AUDIO_SKIP_MISSING=1
```

也可以直接调用底层脚本，按筛选条件导入一张或一批谱：

```bash
.venv/bin/python scripts/build_segment_chunk_audio_lancedb.py \
  --song-id 32 \
  --chart-kind ST \
  --difficulty-index 5
```

默认写入：

```text
outputs/lancedb/simai_pattern_chunks/simai_audio_chunks
outputs/lancedb/simai_pattern_chunks/simai_audio_chunk_index
outputs/audio_chunks/simai_audio_chunks/<chart_name>/*.mp3
outputs/audio_chunks/converted_sources/<song>/track.mp3
```

音频表的 `key` 与 `simai_pattern_chunks` 一致，例如：

```text
00032_ST_5_Master:1-4
```

谱面、分段和 chunk 的关联字段：

```text
simai_pattern_chunk_index / simai_audio_chunk_index
  chart_id          例如 32:ST:5
  difficulty        Master=5, Re:Master=6
  level             原始显示值，例如 13+
  level_value       数值等级，例如 13.6

simai_segments
  key               <chart_id>:<segment_id>
  chart_id
  segment_id        固定为 0-4
  note / peak / charge / slide / handtrip / tricky
  score_vector      按上述顺序排列的六维数组

simai_pattern_chunks / simai_audio_chunks
  chart_id
  segment_id
  segment_key       对应 simai_segments.key
```

已有数据库可直接增量补齐这些字段，不需要重新计算 512 维向量或重新切音频：

```bash
.venv/bin/python scripts/migrate_chart_segment_metadata_lancedb.py
```

如果只想检查会生成哪些行，不切音频、不写库：

```bash
.venv/bin/python scripts/build_segment_chunk_audio_lancedb.py \
  --title "アイデンティティ" \
  --chart-kind ST \
  --difficulty-index 5 \
  --dry-run
```

默认表里保存音频文件路径、时间范围、大小和 sha1。若要把编码后的音频二进制也写进 LanceDB 行：

```bash
.venv/bin/python scripts/build_segment_chunk_audio_lancedb.py --store-audio-bytes
```

对应的一键脚本写法：

```bash
SIMAI_AUDIO_STORE_BYTES=1 scripts/import_audio_chunks.sh
```

若要缺失 `track.mp3` 时直接失败，而不是跳过：

```bash
SIMAI_AUDIO_SKIP_MISSING=0 scripts/import_audio_chunks.sh
```

如果只想导入原始音频是 `track.ogg` 的歌曲，跳过已有 `track.mp3` 的歌曲：

```bash
scripts/import_ogg_audio_chunks.sh
```

等价于：

```bash
SIMAI_AUDIO_ONLY_SOURCE_FORMAT=ogg scripts/import_audio_chunks.sh
```

底层脚本也可以直接传：

```bash
.venv/bin/python scripts/build_segment_chunk_audio_lancedb.py --only-source-audio-format ogg
```

## 音频 embedding 谱面划分

从 LanceDB 的 1898 张谱面中选择 1600 张，并生成 song-disjoint 的
1000/300/300 train、validation、test 划分：

```bash
.venv/bin/python scripts/split_audio_embedding_charts.py
```

选择时优先排除早期版本且 `level_value < 13` 的 298 张谱面；划分优化器会
同时平衡版本、Master/Re:Master、等级、版本与难度联合分布以及 chunk 总量。
同一个 `song_id` 不会跨越不同数据集。

默认输出：

```text
datasets/audio_embedding_charts_1000_300_300.csv
datasets/audio_embedding_charts_1000_300_300_train.csv
datasets/audio_embedding_charts_1000_300_300_validation.csv
datasets/audio_embedding_charts_1000_300_300_test.csv
datasets/audio_embedding_charts_1000_300_300_excluded.csv
datasets/audio_embedding_charts_1000_300_300_summary.json
```

固定随机种子为 `20260715`，可以通过 `--seed` 或 `--out-prefix` 生成独立实验版本。

## 六维字段说明

最终六维都是 `0-200`：

```text
note
peak
charge
slide
handtrip
tricky
```

主导维度：

```text
dominant_dimension
```

中间值常用字段：

```text
density_note_mean
density_cv
burst_density
slide_ratio
slide_density
charge_ratio
charge_density
handtrip_density
tricky_shortest_time
```

公式细节见：

```text
docs/scoring_formulas.md
```

## 当前关键算法口径

- `note` 不计入 touch 音符。
- `peak` 使用相邻四小节的 tap、hold、星星头数量爆发密度。
- `slide` 使用 slide 密度和 slide 占比的融合分数。
- `charge` 使用 hold + touch_hold 的密度和占比融合分数。
- `handtrip` 使用 tap/hold 位移 + slide 路径位移。
- 当 handtrip 的相邻 tap/hold 时刻任意一端 `BPM > 200` 时，最小间隔边界从 `1/16` 改为 `1/12`。
- `tricky` 使用同键三次 tap 的最短时间窗，并会对同一 `lane + beat` 的重复 tap 去重。

## 测试

运行全部相关轻量测试：

```bash
python3 -m unittest discover -s tests -v
```

## 常见检查命令

查看全谱面表规模和错误数：

```bash
python3 - <<'PY'
import json
d = json.load(open("outputs/six_dimension/global_six_dimension_table.json", encoding="utf-8"))
print(len(d["rows"]), d["error_count"], len(d["errors"]))
PY
```

按标题查看某首歌六维：

```bash
python3 - <<'PY'
import json
d = json.load(open("outputs/six_dimension/global_six_dimension_table.json", encoding="utf-8"))
for row in d["rows"]:
    if "初音天地開闢神話" in row["title"]:
        print(json.dumps(row, ensure_ascii=False, indent=2))
PY
```
