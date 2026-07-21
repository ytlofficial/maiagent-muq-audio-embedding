# Simai 六维打分公式

本文档记录当前最新脚本中的实际算法。全谱面六维表由 `scripts/simai_global_six_dimension_table.py` 生成；指定段落打分由 `scripts/simai_segment_scorer.py` 调用同一套全局 baseline；分段器 `scripts/simai_density_segmenter.py` 会复用这些特征来给段落做标注。

最终六维均为 `0-200`：

```text
note
peak
charge
slide
handtrip
tricky
```

## 1. 全局 baseline

全库先对每张谱面生成一行 raw feature，再计算全局分位数 baseline。

当前 baseline 文件：

```text
outputs/six_dimension/global_six_dimension_intermediates.json
```

当前数值：

```json
{
  "chart_count": 1898,
  "density_note_mean_p95": 8.933223381779388,
  "density_peak_q90_p95": 14.00000175000022,
  "density_cv_p05": 0.2980864447426675,
  "density_cv_p95": 0.6662804953558821,
  "burst_density_p98": 16.653580103267295,
  "slide_density_p98": 1.3876486097853686,
  "slide_ratio_p98": 0.284523739406433,
  "charge_density_p98": 0.929161445370295,
  "charge_ratio_p98": 0.18413174959871575,
  "handtrip_density_p98": 8.707022756823223,
  "tricky_intensity_p98": 7.0000000000000355
}
```

其中：

```text
density_note_mean_p95 = P95(density_note_mean)
density_cv_p05 = P05(density_cv)
density_cv_p95 = P95(density_cv)
burst_density_p98 = P98(burst_density)
slide_density_p98 = P98(slide_density)
slide_ratio_p98 = P98(slide_ratio)
charge_density_p98 = P98(charge_density)
charge_ratio_p98 = P98(charge_ratio)
handtrip_density_p98 = P98(handtrip_density)
tricky_intensity_p98 = P98(1 / tricky_shortest_time)
```

`density_peak_q90_p95` 当前保留在输出中，但 `peak` 最终分数已经只使用 `burst_density`。

## 2. 通用变量

对一整张谱面或一个段落：

```text
T = 有效时长，单位秒
N = 总 note 数 = tap + hold + slide + touch + touch_hold
S = slide 数
C = charge 数 = hold + touch_hold
```

比例与密度：

```text
slide_ratio = S / N
slide_density = S / T

charge_ratio = C / N
charge_density = C / T
```

如果 `N = 0` 或 `T = 0`，对应比例或密度记为 `0`。

## 3. note

`note` 只看非 touch 的整体密度。小节密度序列会过滤时长 `< 0.4s` 的小节，并使用 `non_touch_density`。

```text
x_i = 第 i 个有效小节的 non_touch_density
density_note_mean = mean(x_i)
density_cv = std(x_i) / mean(x_i)
```

若 `mean(x_i) = 0`：

```text
density_cv = 0
```

归一化：

```text
d = clip(density_note_mean / density_note_mean_p95, 0, 1)

v = clip(
  (density_cv - density_cv_p05) / (density_cv_p95 - density_cv_p05),
  0,
  1
)

c = v - 0.5
```

参数：

```text
alpha_note = 1.65
lambda_note = 0.35
denominator = 1 + lambda_note / 2 = 1.175
```

公式：

```text
note_raw = d ^ alpha_note * (1 + lambda_note * c)
note = 200 * clip(note_raw / denominator, 0, 1)
```

含义：

- touch 不进入 `note` 的密度计算。
- `alpha_note = 1.65` 会压低中低密度谱面，使全库中位数落在约 `90-100`。
- `density_cv` 越高，`note` 会略微上升，但影响被 `lambda_note = 0.35` 限制。

## 4. peak

`peak` 只使用相邻四小节的爆发密度。

四小节窗口内计数：

```text
burst_note_count = tap_count + hold_count
```

注意：slide 星星头已经被计入 `tap_count`，所以星星头会进入爆发值。

窗口密度：

```text
burst_density_window = burst_note_count / window_duration_seconds
```

取最高窗口：

```text
burst_density = max(burst_density_window)
```

若段落少于四小节，则用实际段落长度作为窗口。

归一化：

```text
b = clip(burst_density / burst_density_p98, 0, 1)
```

参数：

```text
floor_score = 10
alpha_peak = 1.35
```

公式：

```text
peak = clip(10 + 190 * b ^ 1.35, 0, 200)
```

含义：

- 达到或超过 `burst_density_p98` 时为 `200`。
- 低 burst 不归零，最低理论值为 `10`。
- `1.35` 的幂会让低 burst 惩罚更明显。

## 5. slide

`slide` 同时看 slide 密度和 slide 占比，使用加权调和式融合。

归一化：

```text
x = clip(slide_density / slide_density_p98, 0, 1)
y = clip(slide_ratio / slide_ratio_p98, 0, 1)
```

参数：

```text
density_weight = 0.35
ratio_weight = 0.65
epsilon = 0.001
```

如果任一输入为 `0`：

```text
slide = 0
```

否则：

```text
fused = ((1 + epsilon) * x * y)
        / (ratio_weight * y + density_weight * x + epsilon)

slide = 200 * clip(fused, 0, 1)
```

含义：

- 密度高但占比低，不会虚高。
- 占比高但密度低，也不会虚高。
- 两者同时达到 P98 附近时，接近 `200`。

## 6. charge

`charge` 使用和 `slide` 相同的公式，但输入换成 hold 类音符。

```text
charge_count = hold + touch_hold
charge_ratio = charge_count / total_notes
charge_density = charge_count / duration_seconds
```

归一化：

```text
x = clip(charge_density / charge_density_p98, 0, 1)
y = clip(charge_ratio / charge_ratio_p98, 0, 1)
```

公式：

```text
fused = ((1 + epsilon) * x * y)
        / (ratio_weight * y + density_weight * x + epsilon)

charge = 200 * clip(fused, 0, 1)
```

参数同 slide：

```text
density_weight = 0.35
ratio_weight = 0.65
epsilon = 0.001
```

## 7. handtrip

`handtrip` 由两部分组成：

```text
handtrip_total_distance = tap_hold_total_distance + slide_total_distance
handtrip_density = handtrip_total_distance / duration_seconds
```

### 7.1 tap/hold 位移

只计算 tap 和 hold，不计算 touch，也不计算 slide 星星头。

相邻两个 tap/hold 时刻之间必须大于最小间隔才计入：

```text
默认：
  interval_beats > 1/16

若相邻两端任意一端 BPM > 200：
  interval_beats > 1/12
```

高速 BPM 规则只影响 `handtrip` 的 tap/hold 位移，不影响普通 `tap_distance` 输出。

按键距离使用 8 键环形距离：

```text
distance(a, b) = min(abs(a - b), 8 - abs(a - b))
```

因此：

```text
distance(1, 8) = 1
```

多押到单键时，取最近距离：

```text
distance(1/6, 4) = min(distance(1,4), distance(6,4)) = 2
```

多押到多押时：

- 如果两组按键数量相同，并且存在一组配对使每个键的环形距离都 `<= 1`，则该对位移取这组配对中的最大距离。
- 因此 `1/6 -> 2/5` 和 `7/4 -> 8/3` 记为 `1`。
- 否则使用两组按键环形中心的绝对差。

例如：

```text
center(1/6) = 7.5
center(2/4) = 3
distance(1/6, 2/4) = abs(7.5 - 3) = 4.5
```

### 7.2 slide 位移

每条 slide 计算星星头到星星尾的路径距离。

多段 slide 会把每段距离相加：

```text
slide_total_distance = sum(each_slide_path_distance)
```

### 7.3 handtrip 分数

归一化：

```text
h = clip(handtrip_density / handtrip_density_p98, 0, 1)
```

参数：

```text
alpha_handtrip = 1.7
```

公式：

```text
handtrip = 200 * h ^ 1.7
```

含义：

- 达到或超过 `handtrip_density_p98` 时为 `200`。
- 中低位移密度会被非线性压低，让中位数落在约 `90-100`。
- 高速 BPM 下，过密的 12 分以内 tap/hold 位移会被过滤掉一部分。

## 8. tricky

`tricky` 使用“同一个按键出现 3 次 tap 的最短时间区间”。

只看 tap，不看 hold、touch、slide 尾。slide 星星头会作为 tap 参与。

统计前会对同一 `lane + beat` 去重，避免跨小节边界重复的同一颗音被算成两次。

对每个键 `1-8`：

```text
Delta_min = min(t_i+2 - t_i)
```

不要求三次 tap 连续，例如：

```text
1, , 1, 8, 1
```

也满足同键三次。

如果不存在同键三次：

```text
base_score = 0.2
```

否则：

```text
intensity = 1 / Delta_min
u = clip(intensity / tricky_intensity_p98, 0, 1)
base_score = 200 * u
```

然后经过一层“先陡、后平、再陡”的 Hermite 曲线。

参数：

```text
midpoint_score = 84
left_slope = 1.35
middle_slope = 0.20
right_slope = 1.75
```

定义：

```text
x = clip(base_score / 200, 0, 1)
m = midpoint_score / 200 = 0.42
```

Hermite：

```text
H(t, y0, y1, s0, s1)
  = (2t^3 - 3t^2 + 1)y0
  + (t^3 - 2t^2 + t)s0
  + (-2t^3 + 3t^2)y1
  + (t^3 - t^2)s1
```

最终：

```text
if x <= 0.5:
  tricky = 200 * clip(
    H(x / 0.5, 0, m, left_slope * 0.5, middle_slope * 0.5),
    0,
    1
  )
else:
  tricky = 200 * clip(
    H((x - 0.5) / 0.5, m, 1, middle_slope * 0.5, right_slope * 0.5),
    0,
    1
  )
```

含义：

- 没有同键三次时接近 `0`。
- 中段 tricky 会被压缩。
- 高 tricky 段会重新快速上升。

## 9. dominant dimension

六维分数：

```text
scores = {
  note,
  peak,
  charge,
  slide,
  handtrip,
  tricky
}
```

主导维度：

```text
dominant_dimension = argmax(scores)
```

## 10. 全曲、段落与分段器的关系

全曲六维表：

```text
python3 scripts/simai_global_six_dimension_table.py
```

会遍历全库谱面，重新计算 baseline，并输出：

```text
outputs/six_dimension/global_six_dimension_table.json
outputs/six_dimension/global_six_dimension_table.csv
outputs/six_dimension/global_six_dimension_intermediates.json
```

指定段落打分：

```text
python3 scripts/simai_segment_scorer.py ...
```

默认读取：

```text
outputs/six_dimension/global_six_dimension_intermediates.json
```

因此段落分数和全曲分数在同一套全局尺度上。

分段器：

```text
python3 scripts/simai_density_segmenter.py ...
```

会先按密度曲线切段，再为每段计算 raw feature 和六维分数。分段器内部的某些标签判断也会使用段内相对统计，但输出的 `scores` 字段使用全局 baseline。
