# Data contract

The trainer receives a data root at runtime. The data root is mounted at
`/workspace` in Docker and is never copied into the image.

## Directory layout

```text
DATA_ROOT/
  datasets/
    audio_embedding_charts_1000_300_300_train.csv
    audio_embedding_charts_1000_300_300_validation.csv
    audio_embedding_charts_1000_300_300_test.csv
  outputs/
    lancedb/simai_pattern_chunks/
    audio_chunks/simai_audio_chunks/
```

Names are configurable in `configs/training.example.yaml`.

## Split CSVs

Each CSV has one row per chart and must contain:

| Column | Type | Constraint |
| --- | --- | --- |
| `chart_id` | string | Unique within a split |
| `song_id` | integer | No overlap between train, validation, and test |
| `difficulty` | integer | `5` for Master or `6` for Re:Master |
| `level_value` | float | Numeric level, for example a `+` level uses its normalized decimal value |

Extra columns are allowed and ignored by the trainer. Do not commit populated
split CSVs to this repository.

## LanceDB tables

The database directory must expose these three tables. Additional columns are
allowed.

### `simai_pattern_chunks`

| Column | Type | Constraint |
| --- | --- | --- |
| `key` | string | Unique chunk key |
| `chart_id` | string | Matches a split row |
| `song_id` | int64 | Matches the split row |
| `segment_id` | int64 | Integer from `0` through `4` |
| `segment_key` | string | Foreign key into `simai_segments.key` |
| `vector` | fixed-size float list | Exactly 512 finite values with nonzero norm |

### `simai_audio_chunks`

| Column | Type | Constraint |
| --- | --- | --- |
| `key` | string | Exact one-to-one match with pattern chunk `key` |
| `chart_id` | string | Matches the pattern row |
| `audio_file` | string | Nonempty MP3 chunk path |

Historical absolute paths are supported. The loader relocates the suffix
starting at `outputs/audio_chunks/` beneath the current `DATA_ROOT`.

### `simai_segments`

| Column | Type | Constraint |
| --- | --- | --- |
| `key` | string | Referenced by `segment_key` |
| `chart_id` | string | Owning chart |
| `note` | double | Segment score |
| `peak` | double | Segment score |
| `charge` | double | Segment score |
| `slide` | double | Segment score |
| `handtrip` | double | Segment score |
| `tricky` | double | Segment score |

Each selected pattern key must have exactly one matching audio row and a valid
segment row. `--dry-run` validates these joins, split isolation, vectors, and
audio paths without loading PyTorch or MuQ.
