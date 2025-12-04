# Multi-Source Meta-Learning với Source Weighting α_s

## Tổng quan

Hệ thống meta-learning multi-source với cơ chế source weighting dựa trên reliability (r_s) và distance (D_s) để tính toán trọng số α_s cho từng source domain. Cơ chế này cho phép model tự động điều chỉnh mức độ tin cậy của từng source domain dựa trên khả năng phân loại và khoảng cách phân phối so với target domain.

## Kiến trúc

### 1. Episode-based Training

Mỗi episode bao gồm:
- **Support sets**: Dữ liệu từ các source domains và target domain để học prototype
- **Query sets**: Dữ liệu từ các source domains và target domain để tính loss

### 2. Prototype Computation

Với mỗi domain, tính prototype (centroid) cho từng class:
- μ_d^0: Prototype cho class normal (0)
- μ_d^1: Prototype cho class anomaly (1)

### 3. Source Weighting Mechanism

#### Reliability Score (r_s)

Đo độ tin cậy của source domain s bằng cách:
1. Dùng prototype của s để phân loại support set của target
2. Tính accuracy: acc_s = số dự đoán đúng / tổng số
3. Tính margin: margin_s = trung bình (d_wrong - d_true)
4. Reliability: r_s = acc_s + β_margin * margin_s

#### Distance Score (D_s)

Đo khoảng cách phân phối giữa source và target:
- D_s = Σ_y (||μ_s^y - μ_T^y||²) cho y ∈ {0, 1}

#### Combined Score

score_s = w_r * r_s - w_D * D_s

#### Source Weights (α_s)

Chuyển scores thành weights bằng softmax:
α_s = softmax(κ * score_s)

## Cách sử dụng

### Training

```bash
python run.py --mode train \
    --source_systems HDFS OpenStack \
    --target_system BGL \
    --parser IBM \
    --epochs 5 \
    --num_episodes 500 \
    --n_support_src 16 \
    --n_query_src 16 \
    --n_support_tgt 16 \
    --n_query_tgt 16 \
    --source_weighting_mode reliability_plus_distance \
    --w_r 1.0 \
    --w_D 1.0 \
    --beta_margin 0.1 \
    --kappa 1.0
```

### Parameters

#### Episode Parameters
- `--num_episodes`: Số lượng episodes để train (default: epochs * 100)
- `--n_support_src`: Số support samples mỗi source domain (default: 16)
- `--n_query_src`: Số query samples mỗi source domain (default: 16)
- `--n_support_tgt`: Số support samples cho target (default: 16)
- `--n_query_tgt`: Số query samples cho target (default: 16)

#### Source Weighting Parameters
- `--source_weighting_mode`: Chế độ weighting
  - `none`: Uniform weights (baseline)
  - `distance_only`: Chỉ dựa trên khoảng cách
  - `reliability_plus_distance`: Full mechanism (default)
- `--w_r`: Trọng số cho reliability component (default: 1.0)
- `--w_D`: Trọng số cho distance component (default: 1.0)
- `--beta_margin`: Trọng số cho margin trong reliability (default: 0.1)
- `--kappa`: Temperature cho softmax (default: 1.0)

## Loss Functions

### L_cls: Classification Loss

Weighted cross-entropy trên query sets:
- Source domains: weighted bởi α_s
- Target domain: weight = 1.0

### L_align: Alignment Loss

Align prototype normal của target với weighted average của source normal prototypes:
L_align = ||μ_T^0 - Σ_s (α_s * μ_s^0) / Σ_s α_s||²

### L_CL: Contrastive Loss

(Placeholder - có thể implement sau)

### Total Loss

L_total = L_cls + λ_align * L_align + λ_CL * L_CL

## Logging

Training sẽ log:
- Loss values (L_total, L_cls, L_align, L_CL) mỗi `log_interval` episodes
- Source weights (α_s) cho từng source domain
- Source scores nếu có

## File Structure

```
core/
├── entities/
│   └── domains.py          # DomainBatch dataclass
├── models/
│   └── mtalog.py            # MTALog model class
└── meta/
    ├── __init__.py
    ├── episode_sampler.py  # Episode sampling logic
    ├── prototypes.py       # Prototype computation
    ├── source_weighting.py # α_s computation
    ├── trainer.py           # MetaTrainer class
    └── config.py            # Configuration
```

## Example Output

```
Episode 10/500 - L_total: 0.5234, L_cls: 0.4123, L_align: 0.0891, L_CL: 0.0000
  Alphas: HDFS: 0.456, OpenStack: 0.544
  Scores: HDFS: 0.123, OpenStack: 0.234
```

## Ablation Studies

Có thể so sánh 3 modes:
1. `none`: Baseline (uniform weights)
2. `distance_only`: Chỉ dựa trên khoảng cách prototype
3. `reliability_plus_distance`: Full mechanism

## References

Cơ chế này được thiết kế dựa trên:
- Prototype-based meta-learning
- Multi-source domain adaptation
- Reliability-based source selection

