# VegFormer 操作手册

 **环境准备 → 模型训练 → 模型测试 → 指标评估 → 结果可视化** 的顺序组织。已有训练权重时，可从测试阶段开始。

## 1. 环境准备

### 1.1 登录并激活环境

登录服务器后执行：

```bash
bash
conda activate vegformer
cd /mnt/disk_10T/huxiang/greenearthnet
```

### 1.2 目录说明

| 用途         | 路径                                                       |
| ------------ | ---------------------------------------------------------- |
| 项目目录     | `/mnt/disk_10T/huxiang/greenearthnet`                      |
| 数据集根目录 | `/mnt/disk_10T/huxiang/datasets/greenearthnet/data/mydata` |
| 输出根目录   | `/mnt/disk_10T/huxiang/datasets/greenearthnet`             |

输出根目录下的子目录：

| 子目录           | 保存内容                                             | 后续用途                 |
| ---------------- | ---------------------------------------------------- | ------------------------ |
| `explain_weight` | 验证集样本的 `attn`、`gate`、`saliency` 可解释性权重 | 可解释性可视化           |
| `preds`          | 模型预测结果                                         | 指标评估、空间图和趋势图 |
| `result`         | 最终可视化结果                                       | 结果查看与整理           |
| `score_output`   | 评估指标                                             | 模型效果分析             |

### 1.3 底层包与关键代码

底层包目录介绍：

```text
/home/huxiang/miniconda3/envs/greenearthnet/lib/python3.8/site-packages/earthnet_models_pytorch/
```

| 相对位置                             | 作用                     |
| ------------------------------------ | ------------------------ |
| `data/mydata_v5_lc_cdl.py`           | 数据集加载代码           |
| `model/` 下的 `myformer_explain_gdd` | 模型代码                 |
| `task/workflow.py`                   | 控制是否保存可解释性权重 |

## 2. 模型训练

**输入**：已准备好的数据集、模型配置文件。  
**输出**：训练好的模型权重（checkpoint）。

### 2.1 准备配置与数据

在项目目录中操作，确认数据集已准备完成，并检查配置文件：

```text
model_configs/contextformer/contextformer6M/seed=42.yaml
```

### 2.2 执行训练，参考如下

```bash
python train.py \
  model_configs/contextformer/contextformer6M/seed=42.yaml \
  --data_dir /mnt/disk_10T/huxiang/datasets/greenearthnet/data/mydata/cdl_gdd_ia
```

### 2.3 查看训练权重

权重输出位置（相对于项目目录）：

```text
experiments/greenearthnet/contextformer/contextformer6M/config_seed=42_benson/checkpoints
```

训练结束后，确认实际生成的 checkpoint 路径，并在测试阶段使用对应文件。

## 3. 模型测试与可解释性权重生成

**输入**：模型配置、训练权重、测试数据。  
**输出**：预测结果；按需生成 `attn`、`gate` 和 `saliency` 权重。

### 3.1 设置 attn / gate 保存选项

在实际使用的底层包中打开 `task/workflow.py`，搜索 `save_attn`， **第 384 行**：

```python
save_attn = False
```

- `True`：保存可解释性权重。
- `False`：不保存可解释性权重。

`attn`、`gate` 输出目录约位于 **第 390 行和第 402 行**。启用保存前，先配置独立输出目录。

### 3.2 执行测试，生成预测结果，参考如下

```bash
python test.py \
  model_configs/contextformer/contextformer6M/seed=42.yaml \
  /mnt/disk_10T/huxiang/greenearthnet/checkpoints/cdl/cur/IA/2012-ia.ckpt \
  --track val_chopped \
  --pred_dir /mnt/disk_10T/huxiang/datasets/greenearthnet/preds/predictions/transformer/cur/IA/2012 \
  --data_dir /mnt/disk_10T/huxiang/datasets/greenearthnet/data/mydata/cdl_gdd_ia
```

| 参数 / 位置    | 含义                                 |
| -------------- | ------------------------------------ |
| 第一个位置参数 | 模型配置文件                         |
| 第二个位置参数 | 训练权重文件                         |
| `--track`      | 使用的数据划分，示例为 `val_chopped` |
| `--pred_dir`   | 预测结果输出目录                     |
| `--data_dir`   | 数据集目录                           |

如已启用可解释性权重保存，测试后同时检查 `attn`、`gate` 输出目录。

### 3.3 单独生成 saliency 权重

打开项目目录下的 `saliency_v2.py`，设置输出目录。位置约为 **第 117 行**，可搜索 `out_dir` 定位：

```python
out_dir = Path("/mnt/disk_10T/huxiang/datasets/greenearthnet/explain_weight/norm-0.046-lr-10/2012-IA-t/saliency")
```

执行命令：

```bash
python saliency_v2.py \
  model_configs/contextformer/contextformer6M/seed=42.yaml \
  /mnt/disk_10T/huxiang/greenearthnet/checkpoints/cdl/cur/IA/2012-ia.ckpt \
  --track val_chopped \
  --data_dir /mnt/disk_10T/huxiang/datasets/greenearthnet/data/mydata/cdl_gdd_ia

```

## 4. 指标评估

**输入**：真实值目录、预测结果目录。  
**输出**：评估指标。

确认真实值与预测值属于同一次实验后执行：

```bash
python eval.py \
  /mnt/disk_10T/huxiang/datasets/greenearthnet/data/mydata/cdl_gdd_ia/val_chopped \
  /mnt/disk_10T/huxiang/datasets/greenearthnet/preds/predictions/transformer/cur/IA/2020 \
  /mnt/disk_10T/huxiang/datasets/greenearthnet/score_output/transformer/cur/IA/2020

```

三个位置参数依次是：**真实值目录、预测结果目录、指标输出目录**。

## 5. 结果可视化

可视化代码位于项目目录下的 `other/display`。运行前，在对应脚本中修改输入与输出目录，并确认所需预测结果或可解释性权重已生成。

### 5.1 脚本速查

| 子目录                 | 功能                       | 脚本 / 操作                                                  |
| ---------------------- | -------------------------- | ------------------------------------------------------------ |
| `diff`                 | 预测值与真实值的差值可视化 | 运行前切换到 `mmst` 环境                                     |
| `explainable/attn`     | 单个样本的 attn 可视化     | `attn_v3_cur.py`                                             |
| `explainable/attn`     | 整个区域的 attn 可视化     | 先运行 `get_attn_csv_cur.py`，再运行 `attn_county_pixel_cur.py` |
| `explainable/gate`     | 单个样本的 gate 可视化     | `gate_v2_cur.py`                                             |
| `explainable/gate`     | 整个区域的 gate 可视化     | `gate_county_lc_cur.py`                                      |
| `explainable/saliency` | saliency 可视化            | 先修改输入与输出目录                                         |
| `map`                  | 单个样本的时空图           | `get_map_12_cur.py`                                          |
| `trend`                | 单个样本的趋势图           | `get_pred_trend_12_cur.py`                                   |

以下各节给出路径配置示例。**示例来自不同实验，不能直接混合作为同一实验的配置。**

### 5.2 attn：单个样本

脚本：`explainable/attn/attn_v3_cur.py`。

依次配置 **attn 权重目录、图像输出目录、真实值目录**：

```python
ATTN_DIR = Path("/mnt/disk_10T/huxiang/datasets/greenearthnet/explain_weight/1km/2012/attn_maps")
OUT_ROOT = Path("/mnt/disk_10T/huxiang/datasets/greenearthnet/result/1km/2012/attn")
NC_ROOT = Path("/mnt/disk_10T/huxiang/datasets/greenearthnet/data/mydata/1km/mini/2012/val_chopped")

```

### 5.3 attn：整个区域

按以下顺序执行两个阶段。

**第一阶段：运行 `get_attn_csv_cur.py`。**

配置 **可解释性权重根目录、真实值目录、第一阶段输出目录**：

```python
EXPLAIN_ROOT = Path(
    "/mnt/disk_10T/huxiang/datasets/greenearthnet/explain_weight/corn_belt_norm_0.046"
)

REAL_ROOT = Path(
    "/mnt/disk_10T/huxiang/datasets/greenearthnet/data/mydata/corn/test"
)

OUT_ROOT = Path(
    "/mnt/disk_10T/huxiang/datasets/greenearthnet/result/corn_belt_norm_0.046/attn_county"
)

```

**第二阶段：运行 `attn_county_pixel_cur.py`。**

配置 **真实值目录、第一阶段结果目录、可视化输出目录和县界 Shapefile**。第二阶段需读取第一阶段生成的结果；请确认脚本中对应的读取路径与第一阶段输出一致。

```python
REAL_ROOT = Path(
    "/mnt/disk_10T/huxiang/datasets/greenearthnet/data/mydata/corn/test"
)

OUT_ROOT = Path(
    "/mnt/disk_10T/huxiang/datasets/greenearthnet/result/corn_belt_norm_0.046/attn_county"
)

VIS_ROOT = OUT_ROOT / "vis"

DEFAULT_SHP = Path(
    "/mnt/disk_10T/huxiang/datasets/greenearthnet/data/shp/tl_2012_us_county/tl_2012_us_county.shp"
)

```

其中 `OUT_ROOT` 沿用第一阶段结果目录，`VIS_ROOT` 为其下的 `vis` 子目录。shp文件路径保持不变。

### 5.4 gate：单个样本

脚本：`explainable/gate/gate_v2_cur.py`。

依次配置 **gate 权重目录、真实值目录、图像输出目录**：

```python
GATE_DIR = Path("/mnt/disk_10T/huxiang/datasets/greenearthnet/explain_weight/corn_belt_norm_0.046/2012-IA-s/gate_maps")
REAL_DIR = Path("/mnt/disk_10T/huxiang/datasets/greenearthnet/data/mydata/cdl_gdd/ood-s_chopped/IA")
OUT_ROOT = Path("/mnt/disk_10T/huxiang/datasets/greenearthnet/result/corn_belt_norm_0.046/2012-IA-s/gate")

```

### 5.5 gate：整个区域

脚本：`explainable/gate/gate_county_lc_cur.py`。

配置 **可解释性权重根目录、真实值目录、图像输出目录和县界 Shapefile**：

```python
EXPLAIN_ROOT = Path(
    "/mnt/disk_10T/huxiang/datasets/greenearthnet/explain_weight/corn_belt_norm_0.046"
)

REAL_ROOT = Path(
    "/mnt/disk_10T/huxiang/datasets/greenearthnet/data/mydata/corn/test"
)

GATE_OUTPUT_ROOT = Path("/mnt/disk_10T/huxiang/datasets/greenearthnet/result/corn_belt_norm_0.046/gate_county")

SHP_PATH = Path(
    "/mnt/disk_10T/huxiang/datasets/greenearthnet/data/shp/tl_2012_us_county/tl_2012_us_county.shp"
)

```

shp文件路径保持不变。

### 5.6 saliency 可视化

进入 `explainable/saliency`，在对应脚本中配置：

- saliency 权重所在目录（ **第 201 行**）。
- 图像输出目录（ **第 207 行**）。

### 5.7 空间图、趋势图与其他可视化

- **空间图**：在 `map/get_map_12_cur.py` 中修改输入与输出目录后运行。
- **趋势图**：在 `trend/get_pred_trend_12_cur.py` 中修改输入与输出目录后运行。
- **反事实实验图**：在 `counterfactual` 目录下选择对应脚本并配置路径。
- **差值图**：在 `diff` 目录下配置路径，运行前执行：

```bash
conda activate mmst

```

建议最后生成差值图，减少环境切换。若继续执行其他步骤，需重新激活对应环境。

## 6. 运行前核对

- [ ] 当前环境正确，底层包修改作用于实际运行的环境。
- [ ] 模型配置、checkpoint、数据划分、年份和区域相互对应。
- [ ] 本次实验使用独立输出目录。
- [ ] 需要 attn / gate 时，已启用保存并配置输出路径。
- [ ] 需要 saliency 时，已单独运行 `saliency_v2.py`。
- [ ] 评估与可视化使用本次实验生成的结果。
- [ ] 区域 attn 可视化已按先 CSV、后绘图的顺序执行。
