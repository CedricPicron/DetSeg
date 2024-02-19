# DetSeg

Research done on object detection and segmentation during my PhD. Enjoy!

## :dvd: Main results
Results on the 2017 COCO validation set. The inference FLOPs and FPS are measured on the first 100 images of the 2017 COCO validation set using an NVIDIA GeForce RTX 3060 Ti GPU.

### Object Detection (FQDet)

| Backbone |   Head   | Epochs |  AP  | Params | GFLOPs |  FPS  | Script | Log | Checkpoint |
|   :-:    |   :-:    |  :-:   | :-:  |  :-:   |  :-:   |  :-:  |  :-:   | :-: |    :-:     |
| R50+FPN  |  FQDet   |   12   | **43.3** | 33.9 M |  99.0  |  20.9 | [script](scripts/r50_fpn_fqdet_12e.sh) | [log](outputs/r50_fpn_fqdet_12e/log.txt) | [checkpoint](https://drive.google.com/drive/folders/1_MYDpsh__lHkAs7XfBLazZijnKyaZB0Q?usp=sharing) |
| R50+TPN  |  FQDet   |   12   | **45.5** | 42.2 M |  107.8  |  13.6 | [script](scripts/r50_tpn_fqdet_12e.sh) | [log](outputs/r50_tpn_fqdet_12e/log.txt) | [checkpoint](https://drive.google.com/drive/folders/1_MYDpsh__lHkAs7XfBLazZijnKyaZB0Q?usp=sharing) |
| R50+DefEnc-P3  |  FQDet   |   12   | **47.2** | 44.1 M |  234.8  |  9.7 |  [script](scripts/r50_def_fqdet_12e.sh) | [log](outputs/r50_defp3_fqdet_12e/log.txt) | [checkpoint](https://drive.google.com/drive/folders/1_MYDpsh__lHkAs7XfBLazZijnKyaZB0Q?usp=sharing) |

### Object Detection (FQDetV2)

| Backbone |   Head   | Epochs |  AP  | Params | GFLOPs |  FPS  | Script | Log | Checkpoint |
|   :-:    |   :-:    |  :-:   | :-:  |  :-:   |  :-:   |  :-:  |  :-:   | :-: |    :-:     |
| R50+FPN  |  FQDetV2   |   12   | **47.0** | 37.9 M |  117.4  |  17.7 | [script](scripts/r50_fpn_fqdetv2_12e.sh) | [log](outputs/r50_fpn_fqdetv2_12e/log.txt) | [checkpoint](https://drive.google.com/drive/folders/1_MYDpsh__lHkAs7XfBLazZijnKyaZB0Q?usp=sharing) |
| R50+DefEnc-P3  |  FQDetV2   |   12   | **50.8** | 48.1 M |  256.1  |  15.5 |  [script](scripts/r50_def_fqdet_12e.sh) | [log](outputs/r50_defp3_fqdetv2_12e/log.txt) | [checkpoint](https://drive.google.com/drive/folders/1_MYDpsh__lHkAs7XfBLazZijnKyaZB0Q?usp=sharing) |
| R50+DefEnc-P2  |  FQDetV2   |   12   | **51.7** | 48.4 M |  747.0  |  6.8 |  [script](scripts/r50_def_fqdet_12e.sh) | [log](outputs/r50_defp2_fqdet_12e/log.txt) | [checkpoint](https://drive.google.com/drive/folders/1_MYDpsh__lHkAs7XfBLazZijnKyaZB0Q?usp=sharing) |
| SwL+DefEnc-P2  |  FQDetV2   |   12   | **58.2** | 218.7 M |  875.4  |  5.8 |  [script](scripts/r50_def_fqdet_12e.sh) | [log](outputs/swl_defp3_fqdet_12e/log.txt) | [checkpoint](https://drive.google.com/drive/folders/1_MYDpsh__lHkAs7XfBLazZijnKyaZB0Q?usp=sharing) |

### Instance Segmentation (EffSeg)

| Backbone |   Head   | Epochs |  AP  | Params | GFLOPs |  FPS  | Script | Log | Checkpoint |
|   :-:    |   :-:    |  :-:   | :-:  |  :-:   |  :-:   |  :-:  |  :-:   | :-: |    :-:     |
| R50+FPN  |  Mask R-CNN++  |   24   | **42.7** | 40.5 M |  226.7  |  10.4 | [script](scripts/r50_fpn_fqdetv2_12e.sh) | [log](outputs/r50_fpn_fqdetv2_12e/log.txt) | [checkpoint](https://drive.google.com/drive/folders/1_MYDpsh__lHkAs7XfBLazZijnKyaZB0Q?usp=sharing) |
| R50+FPN  |  PointRend++   |   24   | **43.2** | 40.8 M |  296.2  |  6.6 | [script](scripts/r50_fpn_fqdetv2_12e.sh) | [log](outputs/r50_fpn_fqdetv2_12e/log.txt) | [checkpoint](https://drive.google.com/drive/folders/1_MYDpsh__lHkAs7XfBLazZijnKyaZB0Q?usp=sharing) |
| R50+FPN  |  RefineMask++  |   24   | **44.0** | 44.2 M |  455.1  |  6.3 | [script](scripts/r50_fpn_fqdetv2_12e.sh) | [log](outputs/r50_fpn_fqdetv2_12e/log.txt) | [checkpoint](https://drive.google.com/drive/folders/1_MYDpsh__lHkAs7XfBLazZijnKyaZB0Q?usp=sharing) |
| R50+FPN  |  EffSeg (ours) |   24   | **43.7** | 41.8 M |  262.6  |  7.6 | [script](scripts/r50_fpn_fqdetv2_12e.sh) | [log](outputs/r50_fpn_fqdetv2_12e/log.txt) | [checkpoint](https://drive.google.com/drive/folders/1_MYDpsh__lHkAs7XfBLazZijnKyaZB0Q?usp=sharing) |

### Panoptic Segmentation (EffSeg)

| Backbone |   Head   | Epochs |  PQ  | Params | GFLOPs |  FPS  | Script | Log | Checkpoint |
|   :-:    |   :-:    |  :-:   | :-:  |  :-:   |  :-:   |  :-:  |  :-:   | :-: |    :-:     |
| R50+FPN  |  Mask R-CNN++  |   24   | **47.3** | 40.6 M |  218.6  |  10.5 | [script](scripts/r50_fpn_fqdetv2_12e.sh) | [log](outputs/r50_fpn_fqdetv2_12e/log.txt) | [checkpoint](https://drive.google.com/drive/folders/1_MYDpsh__lHkAs7XfBLazZijnKyaZB0Q?usp=sharing) |
| R50+FPN  |  PointRend++   |   24   | **48.4** | 40.9 M |  290.2  |  6.3 | [script](scripts/r50_fpn_fqdetv2_12e.sh) | [log](outputs/r50_fpn_fqdetv2_12e/log.txt) | [checkpoint](https://drive.google.com/drive/folders/1_MYDpsh__lHkAs7XfBLazZijnKyaZB0Q?usp=sharing) |
| R50+FPN  |  RefineMask++  |   24   | **48.6** | 44.2 M |  432.7  |  6.6 | [script](scripts/r50_fpn_fqdetv2_12e.sh) | [log](outputs/r50_fpn_fqdetv2_12e/log.txt) | [checkpoint](https://drive.google.com/drive/folders/1_MYDpsh__lHkAs7XfBLazZijnKyaZB0Q?usp=sharing) |
| R50+FPN  |  EffSeg (ours) |   24   | **48.4** | 41.8 M |  262.8  |  7.4 | [script](scripts/r50_fpn_fqdetv2_12e.sh) | [log](outputs/r50_fpn_fqdetv2_12e/log.txt) | [checkpoint](https://drive.google.com/drive/folders/1_MYDpsh__lHkAs7XfBLazZijnKyaZB0Q?usp=sharing) |

## :page_with_curl: Papers and thesis
- [Trident Pyramid Networks for Object Detection](https://arxiv.org/abs/2110.04004) by [Cédric Picron](https://github.com/CedricPicron) and [Tinne Tuytelaars](https://scholar.google.be/citations?user=EuFF9kUAAAAJ).
- [FQDet: Fast-converging Query-based Detector](https://arxiv.org/abs/2210.02318) by [Cédric Picron](https://github.com/CedricPicron), [Punarjay Chakravarty](https://scholar.google.be/citations?user=AyXW9gYAAAAJ&hl), and [Tinne Tuytelaars](https://scholar.google.be/citations?user=EuFF9kUAAAAJ).
- [EffSeg: Efficient Fine-Grained Instance Segmentation using Structure-Preserving Sparsity](https://arxiv.org/abs/2307.01545) by [Cédric Picron](https://github.com/CedricPicron), and [Tinne Tuytelaars](https://scholar.google.be/citations?user=EuFF9kUAAAAJ).
- Designing High-Performing Networks for Multi-Scale Computer Vision by [Cédric Picron](https://github.com/CedricPicron) (PhD thesis).

## :hammer_and_wrench: Installation
- **Environment**: 
  1. Install the `conda` package and environment management system if not already done.
  2. Execute `source setup_env.sh`.

- **Data preparation**:
  1. Download the desired datasets.
  2. Modify the paths in `setup_data.sh` to point to your installation directories.
  3. Execute `source setup_data.sh`.

## :seedling: Usage
- **Training**: Execute `python main.py` with the desired command-line arguments. Some example training scripts, which were used to obtain the results from above, are found in the `scripts` directory.

- **Evalutation**: Execute `python main.py --eval --eval_task $TASK` with the desired command-line arguments, with $TASK chosen from:
  1. analysis: Analyze the computional cost of the given model.
  2. comparison: Compare the results from two different models.
  3. performance: Compute the model performance on the desired benchmark.
  4. profile: Profile the given model.
  5. tide: Perform [TIDE analysis](https://dbolya.github.io/tide/) of given model.
  6. visualize: Visualize the model predictions.
