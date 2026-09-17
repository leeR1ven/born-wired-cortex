# 无标点实验独立核验

结论：PASS。删标点文本、完整训练与逐项预测、859个同目标配对比较均通过独立核验。

- 新结果目录：语言规模实验_language_scaling_20260913\results_no_punct_v1
- 耗时：91.14 秒
- 核验环境：Python 3.13.7 / NumPy 2.4.6 / Unicode 15.1.0
- 新核验器SHA-256：980296f0b38238aab40856d6f456dcdae6e9f20d120d731ee1d2ddf68ca0cb3b
- 复用只读参考实现SHA-256：37f239d9a57c9185adaa999ab6b6600087987f5ae0fc7c066e4aee4d3f789e7f
- 核验计数：{"assertions": 4580436, "derivative_splits": 3, "mapped_input_positions": 952696, "source_hashes": 6, "prediction_rows": 82645, "metric_sections": 105, "replayed_stages": 20, "continuations": 40, "paired_source_hashes": 3}

三个分区分别逐字符重建，仅删除Unicode类别P*字符，空白的内容和顺序完全保留。另固定原859个汉字目标，核对删除操作前后的唯一索引映射。五种配置复用完全未改的核心，从出生连续重放全部训练阶段并核对哈希、活动、计数器和所有评分。终态权重逐元素相等、有限非负；载入后续训128字仍逐项一致。

配对比较逐项重算原模型、新模型及仅屏蔽标点输出三组概率/rank。屏蔽对照保留旧模型与旧输入，只将标点输出概率置零并归一化。词表变化同时影响概率归一化和底噪，概率损失差异包含这一影响。

旧VERIFIED.md、旧verify_results.py、三个核心文件及原results_v1全部文件在核验前后SHA-256完全一致。旧实验与其结论完整保留。

## 分区变换

| 分区 | 原字符数 | 新字符数 | 删除P*数 |
|---|---:|---:|---:|
| train | 776,348 | 658,462 | 117,886 |
| validation | 80,384 | 68,410 | 11,974 |
| test | 95,964 | 81,099 | 14,865 |

## 解释边界

新固定随机测试位置与原位置不同，不能把原含标点1024位置准确率直接作为新集配对基线。配对准确率用同一859个汉字目标计算。删除标点同时改变相邻关系及按字符计量的有效上下文，应解释为输入表示干预。

数值核验支持记录与给定实现一致，不证明语义理解、推理、AGI或跨作品泛化。数据仍来自同两部作品；不排除短片段和近重复。NLL/BPC仅对已知字符计算；同分采用固定词表顺序。

## 配对核验结果

{
  "adjacent_linear": {
    "original": {
      "n": 859,
      "known_n": 857,
      "oov_n": 2,
      "top1": 0.1629802095459837,
      "top5": 0.3550640279394645,
      "mean_nll_known": 5.740482417555192,
      "bpc_known": 8.281765516117165
    },
    "no_punct": {
      "n": 859,
      "known_n": 857,
      "oov_n": 2,
      "top1": 0.1839348079161816,
      "top5": 0.3562281722933644,
      "mean_nll_known": 6.0176370359603375,
      "bpc_known": 8.68161510964974
    },
    "output_mask_only": {
      "n": 859,
      "known_n": 857,
      "oov_n": 2,
      "top1": 0.19324796274738068,
      "top5": 0.3841676367869616,
      "mean_nll_known": 5.611869699241865,
      "bpc_known": 8.096216585211277
    },
    "paired_summary": {
      "n": 859,
      "original_correct": 140,
      "no_punct_correct": 158,
      "helped": 23,
      "harmed": 5,
      "no_punct_predicted_types": 179,
      "no_punct_most_predicted": [
        [
          "了",
          87
        ],
        [
          "人",
          49
        ],
        [
          "的",
          49
        ],
        [
          "道",
          42
        ],
        [
          "一",
          39
        ],
        [
          "來",
          38
        ],
        [
          "里",
          23
        ],
        [
          "們",
          23
        ]
      ],
      "masked_most_predicted": [
        [
          " ",
          85
        ],
        [
          "了",
          70
        ],
        [
          "的",
          47
        ],
        [
          "人",
          46
        ],
        [
          "道",
          37
        ],
        [
          "來",
          33
        ],
        [
          "一",
          20
        ],
        [
          "們",
          20
        ]
      ]
    }
  },
  "adjacent_diminishing": {
    "original": {
      "n": 859,
      "known_n": 857,
      "oov_n": 2,
      "top1": 0.1629802095459837,
      "top5": 0.3550640279394645,
      "mean_nll_known": 5.909446982287737,
      "bpc_known": 8.52552985574277
    },
    "no_punct": {
      "n": 859,
      "known_n": 857,
      "oov_n": 2,
      "top1": 0.1839348079161816,
      "top5": 0.3562281722933644,
      "mean_nll_known": 6.224209121878457,
      "bpc_known": 8.979635633589897
    },
    "output_mask_only": {
      "n": 859,
      "known_n": 857,
      "oov_n": 2,
      "top1": 0.19324796274738068,
      "top5": 0.3841676367869616,
      "mean_nll_known": 5.835283846535561,
      "bpc_known": 8.41853506757633
    },
    "paired_summary": {
      "n": 859,
      "original_correct": 140,
      "no_punct_correct": 158,
      "helped": 23,
      "harmed": 5,
      "no_punct_predicted_types": 179,
      "no_punct_most_predicted": [
        [
          "了",
          87
        ],
        [
          "人",
          49
        ],
        [
          "的",
          49
        ],
        [
          "道",
          42
        ],
        [
          "一",
          39
        ],
        [
          "來",
          38
        ],
        [
          "里",
          23
        ],
        [
          "們",
          23
        ]
      ],
      "masked_most_predicted": [
        [
          " ",
          85
        ],
        [
          "了",
          70
        ],
        [
          "的",
          47
        ],
        [
          "人",
          46
        ],
        [
          "道",
          37
        ],
        [
          "來",
          33
        ],
        [
          "一",
          20
        ],
        [
          "們",
          20
        ]
      ]
    }
  },
  "context_diminishing": {
    "original": {
      "n": 859,
      "known_n": 857,
      "oov_n": 2,
      "top1": 0.0,
      "top5": 0.119906868451688,
      "mean_nll_known": 6.304477322289402,
      "bpc_known": 9.09543816826385
    },
    "no_punct": {
      "n": 859,
      "known_n": 857,
      "oov_n": 2,
      "top1": 0.047729918509895226,
      "top5": 0.16880093131548313,
      "mean_nll_known": 6.195895067215166,
      "bpc_known": 8.938787087339708
    },
    "output_mask_only": {
      "n": 859,
      "known_n": 857,
      "oov_n": 2,
      "top1": 0.0570430733410943,
      "top5": 0.16996507566938301,
      "mean_nll_known": 6.255032375207845,
      "bpc_known": 9.024104188312272
    },
    "paired_summary": {
      "n": 859,
      "original_correct": 0,
      "no_punct_correct": 41,
      "helped": 41,
      "harmed": 0,
      "no_punct_predicted_types": 12,
      "no_punct_most_predicted": [
        [
          "了",
          676
        ],
        [
          "的",
          111
        ],
        [
          "不",
          20
        ],
        [
          "道",
          14
        ],
        [
          "玉",
          10
        ],
        [
          "\n",
          10
        ],
        [
          "人",
          7
        ],
        [
          "一",
          4
        ]
      ],
      "masked_most_predicted": [
        [
          "了",
          625
        ],
        [
          "不",
          105
        ],
        [
          "的",
          59
        ],
        [
          "道",
          24
        ],
        [
          "\n",
          10
        ],
        [
          "玉",
          8
        ],
        [
          "人",
          7
        ],
        [
          "一",
          7
        ]
      ]
    }
  },
  "context_saturating": {
    "original": {
      "n": 859,
      "known_n": 857,
      "oov_n": 2,
      "top1": 0.002328288707799767,
      "top5": 0.09778812572759023,
      "mean_nll_known": 6.678336585625561,
      "bpc_known": 9.63480307346933
    },
    "no_punct": {
      "n": 859,
      "known_n": 857,
      "oov_n": 2,
      "top1": 0.034924330616996506,
      "top5": 0.1408614668218859,
      "mean_nll_known": 6.531499370247688,
      "bpc_known": 9.422961751025726
    },
    "output_mask_only": {
      "n": 859,
      "known_n": 857,
      "oov_n": 2,
      "top1": 0.034924330616996506,
      "top5": 0.14668218859138532,
      "mean_nll_known": 6.653614567350373,
      "bpc_known": 9.599136740302948
    },
    "paired_summary": {
      "n": 859,
      "original_correct": 2,
      "no_punct_correct": 30,
      "helped": 28,
      "harmed": 0,
      "no_punct_predicted_types": 19,
      "no_punct_most_predicted": [
        [
          "了",
          472
        ],
        [
          "的",
          265
        ],
        [
          "不",
          53
        ],
        [
          "一",
          16
        ],
        [
          "來",
          12
        ],
        [
          "道",
          11
        ],
        [
          "人",
          6
        ],
        [
          "子",
          5
        ]
      ],
      "masked_most_predicted": [
        [
          "了",
          461
        ],
        [
          "的",
          281
        ],
        [
          "不",
          42
        ],
        [
          "來",
          15
        ],
        [
          "道",
          14
        ],
        [
          "一",
          11
        ],
        [
          "人",
          7
        ],
        [
          "子",
          4
        ]
      ]
    }
  },
  "long_context_diminishing": {
    "original": {
      "n": 859,
      "known_n": 857,
      "oov_n": 2,
      "top1": 0.0,
      "top5": 0.06752037252619325,
      "mean_nll_known": 6.590568049319149,
      "bpc_known": 9.508179841393984
    },
    "no_punct": {
      "n": 859,
      "known_n": 857,
      "oov_n": 2,
      "top1": 0.025611175785797437,
      "top5": 0.11059371362048893,
      "mean_nll_known": 6.5354497747852625,
      "bpc_known": 9.428660980061588
    },
    "output_mask_only": {
      "n": 859,
      "known_n": 857,
      "oov_n": 2,
      "top1": 0.025611175785797437,
      "top5": 0.11292200232828871,
      "mean_nll_known": 6.54272585286721,
      "bpc_known": 9.439158141827537
    },
    "paired_summary": {
      "n": 859,
      "original_correct": 0,
      "no_punct_correct": 22,
      "helped": 22,
      "harmed": 0,
      "no_punct_predicted_types": 1,
      "no_punct_most_predicted": [
        [
          "了",
          859
        ]
      ],
      "masked_most_predicted": [
        [
          "了",
          859
        ]
      ]
    }
  }
}
