# 扩大语料实验独立核验

状态：PASS。

独立重建六本清洗、整段分块、固定种子分区和追加训练流；核对 442 个来源块、全部文件哈希和 15 组新旧分区长段落交集。
训练前 658,462 字符与原训练流精确一致。四阶段固定使用 7,652 字符的全量训练词表；从此前已核验的旧权重投影后，独立续训剩余 2,183,223 字符，四阶段完整矩阵、活动、clock、learned_characters 和边更新次数均逐位一致。
独立复现 6,144 行预测及 160 条完整轨迹共 10,240 步，核对预测概率、名次、前五、每一步候选注入及全部活动、尾部周期。复算 30,608 行频次与非零出边覆盖、频次分组分数和前文出现诊断。
未导入训练核心、候选控制器或原评分函数；未修改数据或旧模型。

| 训练字符数 | 回忆条件 | 旧测试命中/256 | 新测试命中/256 | 旧 BPC | 新 BPC |
|---:|---|---:|---:|---:|---:|
| 658,462 | recent_only | 38 | 13 | 8.5846 | 12.7867 |
| 658,462 | persistent8_weak | 35 | 11 | 8.5137 | 11.2133 |
| 658,462 | persistent4_strong | 23 | 9 | 9.0170 | 11.4666 |
| 1,000,000 | recent_only | 40 | 20 | 8.5126 | 10.6129 |
| 1,000,000 | persistent8_weak | 37 | 17 | 8.5701 | 9.9722 |
| 1,000,000 | persistent4_strong | 22 | 13 | 9.0967 | 10.3152 |
| 2,000,000 | recent_only | 36 | 25 | 8.5892 | 9.0613 |
| 2,000,000 | persistent8_weak | 32 | 24 | 8.7815 | 9.2534 |
| 2,000,000 | persistent4_strong | 22 | 15 | 9.3192 | 9.7848 |
| 2,841,685 | recent_only | 34 | 25 | 8.6994 | 8.8903 |
| 2,841,685 | persistent8_weak | 30 | 26 | 8.8944 | 9.1301 |
| 2,841,685 | persistent4_strong | 19 | 12 | 9.4149 | 9.7009 |

解释边界：

- known 表示字符在预分配词表中；起始阶段可包含训练频次为 0 的字符。频次分组以当时实际读入字符为准。
- 这些是同书未见块及固定真实 64 字前文的下一字预测；没有验证现代知识、跨书泛化或自由续写的语义正确性。
- 完全长段落交集为 0、16/64 字原样前文未出现，都不能排除近重复、短片段背诵或共同人物情节；本轮没有近重复去重。
- 追加数据同时改变书目组成和字符频次，不能把全部变化归因为数据量单一因素。四阶段共用测试位置，协议没有据测试选择参数。
- 活动范围和尾部周期是动态指标；更宽活动或更长循环不足以证明语言能力改善。

独立核验耗时 123.96 秒。

原始核验摘要：

~~~json
{
  "status": "PASS",
  "elapsed_seconds": 123.96304059999966,
  "counts": {
    "source_hashes": 29,
    "corpus_blocks": 442,
    "source_books": 6,
    "prediction_rows": 6144,
    "frequency_metric_groups": 105,
    "trajectories": 160,
    "trajectory_steps": 10240,
    "stages": 4,
    "character_coverage_rows": 30608
  },
  "stages": [
    {
      "stage": 658462,
      "conditions": {
        "recent_only": {
          "old_hits": 38,
          "new_hits": 13,
          "old_bpc": 8.58463174596918,
          "new_bpc": 12.78672201186121
        },
        "persistent8_weak": {
          "old_hits": 35,
          "new_hits": 11,
          "old_bpc": 8.513703658378454,
          "new_bpc": 11.213269380682016
        },
        "persistent4_strong": {
          "old_hits": 23,
          "new_hits": 9,
          "old_bpc": 9.01697469343734,
          "new_bpc": 11.466557621170878
        }
      }
    },
    {
      "stage": 1000000,
      "conditions": {
        "recent_only": {
          "old_hits": 40,
          "new_hits": 20,
          "old_bpc": 8.512559510339068,
          "new_bpc": 10.612865701391327
        },
        "persistent8_weak": {
          "old_hits": 37,
          "new_hits": 17,
          "old_bpc": 8.570093969898632,
          "new_bpc": 9.972219745650996
        },
        "persistent4_strong": {
          "old_hits": 22,
          "new_hits": 13,
          "old_bpc": 9.096735140118255,
          "new_bpc": 10.315223648593678
        }
      }
    },
    {
      "stage": 2000000,
      "conditions": {
        "recent_only": {
          "old_hits": 36,
          "new_hits": 25,
          "old_bpc": 8.589234395916705,
          "new_bpc": 9.061347610151012
        },
        "persistent8_weak": {
          "old_hits": 32,
          "new_hits": 24,
          "old_bpc": 8.781493915362836,
          "new_bpc": 9.253418402541039
        },
        "persistent4_strong": {
          "old_hits": 22,
          "new_hits": 15,
          "old_bpc": 9.319183202911283,
          "new_bpc": 9.784803952763898
        }
      }
    },
    {
      "stage": 2841685,
      "conditions": {
        "recent_only": {
          "old_hits": 34,
          "new_hits": 25,
          "old_bpc": 8.69937794733093,
          "new_bpc": 8.890285444204325
        },
        "persistent8_weak": {
          "old_hits": 30,
          "new_hits": 26,
          "old_bpc": 8.89440412216919,
          "new_bpc": 9.130079103685844
        },
        "persistent4_strong": {
          "old_hits": 19,
          "new_hits": 12,
          "old_bpc": 9.414902619662971,
          "new_bpc": 9.700937770638225
        }
      }
    }
  ],
  "independently_replayed_new_training_characters": 2183223,
  "starting_checkpoint": "Previously verified old 658462-character model, projected into the fixed full-training vocabulary",
  "full_weight_arrays_exact": true,
  "protocol_sha256": "a74bfeb1ba5ce5a58ca526650dfa398ea0c921aed9a5e382f64af777e261fbc1",
  "manifest_sha256": "578b2ceffe57e914f99d81829c5775afef387146f58455db746305c16bac9424",
  "verifier_sha256": "35ceec3ea09d69455f7250c3cab87741ac576ea28eeb80aab054bc5740fd9abc"
}
~~~


## 追加稀字分层诊断核验

状态：PASS。按全量训练频次独立重建各组候选池、种子19441至19445的固定抽样和主实验256位置排除。共632个不同新测试位置，核对4条件全部2528行的分组、真实前文、目标、词表身份及20组评分汇总；另独立计算每组首/中/末各两行，共120个完整预测与活动。所用起始/全量权重哈希均匹配上述已完整验证的主实验检查点。没有重训。

频次0组120个目标全部不在全量训练词表，命中0且BPC为空；空值不等于损失0。1–9次组128例：弱候选起始和全量均0命中，recent_only均1命中，概率损失在扩大后下降。分层依据全量训练频次，起始阶段这些字可能尚未经历；不应改称起始阶段频次。

本诊断为看到自然样本缺少稀字后的事后分层检查，未用于选参。按出现位置抽样，并非每个不同汉字等权；不报告跨组汇总准确率，不能当作自然文本平均表现。

~~~json
{
  "status": "PASS",
  "distinct_selected_positions": 632,
  "all_group_rows_with_frequency_prefix_and_metrics_checked": 2528,
  "independently_recomputed_predictions": 120,
  "sampling": "first two, middle two, last two per group per condition; chosen without consulting predictions",
  "protocol_sha256": "098656d6abbfd2d6c4a32f66f18b4c06e5a0bf6e230c9a9ec966112d6fbee9a6",
  "verified_main_checkpoint_weights": true,
  "results": {
    "658462_recent_only": {
      "0": {
        "n": 120,
        "hits": 0,
        "bpc": null
      },
      "1-9": {
        "n": 128,
        "hits": 1,
        "bpc": 22.048030975128835
      },
      "10-99": {
        "n": 128,
        "hits": 1,
        "bpc": 20.765179739884406
      },
      "100-999": {
        "n": 128,
        "hits": 1,
        "bpc": 16.754129063075215
      },
      "1000+": {
        "n": 128,
        "hits": 9,
        "bpc": 10.29547362213718
      }
    },
    "658462_persistent8_weak": {
      "0": {
        "n": 120,
        "hits": 0,
        "bpc": null
      },
      "1-9": {
        "n": 128,
        "hits": 0,
        "bpc": 21.817495395283224
      },
      "10-99": {
        "n": 128,
        "hits": 0,
        "bpc": 19.46619493044505
      },
      "100-999": {
        "n": 128,
        "hits": 0,
        "bpc": 14.491513260204865
      },
      "1000+": {
        "n": 128,
        "hits": 8,
        "bpc": 9.197568812230356
      }
    },
    "2841685_recent_only": {
      "0": {
        "n": 120,
        "hits": 0,
        "bpc": null
      },
      "1-9": {
        "n": 128,
        "hits": 1,
        "bpc": 17.01136734030581
      },
      "10-99": {
        "n": 128,
        "hits": 5,
        "bpc": 13.465993623542655
      },
      "100-999": {
        "n": 128,
        "hits": 9,
        "bpc": 10.191453312042945
      },
      "1000+": {
        "n": 128,
        "hits": 18,
        "bpc": 7.754433596206534
      }
    },
    "2841685_persistent8_weak": {
      "0": {
        "n": 120,
        "hits": 0,
        "bpc": null
      },
      "1-9": {
        "n": 128,
        "hits": 0,
        "bpc": 17.061845979517408
      },
      "10-99": {
        "n": 128,
        "hits": 0,
        "bpc": 12.712127812279588
      },
      "100-999": {
        "n": 128,
        "hits": 4,
        "bpc": 10.319467755635795
      },
      "1000+": {
        "n": 128,
        "hits": 16,
        "bpc": 8.260354013283262
      }
    }
  }
}
~~~
