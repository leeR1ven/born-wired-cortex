# 现代中文规模实验独立核验

状态：PASS。

核对原始官方 SHA256、全部整理文件和完整文档哈希；全体完整文档无跨分区重复，保留官方对话留出，另按冻结哈希分区。按原始来源 ID 抽样重建清洗。
独立逐字重放首 3,000,000 字符，与首检查点全部非零权重、活动顺序及计数精确一致；重建整条交替训练流，核对所有阶段流哈希、字符频次、时钟、边更新数和最终边界。所有完整矩阵 SHA256、有限非负权重及覆盖统计通过。
重算所有保存预测的指标汇总；独立复算最终每组首尾固定样本共 72 个预测、18 条固定问法的完整 64 字续写。未重新训练全部数亿字符，也未以大模型判断语义。

解释边界：完整文档去重不排除近重复、共享对话前缀和短片段背诵。测试按文档均匀抽样后每文档选一个位置，不是按自然字符频率加权；reply_first 单列。clarification_keyword_present 只是关键词出现，不能视为理解或合理澄清。实际训练连续拼接不同对话/文章，会产生跨文档边。

本核验不声称复算每一项常用短语计数，也不把所有自然文本预测独立重算；核验范围和数量如下。

~~~json
{
  "status": "PASS",
  "seconds": 64.13430010000047,
  "checks": {
    "file_hashes": 34,
    "raw_files": 4,
    "prepared_documents": 7013977,
    "raw_cleaning_samples": 24,
    "matrix_checkpoints": 5,
    "independent_training_characters": 3000000,
    "metric_prediction_rows": 5760,
    "metric_groups": 45,
    "independent_final_predictions": 72,
    "independent_final_continuations": 18
  },
  "protocol_sha256": "f93399f99d3d495e74dfb4fb1698001d583b371e1107c61527f9cab4a2f33385",
  "verifier_sha256": "31fbbf4532c9cc325a88b6f4d3d9ff17207ea44601dcedeb75493ca837852088",
  "independent_recall_helper_sha256": "35ceec3ea09d69455f7250c3cab87741ac576ea28eeb80aab054bc5740fd9abc",
  "metrics": {
    "3000000": {
      "recent_only": {
        "lccc_base": {
          "n": 128,
          "known": 128,
          "top1": 15,
          "top5": 39,
          "bits_per_known_character": 8.177404770032627
        },
        "wikipedia": {
          "n": 128,
          "known": 128,
          "top1": 17,
          "top5": 36,
          "bits_per_known_character": 9.314921490653818
        },
        "reply_first": {
          "n": 128,
          "known": 128,
          "top1": 4,
          "top5": 20,
          "bits_per_known_character": 8.938706425872093
        }
      },
      "persistent8_weak": {
        "lccc_base": {
          "n": 128,
          "known": 128,
          "top1": 12,
          "top5": 31,
          "bits_per_known_character": 8.442223136692098
        },
        "wikipedia": {
          "n": 128,
          "known": 128,
          "top1": 17,
          "top5": 31,
          "bits_per_known_character": 9.367828195602703
        },
        "reply_first": {
          "n": 128,
          "known": 128,
          "top1": 4,
          "top5": 25,
          "bits_per_known_character": 8.847610431596118
        }
      },
      "persistent4_strong": {
        "lccc_base": {
          "n": 128,
          "known": 128,
          "top1": 10,
          "top5": 18,
          "bits_per_known_character": 8.97353073103936
        },
        "wikipedia": {
          "n": 128,
          "known": 128,
          "top1": 6,
          "top5": 19,
          "bits_per_known_character": 10.04358467331542
        },
        "reply_first": {
          "n": 128,
          "known": 128,
          "top1": 2,
          "top5": 19,
          "bits_per_known_character": 9.012364573306407
        }
      }
    },
    "30000000": {
      "recent_only": {
        "lccc_base": {
          "n": 128,
          "known": 128,
          "top1": 15,
          "top5": 41,
          "bits_per_known_character": 8.301745153748652
        },
        "wikipedia": {
          "n": 128,
          "known": 128,
          "top1": 20,
          "top5": 41,
          "bits_per_known_character": 8.638668480291713
        },
        "reply_first": {
          "n": 128,
          "known": 128,
          "top1": 4,
          "top5": 20,
          "bits_per_known_character": 8.972386788542881
        }
      },
      "persistent8_weak": {
        "lccc_base": {
          "n": 128,
          "known": 128,
          "top1": 12,
          "top5": 31,
          "bits_per_known_character": 8.68261882762021
        },
        "wikipedia": {
          "n": 128,
          "known": 128,
          "top1": 19,
          "top5": 33,
          "bits_per_known_character": 9.17369335081505
        },
        "reply_first": {
          "n": 128,
          "known": 128,
          "top1": 4,
          "top5": 24,
          "bits_per_known_character": 9.086381684302136
        }
      },
      "persistent4_strong": {
        "lccc_base": {
          "n": 128,
          "known": 128,
          "top1": 9,
          "top5": 19,
          "bits_per_known_character": 9.197406373786254
        },
        "wikipedia": {
          "n": 128,
          "known": 128,
          "top1": 6,
          "top5": 14,
          "bits_per_known_character": 9.855007329292471
        },
        "reply_first": {
          "n": 128,
          "known": 128,
          "top1": 3,
          "top5": 19,
          "bits_per_known_character": 9.247748850006285
        }
      }
    },
    "100000000": {
      "recent_only": {
        "lccc_base": {
          "n": 128,
          "known": 128,
          "top1": 17,
          "top5": 41,
          "bits_per_known_character": 8.431149950831434
        },
        "wikipedia": {
          "n": 128,
          "known": 128,
          "top1": 21,
          "top5": 43,
          "bits_per_known_character": 8.754136124468557
        },
        "reply_first": {
          "n": 128,
          "known": 128,
          "top1": 4,
          "top5": 20,
          "bits_per_known_character": 9.069293943085402
        }
      },
      "persistent8_weak": {
        "lccc_base": {
          "n": 128,
          "known": 128,
          "top1": 13,
          "top5": 32,
          "bits_per_known_character": 8.78506915338648
        },
        "wikipedia": {
          "n": 128,
          "known": 128,
          "top1": 19,
          "top5": 35,
          "bits_per_known_character": 9.252925088528013
        },
        "reply_first": {
          "n": 128,
          "known": 128,
          "top1": 4,
          "top5": 24,
          "bits_per_known_character": 9.178965319002936
        }
      },
      "persistent4_strong": {
        "lccc_base": {
          "n": 128,
          "known": 128,
          "top1": 9,
          "top5": 20,
          "bits_per_known_character": 9.273448525662797
        },
        "wikipedia": {
          "n": 128,
          "known": 128,
          "top1": 6,
          "top5": 15,
          "bits_per_known_character": 9.85792228017668
        },
        "reply_first": {
          "n": 128,
          "known": 128,
          "top1": 3,
          "top5": 19,
          "bits_per_known_character": 9.331807992606349
        }
      }
    },
    "300000000": {
      "recent_only": {
        "lccc_base": {
          "n": 128,
          "known": 128,
          "top1": 16,
          "top5": 39,
          "bits_per_known_character": 8.501060036568965
        },
        "wikipedia": {
          "n": 128,
          "known": 128,
          "top1": 17,
          "top5": 42,
          "bits_per_known_character": 8.752820146262316
        },
        "reply_first": {
          "n": 128,
          "known": 128,
          "top1": 3,
          "top5": 20,
          "bits_per_known_character": 9.13633985751435
        }
      },
      "persistent8_weak": {
        "lccc_base": {
          "n": 128,
          "known": 128,
          "top1": 14,
          "top5": 31,
          "bits_per_known_character": 8.833638165891909
        },
        "wikipedia": {
          "n": 128,
          "known": 128,
          "top1": 16,
          "top5": 35,
          "bits_per_known_character": 9.267842411764873
        },
        "reply_first": {
          "n": 128,
          "known": 128,
          "top1": 3,
          "top5": 21,
          "bits_per_known_character": 9.23810682209576
        }
      },
      "persistent4_strong": {
        "lccc_base": {
          "n": 128,
          "known": 128,
          "top1": 9,
          "top5": 20,
          "bits_per_known_character": 9.321132821745982
        },
        "wikipedia": {
          "n": 128,
          "known": 128,
          "top1": 5,
          "top5": 13,
          "bits_per_known_character": 9.882553242659084
        },
        "reply_first": {
          "n": 128,
          "known": 128,
          "top1": 3,
          "top5": 18,
          "bits_per_known_character": 9.380472834920194
        }
      }
    },
    "344787049": {
      "recent_only": {
        "lccc_base": {
          "n": 128,
          "known": 128,
          "top1": 17,
          "top5": 39,
          "bits_per_known_character": 8.502568873122012
        },
        "wikipedia": {
          "n": 128,
          "known": 128,
          "top1": 21,
          "top5": 45,
          "bits_per_known_character": 8.612151795635508
        },
        "reply_first": {
          "n": 128,
          "known": 128,
          "top1": 3,
          "top5": 20,
          "bits_per_known_character": 9.137642132839854
        }
      },
      "persistent8_weak": {
        "lccc_base": {
          "n": 128,
          "known": 128,
          "top1": 15,
          "top5": 31,
          "bits_per_known_character": 8.833334461589237
        },
        "wikipedia": {
          "n": 128,
          "known": 128,
          "top1": 20,
          "top5": 38,
          "bits_per_known_character": 9.12527120941827
        },
        "reply_first": {
          "n": 128,
          "known": 128,
          "top1": 3,
          "top5": 21,
          "bits_per_known_character": 9.237552216917155
        }
      },
      "persistent4_strong": {
        "lccc_base": {
          "n": 128,
          "known": 128,
          "top1": 9,
          "top5": 19,
          "bits_per_known_character": 9.319241620521908
        },
        "wikipedia": {
          "n": 128,
          "known": 128,
          "top1": 6,
          "top5": 14,
          "bits_per_known_character": 9.784045700050603
        },
        "reply_first": {
          "n": 128,
          "known": 128,
          "top1": 3,
          "top5": 18,
          "bits_per_known_character": 9.381334416998415
        }
      }
    }
  }
}
~~~

## 循环活动核验补充

独立复算 4 个固定输入的各 160 步输出，及每条首尾共 16 帧全部活动、正电流候选数、赢家贡献和单步去自连接电流。首次精确状态重现与原 64 字输出均一致。这里的状态不含未参与读出的单调时钟；单步去自连接并未改写实际轨迹。

~~~json
{
  "status": "PASS",
  "independently_replayed_steps": 640,
  "complete_activity_frames": 64,
  "winner_decompositions": 64,
  "single_step_self_edge_counterfactuals": 64,
  "source_sha256": "680081682a4ceb90da81bb95e74994b48b7267fa30fefabe4c962a7942020a13",
  "cycle_audit_sha256": "089a1298f688c9bd6a907f0f59d0bda4089ed8c20654cc6ebaaafc8a18ad5038",
  "results": [
    {
      "prompt": "你好",
      "first_repeat": {
        "first_step": 11,
        "repeated_at": 13,
        "period": 2
      },
      "final_positive_candidates": 11543
    },
    {
      "prompt": "为什么",
      "first_repeat": {
        "first_step": 10,
        "repeated_at": 13,
        "period": 3
      },
      "final_positive_candidates": 11938
    },
    {
      "prompt": "地球",
      "first_repeat": {
        "first_step": 11,
        "repeated_at": 13,
        "period": 2
      },
      "final_positive_candidates": 11543
    },
    {
      "prompt": "请解释一下咕噜帕索是什么意思",
      "first_repeat": {
        "first_step": 8,
        "repeated_at": 11,
        "period": 3
      },
      "final_positive_candidates": 11516
    }
  ]
}
~~~
