# 同步多字符传播独立核验

状态：PASS。

原最终模型只读，完整权重哈希及本轮冻结源哈希一致。独立实现没有导入 population_propagation.py 或 run_population_probe.py。

完整重放四个事前指定详细提示 × 12 个条件 × 160 步，共 7,680 步；逐步核对活动身份/幅度、原始电流前八、休息计数、成员更新、首次精确状态重现，以及首尾指定帧的全部赢家连接贡献。全部 46 × 12 条轨迹的 88,320 步保存记录用于计数连续性和指标汇总重算。其余 42 个提示未独立重算全部电流或精确状态周期。

二值条件在初始感觉痕迹之后，将每个正信号且允许接收的前 K 个单元同步置为 1；足够合格候选时恰好 K 个。候选排名不作为句子输出。对角边屏蔽只影响当前电流计算，原矩阵未改。休息计数只限制新接收，已活动单元仍传出，并已独立核对。

解释细节：top_current/top8_currents 记录的是休息门之前的原始电流，可能指向暂时禁止接收的单元；实际新群体见下一帧 activity。初始感觉种子幅度仍有 .25/.0625；首次推进后才都是 1。带休息条件在播种末统一将所有感觉种子计数设为 D，而非按提示内的历史发放时间计算。

加权条件保留原先衰减后先删弱、注入后再删弱的顺序。K32/预算1 同时改变候选数与总反馈强度；二值规则也改变幅度与残留机制，因此不能将所有差异单独归因于 K。活动轮换、存活或周期变化均不是语义正确性或完整架构能力的评分。

| 条件 | 检出精确周期的提示 | 最终全灭 | 平均最终活动数 | 尾部平均不同成员集合数 |
|---|---:|---:|---:|---:|
| weighted8_spoken | 46/46 | 0 | 8.043 | 2.239 |
| weighted8_silent | 46/46 | 0 | 8.000 | 1.848 |
| weighted8_silent_no_self | 46/46 | 0 | 8.000 | 2.000 |
| weighted32_same_budget | 46/46 | 45 | 0.022 | 1.000 |
| weighted32_scaled_budget | 46/46 | 0 | 32.000 | 1.000 |
| binary8 | 46/46 | 0 | 8.000 | 1.022 |
| binary8_no_self | 46/46 | 0 | 8.000 | 2.000 |
| binary32 | 46/46 | 0 | 32.000 | 1.000 |
| binary32_no_self | 46/46 | 0 | 32.000 | 2.000 |
| binary128_no_self | 46/46 | 0 | 128.000 | 2.000 |
| binary32_no_self_rest1 | 46/46 | 0 | 32.000 | 2.261 |
| binary32_no_self_rest3 | 46/46 | 0 | 32.000 | 4.000 |

完整核验记录：

~~~json
{
  "status": "PASS",
  "seconds": 28.285582600001362,
  "checks": {
    "connection_decomposition_frames": 453,
    "independent_trajectory_steps": 7680,
    "independent_trajectories": 48,
    "aggregate_trajectories": 552,
    "aggregate_step_records": 88320,
    "binary_simultaneous_updates": 4480,
    "resting_outgoing_frames": 1280,
    "incoming_rest_gate_updates": 1280
  },
  "verifier_sha256": "eca43dd78c03fc0fa561517550477c2d7a9979a2d8389cbb65bac58b24652fbb",
  "independent_weighted_helper_sha256": "35ceec3ea09d69455f7250c3cab87741ac576ea28eeb80aab054bc5740fd9abc",
  "protocol_sha256": "112501c3fa917bf0755e5a28a0cb1c3b1e34b9ec9d3511adcd1cdd2d520f1f19",
  "result_sha256": {
    "weighted8_spoken.json": "415993e178b90ab0df5eb79bae7291ad97f6101e00899a357d2b4b51595f4b70",
    "weighted8_silent.json": "f3c9f73609000c0e397412cefb81c719b35b8d03f38231891dd9db0843a3ea0b",
    "weighted8_silent_no_self.json": "e11a7a0ff29ccd91b40aff41e94eb98a0a3b6cf1421d30fd6d403aade9278160",
    "weighted32_same_budget.json": "8028a69e5eb67d94c2a3d5601abf1f49e338266e048c5573d6a17b6f9bd50c80",
    "weighted32_scaled_budget.json": "18ac4e3e547b5f148c2f0bd55d2ab1c8d9ae3666f9c365bb482efc6b671fa2c6",
    "binary8.json": "c13ecea522cc74dd44f3e2cb2111a382dae29caa4338c09b0228ef50ba76b459",
    "binary8_no_self.json": "42796fca36601329a652bc2b50cc27d6c1e793e6b72325d70fcdc726fb425e2e",
    "binary32.json": "dcfdd94d48cbe93c43caa60eecc14158756f8841a66738b1d06b2a4f122677fa",
    "binary32_no_self.json": "21746fc62985437a009edbaae0731c610bfdabad81003b7aa501cc3df9837ab5",
    "binary128_no_self.json": "3dca6ab44b2e70154a590f206f924936b4e39b803f7404875e40c5282e3c7936",
    "binary32_no_self_rest1.json": "5d5971958f0af3b6f41441f020a01796651d3d9d20d084fa3777095ff9c86b7c",
    "binary32_no_self_rest3.json": "db074bca1915a0c0f034d393a4311844cdab52d98261e9a7dc222e3f435b622f"
  },
  "conditions": {
    "weighted8_spoken": {
      "prompts": 46,
      "exact_state_repeats": 46,
      "repeat_periods": {
        "2": 29,
        "1": 3,
        "3": 14
      },
      "final_extinctions": 0,
      "mean_final_active": 8.043478260869565,
      "mean_distinct_lifetime_units": 23.934782608695652,
      "mean_last32_member_sets": 2.239130434782609,
      "mean_late_turnover": 3.4354619565217392,
      "mean_late_effective_count": 1.9797144440701049,
      "mean_late_first_time_units": 0.0
    },
    "weighted8_silent": {
      "prompts": 46,
      "exact_state_repeats": 46,
      "repeat_periods": {
        "4": 39,
        "2": 6,
        "1": 1
      },
      "final_extinctions": 0,
      "mean_final_active": 8.0,
      "mean_distinct_lifetime_units": 15.23913043478261,
      "mean_last32_member_sets": 1.8478260869565217,
      "mean_late_turnover": 0.8478260869565217,
      "mean_late_effective_count": 7.874759031427717,
      "mean_late_first_time_units": 0.0
    },
    "weighted8_silent_no_self": {
      "prompts": 46,
      "exact_state_repeats": 46,
      "repeat_periods": {
        "2": 35,
        "4": 10,
        "6": 1
      },
      "final_extinctions": 0,
      "mean_final_active": 8.0,
      "mean_distinct_lifetime_units": 15.978260869565217,
      "mean_last32_member_sets": 2.0,
      "mean_late_turnover": 1.0434782608695652,
      "mean_late_effective_count": 7.8861111438431175,
      "mean_late_first_time_units": 0.0
    },
    "weighted32_same_budget": {
      "prompts": 46,
      "exact_state_repeats": 46,
      "repeat_periods": {
        "1": 46
      },
      "final_extinctions": 45,
      "mean_final_active": 0.021739130434782608,
      "mean_distinct_lifetime_units": 3.5217391304347827,
      "mean_last32_member_sets": 1.0,
      "mean_late_turnover": 0.0,
      "mean_late_effective_count": 0.021739130434782608,
      "mean_late_first_time_units": 0.0
    },
    "weighted32_scaled_budget": {
      "prompts": 46,
      "exact_state_repeats": 46,
      "repeat_periods": {
        "3": 9,
        "6": 9,
        "2": 27,
        "4": 1
      },
      "final_extinctions": 0,
      "mean_final_active": 32.0,
      "mean_distinct_lifetime_units": 43.369565217391305,
      "mean_last32_member_sets": 1.0,
      "mean_late_turnover": 0.0,
      "mean_late_effective_count": 29.460999042115454,
      "mean_late_first_time_units": 0.0
    },
    "binary8": {
      "prompts": 46,
      "exact_state_repeats": 46,
      "repeat_periods": {
        "1": 45,
        "2": 1
      },
      "final_extinctions": 0,
      "mean_final_active": 8.0,
      "mean_distinct_lifetime_units": 13.543478260869565,
      "mean_last32_member_sets": 1.0217391304347827,
      "mean_late_turnover": 0.021739130434782608,
      "mean_late_effective_count": 8.0,
      "mean_late_first_time_units": 0.0
    },
    "binary8_no_self": {
      "prompts": 46,
      "exact_state_repeats": 46,
      "repeat_periods": {
        "2": 46
      },
      "final_extinctions": 0,
      "mean_final_active": 8.0,
      "mean_distinct_lifetime_units": 15.282608695652174,
      "mean_last32_member_sets": 2.0,
      "mean_late_turnover": 1.0217391304347827,
      "mean_late_effective_count": 8.0,
      "mean_late_first_time_units": 0.0
    },
    "binary32": {
      "prompts": 46,
      "exact_state_repeats": 46,
      "repeat_periods": {
        "1": 46
      },
      "final_extinctions": 0,
      "mean_final_active": 32.0,
      "mean_distinct_lifetime_units": 46.58695652173913,
      "mean_last32_member_sets": 1.0,
      "mean_late_turnover": 0.0,
      "mean_late_effective_count": 32.0,
      "mean_late_first_time_units": 0.0
    },
    "binary32_no_self": {
      "prompts": 46,
      "exact_state_repeats": 46,
      "repeat_periods": {
        "2": 46
      },
      "final_extinctions": 0,
      "mean_final_active": 32.0,
      "mean_distinct_lifetime_units": 47.95652173913044,
      "mean_last32_member_sets": 2.0,
      "mean_late_turnover": 1.0,
      "mean_late_effective_count": 32.0,
      "mean_late_first_time_units": 0.0
    },
    "binary128_no_self": {
      "prompts": 46,
      "exact_state_repeats": 46,
      "repeat_periods": {
        "2": 46
      },
      "final_extinctions": 0,
      "mean_final_active": 128.0,
      "mean_distinct_lifetime_units": 178.67391304347825,
      "mean_last32_member_sets": 2.0,
      "mean_late_turnover": 2.0,
      "mean_late_effective_count": 128.0,
      "mean_late_first_time_units": 0.0
    },
    "binary32_no_self_rest1": {
      "prompts": 46,
      "exact_state_repeats": 46,
      "repeat_periods": {
        "4": 4,
        "2": 41,
        "6": 1
      },
      "final_extinctions": 0,
      "mean_final_active": 32.0,
      "mean_distinct_lifetime_units": 75.76086956521739,
      "mean_last32_member_sets": 2.260869565217391,
      "mean_late_turnover": 32.0,
      "mean_late_effective_count": 32.0,
      "mean_late_first_time_units": 0.0
    },
    "binary32_no_self_rest3": {
      "prompts": 46,
      "exact_state_repeats": 46,
      "repeat_periods": {
        "4": 46
      },
      "final_extinctions": 0,
      "mean_final_active": 32.0,
      "mean_distinct_lifetime_units": 140.56521739130434,
      "mean_last32_member_sets": 4.0,
      "mean_late_turnover": 32.0,
      "mean_late_effective_count": 32.0,
      "mean_late_first_time_units": 0.0
    }
  }
}
~~~
