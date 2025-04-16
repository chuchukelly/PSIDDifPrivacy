import pandas as pd
import numpy as np



# 计算加权使用者数和加权总人数
weighted_ever_users = 1103670.2751 #(data['weight'] * data['ever_used_ec']).sum()
weighted_total_respondents = 39735076.5256#data['weight'].sum()

# 计算加权比例
p = weighted_ever_users / weighted_total_respondents

# 计算标准误
se = np.sqrt(p * (1 - p) /13086)#weighted_total_respondents)

# 95% 置信区间
z = 1.96  # 正态分布的z值
ci_lower = p - z * se
ci_upper = p + z * se

# 打印结果
print(f"95% confidence interval: {p * 100:.1f}[{ci_lower * 100:.1f}, {ci_upper * 100:.1f}]")
