import pandas as pd
import numpy as np

# path 定义CSV文件路径
#file_path = '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/cps_00006self.csv'#fomersmoker.csv'
file_path = '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/synthetic_cps_00006self copye2d0.csv'
# 加载CSV文件，并指定第一行作为列名
df = pd.read_csv(file_path)

# filter samples 过滤样本
#例如过滤出TSRWT不为0的样本：samples = df[df['TSRWT'] != 0]
condition_total = (#(df['TCIG100'] == 98) |
   ((df['ECEV'] == 1) |(df['ECEV'] == 2))
   #(df['ECEV'] == 1)
   #&(df['TSMKER'] == 1)
   #&(df['TSMKER'] == 2)
   #&(df['TSINCEQUIT'] < 365*1)
   #&(df['TSINCEQUIT'] >= 365*0)
   #&((df['TFREQ']==2)|(df['TFREQ']==3))&(df['TCIG100']==2)
   #&(df['SEX'] == 1)
   #&((df['TFREQ']==2)|(df['TFREQ']==3))&(df['TCIG100']==2)
   #(df['ECEV'] == 1)&(df['ECFREQ'] == 3)&
   #((df['TRYSTOP'] == 1)|(df['TRYSTOP'] == 3))
   #((df['THABIT1'] ==2)|(df['THABIT1'] ==3))
   #&((df['RACE']!=652)&(df['RACE']!=651)&(df['RACE']!=100)&(df['RACE']!=200)&(df['RACE']!=300))
   #&(df['RACE']==300)
   #&(df['HISPAN']==0)
   #&((df['EDUC']==10)|(df['EDUC']==20)|(df['EDUC']==30)|(df['EDUC']==1)|(df['EDUC']==2)|(df['EDUC']==40)|(df['EDUC']==50)|(df['EDUC']==60))
   #&((df['EDUC']==71)|(df['EDUC']==73))
   #&((df['EDUC']==81)|(df['EDUC']==91)|(df['EDUC']==92))
   &((df['EDUC']==111)|(df['EDUC']==123)|(df['EDUC']==124)|(df['EDUC']==125))
   #&(df['AGE'] >= 18)&(df['AGE'] <= 24)
   #((df['ECFREQ'] <= 3)|(df['ECFREQ'] == 99))&
   ##((df['ECEV'] == 1)  &((df['ECFREQ'] == 1)|(df['ECFREQ'] == 2)))&
   ###&((df['TSMKER'] == 1)| ((df['TFREQ']==2)|(df['TFREQ']==3))&(df['TCIG100']==2)|((df['TSMKER'] == 2)&(df['TSINCEQUIT'] <99996)&(df['TSINCEQUIT'] >=365*0)))#
   #((df['TSINCEQUIT'] <365*1)&(df['TSINCEQUIT'] >=365*0))
   #(df['TSMKER'] == 1)
   #((df['TFREQ']==2)|(df['TFREQ']==3))&(df['TCIG100']==2)
   #(df['TSINCEQUIT'] <5*365)&(df['TSINCEQUIT'] >=365*3)
)
totalsamples = df[condition_total]
# count the sample after filter 计算过滤出的样本数量
totalsample_count = totalsamples.shape[0]
print(f"totalsample_count: {totalsample_count}")

# 计算符合条件的行中 TBSUPPWT 的总和
tbsuppwt_sum = df.loc[condition_total, 'TBSUPPWT'].sum()
print(f"TBSUPPWT 总和: {tbsuppwt_sum}")

condition = condition_total &(
    #df['TRYSTOP'] ==3 
    #(df['TSINCEQUIT']>=90)&(df['TFREQ']==1)
    (df['ECEV'] == 1) &((df['ECFREQ'] == 1)|(df['ECFREQ'] == 2))
    #df['ECFREQ'] == 2
)
samples = df[condition]
sample_count = samples.shape[0]
print(f"sample_count: {sample_count}")
tbsuppwt= df.loc[condition, 'TBSUPPWT'].sum()
print(f"TBSUPPWT 和: {tbsuppwt}")

# 计算加权使用者数和加权总人数
weighted_ever_users = tbsuppwt #(data['weight'] * data['ever_used_ec']).sum()
weighted_total_respondents = tbsuppwt_sum#data['weight'].sum()

# 计算加权比例
p = weighted_ever_users / weighted_total_respondents

# 计算标准误
se = np.sqrt(p * (1 - p) /totalsample_count)#weighted_total_respondents)

# 95% 置信区间
z = 1.96  # 正态分布的z值
ci_lower = p - z * se
ci_upper = p + z * se

# 打印结果
print(f"95% confidence interval: {p * 100:.1f}[{ci_lower * 100:.1f}, {ci_upper * 100:.1f}]")

