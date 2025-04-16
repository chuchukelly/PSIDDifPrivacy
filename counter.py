import pandas as pd

# path 定义CSV文件路径
file_path = '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/cps_00006self.csv'#fomersmoker.csv'

# 加载CSV文件，并指定第一行作为列名
df = pd.read_csv(file_path)

# filter samples 过滤样本 2866
#例如过滤出TSRWT不为0的样本：samples = df[df['TSRWT'] != 0]
condition = (
    (df['ECFREQ'] >= 4) &(df['ECFREQ'] != 99)
    #(df['TFREQ'] == 98) |
    #(df['THABIT1'] == 98) |
    #(df['TEVERSTOP'] == 98) |
    #(df['SEX'] == 2) 
    #(df['ECYEAR'] == 99) |
    #(df['ECYEAR'] == 98) |
    #(df['ECYEAR'] == 97) |
    #(df['ECYEAR'] == 96) |
    #(df['TSINCEQUIT'] == 99997) |
    #(df['TCIG100'] == 96) |
    #(df['TFREQ'] == 96) |
    #(df['THABIT1'] == 96) |
    #(df['TEVERSTOP'] == 96) |
    #(df['SEX'] == 1)
    #(df['AGE']>= 65)#&(df['AGE']<= 64)
    #(df['RACE'] != 100)&(df['RACE'] != 652)&(df['RACE'] != 651)&(df['RACE'] != 200)&(df['RACE'] != 300)
    #&(df['HISPAN'] == 0)
    #((df['EDUC'] == 1)|(df['EDUC'] == 2)|(df['EDUC'] == 10)|(df['EDUC'] == 20)|(df['EDUC'] == 22)|(df['EDUC'] == 30)|(df['EDUC'] == 40)|(df['EDUC'] == 50)|(df['EDUC'] == 60))
    ###(((df['TSMKER'] ==2 )&(df['TSINCEQUIT'] !=99999 ))|((df['TCIG100'] == 2)&((df['TFREQ'] ==2 )|(df['TFREQ'] ==3 )))|(df['TSMKER'] ==1 ))
    #(df['TCIG100'] == 2)&((df['TFREQ'] ==2 )|(df['TFREQ'] ==3 ))
    #(df['TSINCEQUIT'] <365)#&(df['TSINCEQUIT'] >=1095)
    #((df['TSMKER'] ==2 )&(df['TSINCEQUIT'] !=99999 ))
    #((df['TSINCEQUIT'] >=1825)&(df['TSINCEQUIT'] !=99999))
    #((df['TCIG100'] == 2)&((df['TFREQ'] ==2 )|(df['TFREQ'] ==3 )))
    ###&(df['ECEV'] == 1) & (df['ECFREQ'] == 3)#&(df['ECFREQ'] != 2))
    ###& (df['ECYEAR'] >=95)
    #& (df['ECYEAR'] >=1)& (df['ECYEAR'] <=95)
    #(df['ECEV'] == 99) 
    #(df['ECFREQ'] == 96) 
    #(df['TSINCEQUIT'] == 99996) |
    #(df['TSINCEQUIT'] == 99998)
    #(df['AGE']>= 18)& ((df['TFREQ']== 2)|(df['TFREQ']== 3))
    ##(df['TSRWT'] != 0)
    #(df['TNRWT'] != 0)
    #(df['AGE']>= 65)#&(df['AGE']<= 24)
)
samples = df[condition]
# count the sample after filter 计算过滤出的样本数量
sample_count = samples.shape[0]

print(f"sample_count: {sample_count}")

# 计算符合条件的行中 TBSUPPWT 的总和
tbsuppwt_sum = df.loc[condition, 'TBSUPPWT'].sum()

print(f"TBSUPPWT 总和 (ECEV==1): {tbsuppwt_sum}")

# 保存过滤后的数据到一个新的CSV文件
#output_file_path = '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/cps_00006self.csv'
#samples.to_csv(output_file_path, index=False)

#print(f"过滤后的数据已保存到 {output_file_path}")
