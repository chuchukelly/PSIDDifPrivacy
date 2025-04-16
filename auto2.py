import pandas as pd
import numpy as np

def countneverecquitattempt(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # filter samples 过滤样本
        condition_total = (
            # 在这里定义你的总过滤条件
            # 例如: (df['SomeColumn'] == some_value) & (df['AnotherColumn'] > some_threshold)
            (df['ECEV'] == 2)
            &((df['TRYSTOP']==1)|(df['TRYSTOP']==3))# 替换为你的实际条件
        )
        totalsamples = df[condition_total]

        # count the sample after filter 计算过滤出的样本数量
        totalsample_count = totalsamples.shape[0]

        # 计算符合条件的行中 TBSUPPWT 的总和
        tbsuppwt_sum = df.loc[condition_total, 'TBSUPPWT'].sum()

        condition1 = condition_total & (
            (df['TRYSTOP'] == 3) 
        )
        samples = df[condition1]
        sample_count = samples.shape[0]
        
        tbsuppwt = df.loc[condition1, 'TBSUPPWT'].sum()

        # 计算加权使用者数和加权总人数
        weighted_ever_users = tbsuppwt
        weighted_total_respondents = tbsuppwt_sum

        # 计算加权比例
        p = weighted_ever_users / weighted_total_respondents

        # 计算标准误
        se = np.sqrt(p * (1 - p) / totalsample_count)

        # 95% 置信区间
        z = 1.96  # 正态分布的z值
        ci_lower = p - z * se
        ci_upper = p + z * se
        

        # 返回结果
        return {
            "totalsample_count": totalsample_count,
            "sample_count": sample_count,
            "p1": p,
            "confidence_interval1": (ci_lower, ci_upper)
        }
    
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"文件为空: {file_path}")
    except pd.errors.ParserError:
        print(f"文件解析错误: {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        
def countusedbutstopecquitattempt(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # filter samples 过滤样本
        condition_total = (
            # 在这里定义你的总过滤条件
            # 例如: (df['SomeColumn'] == some_value) & (df['AnotherColumn'] > some_threshold)
            (df['ECEV'] == 1)&(df['ECFREQ'] == 3)
            &((df['TRYSTOP']==1)|(df['TRYSTOP']==3))# 替换为你的实际条件
        )
        totalsamples = df[condition_total]

        # count the sample after filter 计算过滤出的样本数量
        totalsample_count = totalsamples.shape[0]

        # 计算符合条件的行中 TBSUPPWT 的总和
        tbsuppwt_sum = df.loc[condition_total, 'TBSUPPWT'].sum()

        condition1 = condition_total & (
            (df['TRYSTOP'] == 3) 
        )
        samples = df[condition1]
        sample_count = samples.shape[0]
        
        tbsuppwt = df.loc[condition1, 'TBSUPPWT'].sum()

        # 计算加权使用者数和加权总人数
        weighted_ever_users = tbsuppwt
        weighted_total_respondents = tbsuppwt_sum

        # 计算加权比例
        p = weighted_ever_users / weighted_total_respondents

        # 计算标准误
        se = np.sqrt(p * (1 - p) / totalsample_count)

        # 95% 置信区间
        z = 1.96  # 正态分布的z值
        ci_lower = p - z * se
        ci_upper = p + z * se
        

        # 返回结果
        return {
            "totalsample_count": totalsample_count,
            "sample_count": sample_count,
            "p1": p,
            "confidence_interval1": (ci_lower, ci_upper)
        }
    
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"文件为空: {file_path}")
    except pd.errors.ParserError:
        print(f"文件解析错误: {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

def countcurrentuseecquitattempt(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # filter samples 过滤样本
        condition_total = (
            # 在这里定义你的总过滤条件
            # 例如: (df['SomeColumn'] == some_value) & (df['AnotherColumn'] > some_threshold)
            (df['ECEV'] == 1) & ((df['ECFREQ'] == 1) | (df['ECFREQ'] == 2))
            &((df['TRYSTOP']==1)|(df['TRYSTOP']==3))# 替换为你的实际条件
        )
        totalsamples = df[condition_total]

        # count the sample after filter 计算过滤出的样本数量
        totalsample_count = totalsamples.shape[0]

        # 计算符合条件的行中 TBSUPPWT 的总和
        tbsuppwt_sum = df.loc[condition_total, 'TBSUPPWT'].sum()

        condition1 = condition_total & (
            (df['TRYSTOP'] == 3) 
        )
        samples = df[condition1]
        sample_count = samples.shape[0]
        
        tbsuppwt = df.loc[condition1, 'TBSUPPWT'].sum()

        # 计算加权使用者数和加权总人数
        weighted_ever_users = tbsuppwt
        weighted_total_respondents = tbsuppwt_sum

        # 计算加权比例
        p = weighted_ever_users / weighted_total_respondents

        # 计算标准误
        se = np.sqrt(p * (1 - p) / totalsample_count)

        # 95% 置信区间
        z = 1.96  # 正态分布的z值
        ci_lower = p - z * se
        ci_upper = p + z * se
        

        # 返回结果
        return {
            "totalsample_count": totalsample_count,
            "sample_count": sample_count,
            "p1": p,
            "confidence_interval1": (ci_lower, ci_upper)
        }
    
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"文件为空: {file_path}")
    except pd.errors.ParserError:
        print(f"文件解析错误: {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

def write_results_to_file2(csv_file_path, file_path):
    results = [
    countneverecquitattempt(csv_file_path),
    countusedbutstopecquitattempt(csv_file_path),
    countcurrentuseecquitattempt(csv_file_path)
    ]
    with open(file_path, 'w') as file:
        for result in results:
            file.write(f"{result['totalsample_count']} "
            f"{result['sample_count']} "
            f"{result['p1'] * 100}[{result['confidence_interval1'][0] * 100}, {result['confidence_interval1'][1] * 100}] \n")
        
csv_file_path = '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/cps_00006self.csv'
file_path='/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/a.txt'
write_results_to_file2(csv_file_path, file_path)