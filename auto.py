import pandas as pd
import numpy as np


def counttotal(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # filter samples 过滤样本
        condition_total = (
            # 在这里定义你的总过滤条件
            # 例如: (df['SomeColumn'] == some_value) & (df['AnotherColumn'] > some_threshold)
            (df['ECEV'] == 1)| (df['ECEV'] == 2) # 替换为你的实际条件
        )
        totalsamples = df[condition_total]

        # count the sample after filter 计算过滤出的样本数量
        totalsample_count = totalsamples.shape[0]

        # 计算符合条件的行中 TBSUPPWT 的总和
        tbsuppwt_sum = df.loc[condition_total, 'TBSUPPWT'].sum()

        condition1 = condition_total & (
            (df['ECEV'] == 1) 
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
        
        condition2 = condition_total & (
            (df['ECEV'] == 1) & ((df['ECFREQ'] == 1) | (df['ECFREQ'] == 2))
        )
        samples2 = df[condition2]
        sample_count2 = samples2.shape[0]
        
        tbsuppwt2 = df.loc[condition2, 'TBSUPPWT'].sum()

        # 计算加权使用者数和加权总人数
        weighted_ever_users2 = tbsuppwt2
        weighted_total_respondents = tbsuppwt_sum

        # 计算加权比例
        p2 = weighted_ever_users2 / weighted_total_respondents

        # 计算标准误
        se2 = np.sqrt(p2 * (1 - p2) / totalsample_count)

        # 95% 置信区间
        z = 1.96  # 正态分布的z值
        ci_lower2 = p2 - z * se2
        ci_upper2 = p2 + z * se2

        # 返回结果
        return {
            "totalsample_count": totalsample_count,
            "sample_count": sample_count,
            "p1": p,
            "confidence_interval1": (ci_lower, ci_upper),
            "sample_count2": sample_count2,
            "p2": p2,
            "confidence_interval2": (ci_lower2, ci_upper2)
        }
    
    
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"文件为空: {file_path}")
    except pd.errors.ParserError:
        print(f"文件解析错误: {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        
def countmale(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # filter samples 过滤样本
        condition_total = (
            # 在这里定义你的总过滤条件
            # 例如: (df['SomeColumn'] == some_value) & (df['AnotherColumn'] > some_threshold)
            ((df['ECEV'] == 1)| (df['ECEV'] == 2))
            &(df['SEX'] == 1)# 替换为你的实际条件
        )
        totalsamples = df[condition_total]

        # count the sample after filter 计算过滤出的样本数量
        totalsample_count = totalsamples.shape[0]

        # 计算符合条件的行中 TBSUPPWT 的总和
        tbsuppwt_sum = df.loc[condition_total, 'TBSUPPWT'].sum()

        condition1 = condition_total & (
            (df['ECEV'] == 1) 
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
        
        condition2 = condition_total & (
            (df['ECEV'] == 1) & ((df['ECFREQ'] == 1) | (df['ECFREQ'] == 2))
        )
        samples2 = df[condition2]
        sample_count2 = samples2.shape[0]
        
        tbsuppwt2 = df.loc[condition2, 'TBSUPPWT'].sum()

        # 计算加权使用者数和加权总人数
        weighted_ever_users2 = tbsuppwt2
        weighted_total_respondents = tbsuppwt_sum

        # 计算加权比例
        p2 = weighted_ever_users2 / weighted_total_respondents

        # 计算标准误
        se2 = np.sqrt(p2 * (1 - p2) / totalsample_count)

        # 95% 置信区间
        z = 1.96  # 正态分布的z值
        ci_lower2 = p2 - z * se2
        ci_upper2 = p2 + z * se2

        # 返回结果
        return {
            "totalsample_count": totalsample_count,
            "sample_count": sample_count,
            "p1": p,
            "confidence_interval1": (ci_lower, ci_upper),
            "sample_count2": sample_count2,
            "p2": p2,
            "confidence_interval2": (ci_lower2, ci_upper2)
        }
    
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"文件为空: {file_path}")
    except pd.errors.ParserError:
        print(f"文件解析错误: {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

def countfemale(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # filter samples 过滤样本
        condition_total = (
            # 在这里定义你的总过滤条件
            # 例如: (df['SomeColumn'] == some_value) & (df['AnotherColumn'] > some_threshold)
            ((df['ECEV'] == 1)| (df['ECEV'] == 2))
            &(df['SEX'] == 2)# 替换为你的实际条件
        )
        totalsamples = df[condition_total]

        # count the sample after filter 计算过滤出的样本数量
        totalsample_count = totalsamples.shape[0]

        # 计算符合条件的行中 TBSUPPWT 的总和
        tbsuppwt_sum = df.loc[condition_total, 'TBSUPPWT'].sum()

        condition1 = condition_total & (
            (df['ECEV'] == 1) 
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
        
        condition2 = condition_total & (
            (df['ECEV'] == 1) & ((df['ECFREQ'] == 1) | (df['ECFREQ'] == 2))
        )
        samples2 = df[condition2]
        sample_count2 = samples2.shape[0]
        
        tbsuppwt2 = df.loc[condition2, 'TBSUPPWT'].sum()

        # 计算加权使用者数和加权总人数
        weighted_ever_users2 = tbsuppwt2
        weighted_total_respondents = tbsuppwt_sum

        # 计算加权比例
        p2 = weighted_ever_users2 / weighted_total_respondents

        # 计算标准误
        se2 = np.sqrt(p2 * (1 - p2) / totalsample_count)

        # 95% 置信区间
        z = 1.96  # 正态分布的z值
        ci_lower2 = p2 - z * se2
        ci_upper2 = p2 + z * se2

        # 返回结果
        return {
            "totalsample_count": totalsample_count,
            "sample_count": sample_count,
            "p1": p,
            "confidence_interval1": (ci_lower, ci_upper),
            "sample_count2": sample_count2,
            "p2": p2,
            "confidence_interval2": (ci_lower2, ci_upper2)
        }
    
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"文件为空: {file_path}")
    except pd.errors.ParserError:
        print(f"文件解析错误: {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

def count1824(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # filter samples 过滤样本
        condition_total = (
            # 在这里定义你的总过滤条件
            # 例如: (df['SomeColumn'] == some_value) & (df['AnotherColumn'] > some_threshold)
            ((df['ECEV'] == 1)| (df['ECEV'] == 2))
            &(df['AGE'] >= 18)&(df['AGE'] <= 24)# 替换为你的实际条件
        )
        totalsamples = df[condition_total]

        # count the sample after filter 计算过滤出的样本数量
        totalsample_count = totalsamples.shape[0]

        # 计算符合条件的行中 TBSUPPWT 的总和
        tbsuppwt_sum = df.loc[condition_total, 'TBSUPPWT'].sum()

        condition1 = condition_total & (
            (df['ECEV'] == 1) 
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
        
        condition2 = condition_total & (
            (df['ECEV'] == 1) & ((df['ECFREQ'] == 1) | (df['ECFREQ'] == 2))
        )
        samples2 = df[condition2]
        sample_count2 = samples2.shape[0]
        
        tbsuppwt2 = df.loc[condition2, 'TBSUPPWT'].sum()

        # 计算加权使用者数和加权总人数
        weighted_ever_users2 = tbsuppwt2
        weighted_total_respondents = tbsuppwt_sum

        # 计算加权比例
        p2 = weighted_ever_users2 / weighted_total_respondents

        # 计算标准误
        se2 = np.sqrt(p2 * (1 - p2) / totalsample_count)

        # 95% 置信区间
        z = 1.96  # 正态分布的z值
        ci_lower2 = p2 - z * se2
        ci_upper2 = p2 + z * se2

        # 返回结果
        return {
            "totalsample_count": totalsample_count,
            "sample_count": sample_count,
            "p1": p,
            "confidence_interval1": (ci_lower, ci_upper),
            "sample_count2": sample_count2,
            "p2": p2,
            "confidence_interval2": (ci_lower2, ci_upper2)
        }
    
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"文件为空: {file_path}")
    except pd.errors.ParserError:
        print(f"文件解析错误: {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

def count2544(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # filter samples 过滤样本
        condition_total = (
            # 在这里定义你的总过滤条件
            # 例如: (df['SomeColumn'] == some_value) & (df['AnotherColumn'] > some_threshold)
            ((df['ECEV'] == 1)| (df['ECEV'] == 2))
            &(df['AGE'] >= 25)&(df['AGE'] <= 44)# 替换为你的实际条件
        )
        totalsamples = df[condition_total]

        # count the sample after filter 计算过滤出的样本数量
        totalsample_count = totalsamples.shape[0]

        # 计算符合条件的行中 TBSUPPWT 的总和
        tbsuppwt_sum = df.loc[condition_total, 'TBSUPPWT'].sum()

        condition1 = condition_total & (
            (df['ECEV'] == 1) 
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
        
        condition2 = condition_total & (
            (df['ECEV'] == 1) & ((df['ECFREQ'] == 1) | (df['ECFREQ'] == 2))
        )
        samples2 = df[condition2]
        sample_count2 = samples2.shape[0]
        
        tbsuppwt2 = df.loc[condition2, 'TBSUPPWT'].sum()

        # 计算加权使用者数和加权总人数
        weighted_ever_users2 = tbsuppwt2
        weighted_total_respondents = tbsuppwt_sum

        # 计算加权比例
        p2 = weighted_ever_users2 / weighted_total_respondents

        # 计算标准误
        se2 = np.sqrt(p2 * (1 - p2) / totalsample_count)

        # 95% 置信区间
        z = 1.96  # 正态分布的z值
        ci_lower2 = p2 - z * se2
        ci_upper2 = p2 + z * se2

        # 返回结果
        return {
            "totalsample_count": totalsample_count,
            "sample_count": sample_count,
            "p1": p,
            "confidence_interval1": (ci_lower, ci_upper),
            "sample_count2": sample_count2,
            "p2": p2,
            "confidence_interval2": (ci_lower2, ci_upper2)
        }
    
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"文件为空: {file_path}")
    except pd.errors.ParserError:
        print(f"文件解析错误: {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
    

def count4564(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # filter samples 过滤样本
        condition_total = (
            # 在这里定义你的总过滤条件
            # 例如: (df['SomeColumn'] == some_value) & (df['AnotherColumn'] > some_threshold)
            ((df['ECEV'] == 1)| (df['ECEV'] == 2))
            &(df['AGE'] >= 45)&(df['AGE'] <= 64)# 替换为你的实际条件
        )
        totalsamples = df[condition_total]

        # count the sample after filter 计算过滤出的样本数量
        totalsample_count = totalsamples.shape[0]

        # 计算符合条件的行中 TBSUPPWT 的总和
        tbsuppwt_sum = df.loc[condition_total, 'TBSUPPWT'].sum()

        condition1 = condition_total & (
            (df['ECEV'] == 1) 
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
        
        condition2 = condition_total & (
            (df['ECEV'] == 1) & ((df['ECFREQ'] == 1) | (df['ECFREQ'] == 2))
        )
        samples2 = df[condition2]
        sample_count2 = samples2.shape[0]
        
        tbsuppwt2 = df.loc[condition2, 'TBSUPPWT'].sum()

        # 计算加权使用者数和加权总人数
        weighted_ever_users2 = tbsuppwt2
        weighted_total_respondents = tbsuppwt_sum

        # 计算加权比例
        p2 = weighted_ever_users2 / weighted_total_respondents

        # 计算标准误
        se2 = np.sqrt(p2 * (1 - p2) / totalsample_count)

        # 95% 置信区间
        z = 1.96  # 正态分布的z值
        ci_lower2 = p2 - z * se2
        ci_upper2 = p2 + z * se2

        # 返回结果
        return {
            "totalsample_count": totalsample_count,
            "sample_count": sample_count,
            "p1": p,
            "confidence_interval1": (ci_lower, ci_upper),
            "sample_count2": sample_count2,
            "p2": p2,
            "confidence_interval2": (ci_lower2, ci_upper2)
        }
    
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"文件为空: {file_path}")
    except pd.errors.ParserError:
        print(f"文件解析错误: {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

def count65(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # filter samples 过滤样本
        condition_total = (
            # 在这里定义你的总过滤条件
            # 例如: (df['SomeColumn'] == some_value) & (df['AnotherColumn'] > some_threshold)
            ((df['ECEV'] == 1)| (df['ECEV'] == 2))
            &(df['AGE'] >= 65)# 替换为你的实际条件
        )
        totalsamples = df[condition_total]

        # count the sample after filter 计算过滤出的样本数量
        totalsample_count = totalsamples.shape[0]

        # 计算符合条件的行中 TBSUPPWT 的总和
        tbsuppwt_sum = df.loc[condition_total, 'TBSUPPWT'].sum()

        condition1 = condition_total & (
            (df['ECEV'] == 1) 
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
        
        condition2 = condition_total & (
            (df['ECEV'] == 1) & ((df['ECFREQ'] == 1) | (df['ECFREQ'] == 2))
        )
        samples2 = df[condition2]
        sample_count2 = samples2.shape[0]
        
        tbsuppwt2 = df.loc[condition2, 'TBSUPPWT'].sum()

        # 计算加权使用者数和加权总人数
        weighted_ever_users2 = tbsuppwt2
        weighted_total_respondents = tbsuppwt_sum

        # 计算加权比例
        p2 = weighted_ever_users2 / weighted_total_respondents

        # 计算标准误
        se2 = np.sqrt(p2 * (1 - p2) / totalsample_count)

        # 95% 置信区间
        z = 1.96  # 正态分布的z值
        ci_lower2 = p2 - z * se2
        ci_upper2 = p2 + z * se2

        # 返回结果
        return {
            "totalsample_count": totalsample_count,
            "sample_count": sample_count,
            "p1": p,
            "confidence_interval1": (ci_lower, ci_upper),
            "sample_count2": sample_count2,
            "p2": p2,
            "confidence_interval2": (ci_lower2, ci_upper2)
        }
    
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"文件为空: {file_path}")
    except pd.errors.ParserError:
        print(f"文件解析错误: {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

def countwhite(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # filter samples 过滤样本
        condition_total = (
            # 在这里定义你的总过滤条件
            # 例如: (df['SomeColumn'] == some_value) & (df['AnotherColumn'] > some_threshold)
            ((df['ECEV'] == 1)| (df['ECEV'] == 2))
            &(df['RACE']==100)&(df['HISPAN']==0)# 替换为你的实际条件
        )
        totalsamples = df[condition_total]

        # count the sample after filter 计算过滤出的样本数量
        totalsample_count = totalsamples.shape[0]

        # 计算符合条件的行中 TBSUPPWT 的总和
        tbsuppwt_sum = df.loc[condition_total, 'TBSUPPWT'].sum()

        condition1 = condition_total & (
            (df['ECEV'] == 1) 
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
        
        condition2 = condition_total & (
            (df['ECEV'] == 1) & ((df['ECFREQ'] == 1) | (df['ECFREQ'] == 2))
        )
        samples2 = df[condition2]
        sample_count2 = samples2.shape[0]
        
        tbsuppwt2 = df.loc[condition2, 'TBSUPPWT'].sum()

        # 计算加权使用者数和加权总人数
        weighted_ever_users2 = tbsuppwt2
        weighted_total_respondents = tbsuppwt_sum

        # 计算加权比例
        p2 = weighted_ever_users2 / weighted_total_respondents

        # 计算标准误
        se2 = np.sqrt(p2 * (1 - p2) / totalsample_count)

        # 95% 置信区间
        z = 1.96  # 正态分布的z值
        ci_lower2 = p2 - z * se2
        ci_upper2 = p2 + z * se2

        # 返回结果
        return {
            "totalsample_count": totalsample_count,
            "sample_count": sample_count,
            "p1": p,
            "confidence_interval1": (ci_lower, ci_upper),
            "sample_count2": sample_count2,
            "p2": p2,
            "confidence_interval2": (ci_lower2, ci_upper2)
        }
    
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"文件为空: {file_path}")
    except pd.errors.ParserError:
        print(f"文件解析错误: {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

def counthispan(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # filter samples 过滤样本
        condition_total = (
            # 在这里定义你的总过滤条件
            # 例如: (df['SomeColumn'] == some_value) & (df['AnotherColumn'] > some_threshold)
            ((df['ECEV'] == 1)| (df['ECEV'] == 2))
            &(df['HISPAN']!=0)# 替换为你的实际条件
        )
        totalsamples = df[condition_total]

        # count the sample after filter 计算过滤出的样本数量
        totalsample_count = totalsamples.shape[0]

        # 计算符合条件的行中 TBSUPPWT 的总和
        tbsuppwt_sum = df.loc[condition_total, 'TBSUPPWT'].sum()

        condition1 = condition_total & (
            (df['ECEV'] == 1) 
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
        
        condition2 = condition_total & (
            (df['ECEV'] == 1) & ((df['ECFREQ'] == 1) | (df['ECFREQ'] == 2))
        )
        samples2 = df[condition2]
        sample_count2 = samples2.shape[0]
        
        tbsuppwt2 = df.loc[condition2, 'TBSUPPWT'].sum()

        # 计算加权使用者数和加权总人数
        weighted_ever_users2 = tbsuppwt2
        weighted_total_respondents = tbsuppwt_sum

        # 计算加权比例
        p2 = weighted_ever_users2 / weighted_total_respondents

        # 计算标准误
        se2 = np.sqrt(p2 * (1 - p2) / totalsample_count)

        # 95% 置信区间
        z = 1.96  # 正态分布的z值
        ci_lower2 = p2 - z * se2
        ci_upper2 = p2 + z * se2

        # 返回结果
        return {
            "totalsample_count": totalsample_count,
            "sample_count": sample_count,
            "p1": p,
            "confidence_interval1": (ci_lower, ci_upper),
            "sample_count2": sample_count2,
            "p2": p2,
            "confidence_interval2": (ci_lower2, ci_upper2)
        }
    
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"文件为空: {file_path}")
    except pd.errors.ParserError:
        print(f"文件解析错误: {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

def countblack(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # filter samples 过滤样本
        condition_total = (
            # 在这里定义你的总过滤条件
            # 例如: (df['SomeColumn'] == some_value) & (df['AnotherColumn'] > some_threshold)
            ((df['ECEV'] == 1)| (df['ECEV'] == 2))
            &(df['RACE']==200)&(df['HISPAN']==0)# 替换为你的实际条件
        )
        totalsamples = df[condition_total]

        # count the sample after filter 计算过滤出的样本数量
        totalsample_count = totalsamples.shape[0]

        # 计算符合条件的行中 TBSUPPWT 的总和
        tbsuppwt_sum = df.loc[condition_total, 'TBSUPPWT'].sum()

        condition1 = condition_total & (
            (df['ECEV'] == 1) 
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
        
        condition2 = condition_total & (
            (df['ECEV'] == 1) & ((df['ECFREQ'] == 1) | (df['ECFREQ'] == 2))
        )
        samples2 = df[condition2]
        sample_count2 = samples2.shape[0]
        
        tbsuppwt2 = df.loc[condition2, 'TBSUPPWT'].sum()

        # 计算加权使用者数和加权总人数
        weighted_ever_users2 = tbsuppwt2
        weighted_total_respondents = tbsuppwt_sum

        # 计算加权比例
        p2 = weighted_ever_users2 / weighted_total_respondents

        # 计算标准误
        se2 = np.sqrt(p2 * (1 - p2) / totalsample_count)

        # 95% 置信区间
        z = 1.96  # 正态分布的z值
        ci_lower2 = p2 - z * se2
        ci_upper2 = p2 + z * se2

        # 返回结果
        return {
            "totalsample_count": totalsample_count,
            "sample_count": sample_count,
            "p1": p,
            "confidence_interval1": (ci_lower, ci_upper),
            "sample_count2": sample_count2,
            "p2": p2,
            "confidence_interval2": (ci_lower2, ci_upper2)
        }
    
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"文件为空: {file_path}")
    except pd.errors.ParserError:
        print(f"文件解析错误: {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

def countasian(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # filter samples 过滤样本
        condition_total = (
            # 在这里定义你的总过滤条件
            # 例如: (df['SomeColumn'] == some_value) & (df['AnotherColumn'] > some_threshold)
            ((df['ECEV'] == 1)| (df['ECEV'] == 2))
            &(df['RACE']==651)&(df['HISPAN']==0)# 替换为你的实际条件
        )
        totalsamples = df[condition_total]

        # count the sample after filter 计算过滤出的样本数量
        totalsample_count = totalsamples.shape[0]

        # 计算符合条件的行中 TBSUPPWT 的总和
        tbsuppwt_sum = df.loc[condition_total, 'TBSUPPWT'].sum()

        condition1 = condition_total & (
            (df['ECEV'] == 1) 
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
        
        condition2 = condition_total & (
            (df['ECEV'] == 1) & ((df['ECFREQ'] == 1) | (df['ECFREQ'] == 2))
        )
        samples2 = df[condition2]
        sample_count2 = samples2.shape[0]
        
        tbsuppwt2 = df.loc[condition2, 'TBSUPPWT'].sum()

        # 计算加权使用者数和加权总人数
        weighted_ever_users2 = tbsuppwt2
        weighted_total_respondents = tbsuppwt_sum

        # 计算加权比例
        p2 = weighted_ever_users2 / weighted_total_respondents

        # 计算标准误
        se2 = np.sqrt(p2 * (1 - p2) / totalsample_count)

        # 95% 置信区间
        z = 1.96  # 正态分布的z值
        ci_lower2 = p2 - z * se2
        ci_upper2 = p2 + z * se2

        # 返回结果
        return {
            "totalsample_count": totalsample_count,
            "sample_count": sample_count,
            "p1": p,
            "confidence_interval1": (ci_lower, ci_upper),
            "sample_count2": sample_count2,
            "p2": p2,
            "confidence_interval2": (ci_lower2, ci_upper2)
        }
    
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"文件为空: {file_path}")
    except pd.errors.ParserError:
        print(f"文件解析错误: {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

def countamindinan(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # filter samples 过滤样本
        condition_total = (
            # 在这里定义你的总过滤条件
            # 例如: (df['SomeColumn'] == some_value) & (df['AnotherColumn'] > some_threshold)
            ((df['ECEV'] == 1)| (df['ECEV'] == 2))
            &(df['RACE']==300)&(df['HISPAN']==0)# 替换为你的实际条件
        )
        totalsamples = df[condition_total]

        # count the sample after filter 计算过滤出的样本数量
        totalsample_count = totalsamples.shape[0]

        # 计算符合条件的行中 TBSUPPWT 的总和
        tbsuppwt_sum = df.loc[condition_total, 'TBSUPPWT'].sum()

        condition1 = condition_total & (
            (df['ECEV'] == 1) 
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
        
        condition2 = condition_total & (
            (df['ECEV'] == 1) & ((df['ECFREQ'] == 1) | (df['ECFREQ'] == 2))
        )
        samples2 = df[condition2]
        sample_count2 = samples2.shape[0]
        
        tbsuppwt2 = df.loc[condition2, 'TBSUPPWT'].sum()

        # 计算加权使用者数和加权总人数
        weighted_ever_users2 = tbsuppwt2
        weighted_total_respondents = tbsuppwt_sum

        # 计算加权比例
        p2 = weighted_ever_users2 / weighted_total_respondents

        # 计算标准误
        se2 = np.sqrt(p2 * (1 - p2) / totalsample_count)

        # 95% 置信区间
        z = 1.96  # 正态分布的z值
        ci_lower2 = p2 - z * se2
        ci_upper2 = p2 + z * se2

        # 返回结果
        return {
            "totalsample_count": totalsample_count,
            "sample_count": sample_count,
            "p1": p,
            "confidence_interval1": (ci_lower, ci_upper),
            "sample_count2": sample_count2,
            "p2": p2,
            "confidence_interval2": (ci_lower2, ci_upper2)
        }
    
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"文件为空: {file_path}")
    except pd.errors.ParserError:
        print(f"文件解析错误: {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

def countpacific(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # filter samples 过滤样本
        condition_total = (
            # 在这里定义你的总过滤条件
            # 例如: (df['SomeColumn'] == some_value) & (df['AnotherColumn'] > some_threshold)
            ((df['ECEV'] == 1)| (df['ECEV'] == 2))
            &(df['RACE']==652)&(df['HISPAN']==0)# 替换为你的实际条件
        )
        totalsamples = df[condition_total]

        # count the sample after filter 计算过滤出的样本数量
        totalsample_count = totalsamples.shape[0]

        # 计算符合条件的行中 TBSUPPWT 的总和
        tbsuppwt_sum = df.loc[condition_total, 'TBSUPPWT'].sum()

        condition1 = condition_total & (
            (df['ECEV'] == 1) 
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
        
        condition2 = condition_total & (
            (df['ECEV'] == 1) & ((df['ECFREQ'] == 1) | (df['ECFREQ'] == 2))
        )
        samples2 = df[condition2]
        sample_count2 = samples2.shape[0]
        
        tbsuppwt2 = df.loc[condition2, 'TBSUPPWT'].sum()

        # 计算加权使用者数和加权总人数
        weighted_ever_users2 = tbsuppwt2
        weighted_total_respondents = tbsuppwt_sum

        # 计算加权比例
        p2 = weighted_ever_users2 / weighted_total_respondents

        # 计算标准误
        se2 = np.sqrt(p2 * (1 - p2) / totalsample_count)

        # 95% 置信区间
        z = 1.96  # 正态分布的z值
        ci_lower2 = p2 - z * se2
        ci_upper2 = p2 + z * se2

        # 返回结果
        return {
            "totalsample_count": totalsample_count,
            "sample_count": sample_count,
            "p1": p,
            "confidence_interval1": (ci_lower, ci_upper),
            "sample_count2": sample_count2,
            "p2": p2,
            "confidence_interval2": (ci_lower2, ci_upper2)
        }
    
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"文件为空: {file_path}")
    except pd.errors.ParserError:
        print(f"文件解析错误: {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

def countother(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # filter samples 过滤样本
        condition_total = (
            # 在这里定义你的总过滤条件
            # 例如: (df['SomeColumn'] == some_value) & (df['AnotherColumn'] > some_threshold)
            ((df['ECEV'] == 1)| (df['ECEV'] == 2))
            &((df['RACE']!=652)&(df['RACE']!=651)&(df['RACE']!=100)&(df['RACE']!=200)&(df['RACE']!=300))
            &(df['HISPAN']==0)# 替换为你的实际条件
        )
        totalsamples = df[condition_total]

        # count the sample after filter 计算过滤出的样本数量
        totalsample_count = totalsamples.shape[0]

        # 计算符合条件的行中 TBSUPPWT 的总和
        tbsuppwt_sum = df.loc[condition_total, 'TBSUPPWT'].sum()

        condition1 = condition_total & (
            (df['ECEV'] == 1) 
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
        
        condition2 = condition_total & (
            (df['ECEV'] == 1) & ((df['ECFREQ'] == 1) | (df['ECFREQ'] == 2))
        )
        samples2 = df[condition2]
        sample_count2 = samples2.shape[0]
        
        tbsuppwt2 = df.loc[condition2, 'TBSUPPWT'].sum()

        # 计算加权使用者数和加权总人数
        weighted_ever_users2 = tbsuppwt2
        weighted_total_respondents = tbsuppwt_sum

        # 计算加权比例
        p2 = weighted_ever_users2 / weighted_total_respondents

        # 计算标准误
        se2 = np.sqrt(p2 * (1 - p2) / totalsample_count)

        # 95% 置信区间
        z = 1.96  # 正态分布的z值
        ci_lower2 = p2 - z * se2
        ci_upper2 = p2 + z * se2

        # 返回结果
        return {
            "totalsample_count": totalsample_count,
            "sample_count": sample_count,
            "p1": p,
            "confidence_interval1": (ci_lower, ci_upper),
            "sample_count2": sample_count2,
            "p2": p2,
            "confidence_interval2": (ci_lower2, ci_upper2)
        }
    
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"文件为空: {file_path}")
    except pd.errors.ParserError:
        print(f"文件解析错误: {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

def counteduless12(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # filter samples 过滤样本
        condition_total = (
            # 在这里定义你的总过滤条件
            # 例如: (df['SomeColumn'] == some_value) & (df['AnotherColumn'] > some_threshold)
            ((df['ECEV'] == 1)| (df['ECEV'] == 2))
            &((df['EDUC']==10)|(df['EDUC']==20)|(df['EDUC']==30)|(df['EDUC']==1)|(df['EDUC']==2)|(df['EDUC']==40)|(df['EDUC']==50)|(df['EDUC']==60))# 替换为你的实际条件
        )
        totalsamples = df[condition_total]

        # count the sample after filter 计算过滤出的样本数量
        totalsample_count = totalsamples.shape[0]

        # 计算符合条件的行中 TBSUPPWT 的总和
        tbsuppwt_sum = df.loc[condition_total, 'TBSUPPWT'].sum()

        condition1 = condition_total & (
            (df['ECEV'] == 1) 
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
        
        condition2 = condition_total & (
            (df['ECEV'] == 1) & ((df['ECFREQ'] == 1) | (df['ECFREQ'] == 2))
        )
        samples2 = df[condition2]
        sample_count2 = samples2.shape[0]
        
        tbsuppwt2 = df.loc[condition2, 'TBSUPPWT'].sum()

        # 计算加权使用者数和加权总人数
        weighted_ever_users2 = tbsuppwt2
        weighted_total_respondents = tbsuppwt_sum

        # 计算加权比例
        p2 = weighted_ever_users2 / weighted_total_respondents

        # 计算标准误
        se2 = np.sqrt(p2 * (1 - p2) / totalsample_count)

        # 95% 置信区间
        z = 1.96  # 正态分布的z值
        ci_lower2 = p2 - z * se2
        ci_upper2 = p2 + z * se2

        # 返回结果
        return {
            "totalsample_count": totalsample_count,
            "sample_count": sample_count,
            "p1": p,
            "confidence_interval1": (ci_lower, ci_upper),
            "sample_count2": sample_count2,
            "p2": p2,
            "confidence_interval2": (ci_lower2, ci_upper2)
        }
    
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"文件为空: {file_path}")
    except pd.errors.ParserError:
        print(f"文件解析错误: {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

def countedu12ged(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # filter samples 过滤样本
        condition_total = (
            # 在这里定义你的总过滤条件
            # 例如: (df['SomeColumn'] == some_value) & (df['AnotherColumn'] > some_threshold)
            ((df['ECEV'] == 1)| (df['ECEV'] == 2))
            &((df['EDUC']==71)|(df['EDUC']==73))# 替换为你的实际条件
        )
        totalsamples = df[condition_total]

        # count the sample after filter 计算过滤出的样本数量
        totalsample_count = totalsamples.shape[0]

        # 计算符合条件的行中 TBSUPPWT 的总和
        tbsuppwt_sum = df.loc[condition_total, 'TBSUPPWT'].sum()

        condition1 = condition_total & (
            (df['ECEV'] == 1) 
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
        
        condition2 = condition_total & (
            (df['ECEV'] == 1) & ((df['ECFREQ'] == 1) | (df['ECFREQ'] == 2))
        )
        samples2 = df[condition2]
        sample_count2 = samples2.shape[0]
        
        tbsuppwt2 = df.loc[condition2, 'TBSUPPWT'].sum()

        # 计算加权使用者数和加权总人数
        weighted_ever_users2 = tbsuppwt2
        weighted_total_respondents = tbsuppwt_sum

        # 计算加权比例
        p2 = weighted_ever_users2 / weighted_total_respondents

        # 计算标准误
        se2 = np.sqrt(p2 * (1 - p2) / totalsample_count)

        # 95% 置信区间
        z = 1.96  # 正态分布的z值
        ci_lower2 = p2 - z * se2
        ci_upper2 = p2 + z * se2

        # 返回结果
        return {
            "totalsample_count": totalsample_count,
            "sample_count": sample_count,
            "p1": p,
            "confidence_interval1": (ci_lower, ci_upper),
            "sample_count2": sample_count2,
            "p2": p2,
            "confidence_interval2": (ci_lower2, ci_upper2)
        }
    
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"文件为空: {file_path}")
    except pd.errors.ParserError:
        print(f"文件解析错误: {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

def countedusomecollege(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # filter samples 过滤样本
        condition_total = (
            # 在这里定义你的总过滤条件
            # 例如: (df['SomeColumn'] == some_value) & (df['AnotherColumn'] > some_threshold)
            ((df['ECEV'] == 1)| (df['ECEV'] == 2))
            &((df['EDUC']==81)|(df['EDUC']==91)|(df['EDUC']==92))# 替换为你的实际条件
        )
        totalsamples = df[condition_total]

        # count the sample after filter 计算过滤出的样本数量
        totalsample_count = totalsamples.shape[0]

        # 计算符合条件的行中 TBSUPPWT 的总和
        tbsuppwt_sum = df.loc[condition_total, 'TBSUPPWT'].sum()

        condition1 = condition_total & (
            (df['ECEV'] == 1) 
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
        
        condition2 = condition_total & (
            (df['ECEV'] == 1) & ((df['ECFREQ'] == 1) | (df['ECFREQ'] == 2))
        )
        samples2 = df[condition2]
        sample_count2 = samples2.shape[0]
        
        tbsuppwt2 = df.loc[condition2, 'TBSUPPWT'].sum()

        # 计算加权使用者数和加权总人数
        weighted_ever_users2 = tbsuppwt2
        weighted_total_respondents = tbsuppwt_sum

        # 计算加权比例
        p2 = weighted_ever_users2 / weighted_total_respondents

        # 计算标准误
        se2 = np.sqrt(p2 * (1 - p2) / totalsample_count)

        # 95% 置信区间
        z = 1.96  # 正态分布的z值
        ci_lower2 = p2 - z * se2
        ci_upper2 = p2 + z * se2

        # 返回结果
        return {
            "totalsample_count": totalsample_count,
            "sample_count": sample_count,
            "p1": p,
            "confidence_interval1": (ci_lower, ci_upper),
            "sample_count2": sample_count2,
            "p2": p2,
            "confidence_interval2": (ci_lower2, ci_upper2)
        }
    
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"文件为空: {file_path}")
    except pd.errors.ParserError:
        print(f"文件解析错误: {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

def countedubdorhigh(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # filter samples 过滤样本
        condition_total = (
            # 在这里定义你的总过滤条件
            # 例如: (df['SomeColumn'] == some_value) & (df['AnotherColumn'] > some_threshold)
            ((df['ECEV'] == 1)| (df['ECEV'] == 2))
            &((df['EDUC']==111)|(df['EDUC']==123)|(df['EDUC']==124)|(df['EDUC']==125))# 替换为你的实际条件
        )
        totalsamples = df[condition_total]

        # count the sample after filter 计算过滤出的样本数量
        totalsample_count = totalsamples.shape[0]

        # 计算符合条件的行中 TBSUPPWT 的总和
        tbsuppwt_sum = df.loc[condition_total, 'TBSUPPWT'].sum()

        condition1 = condition_total & (
            (df['ECEV'] == 1) 
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
        
        condition2 = condition_total & (
            (df['ECEV'] == 1) & ((df['ECFREQ'] == 1) | (df['ECFREQ'] == 2))
        )
        samples2 = df[condition2]
        sample_count2 = samples2.shape[0]
        
        tbsuppwt2 = df.loc[condition2, 'TBSUPPWT'].sum()

        # 计算加权使用者数和加权总人数
        weighted_ever_users2 = tbsuppwt2
        weighted_total_respondents = tbsuppwt_sum

        # 计算加权比例
        p2 = weighted_ever_users2 / weighted_total_respondents

        # 计算标准误
        se2 = np.sqrt(p2 * (1 - p2) / totalsample_count)

        # 95% 置信区间
        z = 1.96  # 正态分布的z值
        ci_lower2 = p2 - z * se2
        ci_upper2 = p2 + z * se2

        # 返回结果
        return {
            "totalsample_count": totalsample_count,
            "sample_count": sample_count,
            "p1": p,
            "confidence_interval1": (ci_lower, ci_upper),
            "sample_count2": sample_count2,
            "p2": p2,
            "confidence_interval2": (ci_lower2, ci_upper2)
        }
    
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"文件为空: {file_path}")
    except pd.errors.ParserError:
        print(f"文件解析错误: {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        


# 示例调用
def write_results_to_file(csv_file_path, file_path):
    results = [
    counttotal(csv_file_path),
    countmale(csv_file_path),
    countfemale(csv_file_path),
    count1824(csv_file_path),
    count2544(csv_file_path),
    count4564(csv_file_path),
    count65(csv_file_path),
    countwhite(csv_file_path),
    counthispan(csv_file_path),
    countblack(csv_file_path),
    countasian(csv_file_path),
    countamindinan(csv_file_path),
    countpacific(csv_file_path),
    countother(csv_file_path),
    counteduless12(csv_file_path),
    countedu12ged(csv_file_path),
    countedusomecollege(csv_file_path),
    countedubdorhigh(csv_file_path),
    ]
    with open(file_path, 'w') as file:
        for result in results:
            file.write(f"{result['totalsample_count']} "
            f"{result['sample_count']} "
            f"{result['p1'] * 100}[{result['confidence_interval1'][0] * 100}, {result['confidence_interval1'][1] * 100}] "
            f"{result['sample_count2']} "
            f"{result['p2'] * 100}[{result['confidence_interval2'][0] * 100}, {result['confidence_interval2'][1] * 100}]\n")

            
    
csv_file_path = '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/cps_00006self.csv'
file_path='/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/a.txt'
write_results_to_file(csv_file_path, file_path)