import matplotlib.pyplot as plt
import numpy as np
import re
import scipy.stats as stats

def extract_data(file_path, line_numbers):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line_num in line_numbers:
            line = lines[line_num - 1].strip()
            try:
                percentage_match = re.search(r'(\d+\.\d+)\[', line)
                conf_interval_match = re.search(r'\[(\d+\.\d+),\s*(\d+\.\d+)\]', line)
                if percentage_match and conf_interval_match:
                    percentage = float(percentage_match.group(1))
                    conf_interval = [float(conf_interval_match.group(1)), float(conf_interval_match.group(2))]
                    data.append((percentage, conf_interval))
                else:
                    print(f"Invalid format in line {line_num} of file {file_path}")
            except IndexError as e:
                print(f"Error processing line {line_num} of file {file_path}: {e}")
    return data

def average_data(data_list):
    avg_percentages = []
    avg_conf_intervals = []
    for i in range(len(data_list[0])):
        percentages = [data[i][0] for data in data_list]
        conf_intervals = [data[i][1] for data in data_list]
        avg_percentage = np.mean(percentages)
        avg_conf_interval = [np.mean([ci[0] for ci in conf_intervals]), np.mean([ci[1] for ci in conf_intervals])]
        avg_percentages.append(avg_percentage)
        avg_conf_intervals.append(avg_conf_interval)
    return avg_percentages, avg_conf_intervals

def compute_max_conf_interval(datalist):
    
    # 计算样本均值
    mean = np.mean(datalist) 
    # 计算样本标准差
    std_dev = np.std(datalist, ddof=1)  # 使用ddof=1来计算样本标准差
    # 样本大小
    n = len(datalist)
    # 计算t值
    t_value = 1.96
    # 计算置信区间
    margin_of_error = t_value * (std_dev / np.sqrt(n))
    max_upper_conf_interval = (mean - margin_of_error, mean + margin_of_error)
    max_upper=mean
    
    print("Max Upper Confidence Interval:", max_upper_conf_interval)
    print("upper_bounds:", datalist)
    return max_upper, max_upper_conf_interval

def plot_data(orig_data, avg_data_list, file_labels, x_interval,exact_numbers_list,lower_numbers_list,upper_numbers_list):
    plt.figure(figsize=(14, 7))
    x_values = np.arange(1, len(orig_data) + 1)

    orig_percentages = [data[0] for data in orig_data]
    orig_conf_intervals = [data[1] for data in orig_data]
    orig_lower_bounds = [ci[0] for ci in orig_conf_intervals]
    orig_upper_bounds = [ci[1] for ci in orig_conf_intervals]

    plt.errorbar(x_values, orig_percentages, yerr=[np.array(orig_percentages) - np.array(orig_lower_bounds), np.array(orig_upper_bounds) - np.array(orig_percentages)], fmt='o', label='Original')

    offset = len(orig_percentages)
    avg_x_values = []
    for i, avg_data in enumerate(avg_data_list):
        avg_percentages = avg_data[0]
        avg_conf_intervals = avg_data[1]
        avg_lower_bounds = [ci[0] for ci in avg_conf_intervals]
        avg_upper_bounds = [ci[1] for ci in avg_conf_intervals]

        current_x_values = np.arange(offset + 1, offset + len(avg_percentages) + 1)
        avg_x_values.extend(current_x_values)
        #plt.errorbar(current_x_values, avg_percentages, yerr=[np.array(avg_percentages) - np.array(avg_lower_bounds), np.array(avg_upper_bounds) - np.array(avg_percentages)], fmt='o', label=file_labels[i])
        exact, exact_conf_interval = compute_max_conf_interval(exact_numbers_list[i])
        plt.errorbar(current_x_values, avg_percentages, yerr=[[exact - exact_conf_interval[0]], [exact_conf_interval[1] - exact]], fmt='o', label=file_labels[i])
  
        #print ("kanzheli",avg_x_values)
        #for x, lower, upper in zip(current_x_values, avg_lower_bounds, avg_upper_bounds):
            #plt.plot(x + 0.2, lower, 'o', mfc='none', mec='blue')  # Hollow point for the lowest confidence interval
            #plt.plot(x + 0.2, upper, 'o', mfc='none', mec='red')   # Hollow point for the highest confidence interval

        max_upper, max_upper_conf_interval = compute_max_conf_interval(upper_numbers_list[i])
        plt.errorbar([current_x_values[-1] + 0.2], [max_upper], yerr=[[max_upper - max_upper_conf_interval[0]], [max_upper_conf_interval[1] - max_upper]], fmt='o', mfc='none', mec='red', ecolor='red',label='Upper bound 95% CI')
        min_upper, min_upper_conf_interval = compute_max_conf_interval(lower_numbers_list[i])
        plt.errorbar([current_x_values[-1] + 0.3], [min_upper], yerr=[[min_upper - min_upper_conf_interval[0]], [min_upper_conf_interval[1] - min_upper]], fmt='o', mfc='none', mec='blue', ecolor='blue',label='Lower bound 95% CI')

        offset += len(avg_percentages)

    ticks = np.concatenate((x_values, avg_x_values))
    labels = ['orig'] + file_labels
    plt.xticks(ticks=ticks[:len(labels)], labels=labels, rotation=45, ha='right')

    plt.xlabel('Files')
    plt.ylabel('Percentage %')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Percentage and Confidence Interval Comparison')
    plt.tight_layout()
    plt.show()

def main():
    line_numbers = [1]  # 替换为实际的行号start from  1
    x_interval = 10  # 替换为实际的x数量

    files = [
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_orig.txt',
        
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps1_randomseed111_gen161725_beta0.3_theta4_thr20_seed21.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps1_randomseed111_gen161725_beta0.3_theta4_thr20_seed24.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps1_randomseed111_gen161725_beta0.3_theta4_thr20_seed27.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps1_randomseed111_gen161725_beta0.3_theta4_thr20_seed40.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps1_randomseed111_gen161725_beta0.3_theta4_thr20_seed50.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps1_randomseed111_gen161725_beta0.3_theta4_thr20_seed53.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps1_randomseed111_gen161725_beta0.3_theta4_thr20_seed63.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps1_randomseed111_gen161725_beta0.3_theta4_thr20_seed78.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps1_randomseed111_gen161725_beta0.3_theta4_thr20_seed80.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps1_randomseed111_gen161725_beta0.3_theta4_thr20_seed89.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps2_randomseed111_gen161725_beta0.3_theta4_thr20_seed21.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps2_randomseed111_gen161725_beta0.3_theta4_thr20_seed24.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps2_randomseed111_gen161725_beta0.3_theta4_thr20_seed27.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps2_randomseed111_gen161725_beta0.3_theta4_thr20_seed40.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps2_randomseed111_gen161725_beta0.3_theta4_thr20_seed50.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps2_randomseed111_gen161725_beta0.3_theta4_thr20_seed53.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps2_randomseed111_gen161725_beta0.3_theta4_thr20_seed63.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps2_randomseed111_gen161725_beta0.3_theta4_thr20_seed78.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps2_randomseed111_gen161725_beta0.3_theta4_thr20_seed80.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps2_randomseed111_gen161725_beta0.3_theta4_thr20_seed89.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps10_randomseed111_gen161725_beta0.3_theta4_thr20_seed21.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps10_randomseed111_gen161725_beta0.3_theta4_thr20_seed24.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps10_randomseed111_gen161725_beta0.3_theta4_thr20_seed27.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps10_randomseed111_gen161725_beta0.3_theta4_thr20_seed40.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps10_randomseed111_gen161725_beta0.3_theta4_thr20_seed50.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps10_randomseed111_gen161725_beta0.3_theta4_thr20_seed53.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps10_randomseed111_gen161725_beta0.3_theta4_thr20_seed63.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps10_randomseed111_gen161725_beta0.3_theta4_thr20_seed78.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps10_randomseed111_gen161725_beta0.3_theta4_thr20_seed80.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps10_randomseed111_gen161725_beta0.3_theta4_thr20_seed89.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps100_randomseed111_gen161725_beta0.3_theta4_thr20_seed21.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps100_randomseed111_gen161725_beta0.3_theta4_thr20_seed24.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps100_randomseed111_gen161725_beta0.3_theta4_thr20_seed27.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps100_randomseed111_gen161725_beta0.3_theta4_thr20_seed40.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps100_randomseed111_gen161725_beta0.3_theta4_thr20_seed50.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps100_randomseed111_gen161725_beta0.3_theta4_thr20_seed53.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps100_randomseed111_gen161725_beta0.3_theta4_thr20_seed63.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps100_randomseed111_gen161725_beta0.3_theta4_thr20_seed78.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps100_randomseed111_gen161725_beta0.3_theta4_thr20_seed80.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg0_eps100_randomseed111_gen161725_beta0.3_theta4_thr20_seed89.txt',
        
        

        

    ]

    orig_data = extract_data(files[0], line_numbers)

    avg_data_list = []
    file_labels = []
    exact_numbers_list=[]
    upper_numbers_list=[]
    lower_numbers_list=[]
    for i in range(1, len(files), x_interval):
        interval_data = [extract_data(files[j], line_numbers) for j in range(i, min(i + x_interval, len(files)))]
        #print ("interval_data:", interval_data)
        exact_numbers= [item[0][0] for item in interval_data]
        upper_numbers = [item[0][1][1] for item in interval_data]
        lower_numbers = [item[0][1][0] for item in interval_data]
        #print ("upper_numbers:", upper_numbers)
        if interval_data:
            avg_data = average_data(interval_data)
            avg_data_list.append(avg_data)
            exact_numbers_list.append(exact_numbers)
            upper_numbers_list.append(upper_numbers)
            lower_numbers_list.append(lower_numbers)
            label = f'deg{files[i].split("_deg")[1].split("_eps")[0]}_eps{files[i].split("_eps")[1].split("_")[0]}'
            file_labels.append(label)

    plot_data(orig_data, avg_data_list, file_labels, x_interval,exact_numbers_list,lower_numbers_list,upper_numbers_list)

if __name__ == "__main__":
    main()
