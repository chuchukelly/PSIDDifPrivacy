import matplotlib.pyplot as plt
import numpy as np
import re

def parse_deg_eps(filename):
    # 使用正则表达式从文件名中提取 deg 和 eps 的值
    deg_match = re.search(r'deg(\d+)', filename)
    eps_match = re.search(r'eps(\d+)', filename)
    deg = deg_match.group(1) if deg_match else 'orig'
    eps = eps_match.group(1) if eps_match else 'orig'
    return deg, eps

def extract_data_from_file(file_path):
    values = []
    errors = []
    with open(file_path, 'r') as file:
        for line in file:
            matches = re.findall(r'(\d+\.\d+)\[(\d+\.\d+),\s*(\d+\.\d+)\]', line)
            if matches:
                for value, lower, upper in matches:
                    values.append(float(value))
                    errors.append((float(upper) - float(lower)) / 2)
            else:
                print(f"No matches found in line: {line}")
    return values, errors

def plot_combined_bar_with_error(files):
    all_values = []
    all_errors = []
    group_labels = []
    legends = ['Never used E-c', 'Stopped use E-c', 'Current used E-c']
    group_width = 3  # 每组3个柱形
    group_gap = 1  # 组之间的间隔

    for file_index, file_path in enumerate(files):
        values, errors = extract_data_from_file(file_path)
        if values and errors:
            all_values.append(values)
            all_errors.append(errors)
            #group_labels.append(f'File {file_index + 1}')

            # 提取并记录 deg 和 eps
            deg, eps = parse_deg_eps(file_path)
            if deg== 'orig':
                group_labels.append('orig')
            else:
                group_labels.append(f'deg{deg}, eps{eps}')
            #legends.append(f'deg{deg}, eps{eps}')

    if not all_values:
        print("No valid data found.")
        return

    # 转换为numpy数组以便处理
    all_values = np.array(all_values)
    all_errors = np.array(all_errors)

    # 生成x轴位置
    x = np.arange(len(files) * group_width + (len(files) - 1) * group_gap)
    x_positions = []
    for i in range(len(files)):
        start = i * (group_width + group_gap)
        x_positions.extend([start, start + 1, start + 2])
    x_positions = np.array(x_positions)

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(14, 8))

    for i in range(group_width):
        ax.bar(x_positions[i::group_width], all_values[:, i], width=0.8, yerr=all_errors[:, i], capsize=5, label=legends[i])

    ax.set_xlabel('Groups')
    ax.set_ylabel('Percentages %')
    ax.set_title('Quit attempt')
    ax.legend(loc='upper right')

    # 修正xticks的数量和标签
    plt.xticks(x_positions[1::group_width], group_labels, rotation=20)
    plt.tight_layout()
    plt.show()

def main():
    files = [
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_orig_2.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg2_eps10_randomseed123_gen161725_beta0.3_theta4_thr20_seed6_2.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg2_eps10_randomseed123_gen161725_beta0.3_theta4_thr20_seed11_2.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg2_eps10_randomseed123_gen161725_beta0.3_theta4_thr20_seed34_2.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg2_eps10_randomseed123_gen161725_beta0.3_theta4_thr20_seed52_2.txt',
        '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg2_eps10_randomseed123_gen161725_beta0.3_theta4_thr20_seed98_2.txt'
    ]
    plot_combined_bar_with_error(files)

if __name__ == "__main__":
    main()
