
from datasynthesis import midoutput
from auto import write_results_to_file
from auto2 import write_results_to_file2

if __name__ == '__main__':
    input_data = '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/cps_00006self copy.csv'
    num_tuples_to_generate = 161725

    file_path = '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_orig.txt'
    write_results_to_file(input_data, file_path)
    file_path = '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_orig_2.txt'
    write_results_to_file2(input_data, file_path)

    # define (degree, epsilon) you want
    combinations = [
        #(0, 1),
        #(0, 2),
        #(0, 10),
        #(1, 1),
        (1, 2),
        (1, 10),
        (2, 1),
        (2, 2),
        (2, 10)
    ]
    randomseed=111 #自定义
    # 依次运行每个组合
    for degree, epsilon in combinations:
        print(f"Running midoutput for degree={degree} and epsilon={epsilon}")
        midoutput(input_data, degree, epsilon, num_tuples_to_generate,randomseed)
