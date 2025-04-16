from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network
from auto import write_results_to_file
from auto2 import write_results_to_file2

import pandas as pd
import matplotlib.pyplot as plt
import os
import random


#Runs one iteration of creating the Bayesian network, generating the data and comparing it to the original data
def runDataSynth(degree, epsilon, gennum, beta, theta, threshold_value, null_values, description_file, synthetic_data, seed,input_data,categorical_attributes,candidate_keys):
    
    
        print("Data generated for epsilon " + str(epsilon) + " and degree " + str(degree))
    #Generates the Bayesian Network
        describer = DataDescriber(category_threshold=threshold_value,  null_values=null_values)
        describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data, 
                                                        epsilon=epsilon, 
                                                        beta = beta,
                                                        theta = theta,
                                                        k=degree,
                                                        attribute_to_is_categorical=categorical_attributes,
                                                        attribute_to_is_candidate_key=candidate_keys,
                                                        seed = seed)
        describer.save_dataset_description_to_file(description_file)

        #Displays the bayesian network
        print("Bayesian for epsilon " + str(epsilon) + " and degree " + str(degree))
        display_bayesian_network(describer.bayesian_network)
        #generates the synthetic data and saves it to synthetic data
        generator = DataGenerator()
        generator.generate_dataset_in_correlated_attribute_mode(gennum, description_file)
        generator.save_synthetic_data(synthetic_data)

        # Read both datasets using Pandas.
        input_df = pd.read_csv(input_data, skipinitialspace=True)
        synthetic_df = pd.read_csv(synthetic_data)
        # Read attribute description from the dataset description file.
        attribute_description = read_json_file(description_file)['attribute_description']
        inspector = ModelInspector(input_df, synthetic_df, attribute_description)
    

        print("Comparison for epsilon " + str(epsilon) + " and degree " + str(degree))
    
    #path = 'datasynthplt/' + "epsilon" + str(epsilon) + "degree" + str(degree)
    #isExist = os.path.exists(path)
    #if not isExist:
        #os.makedirs(path)
    #for attribute in synthetic_df.columns:
        #photo = path + "/" + str(attribute) +'.png'
        #inspector.compare_histograms(attribute)
        #plt.savefig(photo, dpi=300, bbox_inches='tight')
        ##plt.cla()
        ##plt.close()
        
    #inspector.mutual_information_heatmap()
    #photo = path + '/heatmap.png'
    #plt.savefig(photo, dpi=300, bbox_inches='tight')
    ##plt.cla()
    ##plt.close()
    
        return synthetic_df, synthetic_data

def midoutput(input_data, degree, epsilon, num_tuples_to_generate,randomseed):
    # 定义相关参数
    threshold_value = 20
    categorical_attributes = {
        'SEX': True, 'RACE': True, 'HISPAN': True, 'EDUC': True, 'TCIG100': True, 'TFREQ': True,
        'TSMKER': True, 'THABIT1': True, 'TPLAN6': True, 'TRYSTOP': True, 'TRYQUIT': True, 'TEVERSTOP': True, 'ECEV': True,
        'ECFREQ': True
    }
    candidate_keys = {}
    null_values = []
    beta = 0.3
    theta = 4
    random.seed(randomseed)
    #random_numbers = [random.randint(0, 100) for _ in range(10)]
    random_numbers = random.sample(range(101), 10)
    print (random_numbers)
    for seed in random_numbers:
        description_file = f"/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/descrip_deg{degree}_eps{epsilon}_randomseed{randomseed}_gen{num_tuples_to_generate}_beta{beta}_theta{theta}_thr{threshold_value}_seed{seed}.json"
        synthetic_data = f"/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/synthetic_cps_00006self copy_deg{degree}_eps{epsilon}_randomseed{randomseed}_gen{num_tuples_to_generate}_beta{beta}_theta{theta}_thr{threshold_value}_seed{seed}.csv"

        # 调用 runDataSynth 函数
        synth_data, synth_data_path = runDataSynth(degree, epsilon, num_tuples_to_generate, beta,
                                               theta, threshold_value, null_values, description_file, synthetic_data, seed, 
                                               input_data, categorical_attributes, candidate_keys)

        log_file = f"/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg{degree}_eps{epsilon}_randomseed{randomseed}_gen{num_tuples_to_generate}_beta{beta}_theta{theta}_thr{threshold_value}_seed{seed}.txt"
        write_results_to_file(synth_data_path, log_file)
        log_file2 = f"/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/runDataSynth_log_deg{degree}_eps{epsilon}_randomseed{randomseed}_gen{num_tuples_to_generate}_beta{beta}_theta{theta}_thr{threshold_value}_seed{seed}_2.txt"
        write_results_to_file2(synth_data_path, log_file2)
    

if __name__ == '__main__':
    # 示例调用 midoutput 函数
    input_data = '/Users/xunuo/Desktop/untitled folder 2/PSIDDifPrivacy/cps_00006self copy.csv'
    degree = 2
    epsilon = 100
    num_tuples_to_generate = 161725

    midoutput(input_data, degree, epsilon, num_tuples_to_generate,111)