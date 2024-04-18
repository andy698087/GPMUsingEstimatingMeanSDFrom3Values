# There are two code files. This one, using no multithread, runs slowly, but easier to read and obtains results identical to at least 14 digits as the multithread used for the simulations.

import pandas as pd
import numpy as np
from scipy.stats import norm, chi2
from datetime import datetime
from math import sqrt, log, exp
import os
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects
from rpy2.robjects.vectors import StrVector
import re


def get_estimated_Mean_SD_from_Samples(rSampleOfRandoms, method = ''):
    # estimated means and standard deviations from 3 quartiles
    rSampleOfRandoms1 = rSampleOfRandoms[:N1]
    rSampleOfRandoms2 = rSampleOfRandoms[N1:(N1+N2)]

    q1_1 = pd.Series(rSampleOfRandoms1).quantile(.25)
    median_1 = pd.Series(rSampleOfRandoms1).quantile(.5)
    q3_1 = pd.Series(rSampleOfRandoms1).quantile(.75)

    q1_2 = pd.Series(rSampleOfRandoms2).quantile(.25)
    median_2 = pd.Series(rSampleOfRandoms2).quantile(.5)
    q3_2 = pd.Series(rSampleOfRandoms2).quantile(.75)
    
    if method == 'Luo_Wan':
        # estimated means and standard deviations from 3 quartiles, based on Luo et al's and Wan et al's method
        rSampleMean1, rSampleSD1 = ThreeValues_to_Mean_SD_Luo_Wan(q1_1, median_1, q3_1, N1)
        rSampleMean2, rSampleSD2 = ThreeValues_to_Mean_SD_Luo_Wan(q1_2, median_2, q3_2, N2)
    
    elif method in ['qe', 'bc', 'mln']:            
        robjects.r(f"""
        library(estmeansd)
        set.seed(1)
        mean_sd1 <- {method}.mean.sd(q1.val = {q1_1}, med.val = {median_1}, q3.val = {q3_1}, n = {N1})
        mean_sd2 <- {method}.mean.sd(q1.val = {q1_2}, med.val = {median_2}, q3.val = {q3_2}, n = {N2})
                """)

        rSampleMean1, rSampleSD1 = robjects.r['mean_sd1'][0][0], robjects.r['mean_sd1'][1][0]
        rSampleMean2, rSampleSD2 = robjects.r['mean_sd2'][0][0], robjects.r['mean_sd2'][1][0]

    else:
        print('not right method in get_estimated_Mean_SD_from_Samples')

    return rSampleMean1, rSampleSD1, rSampleMean2, rSampleSD2

def ThreeValues_to_Mean_SD_Luo_Wan(q1, median, q3, N):
    # the simplified optimal weight from Luo et al; also validated from Shi et al's supplementary Excel file
    opt_weight = 0.7 + 0.39/N
    # estimating mean based on Luo et al's method
    est_mean = opt_weight * (q1 + q3)/2 + (1-opt_weight) * median
    # estimating standard deviations based on Wan et al's method
    est_SD = (q3 - q1)/(2 * norm.ppf((0.75 * N - 0.125)/(N + 0.25))) # loc=0, scale=1
    return est_mean, est_SD

def transform_from_raw_to_log_mean_SD(Mean, SD):        
    # transforming raw means and standard deviations to log scale 
    CV = SD/Mean
    CVsq = CV**2
    # Mean in log scale
    MeanLogScale_1 = log(Mean/sqrt(CVsq + 1)) 
    # SD in log scale
    SDLogScale_1 = sqrt(log((CVsq + 1)))    
    return MeanLogScale_1, SDLogScale_1

def first_two_moment(Mean1, SD1, Mean2, SD2):
    # transforming raw means and standard deviations to log scale using first two moments
    MeanLogScale1, SDLogScale1 = transform_from_raw_to_log_mean_SD(Mean1, SD1)
    MeanLogScale2, SDLogScale2 = transform_from_raw_to_log_mean_SD(Mean2, SD2)                
    return MeanLogScale1, SDLogScale1, MeanLogScale2, SDLogScale2

def Pivot_calculation_2SD(MeanLogScale, SDLogScale, N, U, Z):
    # calculating log ratio of standard deviations using generalized pivotal method
    return np.exp(MeanLogScale- np.sqrt((SDLogScale**2 * (N-1))/U) * (Z/sqrt(N)) ) * np.sqrt((np.exp((SDLogScale**2 * (N-1))/U) - 1) * np.exp((SDLogScale**2 * (N-1))/U))

def Pivot_calculation_2mean(MeanLogScale, SDLogScale, N, U, Z):
    # calculating log ratio of means using generalized pivotal method
    sqrt_U = np.sqrt(U)
    return MeanLogScale - (Z/sqrt(N)) * (SDLogScale/(sqrt_U/sqrt(N-1))) + 0.5 * (SDLogScale/(sqrt_U/sqrt(N-1))) ** 2

def GPM_log_ratio_SD(SampleMeanLog1, SampleSDLog1, SampleMeanLog2, SampleSDLog2):
    # calculating log ratio of standard deviations using generalized pivotal method
    # group 1 pivot calculation
    Pivot1 = Pivot_calculation_2SD(SampleMeanLog1, SampleSDLog1, N1, U1, Z1)
    # group 2 pivot calculation
    Pivot2 = Pivot_calculation_2SD(SampleMeanLog2, SampleSDLog2, N2, U2, Z2)

    # Equation 2, generalized pivotal statistics
    pivot_statistics = np.log(Pivot1) - np.log(Pivot2)

    # Calculate ln ratio and SE ln ratio by percentile and Z statistics
    ln_ratio = pd.Series(pivot_statistics).quantile(.5)
    se_ln_ratio = (pd.Series(pivot_statistics).quantile(.75) - pd.Series(pivot_statistics).quantile(.25))/(norm.ppf(.75) - norm.ppf(.25))
    
    return ln_ratio, se_ln_ratio

def GPM_log_ratio_Mean(SampleMeanLog1, SampleSDLog1, SampleMeanLog2, SampleSDLog2):
    # calculating log ratio of means using generalized pivotal method
    # group 1 pivot calculation
    Pivot1 = Pivot_calculation_2mean(SampleMeanLog1, SampleSDLog1, N1, U1, Z1)

    # group 2 pivot calculation
    Pivot2 = Pivot_calculation_2mean(SampleMeanLog2, SampleSDLog2, N2, U2, Z2)

    # Equation 2, generalized pivotal statistics
    pivot_statistics = Pivot1 - Pivot2 
    
    # Calculate ln ratio and SE ln ratio by percentile and Z statistics
    ln_ratio = pd.Series(pivot_statistics).quantile(.5)
    se_ln_ratio = (pd.Series(pivot_statistics).quantile(.75) - pd.Series(pivot_statistics).quantile(.25))/(norm.ppf(.75) - norm.ppf(.25))

    return ln_ratio, se_ln_ratio

def Coverage(ln_ratio, se_ln_ratio, ideal):

    # Calculate the confidence intervals with z_score of alpha = 0.05
    lower_bound = ln_ratio - z_score * se_ln_ratio
    upper_bound = ln_ratio + z_score * se_ln_ratio   
    
    intervals_include_zero = (lower_bound < ideal) and (upper_bound > ideal)
    # 1 as True, 0 as False, check coverage
    return int(intervals_include_zero)  

# import R's utility package
utils = rpackages.importr('utils')

# select the first mirror in the CRAN for R packages
utils.chooseCRANmirror(ind=1) 

# name of R package we needed
packnames = ('estmeansd',)
# Check if the package have already been installed.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]

# StrVector is R vector of strings. We selectively install what we need based on packnames.
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))
    print(f"installing R packages: {names_to_install}")    
else:
    print("R package estmeansd has been installed")


# Calculate z-score for alpha = 0.05, ppf is the percent point function that is inverse of cumulative distribution function
z_score = norm.ppf(1 - 0.05 / 2)

# the number for pivot
nSimulForPivot = 100000-1

# choosing a seed
seed_value = 12181988

# Generate 4 set of random numbers each with specified seed, will be used for U1, U2, Z1, and Z2 later
np.random.seed(seed_value - 1)
random_numbers1_1 = np.random.rand(nSimulForPivot)

np.random.seed(seed_value - 2)
random_numbers1_2 = np.random.rand(nSimulForPivot)

np.random.seed(seed_value - 3)
random_numbers2_1 = np.random.rand(nSimulForPivot)

np.random.seed(seed_value - 4)
random_numbers2_2 = np.random.rand(nSimulForPivot)

    
# assign output folder, create new one if not existed
folder = "MeanSD_From3ValuesInRaw_BCQEMLN_nSim1M"
os.makedirs(folder, exist_ok = True)

# check if there are existed files from previous run; for continuing previous run
done_list = []
pattern = r"MeanSD_From5Values_nMonte_(\d+)_N_(\d+)_CV_(\d\.\d+)_(\d{8}\d{6})_(\w+).csv"
matching_files = [file for file in os.listdir(folder) if re.match(pattern, file)]
for file_name in matching_files:
    match = re.match(pattern, file_name)
    done_list.append((match.group(2), match.group(3), match.group(5)))

# number of Monte Carlo simulations
nMonte = 1000000
# Sample size, we choose 15, 27, 51, notation "n" in the manuscript
for N in [15, 27, 51]:
    # Sample size, we choose 15, 27, 51, notation "n" in the manuscript
    N1 = N
    N2 = N1

    # Generate random number for later used in calculating Ui and Zi in generalized pivotal method
    # group 1 pivot calculation
    U1 = chi2.ppf(random_numbers1_1, N1 - 1 )
    Z1 = norm.ppf(random_numbers2_1)

    # group 2 pivot calculation     
    U2 = chi2.ppf(random_numbers1_2, N2 - 1 )
    Z2 = norm.ppf(random_numbers2_2)

    # coefficient of variation, we choose 0.15, 0.3, 0.5
    for CV in [0.15, 0.3, 0.5]:

        # coefficient of variation, we choose 0.15, 0.3, 0.5
        CV1 = CV
        CV2 = CV1

        # Mean in log scale in the manuscript
        rMeanLogScale1 = 0
        rMeanLogScale2 = rMeanLogScale1

        # Standard deviation in log scale
        rSDLogScale1 = sqrt(log(1 + CV1 ** 2)) 
        rSDLogScale2 = rSDLogScale1


        for method in ['valid','Luo_Wan', 'qe', 'bc', 'mln']:
            print(method)
            
            # if already done in previous run, skip this setting
            if (str(N),str(CV),method) in done_list:
                print(f"N={str(N)}, CV={str(CV)}, method={method} has been done and skip")
                continue
            
            # record the datetime at the start
            start_time = datetime.now() 
            print('start_time:', start_time) 
            print(f"Start GPM_MC_nMonteSim_{nMonte}_N_{N}_CV_{CV}_{method}_{start_time.strftime('%Y%m%d%H%M%S')}")

            #collecting results
            dict_results = {'ln_ratio_Mean': [], 'se_ln_ratio_Mean': [], 'coverage_Mean': [], 'ln_ratio_SD': [], 'se_ln_ratio_SD': [], 'coverage_SD': []}
            # the pre-determined list of seeds, using number of nMonte
            list_seeds = [i for i in range(seed_value, seed_value + nMonte)] 

            for seed_ in list_seeds:
                # Calculate the mean and standard deviation of a sample generated from a random generator of a normal distribution
                np.random.seed(seed_)
                
                # valid code for generalized confidence interval
                if method == 'valid': 
                    # generate log-normal distributed numbers in log scale, using mean of rMeanLogScale and standard deviation of rSDLogScale
                    rSampleOfRandoms = [norm.ppf(i,loc=rMeanLogScale1, scale=rSDLogScale1) for i in np.random.rand(N1+N2)] 

                    rSampleOfRandoms1 = rSampleOfRandoms[:N1]
                    rSampleOfRandoms2 = rSampleOfRandoms[N1:N1+N2] 

                    # the mean of rSampleOfRandoms1
                    rSampleMeanLogScale1 = np.mean(rSampleOfRandoms1)  
                    # the standard deviation of rSampleOfRandoms1, delta degree of freeden = 1
                    rSampleSDLogScale1 = np.std(rSampleOfRandoms1, ddof=1) 
                    rSampleMeanLogScale2 = np.mean(rSampleOfRandoms2)
                    rSampleSDLogScale2 = np.std(rSampleOfRandoms2, ddof=1)

                # methods estimating means and standard deviations from medians and interquartile ranges
                elif method in ['Luo_Wan', 'qe', 'bc', 'mln']:
                    # generate log-normal distributed numbers, using mean of rMeanLogScale and standard deviation of rSDLogScale
                    rSampleOfRandoms = [norm.ppf(i,loc=rMeanLogScale1, scale=rSDLogScale1) for i in np.random.rand(N1+N2)] 
                    # transform to samples from log to raw scale            
                    rSampleOfRandoms = np.exp(rSampleOfRandoms)

                    # calculate estimated sample mean and SD from medians and interquartile ranges using the specified method
                    rSampleMean1, rSampleSD1, rSampleMean2, rSampleSD2 = get_estimated_Mean_SD_from_Samples(rSampleOfRandoms, method = method)
                    
                    # transform estimated ample mean and SD in log scale using estimated Mean and SD
                    rSampleMeanLogScale1, rSampleSDLogScale1, rSampleMeanLogScale2, rSampleSDLogScale2 = first_two_moment(rSampleMean1, rSampleSD1, rSampleMean2, rSampleSD2)

                # generate 'ln_ratio' and 'se_ln_ratio' for standard deviations with sample mean and SD using generalized pivotal method
                ln_ratio, se_ln_ratio = GPM_log_ratio_SD(rSampleMeanLogScale1, rSampleSDLogScale1, rSampleMeanLogScale2, rSampleSDLogScale2)
                dict_results['ln_ratio_SD'].append(ln_ratio)
                dict_results['se_ln_ratio_SD'].append(se_ln_ratio)

                # check coverage for standard deviations of each rows
                coverage_SD = Coverage(ln_ratio, se_ln_ratio, 0)
                dict_results['coverage_SD'].append(coverage_SD)                
                
                # generate 'ln_ratio' and 'se_ln_ratio' for means with sample mean and SD using GPM
                ln_ratio, se_ln_ratio = GPM_log_ratio_Mean(rSampleMeanLogScale1, rSampleSDLogScale1, rSampleMeanLogScale2, rSampleSDLogScale2)
                dict_results['ln_ratio_Mean'].append(ln_ratio)
                dict_results['se_ln_ratio_Mean'].append(se_ln_ratio)

                # check coverage of each rows for means
                coverage_Mean = Coverage(ln_ratio, se_ln_ratio, 0)
                dict_results['coverage_Mean'].append(coverage_Mean)
            
            # compute the mean of the list of coverage (0 or 1), it equals to the percentage of coverage
            mean_coverage_SD = np.mean(dict_results['coverage_SD'])
            mean_coverage_Mean = np.mean(dict_results['coverage_Mean'])
            
            # record the datetime at the end
            end_time = datetime.now() 
            # print the datetime at the end
            print('end_time:', end_time) 
            # calculate the time taken
            time_difference = end_time - start_time
            print('time_difference:', time_difference) 
            # print out the percentage of coverage
            print('coverage SD: %s' %(mean_coverage_SD,)) 
            print('coverage Mean: %s' %(mean_coverage_Mean,)) 

            output_dir = f"MeanSD_From5Values_nMonte_{nMonte}_N_{N1}_CV_{CV1}_{end_time.strftime('%Y%m%d%H%M%S')}"
            output_dir = os.path.join(folder,output_dir)
            
            # save the results to the csv
            pd.DataFrame(dict_results).to_csv(output_dir + f'_{method}.csv')
            print('csv save to ' + output_dir + f'_{method}.csv')
            
quit()
