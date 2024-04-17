# This one, using the multithread, runs quickly (about 10 min for each 100,000 simulations). It obtains results identical to at least 14 digits as the one using no multithread.
# Run on rapidsai/rapidsai-notebooks container (https://hub.docker.com/r/rapidsai/rapidsai-notebooks) with the Estimate3Vaues.yml enviroment.

import pandas as pd
import numpy as np
from scipy.stats import norm, chi2
from datetime import datetime
from math import sqrt, log, exp
import dask.dataframe as dd
import os
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects
from rpy2.robjects.vectors import StrVector
import re

# import R's utility package
utils = rpackages.importr('utils')

# select the first mirror in the CRAN for R packages
utils.chooseCRANmirror(ind=1) 

# R package names we needed
packnames = ('estmeansd',)
# Check if the package have already been installed.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]

# StrVector is R vector of strings. We selectively install what we needs based on packnames.
if len(names_to_install) > 0:
    print(f"installing R packages: {names_to_install}")
    utils.install_packages(StrVector(names_to_install))
else:
    print("R package estmeansd has been installed")

class SimulPivotMC(object):
    def __init__(self, nMonteSim, N, CV):
        # number of Monte Carlo Simulation
        self.nMonte = nMonteSim

        # Calculate z-score for alpha = 0.05, ppf is the percent point function that is inverse of cumulative distribution function
        self.z_score = norm.ppf(1 - 0.05 / 2)

        # Sample size, we choose 15, 27, 51, notation "n" in the manuscript
        self.N1 = N
        self.N2 = self.N1

        # coefficient of variation, we choose 0.15, 0.3, 0.5
        self.CV1 = CV
        self.CV2 = self.CV1

        # Mean in log scale in the manuscript
        self.rMeanLogScale1 = 0
        self.rMeanLogScale2 = self.rMeanLogScale1
        
        # Standard deviation in log scale
        self.rSDLogScale1 = sqrt(log(1 + self.CV1 ** 2)) 
        self.rSDLogScale2 = self.rSDLogScale1

        # the number for pivot
        nSimulForPivot = 100000-1

        # choosing a seed
        self.seed_value = 12181988

        # Generate 4 set of random numbers each with specified seed, will be used for U1, U2, Z1, and Z2 later
        np.random.seed(self.seed_value - 1)
        random_numbers1_1 = np.random.rand(nSimulForPivot)

        np.random.seed(self.seed_value - 2)
        random_numbers1_2 = np.random.rand(nSimulForPivot)

        np.random.seed(self.seed_value - 3)
        random_numbers2_1 = np.random.rand(nSimulForPivot)

        np.random.seed(self.seed_value - 4)
        random_numbers2_2 = np.random.rand(nSimulForPivot)

        # Generate random number for later used in calculating Ui and Zi in generalized pivotal method
        # group 1 pivot calculation
        self.U1 = chi2.ppf(random_numbers1_1, self.N1 - 1 )
        self.Z1 = norm.ppf(random_numbers2_1)

        # group 2 pivot calculation     
        self.U2 = chi2.ppf(random_numbers1_2, self.N2 - 1 )
        self.Z2 = norm.ppf(random_numbers2_2)
    
    # the main process, method = ['valid', 'Luo_Wan', 'qe', 'bc', 'mln'], valid means valid code for generalized confidence interval
    # Luo_Wan means Luo et al.’s and Wan et al.’s methods
    # qe and bc means McGrath et al.’s quantile estimation and Box-Cox transformation
    # mln means Cai et al.'s maximum likelihood estimation method for non-normal distribution  
    def main(self, method):
        # the pre-determined list of seeds, using number of nMonte
        list_seeds = [i for i in range(self.seed_value, self.seed_value + self.nMonte)] 
        # put the list of seeds into a table (a.k.a DataFrame) with one column named "Seeds"  
        df = pd.DataFrame({'Seeds':list_seeds}) 
        meta=('float64', 'float64')
        # valid code for generalized confidence interval
        if method == 'valid': 
            print(f'method:{method}')
            print('Samples_normal')
            # generate log-normal distributed numbers in log scale, using mean of rMeanLogScale and standard deviation of rSDLogScale
            df['rSampleOfRandoms'] = df.apply(self.Samples_normal, args=('Seeds',), axis=1) 
            # record mean and SD in raw scale, not use for later calculation
            df[['rSampleMeanRaw1', 'rSampleSDRaw1', 'rSampleMeanRaw2', 'rSampleSDRaw2']] = df['rSampleOfRandoms'].apply(lambda x: [np.exp(item) for item in x]).apply(self.Mean_SD).tolist()
            df_record = df[['rSampleMeanRaw1', 'rSampleSDRaw1', 'rSampleMeanRaw2', 'rSampleSDRaw2']].copy()
            
            print('dask')
            # put the table into dask, a progress that can parallel calculating each rows using multi-thread
            df = dd.from_pandas(df['rSampleOfRandoms'], npartitions=16) 
            print('Mean_SD')
            # calculate sample mean and SD using Mean_SD
            df = df.apply(self.Mean_SD, meta=meta)                   
            
        # methods estimating means and standard deviations from medians and interquartile ranges
        elif method in ['Luo_Wan', 'qe', 'bc', 'mln']:
            print(f'method:{method}')
            print('Samples_normal + exponential')
            # generate log-normal distributed numbers in raw scale, using mean of rMeanLogScale and standard deviation of rSDLogScale
            df['rSampleOfRandoms'] = df.apply(self.Samples_normal, args=('Seeds',), axis=1).apply(lambda x: [np.exp(item) for item in x])            
            
            print(f'get_estimated_Mean_SD_from_Samples with method = {method}')
            # calculate estimated sample mean and SD from medians and interquartile ranges using the specified method
            df[['rSampleMeanRaw1', 'rSampleSDRaw1', 'rSampleMeanRaw2', 'rSampleSDRaw2']] = df['rSampleOfRandoms'].apply(self.get_estimated_Mean_SD_from_Samples, args=(method,)).tolist()
        
            df_record = df[['rSampleMeanRaw1', 'rSampleSDRaw1', 'rSampleMeanRaw2', 'rSampleSDRaw2']].copy()

            print('dask')
            # put the table into dask, a progress that can parallel calculating each rows using multi-thread
            df = dd.from_pandas(df[['rSampleMeanRaw1', 'rSampleSDRaw1', 'rSampleMeanRaw2', 'rSampleSDRaw2']], npartitions=35) 
            meta = ('float64', 'float64')

            print('first_two_moment')
            # transform estimated ample mean and SD in log scale using estimated Mean and SD
            df = df.apply(self.first_two_moment, axis=1, args=(0,1,2,3), meta=meta) 
                           
        else:
            # if no method choosed, print this warning
            print('no method in main')

        df_record[['rSampleMeanLogScale1', 'rSampleSDLogScale1', 'rSampleMeanLogScale2', 'rSampleSDLogScale2']] = df.compute().tolist()
        
        print('GPM_log_ratio_Mean')
        # generate 'ln_ratio' and 'se_ln_ratio' for means with sample mean and SD using GPM
        df_Mean = df.apply(self.GPM_log_ratio_Mean, args=(0,1,2,3), meta=meta)  
        df_record[['ln_ratio_Mean', 'se_ln_ratio_Mean']] = df_Mean.compute().tolist()

        print('Coverage Mean')
        # check coverage of each rows for means
        df_Mean = df_Mean.apply(self.Coverage, args=(0,1,0), meta=meta) 
        df_record['coverage_Mean'] = df_Mean.compute().tolist()        

        print('Mean coverage_Mean')
        # compute the mean of the list of coverage (0 or 1), it equals to the percentage of coverage
        coverage_Mean = df_record['coverage_Mean'].mean()
        
        print('GPM_log_ratio_SD')
        # generate 'ln_ratio' and 'se_ln_ratio' for standard deviations with sample mean and SD using generalized pivotal method
        df_SD = df.apply(self.GPM_log_ratio_SD, args=(0,1,2,3), meta=meta)  
        df_record[['ln_ratio_SD', 'se_ln_ratio_SD']] = df_SD.compute().tolist()

        print('Coverage SD')
        # check coverage for standard deviations of each rows
        df_SD = df_SD.apply(self.Coverage, args=(0,1,0), meta=meta) 
        df_record['coverage_SD'] = df_SD.compute().tolist()        

        print('Mean coverage_SD')
        # compute the mean of the list of coverage (0 or 1), it equals to the percentage of coverage
        coverage_SD = df_record['coverage_SD'].mean()

        return coverage_SD, coverage_Mean, df_record, self.nMonte, self.N1, self.CV1, method

    
    def Samples_normal(self, row, seed_):        
        # using seed from pre-determined list
        np.random.seed(row[seed_]) 
        # generate log-normal distribution, using mean of rMeanLogScale and standard deviation of rSDLogScale
        rSampleOfRandoms = [(norm.ppf(i,loc=self.rMeanLogScale1, scale=self.rSDLogScale1)) for i in np.random.rand(self.N1+self.N2)] 
        
        return rSampleOfRandoms

    def Mean_SD(self, row):

        rSampleOfRandoms1 = row[:self.N1]
        rSampleOfRandoms2 = row[self.N1:(self.N1+self.N2)]
        
        # the mean of rSampleOfRandoms1
        rSampleMean1 = np.mean(rSampleOfRandoms1)  
        # the standard deviation of rSampleOfRandoms1, delta degree of freeden = 1
        rSampleSD1 = np.std(rSampleOfRandoms1, ddof=1) 
        rSampleMean2 = np.mean(rSampleOfRandoms2)
        rSampleSD2 = np.std(rSampleOfRandoms2, ddof=1)

        return rSampleMean1, rSampleSD1, rSampleMean2, rSampleSD2
    
    def get_estimated_Mean_SD_from_Samples(self, row, method = ''):
        # estimated means and standard deviations from 3 quartiles
        rSampleOfRandoms1 = row[:self.N1]
        rSampleOfRandoms2 = row[self.N1:(self.N1+self.N2)]

        q1_1 = pd.Series(rSampleOfRandoms1).quantile(.25)
        median_1 = pd.Series(rSampleOfRandoms1).quantile(.5)
        q3_1 = pd.Series(rSampleOfRandoms1).quantile(.75)

        q1_2 = pd.Series(rSampleOfRandoms2).quantile(.25)
        median_2 = pd.Series(rSampleOfRandoms2).quantile(.5)
        q3_2 = pd.Series(rSampleOfRandoms2).quantile(.75)
        
        if method == 'Luo_Wan':
            # estimated means and standard deviations from 3 quartiles, based on Luo et al's and Wan et al's method
            rSampleMean1, rSampleSD1 = self.ThreeValues_to_Mean_SD_Luo_Wan(q1_1, median_1, q3_1, self.N1)
            rSampleMean2, rSampleSD2 = self.ThreeValues_to_Mean_SD_Luo_Wan(q1_2, median_2, q3_2, self.N2)
        
        elif method in ['qe', 'bc', 'mln']:            
            robjects.r(f"""
            library(estmeansd)
            set.seed(1)
            mean_sd1 <- {method}.mean.sd(q1.val = {q1_1}, med.val = {median_1}, q3.val = {q3_1}, n = {self.N1})
            mean_sd2 <- {method}.mean.sd(q1.val = {q1_2}, med.val = {median_2}, q3.val = {q3_2}, n = {self.N2})
                    """)

            rSampleMean1, rSampleSD1 = robjects.r['mean_sd1'][0][0], robjects.r['mean_sd1'][1][0]
            rSampleMean2, rSampleSD2 = robjects.r['mean_sd2'][0][0], robjects.r['mean_sd2'][1][0]

        else:
            print('not right method in get_estimated_Mean_SD_from_Samples')

        return rSampleMean1, rSampleSD1, rSampleMean2, rSampleSD2

    def ThreeValues_to_Mean_SD_Luo_Wan(self, q1, median, q3, N):
        # the simplified optimal weight from Luo et al; also validated from Shi et al's supplementary Excel file
        opt_weight = 0.7 + 0.39/N
        # estimating mean based on Luo et al's method
        est_mean = opt_weight * (q1 + q3)/2 + (1-opt_weight) * median
        # estimating standard deviations based on Wan et al's method
        est_SD = (q3 - q1)/(2 * norm.ppf((0.75 * N - 0.125)/(N + 0.25))) # loc=0, scale=1

        return est_mean, est_SD
    
    def first_two_moment(self, row, col_SampleMean1, col_SampleSD1, col_SampleMean2, col_SampleSD2):
        
        SampleMean1 = row.iloc[col_SampleMean1]
        SampleSD1 = row.iloc[col_SampleSD1]

        SampleMean2 = row.iloc[col_SampleMean2]
        SampleSD2 = row.iloc[col_SampleSD2]
        # transforming raw means and standard deviations to log scale using first two moments
        rSampleMeanLogScale1, rSampleSDLogScale1 = self.transform_from_raw_to_log_mean_SD(SampleMean1, SampleSD1)
        rSampleMeanLogScale2, rSampleSDLogScale2 = self.transform_from_raw_to_log_mean_SD(SampleMean2, SampleSD2)
                   
        return rSampleMeanLogScale1, rSampleSDLogScale1, rSampleMeanLogScale2, rSampleSDLogScale2

    def transform_from_raw_to_log_mean_SD(self, Mean, SD):        
        # transforming raw means and standard deviations to log scale 
        CV = SD/Mean
        CVsq = CV**2
        # Mean in log scale
        MeanLogScale_1 = log(Mean/sqrt(CVsq + 1)) 
        # SD in log scale
        SDLogScale_1 = sqrt(log((CVsq + 1)))
        
        return MeanLogScale_1, SDLogScale_1
    
    def GPM_log_ratio_SD(self, row, col_SampleMeanLog1, col_SampleSDLog1, col_SampleMeanLog2, col_SampleSDLog2):
        # calculating log ratio of standard deviations using generalized pivotal method
        # group 1 pivot calculation
        SampleMeanLog1 = row[col_SampleMeanLog1]
        SampleSDLog1 = row[col_SampleSDLog1]
        Pivot1 = self.Pivot_calculation_2SD(SampleMeanLog1, SampleSDLog1, self.N1, self.U1, self.Z1)
        # group 2 pivot calculation
        SampleMeanLog2 = row[col_SampleMeanLog2]
        SampleSDLog2 = row[col_SampleSDLog2]
        Pivot2 = self.Pivot_calculation_2SD(SampleMeanLog2, SampleSDLog2, self.N2, self.U2, self.Z2)

        # generalized pivotal statistics
        pivot_statistics = np.log(Pivot1) - np.log(Pivot2)

        # Calculate ln ratio and SE ln ratio by percentile and Z statistics
        ln_ratio = pd.Series(pivot_statistics).quantile(.5)
        se_ln_ratio = (pd.Series(pivot_statistics).quantile(.75) - pd.Series(pivot_statistics).quantile(.25))/(norm.ppf(.75) - norm.ppf(.25))
        
        return ln_ratio, se_ln_ratio
    
    def GPM_log_ratio_Mean(self, row, col_SampleMeanLog1, col_SampleSDLog1, col_SampleMeanLog2, col_SampleSDLog2):
        # calculating log ratio of means using generalized pivotal method
        # group 1 pivot calculation
        SampleMeanLog1 = row[col_SampleMeanLog1]
        SampleSDLog1 = row[col_SampleSDLog1]
        Pivot1 = self.Pivot_calculation_2mean(SampleMeanLog1, SampleSDLog1, self.N1, self.U1, self.Z1)

        # group 2 pivot calculation
        SampleMeanLog2 = row[col_SampleMeanLog2]
        SampleSDLog2 = row[col_SampleSDLog2]
        Pivot2 = self.Pivot_calculation_2mean(SampleMeanLog2, SampleSDLog2, self.N2, self.U2, self.Z2)

        # generalized pivotal statistics
        pivot_statistics = Pivot1 - Pivot2 
        
        # Calculate ln ratio and SE ln ratio by percentile and Z statistics
        ln_ratio = pd.Series(pivot_statistics).quantile(.5)
        se_ln_ratio = (pd.Series(pivot_statistics).quantile(.75) - pd.Series(pivot_statistics).quantile(.25))/(norm.ppf(.75) - norm.ppf(.25))

        return ln_ratio, se_ln_ratio
    
    def Pivot_calculation_2SD(self, rSampleMeanLogScale, rSampleSDLogScale, N, U, Z):
        return np.exp(rSampleMeanLogScale- np.sqrt((rSampleSDLogScale**2 * (N-1))/U) * (Z/sqrt(N)) ) * np.sqrt((np.exp((rSampleSDLogScale**2 * (N-1))/U) - 1) * np.exp((rSampleSDLogScale**2 * (N-1))/U))

    def Pivot_calculation_2mean(self, rSampleMeanLogScale, rSampleSDLogScale, N, U, Z):
        sqrt_U = np.sqrt(U)
        return rSampleMeanLogScale - (Z/sqrt(N)) * (rSampleSDLogScale/(sqrt_U/sqrt(N-1))) + 0.5 * (rSampleSDLogScale/(sqrt_U/sqrt(N-1))) ** 2

    def Coverage(self, row, col_ln_ratio, col_se_ln_ratio, ideal):
        
        ln_ratio = row[col_ln_ratio]
        se_ln_ratio = row[col_se_ln_ratio]

        # Calculate the confidence intervals with z_score of alpha = 0.05
        lower_bound = ln_ratio - self.z_score * se_ln_ratio
        upper_bound = ln_ratio + self.z_score * se_ln_ratio   
        
        intervals_include_zero = (lower_bound < ideal) and (upper_bound > ideal)
        # 1 as True, 0 as False, check coverage
        return int(intervals_include_zero)  

# assign output folder, create new one if not existed
folder = "MeanSD_From3ValuesInRaw_BCQEMLN_20240418_nSim1M"
os.makedirs(folder, exist_ok = True)

# check if there are existed files from previous run; for continuing previous run
done_list = []
pattern = r"MeanSD_From5Values_nMonte_(\d+)_N_(\d+)_CV_(\d\.\d+)_(\d{8}\d{6})_(\w+).csv"
matching_files = [file for file in os.listdir(folder) if re.match(pattern, file)]
for file_name in matching_files:
    match = re.match(pattern, file_name)
    done_list.append((match.group(2), match.group(3), match.group(5)))

if __name__ == '__main__':
    # number of Monte Carlo simulations
    nMonteSim = 100000
    # Sample size, we choose 15, 27, 51, notation "n" in the manuscript
    for N in [15, 27, 51]:
        # coefficient of variation, we choose 0.15, 0.3, 0.5
        for CV in [0.15, 0.3, 0.5]:
            for method in ['valid','Luo_Wan', 'qe', 'bc', 'mln']:
                print(method)
                
                # if already done in previous run, skip this setting
                if (str(N),str(CV),method) in done_list:
                    print(f"{(str(N),str(CV),method)} has been done and skip")
                    continue
                
                # record the datetime at the start
                start_time = datetime.now() 
                print('start_time:', start_time) 
                print(f"Start GPM_MC_nMonteSim_{nMonteSim}_N_{N}_CV_{CV}_{str(start_time).split('.')[0].replace('-','').replace(' ','').replace(':','')}_{method}")

                # Cal the class SimulPivotMC(), generate variables in the def __init__(self)
                run = SimulPivotMC(nMonteSim, N, CV)  
                # start main()
                coverage_SD, coverage_Mean, df_record, nMonte, N1, CV1, method = run.main(method=method)  
                
                # record the datetime at the end
                end_time = datetime.now() 
                # print the datetime at the end
                print('end_time:', end_time) 
                # calculate the time taken
                time_difference = end_time - start_time
                print('time_difference:', time_difference) 
                # print out the percentage of coverage
                print('coverage SD: %s' %(coverage_SD,)) 
                print('coverage Mean: %s' %(coverage_Mean,)) 

                output_dir = f"MeanSD_From5Values_nMonte_{nMonte}_N_{N1}_CV_{CV1}_{str(end_time).split('.')[0].replace('-','').replace(' ','').replace(':','')}"
                output_dir = os.path.join(folder,output_dir)
                
                # save the results to the csv
                df_record.to_csv(output_dir + f'_{method}.csv')
                print('csv save to ' + output_dir + f'_{method}.csv')
                
quit()
