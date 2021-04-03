#import library

import numpy as np
import pandas as pd
import re
from math import floor, ceil, sqrt, exp, pi
from csv import reader
from random import seed, randrange


#read data yang disubmit
data = pd.read_csv('diabetes_test.csv')

# Function buat tampilin data frame ke website
def print_df(df):
    for i in range(df.shape[1]):
        if(i == 0):
            if(df.columns[0] == 'index'):
                print('')
            else:
                print(df.columns[i])
        else:
            print(df.columns[i])
#    print('')
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            print(df.iloc[i, j])
#        print('')

# Tampilin data, sama jumlah row + col
print_df(data)

print('Your data has ' + str(data.shape[0]) + ' rows and ' + str(data.shape[1]) + ' columns.')

#Function buat STATISTIKA DESKRIPTIF
#### algoritma mean
def summary_data(df):
    summary_df = pd.DataFrame(columns = df.columns, index=['Mean', 'Median', 'Mode', 'Min', 'Max', 'Range', 'Variance', 'Stdev', 'Q1', 'Q3', 'Length'])
    
    def counter(counters):
        counts = {}
        for count in counters:
            try:
                counts[count] += 1
            except KeyError:
                counts[count] = 1
        return counts
    
    
    for key, value in df.iteritems():
        if (isinstance(value[0], (int, np.int64, float, complex))):
            # NUMERIK
            
            #sort
            old_value = value.tolist()
            new_value = []

            while old_value:
                min = old_value[0]  # arbitrary number in list 
                for x in old_value: 
                    if x < min:
                        min = x
                new_value.append(min)
                old_value.remove(min)    
            
            total = 0
            for i in range(len(value)):
                if(pd.isna(value[i]) == True):
                    value.remove(value[i]) 
                total += value[i]
                
            mean_col = total / (len(value))
            
            x_i_minus_mean = 0
            for i in range(len(value)):
                x_i_minus_mean += (value[i] - mean_col)**2
                
            var = x_i_minus_mean / (len(value) - 1)
            stdev = round(np.sqrt(var),5)
                
            #mean
            summary_df.loc[['Mean'],[key]] = round(mean_col, 5);
            
            #min
            summary_df.loc[['Min'],[key]] = round(new_value[0], 5)
            
            #max
            summary_df.loc[['Max'],[key]] = round(new_value[-1], )
            
            #range
            summary_df.loc[['Range'],[key]] = round(new_value[-1] - new_value[0], 5)
            
            #var
            summary_df.loc[['Variance'],[key]] = round(var, 5)
            
            #sd
            summary_df.loc[['Stdev'],[key]] = round(stdev, 5)
            
            #q1 q3 mean
            def quantile(value, x):
                position = (len(value) + 1 ) * x / 4
                small = value[floor(position)-1]
                big = value[ceil(position)-1]
                q_x = small + (big - small) * ((position+1) - floor(position+1))
                return q_x
            
            #q1
            summary_df.loc[['Q1'],[key]] = round(quantile(new_value, 1), 5)
            
            #q3
            summary_df.loc[['Q3'],[key]] = round(quantile(new_value, 3), 5)
            
            #median
            summary_df.loc[['Median'],[key]] = round(quantile(new_value, 2), 5)
            
            
        else:
            # KATEGORIK
            
            for i in range(len(value)):
                if(pd.isna(value[i]) == True):
                    value.remove(value[i]) 
                    
            #mean
            summary_df.loc[['Mean'],[key]] = '-'
            
            #median
            summary_df.loc[['Median'],[key]] = '-'
            
            #min
            summary_df.loc[['Min'],[key]] = '-'
            
            #max
            summary_df.loc[['Max'],[key]] = '-'
            
            #range
            summary_df.loc[['Range'],[key]] = '-'
            
            #var
            summary_df.loc[['Variance'],[key]] = '-'
            
            #sd
            summary_df.loc[['Stdev'],[key]] = '-'
            
            #q1
            summary_df.loc[['Q1'],[key]] = '-'
            
            #q3
            summary_df.loc[['Q3'],[key]] = '-'

        c = counter(value)
        diction = dict(c)
        mode = [j for j, x in diction.items() if x == max(list(c.values()))] 
            
        if len(mode) == len(value):
            mode = 'No mode'
        else:
            mode = ', '.join(map(str, mode))
        summary_df.loc[['Mode'],[key]] = mode
        
        summary_df.loc[['Length'],[key]] = len(value)
    return(summary_df)
      
summaries = summary_data(data)

print_df(summaries.reset_index())



#Function buat CLASSIFICATION NAIVE BAYES
#read data yang disubmit
data = pd.read_csv('iris.csv')

#pastiin independentnya numerik semua
def check_independent_var(df):
    flag = 0;
    for key, value in df.iteritems():
        if(key != df.columns[-1]):
            if (isinstance(value[0], (int, np.int64, float, complex)) == False):
                flag = 1
    return flag

# Pisahin antar class
#indexnya
def separated_class_index(df):
    separated = dict()
    for i in range(df.shape[0]):
        class_value = df.iloc[i][-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(i)
    return separated

#bind df dengan index yang uda di split tadi
def separated_class(df):
    separated = separated_class_index(df)
    separated_df = dict()
    for class_value, rows in separated.items():
        separated_df[class_value] = df.iloc[separated[class_value]]
    return separated_df

#summary untuk tiap kelas
def summary_by_class(df):
    separated = separated_class(df)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summary_data(rows.reset_index()).loc[['Mean', 'Stdev', 'Length']].drop(['index', rows.columns[-1]], axis = 1)
    return summaries

#hitung probnya dengan rumus gaussian
def gaussian_prob(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    prob = (1 / (sqrt(2 * pi) * stdev)) * exponent
    return prob

# hitung prob dari setiap objek berdasarkan class
def class_prob(summary, rows):
    total = sum([summary[label].reset_index(drop = True).iloc[2, 0] for label in summary])
    prob = dict()
    for class_value, class_summaries in summary.items():
        flag_df = summary[class_value]
        prob[class_value] = flag_df.loc[['Length']].iloc[:,0].item() / total
        for i in range(class_summaries.shape[1]):
            mean = flag_df.loc[['Mean']].iloc[:,i].item()
            stdev = flag_df.loc[['Stdev']].iloc[:,i].item()
            prob[class_value] *= gaussian_prob(rows[i], mean, stdev)
    return prob

#Dari prob yang uda diitung tadi, dibandingin prob nya, classnya di assign ke prob terbesar
def predict_class(summary, rows):
    prob = class_prob(summary, rows)
    best_class, best_prob = None, -1
    for class_value, prob_value in prob.items():
        if best_class is None or prob_value > best_prob:
            best_prob = prob_value
            best_class = class_value
    return best_class

#prediksi dengan naive bayes (gabungin semua func diatas)
def naive_bayes(data):
    df = data.copy()
    class_colnames = df.columns[-1]
    
    summaries = summary_by_class(df)
    df['predicted_class'] = None
    
    for i in range(df.shape[0]):
        df['predicted_class'][i] = predict_class(summaries, df.iloc[i])
    return df[[class_colnames, 'predicted_class']]

#Keluarin accuracy dari hasil prediksi
def naive_bayes_accuracy(df):
    acc = 0
    len_data = df.shape[0]
    for i in range(len_data):
        if(df.iloc[i, 0] == df.iloc[i, 1]):
            acc += 1
    
    accuracy = acc / len_data
    return accuracy

prediction = naive_bayes(data)

print_df(prediction.reset_index())

acc_df = naive_bayes_accuracy(prediction)

print('By using naive bayes classification, the accuracy is ' + str(round(acc_df*100,3)) + ' %')