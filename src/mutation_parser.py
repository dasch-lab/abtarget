# Program to convert yaml file to dictionary
import yaml
import re
import pandas as pd 
import csv
import os
import random
import yaml
random.seed(22)

# opening a file
'''with open('/disk1/abtarget/abcd_dataset.yml', 'r') as stream:
    try:
        # Converts yaml document to python object
        d=yaml.safe_load(stream)
        # Printing dictionary
        print(d)
    except yaml.YAMLError as e:
        print(e)'''


def parse_nohup(nohup_file):
    with open(nohup_file,'r') as stream:
        data_lines = stream.readlines()

    result = pd.DataFrame({'target' : [], 'chain': [], 'model': [], 'generation':[],'mutation':[]})
    for line in data_lines:
        if 'UserWarning' in line:
            continue
        if 'Regression weights not found' in line:
            continue

        if line.startswith('#\tS'):
            # target = [line.strip('# Sequence ')]
            name = line.split('#\tSequence\t')[1].strip('\n')
            target = name.split('_')[0]
            chain = name.split('_')[1]
            # continue
        elif line.startswith('##'):
            model_name = [line.strip('##\tModel\t').strip('\n').split(':')[1].strip('\t')]
        elif line.startswith('Gen'):
            generation = int(line.split(':')[0].strip('Gen '))
            mutation = line.split(':')[1].strip('\n')
            # continue
        elif 'Converged' in line:
            element = {'target' : target, 'chain': chain, 'model': model_name, 'generation':generation,'mutation':mutation}
            element = pd.DataFrame.from_dict(element)
            result = result.append(element, ignore_index=True)
        #    # generation += 1
        else:
            continue
    # convert "generation" from Float to int
    result = result.astype({'generation':'int'})
    result.sort_values(by = ['model', 'target'])
    return result

def concat_rows(rows):
        row = rows.iloc[0]

        if len(rows) > 1:
            row1 = rows.iloc[0]
            row2 = rows.iloc[1]
            element = {'name': [row1['target']], 'VH': [row1['mutation']], 'VL': [row2['mutation']], 'target': 'NonProtein', 'label':[1]}
            element = pd.DataFrame.from_dict(element)
            return element

def write_csv(csv_file, df):
    with open(csv_file, 'w') as f:
        mywriter = csv.writer(f, delimiter=',')
        mywriter.writerows(df)

#def write_csv(path, list):
#    with open(path,'w') as file:
#        file.write("\n".join(str(item) for item in list))

def split_n(path, num):
    # Create the dataset object
    dataset = pd.read_csv(path, sep=",")
    dataset_protein = dataset.loc[dataset['label'] == 0]
    dataset_nonprotein = dataset.loc[dataset['label'] == 1]
    test_protein = dataset_protein.sample(n = num, random_state = 22)
    test_nonprotein = dataset_nonprotein.sample(n = num, random_state = 22)
    testdf = pd.concat([test_protein, test_nonprotein])
    traindf = dataset[~dataset['name'].isin(testdf['name']).dropna()]
    #trainList = [name.split('.')[0].split('_')[0] for name in dataset['name'] if name not in testList]

    testdf.to_csv('/disk1/abtarget/dataset/split/test.csv', index = False)
    traindf.to_csv('/disk1/abtarget/dataset/split/train.csv', index = False)
    
    return traindf


if __name__ == '__main__':
    #filepath = '/disk1/abtarget/nohup_test08Mar23.out'
    #data = parse_file(filepath)

    #traindf = split_n('/disk1/abtarget/dataset/abdb_dataset_noaug.csv', 100)
    traindf = pd.read_csv('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_train1_norep.csv', sep=",")

    results = parse_nohup('/disk1/abtarget/nohup_test03May23_sara.out')

    #a = [name.split('.')[0].split('_')[0] for name in traindf['name']]
    #name = pd.DataFrame([name.split('.')[0].split('_')[0] for name in traindf['name']], columns = ['name'])

    name = pd.DataFrame([name.split('.')[0].upper() for name in traindf['name']], columns = ['name'])

    traindf['name'] = name

    
    df2 = results.groupby(['target','model']).apply(concat_rows).reset_index(drop=True)
    #df2 = df2[df2['name'].isin(name['name']).dropna()]
    #df3 = df2[df2['name'].isin(name['name']).dropna()]
    #df4 = df2[~df2['name'].isin(name['name']).dropna()]
    df2.to_csv('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_train1_norep_aug.csv', mode = 'a', index = False, header=False)

    '''df3 = traindf[traindf['label'] == 1]
    df3 = df3[~df3['name'].isin(df2['name']).dropna()]
    df4 = pd.DataFrame(columns=['name','seq'])
    dict = {}
    
    for index, row in df3.iterrows():
        name = row['name']
        vh = row['VH']
        vl = row['VL']
        #df4.loc[len(df4)] = [name+'_H', vh]
        #df4.loc[len(df4)] = [name+'_L', vl]
        dict[name+'_H'] = vh
        dict[name+'_L'] = vl
        #print(df4)
    
    #file=open("sabdab_dataset.yml","w")
    #yaml.dump(dict,file)
    #file.close()

    
    #df3.to_csv('/disk1/abtarget/dataset/split/aug_train.csv', index = False)
    #df4.to_csv('/disk1/abtarget/dataset/split/aug_test.csv', index = False)
    print('done')'''
