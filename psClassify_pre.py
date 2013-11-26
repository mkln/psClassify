## IMPORT STUFF

import sys
sys.path.append('/home/desktop/patstat_data/all_code/dbUtils')
import company_legal_id as legal
import consolidate_df as cd
import pandas as pd
import numpy as np
import random
import os
import re
from smart_type_convert import as_str

root_folder = '/home/desktop/patstat_data/all_code/psClean/code/clean/patstat_geocoded/'
labeled_data_folder = '/home/desktop/patstat_data/all_code/psClassify/labeled_data/'
r_data_folder = '/home/desktop/patstat_data/all_code/psClassify/r_data/'

countries = [ 'at', 'bg', 'be', 
        'it', 'gb', 'fr', 'de', 'sk', 'se', 'pt', 'pl',
        'hu', 'ie', 'ee', 'es', 'cy', 
         'nl', 'si', 'ro', 'dk', 
        'cz', 'lt', 'lu', 'lv', 'mt', 'fi', 'gr']

    
common_names = pd.read_csv('/home/desktop/patstat_data/all_code/dbUtils/country_common_person_names.csv',sep='\t')
find_legal = legal.FindLegalId()

## DEFINE STUFF

def wordcount(name):
    return len(name.split())
    
def avg_word_len(name):
    return np.mean([len(s) for s in name.split()])
    
def has_three_words(name):
    return 1 if len(name.split()) == 3 else 0   
        
def has_two_words(name):
    return 1 if len(name.split()) == 2 else 0       


def has_legal_out(row):
    name = row['name']
    country = row['country']
    identifier = find_legal.separate_comp_legalid(name, country, 'outside')[1]
    return 1 if len(identifier) else 0
    
def has_legal_in(row):
    name = row['name']
    country = row['country']
    identifier = find_legal.separate_comp_legalid(name, country, 'inside')[1]
    return 1 if len(identifier) else 0
    
def high_patent_ct(howmany, cutoff=100):
    return 1 if howmany>cutoff else 0
    
def maybe_foreign_legal(name):
    it_does = find_legal.maybe_foreign(name)
    return 1 if it_does else 0
    
def common_first_name(row, commons_df = common_names):
    country = row['country']
    name = row['name']
    common_list = list(commons_df[commons_df.country == country]['names'])[0].split()
    list_words = name.split()
    for w in list_words:
        if w in common_list:
            return 1
    return 0
    
_digits = re.compile('\d')
def only_letters(d):
    d2 = d.replace(' ','')
    return 0 if (bool(_digits.search(d2)) or not d2.isalnum()) else 1

## LOAD STUFF
if 'labels.csv' in os.listdir(labeled_data_folder):
    old_data = pd.read_csv(labeled_data_folder + 'labels.csv', sep='\t')
    old_data = old_data[['patstat_id', 'is_person']].set_index('patstat_id')
    already_loaded = 1
else:
    already_loaded = 0

df_large = pd.DataFrame()
for country in countries:
    print 'loading', country.upper()
    f_in = root_folder + 'patstat_geocoded_%s.csv' % country
    df = pd.read_csv(f_in, sep='\t')
    df['country'] = country           
    df_large = df_large.append(df)
    
df_large.columns = ['applicant_seq', 
                    'coauthors', 
                    'country', 
                    'extracted_address', 
                    'inventor_seq', 
                    'ipc_code',
                    'lat', 'lng',
                    'name_abbreviated',
                    'patent_ct',
                    'patstat_id',
                    'name',
                    'year'] 
                    
df_large = df_large[['country', 'name', 
                    'patstat_id', 
                    'patent_ct', 'name_abbreviated',
                    'applicant_seq', 'inventor_seq']]

df_large = df_large.reset_index(drop=True)
"""                
# CONSOLIDATE DATA

mean_app = df_large.applnt_seq.values.mean()
df_large['applnt_seq'] = df_large['applnt_seq'].astype(str)
                          
consolidate_dict = {'name': cd.consolidate_unique,
                    'country': cd.consolidate_unique,
                    'patent_ct': sum,
                    'patstat_id': cd.consolidate_id,
                    'applnt_seq': cd.consolidate_id,
                    ''}
                    
print 'consolidating dataframe...'
df_large = cd.consolidate(df_large, ['name', 'country'], consolidate_dict)
df_large = df_large[['patstat_id','name','country','applnt_seq','patent_ct']].set_index('patstat_id')


# ADD VARS

print 'adding applicant score...'
app_score = []
for i, row in df_large.iterrows():
    if row['applnt_seq']:
        score = sum([int(s.replace("'",""))-mean_app for s in row['applnt_seq'].split('**')])
    else:
        score = None
    app_score.append(score)

df_large['applnt_score'] = app_score
"""

old_data = pd.merge(old_data.reset_index(), df_large, how='inner')
old_data = old_data.set_index('patstat_id')
old_data = old_data[['is_person']]

df_large['name'] = df_large['name'].apply(as_str)
print 'adding word count...'
df_large['word_count'] = df_large['name'].apply(wordcount)

print 'adding average word length...'
df_large['avg_word_len'] = df_large['name'].apply(avg_word_len)

print 'adding total string length...'
df_large['string_len'] = df_large['name'].apply(len)

#print 'adding dummy: has 2 words...'
#df_large['has_2_words'] = df_large['name'].apply(has_two_words)

#print 'adding dummy: has 3 words...'
#df_large['has_3_words'] = df_large['name'].apply(has_three_words)

print 'adding dummy: only letters...'
df_large['only_letters'] = df_large['name'].apply(only_letters)

print 'adding dummy: large patent count...'
df_large['lots_of_patents'] = df_large['patent_ct'].apply(high_patent_ct)

print 'adding dummy: has legal id outside...'
df_large['has_legal_out'] = [has_legal_out(r) for i, r in df_large.iterrows()]

print 'adding dummy: has legal id inside...'
df_large['has_legal_in'] = [has_legal_in(r) for i, r in df_large.iterrows()]

print 'adding dummy: possibly foreign legal id...'
df_large['maybe_foreign_legal'] = df_large['name'].apply(maybe_foreign_legal)

print 'adding dummy: has first name...'
df_large['has_first_name'] = [common_first_name(r) for i, r in df_large.iterrows()]

# CERTAINLY THEY ARE NOT PERSON NAMES...

df_large['certain_not_person'] = 0
# has country-appropriate legal id at beginning or end of name
df_large.certain_not_person[df_large.has_legal_out == 1] = 1
# has numbers or non-alphanum characters in name
df_large.certain_not_person[df_large.only_letters == 0] = 1
# has more than 100 patents
df_large.certain_not_person[df_large.lots_of_patents == 1] = 1
# has same name as entity to which we assigned a 1
df_copy = pd.DataFrame(df_large.groupby('name').size(), columns=['size'])
dup_names = set(df_copy[df_copy.size>1].index)
certains = set(df_large[df_large.certain_not_person == 1].name)
dup_certain = list(set.intersection(dup_names, certains))

df_large = df_large.set_index('name')
df_large['certain_not_person'].loc[dup_certain] = 1
df_large = df_large.reset_index()

# RANDOM SAMPLE WHERE WE DONT HAVE LABEL

if not already_loaded:
    random.seed(1)
else:
    random.seed(len(old_data))
random_sample = random.sample(df_large[df_large.certain_not_person == 0].index, 100)

# CLASSIFY EACH ROW

assigned_class = []
for nn, pid in enumerate(random_sample):
    print nn, 'done.'
    os.system('clear')
    print df_large.loc[pid][['applnt_seq', 'name', 'patent_ct']]
    
    hand_label = -1
    while not hand_label in ['0','1','exit']:
        print '\n\n\n\n', '[1] PERSON NAME     or      [0] NOT PERSON NAME ?'
        hand_label = raw_input('>>')
    
    if hand_label != 'exit':
        assigned_class.append((pid, int(hand_label)))
    else:
        break

labeled_data = pd.DataFrame(assigned_class, columns=['patstat_id', 'is_person'])
labeled_data = labeled_data.set_index('patstat_id')
if already_loaded:
    labeled_data = old_data.append(labeled_data)

# SAVE
labeled_data.to_csv(labeled_data_folder + 'labels.csv', sep='\t', mode='w')

#labeled_data = old_data
labeled_data = labeled_data.join(df_large.set_index('patstat_id'), how='outer')
labeled_data = labeled_data[pd.notnull(labeled_data.name)]
### MOVE TO R ###
# export only if certain_not_person

# ALL DATA - no labels
R_all = labeled_data #.drop(['name', 'country', 'applnt_seq'], 1)
R_all.to_csv(r_data_folder + 'r_input.csv', sep='\t')
