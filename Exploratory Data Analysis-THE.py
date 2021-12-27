
#add this line in new_branch

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import re
import os
#import string

#Load and clean data
def load_excel_data(sheet_name, data_set_name = 'Dataset.xlsx', header = 0, index_col = None):
    df = pd.read_excel(data_set_name, sheet_name, header = header, index_col = index_col)  
    if sheet_name in ['WUR2017','WUR2018','WUR2019']:
        
        for col in df.columns[2:]:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(replace_range)
                
                df[col].replace("[a-zA-Z]+", np.nan, regex = True, inplace = True)
        
        df['rank_label'].replace('\D','',regex = True, inplace = True)
        df['rank_label'] = df['rank_label'].astype(int)
    return df

def replace_range(item):
    if re.match("[\d]*–[\d]*", str(item)):
        for digit in re.finditer("(?P<low>^[\d]*)–(?P<high>[\d]*$)", item):
            low_range = int(digit.groupdict()['low'])
            high_range = int(digit.groupdict()['high'])
            mean = (low_range + high_range) /2
            return int(mean)
        
    elif re.match("[\d]*\.\d–[\d]*\.\d", str(item)):
        for digit in re.finditer("(?P<low>^[\d]*\.\d)–(?P<high>[\d]*\.\d$)", item):
            low_range = float(digit.groupdict()['low'])
            high_range = float(digit.groupdict()['high'])
            mean = (low_range + high_range) /2
            return float(mean)
    else:
        return item

# Where to save the figures
PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure ", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# Load the four Dataframes
df_2017, df_2018, df_2019, df_keys = map(load_excel_data, ['WUR2017','WUR2018','WUR2019','Key'])


df_2017.replace({'Hong Kong University of Science and Technology':'The Hong Kong University of Science and Technology'},
              inplace=True)

concatenate_dfs = pd.concat([df_2017, df_2018, df_2019], keys = ['2017','2018','2019']).reset_index(level = 0).reset_index()

#checks
describe = concatenate_dfs.describe()

concatenate_dfs.columns[concatenate_dfs.dtypes=='object']

#%%
#correlations

concatenate_dfs_copy = concatenate_dfs.copy()
concatenate_dfs_copy.isnull().sum() # the columns teaching, research and international have NaN values, better to drop them
concatenate_dfs_copy.dropna(inplace = True)
concatenate_dfs_copy['rank_label'] = concatenate_dfs_copy['rank_label'].astype(int)
concatenate_dfs_copy['overall_score'] = concatenate_dfs_copy['overall_score'].astype(float)

concatenate_dfs_copy.rename(columns={"rank_label": "rank", "overall_score": "overall", "teaching_score": "teaching",
                                "research_score": "research","citation_score": "citation","industry_score": "industry",
                                "international_score": "international"}, inplace = True)

concatenate_dfs_copy = concatenate_dfs_copy.sort_values(by = 'rank')

def compute_correlation(col1, col2):
    
    x_data = concatenate_dfs_copy[col1]
    y_data = concatenate_dfs_copy[col2]
    corr_score = x_data.corr(y_data)

    return x_data, y_data, corr_score  

x_data, y_data, corr_score = compute_correlation('rank', 'overall')
#x_data, y_data, corr_score = compute_correlation('citation', 'overall')
#x_data, y_data, corr_score = compute_correlation('teaching', 'overall')
#x_data, y_data, corr_score = compute_correlation('research', 'overall')
x_data, y_data, corr_score = compute_correlation('international', 'overall')

from scipy.stats import ttest_ind
ttest_ind(x_data, y_data)

from scipy.optimize import curve_fit

def objective(x, a, b):
	return a * x + b

def plot_correlation(x_data, y_data, corr_score):
    
    popt, _ = curve_fit(objective, x_data, y_data)
    a, b = popt
    string = 'y = {} * x + {}'.format(a,b)
    print(string)
    fig, ax = plt.subplots()
    ax.scatter(x_data, y_data, label = 'Data')
    x_line = np.arange(min(x_data), max(x_data), 1)
    y_line = objective(x_line, a, b)   
    plt.plot(x_line, y_line, '--', color='red', label = 'Linear Fitting')
    
    ax.set_xlabel('Industry Score',fontsize=12)
    ax.set_ylabel('Overall Score',fontsize=12)
    ax.legend(loc = 'best')
    ax.set_ylim([0, 100])
    #ax.set_xlim([0, 100])
    plt.text(min(x_data)*1.5, min(y_data)*1.1, 'Corr score = {:.1}'.format(corr_score), fontsize=13, fontweight="bold")

    ax.spines['right'].set_visible(False)  
    ax.spines['top'].set_visible(False)  

plot_correlation(x_data, y_data, corr_score)
save_fig("Correlation industry-overall")

#SCATTER MATRIX
from pandas.plotting import scatter_matrix

attributes = ["rank", "overall", "teaching",
              "research", "citation", "industry",
              "international"]
sm = scatter_matrix(concatenate_dfs_copy[attributes], figsize=(17.5, 11), diagonal = 'hist',hist_kwds={'bins':20})
_ = [s.tick_params('x',rotation = 0) for s in sm.reshape(-1)]

[s.xaxis.get_label().set_fontsize(11.5) for s in sm.reshape(-1)]
[s.yaxis.get_label().set_fontsize(11.5) for s in sm.reshape(-1)]
save_fig("scatter plot")
#%%
# Polynomial Regression
def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results
polyfit(x_data, y_data,1)
#%%
#PLOT SCORES
overall_ranking_df = concatenate_dfs[concatenate_dfs['subject'] == 'Overall'].set_index('university')#.sort_values(by = 'rank_label')
fig, ax = plt.subplots(figsize = (15,6))
for index, year in enumerate(['2017','2018','2019']):
    current_year = overall_ranking_df[overall_ranking_df['level_0'] == year].sort_values(by = ['university'])
    width = 0.25
    x_locations = np.arange(1-width, 11-width,1) + index*width
    ax.bar(x_locations, current_year['overall_score'].astype(float),width = width, label = year)
    for i,v in enumerate(current_year['overall_score'].astype(float)):
        plt.text(x_locations[i] -0.08, float(v) + 1, int(v))

ax.spines['right'].set_visible(False)  
ax.spines['left'].set_visible(False)  
ax.spines['top'].set_visible(False)  
ax.get_yaxis().set_visible(False)   
ax.legend(loc = 'upper left')  
new_labels = []
for name in current_year.index:
    new_name = '\n'.join(name.split())
    new_labels += [new_name]
plt.xticks(np.arange(1,len(overall_ranking_df.index.unique())+1),new_labels,fontsize=11);
ax.set_title('Overall University Ranking from 2017 to 2019',fontsize=13, fontweight="bold")
ax.set_ylim([0, 100])

#%%
# plot overall university ranking over the 3 year period
overall_ranking_df = concatenate_dfs[concatenate_dfs['subject'] == 'Overall'].set_index('university')#.sort_values(by = 'rank_label')
fig, ax = plt.subplots(figsize = (15,6))
for index, year in enumerate(['2017','2018','2019']):
    current_year = overall_ranking_df[overall_ranking_df['level_0'] == year].sort_values(by = ['university','rank_label'])
    width = 0.25
    x_locations = np.arange(1-width, 11-width,1) + index*width
    ax.bar(x_locations, current_year['rank_label'].astype(int),width = width, label = year)
    for i,v in enumerate(current_year['rank_label']):
        plt.text(x_locations[i] -0.08, float(v) + 1, v)

ax.spines['right'].set_visible(False)  
ax.spines['left'].set_visible(False)  
ax.spines['top'].set_visible(False)  
ax.get_yaxis().set_visible(False)   
ax.legend(loc = 'upper left')  
new_labels = []
for name in current_year.index:
    new_name = '\n'.join(name.split())
    new_labels += [new_name]
plt.xticks(np.arange(1,len(overall_ranking_df.index.unique())+1),new_labels,fontsize=11);
ax.set_title('Overall University Ranking from 2017 to 2019',fontsize=13, fontweight="bold")
ax.set_ylim([0, 60])
plt.gca().invert_yaxis()
#save_fig("overall ranking over 3 years")

current_year = overall_ranking_df[overall_ranking_df['level_0'] == '2018']
current_year['rank_label']
for name in current_year.index.unique():
    print(name)
    new_name = '\n'.join(name.split())
    new_labels += [new_name]
#%%
#OPTION BARH

overall_ranking_copy = overall_ranking_df.reset_index()
year_dict = {}
for index, year in enumerate(['2019','2018','2017']):
    year_dict[year] = overall_ranking_copy[overall_ranking_copy['level_0'] == year][['university','rank_label']].sort_values(by = ['university'])

merge_2017_2018 = year_dict['2017'].set_index('university').merge(year_dict['2018'].set_index('university'), how='inner' ,left_index = True, right_index = True)
merge_all_years = merge_2017_2018.merge(year_dict['2019'].set_index('university'), how='inner' ,left_index = True, right_index = True)
merge_all_years.rename(columns={"rank_label_x": "2017", "rank_label_y": "2018", "rank_label": "2019"}, inplace = True)
#if ranked from 2017:
#merge_all_years = merge_all_years.sort_values(by = ['rank_label_x','rank_label_y'])
#labels_barh = ['Carnegie\nMellon\nUniversity',
# 'National\nUniversity\nof Singapore',
# 'Peking\nUniversity',
# 'École Polytechnique\nFédérale\nde Lausanne',
# 'University\nof\nMelbourne',
# 'Georgia\nInstitute\nof Technology',
# 'KU\nLeuven',
# 'University\nof\nHong Kong',
# 'Technical\nUniversity\nof Munich',
# 'The Hong Kong\nUniversity\nof\nScience\nand Technology']

#if ranked from 2019:
merge_all_years = merge_all_years.sort_values(by = ['2019'])
labels_barh = ['National\nUniversity\nof Singapore',
 'Carnegie\nMellon\nUniversity',
 'Peking\nUniversity',
 'University\nof Melbourne',
 'Georgia Institute\nof Technology',
 'École\nPolytechnique\nFédérale\nde Lausanne',
 'University\nof\nHong Kong',
 'The Hong Kong\nUniversity of\nScience and Technology',
 'Technical\nUniversity\nof Munich',
 'KU\nLeuven']

fig, ax = plt.subplots(figsize = (8,10))
for index, year in enumerate(['2019','2018','2017']):   

    y_locations = np.arange(10.3,1.,-1)-index*0.3
    ax.barh(y_locations, merge_all_years[year].astype(int),height = 0.3, label = year)
    for i,v in enumerate(merge_all_years[year]):
        plt.text(float(v) + 1, y_locations[i] -0.08, v)
        
ax.spines['right'].set_visible(False)  
ax.spines['bottom'].set_visible(False)  
ax.spines['top'].set_visible(False)  
ax.get_xaxis().set_visible(False)   
ax.legend(loc = 'upper right')  

def split_join(df):
    new_labels = []
    for name in df.index:
        new_name = '\n'.join(name.split())
        new_labels += [new_name]
    return new_labels

#labels_barh = split_join(merge_all_years)

plt.yticks(np.arange(1,len(overall_ranking_df.index.unique())+1), labels_barh[::-1], fontsize=11);
ax.set_title('Overall University Ranking from 2017 to 2019',fontsize=13, fontweight="bold")
ax.set_xlim([0, 60])
save_fig("overall ranking over 3 years_barh_sorted_2019") 

#%%
pillars_list = ['teaching_score','research_score', 'citation_score','industry_score','international_score']
colors = ['#FF8F8F','#EDFE96','#96FEA5','#96EFFE','#C396FE']
          
def boxplot_year(year):
    university_year = concatenate_dfs[concatenate_dfs['level_0'] == year].groupby('university').apply(lambda row: 
        row[['university','subject','teaching_score','research_score', 'citation_score','industry_score','international_score']]).set_index('university')
    CMU_stats = university_year[university_year['subject'] == 'Overall'].loc['Carnegie Mellon University'][pillars_list]
    NUS_stats = university_year[university_year['subject'] == 'Overall'].loc['National University of Singapore'][pillars_list]
    
    #BOX PLOT
    ax, box1= concatenate_dfs[concatenate_dfs['level_0'] == year].boxplot(column = pillars_list, grid = False,
                   return_type = 'both', whis='range', figsize = (10,4),patch_artist=True, zorder=2)
    ax.set_title('{} - Score distribution of each pillar'.format(year),fontsize=13, fontweight="bold")
    ax.spines['right'].set_visible(False)  
    ax.spines['left'].set_visible(False)  
    ax.spines['top'].set_visible(False)  
    ax.grid(axis='y',linewidth=0.5,zorder=1)
    ax.set_ylim([0, 119])
    for index, box in enumerate(box1['boxes']):
        plt.setp(box, facecolor = colors[index])
    for index, box in enumerate(box1['medians']):
        plt.setp(box, color = 'k',linewidth = 1.2)
    
    ax.scatter(range(1,6), NUS_stats.values, color = 'k', zorder=3, label = 'National University of Singapore')
    #ax.scatter(range(1,6), HK_science_tech_stats.values, color = 'r', zorder=3, label = 'The Hong Kong University of Science and Technology')
    #for uni in university_2017.index.unique():
        #uni_stats = university_2017[university_2017['subject'] == 'Overall'].loc[uni][pillars_list]
        #ax.scatter(range(1,6), uni_stats.values, zorder=3, label = uni)
    
    ax.legend(loc = 'best')
    plt.xticks(np.arange(1,6), ['Teaching\nScore','Research\nScore', 'Citation\nScore','Industry\nScore','international\nScore'], fontsize=11);
    #ax.set_xlabel('Score Type', fontsize=12, fontweight="bold")
    #ax.set_ylabel('Score', fontsize=12, fontweight="bold")
    #save_fig("Score distribution of each pillar_{}".format(year))
    
    #Get data from the boxplot
    def get_box_plot_data(labels, bp):
        rows_list = []
    
        for i, pillar in enumerate(labels):
            dict1 = {}
            dict1['label'] = labels[i]
            dict1['lower_whisker'] = bp['whiskers'][i*2].get_ydata()[1]
            dict1['lower_quartile'] = np.percentile(concatenate_dfs[concatenate_dfs['level_0'] == '2017'][pillar],25)
            dict1['median'] = bp['medians'][i].get_ydata()[1]
            dict1['upper_quartile'] = np.percentile(concatenate_dfs[concatenate_dfs['level_0'] == '2017'][pillar],75)
            dict1['upper_whisker'] = bp['whiskers'][(i*2)+1].get_ydata()[1]
            rows_list.append(dict1)
    
        return pd.DataFrame(rows_list)
    boxplot_data = get_box_plot_data(pillars_list, box1)
    return boxplot_data, NUS_stats
boxplot_data, stats = boxplot_year('2017')

#%%
from math import pi

def radar_plot(uni_name,acronim, year):
    df = concatenate_dfs[(concatenate_dfs['university'] == uni_name) & (
            concatenate_dfs['level_0'] == year)][
    ['subject','teaching_score','research_score', 'citation_score','industry_score','international_score']].set_index('subject').sort_values(by = ['subject'])
    
    pillars_list = ['Teaching\n(30%)','Research\n(30%)', 'Citation\n(30%)','Industry\n(2.5%)','International\n(7.5%)']
    colors = ['b','r','g','c','m','k']
    
    N = len(pillars_list)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    size = len(df.index)
    if size <= 8:
        fig, (ax,ax1,ax2) = plt.subplots(1, 3, subplot_kw=dict(projection='polar'), figsize = (18,8))
        axes = (ax,ax1,ax2)
        floor_div = 2
    else :
        fig, (ax,ax1,ax2,ax3) = plt.subplots(1, 4, subplot_kw=dict(projection='polar'), figsize = (21,7))
        axes = (ax,ax1,ax2,ax3)
        floor_div = 3

    #OVERALL_stats
    values_df= df.loc['Overall']
    values_df= values_df.append(pd.Series(values_df[0]))
    #print(values_df)
    
    # Plot data
    ax.plot(angles, values_df, linewidth=1.5, linestyle='solid', color = 'b', label = 'Overall')
    ax.fill(angles, values_df, color = 'b', alpha=0.2)
    ax.set_rlabel_position(0)
    ax.set_yticklabels(['20','40','60','80'],color="black", size=9.5)
    
    first_slicing = size//floor_div
    for index, subject in enumerate(df.iloc[:first_slicing].index):
        if subject == 'Overall':
            continue
        values_df= df.loc[subject]
        values_df= values_df.append(pd.Series(values_df[0]))
        ax1.set_rlabel_position(0)
    
        # Plot data
        ax1.plot(angles, values_df, linewidth=1.5, linestyle='solid', color = colors[index], label = subject)
        ax1.fill(angles, values_df, color = colors[index], alpha=0.1)
    
    if size <=8:
        end_slicing = size
    else:
        end_slicing = 2*(size//3)+1
    for index, subject in enumerate(df.iloc[first_slicing:end_slicing].index):
        if subject == 'Overall':
            continue
        values_df= df.loc[subject]
        values_df= values_df.append(pd.Series(values_df[0]))
        ax2.set_rlabel_position(0)
        # Plot data
        ax2.plot(angles, values_df, linewidth=1.5, linestyle='solid', color = colors[::-1][index], label = subject)
        ax2.fill(angles, values_df, color = colors[index], alpha=0.1)
    
    if size > 8:
        for index, subject in enumerate(df.iloc[end_slicing:].index):
            if subject == 'Overall':
                continue
            values_df= df.loc[subject]
            values_df= values_df.append(pd.Series(values_df[0]))
            ax3.set_rlabel_position(0)
            # Plot data
            ax3.plot(angles, values_df, linewidth=1.5, linestyle='solid', color = colors[index], label = subject)
            ax3.fill(angles, values_df, color = colors[index], alpha=0.1)
    
    for axis in axes:
        ticklabels = axis.get_xticklabels()
        ticklabels[0].set_ha("left")
        ticklabels[1].set_ha("left")
        ticklabels[2].set_ha("right")
        ticklabels[3].set_ha("right")
        ticklabels[4].set_ha("left")
        axis.set_xticklabels(pillars_list,color = 'grey')
        axis.set_yticks([20,40,60,80,100],['20','40','60','80','100'])
        axis.set_yticklabels(['20','40','60','80'],color="grey", size=9.5)
        axis.legend(loc = (0.3,1.1))
    
    plt.setp(axes, xticks=angles, xticklabels=pillars_list)
    fig.tight_layout()
    title = '{} - {} Score of single departments'.format(acronim, year)
    print(title)
    fig.suptitle('{}: {} Score of single departments'.format(acronim, year),fontsize=13, fontweight="bold")
    #save_fig(title) 
    return df

#df_HKUST_2017 = radar_plot('The Hong Kong University of Science and Technology','HKUST', '2017')
#df_HKUST_2019 = radar_plot('The Hong Kong University of Science and Technology','HKUST', '2019')

df_EPFL_2017 = radar_plot('École Polytechnique Fédérale de Lausanne','EPFL', '2017')
df_EPFL_2019 = radar_plot('École Polytechnique Fédérale de Lausanne','EPFL', '2019')

#df_NUS_2017 = radar_plot('National University of Singapore','NUS', '2019')

#df_EPFL_2017 = radar_plot('KU Leuven','KUL', '2017')
#df_EPFL_2019 = radar_plot('KU Leuven','KUL', '2019')
#%%
#pie chart
y = np.array([30, 30, 30, 7.5, 2.5])
attributes_sizes = np.array([10, 10, 10, 10, 10])*1.5

attributes = ["Teaching",
              "Research", "Citation", "International\nOutlook","Industry"]

fig,ax = plt.subplots(figsize = (10,10))             
partches, texts, autopct = ax.pie(y, labels = attributes,autopct='%1.1f%%', explode = [0.025]*5)
for index, text in enumerate(texts):
    text.set_fontsize(attributes_sizes[index])
    text.set_weight('bold')
[autopc.set_fontsize(15) for autopc in autopct]
#save_fig("pie chart")
#%%
#Best
departments = concatenate_dfs['subject'].unique()
best_departments = df_2019.groupby('subject').apply(lambda row: row[row['rank_label'] == row['rank_label'].min()][['rank_label','university']])
concatenate_dfs[concatenate_dfs['rank_label'] == concatenate_dfs['rank_label'].max()]['rank_label']
concatenate_dfs['rank_label'].idxmin()
