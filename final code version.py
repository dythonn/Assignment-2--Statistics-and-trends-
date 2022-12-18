import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as ani
import seaborn as sns
from datetime import datetime


# from matplotlib import cm

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)




def read_world_bank_data(filename):
    """ Returns the dataframe with Year column and Country column
        as the dataframe had Countries already in column"""
    
    df = pd.read_csv(filename)
    year_list = list(df.iloc[:, 4:].columns)
    year_df = pd.DataFrame(year_list)
   
    df = pd.melt(df, id_vars=['Country Name','Series Name','Series Code','Country Code'], value_vars = year_list,)
    df['variable'] = df['variable'].str[0:4] # Clean the variable 
    df.rename({'variable': 'Year', 'value': 'Values in %'}, axis=1, inplace=True) # rename the 'variable' column to 'Year' and 'value' column to 'Values in %'
    df['Year'] = pd.to_datetime(df['Year']) # Change the data type of the 'Year' column from object to datetime format
    
    
    return year_df, df


# Call the function to read the datafile
df_by_year, df_by_country = read_world_bank_data('final dataset.csv')
print(df_by_year.head())
print(df_by_country.head())

# =============================================================================
#                             Pre-processing
# =============================================================================


print (df_by_country.info())

#                   Since we have a string in column Values in %
#                           We have to remove them

print(df_by_country.shape)
#replace every '..' to NaN
df_by_country = df_by_country.replace('..', np.nan)
print (df_by_country.shape)

# Making the Values in percentage data to float
df_by_country['Values in %'] = df_by_country['Values in %'].astype(float) #Change the data type of the 'GDP' to numeric


#                   looking for NaN values (dataframe)
print ("")
print(" Total NaN in Dataframe ".center(66,'*'))
print ("")
look_for_nan_df = df_by_country.isnull().sum().sum()
print (look_for_nan_df)

#                   looking for NaN values by columns
print ("")
print(" NaN values by columns ".center(66,'*'))
print ("")
look_for_nan_col = df_by_country.isnull().sum()
print (look_for_nan_col)

#                Dropping the NaN values from DataFrame

df_by_country = df_by_country.dropna()

print (df_by_country.describe())
print (df_by_country)


#                Converting Datetime format to Years only

df_by_country['Year'] = df_by_country['Year'].dt.year

print(df_by_country.head())

df_dup = df_by_country

print (df_dup["Country Code"])
pak = df_dup[df_dup["Country Code"] == 'PAK']
afg = df_dup[df_dup["Country Code"] == 'AFG']
ind = df_dup[df_dup["Country Code"] == 'IND']
usa = df_dup[df_dup["Country Code"] == 'USA']
bgd = df_dup[df_dup["Country Code"] == 'BGD']
gbr = df_dup[df_dup["Country Code"] == 'GBR']
zmb = df_dup[df_dup["Country Code"] == 'ZMB']


""" Indicators For LinePlot """

# Pak
pak_indicator = pak[(pak["Series Code"] == 'SE.PRM.GINT.MA.ZS')]
pak_indicator1 = pak[ (pak["Series Code"] == 'SE.PRM.GINT.FE.ZS')]
pak_indicator2 = pak[(pak["Series Code"] == 'SL.UEM.ADVN.ZS')]

# Zambia
zmb_indicator = zmb[(zmb["Series Code"] == 'SE.PRM.GINT.MA.ZS')]
zmb_indicator1 = zmb[ (zmb["Series Code"] == 'SE.PRM.GINT.FE.ZS')]
zmb_indicator2 = zmb[(zmb["Series Code"] == 'SL.UEM.ADVN.ZS')]

# USA
usa_indicator = usa[(usa["Series Code"] == 'SE.PRM.GINT.MA.ZS')]
usa_indicator1 = usa[ (usa["Series Code"] == 'SE.PRM.GINT.FE.ZS')]
usa_indicator2 = usa[(usa["Series Code"] == 'SL.UEM.ADVN.ZS')]



""" Indicators For BarPlot """

# Pak

bar_pak_indicator = pak[(pak["Series Code"] == 'SE.TER.CUAT.MS.MA.ZS')]
bar_pak_indicator1 = pak[ (pak["Series Code"] == 'SE.TER.CUAT.MS.FE.ZS')]
bar_pak_indicator2 = pak[(pak["Series Code"] == 'SL.TLF.ADVN.MA.ZS')]
bar_pak_indicator3 = pak[(pak["Series Code"] == 'SE.PRM.GINT.FE.ZS')]

# Zambia

bar_zmb_indicator = zmb[(zmb["Series Code"] == 'SE.TER.CUAT.MS.MA.ZS')]
bar_zmb_indicator1 = zmb[ (zmb["Series Code"] == 'SE.TER.CUAT.MS.FE.ZS')]
bar_zmb_indicator2 = zmb[(zmb["Series Code"] == 'SE.PRM.GINT.MA.ZS')]
bar_zmb_indicator3 = zmb[(zmb["Series Code"] == 'SE.PRM.GINT.FE.ZS')]

# USA

bar_usa_indicator = usa[(usa["Series Code"] == 'SE.TER.CUAT.MS.MA.ZS')]
bar_usa_indicator1 = usa[ (usa["Series Code"] == 'SE.TER.CUAT.MS.FE.ZS')]
bar_usa_indicator2 = usa[(usa["Series Code"] == 'SE.PRM.GINT.MA.ZS')]
bar_usa_indicator3 = usa[(usa["Series Code"] == 'SE.PRM.GINT.FE.ZS')]

# AFG

bar_afg_indicator = afg[(afg["Series Code"] == 'SE.TER.CUAT.MS.MA.ZS')]
bar_afg_indicator1 = afg[ (afg["Series Code"] == 'SE.TER.CUAT.MS.FE.ZS')]
bar_afg_indicator2 = afg[(afg["Series Code"] == 'SE.PRM.GINT.MA.ZS')]
bar_afg_indicator3 = afg[(afg["Series Code"] == 'SE.PRM.GINT.FE.ZS')]




def country_counter(con, sc):
    """Takes a Country and indicator and pass their value"""
    
    country = df_dup[df_dup["Country Code"] == con]
    indicator = country [(country ["Series Code"] == 'SE.PRM.GINT.MA.ZS')]
    indicator1 = country [(country["Series Code"] == 'SE.PRM.GINT.FE.ZS')]
    indicator2 = country [(country ["Series Code"] == 'SE.XPD.TOTL.GB.ZS')]
    
    return indicator,indicator1,indicator2



print (country_counter("PAK","SL.UEM.ADVN.ZS"))
# indicator = indicator.set_index('Year')

print (zmb_indicator, zmb_indicator1, zmb_indicator2)
print (pak_indicator, pak_indicator1, pak_indicator2)
print (usa_indicator, usa_indicator1, usa_indicator2)


# =============================================================================
#                             Line Graph
# =============================================================================


def line_graph(country):
    """ Takes in Country Name and returns the statistics/values 
                        for three countries """
    
    plt.figure(figsize=(11, 8))
    
    
    # t_leg_1 = ("Zambia")
    # t_leg_2 = ("Pakistan")
    # t_leg_3 = ("United States of America")
    
    
    # takes x and y as inputs to the plot
    one_indi_yr = pak_indicator['Year']
    one_indi_ = pak_indicator['Values in %']
    two_indi_yr = pak_indicator1['Year']
    two_indi_ = pak_indicator1['Values in %']
    three_indi_yr = pak_indicator2['Year']
    three_indi_ = pak_indicator2['Values in %']
    
    one_indi_yr_z = zmb_indicator['Year']
    one_indi_z = zmb_indicator['Values in %']
    two_indi_yr_z = zmb_indicator1['Year']
    two_indi_z = zmb_indicator1['Values in %']
    three_indi_yr_z = zmb_indicator2['Year']
    three_indi_z = zmb_indicator2['Values in %']
    
    one_indi_yr_u = usa_indicator['Year']
    one_indi_u = usa_indicator['Values in %']
    two_indi_yr_u = usa_indicator1['Year']
    two_indi_u = usa_indicator1['Values in %']
    three_indi_yr_u = usa_indicator2['Year']
    three_indi_u = usa_indicator2['Values in %']
    
    if(country=="PAK"):
        plt.plot(one_indi_yr, one_indi_, linewidth = 1.2)
        plt.plot(two_indi_yr, two_indi_, linewidth = 1.2)
        plt.plot(three_indi_yr, three_indi_, linewidth = 1.2)
        plt.title('Pakistan', fontsize = 18)
        plt.legend(['Gross intake ratio in first grade of primary education, male', 'Gross intake ratio in first grade of primary education, female', 'Government expenditure on education, total (% of government expenditure)'], loc='center', fontsize = 15.5, frameon = False)
        
    elif(country=="ZMB"):
        plt.plot(one_indi_yr_z, one_indi_z, linewidth = 1.2)
        plt.plot(two_indi_yr_z, two_indi_z, linewidth = 1.2)
        plt.plot(three_indi_yr_z, three_indi_z, linewidth = 1.2)
        plt.title('Zambia', fontsize = 18)
        plt.legend(['Gross intake ratio in first grade of primary education, male', 'Gross intake ratio in first grade of primary education, female', 'Government expenditure on education, total (% of government expenditure)'], loc='center', fontsize = 15.5, frameon = False)
        
    elif(country=="USA"):
        plt.plot(one_indi_yr_u, one_indi_u, linewidth = 1.2)
        plt.plot(two_indi_yr_u, two_indi_u, linewidth = 1.2)
        plt.plot(three_indi_yr_u, three_indi_u, linewidth = 1.2) 
        plt.title('United States of America', fontsize = 18)
        plt.legend(['Gross intake ratio in first grade of primary education, male', 'Gross intake ratio in first grade of primary education, female', 'Government expenditure on education, total (% of government expenditure)'], loc='center', fontsize = 15.5, frameon = False)
            
    plt.xlabel('Years', fontsize = 17)
    plt.ylabel('Percentage', fontsize = 17)
    
    
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.tight_layout()
    
    plt.savefig('linePlotLanguages.png')
    plt.show()
    

line_graph("ZMB")
line_graph("USA")
line_graph("PAK")



# =============================================================================
#                               Bar Plot
# =============================================================================



def bar_plot(value_in_per):
    """ Takes a percentage values data and plot with respect to countries
                    and returns the statistics/values """

    N = 4
    ind = np.arange(N)  # the x locations for the groups
    width = 0.22       # the width of the bars

    
    fig = plt.figure()
    # ax = fig.add_subplot(111)
    ax = fig.add_axes([0,0,1,1])
    
    # ddd=bar_zmb_indicator.iloc[2]
    # print(ddd)
    # print(ddd['Values in %'])
    yvals = [pak_indicator.iloc[2]['Values in %'],pak_indicator.iloc[3]['Values in %'],pak_indicator.iloc[4]['Values in %'],pak_indicator.iloc[5]['Values in %']]
    zvals = [zmb_indicator.iloc[2]['Values in %'],zmb_indicator.iloc[3]['Values in %'],zmb_indicator.iloc[4]['Values in %'],zmb_indicator.iloc[5]['Values in %']]
    kvals = [usa_indicator.iloc[2]['Values in %'],usa_indicator.iloc[3]['Values in %'],usa_indicator.iloc[4]['Values in %'],usa_indicator.iloc[5]['Values in %']]
    
    rects1 = ax.bar(ind, yvals, width, color='r')
    rects2 = ax.bar(ind+width, zvals, width, color='g')
    rects3 = ax.bar(ind+width*2, kvals, width, color='b')
    
    ax.set_ylabel('Percentage')
    ax.set_xticks(ind+width)
    ax.set_xticklabels( ('2014', '2016', '2017', '2018') )
    ax.legend( (rects1[0], rects2[0], rects3[0]), ('Pakistan', 'Zambia', 'USA') )
    
    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                    ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.show()
    

bar_plot("")


# =============================================================================
#                             Heatmap Correlation
# =============================================================================


def heatmap_bar(name):
    """Takes Values in % of the indicators data and compares the correlation"""
    
    cc=pd.DataFrame()
    if(name=="PAK"):
        cc["Master's or equivalent, population 25+, male"]=bar_pak_indicator['Values in %'][0:7].values
        cc["Master's or equivalent, population 25+, female"]=bar_pak_indicator1['Values in %'][0:7].values
        cc["Unemployment with advanced education"]=bar_pak_indicator2['Values in %'][0:7].values
        # print(indicator1['Values in %'][0:7].values)
        sns.heatmap(cc.corr(), annot = True);
        plt.show()
    
    elif(name=="ZMB"):
        cc["Master's or equivalent, population 25+, male"]=bar_zmb_indicator['Values in %'][0:7].values
        cc["Master's or equivalent, population 25+, female"]=bar_zmb_indicator1['Values in %'][0:7].values
        cc["Unemployment with advanced education"]=bar_zmb_indicator2['Values in %'][0:7].values
        # print(indicator1['Values in %'][0:7].values)
        sns.heatmap(cc.corr(), annot = True);
        plt.show()
        
    elif(name=="USA"):
        cc["Master's or equivalent, population 25+, male"]=bar_usa_indicator['Values in %'][0:6].values
        cc["Master's or equivalent, population 25+, female"]=bar_usa_indicator1['Values in %'][0:6].values
        cc["Unemployment with advanced education"]=bar_usa_indicator2['Values in %'][0:6].values
        # print(bar_afg_indicator2['Values in %'][0:5].values)
        sns.heatmap(cc.corr(), annot = True);
        plt.show()
    
heatmap_bar("PAK")
heatmap_bar("ZMB")
heatmap_bar("USA")

     

