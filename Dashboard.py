import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
from functions import *
import sys
import os
import radioactivedecay as rad
sys.path.append('../')


st.title('Dosimetry profiling')

cols=st.columns(2)

st.subheader('Insert the first-order rate constants for your system')
widget_selector=cols[0].radio('Choose your favorite widget', options=['Slider', 'Number input'])
half_life_selector=cols[1].radio('Choose between half-lives and rate constants', options=['Rate constants', 'Half-lives'])

if widget_selector=='Slider' and half_life_selector=='Rate constants':
    k1=st.slider('k1', min_value=0.00001, max_value=10.0,value=0.05,format='%.5f')
    k2=st.slider('k2', min_value=0.00001, max_value=10.0,value=0.001,format='%.5f')
    idg=st.slider('% Injected dose / g', min_value=0.00000,max_value=100.0, value=1. )
    init_activity=st.slider('Initial activity in MBq', min_value=0.00001, max_value=100., value=1.)


elif widget_selector=='Number input' and half_life_selector=='Rate constants':
    k1=st.number_input('k1', min_value=0.00001,value=0.05, format='%.5f')
    k2=st.number_input('k2', min_value=0.00001,value=0.001, format='%.5f')
    idg=st.number_input('% Injected dose / g', min_value=0.00001,max_value=100.0, value=1. )
    init_activity=st.number_input('Initial activity in MBq', min_value=0.00001, max_value=100., value=1.)

elif widget_selector=='Number input' and half_life_selector=='Half-lives':
    lambda1=st.number_input('λ1', min_value=0.00001,value=25., format='%.5f')
    lambda2=st.number_input('λ2', min_value=0.00001,value=50., format='%.5f')
    idg=st.number_input('% Injected dose / g', min_value=0.00001,max_value=100.0, value=1. )
    init_activity=st.number_input('Initial activity in MBq', min_value=0.00001, max_value=100., value=1.)
    k1=np.log(2)/lambda1
    k2=np.log(2)/lambda2
    
elif widget_selector=='Slider' and half_life_selector=='Half-lives':
    lambda1=st.slider('λ1', min_value=0.00001,value=25., max_value=100., format='%.5f')
    lambda2=st.slider('λ2', min_value=0.00001,value=50., max_value=100., format='%.5f')
    idg=st.slider('Injected dose / g', min_value=0.00001,max_value=100.0, value=1. )
    init_activity=st.slider('Initial activity in MBq', min_value=0.00001, max_value=100., value=1.)
    k1=np.log(2)/lambda1
    k2=np.log(2)/lambda2

t_max_graph=st.number_input('Select the maximum time (in h) for the graph', min_value=0.00001, max_value=10000., value=1000.)

half_life_intake=np.log(2)/k1
half_life_excretion=np.log(2)/k2

x_axis_plot=10*(max(half_life_excretion, half_life_intake))

y_bateman=[]
time=[]


for i in np.linspace(0, t_max_graph, 1000):
    y_bateman.append(bateman(i, 1,k1, k2))
    time.append(i)

time=np.array(time, dtype=float)
y_bateman=np.array(y_bateman, dtype=float)
y_bateman=y_bateman/(max(y_bateman)) * (idg)
bateman_df=pd.DataFrame([time, y_bateman])
bateman_df=bateman_df.T
bateman_df.columns=['time', 'C(t)']

chart1=alt.Chart(bateman_df).mark_line().encode(alt.X('time'), alt.Y('C(t)', axis=alt.Axis(format=".3")).title('% ID / g'))

st.altair_chart(chart1, theme=None, use_container_width=True)

csv_bateman=bateman_df.to_csv().encode('utf-8')

st.download_button('Download data from chart as .csv file', data=csv_bateman, file_name='kinetics_non_decay.csv')

y_bateman_activity=y_bateman*init_activity

l=[]
l1=[]

for i in os.listdir('Nuclide data'):
    j=str.replace(i,'mird-', '',)
    l.append(j)

for i in range(len(l)):
    m=str.replace(l[i],'-table-0.csv', '')
    l1.append(m)

l1.remove('.DS_Store')
l1.remove('.ipynb_checkpoints')
l1=np.array(l1, dtype='str')
l1=np.sort(l1)

l2=[]
for i in range(len(l1)):
    my_string=l1[i]
    number=''.join([j for j in my_string if j.isdigit()])
    letters=''.join([j for j in my_string if j.isalpha()])
    letters=letters.capitalize()
    element=number + ' ' +letters
    l2.append(element)

nuclide_selection_df=pd.DataFrame([l1,l2], ['Unformatted', 'Formatted']).T
nuclide_selection_df.sort_values(by=['Formatted'], ascending=True)


sel_nucl=st.selectbox('Nuclide', nuclide_selection_df['Formatted'])

unformatted_sel_nucl=nuclide_selection_df[nuclide_selection_df['Formatted']==sel_nucl].index
unformatted_nuclide_name=nuclide_selection_df['Unformatted'].iloc[unformatted_sel_nucl]
unformatted_nuclide_name=unformatted_nuclide_name.values[0]


st.write('The half-life of the selected nuclide (expressed in readable format) is : ' + rad.Nuclide(str(sel_nucl)).half_life('readable'))

half_life_nuc=rad.Nuclide(str(sel_nucl)).half_life('hours')

k_decay=np.log(2)/rad.Nuclide(str(sel_nucl)).half_life('hours')

timescale=min([half_life_excretion, half_life_nuc])

time=np.array(np.linspace(0,20*timescale, 10000))

activity_vs_time=activity_calc(np.linspace(0,20*timescale, 10000), k1, k2, k_decay, idg, init_activity)

activity_vs_time_df=pd.DataFrame([time, activity_vs_time])

activity_vs_time_df=activity_vs_time_df.T
activity_vs_time_df.columns=['time', 'Activity (t)']



chart2=alt.Chart(activity_vs_time_df).mark_line(clip=True).encode(alt.X('time', scale=alt.Scale(domain=[0, t_max_graph])), alt.Y('Activity (t)').title('Activity (MBq/g)'))

st.altair_chart(chart2, theme=None, use_container_width=True)

particle_table=read_nuclide_csv(unformatted_nuclide_name)


beta='β'
alpha='α'
auger='Aug'
conv_elec='ce'

st.subheader('Include particles for dosimetry')

columns0=st.columns(4)

alpha_disabler=False

particle_table

if extract_dose_particle(particle_table, alpha)!=0.0:
    alpha_selector=columns0[0].checkbox('α', disabled=alpha_disabler, value=True)
    st.write(extract_dose_particle(particle_table, alpha))
    alpha_dose=extract_dose_particle(particle_table, alpha)
else:
    alpha_selector=columns0[0].checkbox('α', disabled=True, value=False)

if extract_dose_particle(particle_table, beta)!=0.0:
    beta_selector=columns0[1].checkbox('β', value=True)
    beta_dose=extract_dose_particle(particle_table, beta)
else:
    beta_selector=columns0[1].checkbox('β', disabled=True, value=False)

if extract_dose_particle(particle_table, auger)!=0.0:
    auger_selector=columns0[2].checkbox('Auger',value=True)
    auger_dose=extract_dose_particle(particle_table, auger)
else:
    auger_selector=columns0[2].checkbox('Auger', disabled=True, value=False)

if extract_dose_particle(particle_table, conv_elec)!=0.0:
    ce_selector=columns0[3].checkbox('Conversion Electrons', value=True)
    ce_dose=extract_dose_particle(particle_table, conv_elec)
else:
    ce_selector=columns0[3].checkbox('Conversion Electrons', disabled=True, value=False)

number_of_decays=np.trapz(activity_vs_time)

cumulative_dose_df=pd.DataFrame()

auc_to_t=[]

for i in range(len(time)):
    auc_to_t.append(np.trapz(activity_vs_time[0:i], time[0:i]))

auc_to_t=np.array(auc_to_t)*1E6*3600

print(auc_to_t)
auc_to_t_df=pd.DataFrame([time, auc_to_t])
auc_to_t_df=auc_to_t_df.T
auc_to_t_df.columns=['time', 'Decays total']


dose_particle_df=pd.DataFrame([time])
dose_particle_df=dose_particle_df.T
dose_particle_df.columns=['time']



if alpha_selector==True:
    cumulative_alpha_dose=number_of_decays*alpha_dose*1.6022e-13*1E6*3600*1000
    cumulative_dose_df.insert(len(cumulative_dose_df.columns),'Alpha Total Dose', [cumulative_alpha_dose])
    alpha_contribution=auc_to_t*alpha_dose*1.6022e-13*1000
    dose_particle_df.insert(len(dose_particle_df.columns), 'Alpha Dose', alpha_contribution)
    
if beta_selector==True:
    cumulative_beta_dose=number_of_decays*beta_dose*1.6022e-13*1E6*3600*1000
    cumulative_dose_df.insert(len(cumulative_dose_df.columns),'Beta Total Dose', [cumulative_beta_dose])
    beta_contribution=auc_to_t*beta_dose*1.6022e-13*1000
    dose_particle_df.insert(len(dose_particle_df.columns), 'Beta Dose', beta_contribution)
    
if auger_selector==True:
    cumulative_auger_dose=number_of_decays*auger_dose*1.6022e-13*1E6*3600*1000
    cumulative_dose_df.insert(len(cumulative_dose_df.columns),'Auger Total Dose', [cumulative_auger_dose])
    auger_contribution=auc_to_t*auger_dose*1.6022e-13*1000
    dose_particle_df.insert(len(dose_particle_df.columns), 'Auger Dose', auger_contribution)

if ce_selector==True:
    cumulative_ce_dose=number_of_decays*ce_dose*1.6022e-13*1E6*3600*1000
    cumulative_dose_df.insert(len(cumulative_dose_df.columns),'CE Total Dose', [cumulative_ce_dose])
    ce_contribution=auc_to_t*ce_dose*1.6022e-13*1000
    dose_particle_df.insert(len(dose_particle_df.columns), 'CE Dose', ce_contribution)


dose_particle_df['Sum']=dose_particle_df[dose_particle_df.columns[1:]].sum(axis=1)

st.subheader('Cumulative dose')

cumulative_dose_df

st.subheader('Time-dependent dose per particle')


dose_particle_df1=dose_particle_df[::50]

df2=dose_particle_df1.melt(id_vars='time')


chart3=alt.Chart(auc_to_t_df).mark_line().encode(alt.X('time').title('time(h)'), alt.Y('Decays total', axis=alt.Axis(tickCount=5, format=".1e"))).interactive()

selection = alt.selection_point(fields=['value'], bind='legend')

chart4=alt.Chart(df2).mark_line().encode(alt.X('time').title('time(h)'), alt.Y('value', axis=alt.Axis(format=".3")).title('Dose (Gy)'), alt.Color('variable').title('Type')).interactive()

dose_particle_df

required_dose=st.number_input('Required total dose', min_value=0., max_value=max(dose_particle_df['Sum']))

y_line=alt.Chart(df2).mark_rule(color='red', strokeWidth=1, strokeOpacity=0.02, fillOpacity=0.02).encode(y=alt.datum(float(required_dose)))

st.altair_chart(chart4 + y_line , theme=None, use_container_width=True)

index_required_dose=dose_particle_df1[dose_particle_df['Sum']>=required_dose].index[0]

time_required_dose=dose_particle_df1['time'][index_required_dose]


st.write('The time for delivering the total required dose of %.2f Gy is %.2f hours' % (required_dose, time_required_dose))

csv_dose_particle=dose_particle_df.to_csv().encode('utf-8')

columns1=st.columns(2)

st.download_button('Download data from chart as .csv file', data=csv_dose_particle, file_name='dose_vs_time_vs_particle.csv', key='aaa')


dose_particle_df_normalized=dose_particle_df[['time','Sum']]
dose_particle_df_normalized['Sum']=dose_particle_df_normalized['Sum']/init_activity


st.subheader('Initial activity for 30 or 50 Gy at a certain time')
st.caption('The time choice is indicative for the calculation, the exact timepoint will be reported once the calculation is completed, this is recalculated given the kinetic parameters defined at the beginning. Hence, the results depend only on the PK profile of the drug and on the % IDG deriving from the biodistribution. You can double check the numbers with the graph just above.')

dose_radio_button=st.radio('Choose which dose in Gy you want to reach', options=[30, 50])
time_required_dose=st.number_input('Time for reaching this dose', min_value=0.000001, max_value=time[-1], value=10.)

try:
    dose_particle_df_normalized['Diff']= (dose_particle_df_normalized['time']-time_required_dose).abs()
    min_time_row=dose_particle_df_normalized[dose_particle_df_normalized['Diff']==dose_particle_df_normalized['Diff'].min()]
    
    
    if dose_radio_button==50:
        selected_time=min_time_row['time']
        a0_fifty_gy=50./min_time_row['Sum'].values
        a0_fifty_gy=a0_fifty_gy[0]
        st.write('To reach 50 Gy at %.1f hours, you will need %.1f MBq of initial activity' % (selected_time, a0_fifty_gy))
    
    if dose_radio_button==30:
        selected_time=min_time_row['time']
        a0_fifty_gy=30./min_time_row['Sum'].values
        a0_fifty_gy=a0_fifty_gy[0]
        st.write('To reach 30 Gy at %.1f hours, you will need %.1f MBq of initial activity' % (selected_time, a0_fifty_gy))
    
except:
    st.warning('The kinetic parameters of the model are incompatible with initial activity estimation, modify and try again')