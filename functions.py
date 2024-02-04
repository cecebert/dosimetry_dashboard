""""
Read the decay data for the nuclide of choice from the file list
"""
import pandas as pd

def read_nuclide_csv(nuclide_id):
    nuclide_df=pd.read_csv('mird-'+str(nuclide_id)+'-table-0.csv', usecols=[0, 1, 2, 3], names=['Particle', 'Probability', 'Energy', 'Dose'])
    nuclide_df=nuclide_df.replace('×10', 'e', regex=True)
    return nuclide_df

""""
Extract the dose delivered per particle from the dataframe of the chosen nuclide, the particle is specified as b+ b- auger and ce
"""

def extract_dose_particle(df, particle_name):
    particle_list_1=df['Particle'][df['Particle'].str.find(particle_name)!=-1]
    particle_list_2=df['Particle'][df['Particle'].str.find('ray')!=-1]
    list_union=set(particle_list_1).intersection(set(particle_list_2))
    df=df[~df['Particle'].isin(list_union)]
    particle_list=df['Particle'][df['Particle'].str.find(particle_name)!=-1]
    df=df[df['Particle'].isin(particle_list)]
    print(df)
    sum_e=df['Dose'].astype(float).sum()
    return sum_e

def bateman(t,a0, k1, k2):
    import numpy as np
    y= (a0*k1*np.exp(-t*(k2)))/(k1-k2) - (a0*k1*np.exp(-t*(k1)))/(k1-k2)
    return y

def bateman_decayed(t,a0, k1, k2, k_dec):
    import numpy as np
    y= (a0*k1*np.exp(-t*(k2+k_dec)))/(k1-k2) - (a0*k1*np.exp(-t*(k1+k_dec)))/(k1-k2)
    return y

def max_bateman(t,a0, k1, k2):
    import numpy as np
    y= k1*a(t)-k2*b(t)-k_dec*b(t)
    return y

def activity_calc(t, k_1 , k_2, k_dec, idg, a0):
    import numpy as np
    y=a0*idg/100*(-np.exp(k_1*t) + np.exp(k_2*t))*np.exp(-k_dec*t)*np.exp(-t*(k_1 + k_2))/(np.exp(k_1*np.log(k_2/k_1)/(k_1 - k_2)) - np.exp(k_2*np.log(k_2/k_1)/(k_1 - k_2)))
    return y