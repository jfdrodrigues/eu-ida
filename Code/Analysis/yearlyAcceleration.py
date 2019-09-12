#this script generates tables and plots
#with yearly contribution of factors

import csv
import numpy as np
import pandas as pd
import pickle
import time
import copy
import matplotlib.figure as pltfig
import matplotlib.pyplot as plt
import seaborn as sb
import pdb

#importing codes of member states
msfile = open('../../Data/Eurostat/codes_countries.txt')
msreader = csv.reader(msfile, delimiter = '\t')
mscodes = list(msreader)
nms = len(mscodes)

#time
ntim = 16
timcodes = [str(2000+t) for t in range(ntim)]

dtim = []
for k in range(1,ntim):
    dtim.append('%s-%s' % (timcodes[k-1], timcodes[k]))

#fossil carriers
foscodes = ['fos_anthracite', 'fos_other_bituminous_coal', 'fos_sub_bituminous_coal', 'fos_lignite', 'fos_peat', 'fos_oil', 'fos_gas']
nfos = len(foscodes)

#total fossil is 'fossil'

#non fossil carriers
rencodes = ['ren_nuclear', 'ren_hydro', 'ren_wind', 'ren_solar', 'ren_other']
nren = len(rencodes)

#sector codes of final energy use
fincodes = [ 'fin_residential',
 'fin_services',
 'fin_agriculture',
 'fin_iron_steel',
 'fin_chemical',
 'fin_non_ferrous_metals',
 'fin_non_metalic_minerals',
 'fin_transport_equipment',
 'fin_machinery',
 'fin_mining',
 'fin_food',
 'fin_paper',
 'fin_wood',
 'fin_construction',
 'fin_textile',
 'fin_non_specified_industry',
 'fin_rail',
 'fin_other_transport',
 'fin_other_sector']
nfin = len(fincodes)

#sector codes of final energy use
factorcodes = [ 'share_fossil',
 'share_non_fossil',
 'transmission_efficiency',
 'trade_share',
 'final_use_efficiency',
 'GVA_per_capita',
 'population']
nfactor = len(factorcodes)

#########################################################

#importing input data
with open('../../Data/Processed/unified_val.pkl', 'rb') as f:  
    val = pickle.load(f)

dirval = val['emval']
fosval = val['fosval']
totval = val['totval']
renval = val['renval']
finval = val['finval']
traval = val['traval']
gvaval = val['gvaval']
popval = val['popval']

#importing input data
with open('../../Data/Processed/decomposition_full.pkl', 'rb') as f:  
    ida = pickle.load(f)

drenmin = ida['drenmin']
drenmed = ida['drenmed']
drenmax = ida['drenmax']
emeu = ida['emeu']
demeu = ida['demeu']
dfactor = np.zeros((nfos, nms, nms, nfin, ntim-1, nfactor))
dfactor[:,:,:,:,:,0] = ida['dshafos']
dfactor[:,:,:,:,:,1] = ida['dsharen']
dfactor[:,:,:,:,:,2] = ida['dtraeff']
dfactor[:,:,:,:,:,3] = ida['dtrasha']
dfactor[:,:,:,:,:,4] = ida['dfingva']
dfactor[:,:,:,:,:,5] = ida['dgvacap']
dfactor[:,:,:,:,:,6] = ida['dpopval']

#logarithmic mean function
def logmean(a,b):
    c = (a - b)/(np.log(a) - np.log(b))    
    return c

#calculate emissions
#emeu = emcoef * shafos * sharen * traeff * trasha * fingva * gvacap * pop
#emissions in the EU by ROW demand are 2%, we ignore

#check that decomposition yields same correct total
demeualt = np.zeros((nfos, nms, nms, nfin, ntim-1))
for kfactor in range(nfactor):
    demeualt += dfactor[:,:,:,:,:,kfactor]

#there is a 0.9% discrepancy in EU change 2000-2015 total
#emissions, we believe related to unaccounted exports
#(1 - demeualt.sum()/demeu.sum())*100    

#########################################################

writer = pd.ExcelWriter('../../Text/Discussion/yearly-decomposition.xlsx', engine = 'xlsxwriter')

#########################################################
#Writing table with fossil
str_label = ['Nuclear', 'Hydro', 'Wind', 'Solar', 'Other']

#incorporating numerical values
ytmp1 = drenmed.sum(axis=(0))/1000

dat = ytmp1
#build table
tmp = pd.DataFrame(data = list(dat), index = dtim, columns = str_label)
tmp.to_excel(writer, sheet_name = 'Renewable(MtCO2)')

#generate plot
plt.close()
fig0 = plt.figure()
fig0.subplots_adjust(bottom=0.25)

plt.plot(dat)
plt.plot(np.zeros(ntim-1),'k')
plt.legend(str_label, loc = 'upper left')
plt.title('Yearly decomposition of non-fossil electricity types')
plt.xlabel('Period')
plt.ylabel('Carbon emissions (MtCO2)')
plt.xticks(np.arange(15), dtim, rotation = 90)

#ax.set_aspect(1.0)
plt.savefig('../../Text/Discussion/yearly-renewable.jpg', dpi = 200)

#plt.show()
#########################################################


#########################################################
#Writing table with fossil
str_label = ['Residential', 'Chemical', 'Iron & steel', 'NS industry', 'Other', 'Machinery', 'Services']

#incorporating numerical values
ytmp1 = dfactor[:,:,:,:,:,4].sum(axis = (0,1,2)).T/1000

vpos = (ytmp1.sum(axis = 0)).argsort()
ytmp2 = ytmp1[:,vpos]
ytmp1 = np.zeros((ntim-1,7))
ytmp1[:,:4] = ytmp2[:,:4]
ytmp1[:,-2:] = ytmp2[:,-2:]
ytmp1[:,4] = ytmp2[:,4:-2].sum(axis = 1)

dat = ytmp1
#build table
tmp = pd.DataFrame(data = list(dat), index = dtim, columns = str_label)
tmp.to_excel(writer, sheet_name = 'Final(MtCO2)')

#generate plot
plt.close()
fig0 = plt.figure()
fig0.subplots_adjust(bottom=0.25)

plt.plot(dat)
plt.plot(np.zeros(ntim-1),'k')
plt.legend(str_label, loc = 'upper left')
plt.title('Yearly decomposition of final use of electricity')
plt.xlabel('Period')
plt.ylabel('Carbon emissions (MtCO2)')
plt.xticks(np.arange(15), dtim, rotation = 90)

#ax.set_aspect(1.0)
plt.savefig('../../Text/Discussion/yearly-final.jpg', dpi = 200)

#plt.show()
#########################################################


#########################################################
#Writing table with fossil
str_factor = ['Anthracite', 'Other bituminous coal', 'Sub-bituminous coal', 'Lignite', 'Peat', 'Oil', 'Gas']
str_factor = ['Anthracite', 'Bituminous coal', 'Lignite', 'Other coal', 'Oil', 'Gas']

#incorporating numerical values
z = dfactor[:,:,:,:,:,0].sum(axis = (1,2,3)).T/1000

y = np.zeros((ntim-1,6))
y[:,0] = z[:,0]
y[:,1] = z[:,1]
y[:,2] = z[:,3]
y[:,4] = z[:,5]
y[:,5] = z[:,6]
y[:,3] = y[:,3] + z[:,2]
y[:,3] = y[:,3] + z[:,4]

dat = y


#build table
tmp = pd.DataFrame(data = list(dat), index = dtim, columns = str_factor)
tmp.to_excel(writer, sheet_name = 'Fossil(MtCO2)')

#generate plot
plt.close()
fig0 = plt.figure()
fig0.subplots_adjust(bottom=0.25)

plt.plot(dat)
plt.plot(np.zeros(ntim-1),'k')
plt.legend(str_factor, loc='upper left')
plt.title('Yearly decomposition of fossil fuel carriers')
plt.xlabel('Period')
plt.ylabel('Carbon emissions (MtCO2)')
plt.xticks(np.arange(15), dtim, rotation = 90)

#ax.set_aspect(1.0)
plt.savefig('../../Text/Discussion/yearly-fossil.jpg', dpi = 200)

#plt.show()
#########################################################

#########################################################
#Writing table with factors
str_factor = ['Fossil', 'Non-fossil', 'Transport eff.', 'Trade share', 'Final use eff.', 'GDP per cap', 'Population']

#incorporating numerical values
dat = dfactor.sum(axis = (0,1,2,3))/1000

#build table
tmp = pd.DataFrame(data = list(dat), index = dtim, columns = str_factor)
tmp.to_excel(writer, sheet_name = 'Factors(MtCO2)')

#generate plot
plt.close()
fig0 = plt.figure()
fig0.subplots_adjust(bottom=0.25)

plt.plot(dat)
plt.plot(np.zeros(ntim-1),'k')
plt.legend(str_factor)
plt.title('Yearly decomposition by factor')
plt.xlabel('Period')
plt.ylabel('Carbon emissions (MtCO2)')
plt.xticks(np.arange(15), dtim, rotation = 90)

#ax.set_aspect(1.0)
plt.savefig('../../Text/Discussion/yearly-factor.jpg', dpi = 200)

#plt.show()
#########################################################


writer.save()
writer.close()

