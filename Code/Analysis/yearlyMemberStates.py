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

writer = pd.ExcelWriter('../../Text/Discussion/yearly-member-states.xlsx', engine = 'xlsxwriter')


#########################################################
#Writing table with factors
mscodes[4][1]='Germany'
mscodes[24][1]='UK'
str_col = [mscode[1] for mscode in mscodes]

#incorporating numerical values
dat = dfactor.sum(axis = (0,1,3,5)).T

dat = dat / popval[:,:-1].T * 1000

dmean = np.median(dat,axis=1)
dsd = np.sqrt(np.var(dat,axis=1))
dmax = np.max(dat,axis=1)
dmin = np.min(dat,axis=1)

#build table
tmp = pd.DataFrame(data = list(dat), index = dtim, columns = str_col)
tmp.to_excel(writer, sheet_name = 'YearlyPerCapita(tCO2cap)')

#generate plot
plt.close()
fig0 = plt.figure()
fig0.subplots_adjust(bottom=0.25)

plt.boxplot(dat.T,patch_artist=True,
            boxprops=dict(facecolor='white', color='blue'),
            capprops=dict(color='blue'),
            whiskerprops=dict(color='blue'),
            flierprops=dict(color='white', markeredgecolor='blue'),
            medianprops=dict(color='brown'))
#plt.plot(dmean)
#plt.plot(dmean+dsd)
#plt.plot(dmean-dsd)
#plt.plot(dmin)
#plt.plot(dmax)
plt.plot(np.arange(1,16),np.zeros(ntim-1),'k')
#plt.legend(['Mean',])
plt.title('Yearly decomposition by country')
plt.xlabel('Period')
plt.ylabel('Carbon emissions per capita (tCO2/hab)')
plt.xticks(np.arange(1,16), dtim, rotation = 90)

#ax.set_aspect(1.0)
plt.savefig('../../Text/Discussion/yearly-countries.jpg', dpi = 200)

#plt.show()
#########################################################


writer.save()
writer.close()

