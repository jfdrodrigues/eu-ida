#this script generates tables with summary information
#of per capita decomposition factors from 2007 to 2015


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

#########################################################
#Writing table with factor split by region/time
writer = pd.ExcelWriter('../../Text/Results/per-capita-2007-2015-IDA.xlsx', engine = 'xlsxwriter')
mscodes[4][1]='Germany'
mscodes[24][1]='UK'
str_col = [mscode[1] for mscode in mscodes]
str_row = ['Fossil', 'Non-fossil', 'Transport', 'Trade share', 'Final use', 'GDP per cap', 'Population']

#incorporating numerical values
dat = np.zeros((nms, nfactor))
for kreg in range(nms):
    for kfac in range(nfactor):
        valvec = dfactor.sum(axis = (0,1,3))[kreg, :, kfac]
        dat[kreg,kfac] = valvec[7:].sum() 

datfac = dat.sum(0)
datms = dat.sum(1)

#dividing by population
#toe/cap
for kreg in range(nms):
    dat[kreg,:] = dat[kreg,:] / popval[kreg, 7] * 1000
    datms[kreg] = datms[kreg] / popval[kreg, 7] * 1000
datfac = datfac / popval[:, 7].sum() * 1000

#sorting
sortfac = datfac.argsort()
sortms = datms.argsort()

datfac = datfac[sortfac]
dat = dat[:,sortfac]
str_row = [str_row[sortfac[k]] for k in range(nfactor)] 
datms = datms[sortms]
dat = dat[sortms,:]
str_col = [str_col[sortms[k]] for k in range(nms)] 

#build table
str_col.append('EU')
str_row.append('All')

dat = np.concatenate((dat,datms.reshape(nms,1)),axis = 1)
datfac = np.concatenate((datfac.reshape(1,nfactor),[[0]]),axis = 1)
dat = np.concatenate((dat,datfac),axis = 0)

#dat.concatenate(datms,axis = 1)

tmp = pd.DataFrame(data = list(dat), index = str_col, columns = str_row)
tmp.to_excel(writer, sheet_name = 'Contribution(tCO2cap)')

writer.save()
writer.close()
#########################################################

#########################################################
#generate heatmap
plt.close()
fig0 = plt.figure()
#fig, ax = plt.subplots()
fig0.subplots_adjust(left=0.2, right=1, bottom = 0.25)
sb.heatmap(tmp.transpose(), center=0, cmap=sb.diverging_palette(220, 20, as_cmap=True))
plt.title('Per capita decomposition of emissions (tCO2/cap) in 2007-2015')
plt.ylabel('Factor')
plt.xlabel('Country')

#ax.set_aspect(1.0)
plt.savefig('../../Text/Results/heatmap.jpg', dpi = 200)

#plt.show()
#########################################################

