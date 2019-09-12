#this script generates tables with the full 
#decomposition results

import csv
import numpy as np
import pandas as pd
import pickle
import time
import copy
import matplotlib.pyplot as plt
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


#importing input data
with open('../../Data/Processed/decomposition_full.pkl', 'rb') as f:  
    ida = pickle.load(f)

#########################################################

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
#Split by factor
#yref0 = emeu[:,:,:,:,0].sum(axis = (0,1,2,3))/1000
#ytmp1 = dfactor.sum(axis=(0,1,2,3))/1000

#str_label = ['Share fossil', 'Share non-foss.', 'Transport eff.', 'Trade share', 'Final use eff.', 'GDP per capita', 'Population']
#str_title = 'Decomposition by factor'

#tmp = pd.DataFrame(data = list(val_top), index = ['2000-2007', '2007-2015', '2000-2015'], columns = str_label)
#tmp.to_excel(writer, sheet_name = 'Factors_(MtCO2)_agg')

#tmp.to_excel(writer, sheet_name = 'Emissions_(MtCO2)_year')
#########################################################


#########################################################
#Writing table with factor split by region/time
writer = pd.ExcelWriter('../../Text/Tables/EU-IDA-factor_production(tCO2).xlsx', engine = 'xlsxwriter')
str_col = [timcodes[k]+'-'+timcodes[k+1] for k in range(ntim-1)]
str_row = ['Share fossil electricity', 'Share non-fossil electricity', 'Electricity transport efficiency', 'Trade share', 'Final use efficiency', 'GDP per capita', 'Population']

dat = dfactor.sum(axis = (0,1,2,3)).T
tmp = pd.DataFrame(data = list(dat), index = str_row, columns = str_col)
tmp.to_excel(writer, sheet_name = 'EU')

for kreg in range(nms):

    dat = dfactor[:,kreg,:,:,:,:].sum(axis = (0,1,2)).T

#    emtmp = emeu[:,:,kreg,:,:].sum(axis = (1,2))
#    emprint = np.concatenate((emprint, emtmp))

    tmp = pd.DataFrame(data = list(dat), index = str_row, columns = str_col)
    tmp.to_excel(writer, sheet_name = mscodes[kreg][0])


writer.save()
writer.close()
#########################################################

#########################################################
#Writing table with factor split by region/time
writer = pd.ExcelWriter('../../Text/Tables/EU-IDA-factor_consumption(tCO2).xlsx', engine = 'xlsxwriter')
str_col = [timcodes[k]+'-'+timcodes[k+1] for k in range(ntim-1)]
str_row = ['Share fossil electricity', 'Share non-fossil electricity', 'Electricity transport efficiency', 'Trade share', 'Final use efficiency', 'GDP per capita', 'Population']

dat = dfactor.sum(axis = (0,1,2,3)).T
tmp = pd.DataFrame(data = list(dat), index = str_row, columns = str_col)
tmp.to_excel(writer, sheet_name = 'EU')

for kreg in range(nms):

    dat = dfactor[:,:,kreg,:,:,:].sum(axis = (0,1,2)).T

#    emtmp = emeu[:,:,kreg,:,:].sum(axis = (1,2))
#    emprint = np.concatenate((emprint, emtmp))

    tmp = pd.DataFrame(data = list(dat), index = str_row, columns = str_col)
    tmp.to_excel(writer, sheet_name = mscodes[kreg][0])


writer.save()
writer.close()
#########################################################

#########################################################
#Writing table with factor split by fossil fuel
writer = pd.ExcelWriter('../../Text/Tables/EU-IDA-fossil_consumption(tCO2).xlsx', engine = 'xlsxwriter')
str_col = [timcodes[k]+'-'+timcodes[k+1] for k in range(ntim-1)]
str_row = ['Anthracite', 'Other bituminous coal', 'Sub-bituminous coal', 'Lignite', 'Peat', 'Oil', 'Gas']

dat = dfactor[:,:,:,:,:,0].sum(axis = (1,2,3))
tmp = pd.DataFrame(data = list(dat), index = str_row, columns = str_col)
tmp.to_excel(writer, sheet_name = 'EU')

for kreg in range(nms):

    dat = dfactor[:,:,kreg,:,:,0].sum(axis = (1,2))

#    emtmp = emeu[:,:,kreg,:,:].sum(axis = (1,2))
#    emprint = np.concatenate((emprint, emtmp))

    tmp = pd.DataFrame(data = list(dat), index = str_row, columns = str_col)
    tmp.to_excel(writer, sheet_name = mscodes[kreg][0])


writer.save()
writer.close()
#########################################################

#########################################################
#Writing table with factor split by fossil fuel
writer = pd.ExcelWriter('../../Text/Tables/EU-IDA-fossil_production(tCO2).xlsx', engine = 'xlsxwriter')
str_col = [timcodes[k]+'-'+timcodes[k+1] for k in range(ntim-1)]
str_row = ['Anthracite', 'Other bituminous coal', 'Sub-bituminous coal', 'Lignite', 'Peat', 'Oil', 'Gas']

dat = dfactor[:,:,:,:,:,0].sum(axis = (1,2,3))
tmp = pd.DataFrame(data = list(dat), index = str_row, columns = str_col)
tmp.to_excel(writer, sheet_name = 'EU')

for kreg in range(nms):

    dat = dfactor[:,kreg,:,:,:,0].sum(axis = (1,2))

#    emtmp = emeu[:,:,kreg,:,:].sum(axis = (1,2))
#    emprint = np.concatenate((emprint, emtmp))

    tmp = pd.DataFrame(data = list(dat), index = str_row, columns = str_col)
    tmp.to_excel(writer, sheet_name = mscodes[kreg][0])


writer.save()
writer.close()
#########################################################

#########################################################
#Writing table with factor split by final category
writer = pd.ExcelWriter('../../Text/Tables/EU-IDA-final_production(tCO2).xlsx', engine = 'xlsxwriter')
str_col = [timcodes[k]+'-'+timcodes[k+1] for k in range(ntim-1)]
str_row = [ 'Residential',
 'Services',
 'Agriculture',
 'Iron and steel industry',
 'Chemical and petrochemical industry',
 'Non-ferrous metals industry',
 'Non-metalic minerals',
 'Transport equipment',
 'Machinery industry',
 'Mining and quarrying',
 'Food and tobacco',
 'Paper, pulp and print',
 'Wood and wood products',
 'Construction',
 'Textile and leather',
 'Non-specified (industry)',
 'Rail transport',
 'Other transport',
 'Other sector']

dat = dfactor[:,:,:,:,:,4].sum(axis = (0,1,2))
tmp = pd.DataFrame(data = list(dat), index = str_row, columns = str_col)
tmp.to_excel(writer, sheet_name = 'EU')

for kreg in range(nms):

    dat = dfactor[:,kreg,:,:,:,4].sum(axis = (0,1))

#    emtmp = emeu[:,:,kreg,:,:].sum(axis = (1,2))
#    emprint = np.concatenate((emprint, emtmp))

    tmp = pd.DataFrame(data = list(dat), index = str_row, columns = str_col)
    tmp.to_excel(writer, sheet_name = mscodes[kreg][0])


writer.save()
writer.close()
#########################################################

#########################################################
#Writing table with factor split by final category
writer = pd.ExcelWriter('../../Text/Tables/EU-IDA-final_consumption(tCO2).xlsx', engine = 'xlsxwriter')
str_col = [timcodes[k]+'-'+timcodes[k+1] for k in range(ntim-1)]
str_row = [ 'Residential',
 'Services',
 'Agriculture',
 'Iron and steel industry',
 'Chemical and petrochemical industry',
 'Non-ferrous metals industry',
 'Non-metalic minerals',
 'Transport equipment',
 'Machinery industry',
 'Mining and quarrying',
 'Food and tobacco',
 'Paper, pulp and print',
 'Wood and wood products',
 'Construction',
 'Textile and leather',
 'Non-specified (industry)',
 'Rail transport',
 'Other transport',
 'Other sector']

dat = dfactor[:,:,:,:,:,4].sum(axis = (0,1,2))
tmp = pd.DataFrame(data = list(dat), index = str_row, columns = str_col)
tmp.to_excel(writer, sheet_name = 'EU')

for kreg in range(nms):

    dat = dfactor[:,:,kreg,:,:,4].sum(axis = (0,1))

#    emtmp = emeu[:,:,kreg,:,:].sum(axis = (1,2))
#    emprint = np.concatenate((emprint, emtmp))

    tmp = pd.DataFrame(data = list(dat), index = str_row, columns = str_col)
    tmp.to_excel(writer, sheet_name = mscodes[kreg][0])


writer.save()
writer.close()
#########################################################

#########################################################
#Writing table with factor split by non-fossil electricity
writer = pd.ExcelWriter('../../Text/Tables/EU-IDA-renewable(tCO2).xlsx', engine = 'xlsxwriter')
str_col = [timcodes[k]+'-'+timcodes[k+1] for k in range(ntim-1)]
str_row = ['Nuclear', 'Hydro', 'Wind', 'Solar', 'Other']

dat = drenmed.sum(axis = (0)).T
tmp = pd.DataFrame(data = list(dat), index = str_row, columns = str_col)
tmp.to_excel(writer, sheet_name = 'EU')

for kreg in range(nms):

    dat = drenmed[kreg,:,:].T

#    emtmp = emeu[:,:,kreg,:,:].sum(axis = (1,2))
#    emprint = np.concatenate((emprint, emtmp))

    tmp = pd.DataFrame(data = list(dat), index = str_row, columns = str_col)
    tmp.to_excel(writer, sheet_name = mscodes[kreg][0])

writer.save()
writer.close()
#########################################################

#########################################################
#Writing table with factor split by non-fossil electricity
writer = pd.ExcelWriter('../../Text/Tables/EU-IDA-renewable_min(tCO2).xlsx', engine = 'xlsxwriter')
str_col = [timcodes[k]+'-'+timcodes[k+1] for k in range(ntim-1)]
str_row = ['Nuclear', 'Hydro', 'Wind', 'Solar', 'Other']

dat = drenmin.sum(axis = (0)).T
tmp = pd.DataFrame(data = list(dat), index = str_row, columns = str_col)
tmp.to_excel(writer, sheet_name = 'EU')

for kreg in range(nms):

    dat = drenmin[kreg,:,:].T

#    emtmp = emeu[:,:,kreg,:,:].sum(axis = (1,2))
#    emprint = np.concatenate((emprint, emtmp))

    tmp = pd.DataFrame(data = list(dat), index = str_row, columns = str_col)
    tmp.to_excel(writer, sheet_name = mscodes[kreg][0])

writer.save()
writer.close()
#########################################################

#########################################################
#Writing table with factor split by non-fossil electricity
writer = pd.ExcelWriter('../../Text/Tables/EU-IDA-renewable_max(tCO2).xlsx', engine = 'xlsxwriter')
str_col = [timcodes[k]+'-'+timcodes[k+1] for k in range(ntim-1)]
str_row = ['Nuclear', 'Hydro', 'Wind', 'Solar', 'Other']

dat = drenmax.sum(axis = (0)).T
tmp = pd.DataFrame(data = list(dat), index = str_row, columns = str_col)
tmp.to_excel(writer, sheet_name = 'EU')

for kreg in range(nms):

    dat = drenmax[kreg,:,:].T

#    emtmp = emeu[:,:,kreg,:,:].sum(axis = (1,2))
#    emprint = np.concatenate((emprint, emtmp))

    tmp = pd.DataFrame(data = list(dat), index = str_row, columns = str_col)
    tmp.to_excel(writer, sheet_name = mscodes[kreg][0])

writer.save()
writer.close()
#########################################################

#########################################################
#Writing table with factor split by trade
writer = pd.ExcelWriter('../../Text/Tables/EU-IDA-trade(tCO2).xlsx', engine = 'xlsxwriter')
str_col = [timcodes[k]+'-'+timcodes[k+1] for k in range(ntim-1)]
str_row = [mstmp[0] for mstmp in mscodes]

for ktim in range(ntim-1):

    dat = dfactor[:,:,:,:,ktim,3].sum(axis = (0,3))

#    emtmp = emeu[:,:,kreg,:,:].sum(axis = (1,2))
#    emprint = np.concatenate((emprint, emtmp))

    tmp = pd.DataFrame(data = list(dat), index = str_row, columns = str_row)
    tmp.to_excel(writer, sheet_name = str_col[ktim])


writer.save()
writer.close()
#########################################################

