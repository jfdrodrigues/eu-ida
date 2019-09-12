#this script generates the total emissions by
#energy carrier, country of emission, country of 
#consumption and sector of consumption

import csv
import numpy as np
import pandas as pd
import pickle
import time
import copy


#importing codes of member states
msfile = open('../../Data/Eurostat/codes_countries.txt')
msreader = csv.reader(msfile, delimiter = '\t')
mscodes = list(msreader)
nms = len(mscodes)

#time
ntim = 16
timcodes = [str(2000+t) for t in range(ntim)]

#fossil carriers
carcodes = ['fos_anthracite', 'fos_other_bituminous_coal', 'fos_sub_bituminous_coal', 'fos_lignite', 'fos_peat', 'fos_oil', 'fos_gas']
nfos = len(carcodes)

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

#########################################################
#calculate intensive variables
#emeu = emcoef * sharfos * sharen * traeff * trasha * fingva * gvacap * pop
#emrow = emcoef * sharfos * sharen * traeff * rowval

gvacap = gvaval / popval
fingva = np.zeros((nfin, nms, ntim))
for k in range(nfin):
    tmp = (gvaval != 0) / (gvaval + (gvaval == 0))
    fingva[k,:,:] = finval[k,:,:] * tmp

traref = copy.copy(traval)

#exports of energy to the rest of the world
#shape nms * ntim
rowval = traval[:-1,-1,:]

#share of intra-eu imports
fintot = finval.sum(axis = 0)
tratot = traval[:,:-1,:].sum(axis = 0)

#generate shares
trasha = np.zeros((nms, nms, ntim))
for kms1 in range(nms):
    for kms2 in range(nms):
        for ktim in range(ntim):
            tmp1 = tratot[kms2, ktim]
            tmp1 = (tmp1 != 0) / (tmp1 + (tmp1 == 0))     
            tmp2 = traval[kms1, kms2, ktim]
            trasha[kms1,kms2,ktim] = tmp2 * tmp1
            if(trasha[kms1,kms2,ktim] < 0):
                print(kms1, kms2, ktim, trasha[kms1,kms2,ktim])

#traeff = distribution efficiency
#on top sum of all electricity produced
#on bottom sum of all electricity consumed
traeff = np.zeros((nms, ntim))
for kms in range(nms):
    for ktim in range(ntim):
        tmp1 = totval[kms, ktim] + renval[:-1,kms,ktim].sum()
        tmp2 = traval[kms,:,ktim].sum()
        traeff[kms, ktim] = tmp1 * (tmp2 != 0) / (tmp2 + (tmp2 == 0))

#sharen = share of fossil in total energy production
#on top sum of all electricity produced
#on bottom sum of all electricity consumed
sharen = np.zeros((nms, ntim))
for kms in range(nms):
    for ktim in range(ntim):
        tmp1 = totval[kms, ktim]
        tmp2 = totval[kms, ktim] + renval[:,kms,ktim].sum()
        sharen[kms, ktim] = tmp1 * (tmp2 != 0) / (tmp2 + (tmp2 == 0))

#shafos = share of each fossil carrier in total fossil
shafos = np.zeros((nfos, nms, ntim))
for kfos in range(nfos):
    for kms in range(nms):
        for ktim in range(ntim):
            tmp1 = fosval[kfos, kms, ktim]
            tmp2 = totval[kms, ktim]
            shafos[kfos, kms, ktim] = tmp1 * (tmp2 != 0) / (tmp2 + (tmp2 == 0))

#emcoef = emission coefficient
emcoef = np.zeros((nfos, nms, ntim))
for kfos in range(nfos):
    for kms in range(nms):
        for ktim in range(ntim):
            tmp1 = dirval[kfos, kms, ktim]
            tmp2 = fosval[kfos, kms, ktim]
            emcoef[kfos, kms, ktim] = tmp1 * (tmp2 != 0) / (tmp2 + (tmp2 == 0))

'''
#emissions: carrier, country, country, sector, year
emeval = np.zeros((ncar, nms, ntim))
for (ktim, timval) in enumerate(timcodes):
    for (kms, msval) in enumerate(mscodes):
        for (kcar, carval) in enumerate(carcodes):
            tmp = em[msval[0]][timval][carval]
            emval[kcar, kms, ktim] = tmp
'''

#saving
#emeu = emcoef * sharfos * sharen * traeff * trasha * fingva * gvacap * pop
#emrow = emcoef * sharfos * sharen * traeff * rowval

coef = {'totval': totval, 'renval': renval, 'dirval': dirval, 'emcoef': emcoef, 'shafos': shafos, 'sharen': sharen, 'traeff': traeff, 'trasha': trasha, 'fingva': fingva, 'gvacap': gvacap, 'popval': popval, 'rowval': rowval}
with open('../../Data/Processed/coefficients.pkl', 'wb') as f:  
    pickle.dump(coef, f)














