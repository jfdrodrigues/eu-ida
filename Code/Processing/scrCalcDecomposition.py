#this script generates the total emissions by
#energy carrier, country of emission, country of 
#consumption and sector of consumption

import csv
import numpy as np
import pandas as pd
import pickle
import time
import copy
import sys
import itertools
eps = sys.float_info.epsilon


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
with open('../../Data/Processed/coefficients.pkl', 'rb') as f:  
    coef = pickle.load(f)

totval = coef['totval']
renval = coef['renval']
dirval = coef['dirval']
emcoef = coef['emcoef']
shafos = coef['shafos']
sharen = coef['sharen']
traeff = coef['traeff']
trasha = coef['trasha']
fingva = coef['fingva']
gvacap = coef['gvacap']
rowval = coef['rowval']
popval = coef['popval']

#logarithmic mean function
def logmean(a,b):
    if((a==0) & (b ==0)):
        c = 0
    else:
        c = (a - b)/(np.log((a+eps)/(b+eps)))    
    return c

#calculate emissions
#emeu = emcoef * sharfos * sharen * traeff * trasha * fingva * gvacap * pop
#emrow = emcoef * sharfos * sharen * traeff * rowval
#emissions in the EU by ROW demand are 2%, we ignore

emeu = np.zeros((nfos, nms, nms, nfin, ntim))
for kfos in range(nfos):
    for kms1 in range(nms):
        for kms2 in range(nms):
            for kfin in range(nfin):
                for ktim in range(ntim):
                    tmp1 = emcoef[kfos, kms1, ktim]
                    tmp2 = shafos[kfos, kms1, ktim]
                    tmp3 = sharen[kms1, ktim]
                    tmp4 = traeff[kms1, ktim]
                    tmp5 = trasha[kms1, kms2, ktim]
                    tmp6 = fingva[kfin, kms2, ktim]
                    tmp7 = gvacap[kms2, ktim]
                    tmp8 = popval[kms2, ktim]
                    emeu[kfos, kms1, kms2, kfin, ktim] = tmp1 * tmp2 * tmp3 * tmp4 * tmp5 * tmp6 * tmp7 * tmp8

#total change
demeu = np.zeros((nfos, nms, nms, nfin, ntim-1))
#effect of population
dpopval = np.zeros((nfos, nms, nms, nfin, ntim-1))
#effect of GVA per capita
dgvacap = np.zeros((nfos, nms, nms, nfin, ntim-1))
#effect of sectoral use of energy per GVA
dfingva = np.zeros((nfos, nms, nms, nfin, ntim-1))
#effect of trade in electricity
dtrasha = np.zeros((nfos, nms, nms, nfin, ntim-1))
#effect of transport efficiency
dtraeff = np.zeros((nfos, nms, nms, nfin, ntim-1))
#effect of share of renewables in total production
dsharen = np.zeros((nfos, nms, nms, nfin, ntim-1))
#effect of share of fossils in total fossil
dshafos = np.zeros((nfos, nms, nms, nfin, ntim-1))
for kfos in range(nfos):
    for kms1 in range(nms):
        for kms2 in range(nms):
            for kfin in range(nfin):
                for ktim in range(ntim - 1):
                    tmp1 = emeu[kfos, kms1, kms2, kfin, ktim]
                    tmp2 = emeu[kfos, kms1, kms2, kfin, ktim+1]
                    demeu[kfos, kms1, kms2, kfin, ktim] = tmp2 - tmp1
                    tmp3 = logmean(tmp1, tmp2)
#effect of share of fossil
                    tmp4 = shafos[kfos, kms1, ktim]
                    tmp5 = shafos[kfos, kms1, ktim + 1]
                    tmp6 = np.log((tmp5 + eps)/ (tmp4 + eps))
                    dshafos[kfos, kms1, kms2, kfin, ktim] = tmp6 * tmp3
#effect of share of renewables
                    tmp4 = sharen[kms1, ktim]
                    tmp5 = sharen[kms1, ktim + 1]
                    tmp6 = np.log((tmp5 + eps)/ (tmp4 + eps))
                    dsharen[kfos, kms1, kms2, kfin, ktim] = tmp6 * tmp3
#effect of transport efficiency
                    tmp4 = traeff[kms1, ktim]
                    tmp5 = traeff[kms1, ktim + 1]
                    tmp6 = np.log((tmp5 + eps)/ (tmp4 + eps))
                    dtraeff[kfos, kms1, kms2, kfin, ktim] = tmp6 * tmp3
#effect of trade in electricity
                    tmp4 = trasha[kms1, kms2, ktim]
                    tmp5 = trasha[kms1, kms2, ktim + 1]
                    tmp6 = np.log((tmp5 + eps)/ (tmp4 + eps))
                    dtrasha[kfos, kms1, kms2, kfin, ktim] = tmp6 * tmp3
 #effect of GVA per capita
                    tmp4 = gvacap[kms2, ktim]
                    tmp5 = gvacap[kms2, ktim + 1]
                    tmp6 = np.log((tmp5 + eps)/ (tmp4 + eps))
                    dgvacap[kfos, kms1, kms2, kfin, ktim] = tmp6 * tmp3
#effect of population
                    tmp4 = popval[kms2, ktim]
                    tmp5 = popval[kms2, ktim + 1]
                    tmp6 = np.log((tmp5 + eps)/ (tmp4 + eps))
                    dpopval[kfos, kms1, kms2, kfin, ktim] = tmp6 * tmp3
#effect of sectoral use of energy per GVA
                    tmp4 = fingva[kfin, kms2, ktim]
                    tmp5 = fingva[kfin, kms2, ktim + 1]
                    tmp6 = np.log((tmp5 + eps)/ (tmp4 + eps))
                    dfingva[kfos, kms1, kms2, kfin, ktim] = tmp6 * tmp3

xcnt = 0
#looking for zeros
for kfos in range(nfos):
    for kms1 in range(nms):
        for kms2 in range(nms):
            for kfin in range(nfin):
                for ktim in range(ntim - 1):
                    tmp1 = emeu[kfos, kms1, kms2, kfin, ktim]
                    tmp2 = emeu[kfos, kms1, kms2, kfin, ktim+1]
                    demeu[kfos, kms1, kms2, kfin, ktim] = tmp2 - tmp1
                    tmp3 = logmean(tmp1, tmp2)

                    xmult = 0
#effect of sectoral use of energy per GVA
                    tmp4 = fingva[kfin, kms2, ktim]
                    tmp5 = fingva[kfin, kms2, ktim + 1]
                    if ((tmp4==0) and (tmp5!=0)) or ((tmp4!=0) and (tmp5==0)):
                        xmult = xmult + 1
                        dshafos[kfos, kms1, kms2, kfin, ktim] = 0
                        dsharen[kfos, kms1, kms2, kfin, ktim] = 0
                        dtraeff[kfos, kms1, kms2, kfin, ktim] = 0
                        dtrasha[kfos, kms1, kms2, kfin, ktim] = 0
                        dfingva[kfos, kms1, kms2, kfin, ktim] = tmp2 - tmp1
                        dgvacap[kfos, kms1, kms2, kfin, ktim] = 0
                        dpopval[kfos, kms1, kms2, kfin, ktim] = 0
#effect of trade in electricity
                    tmp4 = trasha[kms1, kms2, ktim]
                    tmp5 = trasha[kms1, kms2, ktim + 1]
                    if ((tmp4==0) and (tmp5!=0)) or ((tmp4!=0) and (tmp5==0)):
                        xmult = xmult + 1
                        dshafos[kfos, kms1, kms2, kfin, ktim] = 0
                        dsharen[kfos, kms1, kms2, kfin, ktim] = 0
                        dtraeff[kfos, kms1, kms2, kfin, ktim] = 0
                        dtrasha[kfos, kms1, kms2, kfin, ktim] = tmp2 - tmp1
                        dfingva[kfos, kms1, kms2, kfin, ktim] = 0
                        dgvacap[kfos, kms1, kms2, kfin, ktim] = 0
                        dpopval[kfos, kms1, kms2, kfin, ktim] = 0
#effect of transport efficiency
                    tmp4 = traeff[kms1, ktim]
                    tmp5 = traeff[kms1, ktim + 1]
                    if ((tmp4==0) and (tmp5!=0)) or ((tmp4!=0) and (tmp5==0)):
                        xmult = xmult + 1
                        dshafos[kfos, kms1, kms2, kfin, ktim] = 0
                        dsharen[kfos, kms1, kms2, kfin, ktim] = 0
                        dtraeff[kfos, kms1, kms2, kfin, ktim] = tmp2 - tmp1
                        dtrasha[kfos, kms1, kms2, kfin, ktim] = 0
                        dfingva[kfos, kms1, kms2, kfin, ktim] = 0
                        dgvacap[kfos, kms1, kms2, kfin, ktim] = 0
                        dpopval[kfos, kms1, kms2, kfin, ktim] = 0
#effect of share of renewables
                    tmp4 = sharen[kms1, ktim]
                    tmp5 = sharen[kms1, ktim + 1]
                    if ((tmp4==0) and (tmp5!=0)) or ((tmp4!=0) and (tmp5==0)):
                        xmult = xmult + 1
                        dshafos[kfos, kms1, kms2, kfin, ktim] = 0
                        dsharen[kfos, kms1, kms2, kfin, ktim] = tmp2 - tmp1
                        dtraeff[kfos, kms1, kms2, kfin, ktim] = 0
                        dtrasha[kfos, kms1, kms2, kfin, ktim] = 0
                        dfingva[kfos, kms1, kms2, kfin, ktim] = 0
                        dgvacap[kfos, kms1, kms2, kfin, ktim] = 0
                        dpopval[kfos, kms1, kms2, kfin, ktim] = 0
#effect of share of fossil
                    tmp4 = shafos[kfos, kms1, ktim]
                    tmp5 = shafos[kfos, kms1, ktim + 1]
                    if ((tmp4==0) and (tmp5!=0)) or ((tmp4!=0) and (tmp5==0)):
                        xmult = xmult + 1
                        dshafos[kfos, kms1, kms2, kfin, ktim] = tmp2 - tmp1
                        dsharen[kfos, kms1, kms2, kfin, ktim] = 0
                        dtraeff[kfos, kms1, kms2, kfin, ktim] = 0
                        dtrasha[kfos, kms1, kms2, kfin, ktim] = 0
                        dfingva[kfos, kms1, kms2, kfin, ktim] = 0
                        dgvacap[kfos, kms1, kms2, kfin, ktim] = 0
                        dpopval[kfos, kms1, kms2, kfin, ktim] = 0
                    if xmult > 1:
                        xcnt = xcnt + 1
print(xcnt)
####################################################

#permutations of non-fossil energy
renperm = np.array(list(itertools.permutations(range(nren))))
nperm = renperm.shape[0]

#calculate emissions
#emeu = emcoef * sharen * elefin

#totcoef = sum(dirval) / totval
#altren = totval / (totval + sum(renval))
#elefin = totval + renval

totcoef = dirval.sum(axis = 0) / totval
elefin = renval.sum(axis = 0) + totval
altren = totval / elefin

#####################################################

toteu = dirval.sum(axis = 0)

#now the decomposition of renewable components
paltren = np.zeros((nms, ntim-1, nren, nperm))
for kms in range(nms):
    for ktim in range(ntim-1):
        tmp1 = toteu[kms, ktim]
        tmp2 = toteu[kms, ktim+1]
        tmp3 = logmean(tmp1, tmp2)

        for kperm in range(nperm):
            tmp7 = totval[kms, ktim]
            tmp8 = totval[kms, ktim + 1]
            for kren in range(nren):
                renpos = renperm[kperm, kren]
                tmp4 = tmp7 + renval[renpos, kms, ktim]
                tmp5 = tmp8 + renval[renpos, kms, ktim+1]
                tmp6 = np.log((tmp8/tmp5 + eps)/ (tmp7/tmp4 + eps))
                paltren[kms, ktim, renpos, kperm] = tmp6 * tmp3
                tmp7 = tmp4
                tmp8 = tmp5


#now averaging and checking match
paltrenmed = np.zeros((nms, ntim-1, nren))
paltrenmin = np.zeros((nms, ntim-1, nren))
paltrenmax = np.zeros((nms, ntim-1, nren))
for kms in range(nms):
    for ktim in range(ntim-1):
        for kren in range(nren):
            tmp = paltren[kms, ktim, kren, :].mean()
            paltrenmed[kms, ktim, kren] = tmp

            tmp = paltren[kms, ktim, kren, :].min()
            paltrenmin[kms, ktim, kren] = tmp

            tmp = paltren[kms, ktim, kren, :].max()
            paltrenmax[kms, ktim, kren] = tmp

#####################################################

ida = {'drenmed': paltrenmed, 'drenmin': paltrenmin, 'drenmax': paltrenmax, 'emeu': emeu, 'demeu': demeu, 'dshafos': dshafos, 'dsharen': dsharen, 'dtraeff': dtraeff, 'dtrasha': dtrasha, 'dfingva': dfingva, 'dgvacap': dgvacap, 'dpopval': dpopval}
with open('../../Data/Processed/decomposition_full.pkl', 'wb') as f:  
    pickle.dump(ida, f)








