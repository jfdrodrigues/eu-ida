#Do everything
import csv
import numpy as np
import pandas as pd
import pickle
import time as time
import pdb


#importing codes of member states
msfile = open('../../Data/Eurostat/codes_countries.txt')
msreader = csv.reader(msfile, delimiter = '\t')
mscodes = list(msreader)
nms = len(mscodes)

#fossil carriers
carcodes = ['fos_anthracite', 'fos_other_bituminous_coal', 'fos_sub_bituminous_coal', 'fos_lignite', 'fos_peat', 'fos_oil', 'fos_gas']
ncar = len(carcodes)

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


#time
ntim = 16
timcodes = [str(2000+t) for t in range(ntim)]


'''
#########################################################
# ENERGY BALANCE
#Importing energy balance
#to clear variables %reset

#importing energy balances
endf = pd.read_csv('../../Data/Eurostat/TSV/nrg_110a.tsv')

#relevant energy balance data:
#   coal, oil and gas into electricity generation
#   total domestic electricity thermal generation
#   total domestic electricity renewable generation
#   electricity sales to final demand
#   imports and exports of electricity

#units in ktoe
#region = 'AT'
#year = '2000'
#extracting the row of year values
#extracting the year 2000 value

#list with stuff to get imported
#cols: product, indic_nrg, new_label
vecprop = [
['2115', 'B_101001', 'fos_anthracite', '1'],
['2116', 'B_101001', 'fos_anthracite', '1'],
['2117', 'B_101001', 'fos_other_bituminous_coal', '1'],
['2118', 'B_101001', 'fos_sub_bituminous_coal', '1'],
['2210', 'B_101001', 'fos_lignite', '1'],
['2230', 'B_101001', 'fos_lignite', '1'],
['2310', 'B_101001', 'fos_peat', '1'],
['2410', 'B_101001', 'fos_oil', '1'],
['3000', 'B_101001', 'fos_oil', '1'],
['4000', 'B_101001', 'fos_gas', '1'],
['6000', 'B_101101', 'fossil', '1'],
['6000', 'B_101102', 'ren_nuclear', '1'],
['5510', 'B_101200', 'ren_hydro', '-1'],
['5520', 'B_101200', 'ren_wind', '-1'],
['5534', 'B_101200', 'ren_solar', '-1'],
['5535', 'B_101200', 'ren_other', '-1'],
['6000', 'B_102010', 'fin_residential', '1'],
['6000', 'B_102035', 'fin_services', '1'],
['6000', 'B_102030', 'fin_agriculture', '1'],
['6000', 'B_101805', 'fin_iron_steel', '1'],
['6000', 'B_101815', 'fin_chemical', '1'],
['6000', 'B_101810', 'fin_non_ferrous_metals', '1'],
['6000', 'B_101820', 'fin_non_metalic_minerals', '1'],
['6000', 'B_101846', 'fin_transport_equipment', '1'],
['6000', 'B_101847', 'fin_machinery', '1'],
['6000', 'B_101825', 'fin_mining', '1'],
['6000', 'B_101830', 'fin_food', '1'],
['6000', 'B_101840', 'fin_paper', '1'],
['6000', 'B_101851', 'fin_wood', '1'],
['6000', 'B_101852', 'fin_construction', '1'],
['6000', 'B_101835', 'fin_textile', '1'],
['6000', 'B_101853', 'fin_non_specified_industry', '1'],
['6000', 'B_101910', 'fin_rail', '1'],
['6000', 'B_101900', 'fin_other_transport', '1'],
['6000', 'B_101910', 'fin_other_transport', '-1'],
['6000', 'B_102020', 'fin_other_sector', '1'],
['6000', 'B_102040', 'fin_other_sector', '1'],
['6000', 'B_100300', 'imports', '1'],
['6000', 'B_100500', 'exports', '1']
]



#####################################################
#function
def funImportBalance(endf, region, vecprop):
    res = {}
    for k in range(2000, 2016):
        year = str(k)        
        res[year] = {}
        #initiate
        for singprop in vecprop:
            res[year][singprop[2]] = 0

    #iterate
    for singprop in vecprop:
        vec = endf.loc[(endf['unit']=='KTOE') & (endf['product']==float(singprop[0])) & (endf['indic_nrg']==singprop[1]) & (endf['geo\\time']== region)]
        if(vec.shape[0] == 0):
            vec = endf.loc[(endf['unit']=='KTOE') & (endf['product']==singprop[0]) & (endf['indic_nrg']==singprop[1]) & (endf['geo\\time']== region)]
#        print(region, year, singprop)        
        for k in range(2000, 2016):
            year = str(k)        
            if(vec.shape[0] == 0):
                res[year][singprop[2]] += 0
            elif (vec[year].iloc[0] == ': '):
                res[year][singprop[2]] += 0
            else:
#                pdb.set_trace()
                res[year][singprop[2]] += float(singprop[3]) * float(vec[year])

    return res
#end of function
####################################################

tic = time.time()
enebal = {}
for mstmp in mscodes:
    region = mstmp[0]
    enebal[region] = {}
    tmp = funImportBalance(endf, region, vecprop)
    enebal[region] = tmp
    toc = time.time() - tic
    print(region, toc)        
    
#####################################
#storing
with open('../../Data/Processed/energy_balance.pkl', 'wb') as f:  
    # store the data as binary data stream
    pickle.dump(enebal, f)

# FINISHED ENERGY BALANCE
#########################################################
'''

#'''
#########################################################
# GENERATE EMISSIONS

with open('../../Data/Processed/energy_balance.pkl', 'rb') as f:  
    enebal = pickle.load(f)

#emissions factors of combustion:
#units tCO2/MWh
dirfac = {}
dirfac['fos_anthracite'] = 0.354
dirfac['fos_other_bituminous_coal'] = 0.341
dirfac['fos_sub_bituminous_coal'] = 0.346
dirfac['fos_lignite'] = 0.364
dirfac['fos_peat'] = 0.382
dirfac['fos_oil'] = 0.267
dirfac['fos_gas'] = 0.202

#1ktoe = 11GWh
kwhtoe = 11

#importing energy balance data
with open('../../Data/Processed/energy_balance.pkl', 'rb') as f:  
    # read the data as binary data stream
    enebal = pickle.load(f)

#emissions in ktCO2
em = {}
for msval in mscodes:
    em[msval[0]] = {}
    for k in range(2000, 2016):
        year = str(k)        
        em[msval[0]][year] = {}
        em[msval[0]][year]['tot'] = 0
        for kdir, dirval in enumerate(dirfac):
            tmp = list(dirfac.keys())[kdir]
            em[msval[0]][year][tmp] = enebal[msval[0]][year][tmp] * kwhtoe *dirfac[dirval]
            em[msval[0]][year]['tot'] += em[msval[0]][year][tmp]
 

#emissions: carrier, country, year
emval = np.zeros((ncar, nms, ntim))
for (ktim, timval) in enumerate(timcodes):
    for (kms, msval) in enumerate(mscodes):
        for (kcar, carval) in enumerate(carcodes):
            tmp = em[msval[0]][timval][carval]
            emval[kcar, kms, ktim] = tmp


#####################################
#storing
#emissions
with open('../../Data/Processed/emissions.pkl', 'wb') as f:  
    # store the data as binary data stream
    pickle.dump(emval, f)

# FINISHED EMISSIONS
########################################################
#'''


#'''
#########################################################
# IMPORT TRADE
#importing energy imports and exports
#nrg_125a = imports electricity
#nrg_135a = exports electricity
impdata = pd.read_csv('../../Data/Eurostat/TSV/nrg_125a.tsv')
expdata = pd.read_csv('../../Data/Eurostat/TSV/nrg_135a.tsv')


with open('../../Data/Processed/energy_balance.pkl', 'rb') as f:  
    enebal = pickle.load(f)


#unit is GWh
#convert to ktoe
gwhktoe = 85.9845 / 1000

#create trade objects
nms = len(mscodes)
ntim = 16
timcodes = [str(2000+t) for t in range(ntim)]

#Intra-EU trade
#the structure is import export time
impval = np.zeros((nms+1, nms+1, ntim))
#the structure is import export time
expval = np.zeros((nms+1, nms+1, ntim))

#fill in imports
for (krow, improw) in impdata.iterrows():
   possor = -1
   postar = -1
   strsor = improw['partner']   
   strtar = improw['geo\\time']   
   if((strsor != 'TOTAL') and (strtar != 'EA19') and (strtar != 'EU28')):  
       for (kms, msselect) in enumerate(mscodes):
          if (msselect[0] == strsor):
            possor = kms
            break 
       for (kms, msselect) in enumerate(mscodes):
          if (msselect[0] == strtar):
            postar = kms
            break 
       for (ktim, vtim) in enumerate(timcodes):
          imptmp = improw[vtim]
          if(imptmp == ': '):
            imptmp = 0
          impval[possor, postar, ktim] += float(imptmp) * gwhktoe

#fill in exports
for (krow, exprow) in expdata.iterrows():
   possor = -1
   postar = -1
   strtar = exprow['partner']   
   strsor = exprow['geo\\time']   
   if((strtar != 'TOTAL') and (strsor != 'EA19') and (strsor != 'EU28')):  
       for (kms, msselect) in enumerate(mscodes):
          if (msselect[0] == strsor):
            possor = kms
            break 
       for (kms, msselect) in enumerate(mscodes):
          if (msselect[0] == strtar):
            postar = kms
            break 
       for (ktim, vtim) in enumerate(timcodes):
          exptmp = exprow[vtim]
          if(exptmp == ': '):
            exptmp = 0
          expval[possor, postar, ktim] += float(exptmp) * gwhktoe


###########################################

#adjusting upward by moving values from row to eu trade
for k1 in range(nms):
    for k2 in range(nms):
        for kt in range(ntim):
            deltaval = expval[k1, k2, kt] - impval[k1, k2, kt] 
            if (deltaval > 0):
                impval[k1, k2, kt] += deltaval
                impval[-1, k2, kt] -= deltaval
            elif (deltaval < 0):
                expval[k1, k2, kt] -= deltaval
                expval[k1, -1, kt] += deltaval


#now creating a unified version
#interior points are average (should be same)
#row is just the good fit
#rowxrow is zero
traval = np.zeros((nms+1, nms+1, ntim))
for kt in range(ntim):
    traval[:-1,:-1,kt] = 0.5 * (impval[:-1,:-1,kt] + expval[:-1,:-1,kt])
    traval[-1,:-1,kt] = impval[-1,:-1,kt]
    traval[:-1,-1,kt] = expval[:-1,-1,kt]
    traval[-1,-1,kt] = 0

#set negatives to zero
for kt in range(ntim):
    for kreg in range(nms):
        tmp = traval[kreg, -1, kt]
        traval[kreg, -1, kt] = max(0,tmp)
        tmp = traval[-1, kreg, kt]
        traval[-1, kreg, kt] = max(0,tmp)

#set negatives to zero
for kt in range(ntim):
    for kreg in range(nms):
        tmp = traval[kreg, -1, kt]
        traval[kreg, -1, kt] = max(0,tmp)
        tmp = traval[-1, kreg, kt]
        traval[-1, kreg, kt] = max(0,tmp)

yval = np.zeros((nms, ntim))
#adding final demand
for (ktim, valtim) in enumerate(timcodes):
   for (kreg, valreg) in enumerate(mscodes):
#member states
      for (kfin, valfin) in enumerate(fincodes):
         tmp = enebal[valreg[0]][valtim][valfin]
         yval[kreg, ktim] += tmp


expval = np.zeros((nms, ntim))
impval = np.zeros((nms, ntim))
#rest of the world
for (ktim, valtim) in enumerate(timcodes):
   expval[:, ktim] = traval[:-1, -1, kt]
   impval[:, ktim] = traval[-1, :-1, kt]

'''
for (ktim, valtim) in enumerate(timcodes):
   rowout = traval[-1,:,ktim].sum()
   rowin = traval[:,-1,ktim].sum()
   if(rowin > rowout):   
      expval[-1, ktim] = rowin - rowout
'''

xval = yval + expval + traval[:-1, :-1,:].sum(axis=1)
vval = xval - impval - traval[:-1, :-1,:].sum(axis=0)


xinv = (xval != 0) / (xval + (xval == 0))
Aval = np.zeros((nms,nms,ntim))
Lval = np.zeros((nms,nms,ntim))
Rval = np.zeros((nms+1,nms+1,ntim))
bval = np.zeros((nms, ntim))
bimp = np.zeros((nms, ntim))
for (ktim, valtim) in enumerate(timcodes):
   Aval[:,:,ktim] = np.dot(traval[:-1,:-1,ktim], np.diag(xinv[:,ktim]))
   bval[:,ktim] = vval[:,ktim] * xinv[:,ktim]
   bimp[:,ktim] = impval[:,ktim] * xinv[:,ktim]
   Lval[:,:,ktim] = np.linalg.inv(np.eye(nms) - Aval[:,:,ktim])      
   Rval[:-1,:-1,ktim] = np.dot(np.dot(np.diag(bval[:,ktim]), Lval[:,:,ktim]), np.diag(yval[:,ktim]))
   Rval[:-1,-1,ktim] = np.dot(np.dot(np.diag(bval[:,ktim]), Lval[:,:,ktim]), expval[:,ktim].reshape((nms,1))).reshape(nms)
   Rval[-1,:-1,ktim] = np.dot(np.dot(bimp[:,ktim].reshape((1,nms)), Lval[:,:,ktim]), np.diag(yval[:,ktim])).reshape(nms)

traval = Rval

with open('../../Data/Processed/energy_trade.pkl', 'wb') as f:  
    pickle.dump(traval, f)


'''
In [42]: 100*(1 - emeu.sum(axis=(0,1,2,3))/dirval.sum(axis=(0,1)))
Out[42]: 
array([2.40299741, 2.41145456, 2.34309663, 2.30749364, 2.17166242,
       2.14961225, 2.12066095, 2.24162364, 2.19012442, 2.34868793,
       2.23914605, 2.37005518, 2.31337941, 2.30299012, 2.41240833,
       2.41703738])
Discrepancy due to exports to RoW is below 2.4%

Specific values are
In [38]: dirval.sum(axis=(0,1))
Out[38]: 
array([1198380.3711, 1221057.442 , 1242726.5345, 1290985.3561,
       1283888.4861, 1274484.2506, 1303649.9729, 1303187.3613,
       1233290.6245, 1132308.5015, 1144335.0479, 1123500.6519,
       1127806.3456, 1058503.4504,  975269.7845,  970260.0281])

In [39]: emeu.sum(axis=(0,1,2,3))
Out[39]: 
array([1169583.32180663, 1191612.19658336, 1213608.25093382,
       1261195.95116907, 1256006.76230056, 1247087.78104503,
       1276003.97700805, 1273974.8053089 , 1206280.0253816 ,
       1105714.1083408 , 1118711.71492908, 1096873.06644698,
       1101715.90576489, 1034126.22056594,  951742.2950225 ,
        946808.48050623])
'''

# FINISHED TRADE
########################################################


#########################################################
# IMPORT POPULATION

#importing population
pop = pd.read_csv('../../Data/Eurostat/TSV/pop.tsv')

#final energy use: sector, country, year
popval = np.zeros((nms, ntim))
for (ktim, timval) in enumerate(timcodes):
    for (kms, msval) in enumerate(mscodes):
            tmp = pop[timval].loc[pop['Code']==msval[0]].iloc[0] 
            popval[kms, ktim] = tmp


#####################################
#storing
with open('../../Data/Processed/population.pkl', 'wb') as f:  
    # store the data as binary data stream
    pickle.dump(popval, f)

# FINISHED POPULATION
########################################################

#########################################################
# IMPORT GVA

#importing GVA
gva = pd.read_csv('../../Data/Eurostat/TSV/nama_10_a64_1_Data.csv')

#unit: Chain linked volumes in MEuro of 2010
gvaval = np.zeros((nms, ntim))

for (cms, kms) in enumerate(mscodes):
    for (ctim, ktim) in enumerate(timcodes):
        vec = gva.loc[(gva['UNIT']=='CLV10_MEUR') & (gva['TIME']== float(ktim)) & (gva['GEO']== kms[0]) & (gva['NACE_R2']== 'TOTAL')] 
#        print('%s %s %s' % (kms[0], ktim, vec.Value.iloc[0].replace(' ','')))
        if (vec.Value.iloc[0].replace(' ','') == ':') :
            gvaval[cms, ctim] = 0
        else:
            gvaval[cms, ctim] = float(vec.Value.iloc[0].replace(' ',''))



#####################################
#storing
with open('../../Data/Processed/gva.pkl', 'wb') as f:  
    # store the data as binary data stream
    pickle.dump(gvaval, f)

# FINISHED GVA
########################################################

#########################################################
# BEGIN UNIFICATION

with open('../../Data/Processed/energy_balance.pkl', 'rb') as f:  
    enebal = pickle.load(f)

#fossil energy use: carrier, country, year
fosval = np.zeros((ncar, nms, ntim))
for (ktim, timval) in enumerate(timcodes):
    for (kms, msval) in enumerate(mscodes):
        for (kcar, carval) in enumerate(carcodes):
            tmp = enebal[msval[0]][timval][carval]
            fosval[kcar, kms, ktim] = tmp

#total fossil energy production: country, year
totval = np.zeros((nms, ntim))
for (ktim, timval) in enumerate(timcodes):
    for (kms, msval) in enumerate(mscodes):
        for (kcar, carval) in enumerate(carcodes):
            tmp = enebal[msval[0]][timval]['fossil']
            totval[kms, ktim] = tmp

#renewable energy use: carrier, country, year
renval = np.zeros((nren, nms, ntim))
for (ktim, timval) in enumerate(timcodes):
    for (kms, msval) in enumerate(mscodes):
        for (kcar, carval) in enumerate(rencodes):
            tmp = enebal[msval[0]][timval][carval]
            renval[kcar, kms, ktim] = tmp

#final energy use: sector, country, year
finval = np.zeros((nfin, nms, ntim))
for (ktim, timval) in enumerate(timcodes):
    for (kms, msval) in enumerate(mscodes):
        for (kcar, carval) in enumerate(fincodes):
            tmp = enebal[msval[0]][timval][carval]
            finval[kcar, kms, ktim] = tmp

#importing input data
with open('../../Data/Processed/emissions.pkl', 'rb') as f:  
    emval = pickle.load(f)


with open('../../Data/Processed/energy_trade.pkl', 'rb') as f:  
    traval = pickle.load(f)

with open('../../Data/Processed/gva.pkl', 'rb') as f:  
    gvaval = pickle.load(f)

with open('../../Data/Processed/population.pkl', 'rb') as f:  
    popval = pickle.load(f)


#save in single file
#####################################
#storing
val = {'emval': emval, 'fosval': fosval, 'renval': renval, 'totval': totval, 'finval': finval, 'traval': traval, 'gvaval': gvaval, 'popval': popval}

with open('../../Data/Processed/unified_val.pkl', 'wb') as f:  
    # store the data as binary data stream
    pickle.dump(val, f)


# FINISHED UNIFICATION
########################################################

