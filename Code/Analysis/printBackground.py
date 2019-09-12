#print plots of background information

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
#plot electricity by source, with separate y-axes

x = np.array(range(2000, 2016))
width = 1
z = dirval.sum(axis = (1)).transpose((1,0))/1000
#z = np.diag(1/z.sum(axis = 1).flatten()) @ z
y = np.zeros((ntim,6))
y[:,0] = z[:,0]
y[:,1] = z[:,1]
y[:,2] = z[:,3]
y[:,4] = z[:,5]
y[:,5] = z[:,6]
y[:,3] = y[:,3] + z[:,2]
y[:,3] = y[:,3] + z[:,4]

w = copy.copy(y)
for k in range(1,5):
    w[:,k] = w[:,k] + w[:,k-1]

#now starting the formatting of the plot
plt.close()
fig, ax1 = plt.subplots()

ax1.bar(list(x), list(w[:,0]), width)
for k in range(1,6):
    ax1.bar(list(x), list(y[:,k]), width,
                bottom=list(w[:,k-1]))

ax1.set_xlabel('Time (year)')
ax1.set_ylabel('Carbon emissions (MtCO2eq)', color='k')
ax1.tick_params('y', colors='k')
plt.xlim((2000-0.5,2015+0.5))

keycodes = ['Anthracite', 'Bituminous coal', 'Lignite', 'Other coal', 'Oil', 'Gas']
plt.legend(keycodes, loc='center left')#, bbox_to_anchor=(0.5, 0.9), labelspacing=-2, frameon=True   )


ax2 = ax1.twinx()
ax2.plot(list(x), list(totval.sum(axis = (0))/1000),color='k')
ax2.set_ylabel('Fossil electricity (Mtoe)', color='k')
ax2.tick_params('y', colors='k')
plt.ylim((0,175))

plt.legend(['Fossil electricity'], loc='lower right')
plt.title('Carbon emissions by source and fossil electricity in the EU')
plt.savefig('../../Text/Background/fig_carbon_source.jpg', dpi = 200)
#plt.show()


writer = pd.ExcelWriter('../../Text/Background/background.xlsx', engine = 'xlsxwriter')

tmp = pd.DataFrame(data = list(y), index = list(x), columns = keycodes)
tmp.to_excel(writer, sheet_name = 'Carbon_emissions_(MtCO2eq)')

tmp = pd.DataFrame(data = list(totval.sum(axis = (0))/1000), index = list(x), columns = ['Fossil electricity (Mtoe)'])
tmp.to_excel(writer, sheet_name = 'Fossil_electricity_(Mtoe)')



#########################################################
#shares of renewables in total electricity produced

x = np.array(range(2000, 2016))
width = 1
y = renval.sum(axis = (1)).transpose((1,0))/1000

#z = np.diag(1/z.sum(axis = 1).flatten()) @ z
#y = np.zeros((ntim,5))
#y[:,0] = z[:,1]
#y[:,1] = z[:,3]
#y[:,3] = z[:,5]
#y[:,4] = z[:,6]
#y[:,2] = y[:,2] + z[:,0]
#y[:,2] = y[:,2] + z[:,2]
#y[:,2] = y[:,2] + z[:,4]

w = copy.copy(y)
for k in range(1,5):
    w[:,k] = w[:,k] + w[:,k-1]

#now starting the formatting of the plot
plt.close()
fig, ax1 = plt.subplots()

ax1.bar(list(x), list(w[:,0]), width)
for k in range(1,5):
    ax1.bar(list(x), list(y[:,k]), width,
                bottom=list(w[:,k-1]))

ax1.set_xlabel('Time (year)')
ax1.set_ylabel('Non-fossil electricity (Mtoe)', color='k')
ax1.tick_params('y', colors='k')
plt.ylim((0,175))
plt.xlim((2000-0.5,2015+0.5))

keycodes = ['Nuclear', 'Hydro', 'Wind', 'Solar', 'Other']
plt.legend(keycodes, loc='lower left')#, bbox_to_anchor=(0.5, 0.9), labelspacing=-2, frameon=True   )


ax2 = ax1.twinx()
ax2.plot(list(x), list(totval.sum(axis = (0))/1000),color='k')
ax2.set_ylabel('Fossil electricity (Mtoe)', color='k')
ax2.tick_params('y', colors='k')
plt.ylim((0,175))

plt.legend(['Fossil electricity'], loc='lower right')
plt.title('Fossil vs. non-fossil electricity generation in the EU')
plt.savefig('../../Text/Background/fig_renewable_source.jpg', dpi = 200)
#plt.show()


tmp = pd.DataFrame(data = list(y), index = list(x), columns = keycodes)

tmp.to_excel(writer, sheet_name = 'Non-fossil_electricity_(Mtoe)')


#########################################################
#shares of final demand in total electricity consumed

x = np.array(range(2000, 2016))
width = 1
y = finval.sum(axis = (1)).transpose((1,0))/1000

yarg = (-y[-1,:]).argsort()
y = y[:,yarg]

#z = np.diag(1/z.sum(axis = 1).flatten()) @ z
#y = np.zeros((ntim,5))
#y[:,0] = z[:,1]
#y[:,1] = z[:,3]
#y[:,3] = z[:,5]
#y[:,4] = z[:,6]
#y[:,2] = y[:,2] + z[:,0]
#y[:,2] = y[:,2] + z[:,2]
#y[:,2] = y[:,2] + z[:,4]

w = copy.copy(y)
for k in range(1,nfin):
    w[:,k] = w[:,k] + w[:,k-1]


#cutting to the first seven elements
wtmp = w
w = np.zeros((16,8))
w[:,:-1] = wtmp[:,:7]
w[:,-1] = wtmp[:,-1]

ytmp = y
y = np.zeros((16,8))
y[:,:-1] = ytmp[:,:7]
y[:,-1] = w[:,-1] - w[:,-2]

#now starting the formatting of the plot
plt.close()
fig, ax1 = plt.subplots()

ax1.bar(list(x), list(w[:,0]), width)
for k in range(1,8):#nfin):
    ax1.bar(list(x), list(y[:,k]), width,
                bottom=list(w[:,k-1]))

ax1.set_xlabel('Time (year)')
ax1.set_ylabel('Electricity consumption (Mtoe)', color='k')
ax1.tick_params('y', colors='k')
#plt.ylim((0,175))
plt.xlim((2000-0.5,2015+0.5))

#The first seven elements have each more than 4%
#and in total 82% of total electricity consumption
#in 2015

keycodes = ['Services', 'Residential', 'Chemical', 'Machinery', 'Paper', 'Food', 'Iron and steel', 'Other']
#keycodes = [fincodes[yarg[k]] for k in range(nfin)]
plt.legend(keycodes, loc='upper left')#, bbox_to_anchor=(0.5, 0.9), labelspacing=-2, frameon=True   )


ax2 = ax1.twinx()
ax2.plot(list(x), list(gvaval.sum(axis = (0))/1000000),color='k')
ax2.set_ylabel('GDP ($10^{12}$ euro)', color='k')
ax2.tick_params('y', colors='k')
#plt.ylim((0,175))

plt.legend(['GDP'], loc='lower right')
plt.title('Electricity consumption and GDP growth in the EU')
plt.savefig('../../Text/Background/fig_final.jpg', dpi = 200)
#plt.show()

tmp = pd.DataFrame(data = list(y), index = list(x), columns = keycodes)
tmp.to_excel(writer, sheet_name = 'Electricity_consumption_(Mtoe)')

tmp = pd.DataFrame(data = list(gvaval.sum(axis = (0))/1000000), index = list(x), columns = ['GDP ($10^{12}$ euro)'])
tmp.to_excel(writer, sheet_name = 'GDP_10^12euro')

#########################################################
#role of trade
x = np.array(range(2000, 2016))
width = 1
yemi = emeu.sum(axis = (0,3))
yene = traval[:nms, :nms, :]
yci = yemi.sum(axis=1) / (yene.sum(axis=1) + (yene.sum(axis=1)==0)) * (yene.sum(axis=1) != 0)

#carbon intensity ratio
yint = np.zeros((ntim,1))
for k in range(ntim):
    tmp1 = 0
    tmp2 = 0
    for ims in range(nms):
        for jms in range(nms):
            if (ims != jms):
                tmp1 += yci[ims,k] * yene[ims, jms, k]            
                tmp2 += yci[jms,k] * yene[ims, jms, k]            
    yint[k,0] = tmp1 / tmp2

ysha = np.zeros((ntim,1))
for k in range(ntim):
    ysha[k,0] = (1-yene[:,:,k].diagonal().sum() / yene[:,:,k].sum()) * 100

#now starting the formatting of the plot
plt.close()
fig, ax1 = plt.subplots()

ax1.bar(list(x), list(ysha.flatten()), width)

ax1.set_xlabel('Time (year)')
ax1.set_ylabel('Share of electricity trade (%)', color='k')
ax1.tick_params('y', colors='k')
#plt.ylim((0,175))
plt.xlim((2000-0.5,2015+0.5))

keycodes = ['Share of trade (%)']
plt.legend(keycodes, loc='upper left')#, bbox_to_anchor=(0.5, 0.9), labelspacing=-2, frameon=True   )


ax2 = ax1.twinx()
ax2.plot(list(x), list(yint.flatten()),color='k')
ax2.set_ylabel('Carbon intensity ratio (-)', color='k')
ax2.tick_params('y', colors='k')
#plt.ylim((0.75,1.2))

plt.legend(['Carbon intensity ratio (-)'], loc='lower right')
plt.title('Share of electricity trade and carbon intensity ratio')
plt.savefig('../../Text/Background/fig_trade.jpg', dpi = 200)
#plt.show()


tmp = pd.DataFrame(data = list(yint.flatten()), index = list(x), columns = ['Carbon intensity ratio'])
tmp.to_excel(writer, sheet_name = 'Carbon_intensity_ratio_(-)')

tmp = pd.DataFrame(data = list(ysha.flatten()), index = list(x), columns = ['Share of electricity trade'])
tmp.to_excel(writer, sheet_name = 'Electricity_trade_(%)')

#########################################################
#extra details on electricity for trade

xene = np.zeros((6,25,16))
xene[1:,:,:] = renval
xene[0,:,:] = totval
xtot = xene.sum(axis=0)

xtra = np.zeros((6,25,25,16))
for kms1 in range(nms):
    for kms2 in range(nms):
        for ktim in range(ntim):
            xtra[:, kms1, kms2, ktim] = xene[:, kms1, ktim] / xtot[kms1, ktim] * traval[kms1, kms2, ktim]

yimp = xtra.sum(axis = 1)
ydom = np.zeros((6,25,16))
for kms in range(nms):
    ydom[:,kms,:] = xtra[:, kms, kms, :] 

yimp = yimp - ydom

yimpagg = yimp.sum(axis=1)
ydomagg = ydom.sum(axis=1)

for ktim in range(ntim):
    ydomagg[:,ktim] = ydomagg[:,ktim] / ydomagg[:,ktim].sum() * 100
    yimpagg[:,ktim] = yimpagg[:,ktim] / yimpagg[:,ktim].sum() * 100

xcol = ['Fossil', 'Nuclear', 'Hydro', 'Wind', 'Solar', 'Other']

tmp = pd.DataFrame(data = ydomagg.T, index = timcodes, columns = xcol)
tmp.to_excel(writer, sheet_name = 'Frac_domestic electricity_(%)')

tmp = pd.DataFrame(data = yimpagg.T, index = timcodes, columns = xcol)
tmp.to_excel(writer, sheet_name = 'Frac_imported electricity_(%)')

#########################################################
#extra details on share of trade

writer.save()
writer.close()










