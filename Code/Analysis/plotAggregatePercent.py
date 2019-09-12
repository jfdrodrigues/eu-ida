#this script generates the total emissions by
#energy carrier, country of emission, country of 
#consumption and sector of consumption

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
#Waterfall plot: single block
def plotWaterfall(ytmp1, rencodes, str_file, str_title, bottom_adjust, y_low, y_high, yref0, loc, ynum1, ynum2, vnum1, vnum2, vdisppos, vdispneg, vdisp1, vdisp2):
    nren = ytmp1.shape[1]

    x_location = np.arange(2*nren+3)
    x_width = 0.8

    for k in range(1,ntim-1):
        ytmp1[k,:] += ytmp1[k-1,:]  

    yfac = np.zeros((nren, ntim))
    yfac[:,1:] = ytmp1.transpose()



#create first block
    val_top = np.zeros(2*nren+3)
    val_top[1:1 + nren] = yfac[:,7] - yfac[:,0]
    val_top[nren + 2:-1] = yfac[:,-1] - yfac[:,7]

    val_abs = np.zeros(2*nren+3)
    val_abs[0] = yref0
    val_abs[nren+1] = (yfac[:,7] - yfac[:,0]).sum()  
    val_abs[-1] = (yfac[:,-1] - yfac[:,7]).sum()

    val_ref = copy.copy(val_top)
    val_bas = np.zeros(2*nren+3)
    val_bas[nren + 1] = val_abs[0]
    val_bas[-1] = val_abs[nren+1] + val_abs[0]
    for k in range(1,2*nren+2):
        if (val_abs[k]!=0):
            continue# - 2*val_top[k]        
        elif ((val_top[k]>0) & (val_ref[k-1]<=0)):
            val_bas[k] = val_abs[k-1] + val_bas[k-1]# - 2*val_top[k]        
    #        print(k,'+-', val_bas[6])    
        elif ((val_top[k]>0) & (val_ref[k-1]>0)):
            val_bas[k] = val_abs[k-1] + val_bas[k-1] + val_top[k-1]         
    #        print(k,'++', val_bas[6])    
        elif ((val_top[k]<=0) & (val_ref[k-1]>0)):
            val_bas[k] = val_abs[k-1] + val_bas[k-1] + val_top[k-1] + + val_top[k]         
            val_top[k] = abs(val_top[k])
    #        print(k,'++', val_bas[6])    
        else:   
            val_bas[k] = val_abs[k-1] + val_bas[k-1] + val_top[k] 
            val_top[k] = abs(val_top[k])
    #        print(k,'--', val_bas[6])    


    plt.close()
    fig0 = plt.figure()
    fig0.subplots_adjust(bottom=bottom_adjust)

#    plt.plot([0,nren+1], [val_abs[0],val_abs[0]], color='k', linewidth = 1)
#    plt.plot([nren+1,2*nren+2], [val_abs[nren+1] + val_abs[0],val_abs[nren+1] + val_abs[0]], color='k', linewidth = 1)

    p1 = plt.bar(x_location, val_bas, x_width, color='white')
    p2 = plt.bar(x_location, val_abs, x_width, bottom=val_bas)
    p3 = plt.bar(x_location, val_top, x_width, bottom=val_bas)

    for k in range(1, 2*nren+3):
        if (val_ref[k]>0):
            y_val = val_abs[k] + val_bas[k]
        else:    
            y_val = val_abs[k] + val_bas[k] + val_top[k]
        plt.plot([k-1 - x_width/2, k + x_width/2], [y_val, y_val], color='k', linestyle='-', linewidth=2)

#adding percentages
    yold  = yref0
    for k in range(2, nren+2):
        if (val_ref[k]>0):
            y_val = val_abs[k] + val_bas[k]
        else:    
            y_val = val_abs[k] + val_bas[k] + val_top[k]
        if y_val > yold:
            vdisp = vdisppos + vnum1[k-2]
        else:
            vdisp = vdispneg
        valtmp = (y_val-yold)/ynum1*100    
        yold = y_val
        plt.annotate('%.1f%%' % valtmp,(k-1.2,y_val+vdisp), rotation = '90')

    valtmp = (y_val-yref0)/ynum1*100    
    plt.annotate('%.1f%%' % valtmp,(k-0.2,y_val+vdisp+vdisp1), rotation = '90')

#    pdb.set_trace() 

    yref0 = y_val
    yold  = yref0
    for k in range(nren+3, 2*nren + 3):
        if (val_ref[k]>0):
            y_val = val_abs[k] + val_bas[k]
        else:    
            y_val = val_abs[k] + val_bas[k] + val_top[k]
        if y_val > yold:
            vdisp = vdisppos + vnum2[k-3-nren]
        else:
            vdisp = vdispneg
        valtmp = (y_val-yold)/ynum2*100    
        yold = y_val
        plt.annotate('%.1f%%' % valtmp,(k-1.2,y_val+vdisp), rotation = '90')

    valtmp = (y_val-yref0)/ynum2*100    
    plt.annotate('%.1f%%' % valtmp,(k-0.2,y_val+vdisp+vdisp2), rotation = '90')

    keycodes = copy.copy(rencodes)
    keycodes.reverse()
    keycodes.append('2000')
    keycodes.reverse()
    keycodes.append('2000-2007')
    for tmp in rencodes:
        keycodes.append(tmp)
    keycodes.append('2007-2015')

    plt.ylabel('Carbon emissions [MtCO2]')
    plt.title(str_title)
    plt.xticks(x_location, keycodes, rotation = 90)
    plt.ylim(y_low, y_high)
    #plt.yticks(np.arange(600, 1200, 100))
    plt.legend((p1[0], p2[0], p3[0]), ('', 'Total', 'Factor'), loc = loc, frameon=False)


#    pdb.set_trace() 
#    ax = fig.add_subplot(111)
    plt.savefig(str_file, dpi = 200)
#    plt.show()

    return
#########################################################

writer = pd.ExcelWriter('../../Text/Results/aggregate.xlsx', engine = 'xlsxwriter')


#########################################################
#Split waterfall plot renewable
loc = 'lower left'
yref0 = 0
yref1 = emeu[:,:,:,:,0].sum(axis = (0,1,2,3))/1000
yref2 = emeu[:,:,:,:,7].sum(axis = (0,1,2,3))/1000
ytmp1 = drenmed.sum(axis=(0))/1000
str_file = '../../Text/Results/fig_waterfall_EU_renewable.jpg'
str_label = ['Nuclear', 'Hydro', 'Wind', 'Solar', 'Other']
str_title = 'Decomposition of non-fossil electricity types'
bottom_adjust = 0.2;
y_low = -175
y_high = 130
vnum1 = np.zeros((nren,1))
vnum2 = np.zeros((nren,1))
#vnum1[5] = 0
vdisppos = 27
vdispneg = -12
vdisp1 = -88
vdisp2 = 0

ytmp3 = copy.copy(ytmp1)
plotWaterfall(ytmp3, str_label, str_file, str_title, bottom_adjust, y_low, y_high, yref0, loc, yref1, yref2, vnum1, vnum2, vdisppos, vdispneg, vdisp1, vdisp2)


tmp = pd.DataFrame(data = list(ytmp1), index = dtim, columns = str_label)
tmp.to_excel(writer, sheet_name = 'Renewable_(MtCO2)_year')

nlabel = len(str_label)
for k in range(1,ntim-1):
    ytmp1[k,:] += ytmp1[k-1,:]  

val_top = np.zeros((3,nlabel))
val_top[0,:] = ytmp1[6,:]
val_top[1,:] = ytmp1[-1,:] - ytmp1[6,:]
val_top[2,:] = ytmp1[-1,:]

tmp = pd.DataFrame(data = list(val_top), index = ['2000-2007', '2007-2015', '2000-2015'], columns = str_label)
tmp.to_excel(writer, sheet_name = 'Renewable_(MtCO2)_agg')
#########################################################


#########################################################
#Split waterfall plot final
loc = 'lower left'
yref0 = 0
ytmp1 = dfactor[:,:,:,:,:,4].sum(axis=(0,1,2))/1000
yref1 = emeu[:,:,:,:,0].sum(axis = (0,1,2,3))/1000
yref2 = emeu[:,:,:,:,7].sum(axis = (0,1,2,3))/1000
ytmp1 = ytmp1.T

ytmp0 = copy.copy(ytmp1)
vpos = (ytmp1.sum(axis = 0)).argsort()
ytmp2 = ytmp1[:,vpos]
ytmp1 = np.zeros((ntim-1,7))
ytmp1[:,:4] = ytmp2[:,:4]
ytmp1[:,-2:] = ytmp2[:,-2:]
ytmp1[:,4] = ytmp2[:,4:-2].sum(axis = 1)

#print(ytmp1[:,0])
str_file = '../../Text/Results/fig_waterfall_EU_final.jpg'

#str_label = [fincodes[vpos[0]], fincodes[vpos[1]], fincodes[vpos[2]], fincodes[vpos[3]], 'Other', fincodes[vpos[-2]], fincodes[vpos[-1]]]
str_label = ['Residential', 'Chemical', 'Iron & steel', 'NS industry', 'Other', 'Machinery', 'Services']

#['Nuclear', 'Hydro', 'Wind', 'Solar', 'Other']
str_title = 'Decomposition of final use of electricity'
bottom_adjust = 0.2;
y_low = -210
y_high = 5
vnum1 = np.zeros((7,1))
vnum2 = np.zeros((7,1))
vnum1[5] = 0
vdisppos = 18
vdispneg = -8
vdisp1 = -27
vdisp2 = -27

ytmp3 = copy.copy(ytmp1)
plotWaterfall(ytmp3, str_label, str_file, str_title, bottom_adjust, y_low, y_high, yref0, loc, yref1, yref2, vnum1, vnum2, vdisppos, vdispneg, vdisp1, vdisp2)

tmp = pd.DataFrame(data = list(ytmp1), index = dtim, columns = str_label)
tmp.to_excel(writer, sheet_name = 'Final_(MtCO2)_year')

nlabel = len(str_label)
ytmp3 = copy.copy(ytmp1)
for k in range(1,ntim-1):
    ytmp3[k,:] += ytmp3[k-1,:]  

val_top = np.zeros((3,nlabel))
val_top[0,:] = ytmp3[6,:]
val_top[1,:] = ytmp3[-1,:] - ytmp3[6,:]
val_top[2,:] = ytmp3[-1,:]

tmp = pd.DataFrame(data = list(val_top), index = ['2000-2007', '2007-2015', '2000-2015'], columns = str_label)
tmp.to_excel(writer, sheet_name = 'Final_(MtCO2)_agg')
#########################################################


#########################################################
#Split waterfall plot fossil carriers
loc = 'lower left'
yref0 = 0
yref1 = emeu[:,:,:,:,0].sum(axis = (0,1,2,3))/1000
yref2 = emeu[:,:,:,:,7].sum(axis = (0,1,2,3))/1000
ytmp2 = dfactor[:,:,:,:,:,0].sum(axis=(1,2,3))/1000
ytmp1 = np.zeros((6,15))
ytmp1[0,:] = ytmp2[0,:]
ytmp1[1,:] = ytmp2[1,:]
ytmp1[2,:] = ytmp2[3,:]
ytmp1[3,:] = ytmp2[2,:]
ytmp1[3,:] += ytmp2[4,:]
ytmp1[4,:] = ytmp2[5,:]
ytmp1[5,:] = ytmp2[6,:]
ytmp1 = ytmp1.T
#Notice that minor types of coal are aggregated
str_file = '../../Text/Results/fig_waterfall_EU_fossil.jpg'
str_label = ['Anthracite', 'Bitum. coal', 'Lignite', 'Other coal', 'Oil', 'Gas']
str_title = 'Decomposition of fossil fuel carriers'
bottom_adjust = 0.2;
y_low = -250
y_high = 50
vnum1 = np.zeros((6,1))
vnum2 = np.zeros((6,1))
vnum1[5] = 0
vdisppos = 25
vdispneg = -12
vdisp1 = -37
vdisp2 = 0

ytmp3 = copy.copy(ytmp1)
plotWaterfall(ytmp3, str_label, str_file, str_title, bottom_adjust, y_low, y_high, yref0, loc, yref1, yref2, vnum1, vnum2, vdisppos, vdispneg, vdisp1, vdisp2)

tmp = pd.DataFrame(data = list(ytmp1), index = dtim, columns = str_label)
tmp.to_excel(writer, sheet_name = 'Fossil_(MtCO2)_year')

nlabel = len(str_label)
for k in range(1,ntim-1):
    ytmp1[k,:] += ytmp1[k-1,:]  

val_top = np.zeros((3,nlabel))
val_top[0,:] = ytmp1[6,:]
val_top[1,:] = ytmp1[-1,:] - ytmp1[6,:]
val_top[2,:] = ytmp1[-1,:]

tmp = pd.DataFrame(data = list(val_top), index = ['2000-2007', '2007-2015', '2000-2015'], columns = str_label)
tmp.to_excel(writer, sheet_name = 'Fossil_(MtCO2)_agg')
#########################################################

'''
#########################################################
#Split waterfall plot region
loc = 'lower center'
yref0 = emeu[:,:,:,:,0].sum(axis = (0,1,2,3))/1000
yref2 = emeu[:,:,:,:,7].sum(axis = (0,1,2,3))/1000
ytmp1 = dfactor.sum(axis=(0,2, 3, 5))/1000
ytmp1 = ytmp1.T


ytmp0 = copy.copy(ytmp1)
vpos = (ytmp1.sum(axis = 0)).argsort()
ytmp2 = ytmp1[:,vpos]
ytmp1 = np.zeros((ntim-1,8))
ytmp1[:,:6] = ytmp2[:,:6]
ytmp1[:,-1:] = ytmp2[:,-1:]
ytmp1[:,6] = ytmp2[:,6:-1].sum(axis = 1)

str_file = '../../Text/Results/fig_waterfall_EU_region.jpg'
str_label = []
for tmp in vpos[:6]:
    str_label.append(mscodes[tmp][0])
str_label.append('Other')
str_label.append(mscodes[vpos[-1]][0])


str_title = 'Decomposition by country'
bottom_adjust = 0.2;
y_low = 870
y_high = 1335
vnum1 = np.zeros((nms,1))
vnum2 = np.zeros((nms,1))
vnum1[5] = 0
vdisppos = 40
vdispneg = -18
vdisp1 = 0
vdisp2 = -57

ytmp3 = copy.copy(ytmp1)
plotWaterfall(ytmp3, str_label, str_file, str_title, bottom_adjust, y_low, y_high, yref0, loc, yref0, yref2, vnum1, vnum2, vdisppos, vdispneg, vdisp1, vdisp2)

tmp = pd.DataFrame(data = list(ytmp1), index = dtim, columns = str_label)
tmp.to_excel(writer, sheet_name = 'Country_(MtCO2)_year')

nlabel = len(str_label)
for k in range(1,ntim-1):
    ytmp1[k,:] += ytmp1[k-1,:]  

val_top = np.zeros((3,nlabel))
val_top[0,:] = ytmp1[6,:]
val_top[1,:] = ytmp1[-1,:] - ytmp1[6,:]
val_top[2,:] = ytmp1[-1,:]

tmp = pd.DataFrame(data = list(val_top), index = ['2000-2007', '2007-2015', '2000-2015'], columns = str_label)
tmp.to_excel(writer, sheet_name = 'Country_(MtCO2)_agg')
#########################################################
'''

#########################################################
#Split waterfall plot factor
loc = 'upper left'
yref0 = emeu[:,:,:,:,0].sum(axis = (0,1,2,3))/1000
yref2 = emeu[:,:,:,:,7].sum(axis = (0,1,2,3))/1000
ytmp1 = dfactor.sum(axis=(0,1,2,3))/1000
str_file = '../../Text/Results/fig_waterfall_EU_factor.jpg'
str_label = ['Share fossil', 'Share non-foss.', 'Transport eff.', 'Trade share', 'Final use eff.', 'GDP per capita', 'Population']
str_title = 'Decomposition by factor'
bottom_adjust = 0.3;
y_low = 800
y_high = 1400
vnum1 = np.zeros((nfactor,1))
vnum2 = np.zeros((nfactor,1))
vnum1[5] = 17
vdisppos = 62
vdispneg = -25
vdisp1 = 0
vdisp2 = -85


ytmp3 = copy.copy(ytmp1)
plotWaterfall(ytmp3, str_label, str_file, str_title, bottom_adjust, y_low, y_high, yref0, loc, yref0, yref2, vnum1, vnum2, vdisppos, vdispneg, vdisp1, vdisp2)

tmp = pd.DataFrame(data = list(ytmp1), index = dtim, columns = str_label)
tmp.to_excel(writer, sheet_name = 'Factors_(MtCO2)_year')

nlabel = len(str_label)
ytmp2 = copy.copy(ytmp1)
for k in range(1,ntim-1):
    ytmp2[k,:] += ytmp2[k-1,:]  

val_top = np.zeros((3,nlabel))
val_top[0,:] = ytmp2[6,:]
val_top[1,:] = ytmp2[-1,:] - ytmp2[6,:]
val_top[2,:] = ytmp2[-1,:]

tmp = pd.DataFrame(data = list(val_top), index = ['2000-2007', '2007-2015', '2000-2015'], columns = str_label)
tmp.to_excel(writer, sheet_name = 'Factors_(MtCO2)_agg')
#########################################################



#########################################################
yref0 = emeu.sum(axis = (0,1,2,3))/1000

tmp = pd.DataFrame(data = list(yref0), index = timcodes, columns = ['Total'])
tmp.to_excel(writer, sheet_name = 'Emissions_(MtCO2)_year')
#########################################################


writer.save()
writer.close()

#########################################################
#########################################################
#########################################################

