import json,math
import pandas as pd
import os.path
import csv
import numpy as np

i=0
groups=[]
while(True):
        f_name="28-01-2019_11_51_32_Peak_table_-1_"+str(i)+".json"
        if os.path.isfile(f_name):
                yo = json.load(open(f_name,"r"))
                groups.append(yo['groups'])
                i+=1
                print(i)
        else:
                break
group_training_data = []
peaks_training_data = []

def prepare_group_features(group):
        group_train_data={}
        features = parse_features(group)
        label = int(group['label']=="g")
        group_train_data.update({"features":features,"label":label})
        return group_train_data

def validate_group(group):
        if group['compound']['tagString']=="C12 PARENT":
        	return group['label']!=""
        return False

def parse_features(group):
        features = {}
        features['AvgPeakAreaFractional'] = getAvgPeakAreaFractional(group)
        features['AvgNoNoiseFraction'] = getAvgNoNoiseFraction(group)
        features['getAvgSymmetry'] = getAvgSymmetry(group) / (getAvgWidth(group) + 1) * math.log(getAvgWidth(group) + 1)
        features['GroupOverlapFrac'] = getGroupOverlapFrac(group)
        features['MaxGaussFitR2'],features['MinGaussFitR2'],features['AvgGaussFitR2']=getMinGaussFitR2(group)
        features['LogMaxSignalBaselineRatio'],features['LogMinSignalBaselineRatio'],features['LogAvgSignalBaselineRatio'] = getLogAvgSignalBaselineRatio(group)
        if (getLogAvgPeakIntensity(group) > 0):
                features['LogAvgPeakIntensity'] = math.log(getLogAvgPeakIntensity(group))
        else:
                features['LogAvgPeakIntensity'] = 0.0
        if (getAvgWidth(group) <= 3.0 and getLogAvgSignalBaselineRatio(group)[2] >= 3.0):
                features['WidthLogAvgSignalBaselineRatio'] = 1
        else:
                features['WidthLogAvgSignalBaselineRatio'] = 0
        features['RelativeMaxPeakIntensity']=getRelMaxPeak(group)
        features['mzCorrelation']=0.0
        features['GroupVariance']=getGroupVariance(group)
        features['RelativeIntensityMax']=0.0
        features['RelativeIntensityMin']=0.0
        features['RelativeIntensityAvg']=0.0
        if(os.path.isfile('cohort.csv')):
                features['IntragroupVarianceMax'],features['IntragroupVarianceMin'],features['IntragroupVarianceAvg']=getIntragroupVariance(group)
                features['MaxNonZeroSample'],features['MinNonZeroSample'],features['AvgNonZeroSample']=getCohortSampleRatio(group)
                features['MaxCohortQuality'],features['MinCohortQuality'],features['AvgCohortQuality']=getCohortQuality(group)
        return features

def prepare_peak_features(peak):
        return peak

def getAvgPeakAreaFractional(group):
        AF_list = []
        for peak in group['peaks']:
                if peak['peakAreaFractional']=="NA":
                        peak['peakAreaFractional']=0
                if peak['peakWidth']=="NA":
                        peak['peakWidth']=0
                if peak['peakWidth'] > 0:
                        AF_list.append(peak['peakAreaFractional'])
        return sum(AF_list)/len(AF_list)

def getAvgNoNoiseFraction(group):
        AF_list = []
        for peak in group['peaks']:
                if peak['noNoiseFraction']=="NA":
                        peak['noNoiseFraction']=0
                if peak['peakWidth']=="NA":
                        peak['peakWidth']=0
                if peak['peakWidth'] > 0:
                        AF_list.append(peak['noNoiseFraction'])
        return sum(AF_list)/len(AF_list)

def getAvgSymmetry(group):
        AF_list = []
        for peak in group['peaks']:
                if peak['symmetry']=="NA":
                        peak['symmetry']=0
                if peak['peakWidth']=="NA":
                        peak['peakWidth']=0
                if peak['peakWidth'] > 0:
                        AF_list.append(peak['symmetry'])
        return sum(AF_list)/len(AF_list)

def getAvgWidth(group):
        AF_list = []
        for peak in group['peaks']:
                if peak['peakWidth']=="NA":
                        peak['peakWidth']=0
                if peak['peakWidth'] > 0:
                        AF_list.append(peak['peakWidth'])
        return sum(AF_list)/len(AF_list)

def getGroupOverlapFrac(group):
        AF_list = []
        for peak in group['peaks']:
                if peak['peakWidth']=="NA":
                        peak['peakWidth']=0
                if peak['peakWidth'] > 0:
                        AF_list.append(pow(math.e, peak['groupOverlapFrac']))
        return math.log(sum(AF_list) / len(AF_list))

def getMinGaussFitR2(group):
        for peak in group['peaks']:
                if peak['gaussFitR2']=="NA":
                        peak['gaussFitR2']=0
        AF_list = [peak['gaussFitR2'] for peak in group['peaks'] if peak['peakWidth']>0]
        return max(AF_list),min(AF_list),(sum(AF_list)/len(AF_list))

def getLogAvgSignalBaselineRatio(group):
        AF_list = []
        for peak in group['peaks']:
                if peak['peakWidth']=="NA":
                        peak['peakWidth']=0
                if peak['peakWidth'] > 0:
                        AF_list.append(math.log(peak['signalBaselineRatio'],2))
        return max(0,pow(2,max(AF_list)))/10.0,max(0,pow(2,min(AF_list)))/10.0,max(0,pow(2,sum(AF_list) / len(AF_list)))/10.0

def getLogAvgPeakIntensity(group):
        AF_list = []
        for peak in group['peaks']:
                if peak['peakWidth']=="NA":
                        peak['peakWidth']=0
                if peak['peakWidth'] > 0:
                        AF_list.append(math.log(peak['peakIntensity'],2))
        return pow(2,sum(AF_list) / len(AF_list))

def getRelMaxPeak(group):
        max=0.0
        max_eic=0.0
        for peak in group['peaks']:
                if peak['peakIntensity']=="NA":
                        peak['peakIntensity']=0
                if peak['peakIntensity']>max:
                        max=peak['peakIntensity']
                        max_eic=getEicMax(peak['eic'])
        return max/max_eic

def getEicMax(eic):
        return max(eic['intensity'])

def getGroupVariance(group):
        p=[0]*len(group['peaks'])
        i=0
        for peak in group['peaks']:
                if peak['peakIntensity']=="NA":
                        peak['peakIntensity']=0
                p[i]=peak['peakIntensity']
                i+=1
        return np.var(p)

def getGroupPeakRow(group):
        p=[]
        for peak in group['peaks']:
                if peak['peakIntensity']!="NA" and peak['peakIntensity']>0:
                        p.append(peak['peakIntensity'])
        row=[max(p),min(p),sum(p)/len(p),group['meanMz']]
        return row

def getIntragroupVariance(group):
        with open('cohort.csv','r')as f:
                reader=csv.reader(f)
                names=list(reader)
        c_table={}
        variance=[]
        cohort_keys=[]
        for i in range(len(names)-1):
                if names[i+1][1] not in cohort_keys:
                        cohort_keys.append(names[i+1][1])
                        c_table[names[i+1][1]]=[]
        for peak in group['peaks']:
                if peak['peakAreaFractional']!="NA":
                        c_table[names[[x[0] for x in names].index(peak['sampleName'])][1]].append(peak['peakIntensity'])
        for key in cohort_keys:
                variance.append(np.var(c_table[key]))
        return max(variance),min(variance),(sum(variance)/len(variance))

def getCohortSampleRatio(group):
        with open('cohort.csv','r')as f:
                reader=csv.reader(f)
                names=list(reader)
        cohort_keys=[]
        c_size={}
        for i in range(len(names)-1):
                if names[i+1][1] not in cohort_keys:
                        cohort_keys.append(names[i+1][1])
                        c_size[names[i+1][1]]=[1,0]
                else:
                        c_size[names[i+1][1]][0]+=1
        for peak in group['peaks']:
                if peak['sampleName']!="NA":
                        c_size[names[[x[0] for x in names].index(peak['sampleName'])][1]][1]+=1
        rat=[]
        for key in cohort_keys:
                rat.append(c_size[key][1]/c_size[key][0])
        return max(rat),min(rat),(sum(rat)/len(rat))

def getCohortQuality(group):
        with open('cohort.csv','r')as f:
                reader=csv.reader(f)
                names=list(reader)
        cohort_keys=[]
        c_quality={}
        for i in range(len(names)-1):
                if names[i+1][1]not in cohort_keys:
                        cohort_keys.append(names[i+1][1])
                        c_quality[names[i+1][1]]=0
        for peak in group['peaks']:
                if peak['quality']!="NA":
                        c_quality[names[[x[0] for x in names].index(peak['sampleName'])][1]]+=peak['quality']
        return max(c_quality.values()),min(c_quality.values()),sum(c_quality.values())/len(c_quality)

def export_training_data(final_data,name):
        if (os.path.isfile('cohort.csv')):
                feature_keys = ['LogAvgSignalBaselineRatio', 'getAvgSymmetry', 'AvgNoNoiseFraction', 'AvgPeakAreaFractional','MaxGaussFitR2', 'MinGaussFitR2','AvgGaussFitR2', 'GroupOverlapFrac','LogMaxSignalBaselineRatio','LogMinSignalBaselineRatio', 'LogAvgPeakIntensity', 'WidthLogAvgSignalBaselineRatio','RelativeMaxPeakIntensity','mzCorrelation','GroupVariance','RelativeIntensityMax','RelativeIntensityMin','RelativeIntensityAvg','IntragroupVarianceMax','IntragroupVarianceMin','IntragroupVarianceAvg','MaxNonZeroSample','MinNonZeroSample','AvgNonZeroSample','MaxCohortQuality','MinCohortQuality','AvgCohortQuality']
        else:
                feature_keys = ['LogMaxSignalBaselineRatio','LogMinSignalBaselineRatio','LogAvgSignalBaselineRatio', 'getAvgSymmetry', 'AvgNoNoiseFraction', 'AvgPeakAreaFractional','MaxGaussFitR2', 'MinGaussFitR2','AvgGaussFitR2','GroupOverlapFrac', 'LogAvgPeakIntensity', 'WidthLogAvgSignalBaselineRatio','RelativeMaxPeakIntensity','mzCorrelation','GroupVariance','RelativeIntensityMax','RelativeIntensityMin','RelativeIntensityAvg','NoOfNonZeroPeaks']
        result_data = [[data['features'][key] for key in feature_keys] + [data['label']] for data in final_data]
        result_df = pd.DataFrame(result_data)
        result_df.columns = feature_keys + ['label']
        result_df.to_csv(name,index=False)

rt_list=[]
grp_peak_list=[] ##0-max	1-min	2-avg	3-mz
for i in range(len(groups)):
        for group in groups[i]:
                peaks = group['peaks']
                try:
                        if validate_group(group):
                                 group_training_data.append(prepare_group_features(group))
                                 rt_list.append(group['meanRt'])
                                 grp_peak_list.append(getGroupPeakRow(group))
                except Exception as e:
                        print ("error - \n",str(e))
for i in range(len(rt_list)):
        group_training_data[i]['features']['mzCorrelation']=len([x for x in rt_list if x>rt_list[i]-0.075 and x<rt_list[i]+0.075])/len(rt_list)
        maxi,mini=grp_peak_list[i][0:2]
        avg=0
        k=0
        for j in range(len(grp_peak_list)):
                if(grp_peak_list[i][3]==grp_peak_list[j][3]):
                        maxi=max(maxi,grp_peak_list[j][0])
                        mini=min(mini,grp_peak_list[j][1])
                        avg+=grp_peak_list[j][2]
                        k+=1
        group_training_data[i]['features']['RelativeIntensityMax']=grp_peak_list[i][0]/maxi
        group_training_data[i]['features']['RelativeIntensityMin']=grp_peak_list[i][1]/mini
        group_training_data[i]['features']['RelativeIntensityAvg']=(grp_peak_list[i][2]*k)/avg

print(len(group_training_data))
train_data,validation_data = group_training_data[0:int(0.8*len(group_training_data))],group_training_data[int(0.8*len(group_training_data)):len(group_training_data)]
export_training_data(train_data,"training_NN.csv")
export_training_data(validation_data,"testing_NN.csv")
