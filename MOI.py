import json,math
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

peak_json = "./datasets/json/pfizer_mini.json"
gg_csv = "./datasets/gg_csv/good_groups_pfizer.csv"
cohort_csv = "./datasets/cohort_csv/cohort_pfizer.csv"

data = json.load(open(peak_json,"r"))
groups = data["groups"]
good_group_id = pd.read_csv(gg_csv,usecols = ["Gid"])
good_group_id = list(good_group_id["Gid"])
print(len(good_group_id))
cohort_data = pd.read_csv(cohort_csv,usecols = ["Sample","Cohort"])

def non_zero_filter(group):
	sample_names = list(cohort_data["Sample"])
	cohort_names = list(cohort_data["Cohort"])
	cohort_keys = list(set(cohort_names))
	c_size = {}
	for peak in group["peaks"]:
		if peak["peakMz"]!="NA":
			k = cohort_names[sample_names.index(peak["sampleName"])]
			if k in c_size.keys():
				c_size[k]+=1
			else:
				c_size[k] = 1
	m = max(list(c_size.values()))
	print(c_size.values())
	if m==0:
		print(group['groupId'],'1')
		return False
	elif (min(list(c_size.values()))/m)<0.2:
		print(group['groupId'],'2')
		return False
	else:
		print(group['groupId'],'3')
		return True

def adduct_corr(group):
	for g in groups:
		if(group['meanRt']>(g['meanRt']-0.005))and(group['meanRt']<(g['meanRt']+0.005)) and not (group['meanMz']==g['meanMz']):
			pr1 = []
			pr2 = []
			pa1 = []
			pa2 = []
			for peak in group['peaks']:
				if peak['rt']=='NA':
					peak['rt'] = 0
					peak['peakAreaTop'] = 0
				pr1.append(peak['rt'])
				pa1.append(peak['peakAreaTop'])
			for peak in g['peaks']:
				if peak['rt']=='NA':
					peak['rt'] = 0
					peak['peakAreaTop'] = 0
				pr2.append(peak['rt'])
				pa2.append(peak['peakAreaTop'])
			p1 = cosine_similarity([np.array(pr1),np.array(pr2)])
			p2 = cosine_similarity([np.array(pa1),np.array(pa2)])
			if p2[0][1]>0.95 and p1[0][1]>0.99:
				print(group['groupId'],'4')
				return True
	print(group['groupId'],'5')
	return False

def cohort_corr(group):
	peak_list = {}
	for peak in group["peaks"]:
		if peak["peakIntensity"]!='NA':
			cohort_name = list(cohort_data["Cohort"])[list(cohort_data["Sample"]).index(peak["sampleName"])]
			if cohort_name in peak_list.keys():
				peak_list[cohort_name].append(peak["peakIntensity"])
			else:
				peak_list[cohort_name] = [peak["peakIntensity"]]
	cohort_means=[]
	for cohort_key in peak_list.keys():
		m = max(peak_list[cohort_key])
		for i in range(len(peak_list[cohort_key])):
			peak_list[cohort_key][i]/=m
		di = np.var(peak_list[cohort_key])/np.mean(peak_list[cohort_key])
		if di<0.02:
			cohort_means.append(np.mean(peak_list[cohort_key])*m)
	if len(cohort_means)>(len(peak_list.keys())/2):
		m2 = max(cohort_means)
		for i in range(len(cohort_means)):
			cohort_means[i]/=m2
		if (np.var(cohort_means)/np.mean(cohort_means))>0.04:
			print(group['groupId'],'6')
			return True
	print(group['groupId'],'7')
	return False

moi_list = [[],[],[]]
for group in groups:
	r = 0
	if group["groupId"] in good_group_id:
		if(non_zero_filter(group)):
			if(adduct_corr(group)):
				r+=1
			if(cohort_corr(group)):
				r+=1
		moi_list[r].append(group["groupId"])
print(moi_list[2],'\n')
print(moi_list[1],'\n')
print("Total MOIs= ",len(moi_list[2])+len(moi_list[1]))
