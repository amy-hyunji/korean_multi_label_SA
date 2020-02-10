import os
import pandas as pd

def search(df, name):
	neutral = 0; happy = 0; sad = 0; anger = 0; surprised = 0;
	total_num = int(len(df.index))
	for i in range(total_num):
		emo = df['label'][i]
		sentence = df['sentence'][i]
		if (emo == 0): neutral += 1
		elif (emo == 1): happy += 1
		elif (emo == 2): sad += 1
		elif (emo == 3): anger += 1
		elif (emo == 4): surprised += 1
		else:
				print("WRONG!!!")
				break
	print("{}: neutral: {}, happy: {}, sad: {}, anger: {}, surprised: {}, total: {}".format(name,neutral, happy, sad, anger, surprised, total_num))

elem = os.listdir("./")
for _elem in elem:
	if not ".csv" in _elem:
		continue
	else:
		df = pd.read_csv(_elem)
		search(df, _elem)
