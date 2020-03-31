import os
import random
import pandas as pd
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

def cleansing (new = False):
	readFile = "./before_cleansing/final_before_cleansing_no_neutral.csv"
	saveFile = "./after_cleansing/final_after_cleansing_no_neutral.csv"
	readDF = pd.read_csv(readFile)
	readSEN = readDF['sentence'].tolist()
	readLAB = readDF['label'].tolist()

	if new:
		saveSEN = []
		saveLAB = []
		length = 0
	else:
		saveDF = pd.read_csv(saveFile)
		saveSEN = saveDF['sentence'].tolist()
		saveLAB = saveDF['label'].tolist()

		length = len(saveSEN)
		readSEN = readSEN[length:]
		readLAB = readLAB[length:]

	path = "/Users/amy_hyunji/Documents/GitHub/korean_multi_label_SA/korean_multi_label_SA/API/driver/chromedriver"
	chrome_options = webdriver.ChromeOptions()
	chrome_options.add_argument('headless')
	driver = webdriver.Chrome(path, options=chrome_options)
	driver.get("http://speller.cs.pusan.ac.kr")

	for i in tqdm(range(len(readSEN))):
		if (i%20 == 0):
			output_df = pd.DataFrame({"sentence": saveSEN, "label": saveLAB})
			output_df.to_csv(saveFile)

		search_sentence = readSEN[i]
		try:
			search_box = driver.find_element_by_name("text1")
			search_box.send_keys(search_sentence)
			search_box.submit()

			m = 0
			while(1):
				try:
					_new = driver.find_element_by_id("ul_{}".format(m))
					_new_array = _new.text.split("\n")
					revised = _new_array[0]
					original = _new_array[1]
					search_sentence = search_sentence.replace(original, revised, 1)
					m += 1
				except:
					break
		
			saveSEN.append(search_sentence)
			saveLAB.append(readLAB[i])

			try:
				renew_btn = driver.find_element_by_id("btnRenew2").click()
			except:
				print("[btnRenew2] Error occured during {} sentence {}".format(i+length, search_sentence))
				break
			
		except:
			print("[Text1] Error occured during {} sentence {}".format(i+length, search_sentence))
			break

	output_df = pd.DataFrame({"sentence": saveSEN, "label": saveLAB})
	output_df.to_csv(saveFile)


def check_num():
	single = "./original/korean_single_turn_utterance.xlsx"
	multi = "./original/korean_multi_turn_utterance.xlsx"
	naver = "./original/naver_movie_rating.xlsx"

	singleDF = pd.read_excel(single)
	multiDF = pd.read_excel(multi)
	naverDF = pd.read_excel(naver)

	singleSEN = singleDF['Sentence'].tolist()
	singleLAB = singleDF['Emotion'].tolist()
	singleLAB = change_label(singleLAB)
	totalPAIR = make_pair(singleSEN, singleLAB, remove_neutral) 

	multiSEN = multiDF['Sentence'].tolist()
	multiLAB = multiDF['Emotion'].tolist()
	multiLAB = change_label(multiLAB)
	totalPAIR += make_pair(multiSEN, multiLAB, remove_neutral)

	naverSEN = naverDF['original'].tolist()
	naverLAB = naverDF['label'].tolist()
	naverLAB = change_label(naverLAB)
	totalPAIR += make_pair(naverSEN, naverLAB, remove_neutral)

	random.shuffle(totalPAIR)
	sentence = []
	label = []
	for sen, lab in totalPAIR:
		sentence.append(sen)
		label.append(lab)
	
	output_df = pd.DataFrame({"sentence": sentence, "label": label})
	output_df.to_csv("./final_before_cleansing_no_neutral.csv")

def remove_neutral():
	root = "before_cleansing/final_remove_dup_before_cleansing.csv"
	save = "before_cleansing/final_before_cleansing_no_neutral.csv"

	rootDF = pd.read_csv(root)
	rootSEN = rootDF['sentence'].tolist()
	rootLAB = rootDF['label'].tolist()

	saveSEN = []; saveLAB = []

	for sen, lab in zip(rootSEN, rootLAB):
		if (lab == 0): continue
		else:
			saveSEN.append(sen)
			saveLAB.append(lab)

	check_labelnum("no_netural", saveLAB)

	_df = pd.DataFrame({"sentence": saveSEN, "label": saveLAB})
	_df.to_csv(save)

def make_pair(sen, lab, remove):
	pair = []
	for _sen, _lab in zip(sen, lab):
		if (remove and _lab == 0):
			continue
		else:
			pair.append([_sen, _lab])
	return pair

def change_label(labelList):
	returnLAB = []
	for label in labelList:
		if (label == 0 or label == "N" or label == "중립"):
			returnLAB.append(0)
		elif (label == 1 or label == "H" or label == "행복"):
			returnLAB.append(1)
		elif (label == 2 or label == "S" or label == "슬픔"):
			returnLAB.append(2)
		elif (label == 3 or label == "A" or label == "분노" or label == "혐오"):
			returnLAB.append(3)
		elif (label == 4 or label == "U" or label == "놀람"):
			returnLAB.append(4)
		elif (label == 5 or label == "F" or label == "공포"):
			returnLAB.append(5)
	return returnLAB

def main_check_labelnum():
	path = "./before_cleansing/final_before_cleansing_no_neutral.csv"
	_df = pd.read_csv(path)
	emotion = _df['label'].tolist()
	check_labelnum(path.split("/")[-1], emotion)

def check_labelnum(name, labelList):
	labelDict = {"neutral": 0, "happy": 0, "sad": 0, "angry": 0, "surprised": 0, "fear": 0}
	for label in labelList:
		if (label == 0 or label == "N" or label =="중립"):
			labelDict['neutral'] += 1
		elif (label == 1 or label == "H" or label == "행복"):
			labelDict['happy'] += 1
		elif (label == 2 or label == "S" or label == "슬픔"):
			labelDict['sad'] += 1
		elif (label == 3 or label == "A" or label == "분노" or label == "혐오"):
			labelDict['angry'] += 1
		elif (label == 4 or label == "U" or label == "놀람"):
			labelDict['surprised'] += 1
		elif (label == 5 or label == "F" or label == "공포"):
			labelDict['fear'] += 1

	print("[{}]: neutral {}, happy: {}, sad: {}, angry: {}, surprised: {}, fear: {}".format(name, labelDict['neutral'], labelDict['happy'], labelDict['sad'], labelDict['angry'], labelDict['surprised'], labelDict['fear']))

def remove_dup():
	inputFile = "./final_before_cleansing_no_neutral.csv"
	inputDF = pd.read_csv(inputFile)
	
	df_sentence = inputDF['sentence'].tolist()
	df_label = inputDF['label'].tolist()
	print("INITIAL sentence: {}".format(len(df_sentence)))

	sentence_set = list(set(df_sentence))
	print("SET sentence: {}".format(len(sentence_set)))

	sentence = []; label = []
	for sen, lab in zip(df_sentence, df_label):
		if sen in sentence_set:
			sentence.append(sen)
			label.append(lab)
			sentence_set.remove(sen)
		else:
			continue

	print("final sentence: {}, label: {}".format(len(sentence), len(label)))
	output_df = pd.DataFrame({"sentence": sentence, "label": label})
	output_df.to_csv("./final_before_cleansing_no_neutral.csv")

if __name__ == "__main__":
	cleansing()	
