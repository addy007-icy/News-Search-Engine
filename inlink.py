from __future__ import division
from bs4 import BeautifulSoup 
import requests
import os, sys


for q in range (6,100):
	with open('Links/'+str(q), 'r') as links:
		data = links.read().split('\n')
		count = 0
		print(q)
		for line in data:
			try:
				url= line
				print(str(count/len(data)*100)+'%')
				first=url[-12:-4]
				page= requests.get(url)

				soup= BeautifulSoup(page.content, 'html.parser')

				myrealdiv= soup.find('div',id="content-body-14269002-"+first)
				title= soup.find('h1', class_='title').get_text()

				file_name= "Documents/"+title
				
				f= open(file_name, 'w')
				f.write(url+'\n\n'+myrealdiv.get_text())
				f.close()
			except:
				print("skipping coz baby im fragility")
				print(url+"\n")
			count= count+1


