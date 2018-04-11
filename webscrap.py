"""
This file generates the links to scrap the text news data for the year give
by import date module
"""


from bs4 import BeautifulSoup
"""
BeautifulSoup version 4 is used for scraping the data
"""
import requests

import date
"""
-> Requets module is used to send and recieve data packets from the requested urls
-> It establishes a https connection with the website server
"""

def release_links():
	links= date.link_date()

	for k in enumerate(links):
		"""
		url variable stores the address of each of the page from thehindu.com
		"""
		url = "http://www.thehindu.com/archive/web"+k[1]

		# print(page.text)
		page= requests.get(url)

		soup= BeautifulSoup(page.content, 'html.parser')
		unlisted_links = soup.find_all('ul', class_='archive-list')

		page_links=[]
		f=open('Links/'+str(k[0]), 'a')


		for i in unlisted_links:
			listed_art= i.find_all('li')
			for j in listed_art:
				page_links.append(j.a['href'])
				f.write(j.a['href']+'\n')

		print(k[1])	
		print(len(page_links))
		f.close()


