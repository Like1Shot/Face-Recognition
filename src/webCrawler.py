from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request
import socket
import os

people = [
["Robert Downey Jr", "Robert Downey", "로버트 다우니", "로버트 다우니 주니어"], 
["Chris Evans", "크리스 에반스"], 
["Chris Hemsworth", "크리스 햄스워스"], 
["Scarlett Johansson", "스칼렛 요한슨"], 
["Mark Ruffalo", "마크 러팔로"], 
["Jeremy Renner", "제레미 레너"], 
["Paul Rudd", "폴 러드"], 
["Brie Larson", "브리 라슨"], 
["Tom Hiddleston", "톰 히들스턴"], 
["Chris Pratt", "크리스 프랫"], 
["Tom Holland", "톰 홀란드"], 
["Benedict Cumberbatch", "베네딕트 컴버배치"], 
["Anthony Mackie", "앤서니 매키"], 
["Samuel L Jackson", "사뮤엘 잭슨"], 
["wonyoung jang", "장원영"], 
["yena choi", "최예나"], 
["chaewon kim", "김채원"], 
["nako yabuki", "야부키 나코"], 
["minju kim", "김민주"], 
["yuri jo", "조유리"], 
["hitomi honda", "혼다 히토미"], 
["sakura miyawaki", "미야와키 사쿠라"], 
["yujin ahn", "안유진"], 
["chaeyeon lee", "이채연"], 
["eunbi kwon", "권은비"], 
["hyewon kang", "강혜원"]
]

socket.setdefaulttimeout(60)
def crawl_keyword(keyword):
	if not os.path.exists(keyword):
		os.makedirs(keyword)

	driver = webdriver.Chrome()
	driver.get("https://www.google.co.kr/imghp?hl=ko&tab=wi&authuser=0&ogbl")
	elem = driver.find_element_by_name("q")
	elem.send_keys(keyword)
	elem.send_keys(Keys.RETURN)

	SCROLL_PAUSE_TIME = 1
	# Get scroll height
	last_height = driver.execute_script("return document.body.scrollHeight")
	while True:
		# Scroll down to bottom
		driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
		# Wait to load page
		time.sleep(SCROLL_PAUSE_TIME)
		# Calculate new scroll height and compare with last scroll height
		new_height = driver.execute_script("return document.body.scrollHeight")
		if new_height == last_height:
		    try:
		        driver.find_element_by_css_selector(".mye4qd").click()
		    except:
		        break
		last_height = new_height

	images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd")
	count = 1
	imgUrl = ""
	for image in images:
		imgUrl = ""
		try:
		    image.click()
		    imgUrl = driver.find_element_by_xpath('/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div/a/img').get_attribute("src")
		    while (imgUrl[:5] == "data:" or imgUrl == ""):
		    	for i in range(10):
		    		time.sleep(0.1)
		    		imgUrl = driver.find_element_by_xpath('/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div/a/img').get_attribute("src")

		    	print(".")
		    	image.click()

		    imgUrl = driver.find_element_by_xpath('/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div/a/img').get_attribute("src")
		    print(imgUrl)
		#except ElementClickInterceptedException as e:
		#    print("Part#1 ElementClickInterceptedException {}".format(e))
		#    pass
		except Exception as e:
		    print("Part#1 {}".format(e))
		    #print(imgUrl)
		    pass
		
		try:
		    #subimgUrl = driver.find_element_by_xpath('/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[3]/div[3]/c-wiz/div/div/div[1]/div[3]/div[1]/div[2]/a[1]/div[1]/img').
		    opener=urllib.request.build_opener()
		    opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
		    urllib.request.install_opener(opener)
		    print(imgUrl)
		    savePath = "{0}/{0}_{1:04d}.jpg".format(keyword, count)
		    urllib.request.urlretrieve(imgUrl, savePath)
		    print(savePath)
		    count = count + 1
		except ConnectionRefusedError:
		    print ("ConnectionRefusedError")
		    pass
		except Exception as e:
		    print("Part#2 {}".format(e))
	driver.close()
	
#keywords = ["Robert Downey Jr", "Robert Downey", "로버트 다우니 주니어", "Chris Evans", "크리스 에반스", "Chris Hemsworth", "크리스 햄스워스", "Scarlett Johansson"]
keywords = ["wonyoung jang", "yena choi", "chaewon kim", "nako yabuki", "minju kim", "yuri jo", "hitomi honda", "sakura miyawaki", "yujin ahn", "chaeyeon lee", "eunbi kwon", "hyewon kang"]


for k in keywords:
	crawl_keyword(k)
