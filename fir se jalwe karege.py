import urllib.request
import json
import csv
import time
import requests
from bs4 import BeautifulSoup
import re
data_infy = {}
data_tcs = {}
lsave=time.time()
def timecheck():
    from datetime import datetime
    import calendar

    currentDT = datetime.now()
    d=calendar.day_name[currentDT.weekday()]

    date_string1 = '9:30'
    b = currentDT.strftime("%H:%M")
    date_string3 = '15:30'
    format = '%H:%M'
    my_date1 = datetime.strptime(date_string1, format)
    my_date3 = datetime.strptime(date_string3, format)
    # This prints '2009-11-29 03:17 AM'
    a=my_date1.strftime(format)
    c=my_date3.strftime(format)
    if b>=a and b<=c and d!="Sunday" and d!="Saturday":
            return 'ok'
    else:
            return "Market Is Close"


def import_web(ticker):
    """
    :param ticker: Takes the company ticker
    :return: Returns the HTML of the page
    """
    url = 'https://www.nseindia.com/live_market/dynaContent/live_watch/get_quote/GetQuote.jsp?symbol='+ticker+'&illiquid=0&smeFlag=0&itpFlag=0'
    req = urllib.request.Request(url, headers={'User-Agent' : "Chrome Browser"}) 
    fp = urllib.request.urlopen(req, timeout=10)
    mybytes = fp.read()
    mystr = mybytes.decode("utf8")
    fp.close()
    return mystr


def get_quote(ticker):
    """
    :param ticker: Takes the company ticker
    :return: None
    """
    ticker = ticker.upper()
    try:
        """fetches a UTF-8-encoded web page, and  extract some text from the HTML"""
        string_html = import_web(ticker)
        get_data(filter_data(string_html),ticker)
    except Exception as e:
        print(e)

          
def filter_data(string_html):          
    searchString = '<div id="responseDiv" style="display:none">'
    #assign: stores html tag to find where data starts
    searchString2 = '</div>'
    #stores:  stores html tag where  data end
    sta = string_html.find(searchString)
    #print(searchString)
    # returns & store: find() method returns the lowest index of the substring (if found). If not found, it returns -1.
    data = string_html[sta + 43:]
    #returns & stores: skips 43 characters and stores the index of substring
    end = data.find(searchString2)
    # returns & store: find() method returns the lowest index of the substring (if found). If not found, it returns -1.
    fdata = data[:end]
    #fetch: stores the fetched data into fdata
    stripped = fdata.strip()
    #removes: blank spaces
    return stripped



def get_data(stripped, company):     
    js = json.loads(stripped)
    datajs = js['data'][0]
    subdictionary = {}
    subdictionary['1. open'] = datajs['open']
    subdictionary['2. high'] = datajs['dayHigh']
    subdictionary['3. low'] = datajs['dayLow']
    subdictionary['4. close'] = datajs['lastPrice']
    subdictionary['5. volume'] = datajs['totalTradedVolume']
    subdictionary['6. 52 Week High'] = datajs['high52']
    subdictionary['7. 52 Week Low'] = datajs['low52']
    subdictionary['8. Previous Close'] = datajs['previousClose']
    print (
        'Adding value at : ',
        js['lastUpdateTime'],
        ' to ',
        company,
        ' Open Price:',datajs["open"],' Day High:',datajs["dayHigh"],' Day Low:',datajs["dayLow"],
        ' Total Traded Volume:',datajs["totalTradedVolume"],' 52 Week High:',datajs["high52"],' 52 Week Low',datajs["low52"],
        ' Previous Close:',datajs["previousClose"],' Stock Price:',datajs["lastPrice"],
        )
    data_infy[js['lastUpdateTime']] = subdictionary
    a = datajs["open"]
    b = datajs["dayHigh"]
    c = datajs["dayLow"]
    d = datajs["totalTradedVolume"]
    e = datajs["high52"]
    f = datajs["low52"]
    g = datajs["previousClose"]
    h = datajs["lastPrice"]
    with open(company+'.csv', 'a')as csvfile:
        writer = csv.writer(csvfile,lineterminator='\n')
        writer.writerow([a,b,c,d,e,f,g,h])
        csvfile.close()

def company():
    url = requests.get("https://en.wikipedia.org/wiki/NIFTY_50").text
    soup = BeautifulSoup(url, 'lxml')
    mytable = soup.find('table',{'id':'constituents'})
    mytable=mytable.find_all('tr')
    b=[]
    list_rows = []
    for row in mytable:
        cells = row.find_all('td')
        str_cells = str(cells)
        clean = re.compile('<.*?>')
        clean2 = (re.sub(clean, '',str_cells))
        if clean2!="[]":
            p3=clean2.split(', ')
            list_rows.append(p3[1])
            
    for i in list_rows:
        i.replace(".", "/")
        i=i.rsplit('.NS')
        if i[0]=='M&amp;M':
            i="M%26M"
            b.append(i)
        else:
            b.append(i[0])
    return b

def main():
    t_list=company()
    try:
        while(True):
            ti=timecheck()
            if ti=='ok':
                for ticker in t_list:
                    print("Starting get_quote for ",ticker)
                    get_quote(ticker)
                print("Taking a nap! Good Night")
                time.sleep(60)
                print("\n\n")
            else:
                print(ti)
                time.sleep(60)
    except Exception as e:
        print(e)
main()
