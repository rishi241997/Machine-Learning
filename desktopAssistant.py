from gtts import gTTS
import speech_recognition as sr
import os
import re
import webbrowser
import smtplib
import requests
from weather import Weather,Unit
import time
import subprocess
import pyttsx3
from playsound import playsound
import random
def talkToMe(audio):
    "speaks audio passed as argument"

    #print(audio)
    #for line in audio.splitlines():
        #os.system("say " + audio)

    #  use the system's inbuilt say command instead of mpg123
    #  text_to_speech = gTTS(text=audio, lang='en')
    #  text_to_speech.save('audio.mp3')
    #  os.system('mpg123 audio.mp3')
    
    print(audio)
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    rate = engine.getProperty('rate')
    engine.setProperty('rate',rate-50)
    engine.setProperty('voice','HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0')
    engine.say(audio)
    engine.runAndWait()

    """print(audio)
    r1=random.randint(1,100)
    ranfile=str(r1)+".mp3"

    tts=gTTS(text=audio,lang='en-us')
    tts.save(ranfile)
    playsound(ranfile)
    os.remove(ranfile)"""

def myCommand():
    "listens for commands"

    r = sr.Recognizer()
    r.energy_threshold = 4000

    with sr.Microphone() as source:
        print('Ready...')
        r.pause_threshold = 0.8
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)

    try:
        command = r.recognize_google(audio).lower()
        print('You said: ' + command + '\n')

    #loop back to continue to listen for commands if unrecognizable speech is received
    except sr.UnknownValueError:
        print('Your last command couldn\'t be heard')
        command = myCommand();

    return command


def assistant(command):
    "if statements for executing commands"

    if 'open youtube' in command:
        url = 'https://www.youtube.com/' 
        webbrowser.open(url)
        talkToMe('opening youtube!')
        print('Done!')


    elif 'search on youtube' in command:
        reg_ex = re.search('search on youtube (.*)', command)
        if reg_ex:
            domain = reg_ex.group(1)
            url = 'https://www.youtube.com/results?search_query=' + domain
            webbrowser.open(url)
            talkToMe('ok !Searching on youtube!')
            print('Done!')
        else:
            pass

    elif 'open facebook' in command:
        url = 'https://www.facebook.com/'
        webbrowser.open(url)
        talkToMe('opening Facebook!')
        print('Done!')

    elif 'search on facebook' in command:
        reg_ex = re.search('search on facebook (.*)', command)
        if reg_ex:
            domain = reg_ex.group(1)
            url = 'https://www.facebook.com/search/top/?q=' + domain
            webbrowser.open(url)
            talkToMe('ok !Searching on facebook!')
            print('Done!')
        else:
            pass

    elif 'open google' in command:
        url = 'https://www.google.com/'
        webbrowser.open(url)
        talkToMe('opening Google!')
        print('Done!')

    elif 'search' in command:
        reg_ex = re.search('search (.*)', command)
        if reg_ex:
            domain = reg_ex.group(1)
            url = 'https://www.google.co.in/search?source=hp&ei=Nr_JWuWtMIrhvgThx6ngCw&q=' + domain
            webbrowser.open(url)
            talkToMe('ok !Searching on google!')
            print('Done!')
        else:
            pass

    elif 'open website' in command:
        reg_ex = re.search('open website (.+)', command)
        if reg_ex:
            domain = reg_ex.group(1)
            url = 'https://www.' + domain
            webbrowser.open(url+".com")
            print('Done!')
        else:
            pass

    elif 'what\'s up' in command:
        talkToMe('Just doing my thing!')

    elif 'how you doing' in command:
        talkToMe('i am doing great! thanks!')


#jokes api
    elif 'joke' in command:
        res = requests.get(
                'https://icanhazdadjoke.com/',
                headers={"Accept":"application/json"}
                )
        if res.status_code == requests.codes.ok:
            talkToMe('ok! here is a joke for You')
            talkToMe(str(res.json()['joke']))
        else:
            talkToMe('oops!I ran out of jokes')


    #quotes api
    elif 'quote' in command:
        res = requests.get("https://andruxnet-random-famous-quotes.p.mashape.com/?cat=famous&count=10",
        headers={
        "X-Mashape-Key": "OqUFMd95Dsmshi65ZDMf4BxnoEBHp1jDkCUjsnWmu9Yj0sg61y",
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"}
        )
        if res.status_code == requests.codes.ok:
            talkToMe('ok! here is a quote for You')
            talkToMe(str(res.json()['quote']))
            talkToMe(str(res.json()['author']))
        else:
            talkToMe('oops!I ran out of quotes')

    #weather api
    elif 'current weather in' in command:
        reg_ex = re.search('current weather in (.*)', command)
        if reg_ex:
            city = reg_ex.group(1)
            weather = Weather()
            location = weather.lookup_by_location(city)
            condition = location.condition
            talkToMe('Ok!')
            talkToMe('The Current weather in %s is %s ! The temperature is %s degree celcius !' % (city, condition.text,condition.temp))

    elif 'weather forecast in' in command:
        reg_ex = re.search('weather forecast in (.*)', command)
        if reg_ex:
            city = reg_ex.group(1)
            weather = Weather()
            location = weather.lookup_by_location(city)
            forecasts = location.forecast
            talkToMe('Ok!')
            for forecast in forecasts:
                talkToMe('On %s it will %s ! The maximum temperature will be %s degree celcius ! The lowest temperature will be %s degree celcius !' % (forecast.date, forecast.text,forecast.high,forecast.low))

    #Email api
    elif 'email' in command:
        talkToMe('May i know the recipient?')
        recipient = myCommand()
        if 'rishabh' in recipient:
            t=1
            mail1= 'jainr578@gmail.com'
            while t:
                talkToMe('ok! What do you want me to say?')
                content = myCommand()
                talkToMe(content+'  !is this right?')
                answer = myCommand()
                if 'yes' in answer:
                    #init gmail SMTP
                    mail = smtplib.SMTP('smtp.gmail.com', 587)

                    #identify to server
                    mail.ehlo()

                    #encrypt session
                    mail.starttls()

                    #login
                    mail.login('fifamobilegamender@gmail.com', 'youwanttohack')

                    #send message
                    mail.sendmail('rishabh',mail1, content)

                    #end mail connection
                    mail.close()

                    talkToMe('Email sent!')
                    t=0
                    
                else:
                    continue
                    
        elif '@gmail.com' in recipient:
            mail1=recipient
            mail1=mail1.replace(" ","")
            print(mail1)
            t=1
            while t:
                talkToMe('ok! What do you want me to say?')
                content = myCommand()
                talkToMe(content+'  !is this right?')
                answer = myCommand()
                if 'yes' in answer or 'right' in answer:
                    #init gmail SMTP
                    mail = smtplib.SMTP('smtp.gmail.com', 587)

                    #identify to server
                    mail.ehlo()

                    #encrypt session
                    mail.starttls()

                    #login
                    mail.login('fifamobilegamender@gmail.com', 'youwanttohack')

                    #send message
                    mail.sendmail('noone',mail1, content)

                    #end mail connection
                    mail.close()

                    talkToMe('Email sent!')
                    t=0
                    
                else:
                    continue
                    

        else:
            talkToMe('I don\'t know what you mean!')

    #GenreBasedMusic
    elif 'play' in command:
        path='C:\Python\musicplaylist'
        if 'dance' in command or 'fun' in command or 'party' in command or 'edm' in command:
            add=path+'\edm\edm.m3u'
            subprocess.Popen("explorer "+add)
            talkToMe('ok! playing')
            print("Done!")

        elif 'happy' in command or 'cheering' in command or 'hip hop' in command:
            add=path+'\happy\happy.m3u'
            subprocess.Popen("explorer "+add)
            talkToMe('ok! playing')
            print("Done!")

        elif 'romantic' in command or 'sad' in command or 'bollywood' in command:
            add=path+"\\romantic\\romantic.m3u"
            subprocess.Popen("explorer "+add)
            talkToMe('ok! playing')
            print("Done!")

        elif 'devotional' in command or 'bhakti' in command or 'religious' in command:
            add=path+'\devotional\devotional.m3u'
            subprocess.Popen("explorer "+add)
            talkToMe('ok! Playing')
            print("Done!")

        elif 'relaxing' in command or 'chilling' in command or 'chillin' in command or 'cool' in command or 'uplifting' in command:
            add=path+"\\relaxing\\relaxing.m3u"
            subprocess.Popen("explorer "+add)
            talkToMe('ok! playing')
            print("Done!")

        elif 'music' in command or 'anything' in command or 'pop' in command or 'any kind' in command:
            add=path+'\general\general.m3u'
            subprocess.Popen("explorer "+add)
            talkToMe('ok! playing')
            print("Done!")

        elif 'rock' in command or 'metal' in command or 'rocking' in command:
            add=path+"\\rock\\rock.m3u"
            subprocess.Popen("explorer "+add)
            talkToMe('ok! playing')
            print("Done!")

        elif 'create' in command:
            talkToMe("OK! creating new playlists")
            os.system("musicplaylistcreator.py")
            talkToMe("playlists created!")

        else:
            talkToMe("Sorry this is not a valid choice")

   
        
        


talkToMe('I am ready for your command')

#loop to continue executing multiple commands
while True:
    assistant(myCommand())
