from selenium import webdriver
from selenium.webdriver.firefox.webdriver import WebDriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common import exceptions
from selenium.webdriver.common.keys import Keys
import time
import csv
import pandas

url = 'https://fangj.github.io/friends/'  # First navigation to Twitter
driver = webdriver.Firefox()  # Can be edited if using Chrome etc

# Used to derive the searching page while inputting the search terms as a hashtag


def base_url(epNum):
    return "https://fangj.github.io/friends/season/{}.html".format(epNum[0])

# Function for trawling through the HTML of the page and collecting the correct elements


def getScriptText():
    scriptText = driver.find_element_by_tag_name("body").text
    #scriptText = scriptText.encode('utf-8')
    print(scriptText)
    return scriptText

    # Function for saving the tweets to a CSV. The commented out line is for use when an empty csv is being used/first set up


def saveScripts():
    textToSave = ""
    with open('epNums.txt', newline='') as f:
        reader = csv.reader(f)
        epNumbers = list(reader)
    for epNum in epNumbers:
        print()
        print('Scraping script{}'.format(epNum))
        driver.get(base_url(epNum))
        scriptText = getScriptText()
     #   scriptText = scriptText.replace(",", "\",\"")

        textToSave = textToSave + scriptText
    text_file = open("allEpisodesText.txt", "w", encoding='utf-8')
    text_file.write(textToSave)
    text_file.close()


# Function used to wait while the web pages are allowed to load
def waiting_function(by_variable, attribute):
    try:
        WebDriverWait(driver, 20).until(
            lambda x: x.find_element(by=by_variable, value=attribute))
    except (NoSuchElementException, TimeoutException):
        print(' {} {} not found'.format(by_variable, attribute))
        exit()


# Main function - logs into Twitter and then continues to scrape tweets indefinitely
if __name__ == '__main__':
    driver.get(url)
    saveScripts()
    driver.quit()
