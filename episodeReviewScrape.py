from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

# First navigation to IMDB review page
url = 'https://www.imdb.com/title/tt0108778/episodes?season='
driver = webdriver.Firefox()  # Can be edited if using Chrome etc

# Used to derive the searching page while inputting the search terms as a hashtag


def getScriptText(season):
    urls = []
    reviews = []
    elementList = []
    counter = 0
    episodeElms = driver.find_elements_by_partial_link_text('The One')
    for episode in episodeElms:
        tempURL = (episode.get_attribute('href'))
        URLPieces = tempURL.split('/')
        urls.append(URLPieces[4])
    for url in urls:
        counter += 1
        driver.get(
            'https://www.imdb.com/title/{}/reviews?ref_=tt_urv'.format(url))
        elementList = driver.find_elements_by_xpath(
            "//div[@class='expander-icon-wrapper spoiler-warning__control']")
        for element in elementList:
            element.click()
        reviewText = driver.find_elements_by_xpath(
            "//div[@class='text show-more__control']")
        for review in reviewText:
            if review.text != '':
                if (counter <= 9):
                    if (season != 10):
                        reviews.append('/////' + 'Season '+str(season) +
                                       'A Episode '+str(counter) + 'A ' + review.text)
                    else:
                        reviews.append('/////' + 'Season '+str(season) +
                                       ' Episode '+str(counter) + 'A ' + review.text)
                else:
                    if (season != 10):
                        reviews.append('/////' + 'Season '+str(season) +
                                       'A Episode '+str(counter) + ' ' + review.text)
                    else:
                        reviews.append('/////' + 'Season '+str(season) +
                                       ' Episode '+str(counter) + ' ' + review.text)
    return reviews

    # Function for saving the tweets to a CSV. The commented out line is for use when an empty csv is being used/first set up


def saveScripts():
    textToSave = ''
    count = 0
    for season in range(1, 11):
        print('Scraping season...'+str(season))
        driver.get(url+str(season))
        reviews = getScriptText(season)
        for x in reviews:
            textToSave = textToSave + '\n' + x
    text_file = open("allEpisodeReviews.txt", "w", encoding='utf-8')
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
    saveScripts()
    driver.quit()
