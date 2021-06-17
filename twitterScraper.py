from selenium import webdriver
from selenium.webdriver.firefox.webdriver import WebDriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common import exceptions
from selenium.webdriver.common.keys import Keys
import time
import pandas

searchTerms = ['Friends', 'Ross', 'Chandler',
               'Joey', 'Monica', 'Phoebe', 'Rachel']

url = 'https://twitter.com/login'
driver = webdriver.Firefox()


def base_url(
    hashtag): return "https://twitter.com/search?q=lang%3Aen%20%23{}&src=typed_query&f=live".format(hashtag)


def getTweetData(card, term):

    # Username
    Username = card.find_element_by_xpath('.//span').text
    Username = Username.encode('ascii', 'ignore')
    print()
    Username = str(Username).replace("b'", "")
    Username = str(Username).replace("\\n", "")
    Username = Username[:-1]
    # print(Username)

    # Twitter Handle
    Handle = card.find_element_by_xpath('.//span[contains(text(), "@")]').text
    # print(Handle)

    # Postdate
    PostDate = card.find_element_by_xpath(
        './/time').get_attribute('datetime')
    print(PostDate)

    # Content
    Comment = card.find_element_by_xpath('.//div[2]/div[2]/div[1]').text
    Response = card.find_element_by_xpath('.//div[2]/div[2]/div[2]').text
    Content = Comment + Response
    Content = Content.encode('ascii', 'ignore')

    Content = str(Content).replace("b'", "")
    Content = str(Content).replace("b\"", "")
    Content = str(Content).replace("\\n", "")
    Content = Content[:-1]
    # print(Content)

# Reply count
    ReplyCount = card.find_element_by_xpath(
        './/div[@data-testid="reply"]').text
    print(ReplyCount)

# Retweet count
    RetweetCount = card.find_element_by_xpath(
        './/div[@data-testid="retweet"]').text
    print(RetweetCount)

# Likes count
    LikesCount = card.find_element_by_xpath('.//div[@data-testid="like"]').text
    print(LikesCount)

    tweet = (term, Username, Handle, PostDate, Content,
             ReplyCount, RetweetCount, LikesCount)
    return tweet


def scrapePage(localUrl, n, term, tweetData):
    driver.get(localUrl)
    driver.implicitly_wait(5)
    driver.find_element_by_link_text('Latest').click()
    data = []
    tweet_ids = set()
    last_pos = driver.execute_script("return window.pageYOffset;")
    scrolling = True
    scrollNo = 0

    while scrolling:
        cards = driver.find_elements_by_xpath('//div[@data-testid="tweet"]')
        for card in cards:
            data = getTweetData(card, term)
            if data:
                tweet_id = ''.join(data)
                if tweet_id not in tweet_ids:
                    tweet_ids.add(tweet_id)
                    tweetData.append(data)

        scrollAttempt = 0
        while True:
            # check scroll position
            driver.execute_script(
                'window.scrollTo(0, document.body.scrollHeight);')
            scrollNo += 1
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" + str(scrollNo))
            if scrollNo >= n:
                scrolling = False
                break
            time.sleep(3)
            curr_position = driver.execute_script("return window.pageYOffset;")
            if last_pos == curr_position:
                scrollAttempt += 1
                if scrollAttempt >= 3:
                    scrolling = False
                    break
                else:
                    time.sleep(3)
            else:
                last_pos = curr_position
                break
    return(tweetData)


def saveTweets(searchTerms, n):
    tweetList = []
    for term in searchTerms:
        print()
        print('Scraping for #{}'.format(term))
        tweetList = scrapePage(base_url(term), n, term, tweetList)
    df = pandas.DataFrame(tweetList)
    df.to_csv('friendsTweetData.csv', encoding='utf-8', index=False, header=[
        "Search Term", "Username", "Handle", "PostDate", "Content", "Comments", "Retweets", "Likes"])
    #old = pandas.read_csv('tweetData.csv', header=None)
    #old = old.append(df, ignore_index=True)
    #new = old.drop_duplicates()
    # new.to_csv('tweetData.csv',
    #           encoding='utf-8', index=False)


def waiting_function(by_variable, attribute):
    try:
        WebDriverWait(driver, 20).until(
            lambda x: x.find_element(by=by_variable, value=attribute))
    except (NoSuchElementException, TimeoutException):
        print(' {} {} not found'.format(by_variable, attribute))
        exit()


if __name__ == '__main__':
    driver.get(url)

    waiting_function('name', 'session[username_or_email]')
    email = driver.find_element_by_name('session[username_or_email]')
    password = driver.find_element_by_name('session[password]')
    email.send_keys('LaneTrance')
    password.send_keys('PivotalData', Keys.ENTER)

    while True:
        saveTweets(searchTerms, 5)
    driver.quit()
