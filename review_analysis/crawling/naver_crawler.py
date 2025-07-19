import os
from dataclasses import dataclass
from typing import Optional
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.remote.webelement import WebElement
from selenium.common.exceptions import TimeoutException
from review_analysis.crawling.base_crawler import BaseCrawler
from utils.logger import setup_logger

logger = setup_logger()

@dataclass
class CrawledReview:
    rating: int
    date: str
    review: str


UNOFFICIAL_FETCH_SCRIPT = """
var callback = arguments[arguments.length - 1];
fetch('https://m.search.naver.com/p/csearch/content/nqapirender.nhn?where=nexearch&pkid=68&fileKey=movieKBPointAPI&u1='+arguments[0]+'&u5=true&u3=sympathyScore&u4=false&u2='+arguments[1]).then(res => res.json()).then(data => callback(data));
"""


class NaverCrawler(BaseCrawler):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.driver: Optional[webdriver.Chrome] = None
        self.reviews: list[CrawledReview] = []

    def start_browser(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        self.driver = webdriver.Chrome(options=options)
        logger.info("브라우저 시작 완료")

    def scrape_reviews(self):
        return self._scrape_reviews("기생충", "161967", 1000)

    def _scrape_reviews(self, movie_name: str, movie_id: str, limit: int = 100):
        self.start_browser()
        url = f"https://search.naver.com/search.naver?query=영화 {movie_name} 관람평"
        if not self.driver:
            raise RuntimeError("브라우저가 시작되지 않았습니다.")
        driver = self.driver
    
        driver.get(url)
        driver.implicitly_wait(10)
        logger.info("리뷰 수집 시작")

        def get_review_count() -> int:
            return driver.execute_script("return document.querySelectorAll('#review_container > li').length;")

        def fetch_unofficial_reviews(page: int):
            return driver.execute_async_script(UNOFFICIAL_FETCH_SCRIPT, movie_id, str(page))

        # Body에 element를 모아둘 div 생성
        driver.execute_script("""
            var div = document.createElement('div');
            div.id = 'review_container';
            document.body.appendChild(div);
        """)

        count = 0
        while get_review_count() < limit:
            count += 1
            reviews = fetch_unofficial_reviews(count)
            html = reviews.get('html', '')
            if not html:
                logger.warning("리뷰 HTML이 비어있습니다. 더 이상 리뷰가 없을 수 있습니다.")
                break
            driver.execute_script(f"document.getElementById('review_container').innerHTML += `{html}`;")
            logger.info(f"요청수: {count}, 수집된 리뷰 수: {get_review_count()}")

        def extract_review(element: WebElement) -> CrawledReview:
            rating = int(element.find_element(By.CSS_SELECTOR, "div.lego_movie_pure_star").text.strip().replace("별점(10점 만점 중)", ""))
            date = element.find_elements(By.CSS_SELECTOR, "dl.cm_upload_info > dd")[1].text.strip()
            review = element.find_element(By.CSS_SELECTOR, "div.area_review_content span._text").text
            return CrawledReview(rating=rating, date=date, review=review)

        review_elements = driver.find_elements(By.CSS_SELECTOR, "div#review_container > li")

        for element in review_elements:
            review = extract_review(element)
            self.reviews.append(review)

        logger.info(f"최종 수집 리뷰 수: {len(self.reviews)}")
        self.driver.quit()
        logger.info("브라우저 종료")
            
    def save_to_database(self):
        if not self.reviews:
            logger.warning("저장할 리뷰가 없습니다.")
            return
        df = pd.DataFrame(self.reviews)
        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, "reviews_naver.csv")
        df.to_csv(save_path, index=False)
        logger.info(f"CSV 저장 완료: {save_path}")