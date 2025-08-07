import os
import csv
from typing import Optional
from app.review.review_repository import ReviewRepository
from app.review.review_schema import *
from review_analysis.preprocessing.base_processor import BaseDataProcessor
from review_analysis.preprocessing.letterboxd_processor import LetterboxdProcessor
from review_analysis.preprocessing.naver_processor import NaverProcessor
from review_analysis.preprocessing.rottentomatoes_processor import RottenTomatoesProcessor


TMP_PREPROCESSED_DIR = "tmp/preprocessed"


class ReviewService:
    def __init__(self, review_repository: ReviewRepository) -> None:
        self.repo = review_repository
        self.preprocessors: dict[SiteName, BaseDataProcessor] = {
            SiteName.LETTERBOXD: LetterboxdProcessor(input_path=os.path.join(TMP_PREPROCESSED_DIR, 'reviews_letterboxd.csv'), output_path=TMP_PREPROCESSED_DIR),
            SiteName.NAVER: NaverProcessor(input_path=os.path.join(TMP_PREPROCESSED_DIR, 'reviews_naver.csv'), output_path=TMP_PREPROCESSED_DIR),
            SiteName.ROTTENTOMATOES: RottenTomatoesProcessor(input_path=os.path.join(TMP_PREPROCESSED_DIR, 'reviews_rottentomatoes.csv'), output_path=TMP_PREPROCESSED_DIR)
        }
        os.makedirs(TMP_PREPROCESSED_DIR, exist_ok=True)

    def __del__(self):
        '''
        소멸자에서 TMP_PREPROCESSED_DIR 폴더를 삭제합니다.
        '''
        if os.path.exists(TMP_PREPROCESSED_DIR):
            for file in os.listdir(TMP_PREPROCESSED_DIR):
                file_path = os.path.join(TMP_PREPROCESSED_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(TMP_PREPROCESSED_DIR)

    def preprocess_reviews(self, site_name: SiteName) -> str:
        '''
        주어진 사이트 이름에 해당하는 리뷰들을 전처리하고, 전처리된 리뷰들을 데이터베이스에 저장합니다.
        :param site_name: SiteName 열거형
        :return: 전처리된 리뷰들의 ID
        '''
        reviews = self.repo.get_reviews_by_site_name(site_name)
        parser = self.preprocessors[site_name]

        reviews = self.repo.get_reviews_by_site_name(site_name)
        if not reviews:
            raise ValueError("Reviews not Found.")
        
        # CSV 파일로 저장하고 전처리
        ReviewService.save_reviews_to_csv(reviews, site_name)
        parser.preprocess()
        parser.feature_engineering()
        parser.save_to_database()
        preprocessed_reviews = ReviewService.get_preprocessed_reviews_from_csv(site_name)
        preprocessed_id = self.repo.save_preprocessed_reviews(site_name, preprocessed_reviews)
        if not preprocessed_reviews:
            raise ValueError("Preprocessed reviews not found.")
        return preprocessed_id
    
    def get_preprocessed_reviews_by_id(self, id: str) -> Optional[list[PreprocessedReview]]:
        '''
        주어진 ID에 해당하는 전처리된 리뷰들을 데이터베이스에서 조회합니다.
        :param id: 전처리된 리뷰의 ID
        :return: 전처리된 리뷰들의 리스트
        '''
        preprocessed_reviews = self.repo.get_preprocessed_reviews_by_id(id)
        if not preprocessed_reviews:
            raise ValueError("Preprocessed reviews not found.")
        return preprocessed_reviews

    @staticmethod
    def save_reviews_to_csv(reviews: list[Review], site_name: SiteName):
        file_path = os.path.join(TMP_PREPROCESSED_DIR, f'reviews_{site_name.value}.csv')
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['rating', 'date', 'review']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for review in reviews:
                writer.writerow({
                    'rating': review.rating,
                    'date': review.date,
                    'review': review.review
                })
    
    @staticmethod
    def get_preprocessed_reviews_from_csv(site_name: SiteName) -> Optional[list[PreprocessedReview]]:
        file_path = os.path.join(TMP_PREPROCESSED_DIR, f'preprocessed_reviews_{site_name.value}.csv')
        if not os.path.exists(file_path):
            return None
        
        reviews = []
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                reviews.append(PreprocessedReview(**row))
        
        return reviews

    