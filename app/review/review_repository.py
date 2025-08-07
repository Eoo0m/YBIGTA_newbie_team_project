import json

from typing import Optional

from app.review.review_schema import Review, SiteName, PreprocessedReview
from app.config import USER_DATA
from database.mongodb_connection import mongo_db
from bson.objectid import ObjectId

class ReviewRepository:
    def __init__(self) -> None:
        pass

    def get_reviews_by_site_name(self, site_name: SiteName) -> Optional[list[Review]]:
        reviews = mongo_db['reviews_' + site_name.value].find()
        return [Review(**review) for review in reviews] if reviews else None
    
    def save_preprocessed_reviews(self, site_name: SiteName, reviews: list[PreprocessedReview]) -> str:
        collection_name = 'preprocessed_reviews'
        if collection_name not in mongo_db.list_collection_names():
            mongo_db.create_collection(collection_name)
        preprocessed_reviews = mongo_db[collection_name]
        result = preprocessed_reviews.insert_one({
            "site_name": site_name.value,
            "data": [review.model_dump() for review in reviews]
        })
        return str(result.inserted_id)

    def get_preprocessed_reviews_by_id(self, id: str) -> Optional[list[PreprocessedReview]]:
        collection_name = 'preprocessed_reviews'
        preprocessed_reviews = mongo_db[collection_name]
        result = preprocessed_reviews.find_one({"_id": ObjectId(id)})
        return [PreprocessedReview(**review) for review in result['data']] if result else None