from fastapi import APIRouter, HTTPException, Depends, status
from pydantic_core import ValidationError
from app.review.review_schema import PreprocessedReview, SiteName, PreprocessReviewResponse
from app.review.review_service import ReviewService
from app.dependencies import get_review_service
from app.responses.base_response import BaseResponse

review = APIRouter(prefix="/review")


@review.post("/preprocess/{site_name}", response_model=BaseResponse[PreprocessReviewResponse], status_code=status.HTTP_200_OK)
def preprocess_reviews(site_name: str, service: ReviewService = Depends(get_review_service)) -> BaseResponse[PreprocessReviewResponse]:
    try:
        site_name = SiteName(site_name.lower())
        preprocess_reviews_id = service.preprocess_reviews(site_name)
        response = PreprocessReviewResponse(id=preprocess_reviews_id, site_name=site_name)
        return BaseResponse(status="success", data=response, message="Preprocessing Success.")
    except ValidationError as exc:
        print(repr(exc.errors()))
        raise HTTPException(status_code=422, detail="Validation Error")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@review.get("/preprocessed/{id}", response_model=BaseResponse[list[PreprocessedReview]], status_code=status.HTTP_200_OK)
def get_preprocessed_reviews(id: str, service: ReviewService = Depends(get_review_service)) -> BaseResponse[list[PreprocessedReview]]:
    try:
        preprocessed_reviews = service.get_preprocessed_reviews_by_id(id)
        if not preprocessed_reviews:
            raise HTTPException(status_code=404, detail="Preprocessed reviews not found")
        return BaseResponse(status="success", data=preprocessed_reviews, message="Retrieved preprocessed reviews successfully.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")
