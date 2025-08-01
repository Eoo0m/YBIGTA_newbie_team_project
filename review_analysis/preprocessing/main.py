import os
import glob
from argparse import ArgumentParser
from typing import Dict, Type
from review_analysis.preprocessing.base_processor import BaseDataProcessor
from review_analysis.preprocessing.rottentomatoes_processor import RottenTomatoesProcessor
from review_analysis.preprocessing.naver_processor import NaverProcessor
from review_analysis.preprocessing.letterboxd_processor import LetterboxdProcessor



# 모든 preprocessing 클래스를 예시 형식으로 적어주세요.
# key는 "reviews_사이트이름"으로, value는 해당 처리를 위한 클래스
PREPROCESS_CLASSES: Dict[str, Type[BaseDataProcessor]] = {
    "reviews_letterboxd": LetterboxdProcessor,
    "reviews_naver": NaverProcessor,
    "reviews_rottentomatoes": RottenTomatoesProcessor
    # key는 크롤링한 csv파일 이름으로 적어주세요! ex. reviews_naver.csv -> reviews_naver
}

REVIEW_COLLECTIONS = glob.glob(os.path.join("database", "reviews_*.csv"))


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=False,
        default="../../database",
        help="Output file dir. Example: ../../database",
    )
    parser.add_argument(
        "-c",
        "--preprocessor",
        type=str,
        required=False,
        choices=PREPROCESS_CLASSES.keys(),
        help=f"Which processor to use. Choices: {', '.join(PREPROCESS_CLASSES.keys())}",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Run all data preprocessors. Default to False.",
    )
    return parser


if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.all:
        print("[INFO] --all 옵션 실행 시작")
        for csv_file in REVIEW_COLLECTIONS:
            print(f"[INFO] 발견한 파일: {csv_file}")

            base_name = os.path.splitext(os.path.basename(csv_file))[0]
            print(f"[INFO] base_name = {base_name}")

            if base_name in PREPROCESS_CLASSES:
                print(f"[INFO] {base_name} 전처리 시작")
                preprocessor_class = PREPROCESS_CLASSES[base_name]
                preprocessor = preprocessor_class(csv_file, args.output_dir)
                preprocessor.preprocess()
                preprocessor.feature_engineering()
                preprocessor.save_to_database()
            else:
                print(f"[SKIP] {base_name} 클래스가 등록되지 않음")

    elif args.preprocessor:
            preprocessor = PREPROCESS_CLASSES[args.preprocessor](f'database/{args.preprocessor}.csv', args.output_dir)
            preprocessor.preprocess()
            preprocessor.feature_engineering()
            preprocessor.save_to_database()

