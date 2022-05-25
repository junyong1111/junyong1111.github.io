from icrawler.builtin import GoogleImageCrawler
import os

# Google에서 크롤링
# 이미지 저장 폴더 경로
save_dir = os.path.join('/')
# GoogleImageCrawler 객체 생성
filters = {
    'size': 'large',
    'license': 'noncommercial,modify', ## 비상업 , 수정가능 옵션 추가
    'color': 'blackandwhite'
    }

google_crawler = GoogleImageCrawler(storage={'root_dir': save_dir})
google_crawler.crawl(keyword='Dog',min_size = (200,200),
                     max_num=50, filters=filters)