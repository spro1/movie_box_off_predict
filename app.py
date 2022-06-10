from flask import Flask, render_template, request, redirect, url_for

# python library
import math
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor

#import xgboost
#from xgboost import XGBRegressor

from sklearn.metrics.pairwise import linear_kernel
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
import joblib

# model load
y1 = joblib.load("static/model/y1.pkl")
y2 = joblib.load("static/model/y2.pkl")
y3 = joblib.load("static/model/y3.pkl")
y4 = joblib.load("static/model/y4.pkl")
y5 = joblib.load("static/model/y5.pkl")
y6 = joblib.load("static/model/y6.pkl")

z1 = joblib.load("static/model/z1.pkl")
z2 = joblib.load("static/model/z2.pkl")
z3 = joblib.load("static/model/z3.pkl")
z4 = joblib.load("static/model/z4.pkl")
z5 = joblib.load("static/model/z5.pkl")
z6 = joblib.load("static/model/z6.pkl")

rf = joblib.load("static/model/random forest.pkl")
gb = joblib.load("static/model/gradient boosting.pkl")


app = Flask(__name__)


@app.route('/', methods=["GET"])
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/result', methods=["GET", "POST"])
def result():
    return render_template('result.html')


@app.errorhandler(404)
def page_not_found(error):
    return redirect(url_for('index'))


@app.errorhandler(500)
def internal_server_error(error):
    return redirect(url_for('index'))


@app.route('/loading', methods=["GET", "POST"])
def loading():
    data = request.form.to_dict()
    return render_template('loading.html', data=data)


@app.route('/pred', methods=["GET", "POST"])
def new_movie():
    if request.method == "POST":
        title = request.form.get("title")
        category = request.form.get('category')
        year = request.form.get("year")
        month = request.form.get("month")
        staff = request.form.get("staff")
        story = request.form.get("story")

        new_movie_data, similar_movie = new_movie_similar(title, story, category, staff, year, month)

        new_movie_data = pd.DataFrame([new_movie_data])
        new_movie_data['category'] = new_movie_data.category.map({'SF': 1, '판타지': 2, '어드벤처': 3, '액션': 4, '사극': 5, '전쟁': 6,
                                                                  '드라마': 7, '범죄': 8, '애니메이션': 9, '코미디': 10, '미스터리': 11,
                                                                  '로맨스': 12, '다큐멘터리': 13, '스릴러': 14, '공포': 15, '공연': 16})

        y1_value = int(y1.predict(new_movie_data)[0])
        y2_value = int(y2.predict(new_movie_data)[0])
        y3_value = int(y3.predict(new_movie_data)[0])
        y4_value = int(y4.predict(new_movie_data)[0])
        y5_value = int(y5.predict(new_movie_data)[0])
        y6_value = int(y6.predict(new_movie_data)[0])

        new_movie_data["2주차_매출"] = y1_value
        new_movie_data["2주차_관객수"] = y2_value
        new_movie_data["2주차_스크린수"] = y3_value
        new_movie_data["2주차_상영횟수"] = y4_value
        new_movie_data["2주차_누적매출액"] = y5_value
        new_movie_data["2주차_누적관객수"] = y6_value

        z1_value = int(z1.predict(new_movie_data)[0])
        z2_value = int(z2.predict(new_movie_data)[0])
        z3_value = int(z3.predict(new_movie_data)[0])
        z4_value = int(z4.predict(new_movie_data)[0])
        z5_value = int(z5.predict(new_movie_data)[0])
        z6_value = int(z6.predict(new_movie_data)[0])

        new_movie_data["3주차_매출"] = z1_value
        new_movie_data["3주차_관객수"] = z2_value
        new_movie_data["3주차_스크린수"] = z3_value
        new_movie_data["3주차_상영횟수"] = z4_value
        new_movie_data["3주차_누적매출액"] = z5_value
        new_movie_data["3주차_누적관객수"] = z6_value

        #xgb_value = int(xgb.predict(new_movie_data)[0])
        rf_value = int(rf.predict(new_movie_data)[0])
        gb_value = int(gb.predict(new_movie_data)[0])
        #voting_value = int(vt.predict(new_movie_data)[0])

        #new_movie_data["box_off_num_xgb"] = xgb_value
        new_movie_data["box_off_num_rf"] = rf_value
        new_movie_data["box_off_num_gb"] = gb_value
        #new_movie_data["box_off_num_vt"] = voting_value

        #print(new_movie_data, similar_movie)

    return render_template('result.html', title=title, story=story, category=category,
                           movie_data=new_movie_data, similar_data=similar_movie)



# 예측 모델
STOP_WORDS = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
okt = Okt()


def morph_and_stopword(s):
    token_ls = []
    # 형태소 분석
    tmp = okt.morphs(s, stem=True)

    # 불용어 처리
    for token in tmp:
        if token not in STOP_WORDS:
            token_ls.append(token)
    return token_ls


def new_movie_similar(title, story, category, staff, year, month):
    new_movie_data = {}
    similar_movie = []
    # 데이터 셋 불러오기
    datasetAll = pd.read_csv('static/data/movie_list.csv', encoding='CP949')
    copy_datasetAll = datasetAll.copy()
    copy_datasetAll = copy_datasetAll[:498]
    data = pd.json_normalize({"movie_name": title, "story": story, "category": category})
    copy_datasetAll = pd.concat([copy_datasetAll, data], ignore_index=True)

    for i, l in zip(copy_datasetAll.index, copy_datasetAll["story"]):
        l = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z ]', '', l)
        l = morph_and_stopword(l)
        l = " ".join(l)

        copy_datasetAll.at[i, 'story'] = l

    x_data = (copy_datasetAll["movie_name"]) + " " + copy_datasetAll["story"] + " " + (copy_datasetAll["category"])

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(x_data)

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(copy_datasetAll.index, index=copy_datasetAll['movie_name']).drop_duplicates()

    choice = []
    # 선택한 영화의 타이틀로부터 해당되는 인덱스를 받아옵니다. 이제 선택한 영화를 가지고 연산할 수 있습니다.
    idx = indices[title]

    # 모든 영화에 대해서 해당 영화와의 유사도를 구합니다.
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 유사도에 따라 영화들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 10개의 영화를 받아옵니다.
    sim_scores = sim_scores[1:11]

    # print(sim_scores)
    movie_indices = [i[0] for i in sim_scores]

    week_price = []
    week_viewer = []
    week_screen = []
    week_playnum = []
    week_total_price = []
    week_total_viewer = []
    naver_reporter = []
    naver_netizen = []
    daum_viewer = []

    for i in range(10):
        choice.append(copy_datasetAll['movie_name'][movie_indices[i]])
        week_price.append(copy_datasetAll['1주차_매출'][movie_indices[i]])
        week_viewer.append(copy_datasetAll['1주차_관객수'][movie_indices[i]])
        week_screen.append(copy_datasetAll['1주차_스크린수'][movie_indices[i]])
        week_playnum.append(copy_datasetAll['1주차_상영횟수'][movie_indices[i]])
        week_total_price.append(copy_datasetAll['1주차_누적매출액'][movie_indices[i]])
        week_total_viewer.append(copy_datasetAll['1주차_누적관객수'][movie_indices[i]])
        naver_reporter.append(copy_datasetAll['naver_reporter'][movie_indices[i]])
        naver_netizen.append(copy_datasetAll['naver_netizen'][movie_indices[i]])
        daum_viewer.append(copy_datasetAll['daum_viewer'][movie_indices[i]])

    week_price = round(np.mean(week_price))
    week_viewer = round(np.mean(week_viewer))
    week_screen = round(np.mean(week_screen))
    week_playnum = round(np.mean(week_playnum))
    week_total_price = round(np.mean(week_total_price))
    week_total_viewer = round(np.mean(week_total_viewer))
    naver_reporter = np.mean(naver_reporter)
    naver_netizen = np.mean(naver_netizen)
    daum_viewer = np.mean(daum_viewer)

    for i in range(10):
        similar_movie.append(choice[i])

    new_movie_data["년"] = year
    new_movie_data["월"] = month
    new_movie_data["naver_reporter"] = naver_reporter
    new_movie_data["naver_netizen"] = naver_netizen
    new_movie_data["daum_viewer"] = daum_viewer
    new_movie_data["staff"] = staff
    new_movie_data["category"] = category
    new_movie_data["1주차_매출"] = week_price
    new_movie_data["1주차_관객수"] = week_viewer
    new_movie_data["1주차_스크린수"] = week_screen
    new_movie_data["1주차_상영횟수"] = week_playnum
    new_movie_data["1주차_누적매출액"] = week_total_price
    new_movie_data["1주차_누적관객수"] = week_total_viewer

    return new_movie_data, similar_movie


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
