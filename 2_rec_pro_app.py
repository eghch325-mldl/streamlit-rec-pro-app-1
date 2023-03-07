import pickle
import pandas as pd
import streamlit as st
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pickle.load(open('df_list.pickle', 'rb'))
master_pro = pickle.load(open('master_pro.pickle', 'rb'))

# scores 평균으로 나누기 함수 : 1번 파일에서 가공 후 dump 했으므로 pass
# def avg_scores(abc):
#     return len(abc) // 7

# lever = 1

# df['scores'] = df['scores'] / df['products'].apply(avg_scores) * lever

# st.set_page_config(layout='wide')
st.header('상품 추천')

arg1 = str(st.text_input('마스터코드 입력 1', ''))
arg2 = str(st.text_input('마스터코드 입력 2', ''))
arg3 = str(st.text_input('마스터코드 입력 3', ''))
arg4 = str(st.text_input('마스터코드 입력 4', ''))
arg5 = str(st.text_input('마스터코드 입력 5', ''))

def get_cosine(arg1=arg1, arg2=arg2, arg3=arg3, arg4=arg4, arg5=arg5):
    try:
        title_1 = str(arg1) + ' ' + str(arg2) + ' ' + str(arg3) + ' ' + str(arg4) + ' ' + str(arg5)
        title_1 = title_1.rstrip()
        title = title_1.split(' ')

        # 조회값 df에 마지막 행으로 추가
        df.loc[len(df)] = [title_1, 0]

        count = CountVectorizer()
        count_matrix = count.fit_transform(df['products'])

        cosine_sim_1 = cosine_similarity(count_matrix, count_matrix)

        cosine_sim_2 = np.array(df['scores'])
        cosine_sim_2 = np.reshape(cosine_sim_2, (-1,1))
        cosine_sim_2 = np.dot(cosine_sim_2, cosine_sim_2.T)

        cosine_sim = cosine_sim_1 + cosine_sim_2

        # 중복값 조회시 천 번째로 선택
        idx = df[df['products'] == title_1].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))

        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        temp = []
        for i in range(len(df)):
            sim = sim_scores[i][0]
            abc = df.iloc[sim,:]['products']
            abc = abc.split(' ')
            for j in abc:
                if str(j) not in title and str(j) not in temp:
                    temp.append(j)
        return temp[:5], title # 결과값 / 조회값 반환
    except:
        return [], []

rec_name = ['','','','','']

# 버튼을 클릭하면 다음 동작 실행
if st.button('Recommend'):
    rec_result = get_cosine()

    rec_code = rec_result[0]
    for i in range(len(rec_code)):
        try:
            rec_name[i] = master_pro[master_pro['code'] == rec_code[i]]['name'].values
        except:
            rec_name[i] = ''

    # 조회 코드 / 상품명 조회
    st.write('조회')
    search_code = rec_result[1]
    search_name = ['','','','','']
    for i in range(len(search_code)):
        try:
            search_name[i] = master_pro[master_pro['code'] == search_code[i]]['name'].values
        except:
            search_name[i] = ''
    st.dataframe(zip(search_code, search_name), width=1000)

    # 추천 목록 코드 / 상품명
    st.write('결과')
    st.dataframe(zip(rec_code, rec_name), width=1000)
