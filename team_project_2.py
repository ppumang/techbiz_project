# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:03:32 2022

version : 1.0

@author: tmlab
"""

# %% 1. 데이터 로드

if __name__ == '__main__':

    import os
    import sys
    import pandas as pd
    import numpy as np
    import warnings

    warnings.filterwarnings("ignore")

    directory = os.path.dirname(os.path.abspath(__file__))
    directory = directory.replace("\\", "/")  # window
    sys.path.append(directory+'/submodule')

    # load data
    # output_directory = directory + '/output_1/'
    # output_directory = directory + '/output_1_3years/'
    # output_directory = directory + '/output_2/'
    output_directory = directory + '/output_2_3years/'
    directory += '/input/'
    file_name = 'search_result_2_3years'
    # file_name = 'search_result_2'
    data = pd.read_excel(directory + file_name + '.xls', skiprows=7)
    LDA_RECENT_PERIOD = 3

    data['file_name'] = file_name

    # 데이터 전처리
    from preprocess import kipris_prep

    data_ = kipris_prep(data)

    # %% 2-1. EDA-출원연도별 동향

    from copy import copy
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False

    # data_input = data_
    data_input = copy(data_)
    # data_input = copy(data_.loc[data_['year_application']>= 2000, :].reset_index(drop= 1))

    year_counts = data_input['year_application'].value_counts().sort_index()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=year_counts.index, y=year_counts.values,
                 marker='o', color='blue', label='count')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.title('Annual Frequency')
    # plt.xticks(rotation=45)  # Rotate x-axis labels
    plt.grid(axis='y')
    plt.tight_layout()  # Ensure everything fits nicely

    # Annotate each point with its frequency
    for i, freq in enumerate(year_counts.values):
        # Adjusting y-offset for better visibility
        plt.text(year_counts.index[i], freq + 5, str(freq), ha='center')

    plt.show()
    # plt.savefig(output_directory+'출원연도별동향.png',
    #             dpi=1000, bbox_inches='tight',)

    print('출원연도별 동향 끝')
    # %% 2-2. EDA-대표 출원인

    from collections import Counter
    c = Counter(data_input['applicant_rep'])
    # c = c.most_common(20)

    sns.set_context("paper", font_scale=1.5)
    # Filter the Counter object
    threshold = 5
    c = {key: val for key, val in c.items() if val >= threshold}
    c = {key: val for key, val in c.items() if str(key) != 'nan'}
    c = {key: val for key, val in c.items() if str(key) != 'unassigned'}
    c = dict(sorted(c.items(), key=lambda item: item[1], reverse=1))
    # Extract data from the Counter object
    labels = list(c.keys())
    frequencies = list(c.values())

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=frequencies, y=labels, palette="viridis")

    # Add titles and labels
    plt.title("Frequency of Items")
    plt.xlabel("Item")
    plt.ylabel("Frequency")

    # Add frequencies on top of each bar
    for i, v in enumerate(frequencies):
        ax.text(v + 0.1, i, str(v), va='center', fontsize=10, color='black')

    # Display the plot
    plt.tight_layout()
    plt.xticks(rotation=90)
    plt.show()
    # plt.savefig(output_directory+'대표출원인.png',
    #             dpi=1000, bbox_inches='tight',)

    print('대표 출원인 끝')

    # %% 2-3. EDA-주요 협력 대상 리스트

    from collections import Counter

    flat_applicant = [item for sublist in data_input['applicants']
                      for item in sublist]
    flat_applicant = [i for i in flat_applicant if 'amore' not in i]

    c = Counter(flat_applicant)
    # c = c.most_common(20)

    sns.set_context("paper", font_scale=1.5)
    # Filter the Counter object
    threshold = 1
    c = {key: val for key, val in c.items() if val >= threshold}
    c = {key: val for key, val in c.items() if str(key) != 'nan'}
    c = {key: val for key, val in c.items() if str(key) != 'unassigned'}
    c = dict(sorted(c.items(), key=lambda item: item[1], reverse=1))
    # Extract data from the Counter object
    labels = list(c.keys())
    frequencies = list(c.values())

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=frequencies, y=labels, palette="viridis")

    # Add titles and labels
    plt.title("Frequency of Items")
    plt.xlabel("Item")
    plt.ylabel("Frequency")

    # Add frequencies on top of each bar
    for i, v in enumerate(frequencies):
        ax.text(v + 0.1, i, str(v), va='center', fontsize=10, color='black')

    # Display the plot
    plt.tight_layout()
    plt.xticks(rotation=90)
    plt.show()
    # plt.savefig(output_directory+'협력대상.png',
    #             dpi=1000, bbox_inches='tight',)

    print('협력대상 끝')

    # %% 2-4. EDA-주요 CPC 리스트
    from collections import Counter
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False

    # data_input = copy(data_.loc[data_['year_application']>= 1990, :].reset_index(drop= 1))
    c = Counter(x for xs in data_input['CPC_mc'] for x in set(xs))
    c = c.most_common(10)

    plt.bar(*zip(*c), color='black')
    plt.show()

    c = Counter(x for xs in data_input['CPC_sc'] for x in set(xs))
    c = c.most_common(10)

    plt.bar(*zip(*c), color='dimgray')
    plt.show()

    c = Counter(x for xs in data_input['CPC_mg'] for x in set(xs))
    c = c.most_common(10)

    plt.bar(*zip(*c), color='darkgray')
    plt.xticks(rotation=45)
    plt.show()

    c = Counter(x for xs in data_input['CPC_sg'] for x in set(xs))
    c = c.most_common(10)

    plt.bar(*zip(*c), color='silver')
    plt.xticks(rotation=45)
    plt.show()
    # plt.savefig(output_directory+'CPC.png',
    #             dpi=1000, bbox_inches='tight',)

    print('cpc 끝')

    # CPC 성장률

    cpc_sc_list_total = data_['CPC_sc'].tolist()
    cpc_mg_list_total = data_['CPC_mg'].tolist()
    cpc_sg_list_total = data_['CPC_sg'].tolist()

    def get_cpc_impact(cpc_list, data):

        cpc_set = set(sum(cpc_list, []))
        cpc_impact_dict = {}

        for cpc in cpc_set:

            index_list = [idx for idx, val in enumerate(
                cpc_list) if cpc in val]
            temp_data = data.loc[index_list, :]
            cpc_impact_dict[cpc] = np.mean(temp_data['자국피인용횟수'])

        return (cpc_impact_dict)

    def CAGR(first, last, periods):
        first = first+1
        last = last+1
        return (last/first)**(1/periods)-1

    def get_cpc_CAGR(cpc_list, data):

        cpc_set = set(sum(cpc_list, []))
        cpc_CAGR_dict = {}

        for cpc in cpc_set:

            index_list = [idx for idx, val in enumerate(
                cpc_list) if cpc in val]
            temp_data = data.loc[index_list, :].reset_index(drop=1)
            temp_data_2000 = temp_data.loc[temp_data['year_application'] == 2008, :].reset_index(
                drop=1)
            temp_data_2019 = temp_data.loc[temp_data['year_application'] == 2016, :].reset_index(
                drop=1)

            cpc_CAGR_dict[cpc] = CAGR(
                len(temp_data_2000), len(temp_data_2019), 8)

        return (cpc_CAGR_dict)

    cpc_sc_CAGR = get_cpc_CAGR(cpc_sc_list_total, data_)
    cpc_mg_CAGR = get_cpc_CAGR(cpc_mg_list_total, data_)
    cpc_sg_CAGR = get_cpc_CAGR(cpc_sg_list_total, data_)

    c = cpc_sc_CAGR
    c = dict(sorted(c.items(), key=lambda item: item[1], reverse=1))
    c = list(c.items())
    c = c[:10]

    print(c)

    plt.plot(*zip(*c), color='dodgerblue', marker='s')
    plt.xticks(rotation=45)
    plt.show()
    # plt.savefig(output_directory+'CPC_sc_CAGR.png',
    #             dpi=1000, bbox_inches='tight',)

    c = cpc_mg_CAGR
    c = dict(sorted(c.items(), key=lambda item: item[1], reverse=1))
    c = list(c.items())
    c = c[:10]

    plt.plot(*zip(*c), color='blue', marker='s')
    plt.xticks(rotation=45)
    plt.show()
    # plt.savefig(output_directory+'CPC_mg_CAGR.png',
    #             dpi=1000, bbox_inches='tight',)

    c = cpc_sg_CAGR
    c = dict(sorted(c.items(), key=lambda item: item[1], reverse=1))
    c = list(c.items())
    c = c[:10]

    plt.plot(*zip(*c), color='indigo', marker='s')
    plt.xticks(rotation=45)
    plt.show()
    # plt.savefig(output_directory+'CPC성장률.png',
    #             dpi=1000, bbox_inches='tight',)

    print('cpc 성장률 끝')
    # %% 3.텍스트 분석 준비

    import spacy
    import textMining
    import re
    import gensim

    corpus = data_input['TAF']
    # python -m spacy download en_core_web_sm

    # p1) removing speical character(optional)
    corpus = textMining.removing_sc(corpus)

    # p2) change type 2 nlp
    nlp = spacy.load("en_core_web_sm")

    data_['TAF_nlp'] = [nlp(i) for i in corpus]

    print('텍스트 분석 준비 끝')

    # %% 4. LDA 적합

    import textMining

    # LDA 파라미터 튜닝
    word_lists = textMining.get_word_list(data_['TAF_nlp'])
    LDA_0 = textMining.LDA_gensim(word_lists)

    tunning_df = LDA_0.tunning_passes(['perplexity', 'coherence', 'diversity'])
    tunning_df = LDA_0.tunning_ab(['perplexity', 'diversity', 'coherence'])
    tunning_df = LDA_0.tunning_k(
        method=['perplexity', 'diversity', 'coherence'])

    # LDA 파라미터 수동 수정
    # LDA_0.k = 200
    # LDA_0.alpha = 0.01
    # LDA_0.refresh_model()

    # LDA 결과 저장
    docTopic_matrix = LDA_0.get_docByTopics()
    wordTopic_matrix = LDA_0.get_wordByTopics()
    topwordTopic_matrix = LDA_0.get_topwordByTopics()
    topic_prop = LDA_0.get_topicProportion()
    title = LDA_0.get_most_similar_doc2topic(
        data_input, title='title', date='year_application')
    topic_time_df = LDA_0.get_TVByTime(data_input, 'year_application')
    print('get_TVByTime DONE')
    topic_time_df_summary = LDA_0.get_summary_TVByTime(LDA_RECENT_PERIOD)
    print('get_summary_TVByTime DONE')

    LDA_0.save(output_directory, 'test.xlsx', data=data_input,
               recent_period=LDA_RECENT_PERIOD)

    import pyLDAvis.gensim_models
    import pyLDAvis

    print('LDA 적합 끝')

    # LDAVis
    vis_data = pyLDAvis.gensim_models.prepare(LDA_0.model_final,
                                              LDA_0.corpus,
                                              LDA_0.dictionary
                                              )

    pyLDAvis.save_html(vis_data, output_directory + 'ldavis_test.html')

    print('LDAVis 끝')

    # %% 5. LDA 주제 기반 기술 포트폴리오

    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(20, 20))
    import seaborn as sns
    sns.set_style("ticks")
    # sns.set_context("notebook")
    sns.set_context("talk")

    input_df = copy(topic_time_df_summary)
    # input_df['CAGR_recent']= np.log(input_df['CAGR_recent'])
    input_df['volume_recent'] = np.log(input_df['volume_recent'])

    sns.scatterplot(data=input_df, x='volume_recent',
                    y='CAGR_recent', marker='o')

    for i in range(len(input_df)):

        plt.annotate(input_df['topic'][i],
                     (input_df['volume_recent'][i],
                     input_df['CAGR_recent'][i]), fontsize=14)

    plt.axvline(np.median(input_df['volume_recent']))
    plt.axhline(np.median(input_df['CAGR']))
    # plt.show()
    plt.savefig(output_directory+'portfolio.png',
                dpi=1000, bbox_inches='tight',)

    print('LDA 주제 기반 기술 포트폴리오 끝')
