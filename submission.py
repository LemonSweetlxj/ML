import helper
import operator
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

def get_train(c0,c1,label0,label1):

    X_t_0 = []     
    for i in c0:
        sen = " ".join(i)
        X_t_0.append(sen)
        
    X_t_all = X_t_0.copy()
    y_t_all = label0.copy()

    X_t_1 = []
    for i in c1:
        sen = " ".join(i)
        X_t_1.append(sen)

    X_t_all.extend(X_t_1)
    y_t_all.extend(label1)  #前360个为0， 后180个为1 [0,0,0,1,1,1]        
    X_train_0 = np.array(X_t_0)   ##所有数据 class 0
    X_train_1 = np.array(X_t_1)   ##所有数据 class 1
    X_train = np.array(X_t_all)   ##class 1 和 0 的合并数据
    y_train = np.array(y_t_all)
    
    return X_train_0, X_train_1, X_train, label0, label1, y_train


def count_vectorize(X_train): #只建一个countv 5718
    cv = CountVectorizer(min_df = 0, token_pattern = '\S+')
##    cv = CountVectorizer()
    count_train = cv.fit_transform(X_train)  ##train里所有的内容
    train_words = cv.get_feature_names()    
    return cv, count_train,train_words

###train， test先不管    
def build_clfs(X_train_0, X_train_1, X_train, label0, label1, y_train, cv, count_train):

    ######8个1比1分类器，1个全部分类器，1比1分类器里6个是随机的
    clf_1 = svm.SVC(C = 4, probability = True, kernel='linear')  
    clf_2 = svm.SVC(C = 4, probability = True, kernel='linear')
    clf_3 = svm.SVC(C = 4, probability = True, kernel='linear')
    clf_4 = svm.SVC(C = 4, probability = True, kernel='linear')
    clf_5 = svm.SVC(C = 4, probability = True, kernel='linear')
    clf_6 = svm.SVC(C = 4, probability = True, kernel='linear')
    clf_7 = svm.SVC(C = 4, probability = True, kernel='linear')
    clf_8 = svm.SVC(C = 4, probability = True, kernel='linear')
    clf_9 = svm.SVC(C = 4, probability = True, kernel='linear')
    clf_10 = svm.SVC(C = 4, probability = True, kernel='linear')
    clf_11 = svm.SVC(C = 4, probability = True, kernel='linear')

    ##向量化，共享单词表
    cv2 = CountVectorizer(vocabulary=cv.vocabulary_)

    #0少1多， 0全部保留， 让1截取， 为了普遍性， 不用具体数值，用len(number)
    if len(X_train_0) < len(X_train_1):
        number = len(X_train_0)

        ##设置标签 前180为1，后180为0
        new_label_1 = [1 for i in range(number)]
        new_label_1.extend(label0)
        y_train_all_1 = np.array(new_label_1)

        #截取
        X_train_1_1 = X_train_1[-number:] #后180个
        X_train_all_1 = np.hstack([X_train_1_1,X_train_0])
        X_train_1_2 = X_train_1[:number]  #前180个
        X_train_all_2 = np.hstack([X_train_1_2,X_train_0])
        X_train_1_3 = np.random.choice(X_train_1, number) #随机选取180个 1号
        X_train_all_3 = np.hstack([X_train_1_3,X_train_0])
        X_train_1_4 = np.random.choice(X_train_1, number) #随机选取180个 2号
        X_train_all_4 = np.hstack([X_train_1_4,X_train_0])
        X_train_1_5 = np.random.choice(X_train_1, number) #随机选取180个 2号
        X_train_all_5 = np.hstack([X_train_1_5,X_train_0])
        X_train_1_6 = np.random.choice(X_train_1, number) #随机选取180个 2号
        X_train_all_6 = np.hstack([X_train_1_6,X_train_0])
        X_train_1_7 = np.random.choice(X_train_1, number) #随机选取180个 2号
        X_train_all_7 = np.hstack([X_train_1_7,X_train_0])
        X_train_1_8 = np.random.choice(X_train_1, number) #随机选取180个 2号
        X_train_all_8 = np.hstack([X_train_1_8,X_train_0])
        X_train_1_9 = np.random.choice(X_train_1, number) #随机选取180个 2号
        X_train_all_9 = np.hstack([X_train_1_9,X_train_0])
        X_train_1_10 = np.random.choice(X_train_1, number) #随机选取180个 2号
        X_train_all_10 = np.hstack([X_train_1_10,X_train_0])


    #1少0多， 1全部保留， 让0截取， 为了普遍性， 不用具体数值，用len(number)
    else:
        number = len(X_train_1)

        ##设置标签 前180为0，后180为1
        new_label_0 = [0 for i in range(number)]
        new_label_0.extend(label1)
        y_train_all_1 = np.array(new_label_0)

        #截取
        X_train_0_1 = X_train_0[-number:] #后180个
        X_train_all_1 = np.hstack([X_train_0_1,X_train_1])
        X_train_0_2 = X_train_0[:number]  #前180个
        X_train_all_2 = np.hstack([X_train_0_2,X_train_1])
        X_train_0_3 = np.random.choice(X_train_0, number) #随机选取180个 1号
        X_train_all_3 = np.hstack([X_train_0_3,X_train_1])
        X_train_0_4 = np.random.choice(X_train_0, number) #随机选取180个 2号
        X_train_all_4 = np.hstack([X_train_0_4,X_train_1])
        X_train_0_5 = np.random.choice(X_train_0, number) #随机选取180个 2号
        X_train_all_5 = np.hstack([X_train_0_5,X_train_1])
        X_train_0_6 = np.random.choice(X_train_0, number) #随机选取180个 2号
        X_train_all_6 = np.hstack([X_train_0_6,X_train_1])
        X_train_0_7 = np.random.choice(X_train_0, number) #随机选取180个 2号
        X_train_all_7 = np.hstack([X_train_0_7,X_train_1])
        X_train_0_8 = np.random.choice(X_train_0, number) #随机选取180个 2号
        X_train_all_8 = np.hstack([X_train_0_8,X_train_1])
        X_train_0_9 = np.random.choice(X_train_0, number) #随机选取180个 2号
        X_train_all_9 = np.hstack([X_train_0_9,X_train_1])
        X_train_0_10 = np.random.choice(X_train_0, number) #随机选取180个 2号
        X_train_all_10 = np.hstack([X_train_0_10,X_train_1])

       
    ##向量化
    count_train_1 = cv2.fit_transform(X_train_all_1)  #360,5718
    count_train_2 = cv2.fit_transform(X_train_all_2)
    count_train_3 = cv2.fit_transform(X_train_all_3)
    count_train_4 = cv2.fit_transform(X_train_all_4)
    count_train_5 = cv2.fit_transform(X_train_all_5)
    count_train_6 = cv2.fit_transform(X_train_all_6)
    count_train_7 = cv2.fit_transform(X_train_all_7)
    count_train_8 = cv2.fit_transform(X_train_all_8)
    count_train_9 = cv2.fit_transform(X_train_all_9)
    count_train_10 = cv2.fit_transform(X_train_all_10)

    ##fit
    clf_1.fit(count_train_1,y_train_all_1)
    clf_2.fit(count_train_2,y_train_all_1)
    clf_3.fit(count_train_3,y_train_all_1)
    clf_4.fit(count_train_4,y_train_all_1)
    clf_5.fit(count_train_5,y_train_all_1)
    clf_6.fit(count_train_6,y_train_all_1)
    clf_7.fit(count_train_7,y_train_all_1)
    clf_8.fit(count_train_8,y_train_all_1)
    clf_9.fit(count_train_9,y_train_all_1)
    clf_10.fit(count_train_10,y_train_all_1)
    clf_11.fit(count_train,y_train)

    return clf_1, clf_2, clf_3, clf_4, clf_5,clf_6, clf_7, clf_8, clf_9,clf_10, clf_11,cv2

#找单词与权重的对应
def get_correspond(clf_1, clf_2, clf_3, clf_4, clf_5, clf_6, clf_7,clf_8, clf_9,clf_10, clf_11,train_words):
    d_index_weight = {}   #取所有weight {index：[weight1, weight2..]}

    index_1 = clf_1.coef_.indices.tolist()
    weight_1 = clf_1.coef_.data.tolist()
    for i in range(len(index_1)):
        if index_1[i] not in d_index_weight:
            d_index_weight[index_1[i]] = [weight_1[i]]

    index_2 = clf_2.coef_.indices.tolist()
    weight_2 = clf_2.coef_.data.tolist()
    for i in range(len(index_2)):
        if index_2[i] not in d_index_weight:
            d_index_weight[index_2[i]] = [weight_2[i]]
        else:
            d_index_weight[index_2[i]].append(weight_2[i]) 

    index_3 = clf_3.coef_.indices.tolist()
    weight_3 = clf_3.coef_.data.tolist()
    for i in range(len(index_3)):
        if index_3[i] not in d_index_weight:
            d_index_weight[index_3[i]] = [weight_3[i]]
        else:
            d_index_weight[index_3[i]].append(weight_3[i]) 

    index_4 = clf_4.coef_.indices.tolist()
    weight_4 = clf_4.coef_.data.tolist()
    for i in range(len(index_4)):
        if index_4[i] not in d_index_weight:
            d_index_weight[index_4[i]] = [weight_4[i]]
        else:
            d_index_weight[index_4[i]].append(weight_4[i])

    index_5 = clf_5.coef_.indices.tolist()
    weight_5 = clf_5.coef_.data.tolist()
    for i in range(len(index_5)):
        if index_5[i] not in d_index_weight:
            d_index_weight[index_5[i]] = [weight_5[i]]
        else:
            d_index_weight[index_5[i]].append(weight_5[i])

    index_6 = clf_6.coef_.indices.tolist()
    weight_6 = clf_6.coef_.data.tolist()
    for i in range(len(index_6)):
        if index_6[i] not in d_index_weight:
            d_index_weight[index_6[i]] = [weight_6[i]]
        else:
            d_index_weight[index_6[i]].append(weight_6[i])

    index_7 = clf_7.coef_.indices.tolist()        
    weight_7 = clf_7.coef_.data.tolist()
    for i in range(len(index_7)):
        if index_7[i] not in d_index_weight:
            d_index_weight[index_7[i]] = [weight_7[i]]
        else:
            d_index_weight[index_7[i]].append(weight_7[i])


    index_8 = clf_8.coef_.indices.tolist()
    weight_8 = clf_8.coef_.data.tolist()
    for i in range(len(index_8)):
        if index_8[i] not in d_index_weight:
            d_index_weight[index_8[i]] = [weight_8[i]]
        else:
            d_index_weight[index_8[i]].append(weight_8[i])

    index_9 = clf_9.coef_.indices.tolist()        
    weight_9 = clf_9.coef_.data.tolist()
    for i in range(len(index_9)):
        if index_9[i] not in d_index_weight:
            d_index_weight[index_9[i]] = [weight_9[i]]
        else:
            d_index_weight[index_9[i]].append(weight_9[i])

    index_10 = clf_10.coef_.indices.tolist()
    weight_10 = clf_10.coef_.data.tolist()
    for i in range(len(index_10)):
        if index_10[i] not in d_index_weight:
            d_index_weight[index_10[i]] = [weight_10[i]]
        else:
            d_index_weight[index_10[i]].append(weight_10[i])

    index_11 = clf_11.coef_.indices.tolist()        
    weight_11 = clf_11.coef_.data.tolist()
    for i in range(len(index_11)):
        if index_11[i] not in d_index_weight:
            d_index_weight[index_11[i]] = [weight_11[i]]
        else:
            d_index_weight[index_11[i]].append(weight_11[i])

    d_index_mean = {}      ##取均值 {index：weight}
    for (k,v) in d_index_weight.items():
        d_index_mean[k] = sum(v)/len(v)

    word_weight_tmp = {}
    for (k,v) in d_index_mean.items():
        word_weight_tmp[train_words[k]] = v

    word_weight_positive = {}      ##取名称并排序(降序，正在前) {word：weight}
    temp = sorted(word_weight_tmp.items(), key = operator.itemgetter(1),reverse = True)
    for pairwise in temp:
        word_weight_positive[pairwise[0]] = pairwise[1]

    word_weight_negative = {}      ##取名称并排(升序，负在前)  {word：weight}
    temp = sorted(word_weight_tmp.items(), key = operator.itemgetter(1),reverse = False)
    for pairwise in temp:
        word_weight_negative[pairwise[0]] = pairwise[1]
    return word_weight_positive, word_weight_negative


def test_clfs(clf_1, clf_2, clf_3, clf_4, clf_5, clf_6, clf_7, clf_8, clf_9, clf_10, clf_11, X_test, y_test,cv2):
    vote_predict = []
    
    count_test = cv2.fit_transform(X_test)
    y_pred_1 = clf_1.predict(count_test)
    y_pred_2 = clf_2.predict(count_test)
    y_pred_3 = clf_3.predict(count_test)
    y_pred_4 = clf_4.predict(count_test)
    y_pred_5 = clf_5.predict(count_test)
    y_pred_6 = clf_6.predict(count_test)
    y_pred_7 = clf_7.predict(count_test)
    y_pred_8 = clf_8.predict(count_test)
    y_pred_9 = clf_9.predict(count_test)
    y_pred_10 = clf_10.predict(count_test)
    y_pred_11 = clf_11.predict(count_test)
    
    for i in range(len(y_pred_1)):
        one_doc = []
        one_doc.append(y_pred_1[i])
        one_doc.append(y_pred_2[i])
        one_doc.append(y_pred_3[i])
        one_doc.append(y_pred_4[i])
        one_doc.append(y_pred_5[i])
        one_doc.append(y_pred_6[i])
        one_doc.append(y_pred_7[i])
        one_doc.append(y_pred_8[i])
        one_doc.append(y_pred_9[i])
        one_doc.append(y_pred_10[i])
        one_doc.append(y_pred_11[i])
        
        if one_doc.count(1) > one_doc.count(0):
            vote_predict.append(1)
        else:
            vote_predict.append(0)

    y_pred = np.array(vote_predict)
                
    return  y_pred


def get_test_data_1(test_file):
    test = []
    with open(test_file,'r') as t:
        for line in t:
            test.append(line.strip())
    X_test = np.array(test)
    y_test = np.array([1 for i in range(len(X_test))])
    return X_test,y_test

def get_test_data_2(test_file):
    with open(test_file, 'r') as f_test:       ##test_file 处理
        test_data = []
        text = f_test.readlines()
        for single_text in text:
            single_text_list = single_text.split(' ')  ##拿到所有词
            test_data.append(single_text_list)
    return test_data


def modify_data(test_data, top_positive, top_negative):
    
    
    modified_data = []
#    all_count = []
    c = 10
    
    for doc in test_data:
        new_doc = set(doc)
        p = 0
        p_words = []
        for i in top_positive:
            if i in new_doc:
                p_words.append(i)
                p += 1 
                if p == c:
                    break
          
        n = 0
        n_words = []
        for j in top_negative:
            if j not in new_doc:
                n_words.append(j)
                n += 1 
                if n == c:
                    break
        tmp_doc = list(new_doc)        
        for i in range(c):
            tmp_doc.remove(p_words[i])
            tmp_doc.append(n_words[i])
            
        doc_string_tmp = ' '.join(tmp_doc)
        
        doc_list = doc_string_tmp.split()
        
        doc_string = ' '.join(doc_list)
        
        modified_data.append(doc_string)
        
    X_modif_test = np.array(modified_data)
    y_modif_test = np.array([1 for i in range(len(X_modif_test))])
    return X_modif_test, y_modif_test, modified_data 
    

def fool_classifier(test_data): ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...
    
    
    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance=helper.strategy() 
    parameters={'gamma':'auto', 'C': 2, 'kernel': 'linear', 'degree':3, 'coef0':0.0}
    
    ##..................................#
    #
    #
    #    ## Your implementation goes here....#
    '''得到全部的数据'''
    c0 = strategy_instance.class0
    label0 = [0 for i in range(len(c0))]  ##class0 的所有数据 360全部保留
    c1 = strategy_instance.class1
    label1 = [1 for i in range(len(c1))]   ##class1 的所有数据  180全部保留
    
    X_train_0, X_train_1, X_train, label0, label1, y_train = get_train(c0,c1,label0,label1)   
    X_test, y_test = get_test_data_1(test_data)
    cv, count_train, train_words = count_vectorize(X_train)
    clf_1, clf_2, clf_3,clf_4, clf_5,clf_6, clf_7, clf_8, clf_9,clf_10, clf_11, cv2 = build_clfs(X_train_0, X_train_1, X_train, label0, label1, y_train, cv, count_train)
    word_weight_positive, word_weight_negative = get_correspond(clf_1, clf_2, clf_3, clf_4, clf_5,clf_6, clf_7,clf_8, clf_9,clf_10, clf_11,train_words)
    top_positive = {k: word_weight_positive[k] for k in list(word_weight_positive)}
    top_negative = {k: word_weight_negative[k] for k in list(word_weight_negative)}
#    print(top_negative)
    y_pred_1 = test_clfs(clf_1, clf_2, clf_3, clf_4, clf_5,clf_6, clf_7,clf_8, clf_9,clf_10, clf_11, X_test, y_test,cv2)
    test_data_cont = get_test_data_2(test_data)
    X_modif_test, y_modif_test, modified_data = modify_data(test_data_cont, top_positive, top_negative)
    y_pred_2 = test_clfs(clf_1, clf_2, clf_3, clf_4, clf_5,clf_6, clf_7,clf_8, clf_9,clf_10, clf_11, X_modif_test, y_modif_test,cv2)

    acc_1 = metrics.accuracy_score(y_test, y_pred_1) ##改词前 越高越好
    print("before modify:", acc_1)
    acc_2 = metrics.accuracy_score(y_modif_test, y_pred_2) ##改词后 越低越好
    print("after modify:", acc_2)    
    #
    #
    #
    ##..................................#
    
    
    ## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory...
    modified_data_file = open("modified_data.txt", "w")
    for i in modified_data:
        modified_data_file.write(i)
        modified_data_file.write('\n')

    modified_data_file.close()
    
    ## You can check that the modified text is within the modification limits.
    modified_data='./modified_data.txt'
    assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance ## NOTE: You are required to return the instance of this class.

fool_classifier('test_data.txt')
