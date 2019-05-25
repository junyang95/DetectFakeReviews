import nltk
from nltk.tokenize import word_tokenize

stopword = nltk.corpus.stopwords.words(
    'english')  # All English Stopwords Dont forget to type in terminal python -m nltk.downloader stopwords
PorterStemmer = nltk.PorterStemmer()
WordNetLemmatizer = nltk.WordNetLemmatizer()

import string
import pandas

pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('display.max_colwidth', 400)  # To extend column width

import sqlalchemy
import re
import sys
import collections

from sklearn.feature_extraction.text import TfidfVectorizer

conn = sqlalchemy.create_engine('postgresql://junyang:123456@localhost:5432/postgres',
                                connect_args={'options': '-csearch_path={}'.format('lyt_project_schema')})

query_fake_positive = "SELECT id, review, fake, polarity FROM reviews_table WHERE fake=1 AND polarity=1 ORDER BY id;"
query_fake_negative = "SELECT id, review, fake, polarity FROM reviews_table WHERE fake=1 AND polarity=0 ORDER BY id;"
query_real_positive = "SELECT id, review, fake, polarity FROM reviews_table WHERE fake=0 AND polarity=1 ORDER BY id;"
query_real_negative = "SELECT id, review, fake, polarity FROM reviews_table WHERE fake=0 AND polarity=0 ORDER BY id;"
query_positive = "SELECT id, review, fake, polarity FROM reviews_table WHERE polarity=1 ORDER BY id;"
query_negative = "SELECT id, review, fake, polarity FROM reviews_table WHERE polarity=0 ORDER BY id;"
query_all = "SELECT id, review, fake, polarity FROM reviews_table ORDER BY id;"

data_fake_positive = pandas.read_sql_query(query_fake_positive, con=conn)  # id = 801~1200
data_fake_negative = pandas.read_sql_query(query_fake_negative, con=conn)  # id = 001~400
data_real_positive = pandas.read_sql_query(query_real_positive, con=conn)  # id = 1201~1600
data_real_negative = pandas.read_sql_query(query_real_negative, con=conn)  # id = 401~800
data_positive = pandas.read_sql_query(query_positive, con=conn)  # id = 801~1200 + 1201~1600
data_negative = pandas.read_sql_query(query_negative, con=conn)  # id = 001~400 + 401~800
data_all = pandas.read_sql_query(query_all, con=conn)  # id = 001~1600


#######################################################################################################################


#######################################################################################################################
def lower_case(text):
    text_lowercase = text.lower()
    return text_lowercase


def remove_punct(text):
    text_nopunct = "".join(
        [char for char in text if char not in string.punctuation])  # It will discard all punctuations
    return text_nopunct


def replace_abbreviations(text):
    # patterns that used to find or/and replace particular chars or words
    # to find chars that are not a letter, a blank or a quotation
    pat_letter = re.compile(r'[^a-zA-Z \']+')
    # to find the 's following the pronouns. re.I is refers to ignore case
    pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
    # to find the 's following the letters
    pat_s = re.compile("(?<=[a-zA-Z])\'s")
    # to find the ' following the words ending by s
    pat_s2 = re.compile("(?<=s)\'s?")
    # to find the abbreviation of not
    pat_not = re.compile("(?<=[a-zA-Z])n\'t")
    # to find the abbreviation of would
    pat_would = re.compile("(?<=[a-zA-Z])\'d")
    # to find the abbreviation of will
    pat_will = re.compile("(?<=[a-zA-Z])\'ll")
    # to find the abbreviation of am
    pat_am = re.compile("(?<=[I|i])\'m")
    # to find the abbreviation of are
    pat_are = re.compile("(?<=[a-zA-Z])\'re")
    # to find the abbreviation of have
    pat_ve = re.compile("(?<=[a-zA-Z])\'ve")

    new_text = pat_letter.sub(' ', text).strip().lower()  # 去符号，去字节前，和字节后的whitspaces（去无意义的空格），全部变小写
    new_text = pat_is.sub(r"\1 is", new_text)
    new_text = pat_s.sub("", new_text)
    new_text = pat_s2.sub("", new_text)
    new_text = pat_not.sub(" not", new_text)
    new_text = pat_would.sub(" would", new_text)
    new_text = pat_will.sub(" will", new_text)
    new_text = pat_am.sub(" am", new_text)
    new_text = pat_are.sub(" are", new_text)
    new_text = pat_ve.sub(" have", new_text)
    new_text = new_text.replace('\'', ' ')
    return new_text


# data_fake_positive['review_clean'] = data_fake_positive['review'].apply(lambda x: replace_abbreviations(x))
# data_fake_negative['review_clean'] = data_fake_negative['review'].apply(lambda x: replace_abbreviations(x))
# data_real_positive['review_clean'] = data_real_positive['review'].apply(lambda x: replace_abbreviations(x))
# data_real_negative['review_clean'] = data_real_negative['review'].apply(lambda x: replace_abbreviations(x))


# data_fake_positive['review_clean'] = data_fake_positive['review'].apply(lambda x: remove_punct(x)).apply(lambda x: lower_case(x))
# data_fake_negative['review_clean'] = data_fake_negative['review'].apply(lambda x: remove_punct(x)).apply(lambda x: lower_case(x))
# data_real_positive['review_clean'] = data_real_positive['review'].apply(lambda x: remove_punct(x)).apply(lambda x: lower_case(x))
# data_real_negative['review_clean'] = data_real_negative['review'].apply(lambda x: remove_punct(x)).apply(lambda x: lower_case(x))
#######################################################################################################################
def tokenize(text):
    tokens = word_tokenize(text)  # W+ means that either a word character (A-Za-z0-9_) or a dash (-) can go there.
    return tokens


# data_fake_positive['review_clean_tokenized'] = data_fake_positive['review_clean'].apply(lambda x: tokenize(x))
# data_fake_negative['review_clean_tokenized'] = data_fake_negative['review_clean'].apply(lambda x: tokenize(x))
# data_real_positive['review_clean_tokenized'] = data_real_positive['review_clean'].apply(lambda x: tokenize(x))
# data_real_negative['review_clean_tokenized'] = data_real_negative['review_clean'].apply(lambda x: tokenize(x))


#######################################################################################################################
def pos_tag(tokenized_list):
    tag = nltk.pos_tag(tokenized_list)
    return tag


# data_fake_positive['review_clean_tokenized_tag'] = data_fake_positive['review_clean_tokenized'].apply(lambda x: pos_tag(x))
# data_fake_negative['review_clean_tokenized_tag'] = data_fake_negative['review_clean_tokenized'].apply(lambda x: pos_tag(x))
# data_real_positive['review_clean_tokenized_tag'] = data_real_positive['review_clean_tokenized'].apply(lambda x: pos_tag(x))
# data_real_negative['review_clean_tokenized_tag'] = data_real_negative['review_clean_tokenized'].apply(lambda x: pos_tag(x))


########################################################################################################################
def get_wordnet_pos(
        treebank_tag):  # https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return False


def lemmatizing(tokenized_tag_list):
    lemmatized = []
    for tokenized_tag in tokenized_tag_list:
        correct_pos = get_wordnet_pos(tokenized_tag[1])
        if correct_pos != False:
            lemmatized_word = WordNetLemmatizer.lemmatize(tokenized_tag[0], correct_pos)
            lemmatized.append(lemmatized_word)
        elif correct_pos == False:
            lemmatized.append(tokenized_tag[0])
    return lemmatized


# data_fake_positive['review_clean_tokenized_lemmatized'] = data_fake_positive['review_clean_tokenized_tag'].apply(lambda x: lemmatizing(x))
# data_fake_negative['review_clean_tokenized_lemmatized'] = data_fake_negative['review_clean_tokenized_tag'].apply(lambda x: lemmatizing(x))
# data_real_positive['review_clean_tokenized_lemmatized'] = data_real_positive['review_clean_tokenized_tag'].apply(lambda x: lemmatizing(x))
# data_real_negative['review_clean_tokenized_lemmatized'] = data_real_negative['review_clean_tokenized_tag'].apply(lambda x: lemmatizing(x))


########################################################################################################################
def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]  # To remove all stopwords
    return text


# data_fake_positive['review_clean_tokenized_lemmatized_nostop'] = data_fake_positive[
#     'review_clean_tokenized_lemmatized'].apply(lambda x: remove_stopwords(x))
# data_fake_negative['review_clean_tokenized_lemmatized_nostop'] = data_fake_negative[
#     'review_clean_tokenized_lemmatized'].apply(lambda x: remove_stopwords(x))
# data_real_positive['review_clean_tokenized_lemmatized_nostop'] = data_real_positive[
#     'review_clean_tokenized_lemmatized'].apply(lambda x: remove_stopwords(x))
# data_real_negative['review_clean_tokenized_lemmatized_nostop'] = data_real_negative[
#     'review_clean_tokenized_lemmatized'].apply(lambda x: remove_stopwords(x))


########################################################################################################################
def long_sentence(tokenized_list):
    sentence = "".join([word + " " for word in tokenized_list])
    return sentence
    # 这里把lemmatized之后的array还原为一整句话


# data_fake_positive['clean_sentence'] = data_fake_positive['review_clean_tokenized_lemmatized_nostop'].apply(
#     lambda x: long_sentence(x))
# data_fake_negative['clean_sentence'] = data_fake_negative['review_clean_tokenized_lemmatized_nostop'].apply(
#     lambda x: long_sentence(x))
# data_real_positive['clean_sentence'] = data_real_positive['review_clean_tokenized_lemmatized_nostop'].apply(
#     lambda x: long_sentence(x))
# data_real_negative['clean_sentence'] = data_real_negative['review_clean_tokenized_lemmatized_nostop'].apply(
#     lambda x: long_sentence(x))
#######################################################################################################################
# def total_word_count(data_X_Y):
#     final_long_sentence = ""
#     rows = data_X_Y.shape[0]
#     for x in range(rows):
#         final_long_sentence = final_long_sentence + data_X_Y.iloc[x]['clean_sentence'] + " "
#     return collections.Counter(final_long_sentence.split())
#######################################################################################################################

###############CountVectorizer###############
from sklearn.feature_extraction.text import CountVectorizer


def getCountVectorizer(data_X_Y):
    count_vect = CountVectorizer()
    X_counts = count_vect.fit_transform(data_X_Y['clean_sentence'])
    # print(X_counts.shape)          #这里可以调取矩阵的大小
    # print(count_vect.get_feature_names())      #这里可以调取每一个feature的名字
    X_counts_df = pandas.DataFrame(X_counts.toarray(), columns=count_vect.get_feature_names())
    #print(X_counts_df)
    # return X_counts_df
    return X_counts


###############Vectorizing Raw Data: N-Grams###############
def getNGramVectorizer(data_X_Y):
    ngram_vect = CountVectorizer(ngram_range=(2, 2))  # It applies only bigram vectorizer
    X_counts = ngram_vect.fit_transform(data_X_Y['clean_sentence'])
    X_counts_df = pandas.DataFrame(X_counts.toarray(), columns=ngram_vect.get_feature_names())
    #print(X_counts_df)
    # return X_counts_df
    return X_counts


###############Vectorizing Raw Data: TF-IDF###############
def getTfidfVectorizer(data_X_Y):
    tfidf_vect = TfidfVectorizer()
    X_tfidf = tfidf_vect.fit_transform(data_X_Y['clean_sentence'])
    X_tfidf_df = pandas.DataFrame(X_tfidf.toarray(), columns=tfidf_vect.get_feature_names())
    #print(X_tfidf_df)
    # X_tfidf_feat = pandas.concat([data_X_Y['body_len'], data_X_Y['punct%'], pandas.DataFrame(X_tfidf.toarray())], axis=1)
    # return X_tfidf_df
    return X_tfidf


#######################################################################################################################
####To get punctuation feature from the scentence####
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count / (len(text) - text.count(" ")), 3) * 100


def getPunctuationFeature(data_X_Y):
    data_X_Y['body_len'] = data_X_Y['review'].apply(lambda x: len(x) - x.count(" "))
    data_X_Y['punct%'] = data_X_Y['review'].apply(lambda x: count_punct(x))


# getPunctuationFeature(data_fake_positive)
# getPunctuationFeature(data_fake_negative)
# getPunctuationFeature(data_real_positive)
# getPunctuationFeature(data_real_negative)


# 这里是画图的部分
import matplotlib.pyplot as plt
import numpy as np

# bins = np.linspace(0, 3000, 30)
# plt.hist(data_fake_positive['body_len'], bins, alpha=0.5, normed=False, label='data_fake_positive_length')
# plt.hist(data_fake_negative['body_len'], bins, alpha=0.5, normed=False, label='data_fake_negative_length')
# plt.hist(data_real_positive['body_len'], bins, alpha=0.5, normed=False, label='data_real_positive_length')
# plt.hist(data_real_negative['body_len'], bins, alpha=0.5, normed=False, label='data_real_negative_length')

# bins = np.linspace(0, 12, 30)
# plt.hist(data_fake_positive['punct%'], bins, alpha=0.5, normed=False, label='data_fake_positive_length')
# plt.hist(data_fake_negative['punct%'], bins, alpha=0.5, normed=False, label='data_fake_negative_length')
# plt.hist(data_real_positive['punct%'], bins, alpha=0.5, normed=False, label='data_real_positive_length')
# plt.hist(data_real_negative['punct%'], bins, alpha=0.5, normed=False, label='data_real_negative_length')

# plt.legend(loc='upper right')
# plt.show()

#######################################################################################################################
###############Linear Classifier###############

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import LinearSVC

from sklearn.neural_network import MLPClassifier


def preProcesstheDataset(data_type):
    data_type['review_clean'] = data_type['review'].apply(lambda x: replace_abbreviations(x))
    data_type['review_clean_tokenized'] = data_type['review_clean'].apply(lambda x: tokenize(x))
    data_type['review_clean_tokenized_tag'] = data_type['review_clean_tokenized'].apply(lambda x: pos_tag(x))
    data_type['review_clean_tokenized_lemmatized'] = data_type['review_clean_tokenized_tag'].apply(
        lambda x: lemmatizing(x))
    data_type['review_clean_tokenized_lemmatized_nostop'] = data_type['review_clean_tokenized_lemmatized'].apply(
        lambda x: remove_stopwords(x))
    data_type['clean_sentence'] = data_type['review_clean_tokenized_lemmatized_nostop'].apply(
        lambda x: long_sentence(x))

    data_type['body_len'] = data_type['review'].apply(lambda x: len(x) - x.count(" "))
    data_type['punct%'] = data_type['review'].apply(lambda x: count_punct(x))
    #print(data_type)


def getModelByVectorizer(data_type, nameVectorizer, attribute, confusionMatrix, train_size, test_size, trials,
                         classifier):
    if nameVectorizer == 'Count':
        features_nd = getCountVectorizer(data_type)
    elif nameVectorizer == 'NGram':
        features_nd = getNGramVectorizer(data_type)
    elif nameVectorizer == 'Tfidf':
        features_nd = getTfidfVectorizer(data_type)

    ########这里把特征变成array的形式#######
    # print(features_nd.toarray())
    ########这两种写法都可以把pandas转换为numpy都array########
    # print(data_type.values[:, 2])
    # print(data_type[attribute].values)        #https://www.reddit.com/r/learnpython/comments/4c6bjc/getting_the_index_number_of_a_specific_row_pandas/

    f1_score_0 = []
    f1_score_1 = []

    for x in range(0, trials):
        X_train, X_test, y_train, y_test = train_test_split(
            features_nd.toarray(),
            data_type[attribute],  # can measure either fake or polarity
            train_size=train_size,
            test_size=test_size,
            random_state=1234 + x
        )

        ###########这里是训练数据集，放入到逻辑回归模型当中#############
        if classifier == 'NB_Multinomial':
            model = MultinomialNB()  # _Naive Bayes classifier for multinomial models
            # Multinomial - good for when your features (categorical or continupous)
            # describe discrete frequency counts (eg. word counts)
        elif classifier == 'NB_Gaussian':
            model = GaussianNB()  # Gaussian Naive Bayes (GaussianNB)
            # Good for making predictions from normally distributed features
        elif classifier == 'NB_Bernoulli':
            model = BernoulliNB(binarize=False)  # _Naive Bayes classifier for multivariate Bernoulli models.
            # Good for making predictions from binary features
        elif classifier == 'LR':
            model = LogisticRegression(solver='lbfgs')
        elif classifier == 'DT':
            model = DecisionTreeClassifier(random_state=0)
        elif classifier == 'SVM':
            model = LinearSVC()
        elif classifier == 'MLP':
            model = MLPClassifier(solver='lbfgs')



        model = model.fit(X=X_train, y=y_train)
        ##########这里将前面交叉验证中取出到测试集，放入到了刚才的回归模型当中##########
        y_pred = model.predict(X_test)

        #########打印，模型给出的predication的答案##########
        # print(y_pred)
        # print(len(y_pred))

        ########同时打印召回率精确率等score, y_test可以理解为正确答案，y_pred可以理解为模型给出的答案#########
        #print(classification_report(y_test, y_pred, digits=3))
        output_dict = classification_report(y_test, y_pred, digits=3, output_dict=True)
        if attribute == 'fake':
            f1_score_0.append(output_dict.get("0").get("f1-score"))
            f1_score_1.append(output_dict.get("1").get("f1-score"))
        elif attribute == 'polarity':
            f1_score_0.append(output_dict.get("0.0").get("f1-score"))
            f1_score_1.append(output_dict.get("1.0").get("f1-score"))

        #####y_test就是答案######
        answers = y_test.values
        answers_mean = []

        indexs = y_test.index.values
        reviews = []

        predictions = y_pred
        predictions_mean = []

        truth = []

        for answer, prediction, index in zip(answers, predictions, indexs):
            review = data_type['review'].values[index]
            reviews.append(review)

            if answer == prediction:
                truth.append('✓')
            elif answer != prediction:
                truth.append('x')

            if attribute == 'polarity':
                if answer == 1:
                    answer = 'P'
                elif answer == 0:
                    answer = 'N'
                answers_mean.append(answer)

                if prediction == 1:
                    prediction = 'P'
                elif prediction == 0:
                    prediction = 'N'
                predictions_mean.append(prediction)

            elif attribute == 'fake':
                if answer == 1:
                    answer = 'F'
                elif answer == 0:
                    answer = 'R'
                answers_mean.append(answer)

                if prediction == 1:
                    prediction = 'F'
                elif prediction == 0:
                    prediction = 'R'
                predictions_mean.append(prediction)

        test_table = pandas.DataFrame({'indexs': indexs,
                                       'answers': answers_mean,
                                       'predictions': predictions_mean,
                                       'turth': truth,
                                       'review': reviews})

        # print(answers)
        # print(answers_mean)
        #
        # print(predictions)
        # print(predictions_mean)
        #
        # print(indexs)

        #print(test_table)
        ########这里用data visulation的方法，y_test可以理解为正确答案，y_pred可以理解为模型给出的答案#######
        # Making the Confusion Matrix
        if confusionMatrix == True:
            cm = confusion_matrix(y_test, y_pred)
            class_label = ["1", "0"]
            df_cm = pandas.DataFrame(cm, index=class_label, columns=class_label)
            sns.heatmap(df_cm, annot=True, fmt='d')
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.show()

    dict_0_1 = {
        "0": f1_score_0,
        "1": f1_score_1
    }
    print(dict_0_1)
    return dict_0_1


#######################################################################################################################
import seaborn as sns
import matplotlib as mpl
from scipy.stats import ttest_ind


#
# mpl.use('agg')


def box_plot(arraydict, namedict, figName, classifier, figTitle):
    if len(arraydict) != len(namedict):
        raise ValueError('The number of arraydict and namedict do not match')

    dataFrame_to_plot = pandas.DataFrame(columns=['Vectorizer', 'Result', 'Model', 'f1-scores'])

    for i, dict, name in zip(range(len(arraydict)), arraydict, namedict):
        print("this is box_plot function")
        list_0 = dict.get("0")
        list_1 = dict.get("1")

        if len(list_0) != len(list_1):
            raise ValueError('The number of f1-scores for 0 and 1 are not match')

        numberofData = len(list_0) + len(list_1)
        for j in range(len(list_0)):
            dataFrame_to_plot.loc[i * numberofData + 2 * j] = [name, '0', classifier, list_0[j]]
            dataFrame_to_plot.loc[i * numberofData + 2 * j + 1] = [name, '1', classifier, list_1[j]]

    print(dataFrame_to_plot)

    Count = dataFrame_to_plot[dataFrame_to_plot['Vectorizer'] == 'Count']
    NGram = dataFrame_to_plot[dataFrame_to_plot['Vectorizer'] == 'NGram']
    Tfidf = dataFrame_to_plot[dataFrame_to_plot['Vectorizer'] == 'Tfidf']

    Count_f1 = Count['f1-scores']
    print("Count_f1 avg: " + str(Count['f1-scores'].mean()))
    NGram_f1 = NGram['f1-scores']
    print("NGram_f1 avg: " + str(NGram['f1-scores'].mean()))
    Tfidf_f1 = Tfidf['f1-scores']
    print("Tfidf_f1 avg: " + str(Tfidf['f1-scores'].mean()))

    print(ttest_ind(Count_f1, NGram_f1))
    print(ttest_ind(Count_f1, Tfidf_f1))
    print(ttest_ind(NGram_f1, Tfidf_f1))

    #sns.set(style="whitegrid")
    #fig, ax = plt.subplots()
    #sns.boxplot(x='Vectorizer', y='f1-scores', hue='Result', data=dataFrame_to_plot, ax=ax, palette='Set3').set_title(figTitle)
    #plt.savefig(figName, dpi=100)
    #plt.show()  # if use mpl.use('agg') then GUI will not show
    #plt.close()



    # data_to_plot = []
    # for columnName in list(dataFrame_to_plot.columns.values):
    #     dataFrameToList = dataFrame_to_plot[columnName].tolist()
    #     data_to_plot.append(dataFrameToList)
    #
    # # Create a figure instance
    # fig = plt.figure(1, figsize=(9, 6))
    #
    # # Create an axes instance
    # ax = fig.add_subplot(111)
    #
    # # Create the boxplot
    # bp = ax.boxplot(data_to_plot)
    #
    # # Save the figure
    # fig.savefig(figName, bbox_inches='tight')
    return dataFrame_to_plot

class crossSize:
    def __init__(self, ratio):
        if ratio > 1 or ratio < 0 :
            raise ValueError('the train_size must between 0 and 1')
        self.ratio = ratio

    def getSize(self):
        return self.ratio


def overView(data, attribute, train_size, test_size, trials, classifier):
    preProcesstheDataset(data)

    dict_Count = getModelByVectorizer(data, 'Count', attribute, False, train_size, test_size, trials, classifier)
    dict_NGram = getModelByVectorizer(data, 'NGram', attribute, False, train_size, test_size, trials, classifier)
    dict_Tfidf = getModelByVectorizer(data, 'Tfidf', attribute, False, train_size, test_size, trials, classifier)

    figTitle = 'Predict ' + str(attribute) + ': Using ' + str(classifier) + ' Classifier to fit ' + str(train_size * 100) + '%/' + str(
            test_size * 100) + '% ratio with ' + str(trials) + ' trials'
    figName = attribute + '_' + str(train_size) + '_' + str(test_size) +'_' + str(trials) + '_' + classifier + '.png'

    dataFrame_to_plot = box_plot([dict_Count, dict_NGram, dict_Tfidf], ['Count', 'NGram', 'Tfidf'], figName, classifier, figTitle)
    dataFrame_to_plot['TranningSize'] = train_size
    return dataFrame_to_plot

# getModelByVectorizer(data_all, 'Count', 'fake', False, 0.8, 0.2, 'NB_Multinomial')
# getModelByVectorizer(data_all, 'Count', 'fake', False, 0.8, 0.2, 'NB_Gaussian')
# getModelByVectorizer(data_all, 'Count', 'fake', False, 0.8, 0.2, 'NB_Bernoulli')

final_dataFrame_to_plot = pandas.DataFrame(columns=['Vectorizer', 'Result', 'Model', 'f1-scores', 'TranningSize'])
for x in range(5):
    #train_size = crossSize(0.50 + x * 0.1)
    #test_size = crossSize(0.50 - x * 0.1)
    train_size = crossSize(0.50 + 0.1 * x)
    test_size = crossSize(0.50 - 0.1 * x)
    dataFrame_to_plot = overView(data_all, 'fake', train_size.getSize(), test_size.getSize(), 10, 'NB_Bernoulli')
    final_dataFrame_to_plot = pandas.concat([final_dataFrame_to_plot, dataFrame_to_plot], ignore_index=True)

final_dataFrame_to_plot.to_csv('NB_Bernoulli.csv', sep='\t', encoding='utf-8', index=True)
print(final_dataFrame_to_plot)

# overView(data_all, 'polarity', 0.80, 0.20, 10, 'Logistic')
# overView(data_all, 'fake', 0.80, 0.20, 15, 'NB_Gaussian')



#######################################################################################################################
