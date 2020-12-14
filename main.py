import os
import sys
import xml.etree.ElementTree as ET
from gensim import corpora, models
import gensim
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
import numpy as np
from pandas import DataFrame
from statsmodels.tsa.stattools import grangercausalitytests
from collections import Counter
from scipy import stats
from datetime import datetime
import math

class PreProcessing:
    def __init__(self):
        self.iem_data = []
        self.nyt_data = []
        self.dictionary = None
        self.doc_array  = [] # list of documents
        self.date_array = [] # list of document dates

    def process_iem(self):
        cwd = os.getcwd()
        f = open(os.path.join(cwd,"iem_data","norm_prices.txt"), "r")
        lines = f.readlines()

        for line in lines:
            if "Dem" in line:
                data = line.split("\t")
                self.iem_data.append([data[0], data[1]])
        f.close()

    def process_nyt(self):
        # Generate txt for each day (each line is different document)
        # Only include paragraphs that contain Bush and/or Gore
        cwd = os.getcwd()
        nyt_path = os.path.join(cwd,"nyt_data")

        # check if data already preprocessed
        if (not os.path.exists(os.path.join(nyt_path,"08_31_00.txt"))):
            for month in os.listdir(nyt_path):
                month_path = os.path.join(nyt_path,month)
                if (os.path.isdir(month_path)):
                    for day in os.listdir(month_path):
                        day_path = os.path.join(month_path, day)
                        # In "day" directory
                        if (os.path.isdir(day_path)):
                            title = month + "_" + day + "_00.txt"
                            # Create new txt for each day
                            new_f = open(os.path.join(nyt_path,title),"w")
                            self.nyt_data.append(new_f)
                            # Process xml
                            for doc in os.listdir(day_path):
                                xml = os.path.join(day_path,doc)
                                print(xml)
                                if os.path.isfile(xml) and '.xml' in xml:
                                    tree = ET.parse(os.path.join(day_path,doc))
                                    root = tree.getroot()
                                    # Make string of paragraphs that contain Bush/Gore
                                    line = ""
                                    for para in root.iter('p'):
                                        txt = para.text
                                        if txt != None and ("Bush" in txt or "Gore" in txt):
                                            line += " " + txt
                                    # Write line to txt
                                    if line != "":
                                        new_f.write(line + "\n")
                                    
                            new_f.close()
        else:
            for file in os.listdir(nyt_path):
                if file.endswith(".txt"):
                    self.nyt_data.append(os.path.join(nyt_path,file))

    def set_dictionary(self):
        tokenizer = RegexpTokenizer(r'\w+')
        stop_words = get_stop_words('en')
        doc_array = []

        for doc in self.nyt_data:
            file_base = os.path.basename(doc).split("_")
            date = file_base[0] + "/" + file_base[1] + "/" + file_base[2].split(".")[0]

            f     = open(doc, "r")
            lines = f.readlines()
            for line in lines: # line = document
                # Got rid of the names of the presidents, since they're the most common words
                line = line.lower()
                
                line = line.replace("bush","")
                line = line.replace("george","")
                line = line.replace("al gore","")
                line = line.replace("gore","")
                line = line.replace("mr","")
                line = line.replace("presidential","")
                line = line.replace("president","")
                line = line.replace("campaign","")
                line = line.replace("vice","")
                line = line.replace("said","")
                line = line.replace("governor","")
                line = line.replace("clinton","")
                
                tokens = tokenizer.tokenize(line)
                stopped_tokens = [i for i in tokens if not i in stop_words]
                self.doc_array.append(stopped_tokens)
                self.date_array.append(date)

        self.dictionary   = corpora.Dictionary(self.doc_array)

    def get_iem(self):
        return self.iem_data

    def get_nyt(self):
        return self.nyt_data

    def get_data(self):
        return self.doc_array

    def get_dates(self):
        return self.date_array

    def get_dict(self):
        return self.dictionary

class Lda:
    def __init__(self, doc_array, dict, u, tn, prior):
        self.doc_array  = doc_array
        self.dict       = dict
        self.lda_model  = None
        self.corpus     = None
        self.prior      = prior
        self.u          = u
        self.num_topics = tn

    def set_corpus(self):
        # each doc is represented by tuples of int id (token) and counts
        self.corpus = [self.dict.doc2bow(doc) for doc in self.doc_array] 

    def generate_lda(self):
        # initialize prior if 1st iteration
        if (not any(self.prior)):
            vocab_len = len(self.dict)
            self.prior = np.full((vocab_len),1.0/vocab_len)

        self.lda_model = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=self.num_topics, id2word=self.dict, passes=3, eta=self.prior, decay = self.u)

    def update_prior(self, new_prior):
        self.prior = new_prior

    def get_top_words(self,topic_impacts):
        words = ""
        for t in topic_impacts:
            for (word, prob) in self.lda_model.get_topic_terms(t, topn=10):
                words = words + str(self.dict.get(word)) + ", "
            words = words + "\n"
        return words

class TopicCausality:
    def __init__(self, lda_model, iem_data, dates, sig):
        self.lda_model = lda_model
        self.iem_data  = iem_data
        self.dates     = dates
        self.sig       = sig
        self.topic_ts  = [] # each index = topic #, containing ts data
        self.topic_impacts = []

    # generate topic time series
    def generate_topic_ts(self):
        # 1. Init self.topic_ts, where each index holds [topic #, [time series]]
        num_topics = self.lda_model.num_topics
        date_set   = list(set(self.dates))
        date_set   = [datetime.strptime(date.replace("00","2000"), '%m/%d/%Y') for date in date_set] 
        date_set   = sorted(date_set)
        
        for i in range(0,num_topics):
            self.topic_ts.append([0.0]*len(date_set))
            
        # 2a. get list of pi for all docs
        docs_pi = self.lda_model.lda_model.get_document_topics(self.lda_model.corpus)

         # 2b. iterate through topic coverages for each doc
        for doc_num in range(0,len(docs_pi)): 
            doc_pi = docs_pi[doc_num]

            date   = datetime.strptime(self.dates[doc_num].replace("00","2000"), '%m/%d/%Y')
            date_index = date_set.index(date)

            for topic,coverage in doc_pi: # each doc has tuple list (topic id, coverage)
                self.topic_ts[topic][date_index] = self.topic_ts[topic][date_index] + coverage

    # perform granger to get top topics
    def granger(self):
        # get dates
        dates = self.getDates(self.iem_data)

        # get sorted IEM time series data
        iem_sorted = self.sortList(self.iem_data, dates)
        
        for topic_index in range(0,len(self.topic_ts)):         
            # format iem and topic lists into dataframe
            combined_sorted = []
            for i in range(0,len(iem_sorted)):
                combined_sorted.append([iem_sorted[i],self.topic_ts[topic_index][i]])
            df = DataFrame(combined_sorted, columns = ['iem','topic'])
            granger_results = grangercausalitytests(df[['iem','topic']], maxlag=5)
            
            # get impact value
            max_granger = 0.0
            for key in granger_results.keys():
                if granger_results[key][0]['params_ftest'][1] > max_granger:
                    max_granger = granger_results[key][0]['params_ftest'][1]
            
            self.topic_impacts.append([topic_index,max_granger])

    # helper function: sort list by time
    def sortList(self,list,dates):
        data = []
        for date in dates:
            i = 0
            date_found = False
            while not date_found:
                if list[i][0] == date:
                    data.append(float(list[i][1]))
                    date_found = True
                else:
                    i = i + 1
        return data

    # helper function: get dates for time series data
    def getDates(self, list):
        dates = []
        for item in list:
            dates.append(item[0])
        return dates

    def get_granger_avg(self):
        sum   = 0.0

        for [topic,impact] in self.topic_impacts:
            sum = sum + impact
        
        return sum/len(self.topic_impacts)

class WordCausality:
    def __init__(self, top_topics, documents, dict):
        self.top_topics = top_topics # includes all topics, but has sig values
        self.sig_topics  = None # TODO: use for prior
        self.documents  = documents
        self.dict       = dict
        self.top_words  = [] # [[topic #, [word1, word2,...]],...]
        self.all_words  = [] # all words
        self.word_count_stream = []
        self.word_correlations = [] # TODO: use for prior

    def get_words(self):
        # get top topics
        topics = self.top_topics.topic_impacts
        top_topics = []
        for topic in topics:
            if topic[1] > 0.95:
                top_topics.append(topic[0])
        self.sig_topics = top_topics

        # initialize topic word list array
        for top_topic in top_topics:
            self.top_words.append([top_topic,[]])

        for top_topic in top_topics:
            topic_words = self.top_topics.lda_model.lda_model.get_topic_terms(top_topic, topn=25)
                
            for i in range(0,len(self.top_words)):
                if self.top_words[i][0] == top_topic:
                    for (word,_) in topic_words:
                        self.top_words[i][1].append(word)
                        self.all_words.append(word)

        # remove duplicates
        for i in range(0,len(self.top_words)):
            self.top_words[i][1] = list(set(self.top_words[i][1]))

        self.all_words = list(set(self.all_words))

    def get_word_stream(self):
        # initialize word stream
        for word in self.all_words:
            self.word_count_stream.append([word, []])

        for doc in self.documents:
            file_base = os.path.basename(doc).split("_")
            date = file_base[0] + "/" + file_base[1] + "/" + file_base[2].split(".")[0]
            f     = open(doc, "r")
            data  = f.read()
            
            for word in self.all_words:
                word_count = data.count(self.dict.get(word))
                # add word count (with date), to word_count_stream
                i = 0
                found_word = False
                while i < len(self.word_count_stream) and not(found_word):
                    if self.word_count_stream[i][0] == word:
                        self.word_count_stream[i][1].append([date, word_count])
                        found_word = True
                    else:
                        i = i + 1

    def get_word_sig(self):
        # get dates
        dates = self.top_topics.getDates(self.top_topics.iem_data)

        # get sorted IEM time series data
        iem_sorted = self.top_topics.sortList(self.top_topics.iem_data, dates)
            
        # get Pearson Correlation betwen word count stream and IEM time series
        for word_stream in self.word_count_stream:
            word_sorted = self.top_topics.sortList(word_stream[1], dates)
            correlation,p_val = stats.pearsonr(word_sorted, iem_sorted)
            self.word_correlations.append([word_stream[0],correlation])

    def get_word_correlations(self):
        return self.word_correlations

class PriorGeneration:
    def __init__(self, top_words, lda_model, sig, probM):
        self.top_words = top_words
        self.lda_model = lda_model
        self.sig       = sig
        self.probM     = probM
        self.pos_sorted_words = [] # [[topic,[w1,w2,...]], [topic 2, [w1,w2,...]], ...]
        self.neg_sorted_words = [] 
        self.prior            = []
        self.correlation_map  = None
        self.purity           = 0.0
    # find words from word_correlations that are in each sign. topic, and rank them by p(word|topic)
    def sort_words_per_topic(self):
        sig_topics = self.top_words.sig_topics
        word_correlations  = self.top_words.word_correlations
        pos_words = []
        neg_words = []
        
        for [word,sig] in word_correlations:
            if sig > 0.0:
                pos_words.append(word)
            else:
                neg_words.append(word)

        # store each sig(C,X,w) at word index
        self.correlation_map = [0.0]*(max(pos_words + neg_words)+1)
    
        for [word,sig] in word_correlations:
            self.correlation_map[word] = sig

        # calculate purity
        pProb = len(pos_words)/(len(pos_words) + len(neg_words))
        nProb = len(neg_words)/(len(pos_words) + len(neg_words))
        entropy = pProb*math.log(pProb, 10) + nProb*math.log(nProb, 10)
        self.purity = 100.0+100.0*entropy

        '''
        print(word_correlations)
        print(self.correlation_map)
        print(pos_words)
        print(neg_words)
        '''
        for topic in sig_topics:
            #print(topic)
            pos_words_list = []
            pos_probSum    = 0.0
            neg_words_list = []
            neg_probSum    = 0.0
            topic_terms = self.lda_model.lda_model.get_topic_terms(topic, topn = len(self.lda_model.dict))

            for (word, prob) in topic_terms:
                if word in pos_words:
                    if (pos_probSum + prob) < self.probM:
                        pos_probSum = pos_probSum + prob
                        pos_words_list.append(word)
                if word in neg_words:
                    if (neg_probSum + prob) < self.probM:
                        neg_probSum = neg_probSum + prob
                        neg_words_list.append(word)            
            self.pos_sorted_words.append([topic,pos_words_list])
            self.neg_sorted_words.append([topic,neg_words_list])
        '''
        print(self.pos_sorted_words)
        print(self.neg_sorted_words)
        '''

    def calc_prior(self):
        vocab_len = len(self.lda_model.dict)
        self.prior = np.full((vocab_len),1.0/vocab_len) 

        # 1. sum all word sig vals for each topic (for avg)
        pos_sums = []
        for [topic,words] in self.pos_sorted_words:
            sum = 0.0
            for word in words:
                sum = sum + self.correlation_map[word]-self.sig
            pos_sums.append(sum)

        neg_sums = []
        for [topic,words] in self.neg_sorted_words:
            sum = 0.0
            for word in words:
                sum = sum + self.correlation_map[word]-self.sig
            neg_sums.append((sum))

        '''
        print(self.pos_sorted_words)
        print(pos_sums)
        print(self.neg_sorted_words)        
        print(neg_sums)  
        '''

        # 2. calculate each prior
        for i in range(0,len(self.pos_sorted_words)):
            words = self.pos_sorted_words[i][1]
            sum   = pos_sums[i]
            for word in words:
                word_prior = ((self.correlation_map[word]-self.sig)/sum)
                if word_prior > self.prior[word-1]:
                    self.prior[word-1] = word_prior

        for i in range(0,len(self.neg_sorted_words)):
            words = self.neg_sorted_words[i][1]
            sum   = neg_sums[i]
            for word in words:
                word_prior = ((self.correlation_map[word]-self.sig)/sum)
                if word_prior > self.prior[word-1]:
                    self.prior[word-1] = word_prior
        
    def get_prior(self):
        return self.prior

def main():
    # create log file
    cwd = os.getcwd()
    f = open(os.path.join(cwd,"log.txt"), "w")    

    # 0a. Preprocess data
    data = PreProcessing()
    data.process_iem()
    data.process_nyt()
    data.set_dictionary()

    # 0b. Get vocab + corpus for each day
    doc_data  = data.get_data()
    dates     = data.get_dates()
    dict      = data.get_dict()
    iem_data  = data.get_iem()
    documents = data.get_nyt()

    # 0c. Define parameters & initialize LDA
    u_list  = [10,50,100,500,1000]
    tn_list = [10,20,30,40]
    prior = []

    # u test with 5 iterations
    f.write("************** Strength of Prior Test **************\n")
    print("************** Strength of Prior Test **************\n")
    for u in u_list:
        f.write("u = " + str(u) + "\n")
        print("u = " + str(u) + "\n")
        tn = 30
        lda_model = Lda(doc_data, dict, u, tn, prior)
        lda_model.set_corpus()

        for i in range(0,5):
            f.write("iteration: " + str(i) + "\n")
            print("iteration: " + str(i) + "\n")
            # 1. Build LDA
            lda_model.generate_lda()
            
            # 2. Topic Level Causality Analysis
            sig = 0.95
            top_topics = TopicCausality(lda_model, iem_data, dates, sig)
            top_topics.generate_topic_ts()
            top_topics.granger()

            # 3. Word Level Causality Analysis
            top_words = WordCausality(top_topics, documents, data.dictionary)
            top_words.get_words()
            top_words.get_word_stream()
            top_words.get_word_sig()

            # 4. Generate and update prior
            sig_cutoff = 0.9
            probM      = 0.2
            word_correlations = top_words.get_word_correlations()
            prior_gen = PriorGeneration(top_words, lda_model, sig, probM)
            prior_gen.sort_words_per_topic()
            prior_gen.calc_prior()
            lda_model.update_prior(prior_gen.get_prior())

            f.write("Average Causality Confidence: " + str(top_topics.get_granger_avg()) + "\n")
            f.write("Average Purity: " + str(prior_gen.purity) + "\n")
            f.write("Top words: \n" + lda_model.get_top_words(top_words.sig_topics) + "\n")
            f.write("-----------------------------------------------------\n")
            print("Average Causality Confidence: " + str(top_topics.get_granger_avg()) + "\n")
            print("Average Purity: " + str(prior_gen.purity) + "\n")
            print("Top words: \n" + lda_model.get_top_words(top_words.sig_topics) + "\n")
            print("-----------------------------------------------------\n")            

    # tn test with 5 iterations
    f.write("************** Number of Topics Test **************\n")
    print("************** Number of Topics Test **************\n")
    for t in tn_list:
        f.write("tn = " + str(t) + "\n")
        print("tn = " + str(t) + "\n")
        u = 50
        lda_model = Lda(doc_data, dict, u, tn, prior)
        lda_model.set_corpus()   

        for i in range(0,5):
            f.write("iteration: " + str(i) + "\n")
            print("iteration: " + str(i) + "\n")
            # 1. Build LDA
            lda_model.generate_lda()
            
            # 2. Topic Level Causality Analysis
            sig = 0.95
            top_topics = TopicCausality(lda_model, iem_data, dates, sig)
            top_topics.generate_topic_ts()
            top_topics.granger()

            # 3. Word Level Causality Analysis
            top_words = WordCausality(top_topics, documents, data.dictionary)
            top_words.get_words()
            top_words.get_word_stream()
            top_words.get_word_sig()

            # 4. Generate and update prior
            sig_cutoff = 0.9
            probM      = 0.2
            word_correlations = top_words.get_word_correlations()
            prior_gen = PriorGeneration(top_words, lda_model, sig, probM)
            prior_gen.sort_words_per_topic()
            prior_gen.calc_prior()
            lda_model.update_prior(prior_gen.get_prior())        

            f.write("Average Causality Confidence: " + str(top_topics.get_granger_avg()) + "\n")
            f.write("Average Purity: " + str(prior_gen.purity) + "\n")
            f.write("Top words: \n" + lda_model.get_top_words(top_words.sig_topics) + "\n")
            f.write("-----------------------------------------------------\n")
            print("Average Causality Confidence: " + str(top_topics.get_granger_avg()) + "\n")
            print("Average Purity: " + str(prior_gen.purity) + "\n")
            print("Top words: \n" + lda_model.get_top_words(top_words.sig_topics) + "\n")
            print("-----------------------------------------------------\n")  

    f.close()
if __name__ == '__main__':
    main()