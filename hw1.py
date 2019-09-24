#!/usr/bin/python
# -*- coding: UTF-8 -*-
import math
import csv
import numpy as np
import string
import xml.etree.ElementTree as ET
import argparse

#Train = 'wm-2019-vsm-model/queries/query-test.xml'
#Model = 'wm-2019-vsm-model/model'
#Output = 'submit.csv'
#CIRB = 'wm-2019-vsm-model/CIRB010'

tfidflist = []
file_word = []
elements = []
index_invert = []

def query_reader(filename):
    all_test = []
    root = ET.parse(filename).getroot()
    for topic in root.findall( 'topic' ):
        concepts = topic.find('concepts').text
        concepts = concepts.strip('\n。')
        concepts = concepts.split('、')
        all_test.append(concepts)
    ##這裏all_test[0]的format長成這樣：['流浪狗', '流浪犬', '動物保護', '動保法', '保育', '人道', '飼主', '寵物', '棄養', '晶片', '收容所', '設施', '安樂死', '結紮', '領養', '罰鍰', '農委會', '獸醫']
    return all_test

def file_word_counter():
    FileList = Model + ''.join( '/file-list' )
    with open( FileList , 'r' , encoding='UTF-8' ) as bigfile:
        totalwordcount = 0
        totalfilecount = 0
        for filename in bigfile:
            filename = filename.split('/')
            i = 0
            filenamestring = CIRB + ''.join('/')
            for _ in filename:
                if i != 0:
                    filenamestring = filenamestring + ''.join(filename[i].strip())
                i = i + 1
                if i != len(filename):
                    filenamestring = filenamestring + ''.join('/')

            root = ET.parse(filenamestring).getroot()
            wordcount = 0
            for paragraph in root.findall( 'doc/text/p' ):
                wordcount = wordcount + len(paragraph.text) - 2
            ##減二是因為len會把前後都加進去，所以要把他們刪掉
            file_word.append( wordcount )
            totalwordcount = wordcount + totalwordcount
            totalfilecount = totalfilecount + 1
        average = totalwordcount / totalfilecount
    return average, totalfilecount

def vocabid_finder(query):
    VocabList = Model + ''.join( '/vocab.all' )
    with open( VocabList , 'r' , encoding='UTF-8' ) as queryidlist:
        counter = 0
        for queryname in queryidlist:
            queryname = queryname.strip()
            if (query == queryname):
                return counter
            counter = counter + 1
    return -1


def query2number(problem):
    intproblem = []
    for question in problem:
        intquestion = []
        #finding query number
        for query in question:
            intquery = []
            for word in query:
                vocabid = vocabid_finder(word)
                intquery.append( vocabid )
            intquestion.append( intquery )
        intproblem.append( intquestion )
    #test
#    for question in intproblem:
#        print(question)
#        for query in question:
#            print(query)
#            print(len(query))
#            print( query[1])
#            for word in query:
#                print( word)
    return intproblem

def invertfilereading():
    InvertFile = Model + ''.join( '/inverted-file' )
    FileList = Model + ''.join( '/file-list' )
    VocabList = Model + ''.join( '/vocab.all' )
    with open( InvertFile , 'r' , encoding='UTF-8' ) as file:
        probability = 0
        row = 0
        number = 1
        index_invert.append(-1)
        for line in file:
            line = line.split()
            elements.append(line)
            if probability == 0 and int(line[0]) >= number:
                for _ in range( int(line[0]) - number):
                    index_invert.append(-1)
                index_invert.append( row)
                number = int( line[0]) + 1
                probability = int( line[2])
            elif probability == 0:
                probability = int( line[2])
            else:
                probability = probability - 1
            row = row + 1
#        print( index_invert)

#    print( len(elements))
#    print( row)
#    print( number)

    return

def okapi( rocchio_state, file_id , tf , qtf , N , df , average): #f沒有除上個字數
    k1 = 1.5
    b = 0.75
    k3 = 0
    alpha = 1
    beta = 0.75
    gamma = 0.15
    dl = file_word[file_id]
    if rocchio_state == 1:
        weight = math.log( (N - df + 0.5) / (df + 0.5)) * ((k1+1)*tf / (k1*(1-b+b*dl/average)+tf)) * ((k3 +1) * qtf /(k3+qtf) * alpha + beta/100 * ((k1+1)*tf / (k1*(1-b+b*dl/average)+tf)))
    elif rocchio_state == 0:
        weight = math.log( (N - df + 0.5) / (df + 0.5)) * ((k1+1)*tf / (k1*(1-b+b*dl/average)+tf)) * ((k3 +1) * qtf /(k3+qtf) * alpha - gamma/46872 * ((k1+1)*tf / (k1*(1-b+b*dl/average)+tf)))
    else:
        weight = math.log( (N - df + 0.5) / (df + 0.5)) * ((k1+1)*tf / (k1*(1-b+b*dl/average)+tf)) * ((k3 +1) * qtf /(k3+qtf) * alpha - gamma/46872 * ((k1+1)*tf / (k1*(1-b+b*dl/average)+tf)))
    return weight
                                                  
def takeSecond(elem):
    return elem[1]

def printcsv( obj , id):
    FileList = Model + ''.join( '/file-list' )
    counter = 0
    chosen = []
    #print( obj )
    for choose in obj:
        if int(choose[1]) <= 0 :
            break
        with open( FileList , 'r' , encoding='UTF-8' ) as bigfile:
            line = bigfile.readlines()
            address = line[choose[0]]
        address = address.strip()
        address = address.lower()
        address = address.split('/')
        chosen.append( address[3] )
        print( address )
        counter = counter + 1
    out = ' '.join( chosen )
    with open(Output, 'a', newline='',encoding='UTF-8' ) as csvfile:
        csvmake = csv.writer(csvfile)
        if( id == 11):
            csvmake.writerow(['query_id', 'retrieved_docs'])
        csvmake.writerow([id,out])
    return


def rocchio_check( rocchio_file , queryid):
    for querycount in rocchio_file:
        if int(querycount[1]) == int(queryid):
            return querycount[0]
    

# checkpoint 0: 本區無意義 / checkpoint 1: 調查vocab_id點 / checkpoint 2: 收錄資料中
# possibility 0: 目前沒事 / possibbility n: 目前還有n個資料需要收錄
def tfidf( i,last_rocchio_file, problem , average, N ):
    rocchio_files = []
    queryid = 11
    for question in problem:
        #print( 'new question')
        filetime = np.zeros(50000)
        for query in question:
            rocchio_que = []
            if len(last_rocchio_file) != 0:
                rocchio_que = rocchio_check( last_rocchio_file , queryid)
            minicounter = 0
            for word in query:
                word_invert = int(index_invert[word])
                checkpoint = 1
                row_counter = 0
                for row in elements:
                    if int(row_counter) < word_invert :
                        _ = 0
                    elif checkpoint == 1:
                        possibility = int(row[2])
                        df = int(row[2])
                        checkpoint = 0
                        #print( row[0], word)
                        if (int(row[0]) == int(word)):
                            if ( minicounter+1 != len(query) and int(row[1]) == query[minicounter+1] ):
                                checkpoint = 2
                                #next = row[1]
                            elif( int(row[1]) == -1):
                                checkpoint = 2
                                #next = row[1]
                        elif (int(row[0]) > int(word)):
                            break
                    elif checkpoint == 2:
                        #print( 'check')
                        possibility = possibility - 1
                        if possibility == 0:
                            checkpoint = 1
                        file_id = int(row[0])
                        time = row[1]
                        rocchio_state = 0
                        for checkrocchio in rocchio_que:
                            if int(checkrocchio[0]) == int(file_id):
                                rocchio_state = 1
                                break
                        if i == 0:
                            rocchio_state = 2
                        weight = okapi( rocchio_state, int(file_id) ,int(time), 1/len(question), N ,int(df),average)
                        #print(file_id ,weight)
                        if weight > 0:
                            filetime[file_id] = filetime[file_id] + weight
                        #print(filetime[file_id] )
                    else:
                        possibility = possibility - 1
                        if possibility == 0:
                            checkpoint = 1
                    row_counter = row_counter + 1
                minicounter = minicounter + 1
                #print( 'word done')
            print( 'query done ')
        print( 'question done')
#        for i in range(np.size(filetime)):
#            if filetime[i] > 0:
#                print(filetime[i])
        filetime_sort = np.argsort(filetime)
        print( 'sort done')
        weight_map = []
        i = np.size(filetime) - 1
        while (i >= np.size(filetime) - 100) :
#            print(filetime_sort[i],filetime[filetime_sort[i]] )
            weight_map.append([filetime_sort[i],filetime[filetime_sort[i]]])
            i = i - 1
        rocchio_files.append( [weight_map , queryid] )
        #printcsv( weight_map , queryid)
        queryid = queryid + 1
    return rocchio_files
        
#read files
#def new():
#    with open( 'ya' , 'r' , encoding='UTF-8' ) as file:
#        for line in file:
#            line = line.split()
#            elements.append(line)
#        ele_array = np.array(elements)
#    file.close()
#    #deal with raw data
#    minicounter = 0
#    for row in len(ele_array):
#        if minicounter == 0:
#            vocab = ele_array[row][0]
#        else:
#            file = ele_array[row][0]
#            times = ele_array[row][1]
#            minicounter = minicounter - 1
#    return


def main():
    global args, use_feedback,Train,Output,Model,CIRB
    parser = argparse.ArgumentParser(description='Process argument.')
    parser.add_argument('-r', dest='use_feedback',action='store_true', help='use feedback', default=False)
    parser.add_argument('-b', dest='use_best',action='store_true', help='use best version', default=True)
    parser.add_argument('-i', dest='Train', default='wm-2019-vsm-model/queries/query-test.xml', help='The input query file.')
    parser.add_argument('-o', dest='Output', default='submit.csv', help='The output ranked list file.')
    parser.add_argument('-m', dest='Model', default='wm-2019-vsm-model/model', help='The input model directory.')
    parser.add_argument('-d', dest='CIRB', default='wm-2019-vsm-model/CIRB010', help='The directiry of NTCIR documents.')
    args = parser.parse_args()
    use_feedback = args.use_feedback
    use_feedback = not args.use_best
    Train = args.Train
    Output = args.Output
    Model = args.Model
    CIRB = args.CIRB
    print( use_feedback)
    args = parser.parse_args()
    print ( ' start running ' )
    query = Train
    problem = query_reader( query )
    print ( ' query analysis complete' )
    average, N = file_word_counter()
    print ( ' file word complete' )
    problem = query2number( problem )
    print ( ' query2number complete' )
    invertfilereading( )
    print ( ' read invert file complete' )
    if use_feedback:
        rocchio_use = 3
        print( 'use rocchio' )
    else :
        rocchio_use = 1
        print( 'use best')
    
    rocchio_file = []
    for i in range(rocchio_use):
        rocchio_file = tfidf( i, rocchio_file, problem , average, N)
        print ( ' rocchio complete: '+str(i) )
    for obj in rocchio_file:
        printcsv( obj[0] , obj[1])


if __name__ == '__main__':
    main()



