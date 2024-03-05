### commun imports
# load spacy small 
import spacy
nlp = spacy.load("en_core_web_sm")

# to create scikit learn class
from sklearn.base import BaseEstimator, TransformerMixin




### EDA FUNCTIONS


def bestDtype(series):
    """
    returns the most memory efficient dtype for a given Series

    parameters :
    ------------
    series : series from a dataframe

    returns :
    ---------
    bestDtype : dtype
    """
    # imports
    import sys
    import pandas as pd
    import gc

    # create a copy()
    s = series.copy()

    # initiate bestDtype with s dtype
    bestDtype = s.dtype

    # initiate a scalar which will contain the min memory
    bestMemory = sys.getsizeof(s)

    # return "cat" or "datetime" if dtype is of kind 'O'
    if s.dtype.kind == "O":
        # return 'datetime64[ns]' if dates are detected
        if s.str.match(r"\d{4}-\d{2}-\d{2} \d{2}\:\d{2}\:\d{2}").all(axis=0):
            bestDtype = "datetime64[ns]"
        else:
            bestDtype = "category"

    # for numericals
    else:
        # test several downcasts
        for typ in ["unsigned", "signed", "float"]:
            sDC = pd.to_numeric(s, downcast=typ)
            # if downcasted Series is different, continue
            if (s == sDC).all() == False:
                continue
            # get memory
            mem = sys.getsizeof(sDC)
            # if best, update bestDtype and bestMemory
            if mem < bestMemory:
                bestMemory = mem
                bestDtype = sDC.dtype
            del sDC
            gc.collect()

    del s
    gc.collect()
    return bestDtype



def findMultipleChars(text, n):
    '''
    given string, return a list of all words containing repeated characters

    parameters :
    ------------
    text - string
    n - int : the number of times a character should be repeated to be indeed considered a repeated character
    '''
    # imports
    import re

    # find n-times repeated letters (lowercase or uppercase)
    iterator = re.finditer(r'\b\w*([a-zA-Z])\1{'+str(n-1)+',}\w*', text)
    
    # return the whole string of each
    return [match.group() for match in iterator]



def findHTML(text):
    '''
    given string, return a list of all HTML special characters

    parameters :
    ------------
    text - string
    '''
    # imports
    import re

    # find HTML 
    iterator = re.finditer(r'([&]{1}[\w\d#]+;)|(<[^>]*>)', text)
    
    # return the whole string of each
    return [match.group() for match in iterator]



### contractions - all patterns in a dict
contractionsPatternsDict = {
    "'" : r'`',
    "aight" : r"\b('aight|a'ight)\b",
    "alright" : r"\baight\b",
    "am not" : r"\bamn't\b",
    "and" : r"\b('n'|n)\b",
    "are not you" : r"\barencha\b",
    "are not" : r"\baren't\b",
    "bout" : r"\b'bout\b",
    "about" : r"\bbout\b",
    "cannot" : r"\bcan't\b",
    "captain" : r"\bcap'n\b",
    "cause" : r"\b('cause|'cuz)\b",
    "come on" : r"\bc'?mon\b",
    "because" : r"\bcause\b",
    "except" : r"\b('cept|cept)\b",
    "come on" : r"\b(cmon|c'mon)\b",
    "could have" : r"\b(could've|couldve)\b",
    "could not have" : r"\b(couldn't've|couldntve|couldnt've|couldn'tve)\b",
    "cup of" : r"\bcuppa\b",
    "dare not" : r"\b(darent|daren't|daresnt|daresn't|dasn't|dasnt)\b",
    "dammit" : r"\b(damn*[\s']?i*t)\b",
    "did not" : r"\b(didn't|didnt)\b",
    "did not you" : r"\bdintcha\b",
    "do not you" : r"\bdontcha\b",
    "does not" : r"\b(doesn't|doesnt)\b",
    "do not" : r"\b(don't|dont)\b",
    "do not know" : r"\bdunno\b",
    "do you" : r"\b(d'ye|dye|d'ya|dya)\b",
    "even" : r"\be’en\b",
    "ever" : r"\be'er\b",
    "them" : r"\b('em|em)\b",
    r"\g<1> is" : \
    r"\b(everybody|everybone|everything|he|how|it|she|somebody|someone|something|so|that|there|this|what|when|where|which|who|why)'?s\b",
    "forecastle" : r"\b(fo'c'sle|focsle|fo'csle|foc'sle)\b",
    "against" : r"\b('gainst|gainst)\b",
    "good day" : r"\b(g'day|gday)\b",
    "given" : r"\b(givn|giv'n)\b",
    "give me" : r"\b(gimme|giz|gi'z)\b",
    "going to" : r"\b(finna|gonna)\b",
    "go not" : r"\b(gon't|gont)\b",
    "got to" : r"\bgotta\b",
    "got you" : r"\bgotcha\b",
    "had not" : r"\b(hadn't|hadnt)\b",
    "had have" : r"\b(had've|hadve)\b",
    "has not" : r"\b(hasn't|hasnt)\b",
    "has to" : r"\bhasta\b",
    "have" : r"\bhav\b",
    "have not" : r"\b(haven't|havent)\b",
    "have to" : r"\bhafta\b",
    "he will" : r"\bhe'll\b",
    "hell of a" : r"\bhelluva\b",
    "yes not" : r"\b(yes'nt|yesnt)\b",
    "here is" : r"\bhere's\b",
    "how do you do" : r"\bhowdy\b",
    "how will" : r"\bhow'll\b",
    "how are" : r"\b(how're|howre)\b",
    "i would have" : r"\b([Ii]'d've|[Ii]'dve|[Ii]d've|[Ii]dve)\b",
    "i would not" : r"\b([Ii]d'nt|[Ii]'dnt|[Ii]'d'nt|[Ii]dnt)\b",
    "i would not have" : r"\b([Ii]'d'nt've|[Ii]dntve|[Ii]dnt've)\b",
    "if and when" : r"\b(if'n|ifn)\b",
    "i will" : r"\b[Ii]'ll\b",
    "i am" : r"\b([Ii]'?m)\b",
    "i am about to" : r"\bImma\b",
    "i am going to" : r"\b([Ii]'?m'?o|[Ii]'?m'?a)\b",
    "is not it" : r"\binnit\b",
    "i do not" : r"\bIon\b",
    "i have" : r"\bI've\b",
    "is not" : r"\b(is'nt|isnt)\b",
    "it would" : r"\b(it'd|itd)\b",
    "it will" : r"\b(it'll|itll)\b",
    "it is" : r"\b(it's|'tis)\b",
    "it was" : r"\b'twas\b",
    "i do not know" : r"\bIdunno\b",
    "kind of" : r"\bkinda\b",
    "let me" : r'\blemme\b',
    "let us" : r"\b(let's|lets)\b",
    "loads of" : r"\bloadsa\b",
    "lot of" : r"\blotta\b",
    "love not" : r"\b(loven't|lovent)\b",
    "madam" : r"\b(ma'am|maam)\b",
    "may not" : r"\b(mayn't|maynt)\b",
    "may have" : r"\b(may've|mayve)\b",
    "i think" : r"\bmethinks\b",
    "might not" : r"\b(mightn't|mightnt)\b",
    "might have" : r"\b(might've|mightve)\b",
    "mine is" : r"\bmine's\b",
    "must not" : r"\b(mustn't|mustnt)\b",
    "must not have" : r"\b(mustn't've|mustnt've|mustn'tve|mustntve)\b",
    "must have" : r"\bmust'?ve\b",
    "beneath" : r"\b'?neath\b",
    "need not" : r"\bneedn'?t\b",
    "and all" : r"\bnal\b",
    "never" : r"\bne'er\b",
    "of" : r"\bo'\b",
    "of the clock" : r"\bo'?clock\b",
    "over" : r"\bo'er\b",
    "old" : r"\bol'\b",
    "ought have" : r"\bought'?ve\b",
    "ought not" : r"\boughtn'?t\b",
    "ought not have" : r"\boughtn'?t'?ve\b",
    "around" : r"\b'round\b",
    "probably" : r"\bprolly\b",
    "shall not" : r"\b(shalln'?t|shan't?)\b",
    "she will" : r"\bshe'll\b",
    "should have" : r"\bshould('?ve|a)\b",
    "should not" : r"\bshouldn'?t\b",
    "should not have" : r"\bshouldn'?t'?ve\b",
    "so are" : r"\bso're\b",
    "so have" : r"\bso'?ve\b",
    "sort of" : r"\bsorta\b",
    "thank you" : r"\b(thanks|thx)\b",
    "that will" : r"\bthat'?ll\b",
    "that would" : r"\bthat'?d\b",
    "there would" : r"\bthere'?d\b",
    "there will" : r"\bthere'?ll\b",
    "there are" : r"\bthere'?re\b",
    "these are" : r"\bthese'?re\b",
    "these have" : r"\bthese'?ve\b",
    "they would" : r"\bthey'?d\b",
    "they would have" : r"\bthey'?d'?ve\b",
    "they will" : r"\bthey'?ll\b",
    "they are" : r"\bthey'?re\b",
    "they have" : r"\bthey'?ve\b",
    "those are" : r"\bthose'?re\b",
    "those have" : r"\bthose'?ve\b",
    "tomorrow" : r"\btmrw\b",
    "without" : r"\b'?thout\b",
    "until" : r"\b'?til\b",
    "used to" : r"\busta\b",
    "it is" : r"\b'?tis\b",
    "to have" : r"\bto'?ve\b",
    "trying to" : r"\btryna\b",
    "it was" : r"\b'?twas\b",
    "between" : r"\b'?tween'\b",
    "it were" : r"\b'?twere'\b",
    r"we \g<1>" : r"\bw'(all|at)\b",
    "want to" : r'\bwanna\b',
    "want to be" : r"\bwannabe\b",
    "was not" : r"\bwasn'?t\b",
    "we would" : r"\bwe'?d\b",
    "we would have" : r"\bwe'?d'?ve\b",
    "we will" : r"\bwe'll\b",
    "we are" : r"\bwe're\b",
    "we have" : r"\bwe'?ve\b",
    "were not" : r"\bweren'?t\b",
    "what are you" : r"\bwhatcha\b",
    "what did" : r"\bwhat'?d\b",
    "what will" : r"\bwhat'?ll\b",
    "what are" : r"\bwhat'?re\b",
    "what have" : r"\bwhat'?ve\b",
    "when did" : r"\bwhen'?d\b",
    "where did" : r"\bwhere'?d\b",
    "where will" : r"\bwhere'?ll\b",
    "where are" : r"\bwhere'?re\b",
    "where have" : r"\bwhere've\b",
    "which had" : r"\bwhich'?d\b",
    "which will" : r"\bwhich'?ll\b",
    "which are" : r"\bwhich'?re\b",
    "which have" : r"\bwhich'?ve\b",
    "who would" : r"\bwho'?d\b",
    "who would have" : r"\bwho'?d'?ve\b",
    "who will" : r"\bwho'?ll\b",
    "who are" : r"\bwho'?re\b",
    "who have" : r"\bwho'?ve\b",
    "why did" : r"\bwhy'?d\b",
    "why are" : r"\bwhy'?re\b",
    "will not" : r"\b(willn'?|won'?|wonno)t\b",
    "would have" : r"\bwould'?ve\b",
    "would not" : r"\bwouldn'?t\b",
    "would not have" : r"\bwouldn'?t'?ve\b",
    "you are not" : r"\by'?ain'?t\b",
    "you all" : r"\by'?all\b",
    "you all would have" : r"\by'?all'?d'?ve\b",
    "you all would not have" : r"\by'?all'?dn'?t'?ve\b",
    "you all are" : r"\by'?all'?re\b",
    "you all are not" : r"\by'?all'?ren'?t\b",
    "you at" : r"\by'?at\b",
    "yes madam" : r"\byes'?m\b",
    "have you ever" : r"\by'?ever\b",
    "you know" : r"\by'?know\b",
    "your" : r"\b(yer|ur)\b",
    "yes sir" : r"\byessir\b",
    "you would" : r"\b(you|u)'?d\b",
    "you will" : r"\b(you|u)'?ll\b",
    "you are" : r"\b(you|u)'?re\b",
    "you have" : r"\b(you|u)'?ve\b",
    "you" : r"\bu\b",
    
}


def findContractions(text):
    '''
    given string, return a list of all contractions

    parameters :
    ------------
    text - string
    '''
    # imports
    import re

    # find contractions using contractionsPatternsDict
    out = []
    
    for pat in contractionsPatternsDict.values() :
        iterator = re.finditer(pat, text)
        
        # return the whole string of each
        out = out + [match.group() for match in iterator]

    return out



def findURL(text):
    '''
    given string, return a list of all url

    parameters :
    ------------
    text - string
    '''
    # imports
    import re

    # find URL 
    iterator = re.finditer(r'(https?:[^\s]+)|(www\.[^\s]+)', text)
    
    # return the whole string of each
    return [match.group() for match in iterator]



def findEmails(text):
    '''
    given string, return a list of all email adresses. 
        
    parameters :
    ------------
    text - string
    '''
    # imports
    import re

    # find email adresses 
    pattern = r'([a-zA-Z0-9._%-]+[a-zA-Z0-9]@([a-zA-Z0-9-]+[a-zA-Z0-9])+\.(([a-z]{2,})|([A-Z]{2,})))'
    iterator = re.finditer(pattern, text)
    
    # return the whole string of each
    return [match.group() for match in iterator]




def findEscapeSequences(text):
    '''
    given string, return a list of all escape sequences

    parameters :
    ------------
    text - string
    '''
    # imports
    import re

    # find escape sequences 
    iterator = re.finditer(r'[\n]|[\r]|[\a]|[\b]|[\\]|[\f]|[\t]|[\v]', text)
    
    # return the whole string of each
    return [match.group() for match in iterator]



def findEmoticons(text):
    '''
    given string, return a list of all emoticons

    parameters :
    ------------
    text - string
    '''
    # imports
    import re
    import emot

    # create emot object
    emot_obj = emot.emot()

    # find emoticons meanings
    emot_text = emot_obj.emoticons(text)
    meaningsList = emot_text["mean"]
    symbolsList = emot_text["value"]

    # keep only the first meaning
    meaningsList = [re.findall(r"^\w*", meaning)[0] for meaning in meaningsList]

    # put symbols and meanings together, first as tuples
    out = list(zip(symbolsList, meaningsList))
    # then in the same string
    out = [pair[0]+" - "+pair[1] for pair in out]
    return out



def findHashtags(text):
    '''
    given string, return a list of all hashtags

    parameters :
    ------------
    text - string
    '''
    # imports
    import re

    # find hashtags 
    iterator = re.finditer(r"(^#\w+|\s#\w+)(#\w+)*", text)
    
    # return the whole string of each
    return [match.group().strip() for match in iterator]




def findMentions(text):
    '''
    given string, return a list of all mentions

    parameters :
    ------------
    text - string
    '''
    # imports
    import re

    # find mentions 
    iterator = re.finditer(r"(^@\w+|\s@\w+)(@\w+)*", text)
    
    # return the whole string of each
    return [match.group().strip() for match in iterator]




def plotWordCloud(listOfStrings, nWordsToPLot = 250, savingFolderAndName = None) :

    '''
    from a list of strings, plot a wordcloud and save image. If image already in the given path, just load it and display

    parameters :
    ------------
    listOfStrings - list of string
    nWordsToPLot - int : the number of strings/words to display on wordcloud, randomly picked. By default : 250
    savingFolderAndName - tuple of strings : containing path and name for saving or loading
                                (path of the folder , name of the image)
                            by default : None, no saving
    output : a wordcloud
    
    '''
    # imports
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    from collections import Counter


    # handle savingFolderAndName
    if savingFolderAndName :
        savingFolderPath = savingFolderAndName[0]
        name = savingFolderAndName[1]
    else :
        savingFolderPath = name = None
        
    if name not in os.listdir(savingFolderPath) :
        # use wordcloud to look at them in a visualisation
        # count them and pick only the nWordsToPLot most frequent
        freq = Counter(listOfStrings).most_common(nWordsToPLot)

        # create our wordcloud object
        wc = WordCloud(
                width=2000,
                height=1200,
                # max_words=nWordsToPLot,
                # max_font_size=75,
                # min_font_size=20,
                background_color="white",
                colormap="cividis",
                random_state=16
                )
        # fit our wordcloud
        wordCloud = wc.fit_words(dict(freq))
        # save
        if savingFolderAndName :
            wordCloud.to_file(savingFolderPath+"/"+name)
    
    else :
        wordCloud = plt.imread(savingFolderPath+"/"+name)
    
        
    # plot using matplotlib imshow
    plt.figure(figsize=(14,14*1200/2000))
    plt.imshow(wordCloud)
       
    
    # no axis
    plt.axis(False)
    
    plt.show()






### DATA LOADING FUNCTIONS






def removeRows (df) :
    '''
    given a dataframe containing "text" and "target" :
        - drop all duplicated observations of "text" with different "target" values
        - drop only duplicates for normal ones (with no incoherency in "target")
        - delete rows where "text" contains only mentions "@USER"
        - delete rows where "text" contains non "printable" ASCII characters


    parameters :
    ------------
    df - DataFrame : a "text" (string) and a "target" (int or bool)

    return :
    --------
    outDf - DataFrame : with rows removed
    '''

    # imports
    import gc
    import pandas as pd
    
    # search for duplicated tweets
    # first create a mask
    mask = df["text"].duplicated(keep=False)
    # filter
    duplicates = df.loc[mask].sort_values(by="text")
    
    # first search for duplicated texts with different "target" values
    duplicatesGB = duplicates.groupby(by="text")["target"].nunique() > 1
    wrongTexts  = duplicatesGB.loc[duplicatesGB == True].index
    wrongIdx = df.loc[df["text"].isin(wrongTexts)].index
    # drop all of them
    outDf = df.drop(index=wrongIdx)
    
    # second, drop normal duplicates
    outDf.drop_duplicates(subset="text", keep="first", inplace=True)

    # deal with "only mentions" or "non printable" tweets
    mask = (outDf["text"].str.contains(r'^(?:@\w+\s*)+$')) | (outDf["text"].str.contains(r"[^\u0000-\u007F]+"))
    outDf = outDf.loc[~mask]

    del mask, wrongTexts, wrongIdx
    gc.collect()

    return outDf





def loadTweetsAndFilter (tweetsPath, nSamples = None, random_state=16) :
    '''
    given a path, load the "training.1600000.processed.noemoticon.csv" dataset (or a sample) and use "removeRows" function

    parameters :
    ------------
    tweetsPath - string
    nSamples - int : the number of samples. By default : None (no sampling)
    random_state - int : for sampling. By default : 16

    return :
    --------
    X - Series : text (the tweet)
    y - Series : target
    '''

    # imports
    import pandas as pd
    import gc
    import numpy as np

    # load "Sentiment140" tweets dataset
    df = pd.read_csv(
        filepath_or_buffer="dataset/training.1600000.processed.noemoticon.csv", 
        encoding="ISO-8859-1",
        usecols=[0,5],
        names=["target","text"]
    )

    # replace 4 with 1
    df["target"] = df["target"].replace(to_replace={4:0,0:1})

    # handle nSamples
    if not nSamples :
        nSamples = len(df)

    # first, sample with more than asked data
    nSamplesPrelim = min(len(df), int(nSamples*1.05))
    df = df.sample(nSamplesPrelim, random_state=random_state)

    # use removeRows function
    df = removeRows(df)

    # sample
    outDf = pd.DataFrame(columns=df.columns)
    # compute the number of samples for each class, to maintain class balanceness
    n0 = (df["target"] == 0).sum()
    n1 = (df["target"] == 1).sum()
    nSamplesPerClass = min(int(np.floor(nSamples/2)), n0, n1)

    # filter and sample
    for label in [0,1] :
        # filter on that label
        filteredDf = df.loc[df["target"] == label]
        # sample
        filteredDf = filteredDf.sample(nSamplesPerClass, random_state=random_state)
        # concat
        outDf = pd.concat([outDf,filteredDf])
        

    # sort index
    outDf = outDf.sort_index()
    # return
    X = outDf["text"]
    y = outDf["target"]

    # adjust y dtype
    y = y.astype(int)
    y = y.astype(bestDtype(y))

    return X, y 







### CLEANING / PREPROCESSING FUNCTIONS






def replaceSTH(textSeries, pattern, replace = "") :
    '''
    given a text Series and a regex pattern, replace matches with a given string 

    parameters :
    ------------
    textSeries - Series : each value is a text document (string type)
    pattern - raw string : regex pattern
    replace - string : to replace the matches. By default : "" (simply remove matches)

    return :
    --------
    out - Series : the same but without HTML special characters codes and tags 
    
    '''
    # imports 
    import pandas as pd
    import re

    # remove HTML special characters codes and tags 
    out = textSeries.apply(lambda doc : re.sub(pattern,replace,doc))
    return out






def dropDuplicatedChars(text, n, keepDouble=False):
    '''
    given a string, remove repeated characters

    parameters :
    ------------
    text - string
    n - int : the number of times a character should be repeated to be indeed considered a repeated character
    keepDouble : wether or not to leave to 2 chars in place of 1

    return :
    --------
    out - string : the same string, but without repeated chars
    
    
    '''
    # imports
    import re
    # handle keepDouble
    if keepDouble :
        repl = r'\1\1'
    else :
        repl = r'\1'

    # replace "n-times" repeated chars
    out = re.sub(
        pattern = r'([A-Z]|[a-z]|[0-9])\1{'+str(n-1)+',}',
        repl = repl,
        string = text
                )
    
    return out



def correctRepeatedChars_text(text) :
    '''
    given a text, remove repeated characters 

    parameters :
    ------------
    text - string 

    return :
    --------
    out - string : the same but with words with repeated characters corrected
    
    '''

    # imports
    import spacy
    from nltk.corpus import wordnet
    import re

    # find words with 3+ repeated chars using findMultipleChars
    find = findMultipleChars(text,3)

    # if no repeated chars, return text
    if len(find) == 0 :
        return text

    else :
        # initiate a dict with corrections
        corrections = {}

        # find a correction for each word with duplicated chars
        for weirdWord in find :
            # possible solutions to test
            possibleSolutions = [
                weirdWord, # maybe the repeated chars are normal
                dropDuplicatedChars(weirdWord, 2), # remove all double chars
                dropDuplicatedChars(weirdWord, 3), # remove all triple chars
                dropDuplicatedChars(weirdWord, 3, keepDouble=True) # remove all triple chars, but keep 2
            ]

            for word in possibleSolutions :
                # if word is in wordnet, keep it
                if word.lower() in wordnet.words() :
                    corrections[weirdWord] = word
                    break
                # if not, check its lemma
                else :
                    # compute lemma
                    lemma = nlp(word.lower())[0].lemma_
                    # if lemma is in wordnet, keep it
                    if lemma in wordnet.words() :
                        corrections[weirdWord] = lemma
                        break

            # if nothing has been found, just keep the "remove all double chars" solution
            if weirdWord not in corrections.keys() :
                corrections[weirdWord] = possibleSolutions[1]

        # replace each word with repeated chars with its correction
        for pb,corr in corrections.items() :
            text = re.sub(
                pattern = pb,
                repl = corr,
                string = text
                )
        return text




def correctRepeatedChars_series (textSeries) :
    '''
    given a text Series, correct words with repeated characters using correctRepeatedChars_text function

    parameters :
    ------------
    textSeries - Series : each value is a text document (string type)

    return :
    --------
    correctedSeries - Series : the same but with words corrected
    
    '''
    # imports 
    import pandas as pd
    
    # use correctRepeatedChars_text function to correct these words
    correctedSeries = textSeries.apply(lambda doc : correctRepeatedChars_text(doc))
    
    return correctedSeries





def correctContractions(textSeries):
    '''
    given a text Series, correct contractions using the dictionary contractionsPatternsDict

    parameters :
    ------------
    textSeries - Series : each value is a text document (string type)

    return :
    --------
    correctedSeries - Series : the same but with contractions corrected

    
    '''
    # imports
    import re

    # patterns


    # initiate correctedSeries
    correctedSeries = textSeries.copy()

    # iterate on each contraction and its correction using contractionsPatternsDict
    for patternCorr, patternCont in contractionsPatternsDict.items() :
        correctedSeries = replaceSTH(
            textSeries = correctedSeries,
            pattern = patternCont,
            replace = patternCorr
        )
        

    return correctedSeries





def replaceEmoticons(textSeries):
    '''
    given a text string, replace emoticons with their meaning using replaceEmoticons_text function

    parameters :
    ------------
    textSeries - Series : each value is a text document (string type)

    return :
    --------
    correctedSeries - Series : the same but with contractions corrected

    
    '''
    # imports
    import re
    import emot
    import pandas as pd
    import gc

    # create emot object
    emot_obj = emot.core.emot()

    # find emoticons meanings and put resutlt in a dataframe
    emotDf = pd.DataFrame(emot_obj.bulk_emoticons(textSeries), index = textSeries.index)
    # add textSeries
    emotDf["text"] = textSeries
    
    # # keep only the first word in meaning (lowercase)
    # emotDf["mean_short"] = emotDf["mean"].apply(lambda l : [re.findall(r"^\w*", meaning)[0].lower() for meaning in l])
    
    # # correct
    # def correct(r) :
    #     corr = r["text"]
    #     for emoticon,mean in zip(r["value"],r["mean_short"]) :
    #         corr = re.sub(r'('+emoticon+')',mean,corr) 
    #     return corr

    # replace emoticon with meaning using its location start and end 
    def correct(r) :
        corr = r["text"]
        for location,mean in zip(r["location"],r["mean"]) :
            start, end = location[0], location[1]
            corr = corr[:start] + mean.lower() + corr[end:]
        return corr
    
    correctedSeries = emotDf.apply(correct, axis=1)
    correctedSeries.name = "text"
    del emotDf
    gc.collect()
    
    return correctedSeries






def generalClean (X, lowercase = True, url = " ", email = " ", hashtag = " ", mention = " ") : 
    '''
    given a text Series X :
        - lowercase
        - remove HTML special characters codes and tags
        - replace url with a string
        - replace email adresses with a string
        - remove escape sequences
        - correct words with 3+ multiple characters
        - correct contractions
        - replace emoticons with their meaning 
        - replace hashtags "#xxxxx" with a string
        - replace mentions "@xxxxx" with a string
        - strip

    parameters :
    ------------
    X - Series : each value is a text document (string type)
    lowercase - bool : wether or not to lowercase. By default : True
    url - string : url replacement. By default : ""
    email - string : email adress repalcement. By default : ""
    hashtag - string : hashtag replacement. By default : ""
    mention - string : mention replacement. By default : ""

    return :
    --------
    outX - Series : the same but without url
    
    '''

    # imports
    import gc
    import pandas as pd

    # create a copy()
    outX = X.copy()
    
    # lowercase
    if lowercase :
        outX = outX.str.lower()

    # deal with cleaning requiring only the replaceSTH (i.e. only a regex pattern)
    # create a dictionary with treatment as keys and as values : dict with patterns and replacement strings
    treatmentDict = {
        treatment : {"pattern" : pat, "replace" : rep}
        for treatment,pat,rep in zip(
            [
                "removeHTMLs", 
                "replaceUrls", 
                "replaceEmails", 
                "removeEscapesSequences", 
                "replaceHashtags",
                "replaceMentions",

            ],
            [
                r'([&]{1}[\w\d#]+;)|(<[^>]*>)', 
                r'(https?:[^\s]+)|(www.[^\s]+)', 
                r'([a-zA-Z0-9._%-]+[a-zA-Z0-9]@([a-zA-Z0-9-]+[a-zA-Z0-9])+\.(([a-z]{2,})|([A-Z]{2,})))',
                r'[\n]|[\r]|[\a]|[\b]|[\\]|[\f]|[\t]|[\v]',
                r'(^#\w+|\s#\w+)(#\w+)*',
                r'(^@\w+|\s@\w+)(@\w+)*'
                
                
            ],
            [
                " ", 
                url, 
                email, 
                " ",
                hashtag,
                mention,  
            ],
        )}
    
    # use replaceSTH for each of them
    for replaceDict in treatmentDict.values() :
        outX = replaceSTH(textSeries=outX, pattern=replaceDict["pattern"], replace=replaceDict["replace"])
    
    # then, cleaning requirering corrections
    for correctionFunc in [correctRepeatedChars_series, correctContractions, replaceEmoticons] :
        outX = correctionFunc(outX)

    # strip
    outX = outX.str.strip()

    return outX







### create a class for  generalClean function (for scikit learn pipeline)
class generalCleaner (BaseEstimator, TransformerMixin) :
    '''
    For scikit learn pipeline, a tweets cleaner using the generalClean function
    '''
    def __init__(self, lowercase = True, url = " ", email = " ", hashtag = " ", mention = " ") :
        '''
        create the cleaner
        '''
        self.lowercase = lowercase
        self.url = url
        self.email = email
        self.hashtag = hashtag
        self.mention = mention

    def fit(self, X , y = None) :
        '''
        for scikit learn pipeline compatibility
        '''
        return self

    def transform(self, X) :
        '''
        for scikit learn pipeline compatibility, call generalClean function
        '''
        return generalClean(X=X, lowercase=self.lowercase, url=self.url, email=self.email, hashtag=self.hashtag, mention=self.mention)









def simpleModelClean (X, customStopWords = [], posToKeep = ["NOUN","ADJ","VERB"], normalization = None ) : 
    '''
    given a text Series :
        - apply lower case
        - remove stopwords (spacy "en_core_web_sm")
        - remove custom stopwords
        - remove punctuations
        - remove spaces
        - remove numbers and number-likes
        - filter on part-of-speech (POS) tagging
        - lemmatize or stemmatize

    parameters :
    ------------
    X - Series : each value is a text document (string type)
    customStopWords - list of strings : list of custom stop-words to remove.By default : [], no removal
    posToKeep - list of strings : list of pos tags to keep (Spacy likes). By default : ["NOUN","ADJ","VERB"]
    normalization - string : one of "lem", "stem", None. By default : None, no normalization

    return :
    --------
    outX - Series of strings : the same one, after cleaning
    
    '''

    # imports
    import pandas as pd
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    # spacy and its "en_core_web_sm" are already loaded (nlp)

    # create an output list for modified texts
    outX = []

    # handle the normalization option
    # first for lemmatization
    if normalization == "lem" :
        # pass text through spacy pipeline
        for doc in nlp.pipe(
            texts=X.str.lower().values,
            disable=['ner']
            ) :
            
            # get rid of stopwords, puntuation, spaces, digits, num-likes and custom stopwords
            # and lemmatize
            tokens = [
                token.lemma_.lower() for token in doc \
                if \
                (
                    not \
                    token.is_stop
                    | token.is_punct
                    | token.is_space
                    | token.is_digit
                    | token.like_num
                )
                and
                (
                    token.pos_ in posToKeep
                )
                and
                (
                    token.lemma_ not in customStopWords
                )
            ]
            # append this doc to the list
            outX.append(" ".join(tokens))

    # then for stemmatization or no normalization
    else :
        # pass text through spacy pipeline
        for doc in nlp.pipe(
            texts=X.str.lower().values,
            disable=['ner']
            ) :
            
            # get rid of stopwords, puntuation, spaces, digits, num-likes and custom stopwords
            # just keep .text (no .lemma_ lemmatization)
            tokens = [
                token.text for token in doc \
                if \
                (
                    not \
                    token.is_stop
                    | token.is_punct
                    | token.is_space
                    | token.is_digit
                    | token.like_num
                )
                and
                (
                    token.pos_ in posToKeep
                )
                and
                (
                    token.lemma_ not in customStopWords
                )
            ]
            # handle stematization if needed
            if normalization == "stem" :
                tokens = [ps.stem(token) for token in tokens]
            # append this doc to the list
            outX.append(" ".join(tokens))
    
    
    # as a Series
    outX = pd.Series(outX, index = X.index)
    
    return outX



### create a class for  simpleModelCleane function (for scikit learn pipeline)
class simpleModelCleaner (BaseEstimator, TransformerMixin) :
    '''
    For scikit learn pipeline, a tweets cleaner using the simpleModelClean function
    '''
    def __init__(self, customStopWords = [], posToKeep = ["NOUN","ADJ","VERB"], normalization = None) :
        '''
        create the cleaner
        '''
        self.customStopWords = customStopWords
        self.posToKeep = posToKeep
        self.normalization = normalization

    def fit(self, X , y = None) :
        '''
        for scikit learn pipeline compatibility
        '''
        return self

    def transform(self, X) :
        '''
        for scikit learn pipeline compatibility, call simpleModelClean function
        '''
        return simpleModelClean(X=X, customStopWords=self.customStopWords, posToKeep=self.posToKeep, normalization=self.normalization)







def advancedModelClean (X, normalization = None ) : 
    '''
    given a text Series :
        - apply lower case
        - remove punctuations
        - remove spaces
        - lemmatize or stemmatize

    parameters :
    ------------
    X - Series : each value is a text document (string type)
    normalization - string : one of "lem", "stem", None. By default : None, no normalization

    return :
    --------
    outX - Series of strings : the same one, after cleaning
    
    '''

    # imports
    import pandas as pd
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    # spacy and its "en_core_web_sm" are already loaded (as nlp)

    # create an output list for modified texts
    outX = []

    # handle the normalization option
    # first for lemmatization
    if normalization == "lem" :
        # pass text through spacy pipeline
        for doc in nlp.pipe(
            texts=X.str.lower().values,
            disable=['ner']
            ) :
            
            # get rid of puntuation and spaces
            # and lemmatize
            tokens = [
                token.lemma_.lower() for token in doc \
                if \
                (
                    not \
                    token.is_punct
                    | token.is_space
                )
            ]
            # append this doc to the list
            outX.append(" ".join(tokens))

    # then for stemmatization or no normalization
    else :
        # pass text through spacy pipeline
        for doc in nlp.pipe(
            texts=X.str.lower().values,
            disable=['ner']
            ) :
            
            # get rid of stopwords, puntuation, spaces, digits, num-likes and custom stopwords
            # just keep .text (no .lemma_ lemmatization)
            tokens = [
                token.text for token in doc \
                if \
                (
                    not \
                    token.is_punct
                    | token.is_space
                )
            ]
            # handle stematization if needed
            if normalization == "stem" :
                tokens = [ps.stem(token) for token in tokens]
            # append this doc to the list
            outX.append(" ".join(tokens))
    
    
    # as a Series
    outX = pd.Series(outX, index = X.index)
    
    return outX



### create a class for  advancedModelClean function (for scikit learn pipeline)
class advancedModelCleaner (BaseEstimator, TransformerMixin) :
    '''
    For scikit learn pipeline, a tweets cleaner using the advancedModelClean function
    '''
    def __init__(self, normalization = None) :
        '''
        create the cleaner
        '''
        self.normalization = normalization

    def fit(self, X , y = None) :
        '''
        for scikit learn pipeline compatibility
        '''
        return self

    def transform(self, X) :
        '''
        for scikit learn pipeline compatibility, call advancedModelClean function
        '''
        return advancedModelClean(X=X, normalization=self.normalization)






### ROC CURVE / PR CURVE FUNCTIONS






def plotROCandPRfromCV (oofProb, modelName, Xtrain, ytrain, kf, style="mean", plot_chance_level=False,palette=None, show=True) :
    '''
    
    plot ROC curve and PrecisionRecall curve for 1 given model, using out of folds scores from a cross validation
    
    parameters :
    ------------
    oofProb - probs obtained from CV
    modelName - string 
    Xtrain - array or dataframe : training data
    yTrain - array or Series : target values for training data, the ones used to obtain cvProbs
    kf - cross-validator : the one used to obtain cvProbs
    style - string : "mean" or "oof"
                        "mean" - plot global curve using the mean of each fold curve
                        "oof" - plot global curve using full oofProb and full ytrain to plot a new curve
    plot_chance_level - bool
    palette - list : list of colors. Default : None, use of seaborn "tab10" palette
    show - bool : wether or not to display the plot. By default : True
    
    output :
    --------
    display curves
    
    '''

    # imports
    from sklearn.metrics import RocCurveDisplay, auc
    from sklearn.metrics import PrecisionRecallDisplay
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    
    # create figure, both axes and palette
    fig,axs = plt.subplots(1,2,figsize=(12,7))
    axs=axs.flatten()
    if style=="mean" :
        fig.suptitle(modelName+"\nROC and Precision-Recall curves\nfrom Cross Validation\n(using the mean of folds curves)")
    if style=="oof" :
        fig.suptitle(modelName+"\nROC and Precision-Recall curves\nfrom Cross Validation\n(using full out-of-fold preds)")
    if not palette :
        palette = sns.color_palette("tab10")
    
    # create x axis values
    linFPR = np.linspace(0, 1, 1000)
    linRecall = np.linspace(0, 1, 1000)
    
    # create lists to receive...
    TPRs = [] #... folds recalls for ROC curve
    aucs=[] #... folds roc aucs
    precisions = [] # ... folds precisions for PR curve
    a_ps=[] # ... folds average precisions




    # iterate on folds
    for k_fold, (train_idx,valid_idx) in enumerate(kf.split(Xtrain,ytrain)) :

        cvProb = oofProb[valid_idx]
        yTrue = ytrain.iloc[valid_idx]


        # for each fold, plot roc curve
        vizRoc = RocCurveDisplay.from_predictions(
                                   y_true=yTrue, 
                                   y_pred=cvProb,
                                   ax=axs[0],
                                   plot_chance_level=plot_chance_level if k_fold==0 else False,
                                   name="fold "+str(k_fold),
                                   color=palette[k_fold],
                                   lw=0.3
                                  )
        # store TPR values interpolated to match our xavis values
        TPRs_interpolations = np.interp(linFPR, vizRoc.fpr, vizRoc.tpr)
        TPRs_interpolations[0]=0
        TPRs.append(TPRs_interpolations)
        # store auc
        aucs.append(vizRoc.roc_auc)


        # for each fold, plot PR curve
        vizPR = PrecisionRecallDisplay.from_predictions(
                                   y_true=yTrue, 
                                   y_pred=cvProb,
                                   ax=axs[1],
                                   plot_chance_level=plot_chance_level if k_fold==0 else False,
                                   name="fold "+str(k_fold),
                                   color=palette[k_fold],
                                   lw=0.3
                                  )
        # store precisions values interpolated to match our xavis values
        recall=vizPR.recall
        recall=np.append(recall,0)
        precision=vizPR.precision
        precision=np.append(ytrain.mean(),precision)
        precisions_interpolations = np.interp(linRecall, np.flip(recall), np.flip(precision))
        precisions_interpolations[0]=1
        precisions.append(precisions_interpolations)
        # store average precision
        a_ps.append(vizPR.average_precision)

    # for ROC curve, use the mean of stored TPRs
    if style == "mean" :
        meanTPR = np.mean(TPRs, axis=0)
        meanTPR[-1] = 1

        mean_auc = auc(linFPR, meanTPR)
        std_auc = np.std(aucs)
        # plot mean ROC curve
        axs[0].plot(
            linFPR,
            meanTPR,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=1.5,
            alpha=0.8
        )
        # plot std range
        std_TPR = np.std(TPRs, axis=0)
        TPRs_upper = np.minimum(meanTPR + std_TPR, 1)
        TPRs_lower = np.maximum(meanTPR - std_TPR, 0)
        axs[0].fill_between(
            linFPR,
            TPRs_lower,
            TPRs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
         )


    # for ROC curve, use the full out of folds scores to plot
    if style == "oof" :
        vizRocOof=RocCurveDisplay.from_predictions(
                                   y_true=ytrain, 
                                   y_pred=oofProb,
                                   ax=axs[0],
                                   plot_chance_level=False,
                                   name="full oof",
                                   color="green",
                                   lw=1.5,
                                   alpha=0.8
                                  )
        # plot std range using same method
        oofTPRs_interpolations = np.interp(linFPR, vizRocOof.fpr, vizRocOof.tpr)
        oofTPRs_interpolations[0]=0
        std_TPR = np.std(TPRs, axis=0)
        TPRs_upper = np.minimum(oofTPRs_interpolations + std_TPR, 1)
        TPRs_lower = np.maximum(oofTPRs_interpolations - std_TPR, 0)
        axs[0].fill_between(
            linFPR,
            TPRs_lower,
            TPRs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
         )


    # for PR curve, use the mean of stored precisions
    if style == "mean" :
        meanPrecision = np.mean(precisions,axis=0)
        meanPrecision[-1] = ytrain.mean()

        mean_a_p = np.sum(np.diff(linRecall) * np.array(meanPrecision)[:-1])
        std_a_p = np.std(mean_a_p)
        # plot mean PR curve
        axs[1].plot(
            linRecall,
            meanPrecision,
            color="b",
            label=r"Mean PR (AP = %0.2f $\pm$ %0.2f)" % (mean_a_p, std_a_p),
            lw=1.5,
            alpha=0.8,
        )
        # plot std range
        std_precision = np.std(precisions, axis=0)
        precisions_upper = np.minimum(meanPrecision + std_precision, 1)
        precisions_lower = np.maximum(meanPrecision - std_precision, 0)
        axs[1].fill_between(
            linRecall,
            precisions_lower,
            precisions_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
         )
    # for PR curve, use the full out of folds scores to plot
    if style == "oof" :
        vizPROof=PrecisionRecallDisplay.from_predictions(
                                   y_true=ytrain, 
                                   y_pred=oofProb,
                                   ax=axs[1],
                                   plot_chance_level=False,
                                   name="full oof",
                                   color="green",
                                   lw=1.5,
                                   alpha = 0.8
                                  )
        # plot std range using same method
        oofPrecisions_interpolations = np.interp(linRecall, np.flip(vizPROof.recall), np.flip(vizPROof.precision))
        oofPrecisions_interpolations[0]=1
        std_precision = np.std(precisions, axis=0)
        precisions_upper = np.minimum(oofPrecisions_interpolations + std_precision, 1)
        precisions_lower = np.maximum(oofPrecisions_interpolations - std_precision, 0)
        axs[1].fill_between(
            linRecall,
            precisions_lower,
            precisions_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
         )

    
    # set axis form and limits
    for ax in axs :
        ax.axis("square")
        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            )



    # set titles and legends
    axs[0].legend(loc="lower right")
    axs[1].legend(loc="best")
    if style == "mean" :
        axs[0].set_title("Mean ROC curve with variability")
        axs[1].set_title("Mean PR curve with variability")
    if style == "oof" :
        axs[0].set_title("full oof ROC curve with variability")
        axs[1].set_title("full oof PR curve with variability")

    if show :
        plt.show()
        return fig
    else :
        plt.close()
        return fig

    





# def plotROCandPRfromTEST(alreadyFittedModelsList, namesList, Xtest, ytest, plot_chance_level = False, palette = None, show = True, testSetName = "Test") :
#     '''
#     plot ROC curve and PrecisionRecall curve for given model(s), using predictions on a test cv
    
#     parameters :
#     ------------
#     alreadyFittedModelsList - list of models, or model type : a list of ALREADY fitted models (or one ALREADY model)
#     namesList - list of string, or string : a list of names, one for each model (or one model name)
#     Xtest - array or dataframe : testing data
#     ytest - array or Series : target values for testing data
#     plot_chance_level - bool : whether or not to plot random classifier curve. BY DEFAULT : False
#     palette - list : list of colors. BY DEFAULT : None (use of seaborn "tab10")
#     show - bool : wether or not to display the plot. By default : True
#     testSetName - string : name of testing set to display in the title. By default : "Test"
    
#     output :
#     --------
#     display curves
    
#     '''
    
#     # imports
#     from sklearn.metrics import RocCurveDisplay
#     from sklearn.metrics import PrecisionRecallDisplay
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     import numpy as np
#     import pandas as pd
    
#     # create figure, both axes and palette
#     fig,axs = plt.subplots(1,2,figsize=(12,6))
#     fig.suptitle("ROC and Precision-Recall curves\nfrom preds on "+testSetName+" set")
#     axs=axs.flatten()
#     if not palette :
#         palette = sns.color_palette("tab10")
    
#     # handle inputs if type is not list
#     if type(alreadyFittedModelsList)!=list :
#         alreadyFittedModelsList=[alreadyFittedModelsList]
#     if type(namesList)==str :
#         namesList=[namesList]
    
#     # add ROC and PR on respectives axes
#     for i,model in enumerate(alreadyFittedModelsList) :
#         RocCurveDisplay.from_estimator(estimator=model,
#                                        X=Xtest,
#                                        y=ytest,
#                                        ax=axs[0],
#                                        plot_chance_level=plot_chance_level if i==0 else False,
#                                        name=namesList[i],
#                                        color=palette[i],
#                                        alpha=0.8
#                                       )
#         PrecisionRecallDisplay.from_estimator(estimator=model,
#                                               X=Xtest,
#                                               y=ytest,
#                                               ax=axs[1],
#                                               plot_chance_level=plot_chance_level if i==0 else False,
#                                               name=namesList[i],
#                                               color=palette[i],
#                                               alpha=0.8
#                                              )

#     # setting ylim and legend    
#     for ax in axs :
#         ax.axis("square")
#         ax.set_ylim(-0.05,1.05)
#         ax.set_xlim(-0.05,1.05)

        
#     axs[0].legend(loc="lower right")
#     axs[1].legend(loc="best")

        
#     if show :
#         plt.show()
#         return fig
#     else :
#         plt.close()
#         return fig





def plotROCandPRfromTEST(probsList, namesList, ytest, plot_chance_level = False, palette = None, show = True, testSetName = "Test") :
    '''
    plot ROC curve and PrecisionRecall curve for given model(s), using predictions on a test cv
    
    parameters :
    ------------
    probsList - list of arrays, or arrays : a list of models array of scores (or one array of scores)
    namesList - list of string, or string : a list of names, one for each model (or one model name)
    ytest - array or Series : target values for testing data
    plot_chance_level - bool : whether or not to plot random classifier curve. BY DEFAULT : False
    palette - list : list of colors. BY DEFAULT : None (use of seaborn "tab10")
    show - bool : wether or not to display the plot. By default : True
    testSetName - string : name of testing set to display in the title. By default : "Test"
    
    output :
    --------
    display curves
    
    '''
    
    # imports
    from sklearn.metrics import RocCurveDisplay
    from sklearn.metrics import PrecisionRecallDisplay
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    
    # create figure, both axes and palette
    fig,axs = plt.subplots(1,2,figsize=(12,6))
    fig.suptitle("ROC and Precision-Recall curves\nfrom preds on "+testSetName+" set")
    axs=axs.flatten()
    if not palette :
        palette = sns.color_palette("tab10")
    
    # handle inputs if type is not list
    if type(probsList)!=list :
        probsList=[probsList]
    if type(namesList)==str :
        namesList=[namesList]
    
    # add ROC and PR on respectives axes
    for i,y_prob in enumerate(probsList) :
        RocCurveDisplay.from_predictions(
                                       y_true=ytest, 
                                       y_pred=y_prob,
                                       ax=axs[0],
                                       plot_chance_level=plot_chance_level if i==0 else False,
                                       name=namesList[i],
                                       color=palette[i],
                                       alpha = 0.8
                                      )
        PrecisionRecallDisplay.from_predictions(
                                   y_true=ytest, 
                                   y_pred=y_prob,
                                   ax=axs[1],
                                   plot_chance_level=plot_chance_level if i==0 else False,
                                   name=namesList[i],
                                   color=palette[i],
                                   alpha = 0.8
                                  )


    # setting ylim and legend    
    for ax in axs :
        ax.axis("square")
        ax.set_ylim(-0.05,1.05)
        ax.set_xlim(-0.05,1.05)

        
    axs[0].legend(loc="lower right")
    axs[1].legend(loc="best")

        
    if show :
        plt.show()
        return fig
    else :
        plt.close()
        return fig







### SIMPLE MODEL (SCIKIT LEARN) FUNCTIONS





def simplePipeCV (
    model, 
    run_name, 
    experiment_name, 
    X, 
    y, 
    kf, 
    params = {}, 
    fit_params = {}, 
    nested = False, 
    outPlot = False, 
    save_time = True, 
    return_auc=False, 
    tagStatus="candidate"
) : 
    '''
    Learning pipe to train and evaluate (using cross validation) a machine learning model (scikit learn) and track metrics (accuracy, roc auc) and plots (roc and precision recall curves).
    Use of mlflow to track :
    - params
    - metrics (accuracy, roc auc, fit time)
    - plots

    parameters :
    ------------
    model - scikit learn
    run_name - string
    experiment_name - string, the name of the mlflow experiment in which we want to save this run
    X - array like, the learning data
    y - array like, the target
    kf - KFold object, for cross validation
    params - dict : model's parameters. By default : {}, base parameters
    fit_params - dict : model's fit paramaters. By default : {}, base parameters
    nested - bool : wether or not to use a nested run
    outPlot - bool : wether or not to display the plots
    save_time - bool : wether or not to log total global fit time (involves to refit on all data). By default : True
    return_auc - bool : wether or not to return the metric roc_auc. By default : False
    tagStatus - str : set tag "status" with a value. By default : "candidate"
    '''
    
    # imports
    from sklearn.metrics import roc_auc_score, accuracy_score
    import time
    import numpy as np
    import mlflow


    # get experiment id
    exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # initiate a mlflow run
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name, nested = nested, tags={"status": tagStatus}) :

        # apply and log params
        model.set_params(**params)
        mlflow.log_params(params)
    
        # create an array to store out of fold scores
        oof_prob = np.zeros(len(y))
        
        # iterate on each fold
        for iFold, (train_index, valid_index) in enumerate(kf.split(X,y)):
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            X_valid = X.iloc[valid_index]
            y_valid = y.iloc[valid_index]
    
            # fit
            model.fit(X_train, y_train, **fit_params)
    
            # predict
            oof_prob[valid_index] = model.predict_proba(X_valid)[:,1]
    
        # metrics
        roc_auc = roc_auc_score(y, oof_prob)
        oof_pred = np.where(oof_prob >= 0.5, 1, 0)
        acc = accuracy_score(y, oof_pred)
        # log
        mlflow.log_metric(key="CV_ROC_AUC", value=roc_auc)
        mlflow.log_metric(key="CV_Accuracy", value=acc)
    
        # plot
        fig = plotROCandPRfromCV (
            oofProb = oof_prob, 
            modelName = run_name, 
            Xtrain=X, 
            ytrain=y, 
            kf=kf, 
            style="mean", 
            plot_chance_level=True, 
            palette=None, 
            show=outPlot
        ) 
        # log fig
        mlflow.log_figure(figure=fig, artifact_file="CV_roc_pr_curves.png")

        if save_time == True :
            # train on all training set 
            start_time = time.time()
            model.fit(X,y)
            fit_time = time.time() - start_time

            # log
            mlflow.log_metric("training_time", fit_time)
            
    if return_auc == True :
        return roc_auc








def simplePipeTest (model, run_name, model_name, experiment_name, X_train, y_train, X_test, y_test, params = {}, fit_params = {}, outPlot = True, 
                    save_time = True, save_model = True, tagStatus="final") : 
    '''
    Learning pipe to train and evaluate (using test data) a machine learning model (scikit learn) and track metrics (accuracy, roc auc) and plots (roc and precision recall curves).
    Use of mlflow to track :
    - params
    - metrics (accuracy, roc auc, fit time)
    - plots
    - model

    parameters :
    ------------
    model - scikit learn
    run_name - string
    model_name - string
    experiment_name - string, the name of the mlflow experiment in which we want to save this run
    X_train - array like, the learning data
    y_train - array like, the target
    X_test - array like, the test data
    y_test - array like, the target
    params - dict : model's parameters. By default : {}, base parameters
    fit_params - dict : model's fit paramaters. By default : {}, base parameters
    outPlot - bool : wether or not to display the plots. By default : True
    save_time - bool : wether or not to log total global fit time (involves to refit on all data). By default : True
    save_model - bool : wether or not to log total the model (involves to refit on all data). By default : False
    tagStatus - str : set tag "status" with a value. By default : "final"
    '''
    
    # imports
    from sklearn.metrics import roc_auc_score, accuracy_score
    import time
    import numpy as np
    import mlflow


    # get experiment id
    exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # initiate a mlflow run
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name, tags={"status": tagStatus}) :

        # apply and log params
        model.set_params(**params)
        mlflow.log_params(params)

        # fit
        start_time = time.time()
        model.fit(X_train, y_train, **fit_params)
        
        # if needed, save training time
        if save_time == True :
            fit_time = time.time() - start_time
            mlflow.log_metric("training_time", fit_time)

        # if needed, save model
        if save_model == True :
            mlflow.sklearn.log_model(
                sk_model=model, 
                artifact_path=model_name, 
                registered_model_name=model_name)

        # predict
        y_prob = model.predict_proba(X_test)[:,1]
    
        # metrics
        roc_auc = roc_auc_score(y_test, y_prob)
        y_pred = np.where(y_prob >= 0.5, 1, 0)
        acc = accuracy_score(y_test, y_pred)
        # log
        mlflow.log_metric(key="test_ROC_AUC", value=roc_auc)
        mlflow.log_metric(key="test_Accuracy", value=acc)
    
        # plot
        fig = plotROCandPRfromTEST(
            probsList=y_prob, 
            namesList=run_name, 
            ytest=y_test, 
            plot_chance_level=True, 
            palette=None, 
            show=outPlot,
            testSetName = "Test"
        )
        # log fig
        mlflow.log_figure(figure=fig, artifact_file="test_roc_pr_curves.png")
            







def fromDictParamGridToDf (dictParamGrid) :
    '''
    from a dict of "parameter_name" : [parameter_value_1, parameter_value_2, ...], create a dataframe with all combinations of parameters
    parameter :
    -----------
    dictParamGrid - dict of lists

    return :
    --------
    paramDf - dataframe, with :
                            - as many rows as combinations
                            - 1 column per parameter
                            - column "paramsStr" with all parameters in a string
                            - and column "paramsDict" with all parameters in a params dict
    '''

    # imports
    import pandas as pd

    # create a list of lists (as many list as parameters)
    # each list with length = number of combinations
    # each list containing the values of one parameters
    listOfLists=[[]]
    for v in dictParamGrid.values() :
        listOfLists=[x+[y] for x in listOfLists for y in v]

    # put each list as column in a dataframe
    paramDf=pd.DataFrame(listOfLists,columns=dictParamGrid.keys())

    # create columns "paramsStr" (all parameters of a combi in a string) and "paramsDict" (all parameters of a combi in a dict)
    paramsStr=paramDf.apply(lambda r : " / ".join(col+" = "+str(r[col]) for col in paramDf.columns),axis=1)
    paramsDict=paramDf.apply(lambda r : {col : r[col] for col in paramDf.columns},axis=1)
    paramDf["paramsStr"]=paramsStr
    paramDf["paramsDict"]=paramsDict

    return paramDf




def fromParamGridToDf (paramGrid) :
    '''
    from a dict of "parameter_name" : [parameter_value_1, parameter_value_2, ...], OR a list of these dict, create a dataframe with all combinations of parameters

    parameter :
    -----------
    paramGrid - dict of lists OR list of dict of lists

    return :
    --------
    paramDf - dataframe, with :
                            - as many rows as combinations
                            - 1 column per parameter
                            - column "paramsStr" with all parameters in a string
                            - and column "paramsDict" with all parameters in a params dict
    '''

    # imports
    import pandas as pd
    
    # if paramGrid is a list, use fromDictParamGridToDf on each
    if type(paramGrid)==list :
        paramDf = pd.concat([fromDictParamGridToDf(subGrid) for subGrid in paramGrid],axis=0).reset_index(drop=True)
        paramDf = paramDf[[col for col in paramDf if col not in  ["paramsStr","paramsDict"]]+["paramsStr","paramsDict"]]
    # else, just use fromDictParamGridToDf
    else :
        paramDf=fromDictParamGridToDf(paramGrid)

    return(paramDf)


    
# def gridSearchSimple(model, run_name_prefix, model_name, experiment_name, X, y, kf, paramsGrid = {}, fit_params = {},save = False) :
#     '''
#     '''

#     # imports
#     import mlflow
#     import pandas as pd

#     # put all parameters in a dataframe
#     paramDf = fromParamGridToDf(paramsGrid)

#     # get experiment id
#     exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

#     # initiate a mlflow run
#     with mlflow.start_run(experiment_id=exp_id, run_name=run_name_prefix+"_PARENT") as parent_run :

#         parent_id = parent_run.info.run_id
    
#         for i,combiRow in paramDf.iterrows() :
#             simplePipeCV (
#                 model=model, 
#                 run_name=run_name_prefix+"_"+combiRow["paramsStr"], 
#                 model_name=model_name, 
#                 experiment_name=experiment_name,
#                 X=X, 
#                 y=y, 
#                 kf=kf, 
#                 params = combiRow["paramsDict"], 
#                 nested=True,
#                 outPlot=False,
#                 save_time=True,
#                 return_auc=False
#             )

#     # focus on child runs
#     query = f"tags.mlflow.parentRunId = '{parent_id}'"
#     results = mlflow.search_runs(experiment_ids=[exp_id], filter_string=query)

#     # 



def optunaSimple(model, run_name_prefix, experiment_name, X, y, kf, n_trials, paramsGrid, fit_params = {}, save_time = True) :
    '''
    from a given grid of parameters, optimize a model using the optuna library and the simplePipeCV function :

    parameters :
    ------------
    model - scikit learn
    run_name_prefix - string
    experiment_name - string, the name of the mlflow experiment in which we want to save the parent run and its children
    X - array like, the learning data
    y - array like, the target
    kf - KFold object, for cross validation
    n_trials - int, the number of optuna trials
    paramsGrid - dict : {"parameter_name" : list of values}
    fit_params - dict : model's fit paramaters. By default : {}, base parameters
    save_time - bool : wether or not to log total global fit time (involves to refit on all data). By default : True

    '''

    # imports
    from sklearn.metrics import roc_auc_score, accuracy_score
    import time
    import mlflow
    import pandas as pd
    import numpy as np
    import optuna

    # create Optuna objective function
    def objective (trial) :

        paramsOptuna = {}
        for param, listOfValues in paramsGrid.items() :
            if type(listOfValues[0]) == list :
                paramsOptuna[param] = trial.suggest_categorical(param, listOfValues)
            else :
                kind = np.array(listOfValues).dtype.kind
                if kind in "OUS" or len(listOfValues) != 2 :
                    paramsOptuna[param] = trial.suggest_categorical(param, listOfValues)
                elif kind in "iu" :
                    paramsOptuna[param] = trial.suggest_int(name=param, low=min(listOfValues), high=max(listOfValues))
                elif kind in "f" :
                    paramsOptuna[param] = trial.suggest_float(name=param, low=min(listOfValues), high=max(listOfValues), log=True)

        auc = simplePipeCV (
            model=model, 
            run_name=run_name_prefix+"_"+str(trial.number), 
            experiment_name=experiment_name,
            X=X, 
            y=y, 
            kf=kf, 
            params = paramsOptuna, 
            nested=True,
            outPlot = False,
            save_time=False, # don't save fit time (preventing refit at each child run)
            return_auc=True # return auc for optuna study
        )

        return auc

    # get experiment id
    exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # initiate the mlflow parent run
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name_prefix+"_PARENT") as parent_run :

        # xxxxxxxxxxxxx
        parent_id = parent_run.info.run_id

        ## use Optuna
        # Initialize the Optuna study
        study = optuna.create_study(direction="maximize")
        # Execute the hyperparameter optimization trials.
        study.optimize(
            objective, 
            n_trials=n_trials, 
            # n_jobs=12
        )

        ## extract best params and use them
        # apply and log best params
        model.set_params(**study.best_params)
        mlflow.log_params(study.best_params)
    
        # create an array to store out of fold scores
        oof_prob = np.zeros(len(y))
        
        # iterate on each fold
        for iFold, (train_index, valid_index) in enumerate(kf.split(X,y)):
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            X_valid = X.iloc[valid_index]
            y_valid = y.iloc[valid_index]
    
            # fit
            model.fit(X_train, y_train, **fit_params)
    
            # predict
            oof_prob[valid_index] = model.predict_proba(X_valid)[:,1]
    
        # metrics
        roc_auc = roc_auc_score(y, oof_prob)
        oof_pred = np.where(oof_prob >= 0.5, 1, 0)
        acc = accuracy_score(y, oof_pred)
        # log
        mlflow.log_metric(key="CV_ROC_AUC", value=roc_auc)
        mlflow.log_metric(key="CV_Accuracy", value=acc)
    
        # plot
        fig = plotROCandPRfromCV (
            oofProb = oof_prob, 
            modelName = run_name_prefix+"_best", 
            Xtrain=X, 
            ytrain=y, 
            kf=kf, 
            style="mean", 
            plot_chance_level=True, 
            palette=None, 
            show=True
        ) 
        # log fig
        mlflow.log_figure(figure=fig, artifact_file="CV_roc_pr_curves.png")

        # log training time
        if save_time == True :
            # train on all training set 
            start_time = time.time()
            model.fit(X,y)
            fit_time = time.time() - start_time
            mlflow.log_metric("training_time", fit_time)

        # set "status" tag to "best"
        mlflow.set_tags(tags={"status" : "best_"+run_name_prefix})
        








### ADVANCED MODEL (KERAS) FUNCTIONS






def get_custom_standardize_func(lowercase=True, url="url", email="email", hashtag="hashtag", mention="mention", normalization=None) :
    '''
    create a function for keras Vectorization layer "standardize" using :
            - generalClean
            - advancedModelClean


    parameters :
    ------------
    lowercase - bool : wether or not to lowercase. By default : True
    url - string : url replacement. By default : ""
    email - string : email adress repalcement. By default : ""
    hashtag - string : hashtag replacement. By default : ""
    mention - string : mention replacement. By default : ""
    normalization - string : one of "lem", "stem", None. By default : None, no normalization

    return :
    --------
    custom_standardize - function : for "standardize" argument of keras TextVectorization layer
    
    '''
    import keras
    # def func
    @keras.saving.register_keras_serializable()
    def custom_standardize(input_data) :
        # imports
        import tensorflow as tf
        import pandas as pd
        import numpy as np

        # Skip if the layer is just initialized (no data yet)
        if tf.is_symbolic_tensor(input_data) : 
            # print("symb")
            return input_data

        else :

            # convert to numpy
            out_data = input_data.numpy()
            # array has dim (len(input),1). Flatten to reduce it
            out_data = out_data.flatten()
            # cast to pandas Series for custom function compatibility
            out_data = pd.Series(out_data)
            # decode bytes objects
            out_data = out_data.apply(lambda x : x.decode())
            # use functions
            out_data = generalClean(X=out_data, lowercase=lowercase, url=url, email=email, hashtag=hashtag, mention=mention)
            out_data = advancedModelClean(X=out_data, normalization=normalization)
            # recast to tensor
            out_data = tf.convert_to_tensor(out_data)
    
            return out_data

    return custom_standardize







def create_text_vectorizer(
    layer_name,
    lowercase = True,
    url = "url", 
    email = "email", 
    hashtag = "hashtag", 
    mention = "mention", 
    normalization = None,
    vocab_size = 10000, 
    sequence_len = 30
) :
    '''
    create a keras TextVectorization layer with a custom "standardize" callable argument using get_custom_standardize_func

    parameters :
    ------------
    layer_name - string
    lowercase - bool : wether or not to lowercase. By default : True
    url - string : url replacement. By default : ""
    email - string : email adress repalcement. By default : ""
    hashtag - string : hashtag replacement. By default : ""
    mention - string : mention replacement. By default : ""
    normalization - string : one of "lem", "stem", None. By default : None, no normalization
    vocab_size - int : Maximum size of the vocabulary for this layer. By default : 10000
    sequence_len - int : output will have its time dimension padded or truncated

    return :
    --------
    custom_standardize - function : for "standardize" argument of keras TextVectorization layer

    '''

    # imports
    from keras.layers import TextVectorization

    # create the layer
    vect_layer = TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_len,
        standardize=get_custom_standardize_func(
            lowercase=lowercase,
            url=url, 
            email=email, 
            hashtag=hashtag, 
            mention=mention, 
            normalization=normalization
            ),
        name=layer_name
        )

    return vect_layer








def get_embedding_matrix(embedding_file_path, vocabulary, verbose = False):

    '''
    from a given path to an pre-trained embedding file and a vocabulary (list of tokens), create an embedding matrix of shape (len(vocab), embedding dim)

    parameters :
    ------------
    embedding_file_path - string : path to pre-trained embedding file
    vocabulary - list of strings : list of tokens (obtained from TextVectorization ".get_vocabulary()" method for example)
    verbose - bool : wether or not to display some infos about embedding. By default : False

    return :
    --------
    embedding_matrix - array : (words index, embedding coordonates)
    
    '''
    

    # imports
    import numpy as np
    

    # initiate a dict to store the embedding
    embeddings_index = {}

    # extract embedding
    if "glove" in embedding_file_path.lower() :
        with open(embedding_file_path, encoding="utf-8") as f:
            for line in f:
                values = line.split(' ') 
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
    if "fasttext" in embedding_file_path.lower() :
        with open(embedding_file_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f :
            for line in f:
                values = line.rstrip().split(' ') 
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        

    # get the embedding dimension
    any_word = list(embeddings_index.keys())[10]
    emb_dim = embeddings_index[any_word].shape[0]

    # get word indexes from the vocabulary
    word_index = dict(zip(vocabulary, range(len(vocabulary))))

    if verbose:
        print(f'Found {len(embeddings_index)} word vectors.')
        print('embedding dimention:', emb_dim)

    # get the vocab size and add 2
    vocab_size = len(vocabulary) + 2
    embedding_matrix = np.zeros(shape=(vocab_size, emb_dim))
    if verbose:
        print('Embedding matrix dimension:',embedding_matrix.shape)
    
    # In the vocabulary, index 0 is reserved for padding
    # and index 1 is reserved for "out of vocabulary" tokens.
    # The first 2 word in the vocabulary are '', '[UNK]'
    hits = 0
    misses = 0

    # iterate on our vocabulary words and get their embedding vector (if it exists)
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1            
    if verbose:
        print("Converted %d words (%d misses)" % (hits, misses))
        
    return embedding_matrix






def create_LSTM(
    model_name, 
    seq_len,
    embedding_matrix = None,
    trainable_embedding = True,
    default_vocab_size = 10000,
    default_embedding_dim = 50,
    LSTM_units = 10, 
    LSTM_dropout = 0,
    LSTM_recurrent_dropout = 0,
    optimizer="adam",
    metrics=["Accuracy", "AUC"]
    
    
    ) :
    '''
    '''

    # imports
    from keras import Input
    from keras.models import Sequential
    from keras.layers import Embedding, Bidirectional, LSTM, Dense
    from keras import initializers


    # initiate model
    model = Sequential(name=model_name)

    # add input
    model.add(
        Input(
            shape=(seq_len,)
        )
    )
    
    # add Embedding layer
    if embedding_matrix is not None :
        input_dim = embedding_matrix.shape[0]
        output_dim = embedding_matrix.shape[1]
        embeddings_initializer = initializers.Constant(embedding_matrix)
        # weights = [embedding_matrix] # to use with the "build()" and "set_weights()" method, see below
    
    
        
    else :
        input_dim = default_vocab_size
        output_dim = default_embedding_dim
        embeddings_initializer = "uniform"
        # weights = None
        
    emb = Embedding(
            input_dim=input_dim,
            output_dim=output_dim,
            embeddings_initializer=embeddings_initializer,
            name="embedding",
            # weights=weights # doesn't work anymore in keras 3
        )
    # handle trainable embedding layer or not
    # emb.build()
    emb.trainable = trainable_embedding
    # add
    model.add(emb)

    # add Bidirectional LSTM
    model.add(
        Bidirectional(
            layer=LSTM(
                units=LSTM_units,
                dropout=LSTM_dropout,
                recurrent_dropout=LSTM_recurrent_dropout,
            ),
            name="bidirectional_LSTM"
        )
    )

    # add Dense layers
    model.add(
        Dense(
            units=16,
            activation="relu",
            name="dense16"
        ),
    )

    model.add(
        Dense(
            units=1,
            activation="sigmoid",
            name="dense1"
        )
    )



    # compile
    model.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics=metrics)
    return model








def advanced_long_pipeline_valid (
    text_vectorizer_params,
    embedding_params,
    model_params,
    run_name,
    experiment_name,
    X_train, 
    y_train, 
    X_valid,
    y_valid,
    early_stopping_patience = 4,
    fit_params = {},
    nested = False, 
    outPlot = False, 
    outSummary = False,
    save_time = True, 
    return_auc = False, 
    tagStatus="candidate"
) :
    '''
    Complete learning pipe to train and evaluate (using validation split) a deep learning model (keras) and track metrics (accuracy, roc auc) and plots (roc and precision recall curves).
    Use of mlflow to track :
    - params
    - metrics (accuracy, roc auc, fit time)
    - plots

    parameters :
    ------------
    model - keras, already compiled
    run_name - string
    experiment_name - string, the name of the mlflow experiment in which we want to save this run
    X - array like, the learning data
    y - array like, the target
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxsplit_valid_random_state - int : for train / validation split
    fit_params - dict : model's fit paramaters. By default : {}, base parameters
    nested - bool : wether or not to use a nested run. By default : False
    outPlot - bool : wether or not to display the plots. By default : False
    outSummary - bool : wether or not to display the summary of the keras model. By default : False
    save_time - bool : wether or not to log total global fit time (involves to refit on all data). By default : True
    return_auc - bool : wether or not to return the metric roc_auc. By default : False
    tagStatus - str : set tag "status" with a value. By default : "candidate"
    '''
    
    # imports
    import time
    import numpy as np
    import mlflow
    from keras.callbacks import EarlyStopping


    # get experiment id
    exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # initiate a mlflow run
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name, nested = nested, tags={"status": tagStatus}) :

        

        ## Text Vectorization
        # create a text_vectorizer and log params
        text_vectorizer = create_text_vectorizer(**text_vectorizer_params)
        mlflow.log_params(params=text_vectorizer_params)
        # .adapt() on train data but...
        # ...first, start a timer for training time (as vectorization is part of training)
        start_time = time.time()  
        text_vectorizer.adapt(X_train)
        # apply on train and validation data
        X_train_vect, X_valid_vect = text_vectorizer(X_train), text_vectorizer(X_valid)

        ## embedding matrix
        # log embedding_params
        mlflow.log_params(params=embedding_params)
        # if a path is given, create the pre-trained embedding matrix
        if embedding_params["embedding_file_path"] is not None :
            embedding_matrix = get_embedding_matrix(
                **embedding_params,
                vocabulary=text_vectorizer.get_vocabulary(),
            )
        else :
            embedding_matrix = None

        ## create the model and log its params
        mlflow.log_params(params=model_params)
        model = create_LSTM(
            **model_params,
            embedding_matrix=embedding_matrix,
            metrics=["Accuracy", "AUC"]
        )

        ## fit
        # create earlystopping
        callbacks = [
            EarlyStopping(
                monitor="val_auc",
                restore_best_weights=True,
                patience=early_stopping_patience
            )
        ]
        mlflow.log_param(key="early_stopping_patience", value=early_stopping_patience)
        # fit
        history = model.fit(
            x=X_train_vect,
            y=y_train,
            validation_data=(X_valid_vect, y_valid),
            callbacks=callbacks,
            **fit_params
        )
        mlflow.log_params(fit_params)

        ## summary
        if outSummary == True :
            model.summary(show_trainable=True)
        
        ## metrics
        fit_time = time.time() - start_time
        _, acc, roc_auc = model.evaluate(X_valid_vect, y_valid)
        # log them
        mlflow.log_metric(key="valid_ROC_AUC", value=roc_auc)
        mlflow.log_metric(key="valid_Accuracy", value=acc)
        if save_time == True : 
            mlflow.log_metric("training_time", fit_time)
            
        ## plot
        # first make validation predictions (scores)
        y_val_prob = model.predict(X_valid_vect)[:,0]
        # then use ploting func
        fig = plotROCandPRfromTEST(
            probsList=y_val_prob, 
            namesList=run_name, 
            ytest=y_valid, 
            plot_chance_level=True, 
            palette=None, 
            show=outPlot,
            testSetName = "Validation"
        )

        # log fig
        mlflow.log_figure(figure=fig, artifact_file="valid_roc_pr_curves.png")
           
    if return_auc == True :
        return roc_auc







def advanced_short_pipeline_valid (
    text_vectorizer_params,
    embedding_params,
    model_params,
    run_name,
    experiment_name,
    X_train_vect, 
    y_train, 
    X_valid_vect,
    y_valid,
    early_stopping_patience = 4,
    fit_params = {},
    nested = False, 
    outPlot = False, 
    save_time = True, 
    return_auc = False, 
    tagStatus="candidate"
) :
    '''
    Truncated learning pipe to train and evaluate (using validation split) a deep learning model (keras) and track metrics (accuracy, roc auc) and plots (roc and precision recall curves).
    Use of mlflow to track :
    - params
    - metrics (accuracy, roc auc, fit time)
    - plots

    "Truncated" because :
    - training data is supposed to be already vectorized
    - embedding matrix is supposed to be already created

    parameters :
    ------------
    model - keras, already compiled
    run_name - string
    experiment_name - string, the name of the mlflow experiment in which we want to save this run
    X - array like, the learning data
    y - array like, the target
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxsplit_valid_random_state - int : for train / validation split
    fit_params - dict : model's fit paramaters. By default : {}, base parameters
    nested - bool : wether or not to use a nested run
    outPlot - bool : wether or not to display the plots
    save_time - bool : wether or not to log total global fit time (involves to refit on all data). By default : True
    return_auc - bool : wether or not to return the metric roc_auc. By default : False
    tagStatus - str : set tag "status" with a value. By default : "candidate"
    '''
    
    # imports
    import time
    import numpy as np
    import mlflow
    from keras.callbacks import EarlyStopping


    # get experiment id
    exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # initiate a mlflow run
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name, nested = nested, tags={"status": tagStatus}) :

        

        ## for Text Vectorization and embedding, just log params
        mlflow.log_params(params=text_vectorizer_params)
        mlflow.log_params(params=embedding_params)

        ## create the model and log its params
        # carefull, not load the whole embedding matrix !
        model_params_to_log = {k:v for k,v in model_params.items() if k != "embedding_matrix"}
        mlflow.log_params(params=model_params_to_log)
        model = create_LSTM(
            **model_params,
            metrics=["Accuracy", "AUC"]
        )

        ## fit
        start_time = time.time()  
        # create earlystopping
        callbacks = [
            EarlyStopping(
                monitor="val_auc",
                restore_best_weights=True,
                patience=early_stopping_patience
            )
        ]
        mlflow.log_param(key="early_stopping_patience", value=early_stopping_patience)
        # fit
        history = model.fit(
            x=X_train_vect,
            y=y_train,
            validation_data=(X_valid_vect, y_valid),
            callbacks=callbacks,
            **fit_params
        )
        mlflow.log_params(fit_params)
    
        ## metrics
        fit_time = time.time() - start_time
        _, acc, roc_auc = model.evaluate(X_valid_vect, y_valid)
        # log them
        mlflow.log_metric(key="valid_ROC_AUC", value=roc_auc)
        mlflow.log_metric(key="valid_Accuracy", value=acc)
        if save_time == True : 
            mlflow.log_metric("training_time", fit_time)
            
        ## plot
        # first make validation predictions (scores)
        y_val_prob = model.predict(X_valid_vect)[:,0]
        # then use ploting func
        fig = plotROCandPRfromTEST(
            probsList=y_val_prob, 
            namesList=run_name, 
            ytest=y_valid, 
            plot_chance_level=True, 
            palette=None, 
            show=outPlot,
            testSetName = "Validation"
        )

        # log fig
        mlflow.log_figure(figure=fig, artifact_file="valid_roc_pr_curves.png")
           
    if return_auc == True :
        return roc_auc









def optunaShortAdvanced(
    text_vectorizer_params,
    embedding_params,
    model_params_grid,
    run_name_prefix,
    experiment_name,
    X_train, 
    y_train, 
    X_valid,
    y_valid,
    early_stopping_patience_grid,
    fit_params_grid,
    n_trials,
    save_time = True
) :
    '''
    from given grids of parameters, optimize a model using the optuna library and the advanced_short_pipeline_valid function :

    parameters :
    ------------
    text_vectorizer_params - {"parameter_name" : value} : used for create_text_vectorizer function, not part of optuna search
    embedding_params - {"parameter_name" : value} : used for get_embedding_matrix function, not part of optuna search
    model_params_grid - {"parameter_name" : list of values} : LSTM grid to test
    run_name_prefix - string
    experiment_name - string : the name of the mlflow experiment in which we want to save the parent run and its children
    X_train - array like : the learning data
    y_train - array like : the target
    X_valid - array like : the validation data
    y_valid - array like : the target of the validation data
    early_stopping_patience_grid - list of int : different values of early stopping patience, for fitting callback
    fit_params_grid - dict : {"fitting_parameter_name" : list of values}
    save_time - bool : wether or not to log training time. By default : True

    '''

    # imports
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import mlflow
    from keras.callbacks import EarlyStopping
    import pandas as pd
    import numpy as np
    import optuna

    # first, create X_train_vect and X_valid_vect
    ## Text Vectorization
    # create a text_vectorizer and log params
    text_vectorizer = create_text_vectorizer(**text_vectorizer_params)

    # .adapt() on train data but...
    # ...first, start a timer for preprocessing time (as vectorization is part of training)
    preproc_start_time = time.time()  
    text_vectorizer.adapt(X_train)
    # apply on train and validation data
    X_train_vect, X_valid_vect = text_vectorizer(X_train), text_vectorizer(X_valid)

    ## embedding matrix
    # if a path is given, create the pre-trained embedding matrix
    if embedding_params["embedding_file_path"] is not None :
        embedding_matrix = get_embedding_matrix(
            **embedding_params,
            vocabulary=text_vectorizer.get_vocabulary(),
        )
    else :
        embedding_matrix = None
    
    preproc_time = time.time() - preproc_start_time
    
    # create Optuna objective function
    def objective (trial) :
        # initiate optuna parameters grids
        model_params_grid_optuna = {}
        fit_params_grid_optuna = {}
        # for each of these...
        for grid, optuna_grid in zip([model_params_grid, fit_params_grid],[model_params_grid_optuna, fit_params_grid_optuna]) :
            # transform each list of values into a optuma trial.suggest_...(...)
            for param, listOfValues in grid.items() :
                # first handle when the param is not meant to be searched with optuna, i.e. is just one value not in a list
                if type(listOfValues) != list :
                    optuna_grid[param] = listOfValues
                else :
                    kind = np.array(listOfValues).dtype.kind
                    if kind in "OUS" or len(listOfValues) != 2 :
                        optuna_grid[param] = trial.suggest_categorical(param, listOfValues)
                    elif kind in "iu" :
                        optuna_grid[param] = trial.suggest_int(name=param, low=min(listOfValues), high=max(listOfValues), step=min(listOfValues))
                    elif kind in "f" :
                        optuna_grid[param] = trial.suggest_float(name=param, low=min(listOfValues), high=max(listOfValues), step=min(listOfValues))

        # add the embedding matrix to the model_params_grid_optuna
        model_params_grid_optuna["embedding_matrix"] = embedding_matrix
        
        # for early_stopping_patience_grid, it is not a dictionnary but just a list of params...
        early_stopping_patience_grid_optuna = trial.suggest_int(
            name="early_stopping_patience", 
            low=min(early_stopping_patience_grid), 
            high=max(early_stopping_patience_grid),
            step=1
        )

        # use advanced_short_pipeline_valid
        auc = advanced_short_pipeline_valid (
            text_vectorizer_params=text_vectorizer_params,
            embedding_params=embedding_params,
            model_params=model_params_grid_optuna,
            run_name=run_name_prefix+"_"+str(trial.number),
            experiment_name=experiment_name,
            X_train_vect=X_train_vect, 
            y_train=y_train, 
            X_valid_vect=X_valid_vect,
            y_valid=y_valid,
            early_stopping_patience=early_stopping_patience_grid_optuna,
            fit_params=fit_params_grid_optuna,
            nested=True, # we need nested runs 
            outPlot=False, # no ploting at each run
            save_time=save_time, 
            return_auc=True, # we need auc for optuna study
            tagStatus="candidate"
        )

        return auc

    # get experiment id
    exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # initiate the mlflow parent run
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name_prefix+"_PARENT") as parent_run :

        # parent_id
        parent_id = parent_run.info.run_id

        # first log the "out of optuna" params
        mlflow.log_params(params=text_vectorizer_params)
        mlflow.log_params(params=embedding_params)

        ## use Optuna
        # Initialize the Optuna study
        study = optuna.create_study(direction="maximize")
        # Execute the hyperparameter optimization trials.
        study.optimize(
            objective, 
            n_trials=n_trials, 
            # n_jobs=12
        )

        # query mlflow results
        query = f"tags.mlflow.parentRunId = '{parent_id}'"
        optunaBestResults = mlflow.search_runs(experiment_names=[experiment_name], filter_string=query)
        optunaBestResults = optunaBestResults.sort_values("metrics.valid_ROC_AUC", ascending=False)
        optunaBestResults = optunaBestResults.iloc[0,:]
        
        ## extract best params and use them
        # log best params
        mlflow.log_params(study.best_params)
        mlflow.log_params(
            {param : value for param,value in model_params_grid.items() if param not in study.best_params.keys()}
        )
    
        # log metrics
        mlflow.log_metric(key="valid_ROC_AUC", value=optunaBestResults["metrics.valid_ROC_AUC"])
        mlflow.log_metric(key="valid_Accuracy", value=optunaBestResults["metrics.valid_Accuracy"])
        if save_time == True :
            mlflow.log_metric(key="training_time", value=preproc_time + optunaBestResults["metrics.training_time"])
    
        # plot
        # load image artifact from mlflow experiment
        artifact_uri = optunaBestResults["artifact_uri"]
        print(artifact_uri)
        roc_pr_curves = mlflow.artifacts.load_image(artifact_uri + "/valid_roc_pr_curves.png")
        # create a figure and plot
        plt.figure(figsize=(14,9))
        plt.imshow(roc_pr_curves)
        # no axis
        plt.axis(False)
        plt.show()
        # log fig
        mlflow.log_image(image=roc_pr_curves, artifact_file="valid_roc_pr_curves.png")

        # set "status" tag to "best"
        mlflow.set_tags(tags={"status" : "best_"+run_name_prefix})










def advanced_long_pipeline_test (
    text_vectorizer_params,
    embedding_params,
    model_params,
    run_name,
    experiment_name,
    X_train, 
    y_train, 
    X_valid,
    y_valid,
    X_test,
    y_test,
    early_stopping_patience = 4,
    fit_params = {},
    outPlot = False, 
    outSummary = False,
    save_time = True, 
    save_model = True,
    save_path = None,
    tagStatus="final"
) :
    '''
    Complete learning pipe to train and evaluate (using test data) a deep learning model (keras) and track metrics (accuracy, roc auc) and plots (roc and precision recall curves).
    Use of mlflow to track :
    - params
    - metrics (accuracy, roc auc, fit time)
    - plots
    - model (create a Sequential "export_model" with the "text_vectorizer" and the "model", then log it)

    parameters :
    ------------
    text_vectorizer_params - dict : parameters for text_vectorizer creator
    embedding_params - dict : parameters for the creation of embedding matrix
    model_params - dict : parameters for model creator
    run_name - string
    experiment_name - string, the name of the mlflow experiment in which we want to save this run
    X_train - array like, the learning data
    y_train - array like, the target
    X_valid - array like, the learning data
    y_valid - array like, the target of validation data
    X_test - array like, the test data
    y_test - array like, the target of test data
    early_stopping_patience - int : patience of EarlyStopping, for fitting callback
    fit_params - dict : model's fit paramaters. By default : {}, base parameters
    outPlot - bool : wether or not to display the plots. By default : False
    outSummary - bool : wether or not to display the summary of the keras model. By default : False
    save_model - bool : wether or not to log the model. By default : False
    save_path - string or None : to save on local. By default, None (no saving)
    save_time - bool : wether or not to log training time. By default : True
    tagStatus - str : set tag "status" with a value. By default : "final"
    '''
    
    # imports
    import time
    import numpy as np
    import mlflow
    from keras.callbacks import EarlyStopping
    from keras.models import Sequential
    from keras import Input
    from joblib import dump


    # get experiment id
    exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # initiate a mlflow run
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name, tags={"status": tagStatus}) :

        
        ## Text Vectorization
        # create a text_vectorizer and log params
        text_vectorizer = create_text_vectorizer(**text_vectorizer_params)
        mlflow.log_params(params=text_vectorizer_params)
        # .adapt() on train data but...
        # ...first, start a timer for training time (as vectorization is part of training)
        start_time = time.time()  
        text_vectorizer.adapt(X_train)
        # apply on train and test data
        X_train_vect, X_valid_vect, X_test_vect = text_vectorizer(X_train), text_vectorizer(X_valid), text_vectorizer(X_test)

        ## embedding matrix
        # log embedding_params
        mlflow.log_params(params=embedding_params)
        # if a path is given, create the pre-trained embedding matrix
        if embedding_params["embedding_file_path"] is not None :
            embedding_matrix = get_embedding_matrix(
                **embedding_params,
                vocabulary=text_vectorizer.get_vocabulary(),
            )
        else :
            embedding_matrix = None

        ## create the model and log its params
        mlflow.log_params(params=model_params)
        model = create_LSTM(
            **model_params,
            embedding_matrix=embedding_matrix,
            metrics=["Accuracy", "AUC"]
        )

        ## fit
        # create earlystopping
        callbacks = [
            EarlyStopping(
                monitor="val_auc",
                restore_best_weights=True,
                patience=early_stopping_patience
            )
        ]
        mlflow.log_param(key="early_stopping_patience", value=early_stopping_patience)
        # fit
        history = model.fit(
            x=X_train_vect,
            y=y_train,
            validation_data=(X_valid_vect, y_valid),
            callbacks=callbacks,
            **fit_params
        )
        mlflow.log_params(fit_params)
        fit_time = time.time() - start_time

        ## summary
        if outSummary == True :
            model.summary(show_trainable=True)

        ## save model
        if save_model == True :
            # first, in order to load the "text_vectorizer" properly with correct serialization of its custom standardize function
            # save custom_standardize_func args
            custom_standardize_args = {k : v for k,v in text_vectorizer_params.items() if k in ["lowercase", "url", "email", "hashtag", "mention", "normalization"]}
            mlflow.log_dict(custom_standardize_args, "custom_standardize_args.json")
            
            # then, put "text_vectorizer" and "model" in a Sequential
            model_export = Sequential(
                [
                    Input(shape=(1,), dtype="string"),
                    text_vectorizer,
                    model,
                ],
                name = "sequential_model_export"
            )
            model_export.build()
            # compile
            model_export.compile(
                optimizer=model_params["optimizer"],
                loss = "binary_crossentropy", 
                metrics=["Accuracy", "AUC"]
            )
            if outSummary == True :
                model_export.summary(show_trainable=True)
            # signature
            signature = mlflow.models.infer_signature(model_input=X_train, model_output=model.predict(X_train_vect))
            # log with mlflow
            mlflow.tensorflow.log_model(
                model=model_export, 
                # artifact_path=run_name+"_export", 
                artifact_path="model", 
                registered_model_name=run_name+"_export",
                signature=signature
            )
            # save on disk
            if save_path is not None :
                model_export.save(save_path+run_name+".keras")
                dump(custom_standardize_args, save_path+run_name+"custom_standardize_args.joblib")
        
        ## metrics
        _, acc_val, roc_auc_val = model.evaluate(X_valid_vect, y_valid)
        _, acc_test, roc_auc_test = model.evaluate(X_test_vect, y_test)
        # log them
        mlflow.log_metric(key="valid_ROC_AUC", value=roc_auc_val)
        mlflow.log_metric(key="valid_Accuracy", value=acc_val)
        mlflow.log_metric(key="test_ROC_AUC", value=roc_auc_test)
        mlflow.log_metric(key="test_Accuracy", value=acc_test)
        if save_time == True : 
            mlflow.log_metric("training_time", fit_time)
            
        ## plot
        # first make test predictions (scores)
        y_test_prob = model.predict(X_test_vect)[:,0]
        # then use ploting func
        fig = plotROCandPRfromTEST(
            probsList=y_test_prob, 
            namesList=run_name, 
            ytest=y_test, 
            plot_chance_level=True, 
            palette=None, 
            show=outPlot,
            testSetName = "Test"
        )

        # log fig
        mlflow.log_figure(figure=fig, artifact_file="test_roc_pr_curves.png")
           
        # return text_vectorizer, model, model_export





def transformer_pipe_test(
    pretrained_model_name_or_path,
    run_name,
    experiment_name,
    X_train, 
    y_train, 
    X_valid,
    y_valid,
    X_test,
    y_test,
    early_stopping_patience = 2,
    fit_params = {},
    outPlot = False, 
    outSummary = False,
    save_time = True, 
    # save_path = None,
    tagStatus="Transformer"
) :
    '''
    Complete learning pipe to train and evaluate (using test data) a transformer model (Transformers from Huggingface) and track metrics (accuracy, roc auc) and plots (roc and precision recall curves).
    Use of mlflow to track :
    - params
    - metrics (accuracy, roc auc, fit time)
    - plots

    parameters :
    ------------
    pretrained_model_name_or_path - string : name of the bert model
    run_name - string
    experiment_name - string, the name of the mlflow experiment in which we want to save this run
    X_train - array like, the learning data
    y_train - array like, the target
    X_valid - array like, the learning data
    y_valid - array like, the target of validation data
    X_test - array like, the test data
    y_test - array like, the target of test data
    early_stopping_patience - int : patience of EarlyStopping, for fitting callback
    fit_params - dict : model's fit paramaters. By default : {}, base parameters
    outPlot - bool : wether or not to display the plots. By default : False
    outSummary - bool : wether or not to display the summary of the keras model. By default : False
    save_path - string or None : to save on local. By default, None (no saving)
    save_time - bool : wether or not to log training time. By default : True
    tagStatus - str : set tag "status" with a value. By default : "Transformer"
    '''

    # imports
    import time
    import numpy as np
    import tensorflow as tf
    import mlflow
    from keras.callbacks import EarlyStopping
    from transformers import AutoConfig
    from transformers import TFAutoModelForSequenceClassification
    from transformers import AutoTokenizer
    from keras.metrics import SparseCategoricalAccuracy
    from keras.optimizers import Adam
    from keras.activations import sigmoid
    from sklearn.metrics import roc_auc_score, accuracy_score
    

    # get experiment id
    exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # initiate a mlflow run
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name, tags={"status": tagStatus}) :

        # instantiate tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, 

            )

        args = dict(
                return_attention_mask=True,
                return_token_type_ids=False,
                padding=True,
                truncation=True,
                return_tensors="tf"
            )
        
        # get encodings
        start_time = time.time()  
        encodings_train, encodings_valid, encodings_test = tokenizer(list(X_train),**args), tokenizer(list(X_valid),**args), tokenizer(list(X_test),**args)

        # build datasets
        if fit_params["batch_size"] is not None :
            batch_size = fit_params["batch_size"]
        else :
            batch_size = 32
        train_dataset = tf.data.Dataset.from_tensor_slices((dict(encodings_train), y_train)).batch(batch_size)
        valid_dataset = tf.data.Dataset.from_tensor_slices((dict(encodings_valid),y_valid)).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((dict(encodings_test),y_test)).batch(batch_size)
        
        # instantitate config
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        config.id2label = {"0" : "POSITIVE" , "1" : "NEGATIVE"}
        config.label2id = {"NEGATIVE": 1, "POSITIVE": 0}
        config.attention_dropout = 0.4
        config.dropout = 0.4
        config.seq_classif_dropout = 0.4
        print(config)
        # instantiate model
        model = TFAutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            # num_labels=2,
            ignore_mismatched_sizes=True,
            config=config
            )
        # log
        mlflow.log_param(key="transformer_name", value=pretrained_model_name_or_path)

        # compile
        model.compile(optimizer=Adam(learning_rate=3e-5), metrics=[SparseCategoricalAccuracy("accuracy")])
        if outSummary == True :
            model.summary(show_trainable=True)

        # create earlystopping
        callbacks = [
            EarlyStopping(
                monitor="val_accuracy",
                restore_best_weights=True,
                patience=early_stopping_patience,
                mode="max"
            )
        ]
        mlflow.log_param(key="early_stopping_patience", value=early_stopping_patience)

        # fit
        model.fit(train_dataset, validation_data=valid_dataset, callbacks=callbacks, **fit_params)
        fit_time = time.time() - start_time
        mlflow.log_params(fit_params)

        # # save model on local
        # if save_path is not None :
        #     model.save_pretrained(save_path+'/clf')
        #     with open(save_path+'/info.pkl', 'wb') as f:
        #         pickle.dump((pretrained_model_name_or_path, args), f)

        # metrics
        # first get scores
        scores_val, scores_test = model.predict(valid_dataset).logits, model.predict(test_dataset).logits
        scores_val, scores_test = sigmoid(scores_val), sigmoid(scores_test)
        scores_val, scores_test = scores_val.numpy()[:,1], scores_test.numpy()[:,1]
        # preds
        preds_val, preds_test = np.where(scores_val >= 0.5, 1, 0), np.where(scores_test >= 0.5, 1, 0)
        # metrics
        roc_auc_val, roc_auc_test = roc_auc_score(y_valid, scores_val), roc_auc_score(y_test, scores_test)
        acc_val, acc_test = accuracy_score(y_valid, preds_val),  accuracy_score(y_test, preds_test)
        # log them
        mlflow.log_metric(key="valid_ROC_AUC", value=roc_auc_val)
        mlflow.log_metric(key="valid_Accuracy", value=acc_val)
        mlflow.log_metric(key="test_ROC_AUC", value=roc_auc_test)
        mlflow.log_metric(key="test_Accuracy", value=acc_test)
        if save_time == True : 
            mlflow.log_metric("training_time", fit_time)

        # then use ploting func
        fig = plotROCandPRfromTEST(
            probsList=scores_test, 
            namesList=run_name, 
            ytest=y_test, 
            plot_chance_level=True, 
            palette=None, 
            show=outPlot,
            testSetName = "Test"
        )

        # log fig
        mlflow.log_figure(figure=fig, artifact_file="test_roc_pr_curves.png")

        return model
    






def use_pipe_test(
    run_name,
    experiment_name,
    X_train, 
    y_train, 
    X_valid,
    y_valid,
    X_test,
    y_test,
    early_stopping_patience = 2,
    fit_params = {},
    outPlot = False, 
    outSummary = False,
    save_time = True, 
    # save_path = None,
    tagStatus="USE"
) :
    '''
    Complete learning pipe to train and evaluate (using test data) a model using universal sentence encoder ("https://tfhub.dev/google/universal-sentence-encoder/4" from tensorflow hub) 
    and track metrics (accuracy, roc auc) and plots (roc and precision recall curves).
    Use of mlflow to track :
    - params
    - metrics (accuracy, roc auc, fit time)
    - plots

    parameters :
    ------------
    run_name - string
    experiment_name - string, the name of the mlflow experiment in which we want to save this run
    X_train - array like, the learning data
    y_train - array like, the target
    X_valid - array like, the learning data
    y_valid - array like, the target of validation data
    X_test - array like, the test data
    y_test - array like, the target of test data
    early_stopping_patience - int : patience of EarlyStopping, for fitting callback
    fit_params - dict : model's fit paramaters. By default : {}, base parameters
    outPlot - bool : wether or not to display the plots. By default : False
    outSummary - bool : wether or not to display the summary of the keras model. By default : False
    save_path - string or None : to save on local. By default, None (no saving)
    save_time - bool : wether or not to log training time. By default : True
    tagStatus - str : set tag "status" with a value. By default : "Transformer"
    '''

    # imports
    import time
    import numpy as np
    import tensorflow as tf
    import tensorflow_hub
    import mlflow
    from keras import Sequential
    from keras.layers import Dropout, Dense
    from keras.callbacks import EarlyStopping
    

    # get experiment id
    exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # initiate a mlflow run
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name, tags={"status": tagStatus}) :

        ## model
        model = Sequential(name=run_name)
        # add USE
        use = tensorflow_hub.KerasLayer(
            handle="https://tfhub.dev/google/universal-sentence-encoder/4",
            dtype=tf.string,
            trainable=True,
            name="USE"
        )
        model.add(use)
        # add dropout
        model.add(Dropout(0.3))
        # add Embedding layer
        model.add(
            Dense(
                units=16,
                activation="relu",
                name="dense16"
            ),
            )
        # add Dense layer
        model.add(
            Dense(
                units=1,
                activation="sigmoid",
                name="dense1"
            )
            )
        # compile
        model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics=["Accuracy", "AUC"])

        # create earlystopping
        callbacks = [
            EarlyStopping(
                monitor="val_auc",
                restore_best_weights=True,
                patience=early_stopping_patience,
                mode="max"
            )
        ]
        mlflow.log_param(key="early_stopping_patience", value=early_stopping_patience)

        # fit
        start_time = time.time()
        model.fit(
                x=X_train,
                y=y_train,
                validation_data=(X_valid, y_valid),
                callbacks=callbacks
            )
        if outSummary == True :
            model.summary()
        mlflow.log_params(fit_params)

        ## metrics
        fit_time = time.time() - start_time
        _, acc_val, roc_auc_val = model.evaluate(X_valid, y_valid)
        _, acc_test, roc_auc_test = model.evaluate(X_test, y_test)
        # log them
        mlflow.log_metric(key="valid_ROC_AUC", value=roc_auc_val)
        mlflow.log_metric(key="valid_Accuracy", value=acc_val)
        mlflow.log_metric(key="test_ROC_AUC", value=roc_auc_test)
        mlflow.log_metric(key="test_Accuracy", value=acc_test)
        if save_time == True : 
            mlflow.log_metric("training_time", fit_time)

        # first make validation predictions (scores)
        scores_test = model.predict(X_test)[:,0]
        fig = plotROCandPRfromTEST(
            probsList=scores_test, 
            namesList=run_name, 
            ytest=y_test, 
            plot_chance_level=True, 
            palette=None, 
            show=outPlot,
            testSetName = "Test"
        )

        # log fig
        mlflow.log_figure(figure=fig, artifact_file="test_roc_pr_curves.png")

        return model
    






def load_mlflow_advanced_model(run_name, experiment_name, from_registry = False) :
    """
    given a run_name, load the advanced model logged in mlflow

    parameters :
    ------------
    run_name - string
    experiment_name - string
    from_registry - bool : wether or not to load the model grom model registry. By default : False (load from artifact uri using mlflow.tensorflow.load_model)

    return :
    --------
    loaded_model - mlflow model or tensorflow model
    """
    # import 
    import mlflow
    import mlflow.pyfunc
    import gc

    # filter mlflow.search_run dataframe on this name
    query = f"tags.mlflow.runName = '{run_name}'"
    results = mlflow.search_runs(experiment_names=["sentiment_analysis"], filter_string=query)
    artifact_uri = results["artifact_uri"].values[0]

    # load custom_standardize_args to serialize the custom_standardize function (used by text_vectorization layer)
    custom_standardize_args = mlflow.artifacts.load_dict(artifact_uri + "/custom_standardize_args.json")
    get_custom_standardize_func(**custom_standardize_args)

    # load model
    if from_registry == True :
        loaded_model = mlflow.pyfunc.load_model("models:/"+run_name+"_export/1")
    else :
        loaded_model = mlflow.tensorflow.load_model(artifact_uri + "/model")

    del query, results, artifact_uri, custom_standardize_args
    gc.collect()
    
    return loaded_model






def predict_with_mlflow_loaded_model(model, X, proba=True) :
    '''
    a function to make prediction with advanced model : first call text_vectorization layer, then the lstm based model
    parameters :
    ------------
    model - sequential : with two layers (text_vectorization, sequential with the lstm model)
    X - string or list like or tensor : text(s) for classification
    proba - bool : wether or not to return proba (in [0,1]) or prediction (0 or 1)
    '''
    # imports
    import numpy as np

    # handle X
    if type(X) == str :
        X = [X]
    if type(X) == list :
        X = np.array(X)

    # use text_vectorization layer
    X_temp = model.layers[0](X)

    # predict scores
    y_prob = model.layers[1].predict(X_temp)[:,0]

    # handle proba and return scores or preds
    if proba == True :
        return y_prob
    else :
        return np.where(y_prob >= 0.5, 1, 0)
    





def train_advanced_and_TFLite(
    nSamples, 
    tweetsPath,     
    text_vectorizer_params,
    embedding_params,
    model_params,
    early_stopping_patience,
    fit_params,
    save_path,
) :

    # exports
    from sklearn.model_selection import train_test_split
    from keras import Sequential
    from keras.layers import Input
    from joblib import dump
    import tensorflow as tf
    from keras.callbacks import EarlyStopping

    # load more data
    nSamples = nSamples
    # use custom function to load the dataset, just with 100000 tweets
    X_l, y_l = loadTweetsAndFilter(
        tweetsPath = tweetsPath,
        nSamples = nSamples,
        random_state=16
    )

    # split
    X_l_train, X_l_test, y_l_train, y_l_test = train_test_split(X_l, y_l, test_size=0.1, random_state=16)
    X_l_tr_train, X_l_tr_valid, y_l_tr_train, y_l_tr_valid = train_test_split(X_l_train, y_l_train, test_size=0.1, random_state=16)

    ## text_vectorizer
    text_vectorizer = create_text_vectorizer(**text_vectorizer_params)
    # adapt
    text_vectorizer.adapt(X_l_tr_train)
    # use it
    X_l_tr_train_vect, X_l_tr_valid_vect, X_l_test_vect = text_vectorizer(X_l_tr_train), text_vectorizer(X_l_tr_valid), text_vectorizer(X_l_test)
    # save it
    # first, in order to load the "text_vectorizer" properly with correct serialization of its custom standardize function
    # save custom_standardize_func args
    custom_standardize_args = {k : v for k,v in text_vectorizer_params.items() if k in ["lowercase", "url", "email", "hashtag", "mention", "normalization"]}
    dump(custom_standardize_args, save_path+"/custom_standardize_args.joblib")
    # 
    Sequential([Input(shape=(1,), dtype="string"),text_vectorizer], name="text_vectorizer_as_a_model").save(save_path+"/text_vectorizer.keras")

    ## embedding matrix
    embedding_matrix = get_embedding_matrix(**embedding_params, vocabulary=text_vectorizer.get_vocabulary())

    ## lstm model
    model = create_LSTM(**model_params, embedding_matrix=embedding_matrix, metrics=["Accuracy", "AUC"])
    # fit
    # create earlystopping
    callbacks = [
        EarlyStopping(
            monitor="val_auc",
            restore_best_weights=True,
            patience=early_stopping_patience
        )
    ]
    model.fit(
        x=X_l_tr_train_vect,
        y=y_l_tr_train,
        validation_data=(X_l_tr_valid_vect, y_l_tr_valid),
        callbacks=callbacks,
        **fit_params
    )
    # save with tf_lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # to avoid error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    # convert
    tflite_model = converter.convert()

    # save
    with open(save_path+"/ltsm_model_TFLite.tflite", 'wb') as f:
        f.write(tflite_model)







def load_prod_advanced_model(load_path) :
    """
    given a path, load text_vectorizer and TFLite model

    parameters :
    ------------
    load_path - string

    return :
    --------
    text_vectorizer, interpreter
    """
    # import 
    from joblib import load
    import tensorflow as tf
    import keras
    
    # load custom_standardize_args to serialize the custom_standardize function (used by text_vectorization layer)
    custom_standardize_args = load(load_path+"/custom_standardize_args.joblib")
    get_custom_standardize_func(**custom_standardize_args)

    # load text_vectorizer
    text_vectorizer = keras.saving.load_model(load_path+"/text_vectorizer.keras").layers[0]

    # load tflite
    interpreter = tf.lite.Interpreter(model_path=load_path+"/ltsm_model_TFLite.tflite")
    
    return text_vectorizer, interpreter





def predict_with_TFLite_loaded_model(text_vectorizer, interpreter, X, proba=True) :
    '''
    a function to make prediction with advanced model : first call text_vectorization layer, then the lstm based model
    parameters :
    ------------
    model - sequential : with two layers (text_vectorization, sequential with the lstm model)
    X - string or list like or tensor : text(s) for classification
    proba - bool : wether or not to return proba (in [0,1]) or prediction (0 or 1)
    '''
    # imports
    import numpy as np
    import tensorflow as tf

    # handle X
    if type(X) == str :
        X = [X]
    if type(X) == list :
        X = np.array(X)

    # use text_vectorization layer
    X_temp = text_vectorizer(X)
    # cast to float32 (default tflite dtype)
    X_temp = tf.cast(X_temp, dtype='float32')

    # get interpreter input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # resize interpreter input
    interpreter.resize_tensor_input(
        input_index=input_details[0]["index"], 
        tensor_size=[len(X),text_vectorizer.output_shape[1]]
        )
    # allocate
    interpreter.allocate_tensors()
    # predict scores
    interpreter.set_tensor(tensor_index=input_details[0]["index"],value=X_temp)
    interpreter.invoke()
    y_prob = interpreter.get_tensor(output_details[0]['index'])[:,0]

    # handle proba and return scores or preds
    if proba == True :
        return y_prob
    else :
        return np.where(y_prob >= 0.5, 1, 0)