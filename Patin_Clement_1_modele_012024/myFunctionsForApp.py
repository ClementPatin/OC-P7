# load spacy small 
import spacy
nlp = spacy.load("en_core_web_sm")



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








def load_prod_advanced_model(load_path) :
    """
    given a path, load text_vectorizer and TFLite model

    parameters :
    ------------
    load_path - string

    returns :
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
    a function to make prediction with TFLite advanced model
    parameters :
    ------------
    text_vectorizer - Text_Vectorization layer 
    interpreter - TFLite interpreter
    X - string or list like or tensor : text(s) for classification
    proba - bool : wether or not to return proba (in [0,1]) or prediction (0 or 1). By default : True

    return :
    --------
    y_prob - array : scores (or predictions if proba=False)
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