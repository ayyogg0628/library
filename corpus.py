def standardize_dates(df, col):
    '''
    This function standardizes the format of date values in a specified column (col) of a DataFrame (df) to the format 'YYYY-MM-DD'. It converts date strings to datetime format.
    '''
    for i in range(len(df)):
        try: df[col][i] = datetime.datetime.strptime(str(dateutil.parser.parse(df[col][i])),'%Y-%m-%d  %H:%M:%S').strftime('%Y-%m-%d')
        except: pass
    return df

def clean_numbers(df, col, decimal_symbol=None):
    '''
    This function cleans a column (col) containing number values in a DataFrame (df). It removes non-numeric characters and converts the values to float. It also replaces the decimal symbol with a period if specified.
    '''
    for i in range(len(df[col])):
        if df[col][i] != df[col][i]:
            continue
        if decimal_symbol:
            df[col][i] = re.sub(f"[^0-9{decimal_symbol}]", "", df[col][i])
            df[col][i] = float(df[col][i].replace(decimal_symbol, '.'))
    return df

def delete_column(df, col):
    '''
    This function deletes a specified column (column_name) from a DataFrame (df) if it exists. It uses the drop() method of pandas DataFrame.
    '''
    df = df.drop([col], axis=1)
    return df

def delete_rows(df, target_col, condition):
    '''
    This function deletes rows from a DataFrame (df) based on a condition (condition) in a specified target column (target_col). It uses the drop() method of pandas DataFrame.
    '''
    indexs = df[(df[target_col]>=condition)].index
    df = df.drop(indexs)
    return df

def fill_nan_and_blanks(df, col):
    '''
    This function replaces missing values (NaN) and blanks in a specified column of a DataFrame (df) with a predefined replacement. It utilizes the fillna() method of pandas DataFrame.
    '''
    df[col] = df[col].replace('', np.nan)
    df[col] = df[col].fillna('redacted')
    return df

def correct_misspelling(df, col):
    '''
    This function corrects misspelled words in a specified column (col) of a DataFrame (df). It identifies possible misspellings, checks if the column contains only strings, and returns a DataFrame with corrected misspellings.
    '''
    word2vec_model = KeyedVectors.load('./support_docs/en_model.kv', mmap='r')
    try:
        lang = langdetect.detect(df[col][df[col].first_valid_index()])
        if lang == 'ru':
            word2vec_model = KeyedVectors.load('./support_docs/ru_model.kv', mmap='r')
    except:
        word2vec_model = KeyedVectors.load('./support_docs/en_model.kv', mmap='r')
    names_cnt = df[col].value_counts().to_dict()
    names_list = list(names_cnt.keys())
    name_correction = {}
    n_clusters = max(int(len(names_list)*0.5), 1)
    vectors = []
    for name in names_list:
        if name != name:
            vectors.append(np.zeros(shape=300))
        else:
            tok = nltk.tokenize.toktok.ToktokTokenizer()
            tokens = []
            for t in tok.tokenize(name):
                tokens.append(t.lower())
            if len(tokens) > 0:
                word_vecs = []
                for t in tokens:
                    try:
                        get_vectors = word2vec_model.get_vector(t)
                    except:
                        get_vectors = np.zeros(shape=300)
                    word_vecs.append(get_vectors)
                vec = np.sum(word_vecs, axis=0)
                vectors.append(vec)
            else:
                vectors.append(np.zeros(shape=300))    
    max_length = len(max(vectors, key=len))
    for i in range(len(vectors)):
        vectors[i] = np.resize(vectors[i], max_length)
    vectors = np.array(vectors)
    clf = KMeans(n_clusters=n_clusters, max_iter=500, random_state=0)
    labels = clf.fit_predict(vectors)
    correction = {}
    for i in range(n_clusters):
        tmp = {}
        for j in range(len(labels)):
            if labels[j]==i:
                tmp[names_list[j]] = names_cnt[names_list[j]]
        if len(tmp)>0:
            tmp = sorted(tmp.items(), key=lambda x:x[1], reverse=True)
            value, cnt = tmp[0]
            if value != value:
                for j in range(1, len(tmp)):
                    value_tmp, cnt_tmp = tmp[j]
                    if not value_tmp!=value_tmp:
                        value = value_tmp
            correction[i] = value
    for j in range(len(names_list)):
        name_correction[names_list[j]] = correction[labels[j]]
    for index, row in df.iterrows():
        name_before = row[col]
        if name_before!=name_before: continue
        df.at[index, col] = name_correction[name_before]
    return df

def compare_and_clean(df, target_col, ref_col):
    '''
    This function compares a target column (target_col) with a reference column (ref_col), and cleans the target column based on the reference. It aims to compare the values in the target column with those in the reference column and clean the target column by replacing its values with the most frequent value corresponding to the same value in the reference column.  
    '''
    name_mapping = {}
    grouped = df[[ref_col, target_col]].copy().groupby(ref_col)
    for name, group in grouped:
        if name!=name or name=='redacted':
            continue
        target_cnts = group[target_col].value_counts()
        if len(target_cnts) == 1 and (target_cnts.index[0]=='redacted' or target_cnts.index[0]!=target_cnts.index[0]):
            continue
        for i in range(len(target_cnts.index)):
            if target_cnts.index[i]!='redacted' and not target_cnts.index[i]!=target_cnts.index[i]:
                most_frequent_target = target_cnts.index[i]
                break
        for i in range(len(target_cnts.index)):
            name_mapping[name] = most_frequent_target
    for ref_name in name_mapping.keys():
        df_tmp = df[df[ref_col]==ref_name]
        df_tmp[target_col] = name_mapping[ref_name]
        df[df[ref_col]==ref_name] = df_tmp
    return df

def translate_between_columns(df, target_col, ref_col):
    '''
    This function translates the values in the target column to the reference column and vice versa, whichever is null. It uses the googletrans library to translate between english and russian.
    '''
    translator = googletrans.Translator()
    for i, row in df.iterrows():
        ref_value = row[ref_col]
        target_value = row[target_col]
        if pd.isnull(target_value) and not pd.isnull(ref_value):
            try:
                translated_value = translator.translate(ref_value, dest='ru').text
            except AttributeError:
                continue
            df.loc[i, target_value] = translated_value
        elif pd.isnull(ref_value) and not pd.isnull(target_value):
            try:
                translated_value = translator.translate(target_value, dest='en').text
            except AttributeError:
                continue
            df.loc[i, ref_col] = translated_value
    return df

def remove_corp_keywords(df, col, keyword_file, file_col, new_col):
    '''
    This function reads in a list of corporate keywords from a column (file_col) in a file (keyword_file), then removes the corporate keywords from the column and creates a new column with the removed keywords.
    '''
    keyword_file = pd.read_csv(f"./support_docs/{keyword_file}", encoding = "windows-1251")
    keywords = list(keyword_file[file_col])
    df[new_col] = np.nan
    for i in range(len(df[col])):
        name = df[col][i]
        if name != name: continue
        keyword = None
        for key in keywords:
            if name != name.replace(key, "", 1):
                name = name.replace(key, "", 1)
                keyword = key
                break
        if keyword!=None:
            df[new_col][i] = keyword
            df[col][i] = df[col][i].replace(keyword, "").strip()
    return df

def lookup_table_and_map_to_new_column(df, col, lookup_file, ref_col, target_col, new_col):
    '''
    This function takes a reference document (lookup_file) to lookup. It then maps the values in the column (col) of the dataframe (df) to the reference document (ref_col), then df creates a new column (new_col) in the dataframe with the mapped values (target_col). 
    '''
    try:
        file = pd.read_csv(f"./support_docs/{lookup_file}", encoding = "windows-1251", usecols=[ref_col, target_col])
    except:
        file = pd.read_excel(f"./support_docs/{lookup_file}", usecols=[ref_col, target_col])
    mapping = dict(zip(file[ref_col], file[target_col]))
    df[new_col] = np.nan
    for i in range(len(df[col])):
        if df[col][i] in mapping.keys():
            df[new_col][i] = mapping[df[col][i]]
    return df

def clean_strings(df, col):
    '''
    This function clean the strings in a column (col) by removing all punctuation marks, quotation marks, and extra space characters for each value, then returns the dataframe (df).
    '''
    for i in range(len(df[col])):
        if df[col][i]!=df[col][i]:
            continue
        name = df[col][i]
        name = (name.replace("'", ' ')).replace('"', ' ')
        name = name.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
        name = re.sub(' +', ' ', name)
        df[col][i] = name.strip()
    return df
