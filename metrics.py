def printMetrics(tokenized_words_list, text_list):
    tokens_per_word_list= []
    tokens_per_group = []
    splited_words_per_group = []
    words_per_group = []
    fertility_per_group = [] 
    for i_doc, word_group in  enumerate(tokenized_words_list):       
        #statistics
        number_words = len(text_list[i_doc].split(' '))
        number_tokens = len(word_group)
        fertility = []

        if number_words >0:
            words_per_group.append(number_words)
            tokens_per_word_list.append(number_tokens/number_words)
            tokens_per_group.append(number_tokens)
            consecutive_subwords = 0
            total_splited_words = 0
            for token in word_group:
                if token[:2] == "##":
                    consecutive_subwords += 1
                else:
                    if consecutive_subwords == 1:
                        total_splited_words += 1
                    
                    if consecutive_subwords >0:
                        fertility.append(consecutive_subwords)
                    consecutive_subwords = 0
            
            if consecutive_subwords >0:
                fertility.append(consecutive_subwords)
            fertility_per_group.append((number_words+sum(fertility))/number_words)
            if total_splited_words >0: splited_words_per_group.append(total_splited_words/number_words)

    if len(tokens_per_word_list) > 0: print('Average Tokens per word',sum(tokens_per_word_list)/len(tokens_per_word_list)) 
    if len(tokens_per_group) > 0: print('Average Tokens per group',sum(tokens_per_group)/len(tokens_per_group)) 
    if len(splited_words_per_group) > 0: print('Average divided words per group',sum(splited_words_per_group)/len(splited_words_per_group)) 
    if len(words_per_group) > 0: print('Average words per group',sum(words_per_group)/len(words_per_group)) 
    if len(fertility_per_group) >0: print('Average fertility per group', sum(fertility_per_group)/len(fertility_per_group))
    