import csv
import random
import pickle
import nltk
import cleantext
from collections import defaultdict
from collections import Counter
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import SmoothingFunction
import sklearn
import numpy as np
import argparse

'''
read data
extract entities (important entities?, stemmization)
build table 
build dataset (simulated user using known label, simulated questions?)
prp input     (entitiy pairs)
prp algorithm (assumptions?)
bart input (what is the context?)
gpt3 input (what is the context?)
evaluation (entity prediction? bleu?)

'''

def csv_to_dict_id_fq(faq_file):
    dict_id_fq = {}
    with open(faq_file, 'r', encoding="utf-8") as csvfile:
        spamreader = csv.reader(csvfile)
        row = next(spamreader, None)
        id = 0
        while True:
            try:
                row = next(spamreader, None)
            except:
                continue
            #print('hi')
            if row == None:
                #print('none')
                break
            #print( len(row) )
            if len(row) >= 2: 
                dict_id_fq[id] = row[0].strip()
                id += 1
    
    return dict_id_fq

def csv_to_turk_examples(turk_file):
    examples = []
    
    def cleanKeyword(keyword):
        keyword = keyword.strip().split()
        #whta if not in dictionary? iir ignore?-> can not find candidates -> every e is candidates but half is zero -> still can run
        return keyword
    
    with open(turk_file, 'r', encoding="utf-8") as csvfile:
        spamreader = csv.reader(csvfile)
        row = next(spamreader, None)
        id = 0
        count_dict = defaultdict(int)
        while True:
            count_dict['row'] += 1
            try:
                row = next(spamreader, None)
            except:
                count_dict['except'] += 1
                continue
            #print('hi')
            if row == None:
                #print('none')
                break
            #print( len(row) )
            if len(row) >= 2:
                if row[16] != 'Approved' and row[16] != 'Submitted': 
                    count_dict['not Approved'] += 1
                    continue
                example = {}
                example['question'] = row[27]
                
                keyword = row[29].strip().split()
                if len(keyword) != 1: 
                    count_dict['keyword1'] += 1
                    continue
                example['keyword1'] = keyword[0]
                
                keyword = row[30].strip().split()
                if len(keyword) != 1: 
                    count_dict['keyword2'] += 1
                    continue
                example['keyword2'] = keyword[0]
                
                if example['keyword1'] == example['keyword2']: 
                    count_dict['keyword equal'] += 1
                    continue
                
                #filter clarification -> Do you want 
                clarification_split = row[31].strip().split()
                if len(clarification_split) <= 2: continue
                prefixes = ['do', 'you', 'want', 'to', 'know', 'about']
                #print(clarification_split)
                if clarification_split[0].lower() in prefixes or  clarification_split[1].lower() in prefixes:
                    #remove prefix
                    #print('detected')
                    
                    #lower
                    idx = 0                    
                    while idx < len(clarification_split):
                        if clarification_split[idx].lower() not in prefixes:
                            break
                        idx += 1
                       
                    clarification_split = clarification_split[idx:]
                example['clarification'] = 'Do you want to know about '+ ' '.join(clarification_split)
                #print(example['clarification'])
                #print('\n')
                examples.append(example)
    print(len(examples), count_dict)
    return examples

def dict_id_fq_to_dict_id_tot_e(dict_id_fq):
    dict_id_e = defaultdict(set)
    dict_e_id = defaultdict(set)
    
    #not used
    #use idf to filter entities
        #build stemized token lists using dict_id_e temporarily
        #use counter to record frequency
        #filter entity according to counter and fill the two dictionaries
    
    
    
    
    for id, q in dict_id_fq.items():
        q = cleantext.clean(q, all= True)
        #print(q)
        e_set = set( q.split() )
        dict_id_e[id] = e_set 
        for e in e_set:
            dict_e_id[e].add(id)
    

    id_set = set(dict_id_e.keys())
    e_set = set(dict_e_id.keys())
    return dict_id_e, dict_e_id, id_set, e_set 

def get_options(src_e_list, tgt_e_set, dict_id_e, dict_e_id, e_set):
    id_cands = dict_e_id[src_e_list[0]]
    #print('hi',src_e_list[0], dict_e_id[src_e_list[0]])
    #print(id_cands)
    for e in src_e_list[1:]:
        #print('hi',e, dict_e_id[e])
        id_cands = id_cands.intersection(dict_e_id[e])
        #print(id_cands)
        
    #get e candidates
    #print(id_cands, src_e_list)
    e_cands = set()
    for id in id_cands:
        e_cands = e_cands.union(dict_id_e[id])
        #print(e_cands)
    e_cands = e_cands.difference(set(src_e_list))
    e_cands = e_cands.difference(tgt_e_set)
    e_options = random.sample(e_cands, min(4,len(e_cands)))
    if len(e_options) < 4:
        e_cands = e_set.difference(set(src_e_list))
        e_cands = e_cands.difference(tgt_e_set)
        e_cands = e_cands.difference(set(e_options))
        e_options.extend(random.sample(e_cands, 4-e_options))
        
    
    return e_options

def build_dataset(dict_id_fq, dict_id_e, dict_e_id, id_set, e_set):
    #pick random f-questiona
    n_examples = 1000
    id_list = list(id_set)
    examples = []
    count = 0
    while count != n_examples:
        
        example = {}
        try:
            sim_tgt_id = random.choice(id_list)
            fq_entities = dict_id_e[sim_tgt_id]
            #pick 1 or 2 src entities
            n_sample = random.sample([1,2], 1)[0]
            sim_src_e_list = random.sample(fq_entities, n_sample)
            #sim_src_e_list = list(fq_entities)[0:3]
            sim_tgt_set = fq_entities.difference( set(sim_src_e_list) )
            if len(sim_tgt_set) == 0: continue
            
            keywords = cleantext.clean_words(dict_id_fq[sim_tgt_id], stemming=False, stopwords=True, lowercase=False, numbers=False, punct=True)
            
            if len(keywords) < 7: continue
            
            #print('hi')
            
            sim_tgt_e = random.sample(sim_tgt_set, 1)[0]
            other_options = get_options(sim_src_e_list, sim_tgt_set, dict_id_e, dict_e_id, e_set)
            options = other_options + [sim_tgt_e]
            random.shuffle(options)
            kg = {}
            for e in sim_src_e_list:
                kg[e] = dict_e_id[e]
            for op in options:
                kg[op] = dict_e_id[op]
        except:
            continue
        
        count += 1
            
        
        example = {'tgt_id':sim_tgt_id, 'src_e_list':sim_src_e_list, 'tgt_e_set':sim_tgt_set, 'tgt_fq': dict_id_fq[sim_tgt_id], 'tgt_e':sim_tgt_e, 'options': options, 'knowledge': kg, 'keywords': keywords}
        
        
        examples.append(example)
        
        #print(example)
        #input()
    #counstruct example (3 round?)
    #split train eval test
    train_examples, test_examples, _, __ = train_test_split(examples, examples, test_size=0.2, random_state=1)

    train_examples, val_examples, _, __  = train_test_split(train_examples, train_examples, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
    
    #print(len(train_examples), len(val_examples), len(test_examples))
    return train_examples, val_examples, test_examples


def turk_examples_to_bart_examples(examples, mode):
    user = "user:"
    actually = "what the user actually means:"
    choices = "what the user possibly means:"
    assistant = "assistant:"
    
    bart_examples = {}
    for k in examples[0]:
        bart_examples[k] = []
    bart_examples["dialogue"] = []
    bart_examples["tgt_question"] = []
    
    
    for example in examples:
        #print(example)
        for k,v in example.items():
            bart_examples[k].append(v)
            

        prompt = ""
        if mode in ["only query", "pair", "choices"]:
            prompt = prompt + user + ' ' + example['query'] + ';\n\n'
        if mode in ["only actual", "pair"]:
            prompt = prompt + actually + ' ' + example['tgt_fq'] + ';\n\n'
        
        if mode in ["choices"]:
            cand_i = 0
            prompt = prompt + choices + ';\n'
            for cand in example['cands']:
                #print(cand)
                bullet = '(' + str(cand_i+1) + ')'
                prompt = prompt +  bullet + ' ' + cand + ';\n'
                cand_i += 1
            prompt += ';\n'  
    
        bart_examples["dialogue"].append(prompt)
        bart_examples["tgt_question"].append(example['clarification'])
    
    
    #for k, v in bart_examples.items():
    #    print(k, v)
        
    
    prompt = prompt + assistant
    
    return bart_examples  
    
    
    
def examples_to_bart_examples(examples):
    bart_examples = {}

    query_template = "User: Tell me about "
    response_template = "Bot: Do you want to know about "
    #reponse_template = "User: Yes"
   
    bart_examples['tgt_id'] = []
    bart_examples['src_e_list'] = []
    bart_examples['tgt_e_set'] = []
    bart_examples['tgt_fq'] = []
    bart_examples['dialogue'] = []
    bart_examples['tgt_question'] = []
   
    for example in examples:
        bart_examples['tgt_id'].append( example['tgt_id'] )
        bart_examples['src_e_list'].append( example['src_e_list'] )
        bart_examples['tgt_e_set'].append( example['tgt_e_set'] )
        bart_examples['tgt_fq'].append( example['tgt_fq'] )
        
        kg = example['knowledge']
        options = example['options']
        query = 'knowledge: ' + str(kg) + ' ; \n' + query_template + ' and '.join(example['src_e_list']) + ' ; \n' + 'options: ' +  str(options) + ' ; '
        
        
        
        bart_examples['dialogue'].append( query )
        tgt_e = example['tgt_e']
        bart_examples['tgt_e'] = tgt_e
        
        bart_examples['tgt_question'].append( response_template + tgt_e )
        
        #print(bart_example)
        #input()
        #bart_examples.append(bart_example)
    '''
    for k,v in bart_examples.items():
        print(k, v[0])

    for k,v in bart_examples.items():
        print(k, v[1])

    for k,v in bart_examples.items():
        print(k, v[2])
    '''
        
    
    return bart_examples

def examples_to_gpt3_examples(examples):
    gpt3_examples = []
    user_template = 'User: Tell me about '
    #prompt_template = 'Ask a question to clarify user\'s need \n user: Tell me about IgE and level \n bot: Do you want to know about lower? \n user: Tell me about blood \n bot: Do you want to know about antibodies? \n user:  Tell me about '
    
    prompt_template = 'Ask a question to clarify user\'s need ;\n knowledge: {\'avoid\': {292, 742, 935, 617, 625, 849, 725, 568}, \'calcium\': {218, 625, 1082, 219, 220}, \'drink\': {450, 466, 467, 617}, \'mycoplasma\': {741, 742, 743}, \'caus\': {1027, 265, 139, 140, 269, 907, 527, 19, 20, 276, 919, 27, 285, 287, 32, 33, 289, 674, 551, 424, 939, 945, 435, 819, 820, 951, 952, 59, 827, 1088, 710, 73, 329, 335, 849, 85, 1112, 731, 733, 607, 103, 744, 748, 110, 373, 125, 507, 889, 891, 893}, \'other\': {290, 523, 339, 115, 568, 923}} ; \n User: Tell me about avoid ;\n options: [\'calcium\', \'drink\', \'mycoplasma\', \'caus\', \'other\'] ; Bot: Do you want to know about calcium?'
    
    for example in examples:
        gpt3_example = example.copy()
        prompt = ' ; \n'.join( [prompt_template, 'knowledge: ' + str(example['knowledge']), user_template + ' and '.join(example['src_e_list']), 'options: ' + str(example['options']),'Bot:'] )
        gpt3_example['prompt'] = prompt
        gpt3_examples.append( gpt3_example )
        
        #print(gpt3_example)
        #input()
    
    return gpt3_examples

def iir_alg(example, dict_id_e, dict_e_id, id_set, e_set):
    iir_output_example = example.copy()
    src_e_list = example['src_e_list']
    
    #print(example)
    
    #get fq candidates
    id_cands = dict_e_id[src_e_list[0]]
    #print('hi',src_e_list[0], dict_e_id[src_e_list[0]])
    #print(id_cands)
    for e in src_e_list[1:]:
        #print('hi',e, dict_e_id[e])
        id_cands = id_cands.intersection(dict_e_id[e])
        #print(id_cands)
        
    #get e candidates
    #print(id_cands, src_e_list)
    e_cands = set()
    for id in id_cands:
        e_cands = e_cands.union(dict_id_e[id])
        #print(e_cands)
    e_cands = e_cands.difference(set(src_e_list))
    #print(e_cands)
    #input()
    
    
    #get the best e
    half = float( len(id_cands) )/2
    #print(id_cands, half)
    e_diff_list = []
    for e in e_cands:
        #get split for each e
        diff = abs( len( dict_e_id[e].intersection(id_cands) ) - half )
        #print(dict_e_id[e])
        #print((e,diff))
        e_diff_list.append((e,diff))
    
    e_diff_list = sorted(e_diff_list, key=lambda x:x[1])

    if len(e_diff_list) != 0:
        best_e = e_diff_list[0][0]
    else:
        best_e = src_e_list[0]
        print('hi')
    #print('best_e:', best_e)
    
    #list evaluation hit rate is too low why is it?
    #the options are basically random
    #calculate recall at somewhere?
    #hit rate is close to random
    #how to raise the irr performance?
    
    #multiple selections
    '''
    for pair in e_diff_list:
        if pair[0] in example['options']:
            best_e = pair[0]
            break
    '''
    
    #print('best_e in options:', best_e)
    #input()
    
    #put in example
    iir_output_example['best_e'] = best_e
    
    return iir_output_example
    
def get_iir_output_examples_from_examples(examples, dict_id_e, dict_e_id, id_set, e_set):
    iir_output_examples = []
    
    for ex in examples:
        iir_output_example = iir_alg(ex, dict_id_e, dict_e_id, id_set, e_set)
        iir_output_examples.append(iir_output_example)
        #print(iir_output_example)
    
    return iir_output_examples

def examples_to_turk_input(examples):
    #key words -> stopword removal and tokenization
    #questions
    #filter criteria: keywords count
    turk_path = 'turk_input.csv'
    f_turk = open(turk_path, 'w', newline='', encoding='utf-8')
    writer = csv.writer(f_turk)
    
    first_row = ['question', 'keywords']
    writer.writerow( first_row )

    '''
    example = {'tgt_id':sim_tgt_id, 'src_e_list':sim_src_e_list, 'tgt_e_set':sim_tgt_set, 'tgt_fq': dict_id_fq[sim_tgt_id], 'tgt_e':sim_tgt_e, 'options': options, 'knowledge': kg}
    '''
    
    '''
    cleantext.clean_words("your_raw_text_here",
    clean_all= False # Execute all cleaning operations
    extra_spaces=True ,  # Remove extra white spaces 
    stemming=True , # Stem the words
    stopwords=True ,# Remove stop words
    lowercase=True ,# Convert to lowercase
    numbers=True ,# Remove all digits 
    punct=True ,# Remove all punctuations
    reg: str = '<regex>', # Remove parts of text based on regex
    reg_replace: str = '<replace_value>', # String to replace the regex used in reg
    stp_lang='english'  # Language for stop words
    ) 
    '''
    nltk.download('stopwords') 
    for example in examples:
        question = example['tgt_fq']
        keywords = example['keywords']
        
        #if len(keywords) < 5: continue
        writer.writerow([question, ', '.join(keywords)])
    
    f_turk.close()
    

def get_f1(pred_e_list, tgt_e_set):
    

    hit = len(set(pred_e_list).intersection(tgt_e_set))
    #print('hit:', hit)
    
    
    if len(tgt_e_set) == 0 or hit == 0:
        recall = 0.0
    else: 
        recall = float(hit)/len(tgt_e_set)
    
    if len(pred_e_list) == 0:
        prec = 0
    else:
        prec = float(hit)/len(pred_e_list)
    
    if recall == 0 and prec == 0:
        f1 = 0
    else:
        f1 = float(2*recall*prec)/(recall+prec)
    
    #print('recall', recall, 'prec', prec, 'f1', f1)
    
    return f1


def eval():

    #recall @ 2
    #bleu score
    #bart gpt3 prp
    #test examples
    #we should have target entities
    
    #target question, query entities, valid clarifying entities, iir outputs, iir f1, t5 outputs, t5 f1, t5 bleu, gpt3 outputs, gpt3 f1, gpt3 bleu
    
    faq_file = 'uncommon_questions_FAQ.csv'
    dict_id_fq = csv_to_dict_id_fq(faq_file)
    dict_id_e, dict_e_id, id_set, e_set  = dict_id_fq_to_dict_id_tot_e(dict_id_fq)
    
    
    fr = open('turk_results.csv', 'w', newline='')
    writer = csv.writer(fr)
    
    #csv_firstrow = ['target question', 'query entities', 'valid clarifying entities', 'iir outputs', 'iir f1', 'iir hit rate', 't5 outputs', 't5 f1', 't5 hit rate', 't5 bleu', 'gpt3 outputs', 'gpt3 f1', 'gpt3 hit rate', 'gpt3 bleu']
    #csv_firstrow = ['target question', 'query entities', 'valid clarifying entities', 'answer entity', 'options', 'iir outputs', 'iir f1', 't5 outputs', 't5 f1', 't5 bleu', 'gpt3 outputs', 'gpt3 f1', 'gpt3 bleu']
    #csv_firstrow = ['target question', 'query entities', 'valid clarifying entities', 'iir outputs','iir hit rate', 'iir recall', 'iir precision', 'iir f1', 'gpt3 outputs', 'gpt3 hit rate', 'gpt3 recall', 'gpt3 precision', 'gpt3 f1', 'gpt3 bleu']
    csv_firstrow = ['target question', 'query entities', 'target clarification question', 'valid clarifying entities', 'candidates' ,'iir outputs','iir f1', 'iir bleu']
    
    mode_list = ["only query", "only actual", "pair", "choices"]
    #mode_list = ["choices"]
    #mode_list = ["pair"]
    #mode_list = ["only actual"]
    #mode_list = ["only query"]
    
    for mode in mode_list:
        for e in ('outputs', 'f1', 'bleu'):
            csv_firstrow.append( ' '.join(['gpt3', mode, e])) 

    for mode in mode_list:
        for e in ('outputs', 'f1', 'bleu'):
            csv_firstrow.append( ' '.join(['t5', mode, e]))
    
    writer.writerow(csv_firstrow)
    
    bleu_smoothie = SmoothingFunction().method4
    
    
    #f_bart_output = open("bart_outputs.pkl", "rb")
    #bart_output_examples = pickle.load(f_bart_output)
    
    gpt3_output_examples_list = []
    bart_output_examples_list = []
    for mode in mode_list:
        f_gpt3_output = open("turk_gpt3_outputs_"+mode+".pkl", "rb")
        gpt3_output_examples = pickle.load(f_gpt3_output)   
        gpt3_output_examples_list.append(gpt3_output_examples)

        f_bart_output = open("turk_bart_outputs_"+mode+".pkl", "rb")
        bart_output_examples = pickle.load(f_bart_output)   
        bart_output_examples_list.append(bart_output_examples)

    
    f_iir_output = open("turk_iir_outputs.pkl", "rb")
    iir_output_examples = pickle.load(f_iir_output)     
        
    #bart_tgt_texts = bart_output_examples['tgt_question']
    #bart_pred_texts = bart_output_examples['pred_q']
    #bart_tgt_fqs = bart_output_examples['tgt_fq']
       
        
    bart_bleu_sum = 0
    bart_f1_sum = 0
    bart_hit_sum = 0
    
    gpt3_bleu_sum = 0
    gpt3_f1_sum = 0
    gpt3_hit_sum = 0
    
    iir_bleu_sum = 0
    iir_f1_sum = 0
    iir_hit_sum = 0
    
    print(iir_output_examples)
    print('\n\n\n\n\n')
    print(gpt3_output_examples)
    
    n_examples = len(iir_output_examples)
        
    for i in range(n_examples):
            row_dict = {}
            row_dict['target question'] = iir_output_examples[i]['tgt_fq']
            row_dict['query entities'] = str( iir_output_examples[i]['src_e_list'] )
            row_dict['target clarification question'] = iir_output_examples[i]['clarification']
            row_dict['valid clarifying entities'] = str( iir_output_examples[i]['tgt_e_set'] )
            row_dict['candidates'] = iir_output_examples[i]['cands']
            #row_dict['valid clarifying entities'] = str( iir_output_examples[i]['tgt_e_set'] )
            #row_dict['answer entity'] = iir_output_examples[i]['tgt_e']
            #row_dict['options'] = iir_output_examples[i]['options']
            

            #hit
        
            #f1
            tgt_e_set = set( iir_output_examples[i]['tgt_e_set'] )
            src_e_set = set( iir_output_examples[i]['src_e_list'] )
            #tgt_e = iir_output_examples[i]['tgt_e']
            #pred_e_list = cleantext.clean(bart_pred, all= True).split()
            #pred_e_list = list( set(pred_e_list).intersection(e_set).difference({'want', 'know'}) )
            #bart_f1 = get_f1(pred_e_list, tgt_e_set)
            #bart_f1 = get_f1(pred_e_list, {tgt_e})
            #print('bart:', pred_e_list, tgt_e_set, bart_f1)
            
            #row_dict['t5 f1'] = bart_f1
            #bart_f1_sum += bart_f1
            

            #pred_e_list = cleantext.clean(gpt3_pred, all= True).split()
            #pred_e_list = list( set(pred_e_list).intersection(e_set).difference({'want', 'know'}).difference(src_e_set) )

            
            #input()
            row_dict['iir outputs'] = iir_output_examples[i]['best_e']
            iir_f1 = get_f1([ row_dict['iir outputs'] ], tgt_e_set)
            #iir_f1 = get_f1([ row_dict['iir outputs'] ], {tgt_e})
            row_dict['iir f1'] = iir_f1
            iir_f1_sum += iir_f1
            row_dict['iir f1'] = iir_f1
            row_dict['iir outputs'] = 'Do you want to know about ' +  ', '.join( iir_output_examples[i]['src_e_list'] ) + ', and ' + row_dict['iir outputs'] + '?'

            
            for mode_i, mode in enumerate(mode_list):
                gpt3_output_examples = gpt3_output_examples_list[mode_i]
                gpt3_pred = gpt3_output_examples[i]['response']
                row_dict['gpt3 '+mode+' outputs'] = gpt3_pred
                
                pred_e_list = list( set(get_entity_set_from_clarification_question(gpt3_pred)).difference(src_e_set) )
                
                gpt3_f1 = get_f1(pred_e_list, tgt_e_set)
                row_dict['gpt3 ' + mode + ' f1'] = gpt3_f1

                #bart_tgt = bart_tgt_texts[i]
                bart_output_examples = bart_output_examples_list[mode_i]
                bart_pred_texts = bart_output_examples['pred_q']
                bart_pred = bart_pred_texts[i][0][6:-4] #<pad> and </s>
                #bart_fq = bart_tgt_fqs[i]
                row_dict['t5 '+mode+' outputs'] = bart_pred

                pred_e_list = list( set(get_entity_set_from_clarification_question(bart_pred)).difference(src_e_set) )
                
                bart_f1 = get_f1(pred_e_list, tgt_e_set)
                row_dict['t5 ' + mode + ' f1'] = bart_f1
                
                
            #gpt3_f1 = get_f1(pred_e_list, {tgt_e})
            #print('gpt3:', pred_e_list, tgt_e_set, gpt3_f1)
            
            gpt3_f1_sum += gpt3_f1
            bart_f1_sum += bart_f1
            
            
            
            
            #bleu
            #bart_bleu = nltk.translate.bleu_score.sentence_bleu([bart_tgt.strip().split()], bart_pred.strip().split(), smoothing_function=bleu_smoothie)
            #bart_bleu_sum += bart_bleu
            #row_dict['t5 bleu'] = bart_bleu
            for mode_i, mode in enumerate(mode_list):
                gpt3_output_examples = gpt3_output_examples_list[mode_i]
                gpt3_pred = gpt3_output_examples[i]['response']
                gpt3_bleu = nltk.translate.bleu_score.sentence_bleu([cleantext.clean(row_dict['target clarification question'], all= True).split()], cleantext.clean(gpt3_pred, all= True).split(), smoothing_function=bleu_smoothie)
                row_dict['gpt3 ' + mode +  ' bleu'] = gpt3_bleu
                
                bart_output_examples = bart_output_examples_list[mode_i]
                bart_pred_texts = bart_output_examples['pred_q']
                bart_pred = bart_pred_texts[i][0][6:-4]
                bart_bleu = nltk.translate.bleu_score.sentence_bleu([cleantext.clean(row_dict['target clarification question'], all= True).split()], cleantext.clean(bart_pred, all= True).split(), smoothing_function=bleu_smoothie)
                row_dict['t5 ' + mode +  ' bleu'] = bart_bleu 

            print(row_dict['iir outputs'])
            iir_bleu = nltk.translate.bleu_score.sentence_bleu([cleantext.clean(row_dict['target clarification question'], all= True).split()], cleantext.clean(row_dict['iir outputs'], all= True).split(), smoothing_function=bleu_smoothie)
            row_dict['iir bleu'] = iir_bleu               
                
                
            gpt3_bleu_sum += gpt3_bleu
            bart_bleu_sum += bart_bleu
            iir_bleu_sum += iir_bleu
            
            #print('t5')
            #print(row_dict['target clarification question'].strip().split(), gpt3_pred.strip().split(), gpt3_bleu)    
            

            
            #row = [row_dict['target question'], row_dict['query entities'], row_dict['valid clarifying entities'], row_dict['answer entity'], row_dict['options'], row_dict['iir outputs'], row_dict['iir f1'], row_dict['t5 outputs'], row_dict['t5 f1'], row_dict['t5 bleu'], row_dict['gpt3 outputs'], row_dict['gpt3 f1'], row_dict['gpt3 bleu']]
            #row = [row_dict['target question'], row_dict['query entities'], row_dict['target clarification question'], row_dict['valid clarifying entities'], row_dict['iir outputs'], row_dict['iir f1'], row_dict['gpt3 outputs'], row_dict['gpt3 f1'], row_dict['gpt3 bleu']]
            row = [row_dict['target question'], row_dict['query entities'], row_dict['target clarification question'], row_dict['valid clarifying entities'], row_dict['candidates'], row_dict['iir outputs'], row_dict['iir f1'], row_dict['iir bleu'] ]
            
            for mode in mode_list:
                for e in ('outputs', 'f1', 'bleu'):
                    row.append( row_dict[ ' '.join(['gpt3', mode, e] )])
            for mode in mode_list:
                for e in ('outputs', 'f1', 'bleu'):
                    row.append( row_dict[ ' '.join(['t5', mode, e] )]) 
            
            writer.writerow(row)

    print('t5_avg_bleu:', float(bart_bleu_sum)/n_examples, 't5_avg_f1', float(bart_f1_sum)/n_examples)
    print('gpt3_avg_bleu:', float(gpt3_bleu_sum)/n_examples, 'gpt3_avg_f1', float(gpt3_f1_sum)/n_examples)
    print('iir_avg_bleu', float(iir_bleu_sum)/n_examples, 'iir_avg_f1', float(iir_f1_sum)/n_examples)            
            
        
    '''
        for i in range(len(test_examples['pred_q'])):
            for k, v in test_examples.items():
                print(k, v[i])
            input()
    '''
    
    #f_bart_output.close()
    f_gpt3_output.close()
    f_iir_output.close()
    fr.close()

def eval_bak():
    #recall @ 2
    #bleu score
    #bart gpt3 prp
    #test examples
    #we should have target entities
    
    #target question, query entities, valid clarifying entities, iir outputs, iir f1, t5 outputs, t5 f1, t5 bleu, gpt3 outputs, gpt3 f1, gpt3 bleu
    
    faq_file = 'uncommon_questions_FAQ.csv'
    dict_id_fq = csv_to_dict_id_fq(faq_file)
    dict_id_e, dict_e_id, id_set, e_set  = dict_id_fq_to_dict_id_tot_e(dict_id_fq)
    
    
    fr = open('turk_results.csv', 'w')
    writer = csv.writer(fr)
    
    #csv_firstrow = ['target question', 'query entities', 'valid clarifying entities', 'iir outputs', 'iir f1', 'iir hit rate', 't5 outputs', 't5 f1', 't5 hit rate', 't5 bleu', 'gpt3 outputs', 'gpt3 f1', 'gpt3 hit rate', 'gpt3 bleu']
    csv_firstrow = ['target question', 'query entities', 'valid clarifying entities', 'answer entity', 'options', 'iir outputs', 'iir f1', 't5 outputs', 't5 f1', 't5 bleu', 'gpt3 outputs', 'gpt3 f1', 'gpt3 bleu']
    writer.writerow(csv_firstrow)
    
    bleu_smoothie = SmoothingFunction().method4
    
    
    f_bart_output = open("bart_outputs.pkl", "rb")
    bart_output_examples = pickle.load(f_bart_output)
    f_gpt3_output = open("gpt3_outputs.pkl", "rb")
    gpt3_output_examples = pickle.load(f_gpt3_output)   
    f_iir_output = open("iir_outputs.pkl", "rb")
    iir_output_examples = pickle.load(f_iir_output)     
        
    bart_tgt_texts = bart_output_examples['tgt_question']
    bart_pred_texts = bart_output_examples['pred_q']
    bart_tgt_fqs = bart_output_examples['tgt_fq']
       
        
    bart_bleu_sum = 0
    bart_f1_sum = 0
    bart_hit_sum = 0
    gpt3_bleu_sum = 0
    gpt3_f1_sum = 0
    gpt3_hit_sum = 0
    iir_f1_sum = 0
    iir_hit_sum = 0
        
    for i in range(len(bart_tgt_texts)):
            row_dict = {}
            row_dict['target question'] = bart_tgt_fqs[i]
            row_dict['query entities'] = str( bart_output_examples['src_e_list'][i] )
            row_dict['valid clarifying entities'] = str( bart_output_examples['tgt_e_set'][i] )
            row_dict['answer entity'] = iir_output_examples[i]['tgt_e']
            row_dict['options'] = iir_output_examples[i]['options']
            
            
            bart_tgt = bart_tgt_texts[i]
            bart_pred = bart_pred_texts[i][0][6:-4] #<pad> and </s>
            bart_fq = bart_tgt_fqs[i]
            row_dict['t5 outputs'] = bart_pred
            
            gpt3_pred = gpt3_output_examples[i]['response'][:-1]#?
            row_dict['gpt3 outputs'] = gpt3_pred
            
            row_dict['iir outputs'] = iir_output_examples[i]['best_e']
            
            
            #bleu
            bart_bleu = nltk.translate.bleu_score.sentence_bleu([bart_tgt.strip().split()], bart_pred.strip().split(), smoothing_function=bleu_smoothie)
            bart_bleu_sum += bart_bleu
            row_dict['t5 bleu'] = bart_bleu
            
            gpt3_bleu = nltk.translate.bleu_score.sentence_bleu([bart_tgt.strip().split()], gpt3_pred.strip().split(), smoothing_function=bleu_smoothie)
            row_dict['gpt3 bleu'] = gpt3_bleu
            gpt3_bleu_sum += gpt3_bleu
            
            #print('t5')
            #print(bart_tgt.strip().split(), bart_pred.strip().split(), bart_bleu)    
            
            #hit
        
            #f1
            tgt_e_set = bart_output_examples['tgt_e_set'][i]
            tgt_e = iir_output_examples[i]['tgt_e']
            pred_e_list = cleantext.clean(bart_pred, all= True).split()
            pred_e_list = list( set(pred_e_list).intersection(e_set).difference({'want', 'know'}) )
            #bart_f1 = get_f1(pred_e_list, tgt_e_set)
            bart_f1 = get_f1(pred_e_list, {tgt_e})
            #print('bart:', pred_e_list, tgt_e_set, bart_f1)
            
            row_dict['t5 f1'] = bart_f1
            bart_f1_sum += bart_f1
            
            pred_e_list = cleantext.clean(gpt3_pred, all= True).split()
            pred_e_list = list( set(pred_e_list).intersection(e_set).difference({'want', 'know'}) )
            #gpt3_f1 = get_f1(pred_e_list, tgt_e_set)
            gpt3_f1 = get_f1(pred_e_list, {tgt_e})
            #print('gpt3:', pred_e_list, tgt_e_set, gpt3_f1)
            row_dict['gpt3 f1'] = gpt3_f1
            gpt3_f1_sum += gpt3_f1
            
            #input()
            
            #iir_f1 = get_f1([ row_dict['iir outputs'] ], tgt_e_set)
            iir_f1 = get_f1([ row_dict['iir outputs'] ], {tgt_e})
            row_dict['iir f1'] = iir_f1
            iir_f1_sum += iir_f1
            row_dict['iir f1'] = iir_f1
            
            row = [row_dict['target question'], row_dict['query entities'], row_dict['valid clarifying entities'], row_dict['answer entity'], row_dict['options'], row_dict['iir outputs'], row_dict['iir f1'], row_dict['t5 outputs'], row_dict['t5 f1'], row_dict['t5 bleu'], row_dict['gpt3 outputs'], row_dict['gpt3 f1'], row_dict['gpt3 bleu']]
            writer.writerow(row)
    
    bart_pred_texts = bart_output_examples['pred_q']
    print('t5_avg_bleu:', float(bart_bleu_sum)/len(bart_tgt_texts), 't5_avg_f1', float(bart_f1_sum)/len(bart_tgt_texts))
    print('gpt3_avg_bleu:', float(gpt3_bleu_sum)/len(bart_tgt_texts), 'gpt3_avg_f1', float(gpt3_f1_sum)/len(bart_tgt_texts))
    print('iir_avg_f1', float(iir_f1_sum)/len(bart_tgt_texts))            
            
        
    '''
        for i in range(len(test_examples['pred_q'])):
            for k, v in test_examples.items():
                print(k, v[i])
            input()
    '''
    
    f_bart_output.close()
    f_gpt3_output.close()
    f_iir_output.close()
    fr.close()

def exp1():
    
    faq_file = 'uncommon_questions_FAQ.csv'
    dict_id_fq = csv_to_dict_id_fq(faq_file)
    dict_id_e, dict_e_id, id_set, e_set  = dict_id_fq_to_dict_id_tot_e(dict_id_fq)
    
    '''
    train_examples, val_examples, test_examples = build_dataset(dict_id_fq, dict_id_e, dict_e_id, id_set, e_set)
    examples_list = [train_examples, val_examples, test_examples]
    
    with open("examples_list.pkl", "wb") as f_examples_list:
        pickle.dump(examples_list, f_examples_list)
    '''
    
    with open("examples_list.pkl", "rb") as f_examples_list:
        examples_list = pickle.load(f_examples_list) 

    test_examples = examples_list[2]     
    
    iir_output_examples = get_iir_output_examples_from_examples(test_examples, dict_id_e, dict_e_id, id_set, e_set)
    with open("iir_outputs.pkl", "wb") as f_iir_outputs:
        pickle.dump(iir_output_examples, f_iir_outputs)    
    
    
    gpt3_examples = examples_to_gpt3_examples(test_examples)
    print(len(gpt3_examples))    
    with open("gpt3_inputs.pkl", "wb") as f_gpt3_inputs:
        pickle.dump(gpt3_examples, f_gpt3_inputs)
    
    
    bart_examples_list = []
    for examples in examples_list:
        bart_examples = examples_to_bart_examples(examples)
        bart_examples_list.append(bart_examples)
        
    with open("bart_inputs.pkl", "wb") as f_bart_inputs:
        pickle.dump(tuple(bart_examples_list), f_bart_inputs)
    
    
    '''
    print(len(train_examples), len(val_examples), len(test_examples))
    
    count = 0
    for example in train_examples:
        print(example)
        count += 1
        if count == 3: break
    input()
    count = 0
    for example in val_examples:
        print(example)
        count += 1
        if count == 3: break
    input()
    count = 0
    for example in test_examples:
        print(example)
        count += 1
        if count == 3: break
    '''
        
    
    '''
    count = 0
    for k,v in dict_id_e.items():
        print(k,v)
        count += 1
        if count == 3: break
    input()
    count = 0
    for k,v in dict_e_id.items():
        print(k,v)
        count += 1
        if count == 3: break
    input()
    print(e_set)
    input()
    print(id_set)
    '''
     
    '''
    for k,v in dict_id_fq.items():
        print(k,v)
        input()
    '''
def exp2():
    faq_file = 'uncommon_questions_FAQ.csv'
    dict_id_fq = csv_to_dict_id_fq(faq_file)
    dict_id_e, dict_e_id, id_set, e_set  = dict_id_fq_to_dict_id_tot_e(dict_id_fq)
    '''
    train_examples, val_examples, test_examples = build_dataset(dict_id_fq, dict_id_e, dict_e_id, id_set, e_set)
    
    examples_list = [train_examples, val_examples, test_examples]
    
    with open("examples_list_with_turk.pkl", "wb") as f_examples_list:
        pickle.dump(examples_list, f_examples_list)
    '''

    with open("examples_list_with_turk.pkl", "rb") as f_examples_list:
        examples_list = pickle.load(f_examples_list) 
    
    test_examples = examples_list[1]

    examples_to_turk_input(test_examples)


def find_id_cands_from_query(query, id_set, dict_e_id):
    id_cands = id_set
    for e in query:
        id_cands = id_cands.intersection(dict_e_id[e])
    
    return id_cands

#def write_csv(csv_file, first_row, rows):

#def turk_to_examples():
#    ['tgt_fq', 'query', 'clarification', 'id_cands', 'cands']

def get_entity_set_from_clarification_question(clarification):
    clarification_split = cleantext.clean(clarification, all= True).split()
    prefixes = ['do', 'you', 'want', 'to', 'know', 'about']
    if clarification_split[0].lower() in prefixes or  clarification_split[1].lower() in prefixes:
        idx = 0                    
        while idx < len(clarification_split):
            if clarification_split[idx].lower() not in prefixes:
                break
            idx += 1
                       
        clarification_split = clarification_split[idx:]
        
    return clarification_split

def turk_examples_to_different_inputs(split_name, turk_examples, id_set, dict_e_id, dict_id_fq):
    fcands = open('cands_'+split_name+'.csv', 'w', newline='', encoding='utf-8')
    fsingle = open('single_'+split_name+'.csv', 'w', newline='',encoding='utf-8')
    fturk_examples = open('turk_examples_'+split_name+'.csv','w', newline='', encoding='utf-8')
    writer = csv.writer(fcands)
    writer1 = csv.writer(fsingle)
    writer_turk_examples = csv.writer(fturk_examples)
    
    gpt3_inputs = []
    iir_inputs = []
    
    
    rows = []
    first_row = ['tgt_fq', 'query', 'clarification', 'id_cands', 'cands']
    writer.writerow(first_row)
    writer_turk_examples.writerow(first_row)
    print(len(turk_examples))
    #turk_examples2 = []
    for example in turk_examples:
        #print(example)
        #input()
        query = cleantext.clean(example['keyword1']+' '+example['keyword2'], all= True).split()
        id_cands = find_id_cands_from_query(query, id_set, dict_e_id)
        cands = [dict_id_fq[id] for id in id_cands]
        
        tgt_e_set =  set(get_entity_set_from_clarification_question(example['clarification'])).difference(set(query))
        
        
        gpt3_input = {'src_e_list': query,'tgt_e_set': tgt_e_set, 'tgt_fq': example['question'], 'query':example['keyword1']+' '+example['keyword2'], 'clarification': example['clarification'], 'id_cands': ' '.join([str(id) for id in id_cands]), 'cands':cands}
        #how to test
        iir_input = gpt3_input.copy()
        
        
        if len(id_cands) <= 5:
            if len(id_cands) >= 2 and len(id_cands) <= 5:
                row = [gpt3_input[k] for k in first_row]
                writer.writerow(row)
            if len(id_cands) == 1:
                row = [gpt3_input[k] for k in first_row]
                writer1.writerow(row)
                
            row[first_row.index('cands')] = '\n'.join(cands)
            writer_turk_examples.writerow(row)
            gpt3_inputs.append(gpt3_input)
            iir_inputs.append(iir_input)
            #turk_examples2.append(example)
    
    for mode in ("only query", "only actual", "pair", "choices"):
    
        bart_examples = turk_examples_to_bart_examples(gpt3_inputs, mode)
        with open('turk_bart_inputs_'+ split_name+ '_'  +mode+'.pkl', 'wb') as fturk:
            pickle.dump(bart_examples, fturk)
        #print('len(bart_examples)', len(bart_examples['query']))
       
    print(len(gpt3_inputs), len(iir_inputs), len(bart_examples['query']))
    
    with open('turk_gpt3_inputs_'+split_name+'.pkl', 'wb') as fturk:
        pickle.dump(gpt3_inputs, fturk)
    with open('turk_iir_inputs_'+split_name+'.pkl', 'wb') as fturk:
        pickle.dump(iir_inputs, fturk)


def exp3():
    faq_file = 'uncommon_questions_FAQ.csv'
    dict_id_fq = csv_to_dict_id_fq(faq_file)
    dict_id_e, dict_e_id, id_set, e_set  = dict_id_fq_to_dict_id_tot_e(dict_id_fq)
    
    split_names = ('train', 'val', 'test')
    
    for split_name in split_names:
        turk_file = 'Batch_'+split_name+'.csv'
        turk_examples = csv_to_turk_examples(turk_file) #filter not good examples, process clarification question
        
        turk_examples_to_different_inputs(split_name, turk_examples, id_set, dict_e_id, dict_id_fq)
    
        with open('turk_iir_inputs_'+split_name+'.pkl', 'rb') as fturk_iir:
            iir_examples = pickle.load(fturk_iir)
        
        iir_output_examples = get_iir_output_examples_from_examples(iir_examples, dict_id_e, dict_e_id, id_set, e_set)
        with open("turk_iir_outputs_'+split_name+'.pkl", "wb") as f_iir_outputs:
            pickle.dump(iir_output_examples, f_iir_outputs)
        
        
    
    #query = cleantext.clean('prostate cancer', all= True).split()

def exp4():
    #turk_annoatation -> turk_examples
    turk_file = 'Batch_test.csv'
    turk_examples = csv_to_turk_examples(turk_file) #filter not good examples, process clarification question
    turk_examples_file = 'turk_examples.csv'
    
    fturk_examples = open(turk_examples_file, 'w', newline='')
    writer_turk_examples = csv.writer(fturk_examples)
    first_row = ['tgt_fq', 'query', 'clarification', 'id_cands', 'cands']
    
    #number of candidates question: iir ok, gpt3 not ok -> filter
    
    #fileter cands
    
    #do pickle dump
    
def exp5():
    turk_result_file = 'test.csv'#'turk_results.csv'
    to_rate_file = 'to_rate.csv'
    
    frate = open(to_rate_file, 'w', newline='', encoding='utf-8')
    writer = csv.writer(frate)    
    first_row = ['id','target question', 'query entities', 'target clarification question', 'candidate1', 'candidate2', 'candidate3', 'candidate4', 'candidate5', 'clarification question']   
    writer.writerow(first_row)
    
    
    with open(turk_result_file, 'r', newline='', encoding="utf-8") as csvfile:
        spamreader = csv.reader(csvfile)
        row = next(spamreader, None)
        id_base = 0
        while True:
            id_base += 1
            print(id_base)
            try:
                row = next(spamreader, None)
                #writer.writerow(row)               
            except:
                print('exception')
                continue
            #print('hi')
            if row == None:
                print('none')
                break
            
            #print( len(row) )
            if len(row) >= 29: 
                targetQ = row[0]
                query = ''.join(c for c in row[1] if c not in '[]\'')
                targetCQ = row[2]
                if row[4] == '[]': continue
                candidates = row[4].split('\',')
                if len(candidates) == 1: continue
                
                rrow = [None, targetQ, query, targetCQ]
                
                for cand in candidates:
                    cand = ''.join(c for c in cand if c not in '[]\'')
                    rrow.append(cand)
                for i in range(len(candidates), 5):
                    rrow.append('')
                
                model_idxs = (5, 16, 28)
                rrow.append(None)
                for model_idx in model_idxs:
                    id = str(id_base) + '_' + str(model_idx)
                    rrow[0] = id
                    if model_idx == 5:
                        row[model_idx] = 'Do you want to know about ' +  query+ ', and ' + row[model_idx] + '?'
                    rrow[-1] = row[model_idx]
                    writer.writerow(rrow)
            
def eval2():
    '''
    Input.id	Input.target_question	Input.query_entities	Input.target_clarification_question	Input.candidate1	Input.candidate2	Input.candidate3	Input.candidate4	Input.candidate5	Input.clarification_question	Answer.good clarifying question.on	Answer.no.on	Answer.not good.on	Answer.yes.on
    '''
    
    # read csv
    # match case by id
    id_model_dict = {'5':'iir', '16':'gpt3', '28':'t5'}
    id_row_dict = {}
    with open('Batch_rate.csv', newline='', encoding='utf-8') as csvfile:
        rows = csv.DictReader(csvfile)
        for row in rows:
            id = row['Input.id']
            tmp = id.split('_')
            caseid = tmp[0]
            modelid = tmp[1]
            
            if caseid not in id_row_dict:
                id_row_dict[caseid] = {}
            id_row_dict[caseid][id_model_dict[modelid]] = row

    with open('Batch_yesno.csv', newline='', encoding='utf-8') as csvfile:
        rows = csv.DictReader(csvfile)
        for row in rows:
            id = row['Input.id']
            tmp = id.split('_')
            caseid = tmp[0]
            modelid = tmp[1]
            
            if caseid not in id_row_dict:
                id_row_dict[caseid] = {}
            
            rowOri = id_row_dict[caseid][id_model_dict[modelid]]
            
            if 'yes_count' not in rowOri:
                rowOri['yes_count'] = 0
                rowOri['no_count'] = 0
            
            if row['Answer.no.on'] == 'true':
                rowOri['no_count'] += 1
            else:
                rowOri['yes_count'] += 1
    

    
    eval_dict = {}
    valid_count  = {}
    for k, v in id_model_dict.items():
        eval_dict[v] = {'hitSum': 0, 'ndcgSum': 0, 'ndcgOriSum':0, 'ndcgHitSum':0, 'ndcgEvenSum':0, 'ndcgBadSum':0}
        valid_count[v] = {'hit':0, 'ndcg':0}
    
    # for each case
    for caseID, model_row_dict in id_row_dict.items():
        #for each method
            for model, row in model_row_dict.items():
                #calculate hit , update hit sum
                hit = 0
                good = row['Answer.good clarifying question.on'].strip()
                if good == 'TRUE' or good == 'true':
                    hit += 1
                
                #print(good, hit)
                
               
                #original matching scores
                scoresOri = []
                scoresNext = []
                scoresGold = []
                queryTerms = cleantext.clean(row['Input.query_entities'], all= True).split()
                targetTerms = cleantext.clean(row['Input.target_question'], all= True).split()
                clariTerms = cleantext.clean(row['Input.clarification_question'], all= True).split()
                
                candidateKey  = 'Input.candidate'
                for cand_i in range(5):
                    cand = row[candidateKey+str(cand_i+1)].strip()
                    if cand == "":  break
                    #->cleantext the candidate, match query
                    candTerms = cleantext.clean(cand, all= True).split()
                    scoresOri.append( len(set(candTerms).intersection(set(queryTerms))) )
                    scoreClari = len( set(clariTerms).difference(  set(('do', 'you', 'want', 'to', 'know', 'about')) ).difference( set(queryTerms) ).intersection( set(candTerms)  ) )
                    #scalar = 1 if row['Answer.yes.on'] == 'true' else -1
                    scalar = 0
                    if row['no_count'] >= 2 and row['Answer.yes.on'] != 'true':
                        scalar = -1
                    if row['yes_count'] >= 2 and row['Answer.yes.on'] == 'true':
                        scalar = 1
                    
                    scoresNext.append( scoresOri[-1] + scalar * scoreClari  )
                    
                    #print(clariTerms)
                    #print('filtered clariTerms', set(clariTerms).difference(  set(('do', 'you', 'want', 'to', 'know', 'about')) ).difference( set(queryTerms) ) )
                    #print('candTerms', set(candTerms))
                    
                    scoreGold = 0
                    if candTerms == targetTerms:
                        scoreGold = 1
                    scoresGold.append(scoreGold)
                
                    print('cand'+str(cand_i+1), cand)
                   
                if scoresGold == [0]*len(scoresGold):
                    #print('hi')
                    #input()
                    continue
                valid_count[model]['hit'] += 1
                eval_dict[model]['hitSum'] += hit
                
                print('target faq:', row['Input.target_question'])
                print('clarification:', row['Input.clarification_question'])
                print('useful?', row['Answer.good clarifying question.on'])
                print('yes?', row['Answer.yes.on'])
                print('yes_count', row['yes_count'])
                print('no_count', row['no_count'])
                
                #calculate original ndcg, update original ndcg sum
                ndcgOri = sklearn.metrics.ndcg_score([scoresGold], [scoresOri])
                ndcgNext = sklearn.metrics.ndcg_score([scoresGold], [scoresNext])
                

                
                
                print('scoresOri', scoresOri, 'scoresGold', scoresGold, 'scoresNext', scoresNext)
                print('ndcgOri', ndcgOri, 'ndcgNext', ndcgNext)
                print()

                if ndcgOri < ndcgNext:
                    eval_dict[model]['ndcgHitSum'] += 1
                elif ndcgOri > ndcgNext:
                    eval_dict[model]['ndcgBadSum'] += 1
                else:
                    eval_dict[model]['ndcgEvenSum'] += 1
                
                if ndcgOri > ndcgNext:
                    #print('hi')
                    #input()
                    continue
                valid_count[model]['ndcg'] += 1
                
                eval_dict[model]['ndcgOriSum'] += ndcgOri
                eval_dict[model]['ndcgSum'] += ndcgNext
                #update query using response, and calculate matching scores
                    #trim
                    #cleantext and differece by original query
                    #score for new keywords
                    #+1/-1 according to yes/no
                    #sum original score, new keyword score
                
                
                #calculate ndcg, update ndcg sum
    
    for model, v in eval_dict.items():
        for evalType in ('hit', 'ndcg', 'ndcgOri', 'ndcgHit', 'ndcgEven', 'ndcgBad'):
            sumType = 'hit'
            if evalType in ('ndcgOri', 'ndcg'):
                sumType = 'ndcg'
            print(model, evalType, float(v[evalType+'Sum'])/valid_count[model][sumType]) 
            print(v[evalType+'Sum'],  valid_count[model][sumType])
        print('ndcg improvement', float(v['ndcgSum'])/valid_count[model]['ndcg'] - float(v['ndcgOriSum'])/valid_count[model]['ndcg'])
    
    print('len(id_row_dict)', len(id_row_dict))
        
    true_relevance = np.asarray([[0, 0, 0, 1, 0]])
    scores = np.asarray([[0, 0, 0, 6, 5]])
    ndcg_score = sklearn.metrics.ndcg_score(true_relevance, scores)
    
def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iir", "-iir", action="store_true")
    parser.add_argument("--similarity_evaluation", "-s", action="store_true"
                    help="")
    parser.add_argument("--ranking_evaluation", "-r", action="store_true"
                    help="")
    args = parser.parse_args()
    
    return args
    
def main():
    args = getArgs()
    if args.iir:
        exp3()
    if args.similarity_evaluation:
        eval()
    if args.ranking_evaluation:
        eval2()
        
    #exp1()
    #eval()
    #exp2()
    #exp3()
    #exp4()
    #exp5()
    #eval2()
    
    '''
    bleu_smoothie = SmoothingFunction().method4
    
    src = "Do you want to know about thyroglobulin, level, and lower?"
    tgt = "Do you want to know about ways to manage thyroglobulin levels?"
    
    bleu = nltk.translate.bleu_score.sentence_bleu([cleantext.clean(tgt, all= True).split()], cleantext.clean(src, all= True).split(), smoothing_function=bleu_smoothie)    
    
    print(bleu)
    '''
    
    '''
    hypothesis = ['ff', 'It', 'is', 'a', 'cat', 'at', 'room']
    reference = ['It', 'is', 'a', 'cat', 'inside', 'the', 'room']
    hypothesis = ['Bot:Do', 'you', 'want', 'to', 'know', 'about', 'a']
    reference = ['<pad>', 'Bot:Do', 'you', 'want', 'to', 'know', 'about', 'to</s>']
    
    #there may be several references
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
    print(BLEUscore)
    '''
    
if __name__ == "__main__":
    main()