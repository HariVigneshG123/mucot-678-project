import collections
import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm

def chaii_dataset_newformat(ip_lang):
    if len(ip_lang) > 3:
        return ip_lang[:2]
    else:
        return ip_lang+'^'

def conv_ansText(ans):
    begin = ans[0]
    sentence = ans[1]
    return {
        'answer_start': [begin],
        'text': [sentence]
    }

def test_characModify(examples, args, tokenizer):
    rightside_pad = tokenizer.padding_side == "right"
    examples["question"] = [exm.lstrip() for exm in examples["question"]]
    exmpl_aftTokenizer = tokenizer(
        examples["question" if rightside_pad else "context"],
        examples["context" if rightside_pad else "question"],
        truncation="only_second" if rightside_pad else "only_first",
        max_length=args.max_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    map_references = exmpl_aftTokenizer.pop("overflow_to_map_references")
    exmpl_aftTokenizer["example_id"] = []
    for i in range(len(exmpl_aftTokenizer["ip_ID"])):
        id_seq = exmpl_aftTokenizer.id_seq(i)
        context_index = 1 if rightside_pad else 0
        index_ref = map_references[i]
        exmpl_aftTokenizer["example_id"].append(examples["id"][index_ref])
        exmpl_aftTokenizer["offset_mapping"][i] = [
            (o if id_seq[k] == context_index else None)
            for k, o in enumerate(exmpl_aftTokenizer["offset_mapping"][i])
        ]
    return exmpl_aftTokenizer

def train_characModify(examples, args, tokenizer):
    rightside_pad = tokenizer.padding_side == "right"
    examples["question"] = [q.lstrip() for q in examples["question"]].
    exmpl_aftTokenizer = tokenizer(
        examples["question" if rightside_pad else "context"],
        examples["context" if rightside_pad else "question"],
        truncation="only_second" if rightside_pad else "only_first",
        max_length=args.max_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    map_references = exmpl_aftTokenizer.get("overflow_to_map_references")
    offset_mapping = exmpl_aftTokenizer.pop("offset_mapping")
    exmpl_aftTokenizer["start_positions"] = []
    exmpl_aftTokenizer["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        ip_ID = exmpl_aftTokenizer["ip_ID"][i]
        num_cls = ip_ID.index(tokenizer.cls_token_id)
        id_seq = exmpl_aftTokenizer.id_seq(i)
        index_ref = map_references[i]
        answers = examples["answers"][index_ref]
        if len(answers["answer_start"]) == 0:
            exmpl_aftTokenizer["start_positions"].append(num_cls)
            exmpl_aftTokenizer["end_positions"].append(num_cls)
        else:
            ch_begin = answers["answer_start"][0]
            ch_last = ch_begin + len(answers["text"][0])
            num_begin_token = 0
            while id_seq[num_begin_token] != (1 if rightside_pad else 0):
                num_begin_token += 1
            num_last_token = len(ip_ID) - 1
            while id_seq[num_last_token] != (1 if rightside_pad else 0):
                num_last_token -= 1
            if not (offsets[num_begin_token][0] <= ch_begin and offsets[num_last_token][1] >= ch_last):
                exmpl_aftTokenizer["start_positions"].append(num_cls)
                exmpl_aftTokenizer["end_positions"].append(num_cls)
            else:
                while num_begin_token < len(offsets) and offsets[num_begin_token][0] <= ch_begin:
                    num_begin_token += 1
                exmpl_aftTokenizer["start_positions"].append(num_begin_token - 1)
                while offsets[num_last_token][1] >= ch_last:
                    num_last_token -= 1
                exmpl_aftTokenizer["end_positions"].append(num_last_token + 1)
    return exmpl_aftTokenizer

def quesAns_pred_postProcess(examples, features, raw_predictions, tokenizer, n_best_size = 20, max_answer_length = 30):
    logit_begin_all, logit_last_all = raw_predictions
    ind_ID_convExpl = {k: i for i, k in enumerate(examples["id"])}
    expl_sampFeature = collections.defaultdict(list)
    for i, feature in enumerate(features):
        expl_sampFeature[ind_ID_convExpl[feature["example_id"]]].append(i)
    predictions = collections.OrderedDict()
    print(f"{len(features)} features split from {len(examples)} prediction samples for Post-processing")
    for example_index, example in enumerate(tqdm(examples)):
        ind_characFeatures = expl_sampFeature[example_index]
        null_minVal = None
        ans_valid = []   
        context = example["context"]
        for feature_index in ind_characFeatures:
            logits_begin = logit_begin_all[feature_index]
            logits_last = logit_last_all[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            num_cls = features[feature_index]["ip_ID"].index(tokenizer.cls_token_id)
            feature_null_score = logits_begin[num_cls] + logits_last[num_cls]
            if null_minVal is None or null_minVal < feature_null_score:
                null_minVal = feature_null_score
            indexes_begin = np.argsort(logits_begin)[-1 : -n_best_size - 1 : -1].tolist()
            indexes_last = np.argsort(logits_last)[-1 : -n_best_size - 1 : -1].tolist()
            for ind_begin in indexes_begin:
                for ind_last in indexes_last:
                    if (
                        ind_begin >= len(offset_mapping)
                        or ind_last >= len(offset_mapping)
                        or offset_mapping[ind_begin] is None
                        or offset_mapping[ind_last] is None
                    ):
                        continue
                    if ind_last < ind_begin or ind_last - ind_begin + 1 > max_answer_length:
                        continue
                    
                    try:
                        ch_begin = offset_mapping[ind_begin][0]
                        ch_last = offset_mapping[ind_last][1]
                    except:
                        continue
                    ans_valid.append(
                        {
                            "score": logits_begin[ind_begin] + logits_last[ind_last],
                            "text": context[ch_begin: ch_last]
                        }
                    )     
        if len(ans_valid) > 0:
            best_answer = sorted(ans_valid, key=lambda x: x["score"], reverse=True)[0]
        else:

            best_answer = {"text": "", "score": 0.0}
        predictions[example["id"]] = best_answer["text"]
    return predictions

def dataset_importAndProcess(args, split, mode, tokenizer):

    print(f'Loading {split} data for {mode}...')
    
    if args.dataset == 'chaii':
        if split == 'train' and args.dataset_augmentation == 'translation':
            data_path =  f'data/chaii-trans/train_translated_train_k{args.dataset_split_k}.csv'
        elif split == 'train' and args.dataset_augmentation == 'transliteration':
            data_path =  f'data/chaii-trans/train_transliterated_train_k{args.dataset_split_k}.csv'
        elif split == 'test':
            data_path =  f'data/chaii/train_test_k{args.dataset_split_k}.csv'
        elif split == 'val':
            data_path =  f'data/chaii/train_val_k{args.dataset_split_k}.csv'

        if mode == 'train':
            map_fn = train_characModify
            langs = args.langs
        elif mode == 'eval':
            map_fn = test_characModify
            langs = [lang for lang in args.langs if not lang.endswith('^')]

        df = pd.read_csv(data_path)
        if 'language' not in df.columns:
            df['language'] = df['tgt'] 
        df['language'] = df['language'].apply(chaii_dataset_newformat)
        df = df[df['language'].isin(['hi', 'ta', 'bn^', 'mr^', 'ml^', 'te^'])]
        if split == 'train' and mode == 'train' and args.min_langs > 1:
            df = df[df['language'].isin(args.langs_for_min_langs_filter)]
            id_counts = df.id.value_counts()
            ids_filtered = id_counts[id_counts>=args.min_langs].index
            df = df[df['id'].isin(ids_filtered)]
        df = df[df['language'].isin(langs)]
        if split == 'train' and mode == 'train':
            ids_filtered = df[df['is_original']==True]['id']
            df = df[df['id'].isin(ids_filtered)]

        df = df.reset_index(drop=True)
        df['answers'] = df[['answer_start', 'answer_text']].apply(conv_ansText, axis=1)
        hf_dataset = Dataset.from_pandas(df)

        hf_dataset_tokenized = hf_dataset.map(map_fn, fn_kwargs={'args':args, 'tokenizer': tokenizer},
         batched=True, batch_size=len(hf_dataset), remove_columns=hf_dataset.column_names)
        print(f'Length: {len(hf_dataset)} -> {len(hf_dataset_tokenized)}')
        print(f'Data loading is COMPLETED')
        print('-'*50)
        return hf_dataset, hf_dataset_tokenized
    else:
        raise NotImplementedError()

def add_pair_idx_column(dataset_train, dataset_train_tokenized):

    print('Pair matching is STARTED...')
    dataset_train_tokenized = dataset_train_tokenized.add_column('feature_idx', range(len(dataset_train_tokenized)))
    
    dataset_train_tokenized = dataset_train_tokenized.add_column('example_idx', 
        dataset_train_tokenized['overflow_to_map_references']) 
    
    example_to_source = {k:v for k, v in enumerate(dataset_train['id'])}
    dataset_train_tokenized = dataset_train_tokenized.add_column('source_idx', 
        [example_to_source[x] for x in dataset_train_tokenized['example_idx']])

    dataset_train_df = dataset_train.to_pandas()[['id', 'is_original']]
    dataset_train_df = dataset_train_df[dataset_train_df['is_original']==True]
    source_idx_to_source_example_idx= {v:k for k, v in dataset_train_df['id'].items()}
    dataset_train_tokenized = dataset_train_tokenized.add_column('source_example_idx', 
        [source_idx_to_source_example_idx[x] for x in dataset_train_tokenized['source_idx']])
    
    prev = dataset_train_tokenized['example_idx'][0]
    local_feature_idx  = 0
    local_feature_idxs = [local_feature_idx]
    for i, curr in enumerate(dataset_train_tokenized['example_idx'][1:], start=1):
        if curr == prev:
            local_feature_idx += 1
        else:
            local_feature_idx = 0
        local_feature_idxs.append(local_feature_idx)
        prev = curr
    dataset_train_tokenized = dataset_train_tokenized.add_column('local_feature_idx', local_feature_idxs)

    print("#features; #examples; #sources")
    print(len(set(dataset_train_tokenized['feature_idx'])), len(set(dataset_train_tokenized['example_idx'])), len(set(dataset_train_tokenized['source_idx'])))
    cols = [col for col in dataset_train_tokenized.column_names if col.endswith('_idx')]
    df = dataset_train_tokenized.to_pandas()[cols]
    res = df.groupby(['source_example_idx', 'local_feature_idx'])['feature_idx'].agg(list)
    get_pair_idx = lambda row: res[row['source_example_idx'], row['local_feature_idx']]
    df['pair_idx'] = df.apply(get_pair_idx, axis=1)
    dataset_train_tokenized = dataset_train_tokenized.add_column('pair_idx', df['pair_idx'].values.tolist())
    print('Pair matching is DONE.')
    print('-'*50)

    return dataset_train_tokenized