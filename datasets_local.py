import collections
import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm

def reformat_lang_chaii(lang):
    if len(lang) > 3:
        return lang[:2]
    else:
        return lang+'^'

def convert_answers(r):
    start = r[0]
    text = r[1]
    return {
        'answer_start': [start],
        'text': [text]
    }

def prepare_train_features(examples, args, tokenizer):
    pad_on_right = tokenizer.padding_side == "right"
    examples["question"] = [q.lstrip() for q in examples["question"]].
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=args.max_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized_examples.get("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

def prepare_test_features(examples, args, tokenizer):
    pad_on_right = tokenizer.padding_side == "right"
    examples["question"] = [q.lstrip() for q in examples["question"]]
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=args.max_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

def postprocess_qa_predictions(examples, features, raw_predictions, tokenizer, n_best_size = 20, max_answer_length = 30):
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
    predictions = collections.OrderedDict()
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")
    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]
        min_null_score = None
        valid_answers = []   
        context = example["context"]
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    
                    try:
                        start_char = offset_mapping[start_index][0]
                        end_char = offset_mapping[end_index][1]
                    except:
                        continue
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )     
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:

            best_answer = {"text": "", "score": 0.0}
        predictions[example["id"]] = best_answer["text"]
    return predictions

def load_dataset(args, split, mode, tokenizer):

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
            map_fn = prepare_train_features
            langs = args.langs
        elif mode == 'eval':
            map_fn = prepare_test_features
            langs = [lang for lang in args.langs if not lang.endswith('^')]

        df = pd.read_csv(data_path)
        if 'language' not in df.columns:
            df['language'] = df['tgt'] 
        df['language'] = df['language'].apply(reformat_lang_chaii)
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
        df['answers'] = df[['answer_start', 'answer_text']].apply(convert_answers, axis=1)
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
        dataset_train_tokenized['overflow_to_sample_mapping']) 
    
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