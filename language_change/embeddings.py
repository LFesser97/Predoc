"""
embeddings.py

Created on Mon Jul 31 2023

@author: Lukas

This file contains all code for creating sbert embeddings.
"""

import argparse
import torch
import json
import logging
import numpy as np
import pandas as pd
import pickle
import random
import os

from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import defaultdict

# import utils.utils as utils
# from sentence_transformers import SentenceTransformer 


logging.basicConfig(level=logging.INFO)


class SBERTContextualEmbeddings:
    """Takes an SBERT model,and returns contextual embeddings for a given sentence"""
    def __init__(self, model):
        self.model=model
    
    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
        tokenized_sentences = {"input_ids": input_ids, "attention_mask": attention_mask}
        return self.model.forward(tokenized_sentences)["token_embeddings"]

        
def get_token_embeddings(sentences: list, model, tokenizer, batch_size: int=32, max_len: int=512,
                         device: str='cuda', pooling_type: str='concat', model_type: str='bert') -> tuple:
    """ 
    Function to get embedding for all tokens for all sentences using a pretrained modelgoo in HuggingFace. 
    """
    model.to(device)
    model.eval()

    embeddings = []
    input_ids = []
    attention_mask = []

    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i + batch_size]
        batch = tokenizer(batch, padding='max_length', truncation=True, max_length=max_len, return_tensors="pt")
        input_ids.append(batch['input_ids'])
        attention_mask.append(batch['attention_mask'])

        with torch.no_grad():
            if model_type=='sbert':
                outputs=SBERTContextualEmbeddings(model).forward(**batch.to(device))
                embeddings.append(outputs)
            else:
                outputs = model(**batch.to(device), output_hidden_states=True)
                # Take the last 4 hidden states and mean pool them
                last_4_layers = outputs.hidden_states[-4:]
                if pooling_type == 'mean':
                    pooled = torch.stack(last_4_layers, dim=0).mean(dim=0)
                elif pooling_type == 'max':
                    pooled = torch.stack(last_4_layers, dim=0).max(dim=0)[0]
                elif pooling_type=='concat':
                    pooled = torch.cat(last_4_layers, dim=-1)
                embeddings.append(pooled)

    # Unbatching
    embeddings = torch.cat(embeddings, dim=0)
    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)

    return embeddings, input_ids, attention_mask


def remove_padding_tokens(embeddings: torch.tensor, input_ids: torch.tensor,
                          attention_mask: torch.tensor) -> tuple:
    """
    Remove padding tokens from the embeddings.
    """
    # Get the indices of the tokens that are not padding for each sentence
    non_padding_indices = [torch.nonzero(i) for i in attention_mask]

    # Get the embeddings for the non padding tokens
    non_padding_embeddings = [embeddings[i, j] for i, j in enumerate(non_padding_indices)]

    # Get the input_ids for the non padding tokens
    non_padding_ids = [input_ids[i, j] for i, j in enumerate(non_padding_indices)]

    # Get the attention_mask for the non padding tokens
    non_padding_attention_mask = [attention_mask[i, j] for i, j in enumerate(non_padding_indices)]

    return non_padding_embeddings, non_padding_ids, non_padding_attention_mask
    

def convert_input_ids_to_tokens(input_id_list: list, tokenizer) -> list:
    """
    Convert the token ids to tokens
    """
    return [tokenizer.convert_ids_to_tokens(input_ids) for input_ids in input_id_list]


# Wordpiece aggregation - take the mean of the token embeddings for each wordpiece
def tokens_generator(toks):
	last_token = ""
	i = 0
	token_start = 0
	while i < len (toks):
		if i == 0:
			last_token = toks[i]
			token_start = i
		elif toks[i].startswith ("##"):
			last_token = last_token + toks[i][2:]
		else:
			yield token_start, i, last_token
			last_token = toks[i]
			token_start = i
		i += 1

	if len (last_token) > 0:
		yield token_start, i, last_token    


def aggregate_wordpiece(tokens_list: list, embedding_tensor: torch.tensor) -> tuple:
    """
    Aggregate the wordpieces to get the word embeddings
    """
    final_token_list=[]
    final_embedding_list=[]

    for i in range(len(tokens_list)):
        tokens=[token for token in tokens_generator(tokens_list[i])]
        tokenized_text = [token for _,_,token in tokens]
        token_boundaries = [(start, ended) for start, ended, _ in tokens]
        token_embeddings = torch.stack([embedding_tensor[i][start:end,:].mean(dim=0) for start, end in token_boundaries])
        final_token_list.append(tokenized_text)
        final_embedding_list.append(token_embeddings)

    return final_token_list,final_embedding_list


def combine_means_with_tokens(token_list: list, token_embeddings_mean_list: list) -> dict:
    """
    Combine the token embeddings with the input ids - a list of means for each token
    """
    # Convert input_ids_list_flat to np array, convert token_embeddings_mean_list to np array
    token_embeddings_mean_list=[ x.cpu().numpy() for x in token_embeddings_mean_list]
    token_embedding_dict=defaultdict(list)

    for i in range(len(token_list)):
        for j in range(len(token_list[i])):
            token_embedding_dict[token_list[i][j]].append({i:token_embeddings_mean_list[i][j]})
        
    return token_embedding_dict


def get_token_embed_dict(sentences, model, tokenizer, batch_size=32, max_len=512,
                         device='cuda',pooling_type="concat",model_type="bert") -> dict:
    """
    Get the token embeddings for each word in the sentence (mean-aggregated wordpiece
    embeddings) in the form word:embedding_list
    """
    embeddings, input_ids, attention_mask = get_token_embeddings(sentences, model, tokenizer, batch_size, max_len, device,pooling_type,model_type)
    non_padding_embeddings, non_padding_ids, non_padding_attention_mask = remove_padding_tokens(embeddings, input_ids, attention_mask)
    token_list=convert_input_ids_to_tokens(non_padding_ids, tokenizer)
    token_list,aggregated_embeddings=aggregate_wordpiece(token_list,non_padding_embeddings)
    token_embedding_dict=combine_means_with_tokens(token_list,aggregated_embeddings)

    return token_embedding_dict


def get_paper_metadata(data: list) -> pd.DataFrame:
    """
    Get the paper's metadata.
    """
    text_list=[]
    for item in data:
        text_list.append(item["abstract"])

    year_list=[]
    for item in data:
        year_list.append(item["year"])
    
    unique_year_list=list(set(year_list))
    
    full_text_list=[]
    full_id_list=[]
    full_year_list=[]

    for year in tqdm(unique_year_list):
        text_list_year=[text_list[i] for i in range(len(text_list)) if year_list[i]==year]
        id_list_year=[i for i in range(len(text_list_year))]
        year_list_year=[year for _ in range(len(text_list_year))]
        full_text_list.extend(text_list_year)
        full_id_list.extend(id_list_year)
        full_year_list.extend(year_list_year)
    
    # Convert to a dataframe
    df=pd.DataFrame()
    df["id"]=full_id_list
    df["year"]=full_year_list
    df["text"]=full_text_list

    return df


def save_embed_dicts_all_years(data, model, tokenizer, batch_size=32, max_len=512,
                               device='cuda', reembed=False, savedir="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/change_detection/sbert_embedding_year/", 
                               pooling_type='concat',model_type="bert") -> dict:
    """
    Get the token embeddings for each word in the sentence (mean-aggregated wordpiece embeddings) 
    in the form word:embedding_list for all the years in the year_list
    """
    text_list=[]
    for item in data:
        text_list.append(item["abstract"])

    year_list=[]
    for item in data:
        year_list.append(item["year"])
    unique_year_list=list(set(year_list))

    if reembed==True: 
        # Delete all the saved embedding dicts
        for year in unique_year_list:
            if os.path.exists(f"{savedir}token_embedding_dict_{year}.pkl"):
                os.remove(f"{savedir}token_embedding_dict_{year}.pkl")   

        # For each year, get the token embedding dict
        unique_words_list=[]
        for year in tqdm(unique_year_list):
            text_list_year=[text_list[i] for i in range(len(text_list)) if year_list[i]==year]

            # Save the token embedding dict for each year as pickle file
            print("saving token embedding dict for year",year)
        
            token_embedd_dict_year=get_token_embed_dict(text_list_year, model, tokenizer, batch_size, max_len, device, pooling_type, model_type)
            unique_tokens_year=list(token_embedd_dict_year.keys())
            unique_words_list.extend(unique_tokens_year)
            print("Saving at",f"{savedir}token_embedding_dict_{year}.pkl")

            with open(f"{savedir}token_embedding_dict_{year}.pkl", 'wb') as f:
                pickle.dump(token_embedd_dict_year, f)
            print("Saved")
            del token_embedd_dict_year

        unique_words_list=list(set(unique_words_list))

        with open(f"{savedir}unique_words_list.pkl", 'wb') as f:
            pickle.dump(unique_words_list, f)

    # Open the unique words list
    with open(f"{savedir}unique_words_list.pkl", 'rb') as f:
        unique_words_list = pickle.load(f)
    
    embedding_dict = defaultdict(dict)

    for year in tqdm(unique_year_list):
        with open(f"{savedir}token_embedding_dict_{year}.pkl", 'rb') as f:
            token_embedding_dict = pickle.load(f)
        for word in unique_words_list:
            if word in token_embedding_dict:
                embedding_dict[word][year]=token_embedding_dict[word]
    
    # Save the embedding dict
    with open(f"{savedir}embedding_dict.pkl", 'wb') as f:
        pickle.dump(embedding_dict, f)
    
    return embedding_dict
    

def reformat_embedding_dict(embedding_dict: dict) -> dict:
    """
    Reformat the embedding dict to have the following structure:
    embedding_dict[word]={year_id_list:[year_id_list], paper_id_list:[paper_id_list],embeddings:[embedding_list]}
    """
    reformatted_embedding_dict=defaultdict(dict)

    for word in tqdm(embedding_dict.keys()):
        reformatted_embedding_dict[word]["year_id_list"]=[]
        reformatted_embedding_dict[word]["paper_id_list"]=[]
        reformatted_embedding_dict[word]["embeddings"]=[]

        for year in embedding_dict[word].keys():
            for paper_id_embedding in embedding_dict[word][year]:
                reformatted_embedding_dict[word]["year_id_list"].append(year)
                reformatted_embedding_dict[word]["paper_id_list"].append(list(paper_id_embedding.keys())[0])
                reformatted_embedding_dict[word]["embeddings"].append(list(paper_id_embedding.values())[0][0])
        
        # Convert the lists to numpy arrays
        reformatted_embedding_dict[word]["year_id_list"]=np.array(reformatted_embedding_dict[word]["year_id_list"])
        reformatted_embedding_dict[word]["paper_id_list"]=np.array(reformatted_embedding_dict[word]["paper_id_list"])
        reformatted_embedding_dict[word]["embeddings"]=np.array(reformatted_embedding_dict[word]["embeddings"])

    return reformatted_embedding_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert-base-multilingual-uncased')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    args = parser.parse_args()

    jstor_data =utils.load_jstor_data(path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/change_detection/filtered_data.jsonl")

    # Get the embeddings for each year
    embedding_dict="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/change_detection/sbert_embedding_year/embedding_dict.pkl"
    
    # Load the embedding dict
    with open(embedding_dict, 'rb') as f:
        embedding_dict = pickle.load(f)

    # Reformat the embedding dict
    reformatted_embedding_dict=reformat_embedding_dict(embedding_dict)

    # Save the reformatted embedding dict
    with open(f"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/change_detection/sbert_embedding_year/reformatted_embedding_dict.pkl", 'wb') as f:
        pickle.dump(reformatted_embedding_dict, f)

    # Open the dict
    with open(f"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/change_detection/sbert_embedding_year/reformatted_embedding_dict.pkl", 'rb') as f:
        reformatted_embedding_dict = pickle.load(f)
    

    # Break the dict into 5 chunks by keys and save each chunk as a pickle file
    keys=list(reformatted_embedding_dict.keys())
    
    # Break in 5 chunks
    random.shuffle(keys)

    keys_chunks=[keys[i:i+int(len(keys)/5)] for i in range(0, len(keys), int(len(keys)/5))]
    for i in range(len(keys_chunks)):
        chunk_dict={}
        for key in keys_chunks[i]:
            chunk_dict[key]=reformatted_embedding_dict[key]
        with open(f"/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/change_detection/sbert_embedding_year/reformatted_embedding_dict_chunk_{i}.pkl", 'wb') as f:
            pickle.dump(chunk_dict, f)
        del chunk_dict

    # Save the metadata
    metadata=get_paper_metadata(jstor_data)

    # Save the metadata as csv
    metadata.to_csv("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/change_detection/sbert_embedding_year/metadata.csv",index=False)