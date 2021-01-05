import tensorflow as tf
import re
import os
import json
import numpy as np
import codecs
from configs.event_config import event_config
from data_processing.event_prepare_data import EventRolePrepareMRC, EventTypeClassificationPrepare
from tensorflow.contrib import predictor

from pathlib import Path

from argparse import ArgumentParser
import datetime
import ipdb

class fastPredictTypeClassification:
    def __init__(self, model_path, config):
        self.model_path = model_path
        self.data_loader = self.init_data_loader(config)
        self.predict_fn = None
        self.config = config

    def load_models(self):
        subdirs = [x for x in Path(self.model_path).iterdir()
                   if x.is_dir() and 'temp' not in str(x)]
        latest = str(sorted(subdirs)[-1])
        predict_fn = predictor.from_saved_model(latest)
        return predict_fn

    def load_models_kfold(self, model_path):
        subdirs = [x for x in Path(model_path).iterdir()
                   if x.is_dir() and 'temp' not in str(x)]
        latest = str(sorted(subdirs)[-1])
        predict_fn = predictor.from_saved_model(latest)
        return predict_fn

    def init_data_loader(self, config):
        event_type_file = os.path.join(config.get("slot_list_root_path"), config.get("event_type_file"))
        # event_type_file = "data/slot_pattern/vocab_all_role_label_noBI_map.txt"

        vocab_file_path = os.path.join(config.get(
            "bert_pretrained_model_path"), config.get("vocab_file"))
        data_loader = EventTypeClassificationPrepare(vocab_file_path, 512, event_type_file)
        # data_loader = EventRoleClassificationPrepare(
        #     vocab_file_path, 512, event_type_file)
        return data_loader

    def parse_test_json(self, test_file):
        id_list = []
        text_list = []
        with codecs.open(test_file, 'r', 'utf-8') as fr:
            for line in fr:
                line = line.strip()
                line = line.strip("\n")
                d_json = json.loads(line)
                id_list.append(d_json.get("id"))
                text_list.append(d_json.get("text"))
        return id_list, text_list

    def predict_single_sample(self, text):
        words_input, token_type_ids = self.data_loader.trans_single_data_for_test(
            text)
        predictions = self.predict_fn({'words': [words_input], 'text_length': [
            len(words_input)], 'token_type_ids': [token_type_ids]})
        label = predictions["output"][0]
        return np.argwhere(label > 0.5)

    def predict_single_sample_prob(self, predict_fn, text):
        words_input, token_type_ids, type_index_in_token_ids = self.data_loader.trans_single_data_for_test(
            text)
        predictions = predict_fn({'words': [words_input], 'text_length': [
            len(words_input)], 'token_type_ids': [token_type_ids], 'type_index_in_ids_list': [type_index_in_token_ids]})
        label = predictions["output"][0]
        return label

    def predict_for_all_prob(self, predict_fn, text_list):
        event_type_prob = []
        for text in text_list:
            prob_output = self.predict_single_sample_prob(predict_fn, text)
            event_type_prob.append(prob_output)
        return event_type_prob

    def predict_for_all(self, text_list):
        event_result_list = []
        for text in text_list:
            label_output = self.predict_single_sample(text)
            event_cur_type_list = [self.data_loader.id2labels_map.get(
                ele[0]) for ele in label_output]
            event_result_list.append(event_cur_type_list)
        return event_result_list


class fastPredictCls:
    def __init__(self, model_path, config, query_map_file):
        self.model_path = model_path
        self.data_loader = self.init_data_loader(config, query_map_file)
        self.predict_fn = None
        self.config = config

    def load_models_kfold(self, model_path):
        subdirs = [x for x in Path(model_path).iterdir()
                   if x.is_dir() and 'temp' not in str(x)]
        latest = str(sorted(subdirs)[-1])
        predict_fn = predictor.from_saved_model(latest)
        return predict_fn

    def init_data_loader(self, config, query_map_file):
        vocab_file_path = os.path.join(config.get(
            "bert_pretrained_model_path"), config.get("vocab_file"))
        slot_file = os.path.join(event_config.get("slot_list_root_path"),
                                 event_config.get("bert_slot_complete_file_name_role"))
        schema_file = os.path.join(event_config.get(
            "data_dir"), event_config.get("event_schema"))
        # query_map_file = os.path.join(event_config.get(
        #         "slot_list_root_path"), event_config.get("query_map_file"))
        data_loader = EventRolePrepareMRC(
            vocab_file_path, 512, slot_file, schema_file, query_map_file)
        return data_loader

    def parse_test_json(self, test_file):
        id_list = []
        text_list = []
        with codecs.open(test_file, 'r', 'utf-8') as fr:
            for line in fr:
                line = line.strip()
                line = line.strip("\n")
                d_json = json.loads(line)
                id_list.append(d_json.get("id"))
                text_list.append(d_json.get("text"))
        return id_list, text_list

    def predict_single_sample_prob(self, predict_fn, text, query_len, token_type_ids):
        # words_input, token_type_ids,type_index_in_token_ids = self.data_loader.trans_single_data_for_test(
        #     text)
        text_length = len(text)
        predictions = predict_fn({'words': [text], 'text_length': [text_length],
                                  'token_type_ids': [token_type_ids]})
        label_prob = predictions["output"][0]
        return label_prob

    def predict_single_sample_av_prob(self, predict_fn, text, query_len, token_type_ids):
        text_length = len(text)
        predictions = predict_fn({'words': [text], 'text_length': [text_length], 'query_length': [query_len],
                                  'token_type_ids': [token_type_ids]})
        start_ids, end_ids, start_probs, end_probs, has_answer_probs = predictions.get("start_ids"), predictions.get(
            "end_ids"), predictions.get("start_probs"), predictions.get("end_probs"), predictions.get(
            "has_answer_probs")
        return start_ids[0], end_ids[0], start_probs[0], end_probs[0], has_answer_probs[0]

    # def predict_for_all_prob(self,predict_fn,text_list):
    #     event_type_prob = []
    #     for text in text_list:
    #         prob_output = self.predict_single_sample_prob(predict_fn,text)
    #         event_type_prob.append(prob_output)
    #     return event_type_prob
    def extract_entity_from_start_end_ids(self, text, start_ids, end_ids, token_mapping):
        # 根据开始，结尾标识，找到对应的实体
        entity_list = []
        start_end_tuple_list = []
        # text_cur_index = 0
        for i, start_id in enumerate(start_ids):
            if start_id == 0:
                # text_cur_index += len(token_mapping[i])
                continue
            if end_ids[i] == 1:
                # start and end is the same
                entity_str = "".join([text[char_index]
                                      for char_index in token_mapping[i]])
                # entity_str = text[text_cur_index:text_cur_index+cur_entity_len]
                entity_list.append(entity_str)
                start_end_tuple_list.append((i, i))
                # text_cur_index += len(token_mapping[i])
                continue
            j = i + 1
            find_end_tag = False
            while j < len(end_ids):
                # 若在遇到end=1之前遇到了新的start=1,则停止该实体的搜索
                if start_ids[j] == 1:
                    break
                if end_ids[j] == 1:
                    entity_str_index_list = []
                    for index in range(i, j + 1):
                        entity_str_index_list.extend(token_mapping[index])
                    start_end_tuple_list.append((i, j))
                    entity_str = "".join([text[char_index]
                                          for char_index in entity_str_index_list])
                    entity_list.append(entity_str)
                    find_end_tag = True
                    break
                else:
                    j += 1
            if not find_end_tag:
                entity_str = "".join([text[char_index]
                                      for char_index in token_mapping[i]])
                start_end_tuple_list.append((i, i))
                entity_list.append(entity_str)
        return entity_list, start_end_tuple_list


class fastPredictMRC:
    def __init__(self, model_path, config, model_type):
        self.model_path = model_path
        self.model_type = model_type
        self.data_loader = self.init_data_loader(config, model_type)
        # self.predict_fn = self.load_models()
        self.config = config

    def load_models(self, model_path):
        subdirs = [x for x in Path(model_path).iterdir()
                   if x.is_dir() and 'tmp' not in str(x)]
        latest = str(sorted(subdirs)[-1])
        # print(latest)
        predict_fn = predictor.from_saved_model(latest)
        return predict_fn

    def init_data_loader(self, config, model_type):
        vocab_file_path = os.path.join(config.get(
            "bert_pretrained_model_path"), config.get("vocab_file"))
        slot_file = os.path.join(event_config.get("slot_list_root_path"),
                                     event_config.get("bert_slot_complete_file_name_role"))
        schema_file = os.path.join(event_config.get(
                "data_dir"), event_config.get("event_schema"))
        query_map_file = os.path.join(event_config.get(
                "slot_list_root_path"), event_config.get("query_map_file"))
        data_loader = EventRolePrepareMRC(
                vocab_file_path, 512, slot_file, schema_file, query_map_file)

        return data_loader

    def parse_test_json(self, test_file):
        id_list = []
        text_list = []
        with codecs.open(test_file, 'r', 'utf-8') as fr:
            for line in fr:
                line = line.strip()
                line = line.strip("\n")
                d_json = json.loads(line)
                id_list.append(d_json.get("id"))
                text_list.append(d_json.get("text"))
        return id_list, text_list

    def predict_single_sample(self, predict_fn, text, query_len, token_type_ids):
        text_length = len(text)
        # print(text)

        predictions = predict_fn({'words': [text], 'text_length': [text_length], 'query_length': [query_len],
                                  'token_type_ids': [token_type_ids]})
        # print(predictions)
        # start_ids, end_ids,start_probs,end_probs = predictions.get("start_ids"), predictions.get("end_ids"),predictions.get("start_probs"), predictions.get("end_probs")
        pred_ids, pred_probs = predictions.get("pred_ids"), predictions.get("pred_probs")
        # return start_ids[0], end_ids[0],start_probs[0], end_probs[0]
        return pred_ids[0], pred_probs[0]

    def extract_entity_from_start_end_ids(self, text, start_ids, end_ids, token_mapping):
        # 根据开始，结尾标识，找到对应的实体
        entity_list = []
        # text_cur_index = 0
        for i, start_id in enumerate(start_ids):
            if start_id == 0:
                # text_cur_index += len(token_mapping[i])
                continue
            if end_ids[i] == 1:
                # start and end is the same
                entity_str = "".join([text[char_index]
                                      for char_index in token_mapping[i]])
                # entity_str = text[text_cur_index:text_cur_index+cur_entity_len]
                entity_list.append(entity_str)
                # text_cur_index += len(token_mapping[i])
                continue
            j = i + 1
            find_end_tag = False
            while j < len(end_ids):
                # 若在遇到end=1之前遇到了新的start=1,则停止该实体的搜索
                if start_ids[j] == 1:
                    break
                if end_ids[j] == 1:
                    entity_str_index_list = []
                    for index in range(i, j + 1):
                        entity_str_index_list.extend(token_mapping[index])

                    entity_str = "".join([text[char_index]
                                          for char_index in entity_str_index_list])
                    entity_list.append(entity_str)
                    find_end_tag = True
                    break
                else:
                    j += 1
            if not find_end_tag:
                entity_str = "".join([text[char_index]
                                      for char_index in token_mapping[i]])
                entity_list.append(entity_str)
        return entity_list

def parse_event_schema(json_file):
    event_type_role_dict = {}
    with codecs.open(json_file, 'r', 'utf-8') as fr:
        for line in fr:
            line = line.strip()
            line = line.strip("\n")
            data_json = json.loads(line)
            event_type = data_json.get("event_type")
            role_list = data_json.get("role_list")
            role_name_list = [ele.get("role") for ele in role_list]
            event_type_role_dict.update({event_type: role_name_list})
    return event_type_role_dict


def parse_schema_type(schema_file):
    schema_event_dict = {}
    with codecs.open(schema_file, 'r', 'utf-8') as fr:
        for line in fr:
            s_json = json.loads(line)
            role_list = s_json.get("role_list")
            schema_event_dict.update(
                {s_json.get("event_type"): [ele.get("role") for ele in role_list]})
    return schema_event_dict


def extract_entity_span_from_muliclass(text, pred_ids, token_mapping):
    buffer_list = []
    entity_list = []
    for index, label in enumerate(pred_ids):
        if label == 0:
            if buffer_list:
                entity_str_index_list = []
                for i in buffer_list:
                    entity_str_index_list.extend(token_mapping[i])
                entity_str = "".join([text[char_index]
                                      for char_index in entity_str_index_list])
                entity_list.append(entity_str)
                buffer_list.clear()
            continue
        elif label == 1:
            if buffer_list:
                entity_str_index_list = []
                for i in buffer_list:
                    entity_str_index_list.extend(token_mapping[i])
                entity_str = "".join([text[char_index]
                                      for char_index in entity_str_index_list])
                entity_list.append(entity_str)
                buffer_list.clear()
            buffer_list.append(index)
        else:
            if buffer_list:
                buffer_list.append(index)
    if buffer_list:
        entity_str_index_list = []
        for i in buffer_list:
            entity_str_index_list.extend(token_mapping[i])
        entity_str = "".join([text[char_index]
                              for char_index in entity_str_index_list])
        entity_list.append(entity_str)
    return entity_list

def parse_kfold_verify(args):
    
    if(args.gpus is not None):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # test_file = "data/test1.json"
    # Path of test dataset json file
    test_file = os.path.join(event_config.get("data_dir"), event_config.get("event_data_file_test"))
    # Path of text multi label classification saved model
    class_type_model_path = event_config.get(args.event_type_model_path)
    event_schema_file = os.path.join(event_config.get("data_dir"), event_config.get("event_schema"))
    event_schema_dict = parse_event_schema(event_schema_file)
    # multi label event type classifer
    fp_type = fastPredictTypeClassification(class_type_model_path, event_config)
    # parse json file to get id and text
    id_list, text_list = fp_type.parse_test_json(test_file)
    # 
    kfold_type_result_list = [] # for prediction in 65 probabilities
    event_type_result_list = [] # for result event type name
    for k in range(1):
        predict_fn = fp_type.load_models_kfold(class_type_model_path.format(k))
        cur_fold_event_type_probs = fp_type.predict_for_all_prob(predict_fn,text_list)
        kfold_type_result_list.append(cur_fold_event_type_probs)

    for i in range(len(text_list)):
        cur_sample_event_type_buffer = [ele[i] for ele in kfold_type_result_list]
        cur_sample_event_type_prob = np.array(cur_sample_event_type_buffer).reshape((-1,65))
        avg_result = np.mean(cur_sample_event_type_prob,axis=0)
        event_label_ids = np.argwhere(avg_result > 0.5)
        event_cur_type_strs = [fp_type.data_loader.id2labels_map.get(
                ele[0]) for ele in event_label_ids]
        event_type_result_list.append(event_cur_type_strs)

    # path of Answerable Verificaion model to predict whether a query is answerable, 
    # 第一阶段 粗读 的 answaerable verifier ,
    # External Front Verifier 
    exterinal_av_model_path = "output/model/final_verify_cls_fold_{}_usingtype_roberta_large_traindev_event_role_bert_mrc_model_desmodified_lowercase/saved_model"
    # verify_av_model_path_old = event_config.get(args.event_verfifyav_model_path)
    verify_av_model_path_old = "output/model/verify_avmrc_fold_{}_usingtype_roberta_large_traindev_event_role_bert_mrc_model_desmodified_lowercase/saved_model"
    # 第二阶段 精读 的 answaerable verifier ,
    # Internal Front Verifier 。 
    interbal_av_model_path = "output/model/final_verify_avmrc_fold_{}_usingtype_roberta_large_traindev_event_role_bert_mrc_model_desmodified_lowercase/saved_model"
    
    fp_cls_old = fastPredictCls(exterinal_av_model_path, event_config, "data/slot_pattern/slot_descrip_old")
    # fp_cls_new = fastPredictCls(cls_model_path, event_config, "data/slot_pattern/slot_descrip")
    fp_answerable_verifier = fastPredictCls(exterinal_av_model_path, event_config, "data/slot_pattern/slot_descrip")

    kfold_eav_hasa_result = []
    kfold_start_result = []
    kfold_end_result = []
    kfold_hasa_result = []
    
    for k in range(1):
        
        # predict_fn_cls_new = fp_answerable_verifier.load_models_kfold(external_av_model_path.format(k))
        # 粗读fn 
        predict_fn_ex_av = fp_answerable_verifier.load_models_kfold(exterinal_av_model_path.format(k))
        # predict_fn_av = fp_cls_new.load_models_kfold(verify_av_model_path_new.format(k))
        # 精读fn
        predict_fn_in_av = fp_answerable_verifier.load_models_kfold(interbal_av_model_path.format(k))
        
        cur_fold_eav_probs_result = {}
        cur_fold_av_start_probs_result = {}
        cur_fold_av_end_probs_result = {}
        cur_fold_av_has_answer_probs_result = {}
        
        for sample_id, event_type_res, text in zip(id_list, event_type_result_list, text_list):
            
            

            if event_type_res is None or len(event_type_res) == 0:
                # submit_result.append({"id": sample_id, "event_list": []})
                cur_fold_eav_probs_result.update({sample_id: []})
                continue
            for cur_event_type in event_type_res:
                cur_event_type = cur_event_type.strip()
                if cur_event_type is None or cur_event_type == "":
                    continue
                corresponding_role_type_list = event_schema_dict.get(cur_event_type)
                cur_event_type_answerable_probs_result = []
                cur_event_av_start_probs_result = []
                cur_event_av_end_probs_result = []
                cur_event_av_hasanswer_probs_result = []
                for cur_role_type in corresponding_role_type_list:
                    
                    has_answer_probs = None
                    label_prob = None
                    start_probs = None
                    end_probs = None
                    
                    cur_query_word = fp_answerable_verifier.data_loader.gen_query_for_each_sample(
                        cur_event_type, cur_role_type)
                    query_token_ids, query_token_len, token_type_ids_ , token_mapping_new = fp_answerable_verifier.data_loader.trans_single_data_for_test(
                        text, cur_query_word, 512)
                    
                    #############################################################################
                    ## Exterinal Answerable Verify, predict answerable probs
                    eav_probs = fp_answerable_verifier.predict_single_sample_prob(predict_fn_ex_av, query_token_ids,
                                                                           query_token_len, token_type_ids_ )
                    #############################################################################
                    # Internal Answerable Verify ，predict start&end labe and answerable probs
                    role_start_ids, role_end_ids, role_start_probs, role_end_probs, iav_probs = fp_cls_old.predict_single_sample_av_prob(
                        predict_fn_in_av, query_token_ids, query_token_len, token_type_ids_ )
                    
                    cur_event_type_answerable_probs_result.append(eav_probs)
                    cur_event_av_hasanswer_probs_result.append(iav_probs)
                    cur_event_av_start_probs_result.append(role_start_probs)
                    cur_event_av_end_probs_result.append(role_end_probs)
                
                cur_fold_eav_probs_result.update({sample_id + "-" + cur_event_type: cur_event_type_answerable_probs_result})
                cur_fold_av_start_probs_result.update(
                    {sample_id + "-" + cur_event_type: cur_event_av_start_probs_result})
                cur_fold_av_end_probs_result.update({sample_id + "-" + cur_event_type: cur_event_av_end_probs_result})
                cur_fold_av_has_answer_probs_result.update(
                    {sample_id + "-" + cur_event_type: cur_event_av_hasanswer_probs_result})
        kfold_eav_hasa_result.append(cur_fold_eav_probs_result)
        kfold_start_result.append(cur_fold_av_start_probs_result)
        kfold_end_result.append(cur_fold_av_end_probs_result)
        kfold_hasa_result.append(cur_fold_av_has_answer_probs_result)

    submit_result = []

    for sample_id, event_type_res, text in zip(id_list, event_type_result_list, text_list):
        event_list = []
        if event_type_res is None or len(event_type_res) == 0:
            submit_result.append({"id": sample_id, "event_list": []})
            continue
        for cur_event_type in event_type_res:
            cur_event_type = cur_event_type.strip()
            if cur_event_type is None or cur_event_type == "":
                continue
            corresponding_role_type_list = event_schema_dict.get(cur_event_type)
            find_key = sample_id + "-" + cur_event_type
            fold_cls_probs_cur_sample = [ele.get(find_key) for ele in kfold_eav_hasa_result]
            fold_start_probs_cur_sample = [ele.get(find_key) for ele in kfold_start_result]
            fold_end_probs_cur_sample = [ele.get(find_key) for ele in kfold_end_result]
            fold_has_probs_cur_sample = [ele.get(find_key) for ele in kfold_hasa_result]
            for index, cur_role_type in enumerate(corresponding_role_type_list):
                cur_eav_fold_probs = [probs[index] for probs in fold_cls_probs_cur_sample]
                
                cur_iav_hasa_fold_probs = [probs[index] for probs in fold_has_probs_cur_sample]

                cur_eav_fold_probs = np.array(cur_eav_fold_probs).reshape((-1, 1))
                cls_avg_result = np.mean(cur_eav_fold_probs, axis=0)

                cur_iav_hasa_fold_probs = np.array(cur_iav_hasa_fold_probs).reshape((-1, 1))
                has_avg_result = np.mean(cur_iav_hasa_fold_probs, axis=0)

                ######
                # EAV * 0.5 + IAV * 0.5
                final_probs_hasa = 0.5 * (cls_avg_result) + 0.5 * (has_avg_result)

                if final_probs_hasa > 0.4:
                    
                    cur_query_word = fp_answerable_verifier.data_loader.gen_query_for_each_sample(
                        cur_event_type, cur_role_type)
                    token_ids, query_len, token_type_ids, token_mapping = fp_answerable_verifier.data_loader.trans_single_data_for_test(
                        text, cur_query_word, 512)

                    token_len = len(token_ids)
                    
                    cur_start_fold_probs = [probs[index] for probs in fold_start_probs_cur_sample]
                    cur_end_fold_probs = [probs[index] for probs in fold_end_probs_cur_sample]

                    cur_start_fold_probs = np.array(cur_start_fold_probs).reshape((-1, token_len, 2))
                    cur_end_fold_probs = np.array(cur_end_fold_probs).reshape((-1, token_len, 2))
                    start_avg_result = np.mean(cur_start_fold_probs, axis=0)
                    end_avg_result = np.mean(cur_end_fold_probs, axis=0)

                    text_start_probs = start_avg_result[query_len:-1, 1]
                    text_end_probs = end_avg_result[query_len:-1, 1]

                    pos_start_probs = (text_start_probs)
                    pos_end_probs = (text_end_probs)

                    start_ids = (pos_start_probs > 0.4).astype(int)
                    
                    end_ids = (pos_end_probs > 0.4).astype(int)
                    token_mapping = token_mapping[1:-1]

                    entity_list, span_start_end_tuple_list = fp_answerable_verifier.extract_entity_from_start_end_ids(
                        text=text, start_ids=start_ids, end_ids=end_ids, token_mapping=token_mapping)
                    
                    for entity in entity_list:
                        if len(entity) > 1:
                            event_list.append({"event_type": cur_event_type, "arguments": [
                                {"role": cur_role_type, "argument": entity}]})
        submit_result.append({"id": sample_id, "event_list": event_list})
    

    with codecs.open(args.submit_result, 'w', 'utf-8') as fw:
        for dict_result in submit_result:
            write_str = json.dumps(dict_result, ensure_ascii=False)
            fw.write(write_str)
            fw.write("\n")
    
    print("finish")

if __name__ == '__main__':
    print(os.listdir("data/slot_pattern/"))
    parser = ArgumentParser()
    parser.add_argument("--model_trigger_pb_dir",
                        default='bert_model_pb', type=str)
    parser.add_argument("--model_role_pb_dir",
                        default='role_bert_model_pb', type=str)
    parser.add_argument("--trigger_predict_res",
                        default="trigger_result.json", type=str)
    parser.add_argument("--submit_result",
                        default="./data/test2allmerge_modifeddes_prob4null_0404threold_8epoch_type05_verify_kfold_notav_modifyneg_dropout15moretype_roberta_large.json",
                        type=str)
    parser.add_argument("--multi_task_model_pb_dir",
                        default="multi_task_bert_model_pb", type=str)
    parser.add_argument("--event_type_model_path",
                        default="type_class_bert_model_pb", type=str)
    parser.add_argument("--event_cls_model_path",
                        default="role_verify_cls_bert_model_pb", type=str)
    parser.add_argument("--event_verfifyav_model_path",
                        default="role_verify_avmrc_bert_model_pb", type=str)
    parser.add_argument("--gpus", type=str, help="cuda visible devices")
    # parser.add_argument("--model_pb_dir", default='base_pb_model_dir', type=str)
    args = parser.parse_args()
    parse_kfold_verify(args)
