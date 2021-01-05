import os
from pathlib import Path

BASE_DIR = Path('slot_extraction')
event_config = {
    'data_dir': 'data',
    'vocab_file': "vocab.txt",
    'slot_list_root_path': os.path.join('data', 'slot_pattern'),
    'slot_file_name': "vocab_trigger_label_map.txt",
    'bert_slot_file_name': "vocab_trigger_label_map.txt",
    'bert_slot_complete_file_name': "vocab_trigger_label_map.txt",
    'bert_slot_complete_file_name_role': "vocab_all_slot_label_noBI_map.txt",
    'query_map_file': "slot_descrip",
    'event_type_file': "vocab_all_event_type_label_map.txt",
    'all_slot_file': "vocab_all_slot_label_map.txt",
    'log_dir': os.path.join('output', 'log'),
    'data_file_name': 'orig_data_train.txt',
    'event_data_file_train': "train.json",
    'event_data_file_eval': "dev.json",
    'event_data_file_test': "test.json",
    'train_valid_data_dir': 'train_valid_data_bert_event',
    'train_data_text_name': 'train_split_data_text.npy',
    'valid_data_text_name': 'valid_split_data_text.npy',
    'train_data_tag_name': 'train_split_data_tag.npy',
    'valid_data_tag_name': 'valid_split_data_tag.npy',
    'test_data_text_name': 'test_data_text.npy',
    'test_data_tag_name': 'test_data_tag.npy',
    "bert_pretrained_model_path":os.path.join('data','chinese_roberta_wwm_ext_L-12_H-768_A-12'),
    # "bert_pretrained_model_path": os.path.join('data', 'chinese_roberta_wwm_ext_L-12_H-1024_A-12_large'),
    # "bert_pretrained_model_path":os.path.join('data','albert_large_zh'),

    # "bert_pretrained_model_path":os.path.join('roberta_zh-master','finetune_roberta_large_wwm'),
    "bert_config_path": "bert_config.json",
    # 'bert_init_checkpoints':'model.ckpt-51000',

    'bert_init_checkpoints': 'bert_model.ckpt',
    "bert_model_dir": os.path.join('output', 'model', 'event_trigger_bert_model', 'checkpoint'),
    "bert_model_pb": os.path.join('output', 'model', 'event_trigger_bert_model', 'saved_model'),
    "role_bert_model_dir": os.path.join('output', 'model',
                                        'wwm_lr_fold_{}_usingtype_roberta_large_traindev_event_role_bert_mrc_model_desmodified_lowercase',
                                        'checkpoint'),
    "role_bert_model_pb": os.path.join('output', 'model',
                                       'wwm_lr_fold_{}_usingtype_roberta_large_traindev_event_role_bert_mrc_model_desmodified_lowercase',
                                       'saved_model'),
    "student_role_bert_model_dir": os.path.join('output', 'model',
                                                'student_usingtype_roberta_large_traindev_event_role_bert_mrc_model_desmodified_lowercase',
                                                'checkpoint'),
    "student_role_bert_model_pb": os.path.join('output', 'model',
                                               'student_usingtype_roberta_large_traindev_event_role_bert_mrc_model_desmodified_lowercase',
                                               'saved_model'),
    "merge_role_bert_model_dir": os.path.join('output', 'model',
                                              'merge_usingtype_roberta_traindev_event_role_bert_mrc_model_desmodified_lowercase',
                                              'checkpoint'),
    "merge_role_bert_model_pb": os.path.join('output', 'model',
                                             'merge_usingtype_roberta_traindev_event_role_bert_mrc_model_desmodified_lowercase',
                                             'saved_model'),
    "merge_continue_role_bert_model_dir": os.path.join('output', 'model',
                                                       'merge_continue_usingtype_roberta_large_traindev_event_role_bert_mrc_model_desmodified_lowercase',
                                                       'checkpoint'),
    "merge_continue_role_bert_model_pb": os.path.join('output', 'model',
                                                      'merge_continue_usingtype_roberta_large_traindev_event_role_bert_mrc_model_desmodified_lowercase',
                                                      'saved_model'),
    "role_verify_cls_bert_model_dir": os.path.join('output', 'model',
                                                   'final_verify_cls_fold_{}_usingtype_roberta_large_traindev_event_role_bert_mrc_model_desmodified_lowercase',
                                                   'checkpoint'),
    "role_verify_cls_bert_model_pb": os.path.join('output', 'model',
                                                  'final_verify_cls_fold_{}_usingtype_roberta_large_traindev_event_role_bert_mrc_model_desmodified_lowercase',
                                                  'saved_model'),
    "role_verify_avmrc_bert_model_dir": os.path.join('output', 'model',
                                                     'final_verify_avmrc_fold_{}_usingtype_roberta_large_traindev_event_role_bert_mrc_model_desmodified_lowercase',
                                                     'checkpoint'),
    "role_verify_avmrc_bert_model_pb": os.path.join('output', 'model',
                                                    'final_verify_avmrc_fold_{}_usingtype_roberta_large_traindev_event_role_bert_mrc_model_desmodified_lowercase',
                                                    'saved_model'),

    "datamodified_role_bert_model_dir": os.path.join('output', 'model',
                                                     'datamodified_usingtype_roberta_large_traindev_event_role_bert_mrc_model_desmodified_lowercase',
                                                     'checkpoint'),
    "datamodified_role_bert_model_pb": os.path.join('output', 'model',
                                                    'datamodified_usingtype_roberta_large_traindev_event_role_bert_mrc_model_desmodified_lowercase',
                                                    'saved_model'),
    "datamodified_small_role_bert_model_dir": os.path.join('output', 'model',
                                                           'datamodified_usingtype_roberta_traindev_event_role_bert_mrc_model_desmodified_lowercase',
                                                           'checkpoint'),
    "datamodified_small_role_bert_model_pb": os.path.join('output', 'model',
                                                          'datamodified_usingtype_roberta_traindev_event_role_bert_mrc_model_desmodified_lowercase',
                                                          'saved_model'),

    "event_schema": "event_schema.json",
    "multi_task_bert_model_dir": os.path.join('output', 'model', 'event_multask_bert_model', 'checkpoint'),
    "multi_task_bert_model_pb": os.path.join('output', 'model', 'event_multask_bert_model', 'saved_model'),
    "type_class_bert_model_dir": os.path.join('output', 'model',
                                              'index_fold_{}_roberta_large_traindev_desmodified_lowercase_event_type_class_bert_model',
                                              'checkpoint'),
    "type_class_bert_model_pb": os.path.join('output', 'model',
                                             'index_fold_{}_roberta_large_traindev_desmodified_lowercase_event_type_class_bert_model',
                                             'saved_model'),
    "type_role_class_bert_model_dir": os.path.join('output', 'model', 'event_type_role_class_bert_model', 'checkpoint'),
    "type_role_class_bert_model_pb": os.path.join('output', 'model', 'event_type_role_class_bert_model', 'saved_model'),

}
# print(os.path.join(config.get("train_valid_data_dir"),config.get("train_data_text_name")))
