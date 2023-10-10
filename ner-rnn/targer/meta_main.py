import json

class ExpSettings:
    
    def __init__(self):
        self.train = None
        self.train_noisy = None
        self.dev = None
        self.test = None
        self.data_io = "connl-ner-2003"
        self.gpu = 0
        self.model = "BiRNNCNNCRF" # choices=['BiRNN', 'BiRNNCNN', 'BiRNNCRF', 'BiRNNCNNCRF']
        self.load = None
        self.save = "%s_tagger.hdf5"
        self.word_seq_indexer = None
        self.epoch_num = 50
        self.patience = 100
        self.evaluator = "f1-connl"
        self.save_best = True
        self.dropout_ratio=0.5
        self.batch_size = 10
        self.opt = "sgd"
        self.lr = 0.01
        self.lr_decay = 0.05
        self.momentum = 0.9
        self.clip_grad = 5
        self.rnn_type = "LSTM" # choices=['Vanilla', 'LSTM', 'GRU']
        self.rnn_hidden_dim = 300
        self.emb_dim = 300
        self.emb_fn = None
        self.emb_delimiter = " "
        self.emb_load_all = False
        self.freeze_word_embeddings = False
        self.check_for_lowercase = True
        self.char_embeddings_dim = 25
        self.char_cnn_filter_num = 30
        self.char_window_size = 3
        self.freeze_char_embeddings = False
        self.word_len = 20
        self.dataset_sort = False
        self.seed_num = None
        self.report_fn = None
        self.cross_folds_num = -1
        self.cross_fold_id = -1
        self.verbose = True
        self.noisy_subsample_factor = None
        
    def __repr__(self):
        return json.dumps(self.__dict__, indent=4, sort_keys=True)
    
    def __str__(self):
        return self.__repr__()
        
def run_exp(train_size, data, subset_size, dev_subset_size, seed):
    exp_settings = ExpSettings()
    if subset_size != -1:
        subset_size_string = f"_subset{subset_size}"
    else:
        subset_size_string = ""
        
    if dev_subset_size != -1:
        dev_subset_size_string = f"_subset{dev_subset_size}"
    else:
        dev_subset_size_string = ""
        
    exp_settings.train = f"/data/users/didelani/ner-rnn/targer/data{train_size}/{data}/train.txt"
    exp_settings.dev = f"/data/users/didelani/ner-rnn/targer/data{train_size}/{data}/dev.txt"
    exp_settings.test = f"/data/users/didelani/ner-rnn/targer/data{train_size}/{data}/test.txt"
    
    if data == "hau":
        exp_settings.emb_fn = "/data/users/didelani/word_embeddings/hau/model"
    elif data == "amh":
        exp_settings.emb_fn = "/data/users/didelani/word_embeddings/cc.am.300.bin"
    elif data == "twi":
        exp_settings.emb_fn = "/data/users/didelani/word_embeddings/twi/fastText_tw_C3_large.bin"
    elif data == "yor":
        exp_settings.emb_fn = "/data/users/didelani/word_embeddings/yor/fastText_yo_C3_large.bin" #cc.yo.300.bin" #yor/fastText_yo_C3_large.bin"
    elif data == "xho":
        exp_settings.emb_fn = "/data/users/didelani/word_embeddings/xho/xhosa.bin"
    elif data == "wol":
        exp_settings.emb_fn = "/data/users/didelani/word_embeddings/wol/wolof.bin"
    else:
        exp_settings.emb_fn = "/data/users/didelani/word_embeddings/"+data+"/"+lang+".bin"

    exp_settings.save = f"/data/users/didelani/ner-rnn//targer/outputs/{data}_subset{subset_size}_seed{seed}.hdf5"
    exp_settings.report_fn = f"/data/users/didelani/ner-rnn/targer/outputs/{data}_subset{subset_size}_seed{seed}.log"
    exp_settings.seed_num = seed

    from main import main_funct
    main_funct(exp_settings)
    
if __name__ == "__main__":

    languages = ['yor']#, 'bam', 'bbj','tsn']
    #languages = ['ibo', 'kin', 'lug', 'swa']
    #languages = ['mos', 'pcm', 'sna', 'zul']
    #languages = ['ewe', 'fon', 'luo', 'nya']
    for lang in languages:
        for seed in range(12345, 12350):
            #for clean_subset_size, limit_devset_size in zip([500, 1000, 2000, 4000],[-1]):  # -1 is 1014
            for clean_subset_size in [-1]:  # -1 is 1014
                limit_devset_size = -1
                run_exp(clean_subset_size, lang, clean_subset_size, limit_devset_size, seed)
