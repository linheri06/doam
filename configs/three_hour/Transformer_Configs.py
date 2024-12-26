class Configs:
    def __init__(self):
        # Basic settings
        self.target = 'OT'
        self.des = 'test'
        self.dropout = 0.05
        self.num_workers = 10
        self.gpu = 0
        self.lradj = 'type1'
        self.devices = '0'
        self.use_gpu = False
        self.use_multi_gpu = False
        self.freq = 'h'
        self.checkpoints = './checkpoints/'
        self.bucket_size = 4
        self.n_hashes = 4
        self.seq_len = 36
        self.label_len = 1
        self.pred_len = 36
        self.e_layers = 2
        self.d_layers = 1
        self.n_heads = 8
        self.factor = 1
        self.d_model = 512
        self.des = 'Exp'
        self.itr = 1
        self.d_ff = 2048
        self.moving_avg = 25
        self.distil = True
        self.output_attention = False
        self.patience = 3
        self.learning_rate = 0.0001
        self.batch_size = 32
        self.embed = 'timeF'
        self.activation = 'gelu'
        self.use_amp = False
        self.loss = 'mse'

        # Training settings
        self.train_epochs = 4
        self.is_training = True

        # Data settings
        self.enc_in = 2
        self.dec_in = 2
        self.c_out = 2
        self.target = "soilMoisture"
        self.root_path = './data_set/'
        self.data_path = 'temperature_humidity_tran1.csv'
        self.model_id = 'temperature_humidity'
        self.model = 'Transformer'
        self.data = 'custom'
        self.features = 'MS'

        