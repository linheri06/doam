class Configs:
    def __init__(self):
        # Basic settings
        self.target = 'OT'
        self.des = 'Exp'
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
        self.seq_len = 24
        self.label_len = 4
        self.pred_len = 1
        self.e_layers = 2
        self.d_layers = 1
        self.n_heads = 8
        self.factor = 1
        self.d_model = 512
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
        self.train_epochs = 8
        self.is_training = True

        # Data settings
        self.enc_in = 5
        self.dec_in = 5
        self.c_out = 5
        self.target = "luuluongden"
        self.root_path = './data_set/'
        self.data_path = 'tatrach_data_handle.csv'
        self.model_id = 'TaTrach_96_24'
        self.model = 'PatchTST'
        self.data = 'custom'
        self.features = 'MS'

        # PatchTST specific settings
        self.fc_dropout = 0.05
        self.head_dropout = 0.0
        self.patch_len = 16
        self.stride = 8
        self.padding_patch = 'end'
        self.revin = 1
        self.affine = 0
        self.subtract_last = 0
        self.decomposition = 0
        self.kernel_size = 25
        self.individual = 0

        # Optimizer settings
        self.pct_start = 0.3
