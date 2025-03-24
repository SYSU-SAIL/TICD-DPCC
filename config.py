class NetworkConfig:
    def __init__(self):
        self.token_size = 256
        self.output_size = 255
        self.leaf_embedding_dim = 128
        self.octant_embedding_dim = 6
        self.dim_feedforward = 300
        self.level_embedding_dim = 6
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 稀疏点云 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.max_octree_level = 14
        self.octree_level = 12
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 密集点云 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # self.max_octree_level = 10
        # self.octree_level = 10
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.trace_parent_num = 3
        self.context_len = 1024
        self.refer_len = 2048
        self.num_cross_level = 2
        self.attention_heads = 4
        self.num_hidden_layers = 4
        self.anchor_num = 512


class TrainingConfig:
    def __init__(self):
        self.num_epochs = 20
        self.dropout_rate = 0
        self.log_interval = 20  # 记录点间隔的迭代次数
        self.lr_scheduler_factor = 0.1
        self.test_sample_rate = 0.2
        self.lr_scheduler_patience = 1
        self.learning_rate = 1e-4
        self.batch_size = 20  # 双卡
        # self.batch_size = 64   # 三卡
        self.shuffle = False
        self.num_workers = 6
        self.clip_grad_norm = 1.0
        self.buffer_size = 10000
        self.depth_decap = {14: 16}


class KittiConfig:
    def __init__(self):
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 工作站路径 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.data_path = "~/Public/Dataset/data_odometry_velodyne/dataset/sequences"
        self.processed_path = "~/Public/Dataset/OCTANT/"
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ AMAX路径 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # self.data_path = "~/Public/Dataset/data_odometry_velodyne/dataset/sequences"
        # self.processed_path = "~/Public/OCTANT/"
        self.train_seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.test_seq = [11, 12, 13]
        self.horizontal_range = 80
        self.vertical_range = 4
        self.min_bound = [-80, -80, -80]
        self.max_bound = [80.016, 80.016, 80.016]


class DenseConfig:
    def __init__(self):
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ AMAX路径 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.mvub_path = "~/Public/Dataset/MVUB"
        self.EiVFB_path = "~/Public/Dataset/8iVFB"
        self.mvub_processed_path = "~/Public/Dataset/OCTANT_DENSE/MVUB"
        self.EiVFB_processed_path = "~/Public/Dataset/OCTANT_DENSE/EiVFB"
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 工作站路径 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # self.mvub_path = "~/Dataset/MVUB"
        # self.EiVFB_path = "~/Dataset/8iVFB"
        # self.mvub_processed_path = "~/Dataset/OCTANT_DENSE/MVUB"
        # self.EiVFB_processed_path = "~/Dataset/OCTANT_DENSE/EiVFB"
        self.train_seq = ['andrew10', 'david10', 'sarah10', 'soldier', 'longdress']
        self.test_seq = ['phil10', 'ricardo10','redandblack','loot']
        self.resolution_range = 1023
        self.offset = -512
        self.min_bound = [-512.0, -512.0, -512.0]
        self.max_bound = [512.01024, 512.01024, 512.01024]


class Config:
    def __init__(self):
        self.network = NetworkConfig()
        self.training = TrainingConfig()
        self.kitti = KittiConfig()
        self.dense = DenseConfig()


config = Config()
