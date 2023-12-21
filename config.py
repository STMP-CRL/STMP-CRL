class Config(object):
    def __init__(self):
        self.path = './data/meetupCA/'
        self.user_dataset = self.path + 'userRating'
        self.group_dataset = self.path + 'groupRating'
        self.user_in_group_path = "./data/meetupCA/groupMember.txt"
        self.embedding_size = 32
        self.epoch = 30
        self.num_negatives = 6
        self.batch_size = 512
        self.lr = [0.001, 0.0005, 0.0001]
        self.drop_ratio = 0.2
        self.save_path = "./res/saveGCN4Neg6.txt"

        self.input_dim = 32
        self.hidden_size = 32
        self.num_layers = 1

        self.gcn_layers = 4
        self.is_split = False

        self.hidden = [64, 32, 16, 8]

        self.topK = [5, 10]

        self.temperature = 0.25
        self.lambda1 = 1
