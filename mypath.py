class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = '/Users/daiwei/Documents/AI/data/UCF-101'

            # Save preprocess data into output_dir
            output_dir = '/Users/daiwei/Documents/AI/data/ucf101'

            return root_dir, output_dir
        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = '/Users/daiwei/Documents/AI/data/hmdb-51'

            output_dir = '/Users/daiwei/Documents/AI/data/VAR/hmdb51'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return '/Users/daiwei/Documents/AI/data/Models/ucf101-caffe.pth'