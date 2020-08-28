import os
# import random
# import shutil
# import time
from datetime import datetime
from math import ceil

import numpy as np
import tensorflow as tf
import sklearn
from sklearn.preprocessing import MinMaxScaler

# from distance.tfversion import distance
from evaluation import MAPs


class ResNetKNN(object):
    def __init__(self, config):
        # Initialize setting
        np.set_printoptions(precision=4)
        
        self.save_dir = config.save_dir
        self.batch_size = config.batch_size
        self.img_model = config.img_model
        self.output_dim = config.output_dim
        
        self.file_name = '{}_distance_{}_output_{}'.format(
                config.dataset,
                config.dist_type,
                config.output_dim)
        self.save_dir = os.path.join(config.save_dir, self.file_name + '.npy')
        self.log_dir = config.log_dir

        # Setup session
        # config_proto = tf.ConfigProto()
        # config_proto.gpu_options.allow_growth = True
        # config_proto.allow_soft_placement = True
        # self.sess = tf.Session(config=config_proto)

        # Create variables and placeholders
        self.model_weights = config.model_weights
        self.model = self.load_model()
        self.scaler = MinMaxScaler()
        

    def load_model(self):
        if self.img_model == "resnet50":
            base_model = tf.keras.applications.ResNet50(include_top=False,
                                                          input_tensor=tf.keras.layers.Input(
                                                              shape=(256, 256, 3)),
                                                          weights="imagenet")
        elif self.img_model == "resnet50v2":
            base_model = tf.keras.applications.ResNet50V2(include_top=False,
                                                          input_tensor=tf.keras.layers.Input(
                                                              shape=(256, 256, 3)),
                                                          weights="imagenet")
        else:
            raise Exception('cannot use such CNN model as ' + self.img_model)
            
        img_output = tf.keras.Sequential()
        img_output.add(tf.keras.models.Model(inputs=base_model.get_input_at(0), 
                                             outputs=base_model.layers[-2].get_output_at(0)))
        img_output.add(tf.keras.layers.Flatten())
        img_output.add(tf.keras.layers.Dense(self.output_dim))
        
        return img_output

    def save_codes(self, database, query, model_file=None):
        if model_file is None:
            model_file = self.save_dir + "_codes.npy"

        model = {
            'db_features': database.output,
            # 'db_reconstr ': np.dot(database.codes, C),
            'db_label': database.label,
            'val_features': query.output,
            # 'val_reconstr': np.dot(query.codes, C),
            'val_label': query.label,
        }
        print("saving codes to %s" % model_file)
        folder = os.path.dirname(model_file)
        if os.path.exists(folder) is False:
            os.makedirs(folder)
        np.save(model_file, np.array(model))
        return

    def save_model(self, model_file=None):
        if model_file is None:
            model_file = self.save_dir

        model = {}
        for layer in self.model.layers:
            model[layer] = layer.get_weights()

        print("saving model to %s" % model_file)
        folder = os.path.dirname(model_file)
        if os.path.exists(folder) is False:
            os.makedirs(folder)

        np.save(model_file, np.array(model))
        return

    def update_codes(self, dataset, batch_size, val_print_freq=100):
        '''
        update codes in batch size
        '''
        total_batch = int(ceil(dataset.n_samples / float(batch_size)))
        dataset.finish_epoch()

        for i in range(total_batch):
            print(i)
            images = dataset.next_batch(batch_size)
            codes, output = self.predict(images)
            
            dataset.feed_batch_codes(batch_size, codes)
            dataset.feed_batch_output(batch_size, output)

            dataset.finish_epoch()

            if i % val_print_freq == 0:
                    print("%s #validation# batch %d/%d" % (datetime.now(), i, total_batch))

    def predict(self, batch):
        output = self.model.predict(batch)
        
        try:
            normalized_output = self.scaler.transform(output)
        except sklearn.exceptions.NotFittedError:
            normalized_output = self.scaler.fit_transform(output)
            
        codes = (normalized_output >=0.5 ).astype(int)
        return codes, normalized_output
        
    def validation(self, img_query, img_database, R=100):
        print("%s #validation# start validation" % (datetime.now()))
        
        # Get codes of database && Update centers
        self.update_codes(img_database, self.batch_size)

        # Get codes of query
        self.update_codes(img_query, self.batch_size)
        
        # Fit scaler
        
        # Evaluation
        print("%s #validation# calculating MAP@%d" % (datetime.now(), R))
        mAPs = MAPs(R)
        self.save_codes(img_database, img_query)
        self.save_model()
        return {
            'map_feature_ip': mAPs.get_mAPs_by_feature(img_database, img_query, dist_type="euclidean"),
            'map_after_sign':  mAPs.get_mAPs_after_sign(img_database, img_query, dist_type="euclidean"),
            'map_after_sign_with_feature_rerank': mAPs.get_mAPs_after_sign_with_feature_rerank(img_database, img_query, dist_type="euclidean")
        }

