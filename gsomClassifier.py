import time
import sys
import os
from os.path import join
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
sys.path.append('../../')
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from util import utilities as Utils
from util import display as Display_Utils
from core4 import gsom as GSOM_Core

from params import params as Params
# from core4 import core_controller as Core

import numpy as np
class GSOMClassifier():
    def __init__(
        self,
        SF=0.83,
        forget_threshold=60,
        temporal_contexts=1,
        learning_itr=100,
        smoothing_irt=50,
        plot_for_itr=4,
    ):
        self.SF = SF
        self.forget_threshold = forget_threshold
        self.temporal_contexts = temporal_contexts
        self.learning_itr = learning_itr
        self.smoothing_irt = smoothing_irt
        self.plot_for_itr = plot_for_itr
        self.output_loc = None
        self.params = None
        self.gsom = None

    def generate_output_config(self, SF, forget_threshold):
        # File Config
        dataset = 'Classifier'
        experiment_id = 'Exp-' + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
        output_save_location = join('output/', experiment_id)

        # Output data config
        output_save_filename = '{}_data_'.format(dataset)
        filename = output_save_filename + str(SF) + '_T_' + str(self.temporal_contexts) + '_mage_' + str(
            forget_threshold) + 'itr'
        plot_output_name = join(output_save_location, filename)

        # Generate output plot location
        output_loc = plot_output_name
        output_loc_images = join(output_loc, 'images/')
        if not os.path.exists(output_loc):
            os.makedirs(output_loc)
        if not os.path.exists(output_loc_images):
            os.makedirs(output_loc_images)

        return output_loc, output_loc_images

    def _grow_gsom(self, inputs, dimensions, plot_for_itr=0, classes=None, output_loc=None):
        self.gsom = GSOM_Core.GSOM(self.params.get_gsom_parameters(), inputs, dimensions, plot_for_itr=plot_for_itr,
                              activity_classes=classes, output_loc=output_loc)
        self.gsom.grow()
        self.gsom.smooth()
        self.gsom_nodemap = self.gsom.assign_hits()

    def run(self, X_train, plot_for_itr=0, classes=None, output_loc=None):
        results = []
        start_time = time.time()
        self._grow_gsom(X_train, X_train.shape[1], plot_for_itr=plot_for_itr, classes=classes, output_loc=output_loc)
        print('Batch', 0)
        print('Neurons:', len(self.gsom_nodemap))
        print('Duration:', round(time.time() - start_time, 2), '(s)\n')

        results.append({
            'gsom': self.gsom_nodemap,
            'aggregated': None
        })

        return results

    def fit(self, input_vector_database, classes):
        # Init GSOM Parameters
        gsom_params = Params.GSOMParameters(self.SF, self.learning_itr, self.smoothing_irt, distance=Params.DistanceFunction.EUCLIDEAN,
                                            temporal_context_count=self.temporal_contexts, forget_itr_count=self.forget_threshold)
        generalise_params = Params.GeneraliseParameters(gsom_params)

        # Process the input files

        self.output_loc, output_loc_images = self.generate_output_config(self.SF, self.forget_threshold)

        # Setup the age threshold based on the input vector length
        generalise_params.setup_age_threshold(input_vector_database.shape[0])
        self.params = generalise_params

        # Process the clustering algorithm
        result_dict = self.run(input_vector_database, self.plot_for_itr, classes, output_loc_images)

        return result_dict, classes

    def predict(self, x_test):
        self.gsom.finalize_gsom_label()
        y_pred = self.gsom.predict(x_test)
        return y_pred

    def dispaly(self, result_dict, classes):
        saved_name = Utils.Utilities.save_object(result_dict, join(self.output_loc, 'gsom_nodemap_SF-{}'.format(self.SF)))
        gsom_nodemap = result_dict[0]['gsom']
        # Display
        display = Display_Utils.Display(result_dict[0]['gsom'], None)
        display.setup_labels_for_gsom_nodemap(classes, 2, 'Latent Space of {} : SF={}'.format("Data", self.SF),
                                              join(self.output_loc, 'latent_space_' + str(self.SF) + '_hitvalues'))
        display.setup_labels_for_gsom_nodemap(classes, 2, 'Latent Space of {} : SF={}'.format("Data", self.SF),
                                              join(self.output_loc, 'latent_space_' + str(self.SF) + '_labels'))

        print('Completed.')


df = pd.read_csv("data/train.csv")
df = df.replace("nan", np.NaN)
# print(df.isnull().sum())
df.fillna(df.mean(), inplace=True)

X = df.iloc[:, [1, 2, 3, 4, 5, 8, 9, 10, 11, 12]].values
y = df.iloc[:, -1].values

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
gsom = GSOMClassifier()
result_dict, classes = gsom.fit(X, y)
gsom.dispaly(result_dict, classes)
y_pred = gsom.predict(X_test)
f1score = f1_score(y_test, y_pred, average='macro')

print(f1_score())

# X = np.load('data/casme_train2.npy')
# y = np.load('data/casme_train_label2.npy')
# X_test = np.load('data/casme_test2.npy')
# y_test = np.load('data/casme_test_label2.npy')
# # print(y)
# # print(y_test)
# # print(X.shape)
# # print(y.shape)
# # print(X_test.shape)
# # print(y_test.shape)
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)
# X_test = scaler.fit_transform(X_test)
# # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# gsom = GSOMClassifier()
# result_dict, classes = gsom.fit(X, y)
# gsom.dispaly(result_dict, classes)
#
# y_pred = gsom.predict(X_test)
#
# print(y_pred)
# print(y_test)
#
# mat = confusion_matrix(y_test, y_pred)
# print (mat)
#
#
# k = 0
# for i in range (len(mat)):
#     k = k + mat[i][i]
# acc = k/len(y_test)*100
#
# print("Accuracy  :   ", acc)



# XGBOOST
gbc = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
gbc.fit(X, y)
y_predict = gbc.predict(X_test)
y_pred = np.where(y_predict.astype(int) > 0.5, 1, 0)
#
#
mat = confusion_matrix(y_test, y_pred)
print (mat)
k = 0
for i in range (len(mat)):
    k = k + mat[i][i]
acc = k/len(y_test)*100
print("Accuracy  :   ", acc)
