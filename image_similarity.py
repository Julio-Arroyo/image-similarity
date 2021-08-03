#  July 29

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from IPython.core.display import Image, display
import matplotlib.pyplot as plt
import numpy as np
import os, random
import scipy.io
from scipy import spatial


prefix = '/Users/jarroyo/OneDrive - California Institute of Technology/SURF 2021 - Kreiman Lab/datasets/array/'

neural_net = VGG16(weights='imagenet')
model = Model(inputs=neural_net.input, outputs=neural_net.get_layer('fc2').output)


def get_1_idx(arr):
    'Given "arr", a numpy array of 0\'s and a single 1, return the index where the 1 is.'
    for i in range(len(arr)):
        if arr[i] == 1:
            return i
    return None


def get_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return model.predict(img)


def infer_target(norm='L2'):
    categories = {16:'cat', 18:'horse', 19:'sheep', 20:'cow', 34:'kite', 78:'teddy bear'}
    subjects = ['subj02-el','subj03-yu','subj05-je','subj07-pr','subj08-bo']
    #  subjects = ['subj03-yu','subj05-je','subj07-pr','subj08-bo']

    #  results: No. of fixations --> (No. of guesses -> No. of trials that took these many guesses)
    results = {1:{1:0, 2:0, 3:0, 4:0, 5:0},
               2:{1:0, 2:0, 3:0, 4:0},
               3:{1:0, 2:0, 3:0},
               4:{1:0, 2:0}}
    num_trials = [0] * 6  # the i-th index will contain the number of trials with i fixations
    
    is_cosine = False
    if norm == 'cosine':
        is_cosine = True
    
    for subject in subjects:
        #  Keys subjperform: 'scoremat' (has the fixations at which targets were found); 
        #  'fixmat' (has the order in which images were looked at), 'timemat' (time in which fixations occurred).
        subjperform = scipy.io.loadmat(prefix + 'psy/subjects_array/' + subject + '/subjperform.mat')
        
        scoremat = subjperform['scoremat']  # has the fixations at which targets were found
        fixmat = subjperform['fixmat']  # has the order in which the images were looked at
        array_info = scipy.io.loadmat(prefix + 'psy/array.mat')
        array_info = array_info['MyData']
        print('SUBJECT: {}'.format(subject))
        
        for i in range(scoremat.shape[0]):
            target_idx = get_1_idx(scoremat[i])  # get idx where target is located
            if target_idx is None or target_idx == 0 or target_idx == 5:
                continue
            else:
                num_trials[target_idx] += 1
            
#             repeated_fixation = False
#             num_repeated_fixations = 0
            
#             sum_features = [0] * 4096
#             potential_targets = [1, 2, 3, 4, 5, 6]
#             #  ef_paths = []  # error fixation paths
#             error_fixations = []
            
#             for j in range(target_idx):
#                 img_num = int(fixmat[i][j])  # image number from 1-6.
#                 img_cate = array_info[i][0][3][img_num - 1][0]  # in number format
#                 img_id = array_info[i][0][4][img_num - 1][0]
#                 img_path = '{}AllObjCollection/cate{}/img{}.jpg'.format(prefix, img_cate, img_id)
#                 error_fixations.append(categories[img_cate])
                
#                 sum_features += get_features(img_path)
#                 #  print('NUM TO BE REMOVED: {}'.format(int(img_num)))
#                 try:
#                     potential_targets.remove(int(img_num))
#                 except ValueError:
#                     #  print('VALUE ERROR ITERATION (i={}, j={})'.format(i, j))
#                     repeated_fixation = True
#                     num_repeated_fixations += 1
            
#             #  print('SUM_FEATURES: {}'.format(sum_features))
#             avg_features = sum_features * (1/ (target_idx))
#             distances = {}  # map: category -> distance
#             #  print('AVG_FEATURES: {}'.format(avg_features))
#             pt_paths = {}  # map: category -> paths
            
#             for potential_target in potential_targets:
#                 img_cate = array_info[i][0][3][potential_target - 1][0]
#                 img_id = array_info[i][0][4][potential_target - 1][0]
#                 img_path = '{}AllObjCollection/cate{}/img{}.jpg'.format(prefix, img_cate, img_id)
#                 pt_paths[img_cate] = img_path
#                 if norm == 'L1':
#                     distances[img_cate] = np.sum(np.absolute(avg_features - get_features(img_path)))
#                 elif norm == 'cosine':
#                     distances[img_cate] = 1 - spatial.distance.cosine(avg_features, get_features(img_path))
#                 elif norm == 'L2':
#                     distances[img_cate] = np.linalg.norm(avg_features - get_features(img_path))
#                 else:
#                     raise ValueError('Norm must be L1, L2, or cosine.')

#             sorted_distances = sorted(distances.items(), key=lambda x: x[1], reverse=is_cosine)
            
#             #  Make inference
#             for k in range(len(sorted_distances)):
#                 if array_info[i][0][1][0] == sorted_distances[k][0]:
#                     num_fixations = target_idx
#                     if repeated_fixation:
#                         num_fixations -= num_repeated_fixations
#                         repeated_fixation = False
#                     #  print('num_fixations: {}; k + 1: {}'.format(num_fixations, k + 1))
#                     results[num_fixations][k + 1] += 1
#                     break
    return num_trials


def plot_results(results1, results2):
    for num_fixations in results1.keys():
#         plt.bar(results[num_fixations].keys(), results[num_fixations].values(), color='blue')
#         x_axis = list(range(7 - num_fixations))
#         plt.xlabel('Number of inferences to find target')
#         plt.ylabel('Frequency')
#         plt.xticks(x_axis)
#         plural = 's'
#         if num_fixations == 1:
#             plural = ''
#         plt.title('{} - Trials with {} error fixation{}'.format(norm, num_fixations, plural))
#         plt.show()
        
        
        l1 = list(results1[num_fixations].values())
        l2 = list(results2[num_fixations].values())
        
#         print('L1 size {}'.format(len(l1)))
#         print('L2 size {}'.format(len(l2)))

        x = np.arange(1,7 - num_fixations)  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, l1, width, label='L1')
        rects2 = ax.bar(x + width/2, l2, width, label='L2')
        
        plural = 's'
        if num_fixations == 1:
            plural = ''

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Frequency')
        ax.set_title('Trials with {} error fixation{}'.format(num_fixations, plural))
        ax.set_xlabel('Number of inferences to find target')
        ax.set_xticks(x)
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        fig.tight_layout()

        plt.savefig('/Users/jarroyo/OneDrive - California Institute of Technology/SURF 2021 - Kreiman Lab/Results Julio Arroyo/array_{}_fixations'.format(num_fixations))
        plt.show()

        
def plot_results(results1, results2, results3, num_trials):
    for num_fixations in results1.keys():
        l1 = list(results1[num_fixations].values())
        l2 = list(results2[num_fixations].values())
        cosine = list(results3[num_fixations].values())
        
        x = np.arange(1,7 - num_fixations)  # the label locations
        width = 0.28  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width, l1, width, label='L1')
        rects2 = ax.bar(x, l2, width, label='L2')
        rects3 = ax.bar(x + width, cosine, width, label='cosine')
        
        plural = 's'
        if num_fixations == 1:
            plural = ''

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Frequency')
        ax.set_title('Trials with {} error fixation{}'.format(num_fixations, plural))
        ax.set_xlabel('Number of inferences to find target')
        ax.set_xticks(x)
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        ax.bar_label(rects3, padding=3)

        fig.tight_layout()

        plt.savefig('/Users/jarroyo/OneDrive - California Institute of Technology/SURF 2021 - Kreiman Lab/Results Julio Arroyo/cosinel1l2_array_{}_fixations'.format(num_fixations))
        plt.show()
    
    # Plotting relative performance
    avg_num_inferences_model = [None] * 6
    results = [results1, results2, results3]  # TODO: Make single results parameter a matrix instead of taking in a number of parameters for each result
    avg_num_inferences_chance = [None, 3, 2.5, 2, 1.5, None]  # i-th element means the average number of inferences needed by chance to correctly guess target given i fixations.
    markers = ['-or', '-sb', '-^g']
    
    for l in range(len(results)):
        for i in range(1, 5):
            total = 0
            for j in range(1, len(results[l][i]) + 1):
                total += results[l][i][j] * j
            avg_num_inferences_model[i] = total / num_trials[i]
        
        relative_performance = [None] * 4
        for i in range(4):
            relative_performance[i] = 100 * ((avg_num_inferences_chance[i + 1] - avg_num_inferences_model[i + 1]) / avg_num_inferences_chance[i + 1])
        x = np.arange(1,5)
        plt.plot(x, relative_performance, markers[l])
        print('Model {}: average number of inferences {}'.format(l, avg_num_inferences_model))
        print('Result {}: relative performance {}'.format(l, relative_performance))
    plt.show()

# TODO: Measure skewness of data? Relative performance, measure over/under chance

# inferences_l1 = infer_target('L1')
# inferences_l2 = infer_target('L2')
# inferences_cosine = infer_target('cosine')
inferences_l1 = {1: {1: 149, 2: 151, 3: 144, 4: 161, 5: 168}, 2: {1: 151, 2: 153, 3: 133, 4: 107}, 3: {1: 133, 2: 103, 3: 106}, 4: {1: 116, 2: 85}}
inferences_l2 = {1: {1: 151, 2: 149, 3: 136, 4: 158, 5: 179}, 2: {1: 149, 2: 159, 3: 122, 4: 114}, 3: {1: 130, 2: 100, 3: 112}, 4: {1: 118, 2: 83}}
inferences_cosine = {1: {1: 135, 2: 147, 3: 167, 4: 148, 5: 176}, 2: {1: 142, 2: 168, 3: 126, 4: 111}, 3: {1: 130, 2: 126, 3: 83}, 4: {1: 123, 2: 78}}
num_trials = infer_target()
plot_results(inferences_l1, inferences_l2, inferences_cosine, num_trials)

print('Inferences L1: {}'.format(inferences_l1))
print('Inferences L2: {}'.format(inferences_l2))
print('Inferences cosine similarity: {}'.format(inferences_cosine))
print('Number of trials by number of error fixations: {}'.format(num_trials))

