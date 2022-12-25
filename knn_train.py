#using KNN to classify the MNIST dataset and save the model
import pickle
from sklearn.neighbors import KNeighborsClassifier
from keras.datasets import mnist
from itertools import chain

(train_x,train_y),(test_x,test_y) = mnist.load_data()

# print(train_x)

train_images = []
for i in range(len(train_x)):
    train_images.append(list(chain.from_iterable(train_x[i])))

test_images = []
for i in range(len(test_x)):
    test_images.append(list(chain.from_iterable(test_x[i])))

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(train_images,train_y)

with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

# with open('knn_model.pkl', 'rb') as f:
#     knn_from_pickle = pickle.load(f)

# predicted = knn_from_pickle.predict(test_images)

# cnt = 0;
# for i in range(len(predicted)):
#     if predicted[i] == test_y[i]:
#         cnt += 1
# print('accuracy: ',cnt/len(predicted))
