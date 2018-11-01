from sklearn.model_selection import train_test_split
from utils import load_spam_data

from keras import optimizers,backend
import json
from keras.layers import Dense,Dropout,Embedding,Input,BatchNormalization,Conv1D,GlobalMaxPooling1D,Concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time


print('Loading data')
x, y, vocabulary, vocabulary_inv,labels = load_spam_data()
print('shape of the X is : ',x.shape[1])
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

sequence_length = x.shape[1]
vocabulary_size = len(vocabulary_inv)
embedding_dim = 256
filter_sizes = [3,4,5]
num_filters = 512
drop = 0.5
epochs = 3
batch_size = 64
weights_path = 'spam_cnn.h5'

index = {
    'word_to_id': vocabulary,
    'labels': labels,
    'shape':sequence_length
}
# may have to store the shape : x.shape[1]

with open('index.json', 'w') as f:
    f.write(json.dumps(index))


# this returns a tensor
print("Creating Model...")
# TextCNN Model


def create_model():
    inputs = Input(shape=(sequence_length,), dtype='int32')
    embed = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
    conv_3 = Conv1D(filters=256, kernel_size=filter_sizes[0], padding="valid", activation="relu", strides=1)(embed)
    conv_4 = Conv1D(filters=256, kernel_size=filter_sizes[1], padding="valid", activation="relu", strides=1)(embed)
    conv_5 = Conv1D(filters=256, kernel_size=filter_sizes[2], padding="valid", activation="relu", strides=1)(embed)
    pool_3 = GlobalMaxPooling1D()(conv_3)
    pool_4 = GlobalMaxPooling1D()(conv_4)
    pool_5 = GlobalMaxPooling1D()(conv_5)
    cat = Concatenate()([pool_3, pool_4, pool_5])
    output = Dropout(0.25)(cat)
    dense1 = Dense(256, activation='relu')(output)
    bn = BatchNormalization()(dense1)
    output = Dense(units=y_train.shape[1], activation='softmax')(bn)
    print(y_train.shape[1])
    model = Model(inputs=inputs, outputs=output)
    model.save_weights('spam_cnn.h5', overwrite=True)
    return model


def evaluation(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = [int(i[1] + 0.5) for i in y_pred]
    target_names = ['class ' + str(i) for i in range(0, y_train.shape[1])]
    y_test = y_test.tolist()
    y_test = [i.index(1.0) for i in y_test]
    print(classification_report(y_test, y_pred, target_names=target_names))
    print('\n')
    print(confusion_matrix(y_test, y_pred))


def train():
    """ GPU parameter """
    with tf.device('/gpu:0'):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1, allow_growth=True)
        tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                         log_device_placement=True,
                                         gpu_options=gpu_options))
        model = create_model()
        model.compile(optimizer=optimizers.Adam(amsgrad=True),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print('Model Summary : ', model.summary())
        callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0),
                     ModelCheckpoint(weights_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
                     ]
        print("training started...")
        tic = time.process_time()
        model.fit(X_train,
                  y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(X_test, y_test),
                  shuffle=1,
                  callbacks=callbacks)
        toc = time.process_time()
        print("training ended...")
        print(" ----- total Computation time = " + str((toc - tic) / 3600) + " hours ------ ")
        backend.set_learning_phase(0)
        sess = backend.get_session()
        builder = tf.saved_model.builder.SavedModelBuilder("./model")
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
        builder.save(False)
        evaluation(model, X_test, y_test)


def load_model(weights_path):
    print("load model...")
    model = create_model()
    model.load_weights(weights_path)
    return model


if __name__ == "__main__":
    train()
    K.clear_session()
    tf.reset_default_graph()


