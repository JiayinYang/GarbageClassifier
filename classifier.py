import numpy as np
import glob
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix


def classifier(z, precision, recall, falsepos, train_dataset, test_dataset, train_labels, test_labels, trainsize, testsize):
    batch_size = 16
    patch_size = 5
    image_size1 = 384 
    image_size2 = 512
    depth = 16
    num_hidden1 = 256
    num_hidden2 = 64
    num_hidden3 = 16
    num_channels = 3
    num_labels = 4
      
    train_dataset = train_dataset.reshape(
            (trainsize, image_size1, image_size2, num_channels)).astype(np.float32)
    test_dataset = test_dataset.reshape(
            (testsize, image_size1, image_size2, num_channels)).astype(np.float32)
     
    def accuracy(predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

    # For each class, compute the confusion matrix, accuracy, precision, recall and specifity
    def confusionMatrix(cls_true, pred, classNum):
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for i in np.arange(0,16):
            if((cls_true[i] == classNum) and (pred[i] == classNum)):
                tp = tp + 1
            if(cls_true[i] != classNum and pred[i] != classNum):
                tn = tn + 1
            if(cls_true[i] == classNum and pred[i] != classNum):
                fn = fn + 1
            if(cls_true[i] != classNum and pred[i] == classNum):
                fp = fp + 1
     
        tp = float(tp)
        tn = float(tn)
        fn = float(fn)
        fp = float(fp)
      

        if((tp + fp) != 0):
            precision[classNum][z] = float(tp) / (float(tp) + float(fp))
        
        
        if((tp + fn) != 0):
            recall[classNum][z] = tp / (tp + fn)
        
        if((tn + fp) != 0):
            falsepos[classNum][z] = float(1.0-tn / (tn + fp))
      
        accuracy = (tp + tn) / batch_size
        print('accuracy')
        print(accuracy)


    def plot_confusion_matrix(labels, predictions):
        cls_true = np.argmax(labels, 1)
        pred = np.argmax(predictions, 1)
        print('label:')
        print(cls_true)
        print('prediction:')
        print(pred)
        print('class 1:')
        confusionMatrix(cls_true, pred, 0)
        print('class 2:')
        confusionMatrix(cls_true, pred, 1)
        print('class 3:')
        confusionMatrix(cls_true, pred, 2)
        print('class 4:')
        confusionMatrix(cls_true, pred, 3)
   
      
    graph = tf.Graph()

    with graph.as_default():
        # Define the training dataset and lables
        tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size1, image_size2, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

        # Validation/test dataset
        tf_test_dataset = tf.constant(test_dataset)
    
        # CNN layer 1 with filter (num_channels, depth) (3, 16)
        cnn1_W = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, depth], stddev=0.1))
        cnn1_b = tf.Variable(tf.zeros([depth]))

        # CNN layer 2 with filter (depth, depth) (16, 16)
        cnn2_W = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, depth, depth], stddev=0.1))
        cnn2_b = tf.Variable(tf.constant(1.0, shape=[depth]))

        cnn3_W = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, depth, depth], stddev=0.1))
        cnn3_b = tf.Variable(tf.constant(1.0, shape=[depth]))

        cnn4_W = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, depth, depth], stddev=0.1))
        cnn4_b = tf.Variable(tf.constant(1.0, shape=[depth]))

        # Compute the output size of the CNN2 as a 1D array.
        size = image_size1 // 16 * image_size2 // 16 * depth

        # FC1 (size, num_hidden1) (size, 256)
        fc1_W = tf.Variable(tf.truncated_normal(
            [size, num_hidden1], stddev=np.sqrt(2.0 / size)))
        fc1_b = tf.Variable(tf.constant(1.0, shape=[num_hidden1]))

        # FC2 (num_hidden1, num_hidden2) (size, 64)
        fc2_W = tf.Variable(tf.truncated_normal(
            [num_hidden1, num_hidden2], stddev=np.sqrt(2.0 / (num_hidden1))))
        fc2_b = tf.Variable(tf.constant(1.0, shape=[num_hidden2]))

        fc3_W = tf.Variable(tf.truncated_normal(
            [num_hidden2, num_hidden3], stddev=np.sqrt(2.0 / (num_hidden2))))
        fc3_b = tf.Variable(tf.constant(1.0, shape=[num_hidden3]))

        # Classifier (num_hidden2, num_labels) (64, 10)
        classifier_W = tf.Variable(tf.truncated_normal(
            [num_hidden3, num_labels], stddev=np.sqrt(2.0 / (num_hidden3))))
        classifier_b = tf.Variable(tf.constant(1.0, shape=[num_labels]))

        # Model.
        def model(data): 
        # First convolution layer with stride = 1 and pad the edge to make the output size the same.
        # Apply ReLU and a maximum 2x2 pool
            conv1 = tf.nn.conv2d(data, cnn1_W, [1, 1, 1, 1], padding='SAME')
            hidden1 = tf.nn.relu(conv1 + cnn1_b)
            pool1 = tf.nn.max_pool(hidden1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

            # Second convolution layer
            conv2 = tf.nn.conv2d(pool1, cnn2_W, [1, 1, 1, 1], padding='SAME')
            hidden2 = tf.nn.relu(conv2 + cnn2_b)
            pool2 = tf.nn.max_pool(hidden2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

            conv3 = tf.nn.conv2d(pool2, cnn3_W, [1, 1, 1, 1], padding='SAME')
            hidden3 = tf.nn.relu(conv3 + cnn3_b)
            pool3 = tf.nn.max_pool(hidden3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

            conv4 = tf.nn.conv2d(pool3, cnn4_W, [1, 1, 1, 1], padding='SAME')
            hidden4 = tf.nn.relu(conv4 + cnn4_b)
            pool4 = tf.nn.max_pool(hidden4, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

            # Flattern the convolution output
            shape = pool4.get_shape().as_list()
            reshape = tf.reshape(pool4, [shape[0], shape[1] * shape[2] * shape[3]])

            # 2 FC hidden layers
            fc1 = tf.nn.relu(tf.matmul(reshape, fc1_W) + fc1_b)
            fc2 = tf.nn.relu(tf.matmul(fc1, fc2_W) + fc2_b)
            fc3 = tf.nn.relu(tf.matmul(fc2, fc3_W) + fc3_b)

            # Return the result of the classifier
            return tf.matmul(fc3, classifier_W) + classifier_b

        # Training computation.
        logits = model(tf_train_dataset)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

        # Optimizer.
        optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        test_prediction = tf.nn.softmax(model(tf_test_dataset))

        
    num_steps = 1000
    
    with tf.Session(graph=graph) as session:
      tf.global_variables_initializer().run()
      print('Initialized')
      for step in np.arange(0,num_steps):
          print(step)
          offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
          batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
          batch_labels = train_labels[offset:(offset + batch_size), :]
          feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
          _, l, predictions = session.run(
              [optimizer, loss, train_prediction], feed_dict=feed_dict)
          print(train_prediction)
          # Show training result
          if ((step != 0) and (step % 1 == 0)):
            plot_confusion_matrix(batch_labels, predictions)
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
    
      # Testing and show testing result
      testpred = test_prediction.eval()
      print('Test accuracy: %.1f%%' % accuracy(testpred, test_labels))
      plot_confusion_matrix(test_labels, testpred)
      return precision, recall, falsepos

     