def train_neural_network(path, model):
    # loading dataset
    images = data_part.load_images(path + "/rsz_images_64x64/", 50, 64)
    images = np.array(images)
    images = clone_images(images, 20)  # clones images for heatmaps
    heatmaps = load_all_heatmaps(20, path)

    print ("loaded images for neural network: " + str(images.shape))
    print ("loaded heatmaps for neural network: " + str(heatmaps.shape))

    # normalization of heatmaps
    if max(heatmaps.flatten()) > 1:
        heatmaps /= max(heatmaps.flatten())

    # spliting dataset

    # test data
    test_images = images[900:1000]
    test_heatmaps = heatmaps[900:1000]
    # validation data
    valid_images = images[800:900]
    valid_heatmaps = heatmaps[800:900]
    # training data
    images = images[0:800]
    heatmaps = heatmaps[0:800]

    graph = tf.Graph()

    with graph.as_default():

        # input to NN - logits
        x = tf.placeholder('float', shape=[None, 64, 64, 3])
        # expected output - labels
        y = tf.placeholder('float', shape=[None, 64, 64])
        # value on dropout layer
        keep_prob = tf.placeholder(tf.float32)

        # creates model of neural network
        prediction = neural_network_model(x, keep_prob)

        # seting up parameters of model
        cross_entropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y)
        )
        train_step = tf.train.FtrlOptimizer(0.2).minimize(cross_entropy)

        validation_cross_entropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y)
        )
        test_cross_entropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y)
        )

        '''
        config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
        '''

        # add variables to log file
        tf.summary.scalar("cross_entropy", cross_entropy)
        tf.summary.scalar("validation_cross_entropy", cross_entropy)
        summary = tf.summary.merge_all()

        saver = tf.train.Saver()

        with tf.Session(graph=graph) as sess:

            sess.run(tf.global_variables_initializer())

            # set up directories for log files
            summary_writer = tf.summary.FileWriter(path + "logs/", sess.graph)
            validation_writer = tf.summary.FileWriter(path + "logs/val/",
                                                      sess.graph)

            end = False

            iteration_count = 0
            for i in range(0, 15):  # 15 iterations of training on whole
                # dataset
                iterations = 30  # 30 epoch of each iteration

                # randomly shuffles given dataset
                training_dataset = make_dataset_for_epoch(images, heatmaps)
                validation_dataset = make_dataset_for_epoch(valid_images,
                                                            valid_heatmaps)

                # sets up variables for feed_dict
                train_data = {
                    x: training_dataset[0],
                    y: training_dataset[1],
                    keep_prob: 0.5
                }
                val_data = {
                    x: validation_dataset[0],
                    y: validation_dataset[1],
                    keep_prob: 1.0
                }

                print ("start training on series: " + str(i))
                iteration_loss = 100000

                for count in range(0, iterations):  # start training
                    _, c = sess.run([train_step, cross_entropy],
                                    feed_dict=train_data)

                    if math.isnan(c) == True:  # too big loss
                        print ("died at: " + str(count))
                        return 0
                        break

                    print ("current loss: ", c)
                    iteration_count += 1
                    # writing trainig loss to log file
                    summary_str = sess.run(summary, feed_dict=train_data)
                    summary_writer.add_summary(summary_str, iteration_count)
                    summary_writer.flush()

                    # validate data
                    val = sess.run(validation_cross_entropy, feed_dict=val_data)
                    print ("validation loss: ", val)
                    # writing validation loss to log file
                    summary_str = sess.run(summary, feed_dict=val_data)
                    validation_writer.add_summary(summary_str, iteration_count)
                    validation_writer.flush()

                    # if validation loss starts rising, stop training
                    if iteration_loss < val:
                        print ("end of trainig, loss starts rising")

                        end = True
                        break
                    iteration_loss = val

                if end:
                    break

            # test model on test data
            test_data = {x: test_images, y: test_heatmaps, keep_prob: 1.0}
            test_loss = sess.run(test_cross_entropy, feed_dict=test_data)
            print('test loss: ' + str(test_loss))

            # saving model
            save_path = saver.save(sess, model)
            print ("saved in: %s" % save_path)
