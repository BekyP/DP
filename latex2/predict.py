def predict(model, directory_with_images,
            directory_for_saved_predictions, heatmap_type,
            metrics_set=None):
    graph = tf.Graph()
    with graph.as_default():

        x = tf.placeholder('float', shape=[None, 64, 64, 3])
        y = tf.placeholder('float', shape=[None, 64, 64])
        keep_prob = tf.placeholder(tf.float32)

        # creates model of neural network
        prediction = neural_network_model(x, keep_prob)
        cross_entropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y)
        )

        '''
        config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
        '''

        # load names of all images in directory
        files = os.listdir(directory_with_images)
        images_for_prediction = []
        images_original = []

        for filename in files:  # loads images
            images_original.append(directory_with_images + filename)
            images_for_prediction.append(data_part.load_images_for_prediction(
                directory_with_images + filename)[0])

        saver = tf.train.Saver()

        with tf.Session(graph=graph) as sess:
            saver.restore(sess, model)  # restores saved model

            predicted_heatmaps = make_prediction(prediction,
                                                 {x: images_for_prediction,
                                                  keep_prob: 1.0})

            if metrics_set == 1:
                directory = input(
                    "enter directory with original heatmaps and fixations:\n"
                )
                # computes metrics
                count_metrics(predicted_heatmaps, len(images_original),
                              directory)

            for map, img in zip(predicted_heatmaps, images_original):
                print ("working on: " + str(img.rsplit('/', 1)[1]))
                # saving predicted heatmaps on image
                data_part.vizualize_heatmap(map, img,
                                            directory_for_saved_predictions,
                                            heatmap_type)
