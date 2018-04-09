def load_coordinates_for_image(file_with_coordinates, path):
    participant = file_with_coordinates[::-1]
    participant = int(participant[4:6][::-1])  # gets participant number
    image_names = np.loadtxt(file_with_coordinates,
                             usecols=(0,),
                             delimiter=',',
                             dtype=str)  # gets names of images
    # gets fixations and durations for images
    data = np.loadtxt(file_with_coordinates, usecols=(1, 2, 3),
                      delimiter=',')
    current_image = image_names[0]
    i = 0
    j = 0
    heatmap = np.zeros([64, 64])
    for name in image_names:
        # print name
        if current_image != name:  # new image, heatmap for old will be saved
            print ("saving: " + str(current_image))
            save_heatmaps(participant - 2, heatmap, i, path + "/heatmaps/")
            current_image = name
            i += 1
            heatmap = np.zeros([64, 64])

        point_x = int(round(data[j][0] * 63))  # X coordinate of fixation
        point_y = int(round(data[j][1] * 63))  # Y coordinate of fixation
        variance = int(data[j][2])

        if point_y > 63 or point_y < 0 \
                or point_x > 63 or point_x < 0:  # view outside of image
            j += 1
            continue
        sqrt_2_pi = 1 / math.sqrt(2 * math.pi * variance)
        # computing Gaussian distribution
        for y in range(0, 64):
            for x in range(0, 64):
                # distance between points
                ni = abs((point_x - x) ** 2) + abs((point_y - y) ** 2)
                # part of the counting
                heatmap[y][x] += sqrt_2_pi * math.e ** (-ni / (2 * variance))

        j += 1

    print ("saving: " + str(current_image))
    save_heatmaps(participant - 2, heatmap, i, path + "/heatmaps/")
