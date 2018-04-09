def load_fixations (path): # loads all recordings of fixations from one export of experiment
    file = open(path, 'r')
    name = file.readlines()[1].rsplit(';', 13)[0] # gets name of first recording
    file = open(path, 'r')
    fixations_y = [] # data neccesary for us - fixations and durations
    fixations_x = []
    durations = []
    images_name = []
    for line in file.readlines()[1:]:
        line_splited = line.rsplit(';', 13) 
        if line_splited[3] == "grayscale.jpg" or line_splited[3] == "FINAL.JPG": 
			continue # skips grayscale and last image of experiment - it`s not part of dataset for NN
            
        if name != line_splited[0]: # new recording, saving fixations 
            save_sorted_by_participants(name, np.array(images_name), 
                                        np.array(fixations_x, dtype = float),
										np.array(fixations_y, dtype = float),
                                        np.array(durations, dtype = float))
            name = line_splited[0]
            fixations_y = [] # empties variables
            fixations_x = []
            durations = []
            images_name = []
            
        Y = int(line_splited[12].rsplit('\n',1)[0]) # fixation outside of image
        if Y < 45 or Y > 1130:
            continue

        fixations_y.append(Y/1200.0) # normalization of fixation
        fixations_x.append(int(line_splited[11])/1920.0)
        durations.append(line_splited[10])
        images_name.append(line_splited[3])

    save_sorted_by_participants(name, np.array(images_name),  # saving last recording
                                np.array(fixations_x, dtype = float), 
								np.array(fixations_y, dtype = float),
                                np.array(durations, dtype = float))
	
	
	
	