# Made by Isaac Joffe

import tensorflow as tf    # for creating and training deep learning models
import numpy as np    # for formatting of deep learning models
import argparse    # for changing program behaviour from command line


"""
The beam-support deep learning model takes the four coordinates of the beam and the support
and determines if they are connected or not. The interface is detailed below:

    Inputs: 
        beam box coordinates (x1, y1, x2, y2)
        support box coordinates (x1, y1, x2, y2)

    Outputs:
        whether the beam and support are connected (con)

The beam-load deep learning model takes the four coordinates of the beam and the load
and determines if they are connected or not. The interface is detailed below:

    Inputs: 
        beam box coordinates (x1, y1, x2, y2)
        load box coordinates (x1, y1, x2, y2)

    Outputs:
        whether the load acts on the beam (con)

The load-number deep learning model takes the four coordinates of the load and the number
and determines if they are connected or not. The interface is detailed below:

    Inputs: 
        load box coordinates (x1, y1, x2, y2)
        number box coordinates (x1, y1, x2, y2)

    Outputs:
        whether the number represents the load (con)

All coordinates must be normalized properly before being inputted into the model. All x-coordinates
must be expressed as a fraction of the largest x-coordinate, and all y-coordinates must be expressed
as a fraction of the largest y-coordinate.

All three models achieve a very high accuracy, even at a low epoch number and using cross-validation.
"""


"""
Function to parse the text files containing the training data.

    Parameters:
        data_files (list of strings) : a list of all files to parse for data
        preprocess (string) : whether or not to preprocess data ('yes' or 'no')
    
    Returns:
        input_data (list of lists) : a list of all input data
        output_data (list of lists) : a list of the corresponding outputs
"""
def load_data(data_files, preprocess):
    # initialize arrays to hold data and labels
    input_data = []
    output_data = []

    # combine data from all files requested by the user
    for file_name in data_files:
        with open(file_name, "r") as file:
            while True:
                # read in each line of the file
                line = file.readline().strip()
                if line == "EOF":
                    break    # EOF sentinel means no more data to read
                elif not line:
                    file.readline()    # skip blank lines and following line containing image number
                else:
                    # convert line string into a list of floating point numbers
                    line = line.split()
                    line = [float(value) for value in line]
                    output = line.pop()    # store corresponding output

                    if preprocess == 'yes':
                        # determine the largest and smallest coordinates for normalization
                        xmin = min(line[0], line[2], line[4], line[6])
                        xmax = max(line[0], line[2], line[4], line[6])
                        ymin = min(line[1], line[3], line[5], line[7])
                        ymax = max(line[1], line[3], line[5], line[7])
                        # convert coordinates to fractional form so smallest is 0 and largest is 1
                        for i in range(len(line)):
                            if i % 2 == 0:
                                line[i] = (line[i] - xmin) / (xmax - xmin)
                            else:
                                line[i] = (line[i] - ymin) / (ymax - ymin)
                        # perform reflections to augment data and increase data size
                        xline = [((1 - value) if (i % 2 == 0) else value) for value in line]
                        yline = [((1 - value) if (i % 2 != 0) else value) for value in line]
                        xyline = [(1 - value) for value in line]

                        # store data in the output variables of function
                        input_data.append(line)
                        input_data.append(xline)
                        input_data.append(yline)
                        input_data.append(xyline)
                        output_data.append(output)
                        output_data.append(output)
                        output_data.append(output)
                        output_data.append(output)
                    else:
                        # no processing, just store raw data
                        input_data.append(line)
                        output_data.append(output)

    return input_data, output_data


"""
Function to create the deep-learning based TensorFlow model to be used.

    Parameters:
        none

    Returns:
        model (TensorFlow sequential model) : skeleton model to be used
"""
def load_model():
    # create deep learning model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(20,)))    # eight inputs representing the four coordinates of each box
    # model.add(tf.keras.layers.Dense(128, activation='relu'))    # hidden layer in the network
    model.add(tf.keras.layers.Dense(32, activation='relu'))    # hidden layer in the network
    model.add(tf.keras.layers.Dense(8, activation='relu'))    # hidden layer in the network
    model.add(tf.keras.layers.Dense(1))    # single output representing whether they are attached

    # compile model
    model.compile(optimizer='adam',    # basic optimizer function
        loss=tf.keras.losses.binary_crossentropy,    # basic loss fucntion
        metrics=[    # useful information on effectiveness of model
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.FalseNegatives(name='fn')]
    )

    return model


"""
Function to fold the data into equal partitions for cross validation.

    Paramters:
        number_of_folds (int) : number of partitions to create from the data
        input_data (list of lists) : a list of all input data from all 215 images
        output_data (list of lists) : a list of the corresponding outputs

    Returns:
        train_input (list of lists) : a list of lists of training input data for each fold
        train_output (list of lists) : a list of lists of correspodning training output data for each fold
        test_input (list of lists) : a list of lists of testing input data for each fold
        test_output (list of lists) : a list of lists of corresponding testing output data for each fold
"""
def fold_data(number_of_folds, input_data, output_data):
    # initilize arrays to hold partitioned input and output data
    train_input = [[] for i in range(number_of_folds)]
    train_output = [[] for i in range(number_of_folds)]
    test_input = [[] for i in range(number_of_folds)]
    test_output = [[] for i in range(number_of_folds)]

    # iterate over each training instance
    for i in range(len(input_data)):
        # iterate over each partition of the dataset
        for j in range(number_of_folds):
            # save each training instance in either the training or testing set
            # split up consecutive instances to diversify each fold
            if i % number_of_folds != j:
                train_input[j].append(input_data[i])
                train_output[j].append(output_data[i])
            else:
                test_input[j].append(input_data[i])
                test_output[j].append(output_data[i])

    return train_input, train_output, test_input, test_output


"""
Function to test the model by performing cross validation.

    Parameters:
        train_input (list of lists) : a list of lists of training input data for each fold
        train_output (list of lists) : a list of lists of correspodning training output data for each fold
        test_input (list of lists) : a list of lists of testing input data for each fold
        test_output (list of lists) : a list of lists of corresponding testing output data for each fold
        num_epochs (int) : number of epochs for model to run when training

    Returns:
        none
"""
def test_model(train_input, train_output, test_input, test_output, num_epochs):
    # initialize variable to hold test results
    test = []

    # train and test model on each fold of the data set
    for i in range(len(train_input)):
        model = load_model()    # recreate a new model each time
        history = model.fit(train_input[i], train_output[i], epochs=num_epochs)    # 50 epochs works best
        test.append(model.evaluate(test_input[i], test_output[i]))    # store results data

    # print out information about the test results
    print("\nAccuracy:\n\tAverage Accuracy: {}\n\tAccuracy Values: {}\n".format(sum([run[1] for run in test])/len(train_input), [run[1] for run in test]))
    print("Precision:\n\tAverage Precision: {}\n\tPrecision Values: {}\n".format(sum([run[2] for run in test])/len(train_input), [run[2] for run in test]))
    print("Recall:\n\tAverage Recall: {}\n\tRecall Values: {}\n".format(sum([run[3] for run in test])/len(train_input), [run[3] for run in test]))
    print("Overall Confusion Matrix:\n\tTP: {}\n\tTN: {}\n\tFP: {}\n\tFN: {}\n".format(sum([run[4] for run in test]), sum([run[5] for run in test]), sum([run[6] for run in test]), sum([run[7] for run in test])))
    return


"""
Function to create a model trained on all possible test cases (the best or golden version).

    Parameters:
        data_files (list of strings) : a list of all files to parse for data
        preprocess (string) : whether or not to preprocess data ('yes' or 'no')
        num_epochs (int) : number of epochs for model to run when training
        model_name (string) : filename to write the golden model to

    Returns:
        none
"""
def create_golden_model(data_files, preprocess, num_epochs, model_name):
    # read in and preprocess training and test data from file
    input_data, output_data = load_data(data_files, preprocess)
    # create deep learning tensorflow model
    model = load_model()
    # fit model based on all training data
    model.fit(input_data, output_data, epochs=num_epochs)
    # save model to a file
    model.save(model_name)
    return


"""
Function to load the golden version of the model from a file.

    Parameters:
        model_name (string) : filename to write the golden model to

    Returns:
        model (TensorFlow sequential model) : golden version of deep learning model
"""
def load_golden_model(model_name):
    # load in model from files
    model = tf.keras.models.load_model(model_name)

    return model


"""
Function to activate the model on a single input case.

    Parameters:
        model (TensorFlow sequyential model) : deep learning model to use
        instance (list) : input test case for model
    
    Returns:
        prediction (float) : output of model for this case
"""
def test_single_instance(model, instance):
    instance = np.reshape(instance, (1, len(instance)))    # ensures input is valid
    # predict output for the spceified input based on specified model
    prediction = model.predict([instance])

    return prediction


# def test_single_ls(model, instance):
#     instance = np.reshape(instance, (1,20))    # ensures input is valid
#     # predict output for the spceified input based on specified model
#     prediction = model.predict([instance])

#     return prediction


"""
Function to support parsing of inputs from user regarding program settings.

    Parameters:
        none

    Returns:
        options (dictionary) : mapping of settings to the inputted values
"""
def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', nargs=1, default=['test'], type=str, choices=['test', 'create', 'both'], help='whether to make models for testing purposes or to create a goldne model')
    parser.add_argument('--source', '--src', nargs='+', type=str, required=True, help='list of files from which model training and test data is drawn')
    parser.add_argument('--preprocess', nargs=1, default=['yes'], type=str, choices=['yes', 'no'], help='whether or not to preprocess source data first')
    parser.add_argument('--folds', nargs=1, default=[5], type=int, help='the number of folds to do cross validation on')
    parser.add_argument('--epochs', nargs=1, default=[50], type=int, help='the number of epochs to undergo when training')
    parser.add_argument('--name', nargs=1, default=['golden_model'], type=str, help='name of golden model to be created')
    options = parser.parse_args()    # analyze command line invocation
    return options


def main():
    # read command line arguments into variables
    options = parse_options()

    if options.mode[0] == 'test' or options.mode[0] == 'both':
        # read in and preprocess training and test data from file
        input_data, output_data = load_data(options.source, options.preprocess[0])
        # partition data set to support cross validation
        train_input, train_output, test_input, test_output = fold_data(options.folds[0], input_data, output_data)
        # test the performance of the model
        test_model(train_input, train_output, test_input, test_output, options.epochs[0])

    if options.mode[0] == 'create' or options.mode[0] == 'both':
        # save the model fitted to all training data
        create_golden_model(options.source, options.preprocess[0], options.epochs[0], options.name[0])
    return


if __name__ == "__main__":
    main()
