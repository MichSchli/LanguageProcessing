__author__ = 'Michael'

from itertools import chain

def cross_validate(data, labels, k, create_and_fit, evaluate, params):
    #Split into evenly sized chunks
    samples_per_fold = len(data)/k
    data_folds = [list(t) for t in zip(*[iter(data)]*samples_per_fold)]
    label_folds = [list(t) for t in zip(*[iter(labels)]*samples_per_fold)]

    #Distribute the remainder evenly over the folds
    leftover_data = data[samples_per_fold*k:]
    leftover_labels = labels[samples_per_fold*k:]
    for i in xrange(len(leftover_data)):
        data_folds[i%k].append((leftover_data[i]))
        label_folds[i%k].append((leftover_labels[i]))

    acc = 0

    #Do the experiments
    for i in xrange(len(data_folds)):
        #Get a view of the data
        train_data = data_folds[:]
        train_labels = label_folds[:]

        #Construct training and test sets
        validate_data = train_data.pop(i)
        validate_labels = train_labels.pop(i)
        train_data = list(chain.from_iterable(train_data))
        train_labels = list(chain.from_iterable(train_labels))

        #Create a classifier and fit it to the data:
        cvf = create_and_fit(train_data, train_labels, *params)

        #Evaluate accuracy
        acc += evaluate(cvf, validate_data, validate_labels)

    return acc/float(k)
