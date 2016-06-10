# exerciseDetector.py
# python 2.7
# Exercise Detection through Machine Learning
# Dectection uses two sensors (one on torso and one on left thigh)
# Current trained activities include :
#      Pushups, Squats, Starjumps, Situps, Standing, Sitting, Plank, Calf-Raises, Leg-Raises, Sleeping
# Results on validated data were very accurate

# Argument -validate
# Uses training data from ./data/training to class Exercises
# Validates data from ./data/Validates
# Results are displayed in a Confusion Matrix

# Argument -stream
# Uses accelerometer and gyroscope values
# These classifications and raw data are uploaded to a database
# where they can be downloaded from an app

import requests
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
import os, time, sys, getopt
from sklearn import grid_search
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from numpy import array
from numpy import fft
import pylab
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from scipy.stats.stats import pearsonr
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve
import serial
import urlparse
import psycopg2
import csv

import threading
import readchar
import json
import urlparse
import psycopg2
import csv

# Debug flag
from sklearn.grid_search import GridSearchCV

DEBUG = True
# Classifier random seed
RSEED = 42
# Classifier filters
FILTERS = {'Pushups': 0, 'Squats': 1, 'Starjumps': 2, 'Situps': 3, 'Standing': 4, 'Sitting': 5, 'Plank': 6, 'Calf-Raises':7, 'Leg-Raises': 8, 'Sleeping':9}
SAMPLING_TIME = 4


#################################################################################

def print_header_start(text):
    print "--------------------------------------------------------------------------------"
    print text

def print_header_end():
    print "--------------------------------------------------------------------------------"


#################################################################################

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def parse_records_from_json(data):
    records = []
    record = []
    first = 1
    flag = 0
    for entry in data:
        for attribute, value in entry.iteritems():
            if (str(attribute) != 'eid'):
                if (str(value) != '1'):
                    # flag if the value is not 1
                    flag = 1
        if (flag == 0):
            # the flag was not tripped, so this is the start of a record
            #print first
            if (first != 1):
                records.append(record)
            record = []
        else:
            # the flag was tripped, so this is a regular entry
            if first: first = 0
            record.append(entry)
            flag = 0
    return records


def calc_mag(x, y, z):
    """
    @brief  Calculates the magnitude across x, y,z
    @params x,y,z : time domain data
    @retval the magnitudes of x,y,z for each time instance
    """
    npx = np.array(x)
    npy = np.array(y)
    npz = np.array(z)
    mags = []

    for i, val in enumerate(x):
        mags.append(np.linalg.norm([npx[i], npy[i], npz[i]]))

    return mags

def produce_model(data, c1, c2 = 0):
    """
    @brief  Add a set of features to the class based on the sensor
    @params data : Sensor values retrieved in the time domain
            c1   : A second sensor to be correlated with
            c2   : A third sensor to be correlated with
    @retval returns an array of processed features
    """
    npdata = np.array(data)
    result = []
    result.append(min(data))
    result.append(max(data))
    result.append(max(data) - min(data)) # range
    result.append(np.mean(npdata))
    result.append(np.std(npdata))
    result.append(pearsonr(data, c1))
    if (c2 != 0):
        result.append(pearsonr(data, c2))
    result.append(abs(fft.fft(data))/len(data)) # Add FFT as feature.
    return result

def process_features(data):
    """
    @brief  Splits all sensor data and produces a model as useful features
    @params data : JSON string of all sensors values
    @retval An array of all features for that data sample
    """
    time_1 = [float(entry['t']) for entry in data if entry['sid'] == '1']
    ax_1 = [float(entry['ax']) for entry in data if entry['sid'] == '1']
    ay_1 = [float(entry['ay']) for entry in data if entry['sid'] == '1']
    az_1 = [float(entry['az']) for entry in data if entry['sid'] == '1']

    gx_1 = [float(entry['gx']) for entry in data if entry['sid'] == '1']
    gy_1 = [float(entry['gy']) for entry in data if entry['sid'] == '1']
    gz_1 = [float(entry['gz']) for entry in data if entry['sid'] == '1']

    time_2 = [float(entry['t']) for entry in data if entry['sid'] == '2']
    ax_2 = [float(entry['ax']) for entry in data if entry['sid'] == '2']
    ay_2 = [float(entry['ay']) for entry in data if entry['sid'] == '2']
    az_2 = [float(entry['az']) for entry in data if entry['sid'] == '2']

    gx_2 = [float(entry['gx']) for entry in data if entry['sid'] == '2']
    gy_2 = [float(entry['gy']) for entry in data if entry['sid'] == '2']
    gz_2 = [float(entry['gz']) for entry in data if entry['sid'] == '2']

    stuff = []

    transformed_dataset = stuff

    transformed_dataset.append(produce_model(ax_1, ay_1, az_1))
    transformed_dataset.append(produce_model(ay_1, ax_1, az_1))
    transformed_dataset.append(produce_model(az_1, ay_1, ax_1))
    transformed_dataset.append(produce_model(ax_2, ay_2, az_2))
    transformed_dataset.append(produce_model(ay_2, ax_2, az_2))
    transformed_dataset.append(produce_model(az_2, ay_2, ax_2))
    transformed_dataset.append(produce_model(gx_1, gy_1, gz_1))
    transformed_dataset.append(produce_model(gy_1, gx_1, gz_1))
    transformed_dataset.append(produce_model(gz_1, gy_1, gx_1))
    transformed_dataset.append(produce_model(gx_2, gy_2, gz_2))
    transformed_dataset.append(produce_model(gy_2, gx_2, gz_2))
    transformed_dataset.append(produce_model(gz_2, gy_2, gx_2))
    transformed_dataset.append(produce_model( \
        calc_mag(ax_1, ay_1, az_1), calc_mag(ax_2, ay_2, az_2), 0))
    transformed_dataset.append(produce_model( \
        calc_mag(gx_1, gy_1, gz_1), calc_mag(gx_2, gy_2, gz_2), 0))

    return transformed_dataset


class TrainingData:

    def __init__(self, validate = False, liveStream = False, liveData=[]):
        """
        Default: Stores data from training files and targets from within data/training
        validate == True : Stores features from files within data/validate
        liveStream == True : Uses date from liveData to store features in data
        """
        self._data = []
        self._target = []
        self._actvities = []
        self._validate = validate
        self._files = []
        self._time_1 = []
        self._time_2 = []

        if (liveStream):
            # print str(liveData)
            data = json.loads(str(liveData))
            transformed_dataset = process_features(data)
            return


        if (validate):
            for file in os.listdir("data/validate"):
                with open('data/validate/' + file) as data_file:
                    data = json.load(data_file)

                print "Validate File: " , file

                # Process Features
                transformed_dataset = process_features(data)

                self._files.append(file)
                self._data.append([item[0] for item in transformed_dataset])
            return


        # Do this for each training file
        for exercise, value in FILTERS.iteritems():

            for file in os.listdir("data/training"):
                if exercise and file.find(exercise) == -1:
                    continue
                with open('data/training/' + file) as data_file:
                    data = json.load(data_file)

                print "File: " , file

                # Process Features
                transformed_dataset = process_features(data)

                self._data.append([item[0] for item in transformed_dataset])
                self._target.append(value) #number
                self._actvities.append(exercise) #activity


    def getData(self):
        return self._data

    def getTarget(self):
        return self._target

    def getFiles(self):
        return self._files

    def getTime(self):
        return [self._time_1, self._time_2]


def clean_samples(record):
    ones = 0
    twos = 0
    target = 0
    extra = 0
    flag = 0
    data = []

    for entry in record:
        if (all(i == '0' for i in entry[1:])):
            flag = 1

        if (all(i == '1' for i in entry[1:])):
            pass

        elif (flag != 1):
            data.append(entry)

    for entry in data:
        if (entry[1] == '1'):
            ones += 1
        else:
            twos += 1

    if (ones == twos):
        return data
    elif (ones > twos):
        target = 1
        extra = ones - twos
    else:
        target = 2
        extra = twos - ones

    for entry in reversed(data):
        if (entry[1] == str(target)):
            data.remove(entry)
            extra -= 1
            if (extra == 0):
                return data


def create_json(record):

    cleaned = clean_samples(record)
    data = []
    e = {}
    for entry in cleaned:
        e['eid'] = entry[0]
        e['sid'] = entry[1]
        e['t'] = entry[2]
        e['ax'] = entry[3]
        e['ay'] = entry[4]
        e['az'] = entry[5]
        e['gx'] = entry[6]
        e['gy'] = entry[7]
        e['gz'] = entry[8]
        data.append(e)
        e = {}

    if (len(data) < 2):
        print("Malformed data.")
        return

    return json.dumps(data)

    mapping = {}
    mapping[str(cleaned[0][0])] = "UNCATEGORISED"

    with open("categorise.csv") as f:
        while True:
            category = f.readline().strip()
            datasets = f.readline().strip()
            eids = datasets.strip().split(",")
            for eid in eids:
                mapping[eid.strip()] = category
            if not datasets: break #EOF

    mapping[''] = "UNCATEGORISED"

    with open("data/data_" + str(cleaned[0][0]) + "_" + mapping[str(cleaned[0][0])] + ".json", "w") as f:
        print "Writing", "data_" + str(cleaned[0][0])+ "_" + mapping[str(cleaned[0][0])] + ".json..."
        f.write("[")
        for item in data[:-1]:
            f.write("%s,\n" % json.dumps(item))
        f.write("%s]\n" % json.dumps(data[-1]))



head_id = 0 # Used to keep track of the mutliple samples uploaded to the database
# read loop for serial port
def read_loop(arg1):

    clf = arg1
    with open('head_id.txt') as f:
        #Read head ID
        for line in f:
            data = line.split()
            head_id = int(data[0])
            #print "Val: ", data, head_id

            output = ''
            parsed_data = []

            print "Waiting for new exercise....."
            while read_data:
                try:

                    data = s.read();
                    if len(data) > 0:
                        output += data
                        if (data[-1]=='\n'):
                            sys.stdout.write('.')
                            print output
                            parsed_json_all = json.loads(output)  #parse rec json to py

                            #check if data or stop/start packet recv
                            if (len(parsed_json_all) > 1):
                                #collect recv data
                                for i in range(4):
                                    parsed_json = parsed_json_all[i]
                                    #append to master array
                                    parsed_data.append([head_id, str(parsed_json['ID']), str(parsed_json['T']), str(parsed_json['AX']), str(parsed_json['AY']), str(parsed_json['AZ']), str(parsed_json['GX']), str(parsed_json['GY']), str(parsed_json['GZ'])])
                            else:
                                parsed_json = parsed_json_all[0]
                                parsed_data.append([head_id, str(parsed_json['ID']), str(parsed_json['T']), str(parsed_json['AX']), str(parsed_json['AY']), str(parsed_json['AZ']), str(parsed_json['GX']), str(parsed_json['GY']), str(parsed_json['GZ'])])
                                if (parsed_json['AX'] == 0):

                                    #reconnect to database
                                    urlparse.uses_netloc.append("postgres")
                                    url = urlparse.urlparse("postgres://obkenkootsqtvz:MaeQz3GQPJ1hIPknnNHcrgCNMB@ec2-54-83-27-147.compute-1.amazonaws.com:5432/d3nbtq2btt810j")
                                    conn = psycopg2.connect(
                                            database=url.path[1:],
                                            user=url.username,
                                            password=url.password,
                                            host=url.hostname,
                                            port=url.port
                                            )
                                    cur = conn.cursor()
                                    #write id to txt file
                                    #print 'ID: ' + head_id
                                    head_id += 1
                                    with open('head_id.txt', 'w') as k:
                                        k.write('%d' % head_id)
                                    k.close()
                                    #write to csv
                                    print "End of sample data"
                                    f = open("sample_data.csv", "wb+")
                                    writer = csv.writer(f, delimiter='\t')
                                    writer.writerows(parsed_data)
                                    f.seek(0)
                                    #send to database
                                    cur.copy_from(f, 'raw_data')
                                    conn.commit();
                                    print "Copied to raw_data"
                                    f.close()
                                    conn.close()
                                    #clear all buffers
                                    output = ''
                                    parsed_json_all = ''

                                    # Get features from data

                                    for i in range(0,len(parsed_data)):
                                        parsed_data[i] = tuple(parsed_data[i])

                                    data = create_json(parsed_data)
                                    # print "Data is this: ",data
                                    dataSet = TrainingData(validate=False,liveStream=True, liveData=data)
                                    samples = dataSet.getData()
                                    pr = clf.predict(samples)

                                    #reconnect to database
                                    urlparse.uses_netloc.append("postgres")
                                    url = urlparse.urlparse("postgres://obkenkootsqtvz:MaeQz3GQPJ1hIPknnNHcrgCNMB@ec2-54-83-27-147.compute-1.amazonaws.com:5432/d3nbtq2btt810j")
                                    conn = psycopg2.connect(
                                            database=url.path[1:],
                                            user=url.username,
                                            password=url.password,
                                            host=url.hostname,
                                            port=url.port
                                            )
                                    cur = conn.cursor()
                                    #write id to txt file
                                    #print 'ID: ' + head_id
                                    #write to csv
                                    name123abc = ""
                                    for name, clas in FILTERS.iteritems():
                                        if (pr[0] == clas):
                                            #true_data.append(clas)
                                            name123abc = name

                                    maxtime = max(max(dataSet.getTime()[0]), max(dataSet.getTime()[1]))
                                    mintime = min(min(dataSet.getTime()[0]), min(dataSet.getTime()[1]))
                                    diff = round((maxtime - mintime) / 12.0 * 2.0, 3)

                                    cur.execute("INSERT INTO learn_data VALUES (%s, %s, %s, %s, %s, %s)", (head_id - 1, name123abc, 0, 0, 0, diff))
                                    conn.commit();
                                    print "Classified Live Sample as: ", name123abc
                                    print "Copied to learn_data"
                                    f.close()
                                    conn.close()
                                    parsed_data = []

                                    print "\nWaiting for new exercise....."
                            output = ''
                            parsed_json_all = ''
                except Exception, e:
                    #except:
                    print "Exception:", e
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                    output = ''
                    parsed_json_all = ''
                    parsed_data = []
                    #print "close serial port"
                    #s.clost()


                    # close serial port
                    print "close serial port"
                    #s.close()

def main(argv):
    """ Main Program """
    if (len(argv) < 1):
        print "Use: -valiate | -stream"
        return 0

    print "Loading training files..."

    exponential_range = [int(pow(10, i)) for i in range(0, 4)]
    feature_range = [int(pow(2, i)) for i in range(0, 20)]
    feature_range.append('auto')
    parameters = {
                  'max_depth': exponential_range,
                  'max_features': ['auto'],
                  'n_estimators': exponential_range
    }
    print "Fitting model with training data....."
    classifier = RandomForestClassifier(random_state=42)
    clf = GridSearchCV(classifier, parameters, verbose=True, n_jobs=8)


    train = TrainingData()

    clf.fit(train.getData(), train.getTarget())

    print clf

    # Check Argument
    if (argv[0] == "-validate"):
        # Download data to be predicted
        validate = TrainingData(True)
        samples = validate.getData()

        print_header_start("Validation Data Classification: ")
        true_data = []
        pr = clf.predict(samples)
        predicted_data = pr
        print validate.getFiles()
        print pr
        for i, val in enumerate(validate.getFiles()):
            sys.stdout.write(val)
            sys.stdout.write(" ---------------> ")
            for name, clas in FILTERS.iteritems():
                if val.find(name) != -1:
                    true_data.append(clas)
                if (pr[i] == clas):
                    print name
            print "-------------------------"
        print_header_end()

        # Plot confusion Matrix
        # confusion matrix C is such that C_{i, j} is equal to the number of observations known to be in group i but predicted to be in group j.
        #true_data = [0,1,3,1]      # The correct classifications of the data
        #predicted_data = pr # The predicted classifications of the data
        true_data_val = []
        predicted_data_val = []

        for i, val in enumerate(true_data):
            for key, value in FILTERS.iteritems():
                if value == val:
                    true_data[i] = key
                    true_data_val.append(value)
        # print true_data

        predicted_data = predicted_data.tolist()
        for i, val in enumerate(predicted_data):
            for key, value in FILTERS.iteritems():
                if value == val:
                    predicted_data[i] = key
                    predicted_data_val.append(value)

        # print predicted_data
        cm = confusion_matrix(true_data, predicted_data, labels=['Pushups', 'Squats', 'Starjumps', 'Situps', 'Standing', 'Sitting', "Plank", "Calf-Raises", "Leg-Raises", "Sleeping"])
        print FILTERS
        print "Confusion Matrix"
        print cm
        plt.matshow(cm)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        return 0

    elif (argv[0] == "-stream"):
        # Stream data in to be classified
        s = serial.Serial(port = 'COM8', baudrate = 115200) #SensorTag

        read_data = True
        t1 = threading.Thread(target=read_loop, args=(clf,))
        t1.start()

        # send loop for serial port
        while True:
          try:
            #for command in ['0' , '1', '2', 'a', 'i', 't']:
              #command=readchar.readchar()
              #s.write(command)
              time.sleep(1)
          except KeyboardInterrupt:
            print "Shutdown"
            s.close()
            break

        read_data = False
        return 0

    else:
        print "Invalid Argument. Use: -validate or -stream"
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
