import numpy as np
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
# Implementation for asynchronous Hopfield neural network
class Hopfield_Neural_Network:
    def __init__(self,nodes,iterations=100,weights=None):
        self.nodes = nodes
        self.iterations = iterations
        try:
            if weights == None:
                self.weights = np.zeros((nodes,nodes))
        except ValueError:
            self.weights = weights
    def store(self,input):
        dW = np.outer(input,input)
        np.fill_diagonal(dW,0)
        self.weights += dW

    def recall(self,input,range_=None):
        if type(range_) == tuple: # Can specify range of nodes to iterate over (i.e. nodes that are "input" are known as correct)
            a = range(range_[0],range_[1])
        else:
            a = self.nodes
        update_sequence = np.random.choice(a, self.iterations)
        for node in update_sequence:
            input[node] = np.sign(np.inner(input,self.weights[:,node]))
        return input
    def setIter(self,iter_):
        self.iterations = iter_
    def save_weights(self,filename):
        np.save(filename, self.weights)

    def load_weights(self,filename):
        weights = np.load(filename)
        if weights.shape == (self.nodes, self.nodes):
            self.weights = weights
        else:
            raise ValueError("Dimensions of specified weight array does not match network weight dimensions!")

import re
class matrix_expansion:
    def __init__(self,active_motors):
        self.active_motors = active_motors
        active_sensors = np.array(active_motors*2)
        active_sensors[len(active_motors):] += 14
        self.active_sensors = active_sensors
        self.shape = ()

    def load_from_file(self,filename):
        f = open(filename,"r")
        matrix = f.read()
        f.close()
        matrix = re.split(",NEW_ROW,",matrix)
        matrix.pop()
        matrix = np.array([np.array(re.split(",", row)).astype(np.float) for row in matrix])
        self.shape = matrix.shape
        return matrix

    def reduced_matrix(self,matrix):
        matrix = matrix[:,self.active_sensors][self.active_motors]
        return matrix
    def expanded_matrix(self,reduced_matrix):
        matrix = np.zeros(self.shape)
        flat = reduced_matrix.flatten()
        matrix = np.zeros((14,28))
        k = 0
        for i in active_motors:
            for j in active_sensors:
                matrix[i,j] = flat[k]
                k += 1
        return matrix

active_motors = [1,3,4,5,10,12]

expander = matrix_expansion(active_motors)
# Front back
filename = "/home/markus/dep/dep_matrices/front_back.dep"
fb_matrix = expander.load_from_file(filename)
fb_reduced = expander.reduced_matrix(fb_matrix)
#fb_expanded = expander.expanded_matrix(fb_reduced)
# Front side
filename = "/home/markus/dep/dep_matrices/front_side.dep"
fs_matrix = expander.load_from_file(filename)
fs_reduced = expander.reduced_matrix(fs_matrix)
#fs_expanded = expander.expanded_matrix(fs_reduced)
# Side down
filename = "/home/markus/dep/dep_matrices/side_down.dep"
sd_matrix = expander.load_from_file(filename)
sd_reduced = expander.reduced_matrix(sd_matrix)
#sd_expanded = expander.expanded_matrix(sd_reduced)
# Zero
zero_reduced = np.zeros(fb_reduced.shape)
matrices = {"fb": fb_reduced, "fs": fs_reduced, "sd": sd_reduced, "zero": zero_reduced}

import pickle
behaviors = ["zero","fb","fs","sd"]
transitions = {("fb","fs"): [], ("fb","sd"): [], ("fs","fb"): [], ("fs","sd"): [], ("sd","fb"): [], ("sd","fs"): []}

fb = pickle.load(open("/home/markus/dep/dep_data/bases/fb.pickle","rb"))
fs = pickle.load(open("/home/markus/dep/dep_data/bases/fs.pickle","rb"))
sd = pickle.load(open("/home/markus/dep/dep_data/bases/sd.pickle", "rb"))
zero = []
for array in fb:
    zero.append(np.zeros(array.shape))
bases = {"fb": fb, "fs": fs, "sd": sd}#, "zero": zero}
# obtained from plot, time indices that meat transition_muscle_2 condition
fb_t = 124
fs_t = 126
sd_t = 117
zero_t = 0
# pos data
pos = {"fb": fb[0][fb_t][active_motors], "fs": fs[0][fs_t][active_motors], "sd": sd[0][sd_t][active_motors], "zero": zero[0][zero_t][active_motors]}
# vel data
vel = {"fb": fb[1][fb_t][active_motors], "fs": fs[1][fs_t][active_motors], "sd": sd[1][sd_t][active_motors], "zero": zero[1][zero_t][active_motors]}
n = len(behaviors)
start = 0.0
width = 1.0/float(n)
brain_ranges = {}
for i in range(1,n+1):
    brain_ranges[behaviors[i-1]] = ((i-1)*width-start, i*width-start)
    brain_id = {}
for behavior in behaviors:
    brain_id[behavior] = (brain_ranges[behavior][0]+brain_ranges[behavior][1])/2

class scalar_sdr:
    def __init__(self, b, w, min_, max_, shape=0, neg=True):
        if type(b) != int or type(w) != int or type(min_) != float or type(max_) != float:
            raise TypeError("b - buckets must be int, w - width must be int, min_ must be float and max_ must be float")
        self.b = b # must be int
        self.w = w # must be int
        self.min = min_ # must be float
        self.max = max_ # must be float
        self.n = b+w-1 # number of units for encoding
        self.ndarray_shape = shape
        self.neg = neg

    def encode(self,input_):
        if input_ > self.max or input_ < self.min:
            raise ValueError("Input outside encoder range!")
        if type(input_) != float:
            raise TypeError("Input must be float!")
        if self.neg:
            output = np.zeros(self.n)-1
        else:
            output = np.zeros(self.n)
        index = int((input_-self.min)/(self.max-self.min)*self.b)
        output[index:index+self.w] = 1
        return output
    def encode_ndarray(self,input_):
        if input_.shape != self.ndarray_shape:
            raise ValueError("Input dimensions do not match specified encoder dimensions!")
        output = []
        for i in np.nditer(input_, order='K'):
            output.append(self.encode(float(i)))
        return np.array(output)

    def decode(self,input_):
        if len(input_) != self.n: # or len(np.nonzero(input_+1)[0]) != self.w: <-- Can't have since the network is not guaranteed to produce this by any means!!!
            raise TypeError("Input does not correspond to encoder encoded data!")
        if len(np.nonzero(input_+1)[0]) == 0:
            return np.nan
        max_ = 0
        output = 0.0
        for i in range(self.b):
            x = np.zeros(self.n)-1
            x[i:i+self.w] = 1
            score = np.sum(np.array(x)*input_)
            if score > max_:
                max_ = score
                output = float(i)/float(self.b)*(self.max-self.min)+self.min
        return output

    def decode_ndarray(self,input_):
        if input_.shape != (reduce(lambda x, y: x*y, self.ndarray_shape)*self.n,): 
            raise ValueError("Input dimensions do not match specified encoder dimensions!")
        input_ = input_.reshape(self.ndarray_shape+(self.n,))
        output = []
        for i in np.ndindex(self.ndarray_shape):
            output.append(self.decode(input_[i]))
        output = np.array(output).reshape(self.ndarray_shape)
        return output
    def set_ndarray_shape(self,shape):
        if type(shape) != tuple:
            raise TypeError("Must provide tuple of array dimensions!")
        self.ndarray_shape = shape

import time
import matplotlib.pyplot as plt

def single_cell_hnn():
    matrix_shape = (1,)
    m_encoder = scalar_sdr(40,22,-0.25,0.25,matrix_shape)
    b_encoder = scalar_sdr(46,20,0.0,1.0)
    # Calculate number of nodes of hopfield net
    nodes = m_encoder.n*reduce(lambda x, y: x*y, m_encoder.ndarray_shape) + b_encoder.n
    # i) Initialize hopfield network for each matrix value
    # ii) Store data for each brain_id+matrix[brain_id] for given matrix index
    nns = {}
    for index in np.ndindex(matrices["zero"].shape):
        hnn = Hopfield_Neural_Network(nodes)
        for id_ in brain_id:
            data = np.array([])
            matrix = matrices[id_][index].reshape(matrix_shape)
            matrix = m_encoder.encode_ndarray(matrix)
            data = np.append(data,matrix.flatten())
            brain_sig = brain_id[id_]
            brain_sig = b_encoder.encode(brain_sig)
            data = np.append(data,brain_sig)
            hnn.store(data)
        nns[index] = hnn
    # Recall matrix values from brain_id
    # setup some arrays to hold data
    nans_avg = []
    diffs = []
    durations = []
    # run recall fo network 'samples' number of times
    samples = 10
    for i in range(samples):
        nans = 0
        diffs_m = []
        # for each index in the DEP matrix
        for index in np.ndindex(matrices["zero"].shape):

            # for each brain id
            for id_ in brain_id:

                # generate data
                data = np.zeros(nodes)-1
                brain_sig = brain_id[id_]
                brain_sig = b_encoder.encode(brain_sig)
                data[-b_encoder.n:] = brain_sig

                # set iteration number
                nns[index].setIter(300)

                # recall matrix value
                t0 = time.time()
                mem = nns[index].recall(data,(0,m_encoder.n))
                duration = time.time()-t0
                durations.append(duration)

                # decode matrix value
                matrix_out = mem[0:m_encoder.n*reduce(lambda x, y: x*y, m_encoder.ndarray_shape)]

                # calculate error of recalled matrix value
                matrix_decoded = m_encoder.decode_ndarray(matrix_out)
                diff_m = m_encoder.decode_ndarray(m_encoder.encode_ndarray(matrices[id_][index].reshape(matrix_shape)).flatten())-matrix_decoded

                if np.isnan(diff_m):
                    nans += 1
                diffs_m.append(diff_m)
        nans_avg.append(nans)
        diffs.append(abs(np.array(diffs_m)))
    # print meta data
    print "nans: ", np.max(nans_avg)
    print "mean: ", np.mean(np.array(diffs).flatten())
    print "max: ", np.max(np.array(diffs).flatten())
    print "std: ", np.std(np.array(diffs).flatten())
    print "duration: ", np.mean(durations)
    # Bar graph plot
    # plotting constants 
    max_ = 0.5
    bars = 40
    # calculate category indices
    categories = []
    for i in range(bars):
        categories.append(i*max_/bars)
    
    width = categories[1]
    # tally number of instances for each category
    values = np.zeros(bars)
    for diff in np.array(diffs).flatten():
        i = int(diff/(max_+0.0000001)*bars)
        values[i] += 1
    # divide by number of samples to obtain normalized score
    values /= samples
    values /= 288.0
    # plot bar graph
    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()
    fontP.set_size('small')
    plt.figure(1)
    plt.title("Average error distribution across 10 samples")
    plt.bar(categories, values, width, align='center')
    plt.xlabel("Error bars - width = %0.4f" %(max_/bars))
    plt.ylabel("Proportion")
    plt.show()
    # print number of values that fall into first category (good approximation of how well the memory does)
    print "Values[0]: ", values[0]

def full_matrix_hnn():

    m_encoder = scalar_sdr(40,22,-0.25,0.25,(6,12))
    b_encoder = scalar_sdr(460,200,0.0,1.0)
    # calculate number of nodes
    nodes = m_encoder.n*reduce(lambda x, y: x*y, m_encoder.ndarray_shape) + b_encoder.n
    # initialize hopfield net
    hnn = Hopfield_Neural_Network(nodes)

    # store data in hnn i.e. brain_id -- matrix[brain_id] pairs
    for id_ in brain_id:
        data = np.array([])
        matrix = matrices[id_]
        matrix = m_encoder.encode_ndarray(matrix)
        data = np.append(data,matrix.flatten())

        brain_sig = brain_id[id_]
        brain_sig = b_encoder.encode(brain_sig)
        data = np.append(data,brain_sig)

        hnn.store(data)

    durations = []
    diffs = []
    samples = 10
    for i in range(samples):
        for id_ in matrices:
            # setup input with complete brain_id signal
            data = np.zeros(m_encoder.n*reduce(lambda x, y: x*y, m_encoder.ndarray_shape))-1
            brain_sig = brain_id[id_]
            brain_sig = b_encoder.encode(brain_sig)
            data = np.append(data,brain_sig)
        
            # set number of iterations -- proportional to number of nodes (i.e. 300*72 = 21600)
            hnn.setIter(21600)

            # recall matrix value
            t0 = time.time()
            mem = hnn.recall(data)
            duration = time.time()-t0
            durations.append(duration)

            # decode recalled matrix
            mem_decoded = m_encoder.decode_ndarray(mem[0:m_encoder.n*reduce(lambda x, y: x*y, m_encoder.ndarray_shape)])

            # calculate recall error
            matrix_decoded = m_encoder.decode_ndarray(m_encoder.encode_ndarray(matrices[id_]).flatten())
            diff = mem_decoded-matrix_decoded
            diffs.append(abs(np.array(diff)))

    # print meta data
    print "mean: ", np.mean(np.array(diffs).flatten())
    print "max: ", np.max(np.array(diffs).flatten())
    print "std: ", np.std(np.array(diffs).flatten())
    print "duration: ", np.mean(durations)

    # Bar graph plot
    # plotting constants 
    max_ = 0.5
    bars = 40

    # calculate category indices
    categories = []
    for i in range(bars):
        categories.append(i*max_/bars)
    width = categories[1]

    # tally number of instances for each category
    values = np.zeros(bars)
    for diff in np.array(diffs).flatten():
        i = int(diff/(max_+0.0000001)*bars)
        values[i] += 1
    # divide by number of samples to obtain normalized score
    values /= samples
    values /= 288.

    # plot bar graph
    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()
    fontP.set_size('small')
    plt.figure(1)
    plt.title("Average error distribution across 10 samples")
    plt.bar(categories, values, width, align='center')
    plt.xlabel("Error bars - width = %0.4f" %(max_/bars))
    plt.ylabel("Proportion")
    plt.show()
    # print number of values that fall into first category (good approximation of how well the memory does)
    print "Values[0]: ", values[0]

class LAM:
    def __init__(self,shape,weights=None):
        self.shape = shape
        try:
            if weights == None:
                self.weights = np.zeros(shape)
        except ValueError:
            self.weights = weights
    
    def store(self,input,output):
        dW = np.outer(input,output)
        self.weights += dW
        
    def recall(self,input):
        u = np.matrix(input)*self.weights
        output = np.sign(u)
        return output
    
    def save_weights(self,filename):
        np.save(filename, self.weights)
        
    def load_weights(self,filename):
        weights = np.load(filename)
        if weights.shape == (self.nodes, self.nodes):
            self.weights = weights
        else:
            raise ValueError("Dimensions of specified weight array does not match network weight dimensions!")

def single_cell_lam():
    matrix_shape = (1,)
    m_encoder = scalar_sdr(100,21,-0.25,0.25,matrix_shape,neg=False)
    b_encoder = scalar_sdr(100,21,0.0,1.0,neg=False)

    nodes = m_encoder.n*reduce(lambda x, y: x*y, m_encoder.ndarray_shape) + b_encoder.n

    nns = {}
    for index in np.ndindex(matrices["zero"].shape):
        lma = LAM((b_encoder.n,m_encoder.n))
        for id_ in brain_id:
            data = np.array([])
            matrix = matrices[id_][index]
            matrix = m_encoder.encode(float(matrix))

            data = np.append(data,matrix.flatten())

            brain_sig = brain_id[id_]
            brain_sig = b_encoder.encode(brain_sig)

            data = np.append(data,brain_sig)

            lma.store(brain_sig,matrix)

        nns[index] = lma

    diffs = []
    durations = []
    samples = 1
    for i in range(samples):
        for index in np.ndindex(matrices["zero"].shape):
            for id_ in brain_id:
                brain_sig = brain_id[id_]
                brain_sig = b_encoder.encode(brain_sig)

                t0 = time.time()
                mem = nns[index].recall(brain_sig)
                duration = time.time()-t0
                durations.append(duration)
                
                matrix_decoded = m_encoder.decode(mem.reshape(m_encoder.n,1))

                diff = m_encoder.decode(m_encoder.encode(float(matrices[id_][index])))-matrix_decoded
                diffs.append(abs(np.array(diff)))

    # print meta data
    print "mean: ", np.mean(np.array(diffs).flatten())
    print "max: ", np.max(np.array(diffs).flatten())
    print "std: ", np.std(np.array(diffs).flatten())
    print "duration: ", np.mean(durations)

    # Bar graph plot
    # plotting constants 
    max_ = 0.5
    bars = 40

    # calculate category indices
    categories = []
    for i in range(bars):
        categories.append(i*max_/bars)
    width = categories[1]

    # tally number of instances for each category
    values = np.zeros(bars)
    for diff in np.array(diffs).flatten():
        i = int(diff/(max_+0.0000001)*bars)
        values[i] += 1
    # divide by number of samples and ids to obtain normalized score
    values /= samples
    values /= 288.

    # plot bar graph
    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()
    fontP.set_size('small')
    plt.figure(1)
    plt.title("Average error distribution across 10 samples")
    plt.bar(categories, values, width, align='center')
    plt.xlabel("Error bars - width = %0.4f" %(max_/bars))
    plt.ylabel("Proportion")
    plt.show()
    # print number of values that fall into first category (good approximation of how well the memory does)
    print "Values[0]: ", values[0]

def full_matrix_lam():
    matrix_shape = matrices["zero"].shape
    m_encoder = scalar_sdr(100,21,-0.25,0.25,matrix_shape,neg=False)
    b_encoder = scalar_sdr(100,21,0.0,1.0,neg=False)

    shape = (b_encoder.n,m_encoder.n*reduce(lambda x, y: x*y, m_encoder.ndarray_shape))

    lma = LAM(shape)
    for id_ in brain_id:
        data = np.array([])
        matrix = matrices[id_]
        matrix = m_encoder.encode_ndarray(matrix)

        data = np.append(data,matrix.flatten())

        brain_sig = brain_id[id_]
        brain_sig = b_encoder.encode(brain_sig)

        data = np.append(data,brain_sig)

        lma.store(brain_sig,matrix)

    diffs = []
    durations = []
    for i in range(1):
        for id_ in brain_id:
            brain_sig = brain_id[id_]
            brain_sig = b_encoder.encode(brain_sig)

            t0 = time.time()
            mem = lma.recall(np.array(brain_sig))
            duration = time.time()-t0
            durations.append(duration)

            matrix_decoded = m_encoder.decode_ndarray(np.array(mem).reshape((reduce(lambda x, y: x*y, m_encoder.ndarray_shape)*m_encoder.n,)))
            diff = m_encoder.decode_ndarray(np.array(m_encoder.encode_ndarray(matrices[id_])).reshape((reduce(lambda x, y: x*y, m_encoder.ndarray_shape)*m_encoder.n,)))-matrix_decoded
            diffs.append(diff)

    # print meta data
    print "mean: ", np.mean(np.array(diffs).flatten())
    print "max: ", np.max(np.array(diffs).flatten())
    print "std: ", np.std(np.array(diffs).flatten())
    print "duration: ", np.mean(durations)

    # Bar graph plot
    # plotting constants 
    max_ = 0.5
    bars = 40

    # calculate category indices
    categories = []
    for i in range(bars):
        categories.append(i*max_/bars)
    width = categories[1]

    # tally number of instances for each category
    values = np.zeros(bars)
    for diff in np.array(diffs).flatten():
        i = int(diff/(max_+0.0000001)*bars)
        values[i] += 1

    values /= 288.

    # plot bar graph
    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()
    fontP.set_size('small')
    plt.figure(1)
    plt.title("Average error distribution across 10 samples")
    plt.bar(categories, values, width, align='center')
    plt.xlabel("Error bars - width = %0.4f" %(max_/bars))
    plt.ylabel("Proportion")
    plt.show()
    # print number of values that fall into first category (good approximation of how well the memory does)
    print "Values[0]: ", values[0]

def single_cell_hnn_vary():
    valuess = []
    means = []
    stds = []
    maxs = []

    # varying m_encoder width:
    #ws = [16,18,20,22,24,26,28,30,32,34]
    # values = [0.92812500000000009, 0.94791666666666663, 0.96076388888888886, 0.97256944444444449, 0.97881944444444435, 0.98298611111111123, 0.98159722222222223, 0.98229166666666656, 0.97986111111111107, 0.98090277777777779, 0.984375]

    # varying b_encoder buckets:
    #ws = [34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70]
    # values = [0.93090277777777786, 0.9447916666666667, 0.9590277777777777, 0.97118055555555549, 0.97361111111111098, 0.98125000000000007, 0.98263888888888884, 0.98125000000000007, 0.98125000000000007, 0.98263888888888884, 0.98159722222222223, 0.98159722222222223, 0.98055555555555551, 0.9784722222222223, 0.98090277777777779, 0.97673611111111114, 0.97673611111111114, 0.97361111111111098, 0.97569444444444442]
    
    # varying b_encoder width:
    #ws = [12,14,16,18,20,22,24,26,28,30,32]
    # [0.83993055555555562, 0.91145833333333337, 0.95277777777777772, 0.97187499999999993, 0.98229166666666656, 0.98263888888888884, 0.97500000000000009, 0.97152777777777777, 0.95729166666666665, 0.94756944444444435, 0.92083333333333328]
    '''
    Note: 20 has better overall performance than 22

    For 20 -
        nans:  0
        mean:  0.001328125
        max:  0.125
        std:  0.00703216622117
        duration:  0.000690119216839
        Values[0]:  0.982291666667

    For 22- 

        nans:  0
        mean:  0.00155815972222
        max:  0.2375
        std:  0.010365105862
        duration:  0.000707071605656
        Values[0]:  0.982638888889
    '''
    #ws = [30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60]
    # m_w = 46
    #ws = [70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100]
    # b_b = 92
    ws = [30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60]
    for w in ws:
        matrix_shape = (1,)
        m_encoder = scalar_sdr(80,46,-0.25,0.25,matrix_shape)
        b_encoder = scalar_sdr(92,w,0.0,1.0)
        # Calculate number of nodes of hopfield net
        nodes = m_encoder.n*reduce(lambda x, y: x*y, m_encoder.ndarray_shape) + b_encoder.n
        # i) Initialize hopfield network for each matrix value
        # ii) Store data for each brain_id+matrix[brain_id] for given matrix index
        nns = {}
        for index in np.ndindex(matrices["zero"].shape):
            hnn = Hopfield_Neural_Network(nodes)
            for id_ in brain_id:
                data = np.array([])
                matrix = matrices[id_][index].reshape(matrix_shape)
                matrix = m_encoder.encode_ndarray(matrix)
                data = np.append(data,matrix.flatten())
                brain_sig = brain_id[id_]
                brain_sig = b_encoder.encode(brain_sig)
                data = np.append(data,brain_sig)
                hnn.store(data)
            nns[index] = hnn
        # Recall matrix values from brain_id
        # setup some arrays to hold data
        nans_avg = []
        diffs = []
        durations = []
        # run recall fo network 'samples' number of times
        samples = 10
        for i in range(samples):
            nans = 0
            diffs_m = []
            # for each index in the DEP matrix
            for index in np.ndindex(matrices["zero"].shape):

                # for each brain id
                for id_ in brain_id:

                    # generate data
                    data = np.zeros(nodes)-1
                    brain_sig = brain_id[id_]
                    brain_sig = b_encoder.encode(brain_sig)
                    data[-b_encoder.n:] = brain_sig

                    # set iteration number
                    nns[index].setIter(300)

                    # recall matrix value
                    t0 = time.time()
                    mem = nns[index].recall(data,(0,m_encoder.n))
                    duration = time.time()-t0
                    durations.append(duration)

                    # decode matrix value
                    matrix_out = mem[0:m_encoder.n*reduce(lambda x, y: x*y, m_encoder.ndarray_shape)]

                    # calculate error of recalled matrix value
                    matrix_decoded = m_encoder.decode_ndarray(matrix_out)
                    diff_m = m_encoder.decode_ndarray(m_encoder.encode_ndarray(matrices[id_][index].reshape(matrix_shape)).flatten())-matrix_decoded

                    if np.isnan(diff_m):
                        nans += 1
                    diffs_m.append(diff_m)
            nans_avg.append(nans)
            diffs.append(abs(np.array(diffs_m)))
        # print meta data
        print "nans: ", np.max(nans_avg)
        print "mean: ", np.mean(np.array(diffs).flatten())
        print "max: ", np.max(np.array(diffs).flatten())
        print "std: ", np.std(np.array(diffs).flatten())
        print "duration: ", np.mean(durations)
        # Bar graph plot
        # plotting constants 
        max_ = 0.5
        bars = 80
        # calculate category indices
        categories = []
        for i in range(bars):
            categories.append(i*max_/bars)
        
        width = categories[1]
        # tally number of instances for each category
        values = np.zeros(bars)
        for diff in np.array(diffs).flatten():
            i = int(diff/(max_+0.0000001)*bars)
            values[i] += 1
        # divide by number of samples to obtain normalized score
        values /= samples
        values /= 288.0
        # print number of values that fall into first category (good approximation of how well the memory does)
        print "Values[0]: ", values[0]
        
        valuess.append(values[0])
        means.append(np.mean(np.array(diffs).flatten()))
        stds.append(np.std(np.array(diffs).flatten()))
        maxs.append(np.max(np.array(diffs).flatten()))
    return ws, valuess, means, stds, maxs

#single_cell_hnn()
#single_cell_lam()
#full_matrix_hnn()
#full_matrix_lam()


ws, values, means, stds, maxs = single_cell_hnn_vary()
print ws
print values

from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('small')
plt.figure(1)
plt.plot(ws,values,'bo',linestyle='--',linewidth=2.0)
#plt.xlabel("Matrix encoder width")
#plt.xlabel("Id encoder buckets")
#plt.xlabel("Id encoder width")
plt.ylabel("Proportion of error below 0.0125")
plt.show()
# print number of values that fall into first category (good approximation of how well the memory does)
