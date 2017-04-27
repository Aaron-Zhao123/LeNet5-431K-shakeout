from __future__ import print_function

# Import MNIST data
import sys
import getopt
import input_data
import os.path
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle

class Usage(Exception):
    def __init__ (self,msg):
        self.msg = msg

# Parameters
# training_epochs = 500
training_epochs = 700
batch_size = 64
display_step = 1

# Network Parameters
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

n_hidden_1 = 300# 1st layer number of features
n_hidden_2 = 100# 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)



'''
pruning Parameters
'''
# sets the threshold
prune_threshold_cov = 0.08
prune_threshold_fc = 1
# Frequency in terms of number of training iterations
prune_freq = 100
ENABLE_PRUNING = 0

def compute_file_name(pcov, pfc):
    name = ''
    name += 'cov' + str(int(pcov[0] * 10))
    name += 'cov' + str(int(pcov[1] * 10))
    name += 'fc' + str(int(pfc[0] * 10))
    name += 'fc' + str(int(pfc[1] * 10))
    return name

# Store layers weight & bias
def initialize_variables(model_number, no_pruning):
    if (no_pruning == False):
        print(model_number)
        with open(model_number,'rb') as f:
            wc1, wc2, wd1, out, bc1, bc2, bd1, bout = pickle.load(f)
        weights = {
            # 5x5 conv, 1 input, 32 outputs
            'cov1': tf.Variable(wc1),
            # 5x5 conv, 32 inputs, 64 outputs
            'cov2': tf.Variable(wc2),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'fc1': tf.Variable(wd1),
            # 1024 inputs, 10 outputs (class prediction)
            'fc2': tf.Variable(out)
        }

        biases = {
            'cov1': tf.Variable(bc1),
            'cov2': tf.Variable(bc2),
            'fc1': tf.Variable(bd1),
            'fc2': tf.Variable(bout)
        }
    else:
        weights = {
            'cov1': tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 20], stddev=0.1)),
            'cov2': tf.Variable(tf.truncated_normal([5, 5, 20, 50], stddev=0.1)),
            'fc1': tf.Variable(tf.truncated_normal([ 4 * 4 * 50, 500])),
            'fc2': tf.Variable(tf.truncated_normal([500, 10]))
        }
        biases = {
            'cov1': tf.Variable(tf.random_normal([20])),
            'cov2': tf.Variable(tf.random_normal([50])),
            'fc1': tf.Variable(tf.random_normal([500])),
            'fc2': tf.Variable(tf.random_normal([10]))
        }
    return (weights, biases)
#store the masks
# weights_mask = {
#     'cov1': tf.Variable(tf.ones([5, 5, NUM_CHANNELS, 32]), trainable = False),
#     'cov2': tf.Variable(tf.ones([5, 5, 32, 64]), trainable = False),
#     'fc1': tf.Variable(tf.ones([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512]), trainable = False),
#     'fc2': tf.Variable(tf.ones([512, NUM_LABELS]), trainable = False)
# }
    # else:
    #     with open('assets.pkl','rb') as f:
    #         (weights, biases, weights_mask) = pickle.load(f)

# weights_mask = {
#     'cov1': np.ones([5, 5, NUM_CHANNELS, 32]),
#     'cov2': np.ones([5, 5, 32, 64]),
#     'fc1': np.ones([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512]),
#     'fc2': np.ones([512, NUM_LABELS])
# }
# Create model
def conv_network(x, weights, biases, keep_prob, c = 10.):
    conv = tf.nn.conv2d(x,
                        weights['cov1'],
                        strides = [1,1,1,1],
                        padding = 'VALID')
    relu = tf.nn.relu(tf.nn.bias_add(conv, biases['cov1']))
    pool = tf.nn.max_pool(
            relu,
            ksize = [1 ,2 ,2 ,1],
            strides = [1, 2, 2, 1],
            padding = 'VALID')

    conv = tf.nn.conv2d(pool,
                        weights['cov2'],
                        strides = [1,1,1,1],
                        padding = 'VALID')
    relu = tf.nn.relu(tf.nn.bias_add(conv, biases['cov2']))
    pool = tf.nn.max_pool(
            relu,
            ksize = [1 ,2 ,2 ,1],
            strides = [1, 2, 2, 1],
            padding = 'VALID')
    '''get pool shape'''
    # pool_shape = pool.get_shape().as_list()
    # reshape = tf.reshape(
    #     pool,
    #     [-1, pool_shape[1]*pool_shape[2]*pool_shape[3]])
    reshape = tf.reshape(pool, [-1, 4*4*50])
    # h1_fc = tf.nn.relu(tf.matmul(reshape, weights['fc1']) + biases['fc1'])
    # h1_fc = tf.tanh(tf.matmul(reshape, weights['fc1']) + biases['fc1'])
    # h1_fc_drop  = tf.nn.dropout(h1_fc, keep_prob)
    h1_fc_shake, rj= shakeout(reshape, weights['fc1'], c, keep_prob)
    h1_fc = tf.nn.relu(tf.matmul(h1_fc_shake, weights['fc1']) + biases['fc1'])
    output = tf.matmul(h1_fc, weights['fc2']) + biases['fc2']
    return output, reshape, h1_fc_shake, rj

def calculate_non_zero_weights(weight):
    count = (weight != 0).sum()
    size = len(weight.flatten())
    return (count,size)

'''
Prune weights, weights that has absolute value lower than the
threshold is set to 0
'''

def prune_weights(pruning_cov, pruning_cov2, pruning_fc, pruning_fc2, weights, weight_mask, biases, biases_mask, parent_dir, f_name):
    keys_cov = ['cov1','cov2','fc1','fc2']
    keys_fc = ['fc1', 'fc2']
    next_threshold = {}
    b_threshold = {}
    for key in keys_cov:
        if (key == 'cov1'):
            weight = weights[key].eval()
            biase = biases[key].eval()
            next_threshold[key] = np.percentile(np.abs(weight),pruning_cov)
            weight_mask[key] = np.abs(weight) > next_threshold[key]
            b_threshold[key] = np.percentile(np.abs(biase),pruning_cov)
            biases_mask[key] = np.abs(biase) > b_threshold[key]
        if (key == "cov2"):
            weight = weights[key].eval()
            biase = biases[key].eval()
            next_threshold[key] = np.percentile(np.abs(weight),pruning_cov2)
            weight_mask[key] = np.abs(weight) > next_threshold[key]
            b_threshold[key] = np.percentile(np.abs(biase),pruning_cov2)
            biases_mask[key] = np.abs(biase) > b_threshold[key]

    for key in keys_fc:
        if (key == "fc1"):
            weight = weights[key].eval()
            biase = biases[key].eval()
            next_threshold[key] = np.percentile(np.abs(weight),pruning_fc)
            weight_mask[key] = np.abs(weight) > next_threshold[key]
            b_threshold[key] = np.percentile(np.abs(biase),pruning_fc)
            biases_mask[key] = np.abs(biase) > b_threshold[key]
        if (key == "fc2"):
            weight = weights[key].eval()
            biase = biases[key].eval()
            next_threshold[key] = np.percentile(np.abs(weight),pruning_fc2)
            weight_mask[key] = np.abs(weight) > next_threshold[key]
            b_threshold[key] = np.percentile(np.abs(biase),pruning_fc2)
            biases_mask[key] = np.abs(biase) > b_threshold[key]
    pf_name = compute_file_name([pruning_cov, pruning_cov2], [pruning_fc, pruning_fc2])
    print(pf_name)
    # sys.exit()
    file_name = parent_dir + 'weights/' + pf_name + '.pkl'

    with open(file_name, 'wb') as f:
        pickle.dump((
            weights['cov1'].eval(),
            weights['cov2'].eval(),
            weights['fc1'].eval(),
            weights['fc2'].eval(),
            biases['cov1'].eval(),
            biases['cov2'].eval(),
            biases['fc1'].eval(),
            biases['fc2'].eval()),f)

    with open(parent_dir + 'masks/' + pf_name + '.pkl', 'wb') as f:
        pickle.dump((weight_mask, biases_mask), f)

# def quantize_a_value(val):
#
#
'''
mask gradients, for weights that are pruned, stop its backprop
'''
def mask_gradients(weights, grads_and_names, weight_masks, biases, biases_mask):
    new_grads = []
    keys = ['cov1','cov2','fc1','fc2']
    for grad, var_name in grads_and_names:
        # flag set if found a match
        flag = 0
        index = 0
        for key in keys:
            if (weights[key]== var_name):
                # print(key, weights[key].name, var_name)
                mask = weight_masks[key]
                new_grads.append((tf.multiply(tf.constant(mask, dtype = tf.float32),grad),var_name))
                flag = 1
        # if flag is not set
        if (flag == 0):
            new_grads.append((grad,var_name))
    return new_grads

'''
plot weights and store the fig
'''
def plot_weights(weights,pruning_info):
        keys = ['cov1','cov2','fc1','fc2']
        fig, axrr = plt.subplots( 2, 2)  # create figure &  axis
        fig_pos = [(0,0), (0,1), (1,0), (1,1)]
        index = 0
        for key in keys:
            weight = weights[key].eval().flatten()
            # print (weight)
            size_weight = len(weight)
            weight = weight.reshape(-1,size_weight)[:,0:size_weight]
            x_pos, y_pos = fig_pos[index]
            #take out zeros
            weight = weight[weight != 0]
            # print (weight)
            hist,bins = np.histogram(weight, bins=100)
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            axrr[x_pos, y_pos].bar(center, hist, align = 'center', width = width)
            axrr[x_pos, y_pos].set_title(key)
            index = index + 1
        fig.savefig('fig_v3/weights'+pruning_info)
        plt.close(fig)

def shakeout(x, weights, c = 10., keep_rate = 0.5):
    # keep rate = 1 -tau
    # random generation of t between (0,1)
    prob = tf.random_uniform(tf.shape(x), dtype=tf.float32, minval = 0., maxval = 1.)
    tau = 1 - keep_rate
    rj = tf.cast(prob > tau, tf.float32)

    alpha = 1 / (1-tau)
    beta = c * tau

    wj = tf.reduce_sum(weights, 1)

    pos = alpha + alpha * beta / wj
    neg = - c / wj
    factor = pos * (rj)+ neg * (1 - rj)
    u = factor * x
    return u, rj


def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -1, 1)

def find_reg(percent):
    pass



'''
Define a training strategy
'''
def main(argv = None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            # opts, args = getopt.getopt(argv[1:],'hp:tc1:tc2:tfc1:tfc2:')
            opts = argv
            threshold = {
                'cov1' : 0.08,
                'cov2' : 0.08,
                'fc1' : 1,
                'fc2' : 1
            }
            PRUNE_ONLY = False
            TRAIN = True
            learning_rate = 1e-4
            dropout = 1
            shakeout_const = 10.
            weight_file_name = 'tmp'
            no_pruning = False
            for item in opts:
                print (item)
                opt = item[0]
                val = item[1]
                if (opt == '-pcov'):
                    pruning_cov = val
                if (opt == '-pcov2'):
                    pruning_cov2 = val
                if (opt == '-pfc'):
                    pruning_fc = val
                if (opt == '-pfc2'):
                    pruning_fc2 = val
                if (opt == '-m'):
                    model_number = val
                if (opt == '-train'):
                    TRAIN = val
                if (opt == '-lr'):
                    learning_rate = val
                if (opt == '-norm1'):
                    lambda_1 = val
                if (opt == '-norm2'):
                    lambda_2 = val
                if (opt == '-dropout'):
                    dropout = val
                if (opt == '-weight_file_name'):
                    weight_file_name = val
                if (opt == '-shakeout_c'):
                    shakeout_const = val
                if (opt == '-parent_dir'):
                    parent_dir = val
                if (opt == '-PRUNE'):
                    PRUNE_ONLY = val
                if (opt == '-TRAIN'):
                    TRAIN = val
                if (opt == '-nopruning'):
                    no_pruning = val
            print('pruning percentage for cov and fc are {},{}'.format(pruning_cov, pruning_fc))
            print('Train values:',TRAIN)
            print('dropout value:',dropout)
        except getopt.error, msg:
            raise Usage(msg)

        # obtain all weight masks
        pruning_cov = int(pruning_cov)
        pruning_cov2 = int(pruning_cov2)
        pruning_fc = pruning_fc
        pruning_fc2 = int(pruning_fc2)

        if (TRAIN == True):
            if (no_pruning):
                weights_mask = {
                    'cov1': np.ones([5, 5, NUM_CHANNELS, 20]),
                    'cov2': np.ones([5, 5, 20, 50]),
                    'fc1': np.ones([4 * 4 * 50, 500]),
                    'fc2': np.ones([500, NUM_LABELS])
                }
                biases_mask = {
                    'cov1': np.ones([20]),
                    'cov2': np.ones([50]),
                    'fc1': np.ones([500]),
                    'fc2': np.ones([10])
                }
            else:
                # weights_mask = {
                #     'cov1': np.ones([5, 5, NUM_CHANNELS, 20]),
                #     'cov2': np.ones([5, 5, 20, 50]),
                #     'fc1': np.ones([4 * 4 * 50, 500]),
                #     'fc2': np.ones([500, NUM_LABELS])
                # }
                # biases_mask = {
                #     'cov1': np.ones([20]),
                #     'cov2': np.ones([50]),
                #     'fc1': np.ones([500]),
                #     'fc2': np.ones([10])
                # }
                with open(parent_dir + 'masks/' + weight_file_name,'rb') as f:
                    (weights_mask,biases_mask) = pickle.load(f)
        else:
            weights_mask = {
                'cov1': np.ones([5, 5, NUM_CHANNELS, 20]),
                'cov2': np.ones([5, 5, 20, 50]),
                'fc1': np.ones([4 * 4 * 50, 500]),
                'fc2': np.ones([500, NUM_LABELS])
            }
            biases_mask = {
                'cov1': np.ones([20]),
                'cov2': np.ones([50]),
                'fc1': np.ones([500]),
                'fc2': np.ones([10])
            }

        mnist = input_data.read_data_sets("MNIST.data/", one_hot = True)
        # tf Graph input
        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_classes])

        keep_prob = tf.placeholder(tf.float32)
        keys = ['cov1','cov2','fc1','fc2']

        x_image = tf.reshape(x,[-1,28,28,1])
        (weights, biases) = initialize_variables(parent_dir+ 'weights/' +weight_file_name, no_pruning)
        # Construct model
        pred, hidden_before, hidden_after, rj= conv_network(x_image, weights, biases, keep_prob, shakeout_const)
        # lambda_1 = 0.00001
        # lambda_2 = 0.0005

        # Define loss and optimizer
        trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))

        correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        # I need to fetch this value
        variables = [weights['cov1'], weights['cov2'], weights['fc1'], weights['fc2'],
                    biases['cov1'], biases['cov2'], biases['fc1'], biases['fc2']]
        org_grads = trainer.compute_gradients(cost, var_list = variables, gate_gradients = trainer.GATE_OP)

        org_grads = [(ClipIfNotNone(grad), var) for grad, var in org_grads]
        new_grads = mask_gradients(weights, org_grads, weights_mask, biases, biases_mask)

        train_step = trainer.apply_gradients(new_grads)
        # train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)


        init = tf.initialize_all_variables()
        # Launch the graph
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(init)
            # restore model if exists
            keys = ['cov1','cov2','fc1','fc2']

            # retain the masks on the weights
            for key in keys:
                sess.run(weights[key].assign(weights[key].eval()*weights_mask[key]))
            prune_info(weights,1)
            # Training cycle
            training_cnt = 0
            pruning_cnt = 0
            train_accuracy = 0
            accuracy_list = np.zeros(20)
            accuracy_mean = 0
            c = 0
            train_accuracy = 0
            iter_cnt = 0

            if (TRAIN == True):
                print('Training starts ...')
                for epoch in range(training_epochs):
                    avg_cost = 0.
                    total_batch = int(mnist.train.num_examples/batch_size)
                    # Loop over all batches
                    for i in range(total_batch):
                        iter_cnt = iter_cnt + 1
                        # execute a pruning
                        batch_x, batch_y = mnist.train.next_batch(batch_size)
                        [_, cost_val, hb, ha, tdata] = sess.run([train_step, cost, hidden_before, hidden_after, rj], feed_dict = {
                                x: batch_x,
                                y: batch_y,
                                keep_prob: dropout})
                        # print(80*'-')
                        # print('check shakeout')
                        # print('before shakeout: {}'.format(hb))
                        # print('after shakeout: {}'.format(ha))
                        # print('test : {}'.format(tdata))

                        training_cnt = training_cnt + 1
                        if (training_cnt % 100 == 0):
                            [c, train_accuracy] = sess.run([cost, accuracy], feed_dict = {
                                x: batch_x,
                                y: batch_y,
                                keep_prob: 1.})
                            accuracy_list = np.concatenate((np.array([train_accuracy]),accuracy_list[0:19]))
                            accuracy_mean = np.mean(accuracy_list)
                            print(accuracy_mean)

                        if (training_cnt % 1000 == 0):
                            weights_info(training_cnt, c, train_accuracy, accuracy_mean)
                            mask_info(weights_mask)

                        # if (accuracy_mean > 0.99 or epoch > 300):
                        if (accuracy_mean > 0.9 or epoch > 500):
                            accuracy_list = np.zeros(20)
                            accuracy_mean = 0
                            print('Training ends')
                            test_accuracy = accuracy.eval({
                                    x: mnist.test.images[:],
                                    y: mnist.test.labels[:],
                                    keep_prob: 1.})
                            print('test accuracy is {}'.format(test_accuracy))
                            # if (epoch > 300 or test_accuracy > 0.9936):
                            if (epoch > 500 or test_accuracy > 0.99):
                                print('stop training...')
                                file_name = parent_dir + 'weights/' + weight_file_name
                                print(file_name)
                                with open(file_name, 'wb') as f:
                                    pickle.dump((
                                        weights['cov1'].eval(),
                                        weights['cov2'].eval(),
                                        weights['fc1'].eval(),
                                        weights['fc2'].eval(),
                                        biases['cov1'].eval(),
                                        biases['cov2'].eval(),
                                        biases['fc1'].eval(),
                                        biases['fc2'].eval()),f)

                                # mask_info(weights_mask)
                                # Test model
                                return (test_accuracy, iter_cnt)
                            else:
                                pass
                        # Compute average loss
                        avg_cost += c / total_batch
                    # Display logs per epoch step
                    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
                print("Optimization Finished!")
                file_name = parent_dir + 'weights/' + weight_file_name
                print(file_name)
                with open(file_name, 'wb') as f:
                    pickle.dump((
                        weights['cov1'].eval(),
                        weights['cov2'].eval(),
                        weights['fc1'].eval(),
                        weights['fc2'].eval(),
                        biases['cov1'].eval(),
                        biases['cov2'].eval(),
                        biases['fc1'].eval(),
                        biases['fc2'].eval()),f)

                # mask_info(weights_mask)
                # Test model
                return (test_accuracy, iter_cnt)
            if (PRUNE_ONLY == True):
                print('hi in pruning')
                prune_weights(  pruning_cov,
                                pruning_cov2,
                                pruning_fc,
                                pruning_fc2,
                                weights,
                                weights_mask,
                                biases,
                                biases_mask,
                                parent_dir,
                                weight_file_name)
                mask_info(weights_mask)
            # Calculate accuracy
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            test_accuracy = accuracy.eval({x: mnist.test.images, y: mnist.test.labels, keep_prob : 1.0})
            print("Accuracy:", test_accuracy)
            return (test_accuracy, iter_cnt)
    except Usage, err:
        print >> sys.stderr, err.msg
        print >> sys.stderr, "for help use --help"
        return 2

def weights_info(iter,  c, train_accuracy, acc_mean):
    print('This is the {}th iteration, cost is {}, accuracy is {}, accuracy mean is {}'.format(
        iter,
        c,
        train_accuracy,
        acc_mean
    ))

def prune_info(weights, counting):
    if (counting == 0):
        (non_zeros, total) = calculate_non_zero_weights(weights['cov1'].eval())
        print('cov1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['cov2'].eval())
        print('cov2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc1'].eval())
        print('fc1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc2'].eval())
        print('fc2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
    if (counting == 1):
        (non_zeros, total) = calculate_non_zero_weights(weights['fc1'].eval())
        print('take fc1 as example, {} nonzeros, in total {} weights'.format(non_zeros, total))

def mask_info(weights):
    (non_zeros, total) = calculate_non_zero_weights(weights['cov1'])
    print('cov1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
    (non_zeros, total) = calculate_non_zero_weights(weights['cov2'])
    print('cov2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
    (non_zeros, total) = calculate_non_zero_weights(weights['fc1'])
    print('fc1 has prunned {} percent of its weights'.format((total-non_zeros)*100/float(total)))
    (non_zeros, total) = calculate_non_zero_weights(weights['fc2'])
    print('fc2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))

def write_numpy_to_file(data, file_name):
    # Write the array to disk
    with file(file_name, 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(data.shape))

        for data_slice in data:
            for data_slice_two in data_slice:
                np.savetxt(outfile, data_slice_two)
                outfile.write('# New slice\n')


if __name__ == '__main__':
    sys.exit(main())
