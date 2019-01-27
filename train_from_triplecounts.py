import numpy as np
import tensorflow as tf
import pandas as pd
import tensor_operations as to
import dill
from collections import defaultdict
import pickle
import sys

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def file_chunker(fname, size):
    data = pd.read_csv(fname,sep = " ", header=None,chunksize=size)
    for chunk in data:
        yield chunk.values

def train_with_structure_counts(triple_counts, embeddings, tensor_rank, n_triple_counts, on_disk=True, tensor_init=None, c_init=None,train_embeddings=False, embedding_prior = 0, word_constants=False, word_c_init=None, symmetric=True,V_init=None,save_path=None, pmi=False,epochs=1,batch_size=50000,learning_rate=1e-3,loss_path=None):
    """
    Given a set of trained word embeddings as well as a dictionary of counts of triples of words,
    train a low-rank tensor on these triple counts according to the syntactic rand-walk model.
    
    Parameters
    ----------
    triple_counts : Counter object, keys are tuples of ints (indexes into vocab) OR a filename
    embeddings : ndarray of shape (N,d)
    tensor_rank : positive integer
    n_triple_counts : int, number of triple counts over which to iterate
    on_disk : boolean
        indicates whether the triple counts are in memory, or on disk in a file
    tensor_init : ndarray that gives initialization for tensor (or None if default initialization)
    c_init : real number that gives initialization for C (or None if default initialization)
    train_embeddings : boolean, indication whether the embeddings should also be trained
    embedding_prior : float >= 0, gives the weight for the prior on the word embeddings
    word_constants : boolean, indicates whether the model should learn word-dependent scalar parameters
    word_c_init : ndarray of shape (N,), gives initialization of the word constants
    symmetric : boolean, indicates whether the tensor is symmetric or not
    V_init : ndarray of shape (N,d), the initialization for the trainable word embeddings
    save_path : string, filepath to save model params
    pmi : boolean, indicates whether to use the PMI formulation of the model
    epochs : int >= 1, how many passes over the entire data to perform
    batch_size : int, size of batch when chunking through the triple counts
    learning_rate : positive float, step size for the optimizer,
    loss_path : string, filepath to save loss function values throughout the optimization

    Returns
    -------
    var_list : list of numpy arrays
        the learned model parameters
    """
    # configure tensorflow settings 
    config = tf.ConfigProto(
        allow_soft_placement=True,
    )
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    N,d = embeddings.shape
    
    var_list = []
    
    # initialize tensor variable
    if tensor_init is None:
        if symmetric:
            T_init = .001*np.random.randn(tensor_rank, d).astype("float32")
            T_init[:d, :] = np.eye(d, dtype="float32")
        else:
            T_init = .1*np.random.randn(tensor_rank, d, 3).astype("float32")
    else:
        T_init = tensor_init.astype("float32")
    T = tf.Variable(tf.convert_to_tensor(T_init), name="T")
    var_list.append(T)
    
    # initialize C variable
    if c_init is None:
        C_init = np.ones(1).astype("float32")
    else:
        C_init = c_init.astype("float32")
    C = tf.Variable(tf.convert_to_tensor(C_init*np.ones(1).astype("float32")),name="C")
    var_list.append(C)
   
    # initialize the word constant variables
    if word_constants:
        if word_c_init is None:
            word_C_init = np.ones(N).astype("float32")
        else:
            word_C_init = word_c_init.astype("float32")
        word_C = tf.Variable(tf.convert_to_tensor(word_C_init),name="word_C")
        var_list.append(word_C)
    
    # initialize embeddings variable if we train them
    if train_embeddings:
        if V_init is not None:
            V = tf.Variable(tf.convert_to_tensor(V_init.astype("float32")),name="V")
        else:
            V = tf.Variable(tf.convert_to_tensor(embeddings.astype("float32")),name="V")
        var_list.append(V)
    
    with sess.as_default():
        # create placeholder variables for chunks of data that will be fed in 
        pvals = tf.placeholder(tf.float32, shape=[None], name='pvals') # batch of log probabilities
        weights = tf.placeholder(tf.float32, shape=[None], name='weights') # batch of truncated counts
        indices = tf.placeholder(tf.int64, shape=[None,3], name="indices") # batch of indices of the word triples
        c1 = tf.gather(word_C, tf.gather(indices, 0, name='0_indices',axis=1), name='c1') # word constants for first word of each triple in the batch 
        c2 = tf.gather(word_C, tf.gather(indices, 1, name='1_indices',axis=1), name='c2') # word constants for second word of each triple in batch
        prior_loss = 0
        if train_embeddings: # code for including trainable embedding parameters
            v1 = tf.gather(V, tf.gather(indices, 0, name='0_indices',axis=1), name='v1')
            v2 = tf.gather(V, tf.gather(indices, 1, name='1_indices',axis=1), name='v2')
            v3 = tf.gather(V, tf.gather(indices, 2, name='2_indices',axis=1), name='v3')
            unique_inds = tf.placeholder(tf.int64, shape=[None], name="unique_inds")
            unique_prior = tf.placeholder(tf.float32, shape=[None, d], name="unique_prior")
            unique_vects = tf.gather(V, unique_inds, name="unique_vects")
            prior_loss = tf.reduce_mean(tf.reduce_sum(tf.square(unique_vects-unique_prior),axis=1))
            
        else:
            v1 = tf.placeholder(tf.float32, shape=[None, d], name='v1') # batch of first word embeddings
            v2 = tf.placeholder(tf.float32, shape=[None, d], name='v2') # batch of second word embeddings
            v3 = tf.placeholder(tf.float32, shape=[None, d], name='v3') # batch of third word embeddings
            
        if pmi: # code for computing the batch loss for the PMI formulation
            uni_1 = tf.placeholder(tf.float32, shape=[None], name="uni1") # batch of log uni counts
            uni_2 = tf.placeholder(tf.float32, shape=[None], name="uni2")
            uni_3 = tf.placeholder(tf.float32, shape=[None], name="uni3")
            co_1 = tf.placeholder(tf.float32, shape=[None], name="co1")
            co_2 = tf.placeholder(tf.float32, shape=[None], name="co2")
            wordpair = tf.placeholder(tf.float32, shape=[None], name="wordpair")
                
            pred = to.trilinear_lowrank_batch_tf(T,v1,v2,v3)
            pmi_vals = pvals + uni_1 + uni_2 + uni_3 - co_1 - co_2 - wordpair
            errors = tf.squared_difference(pmi_vals, pred)
            loss = tf.reduce_mean(tf.multiply(weights,errors))
            tensor_loss = loss
            
        else: # code for computing the batch loss for the log joint probability formulation
            sum_vects = v1 + v2 + v3 + to.bilinear_lowrank_batch_tf(T,v1,v2) # v1 and v2 are the structure words
            pred = tf.reduce_sum(tf.square(sum_vects), axis=1) - C
            if word_constants:
                pred += -c2

            errors = tf.squared_difference(pvals, pred)
            tensor_loss = tf.reduce_mean(tf.multiply(weights,errors))
            loss = (1-embedding_prior)*tensor_loss + embedding_prior*prior_loss 

        # create the tensorflow optimizer
        global_step = tf.Variable(0,name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # traditionally have chosen learning_rate=1e-3
        #optimizer = tf.train.AdagradOptimizer(1e-3)
        train_op = optimizer.minimize(loss, global_step)
            
        sess.run(tf.global_variables_initializer())
            
        chunk_size = batch_size
        save_every = int((n_triple_counts/chunk_size)/50) # save 50 times per epoch
        print_every = int((n_triple_counts/chunk_size)/1000) # print status 1000 times per epoch
        loss_every = int((n_triple_counts/chunk_size)/1000) # save loss value about 1000 times per epoch
            
         
        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch))
             
            if on_disk: # code for iterating through the triple counts if stored on disk rather than in memory
                chunk_iterator = file_chunker(triple_counts, chunk_size)
            else:  # code for iterating through the triple counts if stored in memory
                triple_counts = list(triple_counts.items())
                np.random.shuffle(triple_counts)
                chunk_iterator = chunker(triple_counts, chunk_size)
            for j,batch in enumerate(chunk_iterator):
                if on_disk:
                    v1_inds = batch[:,0]
                    v2_inds = batch[:,1]
                    v3_inds = batch[:,2]
                    weights_batch = batch[:,3]
                    pvals_batch = np.log(weights_batch)
                    if pmi:
                        uni_batch_1 = np.log([[w] for w in v1_inds])
                        uni_batch_2 = np.log([uni_counts[w] for w in v2_inds])
                        uni_batch_3 = np.log([uni_counts[w] for w in v3_inds])
                        co_batch_1 = np.log([co_counts[tuple(r)] for r in np.sort(batch[:,1:],axis=1)])
                        co_batch_2 = np.log([co_counts[tuple(r)] for r in np.sort(batch[:,[0,2]],axis=1)])
                        wordpair_batch = np.log([wordpairs[tuple(r)] for r in batch[:,:2]])
                else:
                    v1_inds = [b[0][0] for b in batch]
                    v2_inds = [b[0][1] for b in batch]
                    v3_inds = [b[0][2] for b in batch]
                    weights_batch = np.array([b[1] for b in batch])
                    pvals_batch = np.log(weights_batch)
                v1_batch = embeddings[v1_inds]
                v2_batch = embeddings[v2_inds]
                v3_batch = embeddings[v3_inds]
                weights_batch = np.clip(weights_batch,0,100)
                    
                feed_dict = {pvals : pvals_batch, weights : weights_batch, indices : batch[:,:3]}
                if pmi:
                    feed_dict.update({uni_1 : uni_batch_1,
                                      uni_2 : uni_batch_2,
                                      uni_3 : uni_batch_3,
                                      co_1 : co_batch_1,
                                      co_2 : co_batch_2,
                                      wordpair : wordpair_batch})
                if train_embeddings:
                    unique_inds_batch = list(set(batch[:,:3].flatten()))
                    unique_prior_batch = embeddings[unique_inds_batch]
                    feed_dict.update({unique_inds : unique_inds_batch,
                                      unique_prior : unique_prior_batch,
                                      indices : batch[:,:3]})
                else:
                    feed_dict.update({v1 : v1_batch,
                                      v2 : v2_batch,
                                      v3 : v3_batch})
                _, tensor_loss_val, step = sess.run([train_op, tensor_loss, global_step],
                                            feed_dict=feed_dict)
                if loss_path is not None:
                    if j%loss_every == 0:
                        with open(loss_path,"a+") as loss_file:
                            loss_file.write(str(tensor_loss_val)+"\n")
                   
                    
                if j%print_every == 0:
                    print("Epoch Completion: {0:.3f}%, Tensor Loss: {1:.3f}, C={2:.2f}, word_c_avg={3:.2f}".format(j*chunk_size*100/n_triple_counts, tensor_loss_val, C.eval(sess)[0],word_C.eval(sess).mean()),end="\r",flush=True)
                if j%save_every == 0:
                    np.savez(save_path,*[v.eval(sess) for v in var_list])
                    
    return [v.eval(sess) for v in var_list]



def main():

    # parameter settings, can be changed by passing parameters to file
    PMI=False
    symmetric=False
    word_constants=True
    on_disk=True
    train_embeddings=False
    triple_counts = "/usr/xtmp/abef/triple_counts/triple_counts_an_shuf.txt"
    n_triple_counts = 233135511 # dep_vo
    #n_triple_counts = 428452249 # dep_an
    tensor_rank = 1000
    embedding_prior=0
    init_file = None
    V=None
    
    #path_to_vects = "../datasets/rw_vectors.txt"
    #save_path = "/usr/xtmp/abef/learned_params_dep_an_rw.npz"
    #loss_path = "/usr/xtmp/abef/loss_vals_dep_an_rw.txt"
    epochs=1
    batch_size=50000
    learning_rate=1e-3
    """
    python train_from_triplecounts.py --counts_file=/usr/xtmp/abef/triple_counts/triple_counts_vo_shuf.txt --vector_file=../datasets/glove_vectors_rescaled.txt --init_file=/usr/xtmp/abef/learned_params_dep_vo_glove_1000.npz --save_path=/usr/xtmp/abef/learned_params_dep_vo_glove_1000.npz --loss_path=/usr/xtmp/abef/loss_vals_dep_vo_glove_1000.txt --epochs=2 --learning_rate=4e-4 --tensor_rank=1000
    """
    """
    python train_from_triplecounts.py --counts_file=/usr/xtmp/abef/triple_counts/triple_counts_vo_shuf.txt --vector_file=../datasets/cbow_vectors_rescaled.txt --init_file=/usr/xtmp/abef/learned_params_dep_vo_cbow_1000.npz --save_path=/usr/xtmp/abef/learned_params_dep_vo_cbow_1000.npz --loss_path=/usr/xtmp/abef/loss_vals_dep_vo_cbow_1000.txt --epochs=2 --learning_rate=4e-4 --tensor_rank=1000
    """
    """
    python train_from_triplecounts.py --counts_file=/usr/xtmp/abef/triple_counts/triple_counts_vo_shuf.txt --vector_file=../datasets/rw_vectors.txt --init_file=/usr/xtmp/abef/learned_params_dep_vo_rw_1000.npz --save_path=/usr/xtmp/abef/learned_params_dep_vo_rw_1000.npz --loss_path=/usr/xtmp/abef/loss_vals_dep_vo_rw_1000.txt --epochs=2 --learning_rate=4e-4 --tensor_rank=1000
    """
    for arg in sys.argv:
        if arg.startswith('--counts_file='): # path to file containing the triple counts
            triple_counts = arg.split('--counts_file=')[1]
        if arg.startswith('--vector_file='): # path to file containing the word embeddings we're using
            path_to_vects = arg.split('--vector_file=')[1]
        if arg.startswith('--init_file='): # path to numpy archive containing initializations of model parameters
            init_file = arg.split('--init_file=')[1]
        if arg.startswith('--save_path='): # where you want to save model parameters
            save_path = arg.split('--save_path=')[1]
        if arg.startswith('--loss_path='): # where you want to save model parameters
            loss_path = arg.split('--loss_path=')[1]
        if arg.startswith('--epochs='):
            epochs = int(arg.split('--epochs=')[1])
        if arg.startswith('--batch_size='):
            batch_size = int(arg.split('--batch_size=')[1])
        if arg.startswith('--learning_rate='):
            learning_rate = float(arg.split('--learning_rate=')[1])
        if arg.startswith('--train_embeddings='):
            train_embeddings=bool(arg.split('--train_embeddings=')[1])
        if arg.startswith('--init_vectors='):
            init_vectors=arg.split('--init_vectors=')[1]
        if arg.startswith('--embedding_prior='):
            embedding_prior=float(arg.split('--embedding_prior=')[1])
        if arg.startswith('--tensor_rank='):
            tensor_rank=int(arg.split('--tensor_rank=')[1])
            
    vectors = np.loadtxt(path_to_vects)
    if init_file is None:
        tensor_init=None
        C_init=None
        word_C_init=None
    else:
        init_vals = np.load(init_file)
        tensor_init=init_vals["arr_0"]
        C_init=init_vals["arr_1"]
        word_C_init=init_vals["arr_2"]
    if train_embeddings:
        V = np.loadtxt(init_vectors)
        #V = init_vals["arr_3"]
        
    vals = train_with_structure_counts(triple_counts, vectors, tensor_rank, n_triple_counts, on_disk=on_disk,
                                       tensor_init=tensor_init, c_init=C_init, word_c_init=word_C_init, 
                                       train_embeddings=train_embeddings,word_constants=word_constants,
                                       symmetric=symmetric,V_init=V,save_path=save_path,pmi=PMI,epochs=epochs,
                                       batch_size=batch_size,learning_rate=learning_rate,loss_path=loss_path,
                                       embedding_prior=embedding_prior
                                      )
    np.savez(save_path,*vals)



if __name__ == '__main__':
    main()
