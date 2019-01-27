import numpy as np
import tensorflow as tf

def augmented_trilinear_lowrank_batch_np(T,t,u,v,w):
    """
    Given a tensor low-rank tensor T and batches of inputs u,v,w,
    return the batch operation T'(u'(i),v'(i),w'(i)), where u'(i) is
    the i-th row of u augmented with a one in its last entry (and
    similarly for v'(i) and w'(i)), and T' is T augmented with the 
    additional parameter vector t.

    Parameters
    ----------
    u,v,w : ndarrays of shape (m,d)
    T : ndarray of shape (r,d)
    t : ndarray of shape (r,1)
    """
    r1 = np.sum(np.dot(T,u.T)*np.dot(T,v.T)*np.dot(T,w.T),axis=0)
    r2 = np.sum(t*(np.dot(T,u.T)*(np.dot(T,v.T+w.T))+np.dot(T,v.T)*np.dot(T,w.T)),axis=0)
    r3 = np.sum(t**2*(np.dot(T,u.T+v.T+w.T)),axis=0)
    return r1+r2+r3+(t**3).sum()
    
def augmented_trilinear_lowrank_batch_tf(T,t,u,v,w):
    """
    Same as augmented_trilinear_lowrank_batch_np but with tensorflow implementation. 
    
    Parameters
    ----------
    u,v,w : tensors of shape (m,d)
    T : tensor of shape (r,d)
    t : tensor of shape (r,1)
    """
    #uT = tf.transpose(u)
    uT = u
    vT = v
    wT = w
    #vT = tf.transpose(v)
    #wT = tf.transpose(w)
    r1 = tf.reduce_sum(tf.matmul(T,uT)*tf.matmul(T,vT)*tf.matmul(T,wT),axis=0)
    r2 = tf.reduce_sum(t*(tf.matmul(T,uT)*(tf.matmul(T,vT+wT))+tf.matmul(T,vT)*tf.matmul(T,wT)),axis=0)
    r3 = tf.reduce_sum(t**2*(tf.matmul(T,uT+vT+wT)),axis=0)
    return r1+r2+r3+tf.reduce_sum(t**3,axis=0) 
    
def trilinear_lowrank_batch_tf(T,u,v,w):
    """
    Given a symmetric low-rank tensor T, compute T(u,v,w) in batch row-wise
    
    Parameters
    ----------
    T : tf tensor of shape (r,d)
    u, v, w : tf tensors of shape (m,)
    """
    if len(T.shape)==2:
        return tf.reduce_sum(tf.matmul(T,tf.transpose(u))*tf.matmul(T,tf.transpose(v))*tf.matmul(T,tf.transpose(w)),axis=0)
    else:
        return tf.reduce_sum(tf.matmul(T[:,:,0],tf.transpose(u))*
                             tf.matmul(T[:,:,1],tf.transpose(v))*tf.matmul(T[:,:,2],tf.transpose(w)),axis=0)

def trilinear_lowrank_batch_np(T,u,v,w):
    """
    Given a symmetric low-rank tensor T, compute T(u,v,w) in batch row-wise
    
    Parameters
    ----------
    T : ndarray of shape (r,d)
    u, v, w : ndarrays of shape (m,)
    """
    return np.sum(np.dot(T,u.T)*np.dot(T,v.T)*np.dot(T,w.T),axis=0)

def trilinear_batch_tf(T,u,v,w):
    """
    Given a full order 3 tensor T and (batch) vectors u,v,w, compute T(u,v,w)
    
    Parameters
    ----------
    T : tf tensor of shape (d,d,d)
    u,v,w : tf tensors of shape (m,d)
    
    Returns
    -------
    out : tf tensor of shape (m,)
    """
    return tf.einsum('jkl,ij,ik,il->i',T,u,v,w)

def trilinear_batch_np(T,u,v,w):
    """
    Given a full order 3 tensor T and (batch) vectors u,v,w, compute T(u,v,w)
    
    Parameters
    ----------
    T : ndarray of shape (d,d,d)
    u,v,w : ndarrays of shape (m,d)
    
    Returns
    -------
    out : ndarray of shape (m,)
    """
    return np.einsum('jkl,ij,ik,il->i',T,u,v,w)

def bilinear_lowrank_batch_tf(T,u,v):
    """
    Given a symmetric or asymmetric low-rank tensor T, compute T(u,v,I) in batch row-wise on u and v
    
    Parameters
    ----------
    T : tf tensor of shape (r,d) if symmetric, of shape (r,d,3) if asymmetric
    u, v : tf tensors of shape (m,d)
    
    Returns
    -------
    out : tf tensor of shape (m,d)
    """
    d = T.shape[1]
    if len(u.shape)==1:
        u = u.reshape((1,d))
    if len(v.shape)==1:
        v = v.reshape((1,d))
    if len(T.shape)==2: # symmetric
        return tf.matmul(tf.matmul(u,tf.transpose(T))*tf.matmul(v,tf.transpose(T)),T)
    else: # asymmetric
        return tf.matmul(tf.matmul(u,tf.transpose(T[:,:,0]))*tf.matmul(v,tf.transpose(T[:,:,1])),T[:,:,2])

def bilinear_lowrank_batch_np(T,u,v):
    """
    Given a symmetric or asymmetric low-rank tensor T, compute T(u,v,I) in batch row-wise on u and v
    
    Parameters
    ----------
    T : ndarray of shape (r,d) if symmetric, (r,d,3) if asymmetric
    u, v : ndarrays of shape (m,d)
    
    Returns
    -------
    out : ndarray of shape (m,d)
    """
    if len(T.shape)==2: # symmetric
        return np.dot((np.dot(u,T.T)*np.dot(v,T.T)),T)
    else:
        return np.dot((np.dot(u,T[:,:,0].T)*np.dot(v,T[:,:,1].T)),T[:,:,2])

def bilinear_lowrank_batch_tf2(T,u,v):
    """
    Given a symmetric low-rank tensor T, compute T(u,v,I) in batch row-wise on u and v
    
    Parameters
    ----------
    T : tf tensor of shape (r,d)
    u, v : tf tensors of shape (m,d)
    
    Returns
    -------
    out : tf tensor of shape (m,d)
    """
    return tf.einsum('ik,lk,im,lm,lj->ij',u,T,v,T,T)

def bilinear_lowrank_batch_np2(T,u,v):
    """
    Given a symmetric low-rank tensor T, compute T(u,v,I) in batch row-wise on u and v
    
    Parameters
    ----------
    T : ndarray of shape (r,d)
    u, v : ndarrays of shape (m,d)
    
    Returns
    -------
    out : ndarray of shape (m,d)
    """
    return np.einsum('ik,lk,im,lm,lj->ij',u,T,v,T,T)

def bilinear_batch_tf(T,u,v):
    """
    Given a full tensor T and batch vectors u and v, compute T(u,v,I) in batch row-wise
    
    Parameters
    ----------
    T : tf tensor of shape (d,d,d)
    u : tf tensor of shape (m,d)
    v : tf tensor of shape (m,d)
    
    Returns
    -------
    out : tf tensor of shape (m,d)
    """
    return tf.einsum('ik,jlk,il->ij',u,T,v)

def bilinear_batch_np(T,u,v):
    """
    Given a full tensor T and batch vectors u and v, compute T(u,v,I) in batch row-wise
    
    Parameters
    ----------
    T : ndarray of shape (d,d,d)
    u : ndarray of shape (m,d)
    v : ndarray of shape (m,d)
    
    Returns
    -------
    out : ndarray of shape (m,d)
    """
    return np.einsum('ik,jlk,il->ij',u,T,v)

if __name__ == "__main__":
    # run tests
    r = 5 # core rank
    d = 10 # embedding dim
    m = 7 # num samples
    
    T = np.random.randn(d,d,d)
    T_tf = tf.convert_to_tensor(T)
    
    t = np.random.randn(r,1)
    t_tf = tf.convert_to_tensor(t)
    
    T_lowrank1 = np.random.randn(r,d,3)
    T_lowrank_tf1 = tf.convert_to_tensor(T_lowrank1)
    
    T_lowrank2 = np.random.randn(r,d)
    T_lowrank_tf2 = tf.convert_to_tensor(T_lowrank2)
    
    T_lowrank3 = np.empty((r,d,3))
    T_lowrank3[:,:,0] = T_lowrank2
    T_lowrank3[:,:,1] = T_lowrank2
    T_lowrank3[:,:,2] = T_lowrank2
    T_lowrank_tf3 = tf.convert_to_tensor(T_lowrank3)
    
    u = np.random.randn(m,d)
    u_tf = tf.convert_to_tensor(u)
    
    v = np.random.randn(m,d)
    v_tf = tf.convert_to_tensor(v)
    
    w = np.random.randn(m,d)
    w_tf = tf.convert_to_tensor(w)
    
    sess = tf.Session()
    with sess.as_default():
        print("Tests to see if numpy and tensorflow implementations match on random inputs:")
        
        #r1 = augmented_trilinear_lowrank_batch_tf(T_lowrank_tf,t_tf,u_tf,v_tf,w_tf).eval()
        #r2 = augmented_trilinear_lowrank_batch_np(T_lowrank,t,u,v,w)
        #print(r1.shape)
        #print("Augmented Trilinear low rank batch test: {}".format(np.allclose(r1,r2)))
        
        #r1 = trilinear_lowrank_batch_tf(T_lowrank_tf,u_tf,v_tf,w_tf).eval()
        #r2 = trilinear_lowrank_batch_np(T_lowrank,u,v,w)
        #print("Trilinear low rank batch test: {}".format(np.allclose(r1,r2)))
        
        #r1 = trilinear_batch_np(T,u,v,w)
        #r2 = trilinear_batch_tf(T_tf,u_tf,v_tf,w_tf).eval()
        #print("Trilinear batch test: {}".format(np.allclose(r1,r2)))
        
        r1 = bilinear_lowrank_batch_np(T_lowrank1,u,v)
        r2 = bilinear_lowrank_batch_tf(T_lowrank_tf1,u_tf,v_tf).eval()
        r3 = bilinear_lowrank_batch_np(T_lowrank2,u,v)
        r4 = bilinear_lowrank_batch_np(T_lowrank3,u,v)
        r5 = bilinear_lowrank_batch_tf(T_lowrank_tf3, u_tf, v_tf).eval()
        print("Bilinear low rank batch test: {}".format(np.allclose(r1,r2)))
        #print("Bilinear low rank batch test: {}".format(np.allclose(r1,r3)))
        print("Bilinear low rank batch test: {}".format(np.allclose(r4,r3)))
        print("Bilinear low rank batch test: {}".format(np.allclose(r4,r5)))
        
        
        #r1 = bilinear_batch_tf(T_tf,u_tf,v_tf).eval()
        #r2 = bilinear_batch_np(T,u,v)
        #print("Bilinear batch test: {}".format(np.allclose(r1,r2)))
