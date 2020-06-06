import tensorflow as tf
import numpy as np

tf_type = tf.float32

class Kernel_Matern1:
    def __init__(self, jitter=0):
        self.jitter = tf.constant(jitter, dtype=tf_type)
        
    def matrix(self, X, amp, ls):
        K = self.cross(X,X,amp, ls)
        K = K + self.jitter * tf.eye(tf.shape(X)[0], dtype=tf_type)
        return K
    
    def cross(self, X1, X2, amp, ls):
        
        X1 = (X1 - tf.reduce_mean(X1, axis=0)) / tf.math.reduce_std(X1, axis=0)
        X2 = (X2 - tf.reduce_mean(X2, axis=0)) / tf.math.reduce_std(X2, axis=0)
        
        norm1 = tf.reshape(tf.reduce_sum(X1**2, 1), [-1, 1])
        norm2 = tf.reshape(tf.reduce_sum(X2**2, 1), [1, -1])
    
        d = tf.math.sqrt(tf.abs(norm1 - 2.0 * tf.matmul(X1, tf.transpose(X2)) + norm2))
        
        sigf = amp
        sigl = ls
        
        K = sigf**2 * tf.exp(-d/sigl)

        return K

    
class Kernel_Matern3:
    def __init__(self, jitter=0):
        self.jitter = tf.constant(jitter, dtype=tf_type)
        
    def matrix(self, X, amp, ls):
        K = self.cross(X,X,amp, ls)
        K = K + self.jitter * tf.eye(tf.shape(X)[0], dtype=tf_type)
        return K
    
    def cross(self, X1, X2, amp, ls):
        
        X1 = (X1 - tf.reduce_mean(X1, axis=0)) / tf.math.reduce_std(X1, axis=0)
        X2 = (X2 - tf.reduce_mean(X2, axis=0)) / tf.math.reduce_std(X2, axis=0)
        
        norm1 = tf.reshape(tf.reduce_sum(X1**2, 1), [-1, 1])
        norm2 = tf.reshape(tf.reduce_sum(X2**2, 1), [1, -1])
    
        d = tf.math.sqrt(tf.abs(norm1 - 2.0 * tf.matmul(X1, tf.transpose(X2)) + norm2))
        
        sigf = amp
        sigl = ls
        
        K = sigf**2 * (1 + tf.math.sqrt(3.0) * d / sigl) * tf.exp(-tf.math.sqrt(3.0) * d / sigl)

        return K
    
class Kernel_Matern5:
    def __init__(self, jitter=0):
        self.jitter = tf.constant(jitter, dtype=tf_type)
        
    def matrix(self, X, amp, ls):
        K = self.cross(X,X,amp, ls)
        K = K + self.jitter * tf.eye(tf.shape(X)[0], dtype=tf_type)
        return K
    
    def cross(self, X1, X2, amp, ls):
        
        X1 = (X1 - tf.reduce_mean(X1, axis=0)) / tf.math.reduce_std(X1, axis=0)
        X2 = (X2 - tf.reduce_mean(X2, axis=0)) / tf.math.reduce_std(X2, axis=0)
        
        norm1 = tf.reshape(tf.reduce_sum(X1**2, 1), [-1, 1])
        norm2 = tf.reshape(tf.reduce_sum(X2**2, 1), [1, -1])
    
        d = tf.math.sqrt(tf.abs(norm1 - 2.0 * tf.matmul(X1, tf.transpose(X2)) + norm2))
        
        sigf = amp
        sigl = ls

        K = sigf**2 * (1 + tf.math.sqrt(5.0) * d / sigl + 5 * tf.square(d) / (3 * tf.square(sigl))) * tf.exp(-tf.math.sqrt(5.0) * d / sigl)

        return K

class Kernel_ARD:
    def __init__(self, jitter=0):
        self.jitter = tf.constant(jitter, dtype=tf_type)
    
    def matrix(self, X, amp, ls):
        K = self.cross(X,X,amp, ls)
        K = K + self.jitter * tf.eye(tf.shape(X)[0], dtype=tf_type)
        return K       

    def cross(self, X1, X2, amp, ls):
        norm1 = tf.reshape(tf.reduce_sum(X1**2, 1), [-1, 1])
        norm2 = tf.reshape(tf.reduce_sum(X2**2, 1), [1, -1])
        K = norm1 - 2.0 * tf.matmul(X1, tf.transpose(X2)) + norm2
        K = amp * tf.exp(-1.0 * K / ls)
        return K

# class Kernel_RBF:
#     def __init__(self, jitter=0):
#         self.jitter = tf.constant(jitter, dtype=tf_type)
    
#     def matrix(self, X, ls):
#         K = self.cross(X,X, ls)
#         K = K + self.jitter * tf.eye(tf.shape(X)[0], dtype=tf_type)
#         return K       

#     def cross(self, X1, X2, ls):
#         norm1 = tf.reshape(tf.reduce_sum(X1**2, 1), [-1, 1])
#         norm2 = tf.reshape(tf.reduce_sum(X2**2, 1), [1, -1])
#         K = norm1 - 2.0 * tf.matmul(X1, tf.transpose(X2)) + norm2
#         K = tf.exp(-1.0 * K / ls)
#         return K

class Kernel_RBF:
    def __init__(self, jitter=0):
        self.jitter = tf.constant(jitter, dtype=tf_type)
    
    def matrix(self, X, amp, ls):
        K = self.cross(X,X, amp, ls)
        K = K + self.jitter * tf.eye(tf.shape(X)[0], dtype=tf_type)
        return K       

    def cross(self, X1, X2, amp, ls):
        norm1 = tf.reshape(tf.reduce_sum(X1**2, 1), [-1, 1])
        norm2 = tf.reshape(tf.reduce_sum(X2**2, 1), [1, -1])
        K = norm1 - 2.0 * tf.matmul(X1, tf.transpose(X2)) + norm2
        K = tf.exp(-1.0 * K / ls)
        return K
        

class NN:
    # init 0 xavier
    # init 1 msra
    def __init__(self, layers, init=0):
        self.num_layers = len(layers)

        if init == 0:
            self.init = self.xavier_init
        else:
            self.init = self.msra_init

        self.weights = []
        self.biases = []
        for l in range(self.num_layers - 1):
            W = self.init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf_type))
            self.weights.append(W)
            self.biases.append(b)
    
    def foward(self, X):
        H = X
        for l in range(self.num_layers-2):
            W = self.weights[l]
            b = self.biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = self.weights[-1]
        b = self.biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev))

    def msra_init(self,size):
        in_dim = size[0]
        out_dim = size[1]    
        msra_stddev = np.sqrt(2.0/(in_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=msra_stddev))
