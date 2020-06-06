import numpy as np
import tensorflow as tf
import math
from scipy.special import logsumexp
from tqdm import tqdm
import ExpUtil as util

class SGPRN:
    def __init__(self, config, label=None, init_type=2):
        
        print('Initialize Scalable GPRN ...')
        
        self.config = config 
        
        if label == None:
            self.signature = self.config['data']['dname']
        else:
            self.signature = label
        
        self.Q = self.config['Q']
        self.in_d = self.config['data']['X_train'].shape[1]
        self.out_d = self.config['data']['Y_train'].shape[1]
        self.jitter = config['jitter']
        
        self.tf_type = util.tf_type

        self.X_train = self.config['data']['X_train'].reshape((-1, self.in_d))
        self.X_test = self.config['data']['X_test'].reshape((-1, self.in_d))
        self.Y_train = self.config['data']['Y_train'].reshape((-1, self.out_d))
        self.Y_test = self.config['data']['Y_test'].reshape((-1, self.out_d))
        self.Y_test_ground = self.config['data']['Y_test_ground'].reshape((-1, self.out_d))
        self.Y_mean = self.config['data']['Y_mean']
        self.Y_std = self.config['data']['Y_std']
        
        self.epochs = config['epochs']
        self.alpha = config['alpha']
        
        self.M = self.X_test.shape[0]
        self.N = self.X_train.shape[0]
        self.D = self.out_d
        self.K = self.Q
        
        DList = self.config['data']['DList']
        self.D1 = int(DList[0])
        self.D2 = int(DList[1])
        self.D3 = int(DList[2])

        self.tf_X_train = tf.compat.v1.placeholder(util.tf_type, [None, self.in_d])
        self.tf_X_test = tf.compat.v1.placeholder(util.tf_type, [None, self.in_d])
        self.tf_Y_train = tf.compat.v1.placeholder(util.tf_type, [None, self.out_d])
        self.tf_Y_test = tf.compat.v1.placeholder(util.tf_type, [None, self.out_d])

        if self.config['kernel'] == 'matern1':
            self.kernel_ard = util.Kernel_Matern1(self.jitter)
        elif self.config['kernel'] == 'matern3':
            self.kernel_ard = util.Kernel_Matern3(self.jitter)
        elif self.config['kernel'] == 'matern5':
            self.kernel_ard = util.Kernel_Matern5(self.jitter)
        elif self.config['kernel'] == 'ard':
            self.kernel_ard = util.Kernel_ARD(self.jitter)
        elif self.config['kernel'] == 'rbf':
            self.kernel_ard = util.Kernel_RBF(self.jitter)
        
        
        # f kernel
        self.tf_log_f_ls = tf.Variable(0, dtype=util.tf_type) 
        self.tf_log_f_amp =tf.Variable(0, dtype=util.tf_type)
        self.tf_log_W_ls = tf.Variable(0, dtype=util.tf_type)
        self.tf_log_W_amp = tf.Variable(0, dtype=util.tf_type)

        # noise variance
        self.tf_log_f_tau = tf.Variable(0, dtype=util.tf_type) 
        self.tf_log_y_tau = tf.Variable(0, dtype=util.tf_type)
    
        # mean tensors
        if init_type == 1:
            self.tf_f = tf.Variable(tf.random.normal([self.N, self.Q, 1]), dtype=util.tf_type)
            self.tf_W = tf.Variable(tf.random.normal([self.N, self.out_d,self.Q]), dtype=util.tf_type) 
        elif init_type == 2:
            self.tf_f = tf.Variable(0.001*tf.random.truncated_normal([self.N, self.Q, 1]), dtype=util.tf_type)
            self.tf_W = tf.Variable(0.001*tf.random.truncated_normal([self.N, self.out_d,self.Q]), dtype=util.tf_type) 
        else:
            self.tf_f = tf.Variable(init_type['F'], dtype=util.tf_type)
            self.tf_W = tf.Variable(init_type['W'], dtype=util.tf_type) 

        self.W_chl_N = tf.linalg.band_part(tf.Variable(tf.random.normal([self.N, self.N],
               mean=0.0, stddev=self.config['init_std']), dtype=util.tf_type), -1,0)                  
        self.W_chl_D1 = tf.linalg.band_part(tf.Variable(tf.random.normal([self.D1, self.D1],
               mean=0.0, stddev=self.config['init_std']), dtype=util.tf_type), -1,0)
        self.W_chl_D2 = tf.linalg.band_part(tf.Variable(tf.random.normal([self.D2, self.D2],
               mean=0.0, stddev=self.config['init_std']), dtype=util.tf_type), -1,0)
        self.W_chl_D3 = tf.linalg.band_part(tf.Variable(tf.random.normal([self.D3, self.D3],
               mean=0.0, stddev=self.config['init_std']), dtype=util.tf_type), -1,0)
        self.W_chl_K = tf.linalg.band_part(tf.Variable(tf.random.normal([self.K, self.K],
               mean=0.0, stddev=self.config['init_std']), dtype=util.tf_type), -1,0)
        
        self.f_chl_N = tf.linalg.band_part(tf.Variable(tf.random.normal([self.N, self.N],
               mean=0.0, stddev=self.config['init_std']), dtype=util.tf_type), -1,0)
        self.f_chl_K = tf.linalg.band_part(tf.Variable(tf.random.normal([self.K, self.K],
               mean=0.0, stddev=self.config['init_std']), dtype=util.tf_type), -1,0)

        self.tf_W_var_N = self.W_chl_N @ tf.transpose(self.W_chl_N) 
        self.tf_W_var_D1 = self.W_chl_D1 @ tf.transpose(self.W_chl_D1)
        self.tf_W_var_D2 = self.W_chl_D2 @ tf.transpose(self.W_chl_D2)
        self.tf_W_var_D3 = self.W_chl_D3 @ tf.transpose(self.W_chl_D3)        
        self.tf_W_var_K = self.W_chl_K @ tf.transpose(self.W_chl_K)
        
        self.tf_f_var_N = self.f_chl_N @ tf.transpose(self.f_chl_N)
        self.tf_f_var_K = self.f_chl_K @ tf.transpose(self.f_chl_K)
        
        self.tf_Y_pred = self.predict(self.tf_X_test)

        self.entropy = self.eval_entropy()
        self.expect_prior = self.eval_expect_prior()
        self.expect_llh = self.eval_expect_llh()
        self.elbo_loss = -(self.entropy + self.expect_prior + self.expect_llh)
        
        # for the convinience of generating posterior samples
        self.epsi_W = tf.compat.v1.placeholder(shape=[self.N, self.D1, self.D2, self.D3, self.K], dtype=self.tf_type)        
        self.epsi_f = tf.compat.v1.placeholder(shape=[self.N, self.K, 1], dtype=self.tf_type)
        self.eta_W = tf.compat.v1.placeholder(shape=[self.M, self.D, self.K], dtype=self.tf_type)
        self.eta_f = tf.compat.v1.placeholder(shape=[self.M, self.K, 1], dtype=self.tf_type)
        
        self.W_test_sample, self.f_test_sample = self.sample_test_posterior(
            self.tf_X_test, self.epsi_W, self.epsi_f, self.eta_W, self.eta_f)

        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.alpha)
        self.minimizer = self.optimizer.minimize(self.elbo_loss)
        self.it = 0
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=True))
        self.sess.run(tf.compat.v1.global_variables_initializer())
    
    def eval_entropy(self):

        logdet_W_N = 2 * tf.linalg.trace(tf.math.log(tf.abs(self.W_chl_N)))
        
        logdet_W_D1 = 2 * tf.linalg.trace(tf.math.log(tf.abs(self.W_chl_D1)))
        logdet_W_D2 = 2 * tf.linalg.trace(tf.math.log(tf.abs(self.W_chl_D2)))
        logdet_W_D3 = 2 * tf.linalg.trace(tf.math.log(tf.abs(self.W_chl_D3)))
        
        logdet_W_K = 2 * tf.linalg.trace(tf.math.log(tf.abs(self.W_chl_K)))

        H_qW = 0.5 * (
            self.K * self.D1 * self.D2 * self.D3 * logdet_W_N +\
            self.K * self.D1 * self.D2 * logdet_W_D3 * self.N +\
            self.K * self.D1 * logdet_W_D2 * self.D3 * self.N +\
            self.K * logdet_W_D1 * self.D2 * self.D3 * self.N +\
            logdet_W_K * self.D1 * self.D2 * self.D3 * self.N
        )
        
        logdet_f_N = 2 * tf.linalg.trace(tf.math.log(tf.abs(self.f_chl_N)))
        logdet_f_K = 2 * tf.linalg.trace(tf.math.log(tf.abs(self.f_chl_K)))
        
        H_qf = 0.5 * (self.K*logdet_f_N + self.N*logdet_f_K)
        
        entropy = tf.cast(H_qW + H_qf, tf.float64)
        
        return entropy
    
    def eval_expect_prior(self):
        
        Kw = self.kernel_ard.matrix(self.tf_X_train, tf.exp(self.tf_log_W_amp), tf.exp(self.tf_log_W_ls))
        Kf = self.kernel_ard.matrix(self.tf_X_train, tf.exp(self.tf_log_f_amp), tf.exp(self.tf_log_f_ls)) + tf.exp(-self.tf_log_f_tau) * tf.eye(self.N)
        
        Wt = tf.transpose(self.tf_W, perm=[1, 2, 0]) # D by K by N
        mu_WWt = tf.tensordot(self.tf_W, Wt, axes=2)
        var_WWt = tf.linalg.trace(tf.linalg.solve(Kw, self.tf_W_var_N)) *\
              tf.linalg.trace(self.tf_W_var_D1)*\
              tf.linalg.trace(self.tf_W_var_D2)*\
              tf.linalg.trace(self.tf_W_var_D3)*\
              tf.linalg.trace(self.tf_W_var_K)
        
        expect_prior_W = tf.cast(
                -0.5 * self.K * self.D1 * self.D2 * self.D3 * tf.linalg.logdet(Kw) -\
                0.5 * tf.linalg.trace(tf.linalg.solve(Kw, mu_WWt)) -\
                0.5 * var_WWt, tf.float64)

        ft = tf.transpose(tf.squeeze(self.tf_f))
        f1 = tf.expand_dims(ft, -1)
        f2 = tf.expand_dims(ft, -2)
        mu_fft = tf.reduce_sum(tf.matmul(f1, f2), axis=[0])
        
        # special handling when rank = 1
        if self.Q == 1:
            mu_fft = tf.reshape(mu_fft, shape=[tf.size(mu_fft), self.Q])
        
        var_fft = tf.linalg.trace(tf.linalg.solve(Kf, self.tf_f_var_N)) *\
                  tf.linalg.trace(self.tf_f_var_K)
        
        expect_prior_f = tf.cast(
                -0.5 * self.K * tf.linalg.logdet(Kf) -\
                0.5 * tf.linalg.trace(tf.linalg.solve(Kf, mu_fft)) -\
                0.5 * var_fft, tf.float64)      

        return expect_prior_f + expect_prior_W
  
    def eval_expect_llh(self):  
        
        ynyn = tf.squeeze(tf.matmul(tf.expand_dims(self.tf_Y_train, 1), tf.expand_dims(self.tf_Y_train, -1)))
        ynWnfn = tf.squeeze(tf.matmul(tf.expand_dims(self.tf_Y_train, 1), tf.matmul(self.tf_W, self.tf_f)))

        WnWn = tf.matmul(tf.transpose(self.tf_W, perm=[0, 2, 1]), self.tf_W)
        expand_const_W = tf.expand_dims(tf.linalg.diag_part(self.tf_W_var_N), -1) *\
            tf.linalg.trace(tf.matmul(self.tf_W_var_D1, tf.eye(self.D1))) *\
            tf.linalg.trace(tf.matmul(self.tf_W_var_D2, tf.eye(self.D2))) *\
            tf.linalg.trace(tf.matmul(self.tf_W_var_D3, tf.eye(self.D3)))
        
        WWvar = tf.tensordot(expand_const_W, tf.expand_dims(self.tf_W_var_K, 0), axes=1)

        fnfn = tf.matmul(self.tf_f, tf.transpose(self.tf_f, perm=[0, 2, 1]))
        expand_const_f = tf.expand_dims(tf.linalg.diag_part(self.tf_f_var_N), -1)
        ffvar = tf.tensordot(expand_const_f, tf.expand_dims(self.tf_f_var_K, 0), axes=1)

        tr_WnWn_fnfn = tf.linalg.trace(tf.matmul(WnWn+WWvar, fnfn+ffvar))
        expt_Wnfn = ynyn - 2*ynWnfn + tr_WnWn_fnfn
        
        expt_llh = tf.cast(
                self.N * self.D * self.tf_log_y_tau * 0.5 -\
                tf.exp(self.tf_log_y_tau) * 0.5 * tf.reduce_sum(expt_Wnfn), 
                tf.float64)
        
         
        return expt_llh
    
    def predict(self, X):
        Kw11 = self.kernel_ard.matrix(self.tf_X_train, tf.exp(self.tf_log_W_amp), tf.exp(self.tf_log_W_ls)) #N by N
        Kw12 = self.kernel_ard.cross(self.tf_X_train, X, tf.exp(self.tf_log_W_amp), tf.exp(self.tf_log_W_ls)) #N by M
        Kw11InvKw12 = tf.linalg.solve(Kw11, Kw12)#N by M
        #D by K by 1 by N
        W_tr = tf.expand_dims(tf.transpose(self.tf_W, perm=[1,2,0]), 2)
        #D by K by M (posterior mean)
        W_mu = tf.squeeze(tf.tensordot(W_tr, Kw11InvKw12, axes=[[3],[0]]))
        #M by D by K
        W_mu = tf.transpose(W_mu, perm=[2,0,1])
        Kw22 = self.kernel_ard.matrix(X, tf.exp(self.tf_log_W_amp), tf.exp(self.tf_log_W_ls))
        #[M] vector, note each W_ij have the same variance
        W_std = tf.sqrt(tf.linalg.tensor_diag_part(Kw22) - tf.reduce_sum(Kw11InvKw12*Kw12, 0))
        #M by 1 by 1
        W_std = tf.expand_dims(tf.expand_dims(W_std,-1), -1)
        #regarding F
        log_tau = self.tf_log_f_tau
        amp = tf.exp(self.tf_log_f_amp)
        ls = tf.exp(self.tf_log_f_ls)
        Kf = self.kernel_ard.matrix(self.tf_X_train, amp, ls)
        K11 = Kf + tf.exp(-self.tf_log_f_tau) * tf.eye(self.N)
        K21 = self.kernel_ard.cross(X, self.tf_X_train, amp, ls)
        #M by N
        K21K11Inv = tf.transpose(tf.linalg.solve(K11, tf.transpose(K21)))
        K22 = self.kernel_ard.matrix(X, amp, ls)
        #N by K by 1 ==> M by K by 1
        f_mu = tf.tensordot(K21K11Inv, self.tf_f, axes=[[1],[0]])
        #[M] 
        f_std = tf.sqrt(tf.linalg.tensor_diag_part(K22) - tf.reduce_sum(K21K11Inv*K21, 1))
        #M by 1 by 1
        f_std = tf.expand_dims(tf.expand_dims(f_std, -1), -1)
        
        Y_pred = tf.squeeze( tf.matmul(W_mu, f_mu) )
        return (Y_pred, W_mu, W_std, f_mu, f_std, self.tf_log_y_tau)
    
    def sample_test_posterior(self, X, epsi_W, epsi_f, eta_W, eta_f):
        """ Sample W|D, f|D """
        scale_W = tf.tensordot(epsi_W, self.W_chl_N, axes=[[0], [0]])
        scale_W = tf.tensordot(scale_W, self.W_chl_D1, axes=[[0], [0]])
        scale_W = tf.tensordot(scale_W, self.W_chl_D2, axes=[[0], [0]])
        scale_W = tf.tensordot(scale_W, self.W_chl_D3, axes=[[0], [0]])
        scale_W = tf.tensordot(scale_W, self.W_chl_K, axes=[[0], [0]])
        scale_W = tf.reshape(scale_W, tf.shape(self.tf_W))
        W_post = self.tf_W + scale_W
        
        scale_f = tf.tensordot(epsi_f, self.f_chl_N, axes=[[0], [0]])
        scale_f = tf.tensordot(scale_f, self.f_chl_K, axes=[[0], [0]])
        scale_f = tf.tensordot(scale_f, tf.eye(1), axes=[[0], [0]])
        f_post = self.tf_f + scale_f  
        
        """ Sample W_star|W, f_star|f """
        Kw11 = self.kernel_ard.matrix(self.tf_X_train, tf.exp(self.tf_log_W_amp), tf.exp(self.tf_log_W_ls)) #N by N
        Kw12 = self.kernel_ard.cross(self.tf_X_train, X, tf.exp(self.tf_log_W_amp), tf.exp(self.tf_log_W_ls)) #N by M
        Kw11InvKw12 = tf.linalg.solve(Kw11, Kw12)#N by M
        #D by K by 1 by N
        W_tr = tf.expand_dims(tf.transpose(W_post, perm=[1,2,0]), 2)
        #D by K by M (posterior mean)
        W_mu = tf.squeeze(tf.tensordot(W_tr, Kw11InvKw12, axes=[[3],[0]]))
        #M by D by K
        W_mu = tf.transpose(W_mu, perm=[2,0,1])
        Kw22 = self.kernel_ard.matrix(X, tf.exp(self.tf_log_W_amp), tf.exp(self.tf_log_W_ls))
        #[M] vector, note each W_ij have the same variance
        W_std = tf.sqrt(tf.diag_part(Kw22) - tf.reduce_sum(Kw11InvKw12*Kw12, 0))
        #M by 1 by 1
        W_std = tf.expand_dims(tf.expand_dims(W_std,-1), -1)
        #regarding F
        log_tau = self.tf_log_f_tau
        amp = tf.exp(self.tf_log_f_amp)
        ls = tf.exp(self.tf_log_f_ls)
        Kf = self.kernel_ard.matrix(self.tf_X_train, amp, ls)
        K11 = Kf + tf.exp(-self.tf_log_f_tau) * tf.eye(self.N)
        K21 = self.kernel_ard.cross(X, self.tf_X_train, amp, ls)
        #M by N
        K21K11Inv = tf.transpose(tf.linalg.solve(K11, tf.transpose(K21)))
        K22 = self.kernel_ard.matrix(X, amp, ls)
        #N by K by 1 ==> M by K by 1
        f_mu = tf.tensordot(K21K11Inv, f_post, axes=[[1],[0]])
        #[M] 
        f_std = tf.sqrt(tf.diag_part(K22) - tf.reduce_sum(K21K11Inv*K21, 1))
        #M by 1 by 1
        f_std = tf.expand_dims(tf.expand_dims(f_std, -1), -1)
        
        
        W_sample = eta_W*W_std + W_mu
        f_sample = eta_f*f_std + f_mu
        
        
        return W_sample, f_sample
    
    def eval_test_post(self):
        nS = self.config['ns']
        samples = []
        for s in range(nS):
            epsi_W_test = np.random.normal(size=[self.N, self.D1, self.D2, self.D3, self.K])
            epsi_f_test = np.random.normal(size=[self.N, self.K, 1])
            eta_W_test = np.random.normal(size=[self.M, self.D, self.K])
            eta_f_test = np.random.normal(size=[self.M, self.K, 1])

            fdict = {
                self.tf_X_train : self.X_train,
                self.tf_Y_train : self.Y_train,
                self.tf_X_test : self.X_test,
                self.tf_Y_test : self.Y_test,
                self.epsi_W : epsi_W_test,
                self.epsi_f : epsi_f_test,
                self.eta_W : eta_W_test,
                self.eta_f : eta_f_test,
            }
            
            W_sample, f_sample, log_tau =\
                self.sess.run([self.W_test_sample, self.f_test_sample, self.tf_log_y_tau], feed_dict=fdict)
            
            samples.append(np.squeeze(np.matmul(W_sample, f_sample)))
            
        return np.array(samples)


    def test_pred_ll(self):

        N_Y_test = self.config['data']['Y_test']
        
        #test log-likelihood
        S = self.config['ns']
        N_Y_test = np.expand_dims(N_Y_test, -1) #M by D by 1
        vals = np.zeros(S)

        for s in range(S):

            epsi_W_test = np.random.normal(size=[self.N, self.D1, self.D2, self.D3, self.K])
            epsi_f_test = np.random.normal(size=[self.N, self.K, 1])
            eta_W_test = np.random.normal(size=[self.M, self.D, self.K])
            eta_f_test = np.random.normal(size=[self.M, self.K, 1])

            fdict = {
                self.tf_X_train : self.X_train,
                self.tf_Y_train : self.Y_train,
                self.tf_X_test : self.X_test,
                self.tf_Y_test : self.Y_test,
                self.epsi_W : epsi_W_test,
                self.epsi_f : epsi_f_test,
                self.eta_W : eta_W_test,
                self.eta_f : eta_f_test,
            }
            
            W_sample, f_sample, log_tau = self.sess.run([self.W_test_sample, self.f_test_sample, self.tf_log_y_tau], feed_dict=fdict)
            vals[s] = -0.5*np.exp(log_tau)*np.sum(np.square(N_Y_test - np.matmul(W_sample, f_sample)))
        
        test_ll = logsumexp(vals) - np.log(S) + 0.5*N_Y_test.size*(log_tau - np.log(2*np.pi))
        test_ll = test_ll/N_Y_test.shape[0]
        
        return test_ll
         
    def fit(self):
        
        self.it = 0
        hist_nrmse = []
        hist_nmae = []
        hist_pred = []
        hist_nelbo = []
        hist_test_samples = []
        
        res = {}
        
        fdict = {self.tf_X_train: self.X_train, self.tf_X_test: self.X_test, self.tf_Y_train: self.Y_train, self.tf_Y_test: self.Y_test}
        print('** Current experiment:', self.signature)
        for it in tqdm(range(self.epochs + 1)):
            self.sess.run(self.minimizer,feed_dict = fdict)

            if it%self.config['interval']==0:
                nelbo = self.sess.run(self.elbo_loss, feed_dict=fdict)
                [N_Y_pred, W_mu, W_std, f_mu, f_std, log_tau] = self.sess.run(self.tf_Y_pred, feed_dict=fdict)

                Y_pred = N_Y_pred * self.Y_std + self.Y_mean             
                r0 = np.sqrt(np.mean(np.square(Y_pred - self.Y_test_ground))) / np.std(self.Y_test_ground) # nrmse
                r1 = np.mean(np.abs(Y_pred - self.Y_test_ground)) / np.std(self.Y_test_ground) # nmae
                samples = self.eval_test_post()
                
                hist_pred.append(N_Y_pred)
                hist_nrmse.append(r0)
                hist_nmae.append(r1)
                hist_nelbo.append(nelbo)
                hist_test_samples.append(samples)

                tqdm.write('it #%d, nelbo= %.2f, nmae= %.5f, nrmse= %.5f' % (it, nelbo, r1, r0))
        
        tf.compat.v1.reset_default_graph()
        
        res['domain'] = self.config['data']['dname']
        res['Y_pred'] = np.array(hist_pred)
        res['nrmse'] = np.array(hist_nrmse)
        res['nmae'] = np.array(hist_nmae)
        res['nelbos'] = np.array(hist_nelbo)
        res['test_samples'] = hist_test_samples
        
        return res
    

    