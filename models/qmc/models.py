'''
Quantum Measurement Classfiication Models
'''

import tensorflow as tf
import layers as layers

class QMClassifier(tf.keras.Model):
    """
    A Quantum Measurement Classifier model.
    Arguments:
        fm_x: Quantum feature map layer for inputs
        fm_y: Quantum feature map layer for outputs
        dim_x: dimension of the input quantum feature map
        dim_y: dimension of the output representation
    """
    def __init__(self, fm_x, fm_y, dim_x, dim_y):
        super(QMClassifier, self).__init__()
        self.fm_x = fm_x
        self.fm_y = fm_y
        self.qm = layers.QMeasureClassif(dim_x=dim_x, dim_y=dim_y)
        self.dm2dist = layers.DensityMatrix2Dist()
        self.cp1 = layers.CrossProduct()
        self.cp2 = layers.CrossProduct()
        self.num_samples = tf.Variable(
            initial_value=0.,
            trainable=False     
            )

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        rho_y = self.qm(psi_x)
        probs = self.dm2dist(rho_y)
        return probs

    @tf.function
    def call_train(self, x, y):
        if not self.qm.built:
            self.call(x)
        psi_x = self.fm_x(x)
        psi_y = self.fm_y(y)
        psi = self.cp1([psi_x, psi_y])
        rho = self.cp2([psi, tf.math.conj(psi)])
        num_samples = tf.cast(tf.shape(x)[0], rho.dtype)
        rho = tf.reduce_sum(rho, axis=0)
        self.num_samples.assign_add(num_samples)
        return rho

    def train_step(self, data):
        x, y = data
        rho = self.call_train(x, y)
        self.qm.weights[0].assign_add(rho)
        return {}

    def fit(self, *args, **kwargs):
        result = super(QMClassifier, self).fit(*args, **kwargs)
        self.qm.weights[0].assign(self.qm.weights[0] / self.num_samples)
        return result

    def get_rho(self):
        return self.weights[2]

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y
        }
        base_config = super().get_config()
        return {**base_config, **config}


class QMClassifierSGD(tf.keras.Model):
    """
    A Quantum Measurement Classifier model trainable using
    gradient descent.

    Arguments:
        input_dim: dimension of the input
        dim_x: dimension of the input quantum feature map
        dim_y: dimension of the output representation
        num_eig: Number of eigenvectors used to represent the density matrix. 
                 a value of 0 or less implies num_eig = dim_x
        gamma: float. Gamma parameter of the RBF kernel to be approximated.
        random_state: random number generator seed.
    """
    def __init__(self, input_dim, dim_x, dim_y, num_eig=0, gamma=1, random_state=None):
        super(QMClassifierSGD, self).__init__()
        self.fm_x = layers.QFeatureMapRFF(
            input_dim=input_dim,
            dim=dim_x, gamma=gamma, random_state=random_state)
        self.qm = layers.QMeasureClassifEig(dim_x=dim_x, dim_y=dim_y, num_eig=num_eig)
        self.dm2dist = layers.DensityMatrix2Dist()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.gamma = gamma
        self.random_state = random_state

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        rho_y = self.qm(psi_x)
        probs = self.dm2dist(rho_y)
        return probs

    def set_rho(self, rho):
        return self.qm.set_rho(rho)

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y,
            "num_eig": self.num_eig,
            "gamma": self.gamma,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}

class QMClassifierSGDF(tf.keras.Model):
    """
    A Quantum Measurement Classifier model trainable using
    gradient descent.
    Arguments:
        input_dim: dimension of the input
        dim_x: dimension of the input quantum feature map
        dim_y: dimension of the output representation
        num_eig: Number of eigenvectors used to represent the density matrix. 
                 a value of 0 or less implies num_eig = dim_x
        gamma: float. Gamma parameter of the RBF kernel to be approximated.
        random_state: random number generator seed.
    """
    def __init__(self, input_dim, dim_x, dim_y, num_eig=0, gamma=1, random_state=None):
        super(QMClassifierSGDF, self).__init__()
        self.fm_x = layers.QFeatureMapRFF(
            input_dim=input_dim,
            dim=dim_x, gamma=gamma, random_state=random_state)
        self.qm = layers.QMeasureClassifEigF(dim_x=dim_x, dim_y=dim_y, num_eig=num_eig)
        self.dm2dist = layers.DensityMatrix2Dist()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.gamma = gamma
        self.random_state = random_state

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        rho_y = self.qm(psi_x)
        probs = self.dm2dist(rho_y)
        return probs

    def set_rho(self, rho):
        return self.qm.set_rho(rho)

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y,
            "num_eig": self.num_eig,
            "gamma": self.gamma,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}

class QMDensity(tf.keras.Model):
    """
    A Quantum Measurement Density Estimation model.
    Arguments:
        fm_x: Quantum feature map layer for inputs
        dim_x: dimension of the input quantum feature map
    """
    def __init__(self, fm_x, dim_x):
        super(QMDensity, self).__init__()
        self.fm_x = fm_x
        self.dim_x = dim_x
        self.qmd = layers.QMeasureDensity(dim_x)
        self.cp = layers.CrossProduct()
        self.num_samples = tf.Variable(
            initial_value=0.,
            trainable=False     
            )

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        probs = self.qmd(psi_x)
        return probs

    @tf.function
    def call_train(self, x):
        if not self.qmd.built:
            self.call(x)
        psi = self.fm_x(x)
        rho = self.cp([psi, tf.math.conj(psi)])
        num_samples = tf.cast(tf.shape(x)[0], rho.dtype)
        rho = tf.reduce_sum(rho, axis=0)
        self.num_samples.assign_add(num_samples)
        return rho

    def train_step(self, data):
        x = data
        rho = self.call_train(x)
        self.qmd.weights[0].assign_add(rho)
        return {}

    def fit(self, *args, **kwargs):
        result = super(QMDensity, self).fit(*args, **kwargs)
        self.qmd.weights[0].assign(self.qmd.weights[0] / self.num_samples)
        return result

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

class QMDensitySGD(tf.keras.Model):
    """
    A Quantum Measurement Density Estimation modeltrainable using
    gradient descent.
    Arguments:
        input_dim: dimension of the input
        dim_x: dimension of the input quantum feature map
        num_eig: Number of eigenvectors used to represent the density matrix. 
                 a value of 0 or less implies num_eig = dim_x
        gamma: float. Gamma parameter of the RBF kernel to be approximated.
        random_state: random number generator seed.
    """
    def __init__(self, input_dim, dim_x, num_eig=0, gamma=1, random_state=None):
        super(QMDensitySGD, self).__init__()
        self.fm_x = layers.QFeatureMapRFF(
            input_dim=input_dim,
            dim=dim_x, gamma=gamma, random_state=random_state)
        self.qmd = layers.QMeasureDensityEig(dim_x=dim_x, num_eig=num_eig)
        self.num_eig = num_eig
        self.dim_x = dim_x
        self.gamma = gamma
        self.random_state = random_state

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        probs = self.qmd(psi_x)
        self.add_loss(-tf.reduce_sum(tf.math.log(probs)))
        return probs

    def set_rho(self, rho):
        return self.qmd.set_rho(rho)

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "gamma": self.gamma,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}

class QMKDClassifier(tf.keras.Model):
    """
    A Quantum Measurement Kernel Density Classifier model.
    Arguments:
        fm_x: Quantum feature map layer for inputs
        dim_x: dimension of the input quantum feature map
        num_classes: int number of classes
    """
    def __init__(self, fm_x, dim_x, num_classes=2):
        super(QMKDClassifier, self).__init__()
        self.fm_x = fm_x
        self.dim_x = dim_x
        self.num_classes = num_classes
        self.qmd = []
        for _ in range(num_classes):
            self.qmd.append(layers.QMeasureDensity(dim_x))
        self.cp = layers.CrossProduct()
        self.num_samples = tf.Variable(
            initial_value=tf.zeros((num_classes,)),
            trainable=False
            )

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        probs = []
        for i in range(self.num_classes):
            probs.append(self.qmd[i](psi_x))
        posteriors = tf.stack(probs, axis=-1)
        posteriors = posteriors / tf.expand_dims(tf.reduce_sum(posteriors, axis=-1), axis=-1)
        return posteriors

    @tf.function
    def call_train(self, x, y):
        if not self.qmd[0].built:
            self.call(x)
        psi = self.fm_x(x) # shape (bs, dim_x)
        rho = self.cp([psi, tf.math.conj(psi)]) # shape (bs, dim_x, dim_x)
        ohy = tf.keras.backend.one_hot(y, self.num_classes)
        ohy = tf.reshape(ohy, (-1, self.num_classes))
        num_samples = tf.squeeze(tf.reduce_sum(ohy, axis=0))
        ohy = tf.expand_dims(ohy, axis=-1) 
        ohy = tf.expand_dims(ohy, axis=-1) # shape (bs, num_classes, 1, 1)
        rhos = ohy * tf.expand_dims(rho, axis=1) # shape (bs, num_classes, dim_x, dim_x)
        rhos = tf.reduce_sum(rhos, axis=0) # shape (num_classes, dim_x, dim_x)
        self.num_samples.assign_add(num_samples)
        return rhos

    def train_step(self, data):
        x, y = data
        rhos = self.call_train(x, y)
        for i in range(self.num_classes):
            self.qmd[i].weights[0].assign_add(rhos[i])
        return {}

    def fit(self, *args, **kwargs):
        result = super(QMKDClassifier, self).fit(*args, **kwargs)
        for i in range(self.num_classes):
            self.qmd[i].weights[0].assign(self.qmd[i].weights[0] /
                                          self.num_samples[i])
        return result

    def get_rhos(self):
        weights = [qmd.weights[0] for qmd in self.qmd]
        return weights

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "num_classes": self.num_classes
        }
        base_config = super().get_config()
        return {**base_config, **config}

class QMKDClassifierSGD(tf.keras.Model):
    """
    A Quantum Measurement Kernel Density Classifier model trainable using
    gradient descent.

    Arguments:
        input_dim: dimension of the input
        dim_x: dimension of the input quantum feature map
        num_classes: number of classes
        num_eig: Number of eigenvectors used to represent the density matrix. 
                 a value of 0 or less implies num_eig = dim_x
        gamma: float. Gamma parameter of the RBF kernel to be approximated
        random_state: random number generator seed
    """
    def __init__(self, input_dim, dim_x, num_classes, num_eig=0, gamma=1, random_state=None):
        super(QMKDClassifierSGD, self).__init__()
        self.fm_x = layers.QFeatureMapRFF(
            input_dim=input_dim,
            dim=dim_x, gamma=gamma, random_state=random_state)
        self.dim_x = dim_x
        self.num_classes = num_classes
        self.qmd = []
        for _ in range(num_classes):
            self.qmd.append(layers.QMeasureDensityEig(dim_x, num_eig))
        self.gamma = gamma
        self.random_state = random_state

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        probs = []
        for i in range(self.num_classes):
            probs.append(self.qmd[i](psi_x))
        posteriors = tf.stack(probs, axis=-1)
        posteriors = (posteriors / 
                      tf.expand_dims(tf.reduce_sum(posteriors, axis=-1), axis=-1))
        return posteriors

    def set_rhos(self, rhos):
        for i in range(self.num_classes):
            self.qmd[i].set_rho(rhos[i])
        return

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "num_classes": self.num_classes,
            "num_eig": self.num_eig,
            "gamma": self.gamma,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}

class QMRegressor(tf.keras.Model):
    """
    A Quantum Measurement Regression model.
    Arguments:
        fm_x: Quantum feature map layer for inputs
        fm_y: Quantum feature map layer for outputs
        dim_x: dimension of the input quantum feature map
        dim_y: dimension of the output quantum feature map
    """
    def __init__(self, fm_x, fm_y, dim_x, dim_y):
        super(QMRegressor, self).__init__()
        self.fm_x = fm_x
        self.fm_y = fm_y
        self.qm = layers.QMeasureClassif(dim_x=dim_x, dim_y=dim_y)
        self.dmregress = layers.DensityMatrixRegression()
        self.cp1 = layers.CrossProduct()
        self.cp2 = layers.CrossProduct()
        self.num_samples = tf.Variable(
            initial_value=0.,
            trainable=False     
            )

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        rho_y = self.qm(psi_x)
        mean_var = self.dmregress(rho_y)
        return mean_var

    @tf.function
    def call_train(self, x, y):
        if not self.qm.built:
            self.call(x)
        psi_x = self.fm_x(x)
        psi_y = self.fm_y(y)
        psi = self.cp1([psi_x, psi_y])
        rho = self.cp2([psi, tf.math.conj(psi)])
        num_samples = tf.cast(tf.shape(x)[0], rho.dtype)
        rho = tf.reduce_sum(rho, axis=0)
        self.num_samples.assign_add(num_samples)
        return rho

    def train_step(self, data):
        x, y = data
        rho = self.call_train(x, y)
        self.qm.weights[0].assign_add(rho)
        return {}

    def fit(self, *args, **kwargs):
        result = super(QMRegressor, self).fit(*args, **kwargs)
        self.qm.weights[0].assign(self.qm.weights[0] / self.num_samples)
        return result

    def get_rho(self):
        return self.weights[2]

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y
        }
        base_config = super().get_config()
        return {**base_config, **config}

class QMRegressor_2(tf.keras.Model):
    """
    A Quantum Measurement Regression model. Output is the raw distribution.
    Arguments:
        fm_x: Quantum feature map layer for inputs
        fm_y: Quantum feature map layer for outputs
        dim_x: dimension of the input quantum feature map
        dim_y: dimension of the output quantum feature map
    """
    def __init__(self, fm_x, fm_y, dim_x, dim_y):
        super(QMRegressor_2, self).__init__()
        self.fm_x = fm_x
        self.fm_y = fm_y
        self.qm = layers.QMeasureClassif(dim_x=dim_x, dim_y=dim_y)
        self.dm2dist = layers.DensityMatrix2Dist()
        self.dmregress = layers.DensityMatrixRegression()
        self.cp1 = layers.CrossProduct()
        self.cp2 = layers.CrossProduct()
        self.num_samples = tf.Variable(
            initial_value=0.,
            trainable=False     
            )

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        rho_y = self.qm(psi_x)
        mean_var = self.dmregress(rho_y)
        dist = self.dm2dist(rho_y)
        return [dist, mean_var]

    @tf.function
    def call_train(self, x, y):
        if not self.qm.built:
            self.call(x)
        psi_x = self.fm_x(x)
        psi_y = self.fm_y(y)
        psi = self.cp1([psi_x, psi_y])
        rho = self.cp2([psi, tf.math.conj(psi)])
        num_samples = tf.cast(tf.shape(x)[0], rho.dtype)
        rho = tf.reduce_sum(rho, axis=0)
        self.num_samples.assign_add(num_samples)
        return rho

    def train_step(self, data):
        x, y = data
        rho = self.call_train(x, y)
        self.qm.weights[0].assign_add(rho)
        return {}

    def fit(self, *args, **kwargs):
        result = super(QMRegressor_2, self).fit(*args, **kwargs)
        self.qm.weights[0].assign(self.qm.weights[0] / self.num_samples)
        return result

    def get_rho(self):
        return self.weights[2]

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y
        }
        base_config = super().get_config()
        return {**base_config, **config}

class QMRegressorSGD(tf.keras.Model):
    """
    A Quantum Measurement Regressor model trainable using
    gradient descent.

    Arguments:
        input_dim: dimension of the input
        dim_x: dimension of the input quantum feature map
        dim_y: dimension of the output quantum feature map
        num_eig: Number of eigenvectors used to represent the density matrix. 
                 a value of 0 or less implies num_eig = dim_x
        gamma: float. Gamma parameter of the RBF kernel to be approximated.
        random_state: random number generator seed.
    """
    def __init__(self, input_dim, dim_x, dim_y, num_eig=0, gamma=1, random_state=None):
        super(QMRegressorSGD, self).__init__()
        self.fm_x = layers.QFeatureMapRFF(
            input_dim=input_dim,
            dim=dim_x, gamma=gamma, random_state=random_state)
        self.qm = layers.QMeasureClassifEig(dim_x=dim_x, dim_y=dim_y, num_eig=num_eig)
        self.dmregress = layers.DensityMatrixRegression()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.gamma = gamma
        self.random_state = random_state

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        rho_y = self.qm(psi_x)
        mean_var = self.dmregress(rho_y)
        return mean_var

    def set_rho(self, rho):
        return self.qm.set_rho(rho)

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y,
            "num_eig": self.num_eig,
            "gamma": self.gamma,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}

class QMRegressorSGDF(tf.keras.Model):
    """
    A Quantum Measurement Regressor model trainable using
    gradient descent.
    Arguments:
        input_dim: dimension of the input
        dim_x: dimension of the input quantum feature map
        dim_y: dimension of the output quantum feature map
        num_eig: Number of eigenvectors used to represent the density matrix. 
                 a value of 0 or less implies num_eig = dim_x
        gamma: float. Gamma parameter of the RBF kernel to be approximated.
        random_state: random number generator seed.
    """
    def __init__(self, input_dim, dim_x, dim_y, num_eig=0, gamma=1, random_state=None):
        super(QMRegressorSGDF, self).__init__()
        self.fm_x = layers.QFeatureMapRFF(
            input_dim=input_dim,
            dim=dim_x, gamma=gamma, random_state=random_state)
        self.qm = layers.QMeasureClassifEigF(dim_x=dim_x, dim_y=dim_y, num_eig=num_eig)
        self.dmregress = layers.DensityMatrixRegression()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.gamma = gamma
        self.random_state = random_state

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        rho_y = self.qm(psi_x)
        mean_var = self.dmregress(rho_y)
        return mean_var

    def set_rho(self, rho):
        return self.qm.set_rho(rho)

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y,
            "num_eig": self.num_eig,
            "gamma": self.gamma,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}


        
class QMRegressorSGD_2(tf.keras.Model):
    """
    A Quantum Measurement Regressor model trainable using
    gradient descent.

    Arguments:
        input_dim: dimension of the input
        dim_x: dimension of the input quantum feature map
        dim_y: dimension of the output quantum feature map
        num_eig: Number of eigenvectors used to represent the density matrix. 
                 a value of 0 or less implies num_eig = dim_x
        gamma: float. Gamma parameter of the RBF kernel to be approximated.
        random_state: random number generator seed.
    """
    def __init__(self, input_dim, dim_x, dim_y, num_eig=0, gamma=1, random_state=None):
        super(QMRegressorSGD_2, self).__init__()
        self.fm_x = layers.QFeatureMapRFF(
            input_dim=input_dim,
            dim=dim_x, gamma=gamma, random_state=random_state)
        self.qm = layers.QMeasureClassifEigF(dim_x=dim_x, dim_y=dim_y, num_eig=num_eig)
        self.dm2dist = layers.DensityMatrix2Dist()
        self.dmregress = layers.DensityMatrixRegression()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.gamma = gamma
        self.random_state = random_state

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        rho_y = self.qm(psi_x)
        mean_var = self.dmregress(rho_y)
        dist = self.dm2dist(rho_y)
        return tf.concat([dist, mean_var],1)


    def set_rho(self, rho):
        return self.qm.set_rho(rho)

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y,
            "num_eig": self.num_eig,
            "gamma": self.gamma,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}
        
class QMFindingsClassifier(tf.keras.Model):
    """
    A Quantum Measurement Classifier model.
    Arguments:
        fm_x: Quantum feature map layer for inputs
        fm_y_1: Quantum feature map layer for outputs on 1st finding
        fm_y_2: Quantum feature map layer for outputs on 2nd finding       
        fm_y: Quantum feature map layer for outputs
        dim_x: dimension of the input quantum feature map
        dim_y_1: dimension of the 1st output representation
        dim_y_2: dimension of the 2nd output representation
        dim_y: dimension of the output representation
    """
    def __init__(self, fm_x, fm_y_1, fm_y_2, fm_y, dim_x, dim_y_1, dim_y_2, dim_y):
        super(QMFindingsClassifier, self).__init__()
        #print("Empieza el init")
        self.fm_x = fm_x
        self.fm_y_1 = fm_y_1
        self.fm_y_2 = fm_y_2
        self.fm_y = fm_y
        self.qm1 = layers.QMeasureClassif(dim_x=dim_x, dim_y=dim_y_1)
        self.qm2 = layers.QMeasureClassif(dim_x=dim_x, dim_y=dim_y_2)
        
        self.qm = layers.QMeasureClassif(dim_x=4, dim_y=dim_y)
        
        self.dm2dist_1 = layers.DensityMatrix2Dist()
        
        self.dm2dist = layers.DensityMatrix2Dist()
        
        self.cp1 = layers.CrossProduct()
        self.cp2 = layers.CrossProduct()
        
        self.cp3 = layers.CrossProduct()
        self.cp4 = layers.CrossProduct()
        
        self.cp5 = layers.CrossProduct()
        self.cp6 = layers.CrossProduct()
        self.cp7 = layers.CrossProduct()
        
        self.num_samples = tf.Variable(
            initial_value=0.,
            trainable=False     
            )

    def call(self, inputs):
   
        psi_x_1 = self.fm_x(inputs)
        #psi_x_2 = self.fm_x(inputs)
              
        rho_y_1 = self.qm1(psi_x_1)
        rho_y_2 = self.qm2(psi_x_1)
        
        psi_x_a = self.cp5([rho_y_1, rho_y_2])
        psi_x_a = tf.reshape(psi_x_a, [-1, 4, 4])
        psi_x_a = self.dm2dist_1(psi_x_a)
   
        rho_y = self.qm(psi_x_a)
        
        probs = self.dm2dist(rho_y)
        
        return probs

    @tf.function
    def call_train_1_2(self, x, y_tupla_1_2):

        y_1, y_2 = y_tupla_1_2

        if not self.qm1.built:
            self.call(x)
        psi_x = self.fm_x(x)

        psi_y_1 = self.fm_y_1(y_1)       
        psi_y_2 = self.fm_y_2(y_2)
        psi_1 = self.cp1([psi_x, psi_y_1])
        psi_2 = self.cp2([psi_x, psi_y_2])

        rho_1 = self.cp3([psi_1, tf.math.conj(psi_1)])
        rho_2 = self.cp4([psi_2, tf.math.conj(psi_2)])

        rho_1 = tf.reduce_sum(rho_1, axis=0)
        rho_2 = tf.reduce_sum(rho_2, axis=0)
     
        num_samples = tf.cast(tf.shape(x)[0], rho_1.dtype)
        self.num_samples.assign_add(num_samples)
                
        return rho_1, rho_2
        
    @tf.function
    def call_train(self, x, y):

        if not self.qm.built:
            self.call(x)
        psi_x = self.fm_x(x)
      
        rho_y_1 = self.qm1(psi_x)
        rho_y_2 = self.qm2(psi_x)
                      
        psi_x_a = self.cp5([rho_y_1, rho_y_2])
        psi_x_a = tf.reshape(psi_x_a, [-1, 4, 4])
        psi_x_a = self.dm2dist_1(psi_x_a)

        psi_y = self.fm_y(y)
        psi = self.cp6([psi_x_a, psi_y])

        rho = self.cp7([psi, tf.math.conj(psi)])
        rho = tf.reduce_sum(rho, axis=0)

        num_samples = tf.cast(tf.shape(x)[0], rho.dtype)
        self.num_samples.assign_add(num_samples)
                
        return rho    

    def train_step(self, data):

        x, y_tupla  = data
        
        y_1, y_2, y = y_tupla
        
        y_tupla_1_2 = (y_1, y_2)

        rho_1, rho_2 = self.call_train_1_2(x, y_tupla_1_2)
        
        self.qm1.weights[0].assign_add(rho_1)
        self.qm2.weights[0].assign_add(rho_2)
        
        rho = self.call_train(x, y)
       
        self.qm.weights[0].assign_add(rho)
        
        return {}

    def fit(self, *args, **kwargs):

        result = super(QMFindingsClassifier, self).fit(*args, **kwargs)
        self.qm1.weights[0].assign(self.qm1.weights[0] / self.num_samples)
        self.qm2.weights[0].assign(self.qm2.weights[0] / self.num_samples)
        self.qm.weights[0].assign(self.qm.weights[0] / self.num_samples)
        return result

    def get_rho(self):     
        return self.weights[2:5]

    def get_config(self):

        config = {
            "dim_x": self.dim_x,
            "dim_y_1": self.dim_y_1,
            "dim_y_2": self.dim_y_2,
            "dim_y": self.dim_y
        }
        base_config = super().get_config()
        return {**base_config, **config}
        
class QMFindingsClassifierSGD(tf.keras.Model):
    """
    A Quantum Measurement Classifier model trainable using
    gradient descent.

    Arguments:
        input_dim: dimension of the input
        dim_x: dimension of the input quantum feature map
        dim_y: dimension of the output representation
        num_eig: Number of eigenvectors used to represent the density matrix. 
                 a value of 0 or less implies num_eig = dim_x
        gamma: float. Gamma parameter of the RBF kernel to be approximated.
        random_state: random number generator seed.
    """
    def __init__(self, input_dim, dim_x, dim_y_1, dim_y_2, dim_y, num_eig_1=0, num_eig_2=0, num_eig=0, gamma=1, random_state=None):
        #print("**********init****************")
        super(QMFindingsClassifierSGD, self).__init__()
        self.fm_x = layers.QFeatureMapRFF(
            input_dim=input_dim,
            dim=dim_x, gamma=gamma, random_state=random_state)
        self.qm_1 = layers.QMeasureClassifEig(dim_x=dim_x, dim_y=dim_y_1, num_eig=num_eig_1)
        self.qm_2 = layers.QMeasureClassifEig(dim_x=dim_x, dim_y=dim_y_2, num_eig=num_eig_2)
        self.qm = layers.QMeasureClassifEig(dim_x=4, dim_y=dim_y, num_eig=num_eig)
        self.dm2dist_1 = layers.DensityMatrix2Dist()
        self.dm2dist = layers.DensityMatrix2Dist()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_y_1 = dim_y_1
        self.dim_y_2 = dim_y_2
        self.gamma = gamma
        self.random_state = random_state
        
        self.cp = layers.CrossProduct()

    def call(self, inputs):
        #print("**********call****************")
        psi_x_1 = self.fm_x(inputs)
        #psi_x_2 = self.fm_x(inputs)
              
        rho_y_1 = self.qm_1(psi_x_1)
        rho_y_2 = self.qm_2(psi_x_1)
        
        psi_x_a = self.cp([rho_y_1, rho_y_2])
        psi_x_a = tf.reshape(psi_x_a, [-1, 4, 4])
        psi_x_a = self.dm2dist_1(psi_x_a)
   
        rho_y = self.qm(psi_x_a)
        
        probs = self.dm2dist(rho_y)
        return probs
        
    def set_rho_1(self, rho_1):
        return self.qm_1.set_rho(rho_1)
        
    def set_rho_2(self, rho_2):
        return self.qm_2.set_rho(rho_2)

    def set_rho(self, rho):
        return self.qm.set_rho(rho)

    def get_config(self):   
        config = {
            "dim_x": self.dim_x,
            "dim_y_1": self.dim_y_1,
            "dim_y_2": self.dim_y_2,
            "dim_y": self.dim_y,
            "num_eig_1": self.num_eig_1,
            "num_eig_2": self.num_eig_2,
            "num_eig": self.num_eig,
            "gamma": self.gamma,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}       
        
class QMKDFindingsClassifier(tf.keras.Model):
    """
    A Quantum Measurement Kernel Density Classifier model.
    Arguments:
        fm_x: Quantum feature map layer for inputs
        dim_x: dimension of the input quantum feature map
        num_classes_1: int number of classes 1
        num_classes_2: int number of classes 2
        num_classes: int number of classes
    """
    def __init__(self, fm_x, dim_x, num_classes_1=2, num_classes_2=2, num_classes=2):
        super(QMKDFindingsClassifier, self).__init__()
        self.fm_x = fm_x
        self.dim_x = dim_x
        self.num_classes_1 = num_classes_1
        self.num_classes_2 = num_classes_2
        self.num_classes = num_classes
        self.qmd_1 = []
        for _ in range(num_classes_1):
            self.qmd_1.append(layers.QMeasureDensity(dim_x))
        self.qmd_2 = []
        for _ in range(num_classes_2):
            self.qmd_2.append(layers.QMeasureDensity(dim_x))
        self.qmd = []    
        for _ in range(num_classes):
            self.qmd.append(layers.QMeasureDensity(4))        
        self.cp = layers.CrossProduct()
        self.cp1 = layers.CrossProduct()
        self.cp2 = layers.CrossProduct()
        self.num_samples = tf.Variable(
            initial_value=tf.zeros((num_classes,)),
            trainable=False
            )

    def call(self, inputs):
        psi_x_1 = self.fm_x(inputs)
        #psi_x_2 = self.fm_x(inputs)
        probs_1 = []
        probs_2 = []
        for i in range(self.num_classes_1):
            probs_1.append(self.qmd_1[i](psi_x_1))
        for i in range(self.num_classes_2):
            probs_2.append(self.qmd_2[i](psi_x_1))
        posteriors_1 = tf.stack(probs_1, axis=-1)
        posteriors_1 = posteriors_1 / tf.expand_dims(tf.reduce_sum(posteriors_1, axis=-1), axis=-1)
        
        posteriors_2 = tf.stack(probs_2, axis=-1)
        posteriors_2 = posteriors_2 / tf.expand_dims(tf.reduce_sum(posteriors_2, axis=-1), axis=-1)
                
        psi_x_a = self.cp([posteriors_1, posteriors_2])
        psi_x_a = tf.reshape(psi_x_a, [-1, 4])
                
        probs = []
        for i in range(self.num_classes):
            probs.append(self.qmd[i](psi_x_a))
            
        posteriors = tf.stack(probs, axis=-1)
        posteriors = posteriors / tf.expand_dims(tf.reduce_sum(posteriors, axis=-1), axis=-1)
        
        return posteriors
        
        

    @tf.function
    def call_train_1_2(self, x, y_tupla_1_2):
       
        y_1, y_2 = y_tupla_1_2
    
        if not self.qmd_1[0].built:
            self.call(x)
            
        psi_x = self.fm_x(x) # shape (bs, dim_x)
        
        rho = self.cp1([psi_x, tf.math.conj(psi_x)]) # shape (bs, dim_x, dim_x)
        
        ohy_1 = tf.keras.backend.one_hot(y_1, self.num_classes)
        ohy_1 = tf.reshape(ohy_1, (-1, self.num_classes))
        
        ohy_1 = tf.expand_dims(ohy_1, axis=-1) 
        ohy_1 = tf.expand_dims(ohy_1, axis=-1) # shape (bs, num_classes, 1, 1)
        rhos_1 = ohy_1 * tf.expand_dims(rho, axis=1) # shape (bs, num_classes, dim_x, dim_x)
        rhos_1 = tf.reduce_sum(rhos_1, axis=0) # shape (num_classes, dim_x, dim_x)
        
        ohy_2 = tf.keras.backend.one_hot(y_2, self.num_classes)
        ohy_2 = tf.reshape(ohy_2, (-1, self.num_classes))
        
        ohy_2 = tf.expand_dims(ohy_2, axis=-1) 
        ohy_2 = tf.expand_dims(ohy_2, axis=-1) # shape (bs, num_classes, 1, 1)
        rhos_2 = ohy_2 * tf.expand_dims(rho, axis=1) # shape (bs, num_classes, dim_x, dim_x)
        rhos_2 = tf.reduce_sum(rhos_2, axis=0) # shape (num_classes, dim_x, dim_x)
        
        num_samples = tf.squeeze(tf.reduce_sum(ohy_1, axis=0))
        self.num_samples.assign_add(num_samples)
        
        return rhos_1, rhos_2

    @tf.function
    def call_train(self, x, y):
 
        if not self.qmd[0].built:
            self.call(x)
        psi = self.fm_x(x) # shape (bs, dim_x)
                      
        probs_1 = []
        probs_2 = []
        for i in range(self.num_classes_1):
            probs_1.append(self.qmd_1[i](psi))
        for i in range(self.num_classes_2):
            probs_2.append(self.qmd_2[i](psi))
        posteriors_1 = tf.stack(probs_1, axis=-1)
        posteriors_1 = posteriors_1 / tf.expand_dims(tf.reduce_sum(posteriors_1, axis=-1), axis=-1)
        
        posteriors_2 = tf.stack(probs_2, axis=-1)
        posteriors_2 = posteriors_2 / tf.expand_dims(tf.reduce_sum(posteriors_2, axis=-1), axis=-1)
        
        psi_x_a = self.cp([posteriors_1, posteriors_2])
        psi_x_a = tf.reshape(psi_x_a, [-1, 4])
                
        rho = self.cp2([psi_x_a, tf.math.conj(psi_x_a)]) # shape (bs, dim_x, dim_x)
        ohy = tf.keras.backend.one_hot(y, self.num_classes)
        ohy = tf.reshape(ohy, (-1, self.num_classes))
        num_samples = tf.squeeze(tf.reduce_sum(ohy, axis=0))
        ohy = tf.expand_dims(ohy, axis=-1) 
        ohy = tf.expand_dims(ohy, axis=-1) # shape (bs, num_classes, 1, 1)
        rhos = ohy * tf.expand_dims(rho, axis=1) # shape (bs, num_classes, dim_x, dim_x)
        rhos = tf.reduce_sum(rhos, axis=0) # shape (num_classes, dim_x, dim_x)
        self.num_samples.assign_add(num_samples)
        return rhos
        
    @tf.function
    def train_step(self, data):

        x, y_tupla = data
        y_1, y_2, y = y_tupla      
        y_tupla_1_2 = (y_1, y_2)
        
        rhos_1, rhos_2 = self.call_train_1_2(x, y_tupla_1_2)
       
        for i in range(self.num_classes_1):
            self.qmd_1[i].weights[0].assign_add(rhos_1[i])
                
        for i in range(self.num_classes_2):
            self.qmd_2[i].weights[0].assign_add(rhos_2[i])
                
        rhos = self.call_train(x, y)
        
        for i in range(self.num_classes):
            self.qmd[i].weights[0].assign_add(rhos[i])
        return {}

    def fit(self, *args, **kwargs):
        
        result = super(QMKDFindingsClassifier, self).fit(*args, **kwargs)
        for i in range(self.num_classes_1):
            self.qmd_1[i].weights[0].assign(self.qmd_1[i].weights[0] /
                                          self.num_samples[i])
        for i in range(self.num_classes_2):
            self.qmd_2[i].weights[0].assign(self.qmd_2[i].weights[0] /
                                          self.num_samples[i])                                  
        for i in range(self.num_classes):
            self.qmd[i].weights[0].assign(self.qmd[i].weights[0] /
                                          self.num_samples[i])
        return result

    def get_rhos(self):
        weights_1 = [qmd.weights[0] for qmd in self.qmd_1]
        weights_2 = [qmd.weights[0] for qmd in self.qmd_2]
        weights = [qmd.weights[0] for qmd in self.qmd]
        return (weights_1, weights_2, weights)

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "num_classes_1": self.num_classes,
            "num_classes_2": self.num_classes,
            "num_classes": self.num_classes
        }
        base_config = super().get_config()
        return {**base_config, **config}   
        
class QMKDFindingsClassifierSGD(tf.keras.Model):
    """
    A Quantum Measurement Kernel Density Classifier model trainable using
    gradient descent.

    Arguments:
        input_dim: dimension of the input
        dim_x: dimension of the input quantum feature map
        num_classes_1: int number of classes 1
        num_classes_2: int number of classes 2
        num_classes: number of classes
        num_eig_1: Number of eigenvectors used to represent the density matrix 1. 
                 a value of 0 or less implies num_eig_1 = dim_x
        num_eig_2: Number of eigenvectors used to represent the density matrix 2. 
                 a value of 0 or less implies num_eig_2 = dim_x
        num_eig: Number of eigenvectors used to represent the density matrix. 
                 a value of 0 or less implies num_eig = dim_x
        gamma: float. Gamma parameter of the RBF kernel to be approximated
        random_state: random number generator seed
    """
    def __init__(self, input_dim, dim_x, num_classes, num_classes_1, num_classes_2, num_eig_1=0, num_eig_2=0, num_eig=0, gamma=1, random_state=None):
        super(QMKDFindingsClassifierSGD, self).__init__()
        self.fm_x = layers.QFeatureMapRFF(
            input_dim=input_dim,
            dim=dim_x, gamma=gamma, random_state=random_state)
        self.dim_x = dim_x
        self.num_classes_1 = num_classes_1
        self.num_classes_2 = num_classes_2
        self.num_classes = num_classes
        self.qmd_1 = []
        for _ in range(num_classes_1):
            self.qmd_1.append(layers.QMeasureDensityEig(dim_x, num_eig_1))
        self.qmd_2 = []
        for _ in range(num_classes_2):
            self.qmd_2.append(layers.QMeasureDensityEig(dim_x, num_eig_2))    
        self.qmd = []
        for _ in range(num_classes):
            self.qmd.append(layers.QMeasureDensityEig(4, num_eig))
        self.gamma = gamma
        self.cp = layers.CrossProduct()
        self.random_state = random_state

    def call(self, inputs):
        psi_x_1 = self.fm_x(inputs)
        #psi_x_2 = self.fm_x(inputs)
        probs_1 = []
        probs_2 = []
        for i in range(self.num_classes_1):
            probs_1.append(self.qmd_1[i](psi_x_1))
        for i in range(self.num_classes_2):
            probs_2.append(self.qmd_2[i](psi_x_1))
        posteriors_1 = tf.stack(probs_1, axis=-1)
        posteriors_1 = posteriors_1 / tf.expand_dims(tf.reduce_sum(posteriors_1, axis=-1), axis=-1)
        
        posteriors_2 = tf.stack(probs_2, axis=-1)
        posteriors_2 = posteriors_2 / tf.expand_dims(tf.reduce_sum(posteriors_2, axis=-1), axis=-1)
                
        psi_x_a = self.cp([posteriors_1, posteriors_2])
        psi_x_a = tf.reshape(psi_x_a, [-1, 4])
                
        probs = []
        for i in range(self.num_classes):
            probs.append(self.qmd[i](psi_x_a))
            
        posteriors = tf.stack(probs, axis=-1)
        posteriors = posteriors / tf.expand_dims(tf.reduce_sum(posteriors, axis=-1), axis=-1)
        return posteriors

    def set_rhos_1(self, rhos):
        for i in range(self.num_classes_1):
            self.qmd_1[i].set_rho(rhos[i])
        return

    def set_rhos_2(self, rhos):
        for i in range(self.num_classes_2):
            self.qmd_2[i].set_rho(rhos[i])
        return

    def set_rhos(self, rhos):
        for i in range(self.num_classes):
            self.qmd[i].set_rho(rhos[i])
        return

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "num_classes_1": self.num_classes_1,
            "num_classes_2": self.num_classes_2,
            "num_classes": self.num_classes,
            "num_eig_1": self.num_eig_1,
            "num_eig_2": self.num_eig_2,
            "num_eig": self.num_eig,
            "gamma": self.gamma,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}             
        
class QMKDFindingsClassifier_2(tf.keras.Model):
    """
    A Quantum Measurement Kernel Density Classifier model.
    Arguments:
        fm_x: Quantum feature map layer for inputs
        fm_x_2: Quantum feature map layer for inputs 2
        dim_x: dimension of the input quantum feature map
        dim_x_2: dimension of the input quantum feature map 2
        num_classes_1: int number of classes 1
        num_classes_2: int number of classes 2
        num_classes: int number of classes
    """
    def __init__(self, fm_x, fm_x_2, dim_x, dim_x_2, num_classes_1=2, num_classes_2=2, num_classes=2):
        super(QMKDFindingsClassifier_2, self).__init__()
        self.fm_x = fm_x
        
        self.fm_x_2 = fm_x_2 #Nuevo
        
        self.dim_x = dim_x
        
        self.dim_x_2 = dim_x_2 #Nuevo
        
        self.num_classes_1 = num_classes_1
        self.num_classes_2 = num_classes_2
        self.num_classes = num_classes
        
        self.qmd_1 = []
        for _ in range(num_classes_1):
            self.qmd_1.append(layers.QMeasureDensity(dim_x))
        self.qmd_2 = []
        for _ in range(num_classes_2):
            self.qmd_2.append(layers.QMeasureDensity(dim_x_2))
        self.qmd = []    
        for _ in range(num_classes):
            self.qmd.append(layers.QMeasureDensity(4))   
                 
        self.cp = layers.CrossProduct()
        self.cp1 = layers.CrossProduct()
        self.cp2 = layers.CrossProduct()
        self.num_samples = tf.Variable(
            initial_value=tf.zeros((num_classes,)),
            trainable=False
            )

    def call(self, inputs):
        #print('--------------call-----------------')
        #print('---------inputs----------', inputs.shape)
        inputs_1 = tf.slice(inputs, [0,0], [-1,2048])
        inputs_2 = tf.slice(inputs, [0,2048], [-1,-1])
        #print('---------inputs_1----------', inputs_1.shape)
        #print('---------inputs_2----------', inputs_2.shape)
        psi_x_1 = self.fm_x(inputs_1)
        psi_x_2 = self.fm_x_2(inputs_2) #Nuevo, falta arraglar porque no recibe las mismas inputs que fm_x
        #print('---------psi_x_1----------', psi_x_1.shape)
        #print('---------psi_x_2----------', psi_x_2.shape)
        probs_1 = []
        probs_2 = []
        for i in range(self.num_classes_1):
            probs_1.append(self.qmd_1[i](psi_x_1))
        #print('-------------listo_el_qmd_1-------------')    
        for i in range(self.num_classes_2):
            probs_2.append(self.qmd_2[i](psi_x_2))#aquí entra el nuevo psi_x_2
        #print('-------------listo_el_qmd_2-------------')
        posteriors_1 = tf.stack(probs_1, axis=-1)
        posteriors_1 = posteriors_1 / tf.expand_dims(tf.reduce_sum(posteriors_1, axis=-1), axis=-1)
        
        posteriors_2 = tf.stack(probs_2, axis=-1)
        posteriors_2 = posteriors_2 / tf.expand_dims(tf.reduce_sum(posteriors_2, axis=-1), axis=-1)
                
        psi_x_a = self.cp([posteriors_1, posteriors_2])
        psi_x_a = tf.reshape(psi_x_a, [-1, 4])
                
        probs = []
        for i in range(self.num_classes):
            probs.append(self.qmd[i](psi_x_a))
            
        posteriors = tf.stack(probs, axis=-1)
        posteriors = posteriors / tf.expand_dims(tf.reduce_sum(posteriors, axis=-1), axis=-1)
        
        return posteriors

    @tf.function
    def call_train_1_2(self, x, y_tupla_1_2):
        #print('--------------call_train_1_2-----------------')
        #print('--------------x-----------------', x.shape)
        x_1 = tf.slice(x, [0,0], [-1,2048])
        x_2 = tf.slice(x, [0,2048], [-1,-1])
        
        y_1, y_2 = y_tupla_1_2
        #print('--------------x-----------------', x.shape)
        if not self.qmd_1[0].built:
            self.call(x)
            
        psi_x = self.fm_x(x_1) # shape (bs, dim_x)
        
        psi_x_2 = self.fm_x_2(x_2) # shape (bs, dim_x)  #Nuevo, falta arreglar porque recibe algo diferente a fm_x
        
        rho = self.cp1([psi_x, tf.math.conj(psi_x)]) # shape (bs, dim_x, dim_x)
        
        rho_2 = self.cp1([psi_x_2, tf.math.conj(psi_x_2)]) # Nuevo
        
        ohy_1 = tf.keras.backend.one_hot(y_1, self.num_classes_1)
        ohy_1 = tf.reshape(ohy_1, (-1, self.num_classes_1))
        
        ohy_1 = tf.expand_dims(ohy_1, axis=-1) 
        ohy_1 = tf.expand_dims(ohy_1, axis=-1) # shape (bs, num_classes, 1, 1)
        rhos_1 = ohy_1 * tf.expand_dims(rho, axis=1) # shape (bs, num_classes, dim_x, dim_x)
        rhos_1 = tf.reduce_sum(rhos_1, axis=0) # shape (num_classes, dim_x, dim_x)
        
        ohy_2 = tf.keras.backend.one_hot(y_2, self.num_classes_2)
        ohy_2 = tf.reshape(ohy_2, (-1, self.num_classes_2))
        
        ohy_2 = tf.expand_dims(ohy_2, axis=-1) 
        ohy_2 = tf.expand_dims(ohy_2, axis=-1) # shape (bs, num_classes, 1, 1)
        rhos_2 = ohy_2 * tf.expand_dims(rho_2, axis=1) # shape (bs, num_classes, dim_x, dim_x)
        rhos_2 = tf.reduce_sum(rhos_2, axis=0) # shape (num_classes, dim_x, dim_x)
        
        num_samples = tf.squeeze(tf.reduce_sum(ohy_1, axis=0))
        self.num_samples.assign_add(num_samples)
        
        return rhos_1, rhos_2

    @tf.function
    def call_train(self, x, y):
        #print('--------------call_train-----------------')
        x_1 = tf.slice(x, [0,0], [-1,2048])
        x_2 = tf.slice(x, [0,2048], [-1,-1])
 
        if not self.qmd[0].built:
            self.call(x)
            
        psi = self.fm_x(x_1) # shape (bs, dim_x)
        
        psi_2 = self.fm_x_2(x_2) # shape (bs, dim_x) #Nuevo, falta arreglarlo
                      
        probs_1 = []
        probs_2 = []
        for i in range(self.num_classes_1):
            probs_1.append(self.qmd_1[i](psi))
        for i in range(self.num_classes_2):
            probs_2.append(self.qmd_2[i](psi_2))
        posteriors_1 = tf.stack(probs_1, axis=-1)
        posteriors_1 = posteriors_1 / tf.expand_dims(tf.reduce_sum(posteriors_1, axis=-1), axis=-1)
        
        posteriors_2 = tf.stack(probs_2, axis=-1)
        posteriors_2 = posteriors_2 / tf.expand_dims(tf.reduce_sum(posteriors_2, axis=-1), axis=-1)
        
        psi_x_a = self.cp([posteriors_1, posteriors_2])
        psi_x_a = tf.reshape(psi_x_a, [-1, 4])
                
        rho = self.cp2([psi_x_a, tf.math.conj(psi_x_a)]) # shape (bs, dim_x, dim_x)
        ohy = tf.keras.backend.one_hot(y, self.num_classes)
        ohy = tf.reshape(ohy, (-1, self.num_classes))
        num_samples = tf.squeeze(tf.reduce_sum(ohy, axis=0))
        ohy = tf.expand_dims(ohy, axis=-1) 
        ohy = tf.expand_dims(ohy, axis=-1) # shape (bs, num_classes, 1, 1)
        rhos = ohy * tf.expand_dims(rho, axis=1) # shape (bs, num_classes, dim_x, dim_x)
        rhos = tf.reduce_sum(rhos, axis=0) # shape (num_classes, dim_x, dim_x)
        self.num_samples.assign_add(num_samples)
        return rhos
        
    @tf.function
    def train_step(self, data):
        #print('--------------train_step-----------------')
        x, y_tupla = data
        y_1, y_2, y = y_tupla      
        y_tupla_1_2 = (y_1, y_2)
        #print('--------------x-----------------', x.shape)
        rhos_1, rhos_2 = self.call_train_1_2(x, y_tupla_1_2)
       
        for i in range(self.num_classes_1):
            self.qmd_1[i].weights[0].assign_add(rhos_1[i])
                
        for i in range(self.num_classes_2):
            self.qmd_2[i].weights[0].assign_add(rhos_2[i])
                
        rhos = self.call_train(x, y)
        
        for i in range(self.num_classes):
            self.qmd[i].weights[0].assign_add(rhos[i])
        return {}

    def fit(self, *args, **kwargs):
        #print('--------------fit-----------------')
        result = super(QMKDFindingsClassifier_2, self).fit(*args, **kwargs)
        for i in range(self.num_classes_1):
            self.qmd_1[i].weights[0].assign(self.qmd_1[i].weights[0] /
                                          self.num_samples[i])
        for i in range(self.num_classes_2):
            self.qmd_2[i].weights[0].assign(self.qmd_2[i].weights[0] /
                                          self.num_samples[i])                                  
        for i in range(self.num_classes):
            self.qmd[i].weights[0].assign(self.qmd[i].weights[0] /
                                          self.num_samples[i])
        return result
        
    def set_rhos_1(self, rhos):
        for i in range(self.num_classes_1):
            self.qmd_1[i].set_rho(rhos[i])
        return

    def set_rhos_2(self, rhos):
        for i in range(self.num_classes_2):
            self.qmd_2[i].set_rho(rhos[i])
        return

    def set_rhos(self, rhos):
        for i in range(self.num_classes):
            self.qmd[i].set_rho(rhos[i])
        return

    def get_rhos(self):
        weights_1 = [qmd.weights[0] for qmd in self.qmd_1]
        weights_2 = [qmd.weights[0] for qmd in self.qmd_2]
        weights = [qmd.weights[0] for qmd in self.qmd]
        return (weights_1, weights_2, weights)

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "dim_x_2": self.dim_x_2,
            "num_classes_1": self.num_classes_1,
            "num_classes_2": self.num_classes_2,
            "num_classes": self.num_classes
        }
        base_config = super().get_config()
        return {**base_config, **config}   
        
class QMKDFindingsClassifierSGD_2(tf.keras.Model):
    """
    A Quantum Measurement Kernel Density Classifier model trainable using
    gradient descent.

    Arguments:
        input_dim: dimension of the input
        dim_x: dimension of the input quantum feature map
        num_classes_1: int number of classes 1
        num_classes_2: int number of classes 2
        num_classes: number of classes
        num_eig_1: Number of eigenvectors used to represent the density matrix 1. 
                 a value of 0 or less implies num_eig_1 = dim_x
        num_eig_2: Number of eigenvectors used to represent the density matrix 2. 
                 a value of 0 or less implies num_eig_2 = dim_x
        num_eig: Number of eigenvectors used to represent the density matrix. 
                 a value of 0 or less implies num_eig = dim_x
        gamma: float. Gamma parameter of the RBF kernel to be approximated
        random_state: random number generator seed
    """
    def __init__(self, input_dim, input_dim_2, dim_x, dim_x_2, num_classes_1, num_classes_2, num_classes, num_eig_1=0, num_eig_2=0, num_eig=0, gamma=1, gamma_2 = 1, random_state=None):
    
        super(QMKDFindingsClassifierSGD_2, self).__init__()
        self.fm_x = layers.QFeatureMapRFF(
            input_dim=input_dim,
            dim=dim_x, gamma=gamma, random_state=random_state)
        self.fm_x_2 = layers.QFeatureMapRFF(
            input_dim=input_dim_2,
            dim=dim_x_2, gamma=gamma_2, random_state=random_state)
        self.dim_x = dim_x
        self.dim_x_2 = dim_x_2
        self.num_classes_1 = num_classes_1
        self.num_classes_2 = num_classes_2
        self.num_classes = num_classes
        self.qmd_1 = []
        for _ in range(num_classes_1):
            self.qmd_1.append(layers.QMeasureDensityEig(dim_x, num_eig_1))
        self.qmd_2 = []
        for _ in range(num_classes_2):
            self.qmd_2.append(layers.QMeasureDensityEig(dim_x_2, num_eig_2))    
        self.qmd = []
        for _ in range(num_classes):
            self.qmd.append(layers.QMeasureDensityEig(4, num_eig))
        self.gamma = gamma
        self.gamma_2 = gamma_2
        self.cp = layers.CrossProduct()
        self.random_state = random_state

    def call(self, inputs):
        #print('--------------call-----------------')
        #print('---------inputs----------', inputs.shape)
        inputs_1 = tf.slice(inputs, [0,0], [-1,2048])
        inputs_2 = tf.slice(inputs, [0,2048], [-1,-1])
        #print('---------inputs_1----------', inputs_1.shape)
        #print('---------inputs_2----------', inputs_2.shape)
        psi_x_1 = self.fm_x(inputs_1)
        psi_x_2 = self.fm_x_2(inputs_2) #Nuevo, falta arraglar porque no recibe las mismas inputs que fm_x
        #print('---------psi_x_1----------', psi_x_1.shape)
        #print('---------psi_x_2----------', psi_x_2.shape)
        probs_1 = []
        probs_2 = []
        for i in range(self.num_classes_1):
            probs_1.append(self.qmd_1[i](psi_x_1))
        #print('-------------listo_el_qmd_1-------------')    
        for i in range(self.num_classes_2):
            probs_2.append(self.qmd_2[i](psi_x_2))#aquí entra el nuevo psi_x_2
        #print('-------------listo_el_qmd_2-------------')
        posteriors_1 = tf.stack(probs_1, axis=-1)
        posteriors_1 = posteriors_1 / tf.expand_dims(tf.reduce_sum(posteriors_1, axis=-1), axis=-1)
        
        posteriors_2 = tf.stack(probs_2, axis=-1)
        posteriors_2 = posteriors_2 / tf.expand_dims(tf.reduce_sum(posteriors_2, axis=-1), axis=-1)
                
        psi_x_a = self.cp([posteriors_1, posteriors_2])
        psi_x_a = tf.reshape(psi_x_a, [-1, 4])
                
        probs = []
        for i in range(self.num_classes):
            probs.append(self.qmd[i](psi_x_a))
            
        posteriors = tf.stack(probs, axis=-1)
        posteriors = posteriors / tf.expand_dims(tf.reduce_sum(posteriors, axis=-1), axis=-1)
        
        return posteriors

    def set_rhos_1(self, rhos):
        for i in range(self.num_classes_1):
            self.qmd_1[i].set_rho(rhos[i])
        return

    def set_rhos_2(self, rhos):
        for i in range(self.num_classes_2):
            self.qmd_2[i].set_rho(rhos[i])
        return

    def set_rhos(self, rhos):
        for i in range(self.num_classes):
            self.qmd[i].set_rho(rhos[i])
        return
        
    def get_rhos(self):
        weights_1 = [qmd.weights[0] for qmd in self.qmd_1]
        weights_2 = [qmd.weights[0] for qmd in self.qmd_2]
        weights = [qmd.weights[0] for qmd in self.qmd]
        return (weights_1, weights_2, weights)

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "dim_x_2": self.dim_x_2,
            "num_classes_1": self.num_classes_1,
            "num_classes_2": self.num_classes_2,
            "num_classes": self.num_classes,
            "num_eig_1": self.num_eig_1,
            "num_eig_2": self.num_eig_2,
            "num_eig": self.num_eig,
            "gamma": self.gamma,
            "gamma_2": self.gamma_2,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}                            
