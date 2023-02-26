import tensorflow as tf
import tensorflow_probability as tfp
import sklearn.metrics.pairwise

# for randomnes
from numpy.random import seed
seed(3)
tf.random.set_seed(3)

class MMD():
    def __init__(self, x, y, kernel='RBF', sigma=0.25):
        self.x = x
        self.y = y
        self.kernel = kernel
        self.sigma = sigma

    def RBF_kernel(sefl,x, y, sigma=0.25):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
        tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
        # kernel = tf.exp(-(tiled_x - tiled_y )** 2 / (2.0 * sigma ** 2))
        return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) /  (2.0 * sigma ** 2))

    def poly_kernel(self, x,y):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
        tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))  

        poly = tfp.math.psd_kernels.Polynomial(
            bias_variance=None, slope_variance=None, shift=None, exponent=2,
            feature_ndims=1, validate_args=False, parameters=None,
            name='Polynomial')
        res = poly.apply(tiled_x,tiled_y)
        return res

    def compute_mmd(self):
        if self.kernel == "RBF":
            x_kernel = self.RBF_kernel(self.x, self.x, self.sigma)
            y_kernel = self.RBF_kernel(self.y, self.y, self.sigma)
            xy_kernel =self.RBF_kernel(self.x, self.y, self.sigma)
        elif self.kernel == "polynomial":
            x_kernel = self.poly_kernel(self.x, self.x)
            y_kernel = self.poly_kernel(self.y, self.y)
            xy_kernel =self.poly_kernel(self.x, self.y)

        return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)
