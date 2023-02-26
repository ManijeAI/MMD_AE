import tensorflow as tf
from Model.metric import *
from tensorflow import keras
from numpy.random import seed
from keras import backend as K
import matplotlib.pyplot as plt

# for randomnes
seed(3)
tf.random.set_seed(3)

class AE():
    def __init__(self, input_shape, layers, lambda_, plot=0):
        self.input_shape = input_shape
        self.layers = layers
        self.lambda_ = lambda_
        self.plot = plot


    def encoder_decoder(self, layers, activation, name="_"):
        # Create encoder
        input_e = keras.Input(shape=(self.input_shape,), name = 'encoder_input_Iemocap')
        x1_e = layers.Dense(layers[0], activation=activation, kernel_initializer='glorot_uniform', bias_initializer='zeros', name= 'en_layer_1_ie')(input_e)
        x2_e= layers.Dense(layers[1], activation=activation, kernel_initializer='glorot_uniform', bias_initializer='zeros', name= 'en_layer_2_ie')(x1_e)
        z_ = layers.Dense(layers[2],activation='tanh', name='z_ie')(x2_e)
        encoder = keras.Model(input, z_, name='encoder'+str(name))

        # Create decoder
        latent_inputs = keras.Input(shape=(layers[2],), name='z_Iemocap')
        x1_d = layers.Dense(layers[1], activation=activation, kernel_initializer='glorot_uniform', name='dec_input_Iemocap')(latent_inputs)
        x2_d = layers.Dense(layers[0], activation=activation, kernel_initializer='glorot_uniform', name='dec_layer_2_ie')(x1_d)
        outputs =layers.Dense(self.input_shape, activation='linear', name='out_layer_ie', kernel_initializer='glorot_uniform', bias_initializer='zeros')(x2_d)
        decoder = keras.Model(latent_inputs, outputs, name='decoder_ie')

        # instantiate AE model (IEMOCAP)
        outputs = decoder(encoder(input_e))

        return [input_e, outputs, z_, x2_e]

    def reconstruction_loss(self, input, output):
        return K.mean(K.sum(K.pow(input - output, 2), axis=-1))


    def MMD_loss(self, source, target, sigma=0.25):
        mmd = MMD(source, target, kernel="RBF", sigma=sigma)
        return mmd.compute_mmd()


    def create_AE(self):
        source_input, source_output, z_source, layer_source = self.encoder_decoder(self.self.layers, activation='tanh')
        target_input, target_output, z_Target, layer_target = self.encoder_decoder(self.self.layers, activation='tanh')

        #_____________________ loss ________________
        # reconstruction_loss
        reconstruction_loss_source =self.reconstruction_loss(source_input, source_output)
        reconstruction_loss_target = self.reconstruction_loss(target_input, target_output)

        # MMD loss
        buttleneck_mmd_loss = self.MMD_loss(z_source, z_Target)
        multilayer_mmd_loss = self.MMD_loss(layer_source, layer_target)
        total_loss = self.lambda_ * (reconstruction_loss_source + reconstruction_loss_target) + buttleneck_mmd_loss + multilayer_mmd_loss

        #_____________________ create_network ________________

        Ae = keras.Model(inputs=[source_input, target_input], outputs=[source_output, target_output], name="ae_tow_net")
        if (self.plot == 1):
            print(Ae.summary(),"\n\n")

        #_____________________ add_optim_loss_to_network ________________

        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.999, beta_2=0.99, epsilon=0.00000001)
        Ae.add_loss(total_loss)
        Ae.compile(optimizer=adam)

        return AE


    def fit(self, source_data, target_data, epochs = 1, batch_size=1, validation_split=0.05):
        Ae = self.create_AE()
        #__________________fit_model________________
        print("training...")
        history = Ae.fit(x={'encoder_input_Iemocap':source_data, 'encoder_input_Emodb':target_data},
                        # y={'decoder_ie':source_train_lbl,'decoder_em':target_train_lbl},
                        y= None,
                        verbose=0, epochs=epochs , batch_size=batch_size, validation_split=validation_split)

        print(history.history['loss'])
        plt.figure(figsize=(5, 3))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        print("finished")

        #__________________return_ae_encoders________
        encoder_source, encoder_target = Ae.layers['encoder_source'], Ae.layers['encoder_target']
        return encoder_source, encoder_target, Ae