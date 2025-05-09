import tensorflow.python.keras.backend as KB
import tensorflow as tf
from hyper_parameters import *


def multiclass_balanced_cross_entropy_loss(y_true, y_pred):
    # TODO: Check behavior when all classes are not present
    # shape = KB.int_shape(y_pred)
    batch_size = BATCH_SIZE
    num_classes = 4

    y_pred_ = KB.clip(y_pred, KB.epsilon(), 1. - KB.epsilon())

    cross_ent = (KB.log(y_pred_) * y_true)
    cross_ent = KB.sum(cross_ent, axis=-2, keepdims=False)
    cross_ent = KB.sum(cross_ent, axis=-2, keepdims=False)
    cross_ent = KB.reshape(cross_ent, shape=(batch_size, num_classes))

    y_true_ = KB.sum(y_true, axis=-2, keepdims=False)
    y_true_ = KB.sum(y_true_, axis=-2, keepdims=False)
    y_true_ = KB.reshape(y_true_, shape=(batch_size, num_classes)) + KB.ones(shape=(batch_size, num_classes))

    cross_ent = cross_ent / y_true_

    return - KB.mean(cross_ent, axis=-1, keepdims=False)


def multiclass_balanced_cross_entropy_loss_unet(y_true, y_pred):
    # shape = KB.int_shape(y_pred)
    CROP_SHAPE = KB.int_shape(y_pred)
    print(CROP_SHAPE)
    batch_size = BATCH_SIZE
    num_classes = 4

    y_true = tf.image.resize_with_crop_or_pad(y_true, target_height=CROP_SHAPE[1], target_width=CROP_SHAPE[2])
    y_true = tf.cast(y_true, tf.float32)

    y_pred_ = KB.clip(y_pred, KB.epsilon(), 1. - KB.epsilon())


    # Calculate balanced cross-entropy loss
    cross_ent = (KB.log(y_pred_) * y_true)
    cross_ent = KB.sum(cross_ent, axis=-2, keepdims=False)
    cross_ent = KB.sum(cross_ent, axis=-2, keepdims=False)
    cross_ent = KB.reshape(cross_ent, shape=(batch_size, num_classes))

    y_true_ = KB.sum(y_true, axis=-2, keepdims=False)
    y_true_ = KB.sum(y_true_, axis=-2, keepdims=False)
    y_true_ = KB.reshape(y_true_, shape=(batch_size, num_classes)) + KB.epsilon()

    cross_ent = (cross_ent / y_true_)

    # calculate dice loss
    g_0 = y_true[:, :, :, 0]
    p_0 = y_pred_[:, :, :, 0]

    true_pos = KB.sum((1. - p_0) * (1. - g_0), keepdims=False)
    false_pos = KB.sum((1. - p_0) * g_0, keepdims=False)
    false_neg = KB.sum(p_0 * (1. - g_0), keepdims=False)
    dice_loss = 1. - ((2. * true_pos) / (2. * true_pos + false_pos + false_neg + KB.epsilon()))

    return - 0.5 * (KB.mean(cross_ent, axis=-1, keepdims=False)) + 0.5 * dice_loss


def binary_prob(x):
    # TODO : test line 2
    pos = KB.expand_dims(x[:, 1], axis=-1)
    neg = KB.ones((BATCH_SIZE, 1), dtype='float32') - pos
    return KB.concatenate([neg, pos], axis=-1)


def binary_prob_out_shape(input_shape):
    shape = list(input_shape)
    return tuple(shape)


def retouch_dual_net(input_shape=(64, 512, 512, 1)):
    """

    :param input_shape: the shape of the input volume. Please note that this is channel last
    :param nb_classes: 
    :return:
    """

    from keras.layers import Conv2D, SpatialDropout2D, Input, UpSampling2D, \
        AveragePooling2D, GlobalMaxPooling2D, Lambda, LeakyReLU, BatchNormalization, concatenate
    from keras.models import Model
    from keras.utils import plot_model
    from custom_layers import Softmax4D
    from keras.optimizers import SGD

    in_img = Input(shape=input_shape)

    conv0_1 = Conv2D(64, [3, 3], strides=(1, 1), padding='same')(in_img)
    conv0_1 = LeakyReLU(alpha=0.3)(conv0_1)
    conv0_2 = Conv2D(64, [3, 3], strides=(2, 2), use_bias=False, padding='same')(conv0_1)
    conv0_2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                 beta_initializer='zeros', gamma_initializer='ones',
                                 moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv0_2)
    conv0_2 = LeakyReLU(alpha=0.3)(conv0_2)

    conv1_1 = Conv2D(128, [3, 3], strides=(1, 1), padding='same')(conv0_2)
    conv1_1 = LeakyReLU(alpha=0.3)(conv1_1)
    conv1_2 = Conv2D(128, [3, 3], strides=(2, 2), use_bias=False, padding='same')(conv1_1)
    conv1_2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                 beta_initializer='zeros', gamma_initializer='ones',
                                 moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv1_2)
    conv1_2 = LeakyReLU(alpha=0.3)(conv1_2)

    conv2_1 = Conv2D(256, [3, 3], strides=(1, 1), padding='same')(conv1_2)
    conv2_1 = LeakyReLU(alpha=0.3)(conv2_1)
    # conv2_2 = Conv2D(256, [3, 3], strides=(1, 1), padding='same')(conv2_1)
    # conv2_2 = LeakyReLU(alpha=0.3)(conv2_2)
    conv2_3 = Conv2D(256, [3, 3], strides=(2, 2), use_bias=False, padding='same')(conv2_1)
    conv2_3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                 beta_initializer='zeros', gamma_initializer='ones',
                                 moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv2_3)
    conv2_3 = LeakyReLU(alpha=0.3)(conv2_3)

    conv3_1 = Conv2D(512, [3, 3], strides=(1, 1), padding='same')(conv2_3)
    conv3_1 = LeakyReLU(alpha=0.3)(conv3_1)
    # conv3_2 = Conv2D(512, [3, 3], strides=(1, 1), padding='same')(conv3_1)
    # conv3_2 = LeakyReLU(alpha=0.3)(conv3_2)
    conv3_3 = Conv2D(512, [3, 3], strides=(2, 2), use_bias=False, padding='same')(conv3_1)
    conv3_3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                 beta_initializer='zeros', gamma_initializer='ones',
                                 moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv3_3)
    conv3_3 = LeakyReLU(alpha=0.3)(conv3_3)

    conv4_1 = Conv2D(1024, [1, 1], strides=(1, 1), use_bias=False, padding='same')(conv3_3)
    conv4_1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                 beta_initializer='zeros', gamma_initializer='ones',
                                 moving_mean_initializer='zeros', moving_variance_initializer='ones')(conv4_1)
    conv4_1 = LeakyReLU(alpha=0.3)(conv4_1)
    conv4_1 = SpatialDropout2D(0.5)(conv4_1)
    # conv4_2 = Conv2D(4096, [1, 1], strides=(1, 1), padding='same')(conv4_1)
    # conv4_2 = SpatialDropout2D(0.5)(conv4_2)

    apool = AveragePooling2D(pool_size=(14, 14), data_format='channels_last')(conv4_1)

    # Slice classification outputs
    sm_IRF = Conv2D(2, [1, 1], strides=(1, 1), padding='same', activation='relu')(apool)
    sm_IRF = Softmax4D(axis=-1)(sm_IRF)
    sm_IRF = GlobalMaxPooling2D(data_format='channels_last')(sm_IRF)
    sm_IRF = Lambda(binary_prob, output_shape=binary_prob_out_shape, name='sm_IRF')(sm_IRF)

    sm_SRF = Conv2D(2, [1, 1], strides=(1, 1), padding='same', activation='relu')(apool)
    sm_SRF = Softmax4D(axis=-1)(sm_SRF)
    sm_SRF = GlobalMaxPooling2D(data_format='channels_last')(sm_SRF)
    sm_SRF = Lambda(binary_prob, output_shape=binary_prob_out_shape, name='sm_SRF')(sm_SRF)

    sm_PED = Conv2D(2, [1, 1], strides=(1, 1), padding='same', activation='relu')(apool)
    sm_PED = Softmax4D(axis=-1)(sm_PED)
    sm_PED = GlobalMaxPooling2D(data_format='channels_last')(sm_PED)
    sm_PED = Lambda(binary_prob, output_shape=binary_prob_out_shape, name='sm_PED')(sm_PED)

    # sm_IRF = Dense(2, activation='softmax', name='sm_IRF')(apool)
    # sm_SRF = Dense(2, activation='softmax', name='sm_SRF')(apool)
    # sm_PED = Dense(2, activation='softmax', name='sm_PED')(apool)

    # Segmentation layers
    seg_1 = UpSampling2D(size=(2, 2))(conv0_2)
    seg_2 = UpSampling2D(size=(4, 4))(conv1_2)
    seg_3 = UpSampling2D(size=(8, 8))(conv2_3)
    # seg_4 = UpSampling2D(size=(16, 16))(conv3_3)
    seg_5 = UpSampling2D(size=(16, 16))(conv4_1)

    concat_1 = concatenate([seg_1, seg_2, seg_3, seg_5], axis=-1)
    seg_out = Conv2D(4, [1, 1], strides=(1, 1), padding='same', activation='relu')(concat_1)
    seg_out = Softmax4D(axis=-1, name='seg_out')(seg_out)

    model = Model(inputs=in_img, outputs=[sm_IRF, sm_SRF, sm_PED, seg_out])

    print (model.summary())
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    sgd = SGD(learning_rate=0.001, momentum=0.5, decay=1e-6, nesterov=False)
    model.compile(optimizer=sgd, loss={'sm_IRF': 'categorical_crossentropy', 'sm_SRF': 'categorical_crossentropy',
                                       'sm_PED': 'categorical_crossentropy',
                                       'seg_out': multiclass_balanced_cross_entropy_loss})

    return model


def retouch_vgg_net(input_shape=(224, 224, 3)):
    from keras.layers import Conv2D, SpatialDropout2D, UpSampling2D, \
        AveragePooling2D, GlobalMaxPooling2D, Lambda, MaxPooling2D, concatenate
    from keras.models import Model
    from custom_layers import Softmax4D
    from keras.optimizers import SGD
    from keras import Input
    from keras.applications.vgg16 import VGG16

    img_input = Input(shape=input_shape)

    # Block 1
    b1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    b1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(b1)
    b1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(b1)

    # Block 2
    b2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(b1)
    b2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(b2)
    b2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(b2)

    # Block 3
    b3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(b2)
    b3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(b3)
    b3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(b3)
    b3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(b3)

    # Block 4
    b4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(b3)
    b4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(b4)
    b4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(b4)
    b4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(b4)

    # Block 5
    b5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(b4)
    b5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(b5)
    b5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(b5)
    b5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(b5)
    b5 = SpatialDropout2D(0.5, name='block5_drop')(b5)

    apool = AveragePooling2D(pool_size=(7, 7), data_format='channels_last', name='apool6')(b5)

    # Slice classification outputs
    sm_IRF = Conv2D(2, (1, 1), strides=(1, 1), padding='same', activation='relu', name='IRF_conv')(apool)
    sm_IRF = Softmax4D(axis=-1, name='IRF_softmax')(sm_IRF)
    sm_IRF = GlobalMaxPooling2D(data_format='channels_last', name='IRF_maxpool')(sm_IRF)
    sm_IRF = Lambda(binary_prob, output_shape=binary_prob_out_shape, name='sm_IRF')(sm_IRF)

    sm_SRF = Conv2D(2, (1, 1), strides=(1, 1), padding='same', activation='relu', name='SRF_conv')(apool)
    sm_SRF = Softmax4D(axis=-1, name='SRF_softmax')(sm_SRF)
    sm_SRF = GlobalMaxPooling2D(data_format='channels_last', name='SRF_maxpool')(sm_SRF)
    sm_SRF = Lambda(binary_prob, output_shape=binary_prob_out_shape, name='sm_SRF')(sm_SRF)

    sm_PED = Conv2D(2, (1, 1), strides=(1, 1), padding='same', activation='relu', name='PED_conv')(apool)
    sm_PED = Softmax4D(axis=-1, name='PED_softmax')(sm_PED)
    sm_PED = GlobalMaxPooling2D(data_format='channels_last', name='PED_maxpool')(sm_PED)
    sm_PED = Lambda(binary_prob, output_shape=binary_prob_out_shape, name='sm_PED')(sm_PED)

    # Segmentation layers
    seg_1 = UpSampling2D(size=(2, 2), name='up1')(b1)
    seg_2 = UpSampling2D(size=(4, 4), name='up2')(b2)
    seg_3 = UpSampling2D(size=(8, 8), name='up3')(b3)
    # seg_4 = UpSampling2D(size=(16, 16), name='up4')(b4)
    seg_5 = UpSampling2D(size=(32, 32), name='up5')(b5)

    concat_1 = concatenate([seg_1, seg_2, seg_3, seg_5], axis=-1, name='seg_concat')
    seg_out = Conv2D(4, [1, 1], strides=(1, 1), padding='same', activation='relu', name='seg_conv')(concat_1)
    seg_out = Softmax4D(axis=-1, name='seg_out')(seg_out)

    model = Model(inputs=img_input, outputs=[sm_IRF, sm_SRF, sm_PED, seg_out])

    base_model = VGG16(weights='imagenet', include_top=False)

    for layer in model.layers:
        if 'block' in layer.name and 'conv' in layer.name:
            vgg_layer = base_model.get_layer(name=layer.name)
            layer.set_weights(vgg_layer.get_weights())
            layer.trainable = False

    model.summary()

    sgd = SGD(learning_rate=0.001, momentum=0.5, decay=1e-6, nesterov=False)
    model.compile(optimizer=sgd, loss={'sm_IRF': 'categorical_crossentropy', 'sm_SRF': 'categorical_crossentropy',
                                       'sm_PED': 'categorical_crossentropy',
                                       'seg_out': multiclass_balanced_cross_entropy_loss})

    return model


def retouch_unet(input_shape=(224, 224, 3), regularize_weight=0.0001):
    from keras.layers import Conv2D, SpatialDropout2D, GlobalAveragePooling2D, UpSampling2D, \
         GlobalMaxPooling2D, MaxPooling2D, Conv2DTranspose, Activation, Cropping2D, BatchNormalization, concatenate
    from keras.models import Model
    from keras.utils import plot_model
    from custom_layers import Softmax4D
    from keras.optimizers import SGD, Adam
    from keras import backend as K
    from keras.regularizers import l2
    from keras import Input
    
    in_image = Input(shape=input_shape)

    conv1_0 = Conv2D(64, (7, 7), activation='relu', name='conv1_0', padding='same', data_format='channels_last',
                     kernel_regularizer=l2(regularize_weight))(
        in_image)
    conv1_1 = Conv2D(64, (3, 3), activation='relu', name='conv1_1', data_format='channels_last',
                     kernel_regularizer=l2(regularize_weight))(conv1_0)
    conv1_2 = Conv2D(64, (3, 3), name='conv1_2', data_format='channels_last', kernel_regularizer=l2(regularize_weight))(
        conv1_1)
    conv1_2 = BatchNormalization(axis=-1, name='bn1')(conv1_2)
    conv1_2 = Activation('relu')(conv1_2)

    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1', data_format='channels_last')(conv1_2)
    conv2_1 = Conv2D(128, (3, 3), activation='relu', name='conv2_1', data_format='channels_last',
                     kernel_regularizer=l2(regularize_weight))(pool1)
    conv2_2 = Conv2D(128, (3, 3), name='conv2_2', data_format='channels_last',
                     kernel_regularizer=l2(regularize_weight))(conv2_1)
    conv2_2 = BatchNormalization(axis=-1, name='bn2')(conv2_2)
    conv2_2 = Activation('relu')(conv2_2)

    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2', data_format='channels_last')(conv2_2)
    conv3_1 = Conv2D(256, (3, 3), activation='relu', name='conv3_1', data_format='channels_last',
                     kernel_regularizer=l2(regularize_weight))(pool2)
    conv3_2 = Conv2D(256, (3, 3), name='conv3_2', data_format='channels_last',
                     kernel_regularizer=l2(regularize_weight))(conv3_1)
    conv3_2 = BatchNormalization(axis=-1, name='bn3')(conv3_2)
    conv3_2 = Activation('relu')(conv3_2)

    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3', data_format='channels_last')(conv3_2)
    conv4_1 = Conv2D(512, (3, 3), activation='relu', name='conv4_1', data_format='channels_last',
                     kernel_regularizer=l2(regularize_weight))(pool3)
    conv4_2 = Conv2D(512, (3, 3), name='conv4_2', data_format='channels_last',
                     kernel_regularizer=l2(regularize_weight))(conv4_1)
    conv4_2 = BatchNormalization(axis=-1, name='bn4')(conv4_2)
    conv4_2 = Activation('relu')(conv4_2)

    upool3 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same', data_format='channels_last', name='upool3')(
        conv4_2)
    upool3 = SpatialDropout2D(0.25, data_format='channels_last')(upool3)
    crop3 = Cropping2D(cropping=((5, 4), (5, 4)), data_format='channels_last')(conv3_2)
    crop3 = SpatialDropout2D(0.25, data_format='channels_last')(crop3)
    merge3 = concatenate([upool3, crop3], axis=-1, name='merge3')
    dconv3_1 = Conv2D(256, (3, 3), activation='relu', name='dconv3_1', data_format='channels_last',
                      kernel_regularizer=l2(regularize_weight))(merge3)
    dconv3_2 = Conv2D(256, (3, 3), name='dconv3_2', data_format='channels_last',
                      kernel_regularizer=l2(regularize_weight))(dconv3_1)
    dconv3_2 = BatchNormalization(axis=-1, name='bn3d')(dconv3_2)
    dconv3_2 = Activation('relu')(dconv3_2)

    upool2 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same', data_format='channels_last', name='upool2')(
        dconv3_2)
    upool2 = SpatialDropout2D(0.25, data_format='channels_last')(upool2)
    crop2 = Cropping2D(cropping=((17, 17), (17, 17)), data_format='channels_last')(conv2_2)
    crop2 = SpatialDropout2D(0.25, data_format='channels_last')(crop2)
    merge2 = concatenate([upool2, crop2], axis=-1, name='merge2')
    dconv2_1 = Conv2D(128, (3, 3), activation='relu', name='dconv2_1', data_format='channels_last',
                      kernel_regularizer=l2(regularize_weight))(merge2)
    dconv2_2 = Conv2D(128, (3, 3), name='dconv2_2', data_format='channels_last',
                      kernel_regularizer=l2(regularize_weight))(dconv2_1)
    dconv2_2 = BatchNormalization(axis=-1, name='bn2d')(dconv2_2)
    dconv2_2 = Activation('relu')(dconv2_2)

    upool1 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same', data_format='channels_last', name='upool1')(
        dconv2_2)
    upool1 = SpatialDropout2D(0.5, data_format='channels_last')(upool1)
    crop1 = Cropping2D(cropping=((42, 42), (42, 42)), data_format='channels_last')(conv1_2)
    crop1 = SpatialDropout2D(0.5, data_format='channels_last')(crop1)
    merge1 = concatenate([upool1, crop1], axis=-1, name='merge1')
    dconv1_1 = Conv2D(64, (3, 3), activation='relu', name='dconv1_1', data_format='channels_last',
                      kernel_regularizer=l2(regularize_weight))(merge1)
    dconv1_2 = Conv2D(64, (3, 3), activation='relu', name='dconv1_2', data_format='channels_last',
                      kernel_regularizer=l2(regularize_weight))(dconv1_1)
    dconv1_2 = SpatialDropout2D(0.25, data_format='channels_last')(dconv1_2)

    # multiscale seg out
    b1_up = UpSampling2D(size=(8, 8), data_format='channels_last')(conv4_2)
    b1_up = Cropping2D(cropping=((14, 14), (14, 14)), data_format='channels_last')(b1_up)
    b2_up = UpSampling2D(size=(4, 4), data_format='channels_last')(dconv3_2)
    b2_up = Cropping2D(cropping=((6, 6), (6, 6)), data_format='channels_last')(b2_up)
    b3_up = UpSampling2D(size=(2, 2), data_format='channels_last')(dconv2_2)
    b3_up = Cropping2D(cropping=((2, 2), (2, 2)), data_format='channels_last')(b3_up)

    merge_out = concatenate([b1_up, b2_up, b3_up, dconv1_2], axis=-1, name='merge_out')

    seg_out = Conv2D(4, (3, 3), activation='relu', name='seg_out_', data_format='channels_last',
                     kernel_regularizer=l2(regularize_weight), padding='same')(merge_out)
    seg_out = Softmax4D(axis=-1, name='seg_out')(seg_out)

    if not TRAIN_CLASSES:
        model = Model(inputs=in_image, outputs=seg_out)
        # sgd = SGD(lr=0.001, momentum=0.9, decay=1e-8, nesterov=False, clipvalue=1.)
        model.compile(optimizer=Adam(learning_rate=ADAM_LR, beta_1=ADAM_BETA_1), loss=multiclass_balanced_cross_entropy_loss_unet)
    else:
        # label out IRF, SRF, PED
        c_out_IRF = Conv2D(1, (3, 3), activation='sigmoid', name='c_IRF', data_format='channels_last',
                           kernel_regularizer=l2(regularize_weight))(conv4_2)
        c_out_IRF = GlobalMaxPooling2D(data_format='channels_last', name='c_out_IRF')(c_out_IRF)

        c_out_SRF = Conv2D(2, (3, 3), activation='sigmoid', name='c_SRF', data_format='channels_last',
                           kernel_regularizer=l2(regularize_weight))(conv4_2)
        c_out_SRF = GlobalAveragePooling2D(data_format='channels_last', name='c_out_SRF')(c_out_SRF)

        c_out_PED = Conv2D(2, (3, 3), activation='sigmoid', name='c_PED', data_format='channels_last',
                           kernel_regularizer=l2(regularize_weight))(conv4_2)
        c_out_PED = GlobalAveragePooling2D(data_format='channels_last', name='c_out_PED')(c_out_PED)

        model = Model(inputs=in_image, outputs=[seg_out, c_out_IRF, c_out_SRF, c_out_PED])
        # sgd = SGD(learning_rate=0.001, momentum=0.9, decay=1e-8, nesterov=False, clipvalue=1.)
        model.compile(optimizer=Adam(learning_rate=ADAM_LR, beta_1=ADAM_BETA_1),
                      loss=[multiclass_balanced_cross_entropy_loss_unet, 'binary_crossentropy', 'binary_crossentropy',
                            'binary_crossentropy'])

    model.summary()
    plot_model(model, to_file='./outputs/model_unet.png', show_shapes=True)

    return model

def retouch_discriminator(input_shape=(224, 224, 3), regularize_weight=0.0001):
    from keras.layers import Conv2D, GlobalAveragePooling2D, Dense, MaxPooling2D, BatchNormalization, \
        concatenate, Activation
    from keras.models import Model
    from keras.utils import plot_model
    from keras.regularizers import l2
    from keras import Input

    in_image = Input(shape=input_shape)
    in_mask = Input(shape=(input_shape[0], input_shape[1], NB_CLASSES))

    conv1_0_im = Conv2D(64, (7, 7), activation='relu', name='conv1_0_', padding='same', data_format='channels_last',
                        kernel_regularizer=l2(regularize_weight))(in_image)
    conv1_1_im = Conv2D(64, (3, 3), activation='relu', name='conv1_1_', data_format='channels_last',
                        kernel_regularizer=l2(regularize_weight))(conv1_0_im)
    conv1_2_im = Conv2D(64, (3, 3), name='conv1_2_', data_format='channels_last',
                        kernel_regularizer=l2(regularize_weight))(conv1_1_im)
    conv1_2_im = BatchNormalization(axis=-1, name='bn1_')(conv1_2_im)
    conv1_2_im = Activation('relu')(conv1_2_im)

    conv1_0_mask = Conv2D(64, (7, 7), activation='relu', name='conv1_0_m', padding='same', data_format='channels_last',
                          kernel_regularizer=l2(regularize_weight))(in_mask)
    conv1_1_mask = Conv2D(64, (3, 3), activation='relu', name='conv1_1_m', data_format='channels_last',
                          kernel_regularizer=l2(regularize_weight))(conv1_0_mask)
    conv1_2_mask = Conv2D(64, (3, 3), name='conv1_2_m', data_format='channels_last',
                          kernel_regularizer=l2(regularize_weight))(conv1_1_mask)
    conv1_2_mask = BatchNormalization(axis=-1, name='bn1_m')(conv1_2_mask)
    conv1_2_mask = Activation('relu')(conv1_2_mask)

    conv1_2 = concatenate([conv1_2_im, conv1_2_mask], axis=-1, name='merge_in')

    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1', data_format='channels_last')(conv1_2)
    conv2_1 = Conv2D(128, (3, 3), activation='relu', name='conv2_1', data_format='channels_last',
                     kernel_regularizer=l2(regularize_weight))(pool1)
    conv2_2 = Conv2D(128, (3, 3), name='conv2_2', data_format='channels_last',
                     kernel_regularizer=l2(regularize_weight))(conv2_1)
    conv2_2 = BatchNormalization(axis=-1, name='bn2')(conv2_2)
    conv2_2 = Activation('relu')(conv2_2)

    pool2 = MaxPooling2D(pool_size=(2, 1), name='pool2', data_format='channels_last')(conv2_2)
    conv3_1 = Conv2D(256, (3, 3), activation='relu', name='conv3_1', data_format='channels_last',
                     kernel_regularizer=l2(regularize_weight))(pool2)
    conv3_2 = Conv2D(256, (3, 3), name='conv3_2', data_format='channels_last',
                     kernel_regularizer=l2(regularize_weight))(conv3_1)
    conv3_2 = BatchNormalization(axis=-1, name='bn3')(conv3_2)
    conv3_2 = Activation('relu')(conv3_2)

    pool3 = MaxPooling2D(pool_size=(2, 1), name='pool3', data_format='channels_last')(conv3_2)
    conv4_1 = Conv2D(512, (3, 3), activation='relu', name='conv4_1', data_format='channels_last',
                     kernel_regularizer=l2(regularize_weight))(pool3)
    conv4_2 = Conv2D(512, (3, 3), name='conv4_2', data_format='channels_last',
                     kernel_regularizer=l2(regularize_weight))(conv4_1)
    conv4_2 = BatchNormalization(axis=-1, name='bn4')(conv4_2)
    conv4_2 = Activation('relu')(conv4_2)

    apool = GlobalAveragePooling2D(data_format='channels_last')(conv4_2)
    disc_out = Dense(2, activation='softmax', name='disc_out', kernel_regularizer=l2(regularize_weight))(apool)

    model = Model(inputs=[in_image, in_mask], outputs=disc_out)
    # sgd = SGD(lr=0.01, momentum=0.9, decay=1e-8, nesterov=False, clipvalue=1.)
    # model.compile(optimizer=sgd, loss=multiclass_balanced_cross_entropy_loss_unet)

    model.summary()
    # plot_model(model, to_file='./outputs/model_unet.png', show_shapes=True)

    return model

def retouch_unet_no_drop(input_shape=(224, 224, 3)):
    from keras.models import Sequential
    from keras.layers import Conv2D, SpatialDropout2D, GlobalAveragePooling2D, Input, Dense, UpSampling2D, \
        AveragePooling2D, GlobalMaxPooling2D, Lambda, MaxPooling2D, Flatten, Conv2DTranspose
    from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
    from tensorflow.python.keras.layers import BatchNormalization
    from tensorflow.python.keras.layers.merge import concatenate
    from keras.layers import Activation
    from keras.models import Model
    from keras.utils import plot_model
    from custom_layers import Softmax4D
    from keras.optimizers import SGD, Adam
    from keras import backend as K
    from keras.layers import Cropping2D

    in_image = Input(shape=input_shape)

    conv1_0 = Conv2D(64, (7, 7), activation='relu', name='conv1_0', padding='same', data_format='channels_last')(
        in_image)
    conv1_1 = Conv2D(64, (3, 3), activation='relu', name='conv1_1', data_format='channels_last')(conv1_0)
    conv1_2 = Conv2D(64, (3, 3), name='conv1_2', data_format='channels_last')(conv1_1)
    conv1_2 = BatchNormalization(axis=-1, name='bn1')(conv1_2)
    conv1_2 = Activation('relu')(conv1_2)

    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1', data_format='channels_last')(conv1_2)
    conv2_1 = Conv2D(128, (3, 3), activation='relu', name='conv2_1', data_format='channels_last')(pool1)
    conv2_2 = Conv2D(128, (3, 3), name='conv2_2', data_format='channels_last')(conv2_1)
    conv2_2 = BatchNormalization(axis=-1, name='bn2')(conv2_2)
    conv2_2 = Activation('relu')(conv2_2)

    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2', data_format='channels_last')(conv2_2)
    conv3_1 = Conv2D(256, (3, 3), activation='relu', name='conv3_1', data_format='channels_last')(pool2)
    conv3_2 = Conv2D(256, (3, 3), name='conv3_2', data_format='channels_last')(conv3_1)
    conv3_2 = BatchNormalization(axis=-1, name='bn3')(conv3_2)
    conv3_2 = Activation('relu')(conv3_2)

    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3', data_format='channels_last')(conv3_2)
    conv4_1 = Conv2D(512, (3, 3), activation='relu', name='conv4_1', data_format='channels_last')(pool3)
    conv4_2 = Conv2D(512, (3, 3), name='conv4_2', data_format='channels_last')(conv4_1)
    conv4_2 = BatchNormalization(axis=-1, name='bn4')(conv4_2)
    conv4_2 = Activation('relu')(conv4_2)

    upool3 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same', data_format='channels_last', name='upool3')(
        conv4_2)
    # upool3 = SpatialDropout2D(0.25, data_format='channels_last')(upool3)
    crop3 = Cropping2D(cropping=((5, 4), (5, 4)), data_format='channels_last')(conv3_2)
    # crop3 = SpatialDropout2D(0.25, data_format='channels_last')(crop3)
    merge3 = concatenate([upool3, crop3], axis=-1, name='merge3')
    dconv3_1 = Conv2D(256, (3, 3), activation='relu', name='dconv3_1', data_format='channels_last')(merge3)
    dconv3_2 = Conv2D(256, (3, 3), name='dconv3_2', data_format='channels_last')(dconv3_1)
    dconv3_2 = BatchNormalization(axis=-1, name='bn3d')(dconv3_2)
    dconv3_2 = Activation('relu')(dconv3_2)

    upool2 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same', data_format='channels_last', name='upool2')(
        dconv3_2)
    # upool2 = SpatialDropout2D(0.5, data_format='channels_last')(upool2)
    crop2 = Cropping2D(cropping=((17, 17), (17, 17)), data_format='channels_last')(conv2_2)
    # crop2 = SpatialDropout2D(0.5, data_format='channels_last')(crop2)
    merge2 = concatenate([upool2, crop2], axis=-1, name='merge2')
    dconv2_1 = Conv2D(128, (3, 3), activation='relu', name='dconv2_1', data_format='channels_last')(merge2)
    dconv2_2 = Conv2D(128, (3, 3), name='dconv2_2', data_format='channels_last')(dconv2_1)
    dconv2_2 = BatchNormalization(axis=-1, name='bn2d')(dconv2_2)
    dconv2_2 = Activation('relu')(dconv2_2)
    # dconv2_2 = SpatialDropout2D(0.5, data_format='channels_last')(dconv2_2)

    upool1 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same', data_format='channels_last', name='upool1')(
        dconv2_2)
    # upool1 = SpatialDropout2D(0.5, data_format='channels_last')(upool1)
    crop1 = Cropping2D(cropping=((42, 42), (42, 42)), data_format='channels_last')(conv1_2)
    # crop1 = SpatialDropout2D(0.5, data_format='channels_last')(crop1)
    merge1 = concatenate([upool1, crop1], axis=-1, name='merge1')
    dconv1_1 = Conv2D(64, (3, 3), activation='relu', name='dconv1_1', data_format='channels_last')(merge1)
    dconv1_2 = Conv2D(64, (3, 3), activation='relu', name='dconv1_2', data_format='channels_last')(dconv1_1)
    # dconv1_2 = SpatialDropout2D(0.25, data_format='channels_last')(dconv1_2)

    seg_out = Conv2D(4, (1, 1), activation='relu', name='outp', data_format='channels_last')(dconv1_2)
    seg_out = Softmax4D(axis=-1, name='seg_out')(seg_out)

    model = Model(inputs=in_image, outputs=seg_out)
    sgd = SGD(lr=0.01, momentum=0.9, decay=1e-8, nesterov=False, clipvalue=2000.)
    model.compile(optimizer=sgd, loss=multiclass_balanced_cross_entropy_loss_unet)

    model.summary()
    # plot(model, to_file='a.png', show_shapes=True)

    return model
