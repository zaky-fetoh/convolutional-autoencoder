import keras.models as models
import keras.layers as layers
import keras.optimizers as opt
import keras.losses as lss
import keras.metrics as met

def get_model(inp =(28,28,1)):

    model = models.Sequential()
    model.add( layers.Conv2D(28,(3,3),activation='relu',input_shape= inp,padding='same'))
    model.add( layers.MaxPool2D((2,2) ) ) #(14,14)

    model.add( layers.Conv2D(64,(3,3), activation='relu',padding='same'  ))
    model.add( layers.MaxPool2D((2,2))) #(7,7)

    model.add(layers.Conv2D(64,(3,3),activation='relu' ,padding='same'))
    model.add( layers.MaxPool2D((2,2)))#(3,3)

    model.add(layers.Flatten() )
    model.add(layers.Dense(4*4*16 , activation='relu') )
    model.add( layers.Reshape((4,4,16)))#$(4,4)

    model.add( layers.UpSampling2D((2,2)))
    model.add( layers.Conv2D(64,(3,3), activation='relu',padding='same' ))#$(8,8)

    model.add( layers.UpSampling2D((2,2)))
    model.add( layers.Conv2D(64,(3,3), activation='relu')) #$(14,14)

    model.add( layers.UpSampling2D((2,2)))
    model.add( layers.Conv2D(28,(3,3), activation='relu', padding= 'same'))

    model.add( layers.Conv2D(1,(3,3), activation='relu',padding='same') )

    model.compile(optimizer= opt.RMSprop(lr=.0001),
                  loss= lss.mean_absolute_percentage_error,
                  metrics=['acc'] )
    return model

def for_model(inp=(28,28,1) ) :

    model = models.Sequential()
    model.add( layers.Conv2D(16,3,activation='relu',padding='same',
                             input_shape=inp)) #28

    model.add( layers.MaxPool2D() )
    model.add( layers.Conv2D(8,3,activation='relu',padding='same'))#14

    model.add( layers.MaxPool2D())
    model.add(layers.Conv2D(8,3,activation='relu',padding='same'))#7

    model.add( layers.MaxPool2D(padding='same')) #4

    model.add(layers.UpSampling2D())
    model.add( layers.Conv2D(8,3,activation='relu',padding='same'))

    model.add(layers.UpSampling2D())
    model.add( layers.Conv2D(8,3,activation='relu'))

    model.add(layers.UpSampling2D())
    model.add( layers.Conv2D(8,3,activation='relu',padding='same'))

    model.add(layers.Conv2D(1,3,activation='relu', padding='same'))

    model.compile(optimizer=opt.Adam(), loss=lss.mean_squared_error,
                  metrics=['mse'] )

    return model

def load_model():
    return models.load_model('model1.h5' )




































