import data_prep as dt
import model as md
import matplotlib.pyplot as plt


def main():
    model = md.for_model()
    tr_gen, val_gen = dt.get_generators()
    hist = model.fit_generator(tr_gen, steps_per_epoch= 60000//64,
                               epochs= 5, validation_data= val_gen,
                               validation_steps=10000//64).history
    return model, hist


hist = None
if __name__ == '__main__' :
    model,hist = main()
    model.save('model2.h5')

    plt.subplot(2,1,1)
    plt.plot(hist['mean_squared_error'],
             label= 'mean_squared_error' )
    plt.plot(hist['val_mean_squared_error'],
             label='val_mean_squared_error')
    plt.legend()

    plt.subplot(2,1,2)

    plt.plot(hist['loss'], label='loss')
    plt.plot(hist['val_loss'],label='val_loss')
    plt.legend()

    plt.show()
