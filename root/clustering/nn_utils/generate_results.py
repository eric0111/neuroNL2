import datetime

from dateutil.utils import today
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix


def generate_results(history, model, X_test, y_test):
    # summarize history for accuracy
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    #plt.show()
    plt.savefig(str(datetime.datetime.now()) + ".png")

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    #plt.show()
    plt.savefig(str(datetime.datetime.now())+".png")

    # model.fit(X_train, y_train, epochs=30)

    # evaluate model
    print('evaluating..')
    y_pred = model.predict(X_test)
    y_test_1d = [i[0] for i in y_test]
    y_pred_1d = [1.0 if i[0] > .5 else 0.0 for i in y_pred]
    acc_score = accuracy_score(y_test_1d, y_pred_1d)
    print("acc_score")
    print(acc_score)

    tn, fp, fn, tp = confusion_matrix(y_test_1d, y_pred_1d).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print("sensitivity", sensitivity)
    print("specificity", specificity)
    print("accuracy", accuracy)
