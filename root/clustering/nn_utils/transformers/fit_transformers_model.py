from keras.callbacks import EarlyStopping, ReduceLROnPlateau

def fit_transformer_model(model, X_train, y_train):
    callbacks = [EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True),
                 ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-5)]

    print(y_train)

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=200,
        batch_size=64,
        callbacks=callbacks,
    )

    return history