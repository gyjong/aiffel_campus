from models import model

def train_model(train_generator, valid_generator, epoch):
    model.fit(train_generator,
            validation_data=valid_generator,
            epochs=epoch)

def eval_model(test_generator):
    _, acc = model.evaluate(test_generator)

    return acc