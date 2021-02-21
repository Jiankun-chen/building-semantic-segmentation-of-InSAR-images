class DefaultCofig(object):

    # log info
    log_path = './logs/'

    # model info
    model = 'RefineNet'
    batch_size = 8
    use_gpu = True
    epochs = 50
    lr = 1e-6
    momentum = 0.9
    decay = 1e-4
    original_lr = 1e-6
    steps = [-1, 1, 100, 150]
    scales = [1, 1, 1, 1]
    workers = 4

    # train&test files path
    images_ang = 'data/nyu_images_ang'
    images_master = 'data/nyu_images_master'
    images_slave = 'data/nyu_images_slave'

    labels = 'data/nyu_labels40'
    depths = 'data/nyu_depths'
    train_split = 'data/train.txt'
    test_split = 'data/test1.txt'

    # saved model path
    saved_model_path = './saved_models/'

    # test model file
    test_model = saved_model_path + 'RefineNet_0112_163114.pkl'

    # predict files path
    predict_images_ang = './data/predict1/images_ang'
    predict_images_master = './data/predict1/images_master'
    predict_images_slave = './data/predict1/images_slave'
    predict_labels = './data/predict1/labels'
    predict_depths = './data/predict1/depths'