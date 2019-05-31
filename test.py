import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from keras.applications.xception import preprocess_input
from models import Baseline, Sirius
import dataset
import const

def main():
    model = Baseline().model
    model.load_weights(const.BEST_SAVE_MODEL)
    test_info, label_encoder = dataset.load_data(type='test')
    test_results_to_csv(model, test_info, label_encoder)


def test_results_to_csv(model, test_info, label_encoder):
    # dev_binary_acc_wcr, dev_GAP_wcr = validate(model, dev_info, label_encoder, wcr=True, crop_p=0.1)
    # test_pred, test_max_p = predict(model, test_info, label_encoder)

    test_pred, test_max_p = predict(model, test_info, label_encoder)

    predictions = pd.DataFrame(columns=['landmarks'], index=test_info.index)
    predictions['landmarks'] = [str(int(tp)) + ' %.16g' % pp for tp, pp in zip(test_pred, test_max_p)]

    test_info_full = pd.read_csv(const.TEST_CSV, index_col=0)

    # Fill the missing values with the most common landmark
    missing = test_info_full[test_info_full.index.isin(test_info.index) != True]
    missing_predictions = pd.DataFrame(index=missing.index)
    missing_predictions['landmarks'] = '9633 0.0'

    completed_predictions = pd.concat([predictions, missing_predictions])

    sorted_predictions = pd.DataFrame(index=test_info_full.index)
    sorted_predictions['landmarks'] = completed_predictions['landmarks']
    sorted_predictions.to_csv('save/first_submission.csv')


#### Validation and prediction
def predict(model, info, label_encoder, load_n_images=1024):
    n = len(info)
    max_p = np.zeros(n)
    pred = np.zeros(n)

    for ind in range(0, len(info), load_n_images):
        imgs = dataset.load_images(info.iloc[ind:(ind+load_n_images)])
        imgs = preprocess_input(imgs)
        proba = model.predict(imgs, batch_size=const.BATCH_SIZE_PREDICT)
        print(proba)

        pred_i = np.argmax(proba, axis=1)
        max_p[ind:(ind+load_n_images)] = proba[np.arange(len(pred_i)), pred_i]
        pred[ind:(ind+load_n_images)] = label_encoder.inverse_transform(pred_i)

        print(ind, '/', len(info), '  -->', pred[ind], max_p[ind])
    print(len(info), '/', len(info), '  -->', pred[-1], max_p[-1])

    return pred, max_p


# This is a version with 12 crops, for the competition I found that
# 22 crops with crop_p=0.05 and crop_p=0.15 worked even better.
def predict_wcr_vote(model, info, label_encoder, load_n_images=1024, crop_p=0.1, n_crops=12):
    max_p = np.zeros(len(info))
    pred = np.zeros(len(info))

    for ind in range(0, len(info), load_n_images):
        all_proba = np.zeros((n_crops, min(load_n_images, len(info)-ind), const.N_CAT))

        imgs = dataset.load_images(info.iloc[ind:(ind+load_n_images)])
        imgs = preprocess_input(imgs)

        #full image
        all_proba[0,:,:] = model.predict(imgs, batch_size=const.BATCH_SIZE_PREDICT)
        all_proba[1,:,:] = model.predict(np.flip(imgs, axis=2), batch_size=const.BATCH_SIZE_PREDICT)

        crops = ['upper left', 'lower left', 'upper right', 'lower right', 'central']
        jnd_0 = 2
        for jnd, crop in enumerate(crops):
            imgs = dataset.load_cropped_images(info.iloc[ind:(ind+load_n_images)], crop_p=crop_p, crop=crop)  # optimize later
            imgs = preprocess_input(imgs)
            all_proba[jnd_0+2*jnd,:,:] = model.predict(imgs, batch_size=const.BATCH_SIZE_PREDICT)
            all_proba[jnd_0+2*jnd+1,:,:] = model.predict(np.flip(imgs, axis=2), batch_size=const.BATCH_SIZE_PREDICT)

        cmax_p = np.zeros((n_crops,imgs.shape[0]))
        cpred = np.zeros((n_crops,imgs.shape[0]))
        for jnd in range(all_proba.shape[0]):
            proba = all_proba[jnd,:,:]
            pred_i = np.argmax(proba, axis=1)
            cmax_p[jnd,:] = proba[np.arange(len(pred_i)),pred_i]
            cpred[jnd,:] = label_encoder.inverse_transform(pred_i)

        for knd in range(imgs.shape[0]):
            c_res = pd.DataFrame({'max_cat':cpred[:,knd], 'max_p':cmax_p[:,knd]})
            c_res = c_res.groupby('max_cat').aggregate('sum') / n_crops
            pred[ind + knd] = c_res['max_p'].idxmax()
            max_p[ind + knd] = c_res.loc[pred[ind + knd]]['max_p']

        print(ind, '/', len(info), '  -->', pred[ind], max_p[ind])
    print(len(info), '/', len(info), '  -->', pred[-1], max_p[-1])

    return pred, max_p


def validate(model, info, label_encoder, load_n_images=1024, wcr=False, crop_p=0.1):
    if wcr:
        pred, max_p = predict_wcr_vote(model, info, label_encoder, load_n_images=load_n_images, crop_p=crop_p)
    else:
        pred, max_p = predict(model, info, label_encoder, load_n_images=load_n_images)

    y = info['landmark_id'].values
    binary_acc = accuracy_score(y, pred)

    sort_ind = np.argsort(max_p)[::-1]

    pred = pred[sort_ind]
    y_true = y[sort_ind]

    GAP = np.sum(np.cumsum(pred == y_true) * (pred == y_true) / np.arange(1, len(y_true) + 1)) / np.sum(y_true >= 0.)
    print("accuracy:", binary_acc, "\n ")
    print("*** GAP:", GAP, "***")
    return binary_acc, GAP


if __name__ == '__main__':
    main()
