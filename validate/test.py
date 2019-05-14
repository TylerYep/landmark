# Move stuff here later - train.py comes first

K.eval(gm_exp)


print(model.history.history['loss'])


plt.plot(model.history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')



plt.plot(model.history.history['batch_GAP'])
plt.xlabel('epoch')
plt.ylabel('batch_GAP')




plt.plot(model.history.history['acc'])
plt.xlabel('epoch')
plt.ylabel('acc')


# #### Validation and prediciton


def predict(info, load_n_images=1024):
    n = len(info)
    max_p = np.zeros(n)
    pred = np.zeros(n)

    for ind in range(0,len(info),load_n_images):
        imgs = load_images(info.iloc[ind:(ind+load_n_images)])
        imgs = preprocess_input(imgs)
        proba = model.predict(imgs, batch_size=batch_size_predict)

        pred_i = np.argmax(proba, axis=1)
        max_p[ind:(ind + load_n_images)] = proba[np.arange(len(pred_i)),pred_i]
        pred[ind:(ind + load_n_images)] = label_encoder.inverse_transform(pred_i)

        print(ind, '/', len(info), '  -->', pred[ind], max_p[ind])

    print(len(info), '/', len(info), '  -->', pred[-1], max_p[-1])

    return pred, max_p


def predict_wcr_vote(info, load_n_images=1024, crop_p=0.1, n_crops = 12):
    n = len(info)
    max_p = np.zeros(n)
    pred = np.zeros(n)

    for ind in range(0,len(info),load_n_images):
        all_proba = np.zeros((n_crops, min(load_n_images, len(info)-ind), n_cat))

        imgs = load_images(info.iloc[ind:(ind+load_n_images)])
        imgs = preprocess_input(imgs)

        #full image
        all_proba[0,:,:] = model.predict(imgs, batch_size=batch_size_predict)
        all_proba[1,:,:] = model.predict(np.flip(imgs, axis=2),
                                         batch_size=batch_size_predict)

        crops = ['upper left', 'lower left', 'upper right', 'lower right', 'central']
        jnd_0 = 2
        for jnd,crop in enumerate(crops):
            imgs = load_cropped_images(info.iloc[ind:(ind+load_n_images)],
                                  crop_p=crop_p, crop=crop)  # optimize later
            imgs = preprocess_input(imgs)
            all_proba[jnd_0+2*jnd,:,:] = model.predict(imgs,
                                                       batch_size=batch_size_predict)
            all_proba[jnd_0+2*jnd+1,:,:] = model.predict(np.flip(imgs, axis=2),
                                                         batch_size=batch_size_predict)

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
            pred[ind + knd]=c_res['max_p'].idxmax()
            max_p[ind + knd]=c_res.loc[pred[ind + knd]]['max_p']

        print(ind,'/',len(info), '  -->', pred[ind], max_p[ind])
    print(len(info),'/',len(info), '  -->', pred[-1], max_p[-1])

    return pred, max_p


# In[256]:


def validate(info, load_n_images=1024, wcr=False, crop_p=0.1):
    if wcr:
        pred, max_p = predict_wcr_vote(info, load_n_images=load_n_images, crop_p=crop_p)
    else:
        pred, max_p = predict(info, load_n_images=load_n_images)

    y = info['landmark_id'].values
    binary_acc = accuracy_score(y, pred)

    sort_ind = np.argsort(max_p)[::-1]

    pred = pred[sort_ind]
    y_true = y[sort_ind]

    GAP = np.sum(np.cumsum(pred == y_true) * (pred == y_true) / np.arange(1, len(y_true) + 1)) / np.sum(y_true >= 0.)

    print("accuracy:", binary_acc, "\n ")
    print("*** GAP:", GAP, "***")

    return binary_acc, GAP


# Validate only on landmark images

# In[257]:


dev_binary_acc, dev_GAP = validate(dev_info, 50)


# Validate on landmark and non-landmark images

# In[123]:


dev_binary_acc, dev_GAP = validate(pd.concat([dev_info, nlm_dev_df]).sample(frac=1), 1024)


# In[ ]:


dev_binary_acc_wcr, dev_GAP_wcr = validate(dev_info, 1024, wcr=True, crop_p=0.1)


# Some checks before actual prediction

# In[ ]:


print(len(test_info))


# In[ ]:


_, _ = predict_wcr_vote(test_info[:10], 512, crop_p=0.1)


# In[ ]:


#test_pred, test_max_p = predict(test_info, 1024)


# And predict!

# In[ ]:


test_pred, test_max_p = predict_wcr_vote(test_info, 512, crop_p=0.1)


# In[ ]:


predictions = pd.DataFrame(columns=['landmarks'], index=test_info.index)
predictions['landmarks'] = [str(int(tp))+' '+ '%.16g' % pp
                            for tp,pp in zip(test_pred, test_max_p)]
predictions.head()


# In[ ]:


test_info_full = pd.read_csv('test.csv', index_col=0)
test_info_full.head()


# Fill the missing values with the most common landmark

# In[ ]:


missing = test_info_full[test_info_full.index.isin(test_info.index)!=True]
missing_predictions = pd.DataFrame(index=missing.index)
missing_predictions['landmarks'] = '9633 0.0'
missing_predictions.head()


# In[ ]:


completed_predictions = pd.concat([predictions, missing_predictions])
print(len(completed_predictions))


# In[ ]:


sorted_predictions = pd.DataFrame(index=test_info_full.index)
sorted_predictions['landmarks'] = completed_predictions['landmarks']
sorted_predictions.tail()


# In[ ]:


sorted_predictions.to_csv('prediction_c12.csv')


# In[ ]:
