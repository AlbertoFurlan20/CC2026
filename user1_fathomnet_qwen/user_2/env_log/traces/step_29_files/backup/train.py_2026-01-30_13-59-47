for j in range(len(proba)):
    if proba[j] > threshold:
        predict.append(classes[j])
predict_list.append(predict)

submission = pd.DataFrame({"image_id": valid_df["id"], "PredictionString": [" ".join(p) for p in predict_list]})
submission.to_csv("submission.csv", index=False)