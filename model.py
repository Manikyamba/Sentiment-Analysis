import numpy as np

def get_sentiment_recommendations(user,user_final_rating,df_clean,tfidf,logreg):
    if (user in user_final_rating.index):
        # get the product recommedation using the trained User-User ML model
        recommendations = list(user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)
        temp = df_clean[df_clean.id.isin(recommendations)]
        #transfor the input data using saved tf-idf vectorizer
        X =  tfidf.transform(temp["reviews_lemmatize"].values.astype(str))
        #predict sentiment using the logistic regression model
        temp["predicted_sentiment"]= logreg.predict(X)
        temp = temp[['name','predicted_sentiment']]
        temp_grouped = temp.groupby('name', as_index=False).count()
        temp_grouped["pos_review_count"] = temp_grouped.name.apply(lambda x: temp[(temp.name==x) & (temp.predicted_sentiment==1)]["predicted_sentiment"].count())
        temp_grouped["total_review_count"] = temp_grouped['predicted_sentiment']
        temp_grouped['pos_sentiment_percent'] = np.round(temp_grouped["pos_review_count"]/temp_grouped["total_review_count"]*100,2)
        res = temp_grouped.sort_values('pos_sentiment_percent', ascending=False)[:5]
        result = res[['name','pos_sentiment_percent']]
        return result
    else:
        return "User name {user} doesn't exist"