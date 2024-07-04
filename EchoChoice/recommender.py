import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


ds = pd.read_csv("data_cleaned.csv")



tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=1, stop_words='english')

tfidf_matrix = tf.fit_transform(ds['content'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

results = {}

for idx, row in ds.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], ds['id'][i]) for i in similar_indices]

    results[row['id']] = similar_items[1:]

print('done!')





def item(id):
    return ds.loc[ds['id'] == id]['content'].tolist()[0].split(' - ')[0]

def recommend(item_id, num):
    print("Recommending " + str(num) + " products similar to " + item(item_id) + "...")
    print("-------")
    recs = results[item_id][:num]
    for rec in recs:
        print("Recommended: " + item(rec[1]) + " (score:" + str(rec[0]) + ")")


recommend(item_id="P012", num=5)



def recommend_content(input_product_description, num_recommendations):
    
    input_matrix = tf.transform([input_product_description])

    cosine_similarities = linear_kernel(input_matrix, tfidf_matrix).flatten()

    similar_indices = cosine_similarities.argsort()[:-num_recommendations-1:-1]

    print(f"Recommending {num_recommendations} products similar to '{input_product_description}'...")
    print("-------")
    for idx in similar_indices:
        print(f"Recommended: {ds['Product Name'][idx]} (Score: {cosine_similarities[idx]:.4f})")


input_description = "Yoga"
num_recommendations = 2
recommend_content(input_description, num_recommendations)