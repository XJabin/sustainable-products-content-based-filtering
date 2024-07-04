from flask import Flask, jsonify, request,render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentBasedRecommender:
    def __init__(self, data_path):
        self.data_path = data_path
        self.ds = None
        self.tfidf_matrix = None
        self.cosine_similarities = None
        self.results = None
        self.tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=1, stop_words='english')

    def load_data(self):
        print(self.data_path)
        self.ds = pd.read_csv(self.data_path)

    def preprocess_data(self):
        self.tfidf_matrix = self.tf.fit_transform(self.ds['content'])
        self.cosine_similarities = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

    def build_similarity_matrix(self):
        self.results = {}
        for idx, row in self.ds.iterrows():
            similar_indices = self.cosine_similarities[idx].argsort()[:-100:-1]
            similar_items = [(self.cosine_similarities[idx][i], self.ds['id'][i]) for i in similar_indices]
            self.results[row['id']] = similar_items[1:]

    def item(self, id):
        return self.ds.loc[self.ds['id'] == id]['content'].tolist()[0].split(' - ')[0]

    def recommend(self, item_id, num):
        if self.results is None:
            raise Exception("Similarity matrix has not been built yet. Call preprocess_data() and build_similarity_matrix() first.")
        
        recs = self.results.get(item_id, [])[:num]  # Get top num recommendations for item_id
        recommended_items = []
        for rec in recs:
            product_details = self.ds.loc[self.ds['id'] == rec[1]].to_dict('records')[0]
            recommended_items.append({
                "id": product_details['id'],
                "product_name": product_details['PdName'],
                "category": product_details['Category'],
                "material": product_details['Material'],
                "manufacturer": product_details['Manufacturer'],
                "certifications": product_details['Certifications'],
                "price": product_details['Price'],
                "sustainability_features": product_details['Sustainability Features'],
                "country_of_origin": product_details['Country of Origin'],
                "score": float(rec[0])
            })
        print(recommended_items)
        return recommended_items

    def recommend_content(self, input_product_description, num_recommendations):
        if self.tfidf_matrix is None or self.cosine_similarities is None:
            raise Exception("Model has not been trained yet. Call preprocess_data() first.")
        
        input_matrix = self.tf.transform([input_product_description])
        cosine_similarities = linear_kernel(input_matrix, self.tfidf_matrix).flatten()
        similar_indices = cosine_similarities.argsort()[:-num_recommendations-1:-1]

        recommended_items = []
        for idx in similar_indices:
            recommended_items.append({
                "product_name": self.ds['content'][idx],
                "score": cosine_similarities[idx]
            })
        return recommended_items
    
    def get_all_products(self):
        if self.ds is None:
            self.load_data()
        return self.ds.to_dict(orient='records')
    
    def get_product_details(self, id):
        if self.ds is None:
            self.load_data()  # Ensure data is loaded
        
        product = self.ds.loc[self.ds['id'] == id].to_dict(orient='records')
        return product[0] if product else None  



# Flask application
app = Flask(__name__)
recommender = ContentBasedRecommender("C:/Users/ranag/Desktop/kabin/recyclazon/EchoChoice/data/cleaned.csv")

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/recommend/<item_id>/<int:num>', methods=['GET'])
def get_recommendations(item_id, num):
    try:
        if recommender.results is None:
            recommender.load_data()
            recommender.preprocess_data()
            recommender.build_similarity_matrix()

        recommendations = recommender.recommend(item_id, num)
        return jsonify(recommendations)
    except Exception as e:
        return str(e), 500

@app.route('/recommend_content', methods=['POST'])
def recommend_content():
    try:
        if recommender.tfidf_matrix is None or recommender.cosine_similarities is None:
            recommender.load_data()
            recommender.preprocess_data()

        data = request.json
        input_description = data['input_description']
        num_recommendations = data['num_recommendations']

        recommendations = recommender.recommend_content(input_description, num_recommendations)
        return jsonify(recommendations)
    except Exception as e:
        return str(e), 500
    
@app.route('/product', methods=['GET'])
def get_all_products():
    try:
        global recommender
        if recommender is None:
            recommender = ContentBasedRecommender("C:/Users/ranag/Desktop/kabin/recyclazon/EchoChoice/data/cleaned.csv")
            recommender.load_data()

        all_products = recommender.get_all_products()

        print(all_products)
        app.logger.info(f"Fetched {len(all_products)} products")
        return render_template('product.html', products=all_products)
    
    except Exception as e:
        app.logger.error(f"Error fetching products: {str(e)}")
        return str(e), 500   
    
   
@app.route('/product/details/<id>', methods=['GET'])
def get_product_details(id):
    try:
        global recommender
        if recommender is None:
            recommender = ContentBasedRecommender("C:/Users/ranag/Desktop/kabin/recyclazon/EchoChoice/data/cleaned.csv")
            recommender.load_data()

        product = recommender.get_product_details(id)
        if recommender.results is None:
            recommender.load_data()
            recommender.preprocess_data()
            recommender.build_similarity_matrix()

        related_prod = recommender.recommend(id,5)
        print(related_prod)
        if product:
            app.logger.info(f"Fetched product details for ID {id}")
            return render_template('productDetails.html', product=product,related_products= related_prod)
        else:
            return "Product not found", 404
    
    except Exception as e:
        app.logger.error(f"Error fetching product details: {str(e)}")
        return str(e), 500


if __name__ == '__main__':
    app.run(debug=True)
