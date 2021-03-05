from sentence_transformers import SentenceTransformer
from flask import Flask, request, render_template
import scipy
import os
import pandas as pd


model = SentenceTransformer('bert-base-nli-mean-tokens')

app = Flask(__name__)

df= pd.read_csv(r"aajtak.csv")

sentences = df['news'].values.tolist()

print("Getting embeddings for sentences ....")
sentence_embeddings = model.encode(sentences)
print("done with getting embeddings for sentences ....")


print('Sample BERT embedding vector - length', len(sentence_embeddings[0]))

def performSearch(query):
	queries = [query]
	query_embeddings = model.encode(queries)

	# Find the closest 3 sentences of the corpus for each query sentence based on cosine similarity
	number_top_matches = 5 #@param {type: "number"}

	print("Semantic Search Results")
	results = []
	for query, query_embedding in zip(queries, query_embeddings):
		distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]

		results = zip(range(len(distances)), distances)
		results = sorted(results, key=lambda x: x[1])

    	
	return results

@app.route("/semanticsearch",	 methods=['GET', 'POST'])
def rec():
	query = '' 
	if(request.method == "POST"):
		print("inside post")
		query = request.form.get('query')
		print(query)
		results = performSearch(query)
		return render_template('semantic_search.html', query=query, results=results, sentences=sentences)
	else:
		return render_template('semantic_search.html', review="" ,results=None)
	

if __name__ == "__main__":
    app.run(host='127.0.0.1',port=9200, debug=True, threaded=True)