# Semantic Pet Vision 🐱🐶 

> **Semantic Image Search with DocArray + Transformers**


**Semantic Pet Vision** is a lightweight **semantic image search engine** that retrieves similar pet images (cats 🐱 and dogs 🐶) based on their **visual meaning**, not just filenames or keywords.  

By leveraging **Hugging Face vision encoders** and **DocArray**, the system transforms images into embeddings and performs similarity search. A simple query like:

> `"cute ginger kitten"`

returns the most semantically relevant image from the dataset.




## 🧩 How It Works
1. **Indexing** – Images are encoded into embeddings using CLIP.
2. **Storage** – Embeddings are stored in `doc.tags` using DocArray.
3. **Search** – Natural language queries are encoded & compared with cosine similarity.
4. **Result** – The closest image is returned as `top_match.jpg`.




## 🚀 Features
- **Semantic Search** → go beyond keywords, search by meaning  
- **Pet Focused** → trained on cats & dogs for fun experimentation  
- **DocArray Powered** → efficient storage + similarity search  
- **MIT Licensed** → free to use, modify, and expand  



## ▶️ Usage & Applications

Run the main script to:

1. Create documents from sample images
2. Generate embeddings
3. Perform a query search

```bash
python main.py
```

Example query:

```
Query: cute ginger kitten
Top match image saved to top_match.jpg
```

Though fun-sized, this project is a microcosm of real-world systems.

- 🔍 Content-based image retrieval (e.g., “find me similar products”)
- 🐕 Pet adoption search engines (search by photo)
- 📷 Duplicate detection in large photo collections

## 📖 Academic Angle

This project demonstrates **semantic retrieval in computer vision**, bridging **representation learning** and **information retrieval**. It’s a practical, compact example of applying **deep embeddings** for multimodal search tasks.




## ✨ Future Work

* Expand dataset beyond cats & dogs
* Integrate a web-based UI for interactive search
* Experiment with multi-modal queries (text + image)



## 📜 License

This project is licensed under the **MIT License** - free for academic, personal, and commercial use.

