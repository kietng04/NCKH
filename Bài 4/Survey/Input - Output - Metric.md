# Bảng Nghiên Cứu Sử Dụng Dataset IMDB

| Tác giả (Năm) | Input | Tiền xử lý | Feature Extraction/Embedding | Phương pháp | Metric | Hiệu suất |
|---------------|-------|------------|------------------------------|-------------|--------|-----------|
| Iqbal et al. (2018) | IMDB reviews | Tokenization, lemmatization, text cleaning | Unigram, Bigram | Maximum entropy | Accuracy | 88% |
| Dholpuria et al. (2018) | IMDB (3000 reviews) | Loại bỏ ký tự không liên quan, ký hiệu, từ lặp và stop words | Count vectorizer | CNN | Accuracy | 99.33% |
| Hourrane et al. (2019) | IMDB | Tokenization, case folding, loại bỏ URLs, HTML tags, từ không liên quan | TF-IDF | Ridge Classifier | Accuracy | 90.54% |
| Jang et al. (2020) | IMDB (50,000 reviews) | Không nêu chi tiết | word2vec | Hybrid CNN và BiLSTM với attention mechanism | Accuracy | 90.26% |
| Thinh et al. (2019) | IMDB (50,000 reviews) | Không nêu chi tiết | Không nêu chi tiết | 1D-CNN with GRU | Accuracy | 90.02% |
| Gifari & Lhaksmana (2021) | IMDB movie reviews | Tokenization, loại bỏ stop words, stemming | TF-IDF | Ensemble (MNB + KNN + LR) | Accuracy | 89.40% |
| Athar et al. (2021) | IMDB (50,000 reviews) | Tokenization, stemming, loại bỏ URLs, stop words, và punctuation | TF-IDF | Ensemble (LR + NB + XGBoost + RF + MLP) | Accuracy | 89.9% |
| Tan et al. (2022) | IMDB | Không nêu chi tiết | RoBERTa embeddings | RoBERTa-LSTM | Accuracy | 92.96% |
| Kokab et al. (2022) | IMDB | Không nêu chi tiết | BERT embeddings | BERT-based CBRNN | Accuracy | 93% |
| AlBadani et al. (2022) | IMDB | Không nêu chi tiết | Graph embeddings | ST-GCN (Sentiment Transformer Graph Convolutional Network) | Accuracy | 94.94% |
| Tan et al. (2022) | IMDB | Không nêu chi tiết | RoBERTa embeddings | Ensemble (RoBERTa-LSTM + RoBERTa-BiLSTM + RoBERTa-GRU) | Accuracy | 94.9% |

## Chú thích:
- CNN: Convolutional Neural Network
- LSTM: Long Short-Term Memory
- BiLSTM: Bidirectional LSTM
- GRU: Gated Recurrent Unit
- TF-IDF: Term Frequency–Inverse Document Frequency
- MNB: Multinomial Naive Bayes
- KNN: K-Nearest Neighbors
- LR: Logistic Regression
- RF: Random Forest
- MLP: Multilayer Perceptron
- CBRNN: Convolutional Bi-directional Recurrent Neural Network
- ST-GCN: Sentiment Transformer Graph Convolutional Network