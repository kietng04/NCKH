# 2. Dataset & Literature Review

## 2.1. Giới thiệu về IMDb 50K Movie Reviews

IMDb 50K Movie Reviews là một bộ dữ liệu chuẩn và phổ biến trong lĩnh vực xử lý ngôn ngữ tự nhiên (NLP), đặc biệt cho các bài toán phân loại cảm xúc (sentiment analysis). Bộ dữ liệu này bao gồm 50.000 đánh giá phim được thu thập từ Internet Movie Database (IMDb), trong đó mỗi đánh giá được gán nhãn "pos" (tích cực) hoặc "neg" (tiêu cực) [1, 2].

### 2.1.1. Đặc điểm cấu trúc

Dataset được chia thành hai phần riêng biệt với sự phân bố cân bằng:
- 25.000 mẫu cho tập huấn luyện (train)
- 25.000 mẫu cho tập kiểm tra (test)
- Mỗi tập đảm bảo sự cân bằng giữa hai nhãn (12.500 đánh giá tích cực và 12.500 đánh giá tiêu cực)

Sự cân bằng này giúp các mô hình máy học tránh được hiện tượng thiên lệch dữ liệu và cho phép đánh giá khách quan hiệu suất phân loại [3].

### 2.1.2. Đặc điểm nổi bật

IMDb 50K Movie Reviews có một số đặc điểm làm nó trở thành lựa chọn phổ biến trong nghiên cứu NLP:

1. **Độ dài văn bản đa dạng**: Từ các câu ngắn đến đoạn văn dài, phản ánh đa dạng phong cách viết của người dùng thực.
2. **Phức tạp về ngôn ngữ**: Chứa nhiều cách diễn đạt mỉa mai, từ ngữ đa nghĩa và ngữ cảnh phức tạp, thách thức các mô hình NLP.
3. **Tính ứng dụng cao**: Ngoài phân loại cảm xúc, dataset còn được sử dụng để nghiên cứu các tác vụ như tóm tắt văn bản hoặc sinh ngôn ngữ tự nhiên.
4. **Hỗ trợ thư viện**: Được tích hợp sẵn trong nhiều thư viện máy học như PyTorch-NLP, TensorFlow Datasets, giúp nhà nghiên cứu dễ dàng truy cập và tiền xử lý.
5. **Minh họa rõ ràng**: Các đánh giá thường kết hợp giữa nội dung phim và cảm xúc cá nhân, ví dụ: "The plot is simplistic, but the dialogue is witty and the characters are likable..." [11], giúp mô hình học được sự tương quan giữa cấu trúc câu và ý nghĩa.

## 2.2. Tổng quan các phương pháp trong Literature Review

Qua nghiên cứu các công trình đã công bố, có thể chia các phương pháp phân tích cảm xúc trên IMDb 50K Movie Reviews thành ba nhóm chính:

### 2.2.1. Phương pháp Machine Learning truyền thống

Các phương pháp này sử dụng kỹ thuật học máy cổ điển kết hợp với trích xuất đặc trưng thủ công:

- **Naive Bayes, SVM, Maximum Entropy**: Iqbal et al. (2018) [4] áp dụng các kỹ thuật này kết hợp với tiền xử lý tokenization, lemmatization và trích xuất đặc trưng Unigram/Bigram, đạt độ chính xác 88%.

### 2.2.2. Phương pháp Deep Learning

Các mô hình học sâu đã chứng minh hiệu quả vượt trội trong phân tích cảm xúc trên IMDb dataset:

- **Mạng tích chập (CNN)**: Dholpuria et al. (2018) [5] sử dụng CNN với Count Vectorizer sau khi loại bỏ ký tự đặc biệt và stop words, đạt độ chính xác 99.33%.
- **Mạng hồi quy (RNN, LSTM, GRU)**: Thinh et al. (2019) [7] áp dụng kiến trúc 1D-CNN kết hợp với GRU, đạt 90.02% độ chính xác.
- **Mô hình lai CNN-LSTM**: Jang et al. (2020) [6] đề xuất mô hình kết hợp CNN và BiLSTM với cơ chế Attention, sử dụng word2vec để mã hóa văn bản, đạt 90.26% độ chính xác.

### 2.2.3. Phương pháp dựa trên Transformer

Các mô hình dựa trên kiến trúc Transformer đã tạo bước đột phá mới trong những năm gần đây:

- **BERT và biến thể**: Kokab et al. (2022) [9] phát triển BERT-based CBRNN kết hợp BERT với mạng tích chập và BiLSTM, đạt 93% độ chính xác.
- **RoBERTa-LSTM**: Tan et al. (2022a) [8] kết hợp RoBERTa với LSTM để học ngữ cảnh và trình tự, đạt 92.96% độ chính xác.
- **Sentiment Transformer GCN (ST-GCN)**: AlBadani et al. (2022) [10] xây dựng đồ thị dị sentiment và học embedding dựa trên quan hệ từ, đạt 94.94% độ chính xác.

### 2.2.4. Phương pháp Ensemble Learning

- **Hybrid Ensemble Models**: Tan et al. (2022b) [12] đề xuất kết hợp nhiều mô hình (RoBERTa-LSTM + RoBERTa-BiLSTM + RoBERTa-GRU) để tận dụng điểm mạnh của từng kiến trúc, đạt 94.9% độ chính xác.

## 2.3. Bảng tổng hợp các phương pháp và kết quả

| Phương pháp          | Mô hình/Kỹ thuật                     | Tiền xử lý                                | Trích xuất đặc trưng       | Độ chính xác | Nguồn                  |
|----------------------|--------------------------------------|-------------------------------------------|----------------------------|--------------|------------------------|
| Machine Learning     | Naive Bayes, SVM, Maximum Entropy    | Tokenization, lemmatization               | Unigram, Bigram            | 88.00%       | Iqbal et al. (2018) [4]|
| Deep Learning        | CNN                                  | Loại bỏ ký tự đặc biệt, stop words        | Count Vectorizer           | 99.33%       | Dholpuria et al. (2018) [5]|
| Deep Learning        | Hybrid CNN + BiLSTM với Attention    | Mã hóa bằng word2vec                      | Word Embedding (word2vec)  | 90.26%       | Jang et al. (2020) [6]|
| Deep Learning        | 1D-CNN + GRU                         | Không đề cập chi tiết                     | Không đề cập               | 90.02%       | Thinh et al. (2019) [7]|
| Deep Learning        | RoBERTa-LSTM                         | Biến đổi văn bản thành embedding RoBERTa  | Contextual Embedding       | 92.96%       | Tan et al. (2022a) [8]|
| Deep Learning        | BERT-based CBRNN                     | BERT tokenization                         | Pretrained BERT Embedding  | 93.00%       | Kokab et al. (2022) [9]|
| Ensemble Learning    | RoBERTa-LSTM + RoBERTa-BiLSTM + GRU  | RoBERTa tokenization                      | Contextual Embedding       | 94.90%       | Tan et al. (2022b) [12]|
| Deep Learning        | ST-GCN                               | Xây dựng đồ thị sentiment                 | Graph-based Embedding      | 94.94%       | AlBadani et al. (2022) [10]|

## 2.4. Phân tích xu hướng và nhận xét

Từ kết quả tổng hợp, có thể rút ra một số nhận xét quan trọng:

1. **Hiệu suất theo phương pháp**: Các phương pháp Machine Learning truyền thống (88%) nhìn chung cho kết quả thấp hơn so với Deep Learning (90-95%) và Ensemble Learning (>94%).
2. **Vai trò của Transformer**: Các mô hình dựa trên Transformer (RoBERTa, BERT) kết hợp với LSTM/GRU đạt hiệu suất cao nhất (>92%).
3. **Tầm quan trọng của cơ chế Attention**: Các kiến trúc tích hợp cơ chế Attention giúp cải thiện độ chính xác đáng kể, như trường hợp của Jang et al. (2020) [6].
4. **Tiềm năng của Graph Neural Networks**: ST-GCN của AlBadani et al. (2022) [10] đạt kết quả cao nhất (94.94%) nhờ khai thác mối quan hệ giữa các từ trong văn bản.
5. **Tiền xử lý dữ liệu**: Việc sử dụng các pretrained embedding (word2vec, GloVe, BERT, RoBERTa) là yếu tố quan trọng góp phần vào hiệu suất cao của các mô hình.

## 2.5. Paper Key: Phân tích và Đề xuất Cải tiến

Từ các kết quả tổng hợp, chúng tôi chọn mô hình "Hybrid CNN + BiLSTM với Attention" của Jang et al. (2020) [6] làm paper key để phân tích chi tiết và đề xuất cải tiến.

### 2.5.1. Tổng quan mô hình Hybrid CNN + BiLSTM với Attention

Mô hình này kết hợp ba thành phần chính tạo nên một kiến trúc mạnh mẽ:
- **CNN**: Trích xuất các đặc trưng cục bộ và nhận diện các mẫu ngôn ngữ quan trọng
- **BiLSTM**: Nắm bắt ngữ cảnh hai chiều và các phụ thuộc xa trong văn bản
- **Cơ chế Attention**: Tập trung vào các phần quan trọng của văn bản khi đưa ra quyết định

Kiến trúc này hoạt động qua các bước:
1. Biến đổi văn bản thành vector embedding sử dụng word2vec
2. Trích xuất các đặc trưng cục bộ bằng các lớp tích chập
3. Đưa các đặc trưng này vào BiLSTM để học ngữ cảnh hai chiều
4. Áp dụng cơ chế Attention để xác định tầm quan trọng của từng phần trong chuỗi
5. Sử dụng lớp Fully Connected cuối cùng để phân loại cảm xúc

### 2.5.2. Ưu điểm của mô hình

- **Kết hợp đa dạng đặc trưng**: CNN nắm bắt thông tin cục bộ, trong khi BiLSTM nắm bắt ngữ cảnh rộng hơn
- **Hiểu ngữ cảnh hai chiều**: BiLSTM xử lý văn bản theo cả hai hướng, giúp hiểu đầy đủ ngữ cảnh
- **Tập trung vào thông tin quan trọng**: Cơ chế Attention giúp mô hình tập trung vào các từ hoặc cụm từ quan trọng nhất để đưa ra quyết định
- **Cân bằng giữa hiệu suất và tính thực tiễn**: Đạt 90.26% độ chính xác với kiến trúc có thể triển khai được

### 2.5.3. Hạn chế và đề xuất cải tiến

Dựa trên phân tích chi tiết, chúng tôi đề xuất một cải tiến đơn giản nhưng tiềm năng:

**Thay thế BiLSTM bằng BiGRU**

- **Hạn chế hiện tại**: BiLSTM có nhiều tham số và tốc độ huấn luyện chậm
- **Đề xuất**: Thay thế BiLSTM bằng BiGRU (Bidirectional Gated Recurrent Unit)
- **Lý do chọn cải tiến này**:
  1. Dễ triển khai: Chỉ cần thay đổi layer LSTM thành GRU trong mã nguồn
  2. Tốc độ huấn luyện nhanh hơn: GRU có ít tham số hơn LSTM (2 cổng thay vì 3 cổng)
  3. Hiệu suất tương đương: Nhiều nghiên cứu cho thấy GRU có thể đạt hiệu suất tương đương LSTM
  4. Tiết kiệm bộ nhớ: Giảm số lượng tham số giúp giảm dung lượng mô hình

## 2.6. Kết luận

Qua tổng quan literature review, có thể thấy rõ sự phát triển của các phương pháp phân tích cảm xúc trên IMDb 50K dataset, từ machine learning truyền thống đến các kiến trúc học sâu phức tạp. Các mô hình lai kết hợp Transformer với mạng hồi quy hoặc tích chập đang chiếm ưu thế với hiệu suất vượt trội.

Đề xuất cải tiến mô hình "Hybrid CNN + BiLSTM với Attention" bằng cách thay thế BiLSTM bằng BiGRU hứa hẹn cải thiện hiệu quả tính toán và thời gian huấn luyện trong khi vẫn duy trì hiệu suất tương đương. Kết quả của các thực nghiệm này sẽ đóng góp vào sự phát triển liên tục của lĩnh vực phân tích cảm xúc và xử lý ngôn ngữ tự nhiên.

## Tài liệu tham khảo

1. Maas, A.; Daly, R.E.; Pham, P.T.; Huang, D.; Ng, A.Y.; Potts, C. (2011). Learning word vectors for sentiment analysis. In Proceedings of the IEEE 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, Portland, OR, USA, 19–24 June 2011; pp. 142–150. https://doi.org/10.5555/2002472.2002491
2. Dang, N.C.; Moreno-García, M.N.; De la Prieta, F. (2020). Sentiment analysis based on deep learning: A comparative study. Electronics, 9(3), 483. https://doi.org/10.3390/electronics9030483
3. Chakriswaran, P.; Vincent, D.R.; Srinivasan, K.; Sharma, V.; Chang, C.Y.; Reina, D.G. (2019). Emotion AI-driven sentiment analysis: A survey, future research directions, and open issues. Applied Sciences, 9(24), 5462. https://doi.org/10.3390/app9245462
4. Iqbal, N.; Chowdhury, A.M.; Ahsan, T. (2018). Enhancing the performance of sentiment analysis by using different feature combinations. In Proceedings of the 2018 IEEE International Conference on Computer, Communication, Chemical, Material and Electronic Engineering (IC4ME2), Rajshahi, Bangladesh, 8–9 February 2018; pp. 1–4. https://doi.org/10.1109/IC4ME2.2018.8465565
5. Dholpuria, T.; Rana, Y.; Agrawal, C. (2018). A sentiment analysis approach through deep learning for a movie review. In Proceedings of the 2018 IEEE 8th International Conference on Communication Systems and Network Technologies (CSNT), Bhopal, India, 24–26 November 2018; pp. 173–181. https://doi.org/10.1109/CSNT.2018.8820244
6. Jang, B.; Kim, M.; Harerimana, G.; Kang, S.U.; Kim, J.W. (2020). Bi-LSTM model to increase accuracy in text classification: Combining Word2vec CNN and attention mechanism. Applied Sciences, 10(17), 5841. https://doi.org/10.3390/app10175841
7. Thinh, N.K.; Nga, C.H.; Lee, Y.S.; Wu, M.L.; Chang, P.C.; Wang, J.C. (2019). Sentiment Analysis Using Residual Learning with Simplified CNN Extractor. In Proceedings of the 2019 IEEE International Symposium on Multimedia (ISM), San Diego, CA, USA, 9–11 December 2019; pp. 335–3353. https://doi.org/10.1109/ISM46123.2019.00073
8. Tan, K.L.; Lee, C.P.; Anbananthen, K.S.M.; Lim, K.M. (2022a). RoBERTa-LSTM: A Hybrid Model for Sentiment Analysis with Transformer and Recurrent Neural Network. IEEE Access, 10, 21517–21525. https://doi.org/10.1109/ACCESS.2022.3151417
9. Kokab, S.T.; Asghar, S.; Naz, S. (2022). Transformer-based deep learning models for the sentiment analysis of social media data. Array, 14, 100157. https://doi.org/10.1016/j.array.2022.100157
10. AlBadani, B.; Shi, R.; Dong, J.; Al-Sabri, R.; Moctard, O.B. (2022). Transformer-based graph convolutional network for sentiment analysis. Applied Sciences, 12(3), 1316. https://doi.org/10.3390/app12031316
11. Ligthart, A.; Catal, C.; Tekinerdogan, B. (2021). Systematic reviews in sentiment analysis: A tertiary study. Artificial Intelligence Review, 54, 4997–5053. https://doi.org/10.1007/s10462-021-09973-3
12. Tan, K.L.; Lee, C.P.; Lim, K.M.; Anbananthen, K.S.M. (2022b). Sentiment Analysis with Ensemble Hybrid Deep Learning Model. IEEE Access, 10, 103694–103704. https://doi.org/10.1109/ACCESS.2022.3203816