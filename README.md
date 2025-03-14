# THÔNG TIN NHÓM

|Tên                                |MSSV      |Portfolio         |Email|
|-----------------------------------|  :----:  |------------------|-----|
|Nguyễn Phan Tuấn Kiệt (Trưởng nhóm)|3122410193|[https://kietng04.github.io/portfolio/](https://kietng04.github.io/portfolio/) |nguyenphantuankiet299@gmail.com|
|Nguyễn Thế Kiên                    |3122410194|[https://tkieen.github.io/PersonalWebsite/](https://tkieen.github.io/PersonalWebsite/) |nguyenthekien62@gmail.com|
|Phạm Văn Kiệt                      |3122410200|[https://vankiet04.github.io/portfolio/](https://vankiet04.github.io/portfolio) |pvk210504@gmail.com|

# A Sentiment Analysis Framework for Movie Evaluation Based on User Comments

## Tổng quan dự án
Dự án này tập trung vào việc xây dựng một khung phân tích cảm xúc (sentiment analysis) để đánh giá phim dựa trên bình luận của người dùng. Cụ thể, nghiên cứu sẽ phát triển và đánh giá các mô hình học máy có khả năng phân loại bình luận phim thành các nhóm tích cực hoặc tiêu cực, giúp hiểu được cảm nhận tổng thể của người xem về một bộ phim.

## Mô tả chi tiết
Phân tích cảm xúc là một lĩnh vực quan trọng trong xử lý ngôn ngữ tự nhiên (NLP), cho phép tự động phát hiện và phân loại ý kiến, cảm xúc từ văn bản. Trong dự án này, nhóm em sẽ tập trung vào việc phân tích bình luận phim từ IMDB - một trong những cơ sở dữ liệu đánh giá phim lớn nhất thế giới. Bộ dữ liệu IMDB 50K reviews bao gồm 50.000 bình luận đã được gán nhãn tích cực/tiêu cực, cung cấp nguồn dữ liệu phong phú để huấn luyện và đánh giá các mô hình phân tích cảm xúc.

## Mục tiêu dự án
1. Xây dựng mô hình mạng nơ-ron Fully Connected để phân loại cảm xúc trong bình luận phim
2. So sánh hiệu suất của mô hình Fully Connected với các mô hình tiên tiến khác như LSTM
3. Đánh giá hiệu quả của các phương pháp biểu diễn từ khác nhau (FastText và GloVe Embedding)
4. Phân tích các yếu tố ảnh hưởng đến độ chính xác của việc phân loại cảm xúc
5. Xây dựng một khung đánh giá toàn diện cho việc phân tích cảm xúc trong lĩnh vực đánh giá phim

## Dữ liệu sử dụng
Bộ dữ liệu [IMDB 50K Movie Reviews từ Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/code?datasetId=134715&sortBy=voteCount&searchQuery=Exploratory+Data+Analysis), bao gồm:
- 50.000 bình luận phim đã được gán nhãn (25.000 tích cực và 25.000 tiêu cực)
- Mỗi bình luận đều có điểm đánh giá từ người dùng
- Bộ dữ liệu được chia thành tập huấn luyện và tập kiểm tra với tỷ lệ 80:20
- Dữ liệu đã được tiền xử lý để loại bỏ thông tin nhận dạng cá nhân

## Phương pháp nghiên cứu

### Tiền xử lý dữ liệu
1. Làm sạch văn bản: loại bỏ HTML tags, dấu câu, chuyển về chữ thường
2. Loại bỏ stopwords (các từ không mang nhiều ý nghĩa ngữ nghĩa)
3. Tokenization: chuyển văn bản thành các token riêng biệt
4. Stemming/Lemmatization: chuyển các từ về dạng gốc
5. Xử lý các từ viết tắt và biểu tượng cảm xúc

### Trích xuất đặc trưng
1. Bag of Words (BoW)
2. TF-IDF (Term Frequency-Inverse Document Frequency)
3. FastText Embedding: sử dụng mô hình FastText để biểu diễn từ dựa trên ngữ cảnh
4. GloVe Embedding: sử dụng biểu diễn từ toàn cục dựa trên thống kê đồng xuất hiện

### Mô hình chính
**Mạng nơ-ron Fully Connected** với cấu trúc:
- Lớp nhập: kích thước tùy thuộc vào phương pháp biểu diễn từ
- Các lớp ẩn với các hàm kích hoạt ReLU
- Lớp Dropout để tránh overfitting
- Lớp xuất với hàm kích hoạt sigmoid cho phân loại nhị phân

### Các mô hình so sánh
1. **LSTM (Long Short-Term Memory)**: Mô hình mạng nơ-ron hồi quy có khả năng nắm bắt phụ thuộc dài hạn trong chuỗi văn bản
2. **Mô hình kết hợp CNN-LSTM**: Sử dụng CNN để trích xuất đặc trưng cục bộ và LSTM để nắm bắt phụ thuộc thời gian
3. **Mô hình truyền thống**: SVM, Naive Bayes, Random Forest

### Đánh giá mô hình
1. Độ chính xác (Accuracy)
2. Độ chính xác (Precision)
3. Độ nhạy (Recall)
4. Điểm F1 (F1-Score)
5. Diện tích dưới đường cong ROC (AUC-ROC)
6. Ma trận nhầm lẫn (Confusion Matrix)

## Công nghệ sử dụng
- **Ngôn ngữ lập trình**: Python 3.8
- **Thư viện chính**:
  - Pandas: xử lý và phân tích dữ liệu
  - NumPy: tính toán số học
  - tqdm: hiển thị thanh tiến trình
  - scikit-learn: các thuật toán học máy và đánh giá
  - TensorFlow/Keras: xây dựng và huấn luyện mô hình deep learning
  - NLTK/spaCy: xử lý ngôn ngữ tự nhiên
  - Matplotlib/Seaborn: trực quan hóa dữ liệu và kết quả

## Kế hoạch thực hiện
1. **Thu thập và khám phá dữ liệu**: Tải bộ dữ liệu IMDB 50K, phân tích phân phối, độ dài bình luận
2. **Tiền xử lý dữ liệu**: Làm sạch văn bản, tokenization, xử lý từ vựng
3. **Trích xuất đặc trưng**: Thử nghiệm các phương pháp biểu diễn từ khác nhau
4. **Xây dựng mô hình cơ sở**: Mô hình Fully Connected đơn giản
5. **Tối ưu hóa siêu tham số**: Điều chỉnh cấu trúc mạng, tỷ lệ học, kích thước batch
6. **Thực hiện các mô hình so sánh**: LSTM, FastText, GloVe
7. **Đánh giá và phân tích kết quả**: So sánh hiệu suất các mô hình
8. **Cải tiến mô hình**: Áp dụng các kỹ thuật nâng cao như attention mechanism, transfer learning
9. **Tài liệu hóa và báo cáo**: Viết báo cáo chi tiết về phương pháp và kết quả

## Kết quả dự kiến
1. Một khung phân tích cảm xúc hoàn chỉnh cho đánh giá phim
2. So sánh chi tiết hiệu suất của các mô hình và phương pháp biểu diễn từ khác nhau
3. Phân tích về các yếu tố ảnh hưởng đến hiệu suất phân loại cảm xúc
4. Mô hình có khả năng phân loại chính xác bình luận phim thành tích cực/tiêu cực
5. Báo cáo chi tiết và mã nguồn có thể tái sử dụng cho các dự án phân tích cảm xúc tương tự

## Ý nghĩa thực tiễn
Khung phân tích cảm xúc này có thể được áp dụng trong nhiều lĩnh vực:
- Hỗ trợ các nhà sản xuất phim đánh giá phản hồi của khán giả
- Giúp người dùng tìm kiếm phim phù hợp với sở thích cá nhân
- Cung cấp công cụ cho các nền tảng streaming để cải thiện hệ thống đề xuất
- Mở rộng sang phân tích cảm xúc trong các lĩnh vực khác như đánh giá sản phẩm, phân tích mạng xã hội

## Thách thức và giới hạn
1. Xử lý ngữ cảnh và sắc thái trong ngôn ngữ (như châm biếm, mỉa mai)
2. Đối phó với sự mất cân bằng trong dữ liệu
3. Xử lý các bình luận dài và phức tạp
4. Tối ưu hóa hiệu suất mô hình cho các ứng dụng thời gian thực

## Hướng phát triển tương lai
1. Mở rộng sang phân tích đa lớp (không chỉ tích cực/tiêu cực mà còn các mức độ khác nhau)
2. Tích hợp phân tích khía cạnh (aspect-based sentiment analysis)
3. Áp dụng các mô hình ngôn ngữ tiên tiến như BERT, GPT
4. Phát triển ứng dụng web/mobile để trực quan hóa kết quả phân tích


# SCHEDULE
**[Google Sheets](https://docs.google.com/spreadsheets/d/1R6TcgMAHRGDLy_LWjGpTvcf478gThrEl4cTsUK8NyQ8/edit?gid=94895279#gid=94895279)**
