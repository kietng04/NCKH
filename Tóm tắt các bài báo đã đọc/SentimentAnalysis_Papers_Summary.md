# Tóm tắt: Phân tích cảm xúc

## Giới thiệu
Phân tích tình cảm là một lĩnh vực quan trọng trong Xử lý Ngôn ngữ Tự nhiên (NLP), tập trung vào việc xác định thái độ (tích cực, tiêu cực, trung lập) được thể hiện trong văn bản. Hai bài báo dưới đây đóng góp vào lĩnh vực này theo những cách khác nhau: bài báo [1] xem xét lại và sửa chữa một kết quả được công bố trước đó, trong khi bài báo [2] đề xuất một kiến trúc mô hình lai mới.

## Bài báo [1]: The Document Vectors Using Cosine Similarity Revisited

Bài báo này [1] tập trung vào việc đánh giá lại mô hình "Document Vectors using Cosine Similarity" (DV-ngrams-cosine) được đề xuất bởi Thongtan và Phienthrakul (2019), vốn được báo cáo là đạt độ chính xác cao nhất (state-of-the-art - SOTA) 97.42% trên tập dữ liệu đánh giá phim IMDB [1]. Mô hình này là một ensemble kết hợp giữa embedding DV-ngrams-cosine và vector Bag-of-N-grams trọng số Naive Bayes (NB-weighted BON) [1].

### Phát hiện lỗi:
Các tác giả của bài báo [1] đã phát hiện ra một lỗi nghiêm trọng trong quy trình đánh giá của nghiên cứu gốc [1]. Cụ thể, trong quá trình tạo vector đầu vào cho bộ phân loại ensemble, việc ghép nối (concatenation) vector DV-ngrams-cosine và vector NB-weighted BON đã được thực hiện không chính xác. Thay vì ghép nối các vector đại diện cho cùng một tài liệu, mã nguồn gốc đã ghép nối các vector từ hai tài liệu khác nhau nhưng thuộc cùng một lớp (tích cực/tiêu cực) và cùng một tập con (huấn luyện/kiểm tra) [1].

### Hậu quả của lỗi:
Việc ghép nối sai này vô tình đã làm rò rỉ thông tin về nhãn trong quá trình kiểm tra. Khi một tài liệu "khó" (khó phân loại) được ghép nối với một tài liệu "dễ" cùng lớp, bộ phân loại có xu hướng dự đoán đúng dựa trên tín hiệu mạnh từ tài liệu "dễ", dẫn đến độ chính xác bị thổi phồng một cách giả tạo [1].

### Kết quả đã sửa chữa:
Sau khi sửa lỗi và đảm bảo việc ghép nối vector được thực hiện cho cùng một tài liệu, độ chính xác kiểm tra của mô hình ensemble giảm đáng kể xuống còn 93.68% [1]. Con số này chỉ cao hơn 0.55% so với việc chỉ sử dụng embedding DV-ngrams-cosine đơn thuần, thay vì mức cải thiện 4.29% như báo cáo ban đầu [1].

### Phân tích bổ sung:
Bài báo [1] so sánh hiệu suất của DV-ngrams-cosine với mô hình RoBERTa dựa trên Transformer trên các tập huấn luyện có kích thước khác nhau. Kết quả cho thấy RoBERTa vượt trội hơn khi có nhiều dữ liệu huấn luyện, nhưng đáng ngạc nhiên là DV-ngrams-cosine lại hoạt động tốt hơn RoBERTa khi tập huấn luyện có nhãn rất nhỏ (10 hoặc 20 tài liệu) [1].

Họ cũng đề xuất một kỹ thuật lấy mẫu con dựa trên trọng số Naive Bayes (NB Sub-Sampling) cho quá trình huấn luyện DV-ngrams-cosine. Kỹ thuật này giúp tăng tốc độ huấn luyện và cải thiện nhẹ chất lượng embedding [1].

Việc kết hợp RoBERTa với DV-ngrams-cosine chỉ mang lại cải thiện rất nhỏ (0.13-0.15%) [1].

### Kết luận [1]:
Nghiên cứu này chỉ ra tầm quan trọng của việc kiểm tra kỹ lưỡng quy trình đánh giá và cung cấp một kết quả chính xác hơn cho mô hình DV-ngrams-cosine trên tập IMDB. Nó cũng nhấn mạnh hiệu quả của các mô hình đơn giản hơn trong các tình huống dữ liệu thấp [1].

## Bài báo [2]: ROBERTa-LSTM: A hybrid model for sentiment analysis with transformer and recurrent neural network

Bài báo này [2] đề xuất một mô hình học sâu lai mới có tên là RoBERTa-LSTM để giải quyết bài toán phân tích tình cảm [2]. Mục tiêu là kết hợp sức mạnh của các mô hình Transformer (cụ thể là RoBERTa) trong việc tạo ra các embedding ngữ cảnh phong phú và khả năng xử lý song song, với sức mạnh của các mô hình mạng nơ-ron hồi quy (RNN, cụ thể là LSTM) trong việc nắm bắt các phụ thuộc tuần tự tầm xa trong văn bản [2].

### Kiến trúc mô hình:
Mô hình sử dụng RoBERTa (một biến thể tối ưu hóa của BERT) đã được huấn luyện trước để tạo ra các vector embedding cho các từ hoặc từ con (subword) trong văn bản đầu vào [2].

Đầu ra từ RoBERTa (các embedding ngữ cảnh) sau đó được đưa vào một lớp Long Short-Term Memory (LSTM) [2]. Lớp LSTM này có nhiệm vụ nắm bắt các mối quan hệ và phụ thuộc theo trình tự thời gian trong chuỗi embedding [2].

Kiến trúc này được thiết kế để RoBERTa xử lý việc mã hóa ngữ nghĩa cục bộ hiệu quả, còn LSTM tập trung vào việc mô hình hóa các mối quan hệ tuần tự dài hạn [2].

### Tiền xử lý và Tăng cường dữ liệu:
Các bước tiền xử lý văn bản tiêu chuẩn được áp dụng, bao gồm chuyển thành chữ thường, loại bỏ từ dừng (stop words), dấu câu, số và ký tự đặc biệt, và chuẩn hóa từ gốc (stemming) [2].

Kỹ thuật tăng cường dữ liệu (data augmentation) sử dụng embedding GloVe được áp dụng để giải quyết vấn đề mất cân bằng dữ liệu, đặc biệt là trên tập dữ liệu Twitter US Airline Sentiment. Kỹ thuật này thay thế các từ bằng các từ đồng nghĩa gần nhất trong không gian vector để tạo thêm mẫu cho các lớp thiểu số [2].

### Thực nghiệm và Kết quả:
Mô hình RoBERTa-LSTM được đánh giá trên ba tập dữ liệu: IMDb, Twitter US Airline Sentiment, và Sentiment140 [2].

Kết quả cho thấy RoBERTa-LSTM vượt trội hơn đáng kể so với các phương pháp học máy truyền thống (Naive Bayes, SVM, KNN, Decision Tree, AdaBoost) và các mô hình học sâu khác (GRU, LSTM, BiLSTM, CNN-LSTM, CNN-BiLSTM) trên cả ba tập dữ liệu [2].

Mô hình đạt điểm F1-score cao: 93% trên IMDb, 91% trên Twitter US Airline (sau khi tăng cường dữ liệu), và 90% trên Sentiment140 [2].

Việc tăng cường dữ liệu đã cải thiện đáng kể hiệu suất trên tập Twitter US Airline (từ F1-score ~81% lên 91%), cho thấy hiệu quả của phương pháp này đối với dữ liệu mất cân bằng [2].

### Kết luận [2]:
Bài báo đề xuất thành công một kiến trúc lai RoBERTa-LSTM hiệu quả cho phân tích tình cảm, tận dụng ưu điểm của cả Transformer và RNN. Mô hình này đạt được kết quả SOTA trên các tập dữ liệu được thử nghiệm, đặc biệt khi kết hợp với kỹ thuật tăng cường dữ liệu phù hợp [2].

## Tổng kết
Hai bài báo này cung cấp những hiểu biết giá trị về phân tích tình cảm. Bài báo [1] nhấn mạnh sự cần thiết của việc đánh giá mô hình cẩn thận và điều chỉnh lại một kết quả SOTA trước đó, đồng thời cho thấy tiềm năng của các mô hình cũ hơn trong điều kiện dữ liệu hạn chế. Bài báo [2] đóng góp một kiến trúc lai mới, RoBERTa-LSTM, chứng minh hiệu quả vượt trội bằng cách kết hợp các điểm mạnh của hai họ mô hình học sâu phổ biến.

## Nguồn:
[1] Zhang, B., & Arefyev, N. (2022). The Document Vectors Using Cosine Similarity Revisited. arXiv preprint arXiv:2205.13357.  
[2] Tan, K. L., Lee, C. P., Anbananthen, K. S. M., & Lim, K. M. (2022). ROBERTa-LSTM: A hybrid model for sentiment analysis with transformer and recurrent neural network. IEEE Access, 10, 21517-21525.
