# BÀI 4

|Họ và tên                          |MSSV      | Nhiệm vụ                                                             |
|-----------------------------------|----------|----------------------------------------------------------------------|
|Nguyễn Phan Tuấn Kiệt (Trưởng nhóm)|3122410193|Tìm dataset, literature review, paper, code EDA, input, output, metric|
|Nguyễn Thế Kiên                    |3122410194|Kiểm tra lại bài tuần 3 và bổ sung reference, làm phần EDA|
|Phạm Văn Kiệt                      |3122410200|Làm phần EDA, giới thiệu, định nghĩa, phân tích, chuẩn bị dữ liệu |
# BÀI 4 tiếp theo

|Họ và tên                          |MSSV      | Nhiệm vụ                                                             |
|-----------------------------------|----------|----------------------------------------------------------------------|
|Nguyễn Phan Tuấn Kiệt (Trưởng nhóm)|3122410193|Viết khảo toàn bộ sát đề tài|
|Nguyễn Thế Kiên                    |3122410194|Làm slide khảo sát đề tài và làm slide khảo sát|
|Phạm Văn Kiệt                      |3122410200| TÌm hiểu nội dung khảo sát, key paper để thuyết trình

# PHÂN CÔNG ĐỀ TÀI CUỐI KHOÁ

## Lộ Trình 8 Tuần: Phát Triển Mô Hình Deep Learning Phân Tích Cảm Xúc

### Tuần 1: Nghiên Cứu Nền Tảng và Chuẩn Bị Môi Trường
- **Công việc 1.1**: Nghiên cứu tổng quan về phân tích cảm xúc *(sentiment analysis)* và các phương pháp deep learning hiện đại
  - Đọc và tóm tắt 5-7 bài báo liên quan đến phân tích cảm xúc sử dụng deep learning
  - Tìm hiểu về dataset IMDB 50K và các đặc điểm của nó
- **Công việc 1.2**: Chuẩn bị môi trường phát triển
  - Cài đặt các thư viện cần thiết *(PyTorch, TensorFlow, sklearn, pandas, numpy)*
  - Tạo cấu trúc thư mục dự án theo tiêu chuẩn
  - Tải và kiểm tra dataset IMDB
- **Đầu ra**: Báo cáo tổng quan về phân tích cảm xúc và môi trường phát triển hoàn chỉnh.

### Tuần 2: Khám Phá và Tiền Xử Lý Dữ Liệu
- **Công việc 2.1**: Phân tích khám phá dữ liệu *(EDA)*
  - Phân tích phân phối của nhãn *(tích cực/tiêu cực)*
  - Phân tích độ dài bình luận
  - Xác định các từ phổ biến nhất
  - Trực quan hóa dữ liệu *(biểu đồ, đồ thị)*
- **Công việc 2.2**: Tiền xử lý dữ liệu
  - Loại bỏ thẻ HTML
  - Chuyển đổi chữ thường
  - Loại bỏ dấu câu và stopwords
  - Phân chia dữ liệu thành tập huấn luyện và tập kiểm tra
- **Đầu ra**: Báo cáo EDA với hình ảnh trực quan và dữ liệu đã được tiền xử lý.

### Tuần 3: Chuẩn Bị Word Embeddings và Thiết Kế Mô Hình
- **Công việc 3.1**: Tải và chuẩn bị pre-trained word embeddings
  - Tải FastText embeddings
  - Tải GloVe embeddings
  - Tạo embedding matrix cho từ điển
- **Công việc 3.2**: Thiết kế kiến trúc mô hình
  - Nghiên cứu và so sánh các kiến trúc LSTM, BiLSTM, và CNN-LSTM
  - Thiết kế lớp đầu vào và output cho mô hình
  - Xác định các tham số mô hình *(learning rate, batch size, epochs)*
- **Đầu ra**: Báo cáo về việc chuẩn bị word embeddings và thiết kế mô hình.

### Tuần 4: Xây Dựng và Huấn Luyện Mô Hình LSTM với FastText
- **Công việc 4.1**: Cài đặt mô hình LSTM với FastText embedding
  - Cài đặt lớp Dataset và DataLoader
  - Cài đặt kiến trúc mô hình LSTM
  - Cài đặt hàm mất mát và phương pháp tối ưu
- **Công việc 4.2**: Huấn luyện và đánh giá mô hình
  - Huấn luyện mô hình với 5 epochs
  - Theo dõi độ chính xác trên tập validation
  - Trực quan hóa quá trình huấn luyện
- **Đầu ra**: Mô hình LSTM với FastText đã được huấn luyện và báo cáo hiệu suất.

### Tuần 5: Xây Dựng và Huấn Luyện Mô Hình LSTM với GloVe
- **Công việc 5.1**: Cài đặt mô hình LSTM với GloVe embedding
  - Tích hợp GloVe embeddings vào mô hình
  - Điều chỉnh kiến trúc nếu cần thiết
- **Công việc 5.2**: Huấn luyện và đánh giá mô hình
  - Huấn luyện mô hình với 5 epochs
  - So sánh với mô hình FastText
  - Phân tích ưu nhược điểm của từng phương pháp embedding
- **Đầu ra**: Mô hình LSTM với GloVe đã được huấn luyện và báo cáo so sánh với FastText.

### Tuần 6: Tối Ưu Hóa Mô Hình và Thử Nghiệm Kiến Trúc Nâng Cao
- **Công việc 6.1**: Tối ưu hóa mô hình hiện tại
  - Tinh chỉnh tham số *(hyperparameter tuning)*
  - Thử nghiệm các kỹ thuật điều chỉnh *(regularization)*
  - Áp dụng cross-validation để đánh giá độ tin cậy
- **Công việc 6.2**: Thử nghiệm với kiến trúc nâng cao
  - Cài đặt mô hình CNN-LSTM kết hợp
  - Huấn luyện và so sánh với các mô hình trước đó
- **Đầu ra**: Báo cáo tối ưu hóa và so sánh các kiến trúc mô hình.

### Tuần 7: Xây Dựng Giao Diện Tương Tác và Đánh Giá Toàn Diện
- **Công việc 7.1**: Xây dựng giao diện tương tác
  - Cài đặt hàm để xử lý đầu vào của người dùng
  - Tạo chức năng dự đoán sentiment từ văn bản tùy ý
- **Công việc 7.2**: Đánh giá toàn diện các mô hình
  - So sánh hiệu suất giữa các mô hình *(FastText, GloVe, CNN-LSTM)*
  - Phân tích các trường hợp dự đoán sai
  - Trực quan hóa so sánh các metric
- **Đầu ra**: Giao diện tương tác và báo cáo đánh giá tổng thể.

### Tuần 8: Viết Báo Cáo và Chuẩn Bị Thuyết Trình
- **Công việc 8.1**: Viết báo cáo nghiên cứu khoa học
  - Tổng hợp các phần: giới thiệu, phương pháp, kết quả, thảo luận
  - Tạo các hình ảnh và bảng minh họa chất lượng cao
  - Rà soát và chỉnh sửa báo cáo
- **Công việc 8.2**: Chuẩn bị thuyết trình và tài liệu bổ sung
  - Tạo slide thuyết trình
  - Chuẩn bị demo trực tiếp của mô hình
  - Tổ chức mã nguồn và tài liệu để chia sẻ
- **Đầu ra**: Báo cáo nghiên cứu hoàn chỉnh, slide thuyết trình và mã nguồn đã được tổ chức.