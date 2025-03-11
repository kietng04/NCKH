# Input
Tập dữ liệu IMDB với 50.000 đánh giá phim là nguồn dữ liệu thực nghiệm, đại diện cho một bài toán phân loại cảm xúc thực tế. Quá trình tiền xử lý (tokenization, normalization, word embedding, sequencing) đảm bảo dữ liệu thô được chuyển đổi thành dạng mà các mô hình học sâu có thể xử lý, phản ánh cách tiếp cận hiện đại trong xử lý ngôn ngữ tự nhiên (NLP).

# Output
Trong bài báo này, output là dự đoán nhãn cảm xúc của các đánh giá phim, được phân loại thành hai lớp:
- 1: Đánh giá tích cực (positive).
- 0: Đánh giá tiêu cực (negative).

Các mô hình học sâu (MLP, CNN, LSTM và CNN_LSTM) được huấn luyện để thực hiện nhiệm vụ phân loại nhị phân (binary classification). Cụ thể:

Đối với tập kiểm tra (20% dữ liệu), mô hình dự đoán nhãn cảm xúc cho từng đánh giá.
Kết quả cuối cùng là tập hợp các dự đoán này, được sử dụng để đánh giá hiệu suất của mô hình.

# Metric (Thước đo đánh giá)
Cụ thể trong bài báo sử dụng accuracy (độ chính xác) để làm metric.
Trong đó accuracy được tính theo công thức:

$$\text{Accuracy} = \frac{\text{Tổng số dự đoán đúng}}{\text{Số lượng dự đoán}} \times 100\%$$

Kết quả độ chính xác của các mô hình được báo cáo như sau:

| Mô hình  | Accuracy (%) |
|----------|--------------|
| CNN_LSTM | 89.20 (cao nhất) |
| CNN      | 87.70        |
| MLP      | 86.74        |
| LSTM     | 86.64        |