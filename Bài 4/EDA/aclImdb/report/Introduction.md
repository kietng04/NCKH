# Thông tin chung
- **Tên dataset:** IMDB Dataset of 50K Movie Reviews
- **Kích thước:** 50,000 bình luận phim
- **Phân chia:** 25,000 mẫu huấn luyện và 25,000 mẫu kiểm tra
- **Cân bằng:** Số lượng bình luận tích cực và tiêu cực bằng nhau
- **Nhãn:** Nhị phân *(tích cực/tiêu cực)*
# Nguồn gốc của dataset
 Dataset này được thu thập từ nguồn công khai của Stanford University, được tạo bởi Andrew L. Maas và các cộng sự trong nghiên cứu "Learning Word Vectors for Sentiment Analysis" *(2011)*. Dataset có sẵn trong thư viện Keras và nhiều nền tảng học máy khác như TensorFlow, PyTorch.
# Khoảng thời gian thu thập dữ liệu
Dataset được thu thập vào khoảng năm 2011 từ các bình luận phim trên trang web IMDB. Các bình luận được chọn từ nhiều giai đoạn khác nhau trong lịch sử điện ảnh.
# Cấu trúc dữ liệu
|Biến	    |Loại dữ liệu   |Mô tả                                            |
|-----------|     :----:    |-------------------------------------------------|
|review	    |text	        |Nội dung bình luận phim do người dùng IMDB viết  |
|sentiment  |integer	    |Nhãn cảm xúc: 1 *(tích cực)* hoặc 0 *(tiêu cực)* |
# Đặc điểm bổ sung
- Các bình luận được chọn có điểm đánh giá rõ ràng: bình luận tích cực có điểm ≥ 7/10, bình luận tiêu cực có điểm ≤ 4/10
- Đã loại bỏ các bình luận trung tính *(5-6 điểm)* để tăng độ phân biệt
- Mỗi bộ phim có tối đa 30 bình luận để tránh thiên lệch
- Chỉ chọn các bình luận có ít nhất 10 câu
# Công dụng tiềm năng
- Huấn luyện mô hình phân tích cảm xúc
- Xử lý ngôn ngữ tự nhiên
- Phân loại văn bản
- Đánh giá hiệu suất các thuật toán học máy
### Dataset này là một tiêu chuẩn phổ biến để đánh giá hiệu suất của các thuật toán và kỹ thuật phân loại văn bản, đặc biệt trong lĩnh vực phân tích cảm xúc.