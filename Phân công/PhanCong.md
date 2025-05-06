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
# Bài 6

|Họ và tên                          |MSSV      | Nhiệm vụ                                                             |
|-----------------------------------|----------|----------------------------------------------------------------------|
|Nguyễn Phan Tuấn Kiệt (Trưởng nhóm)|3122410193|Tóm tắt và duyệt, tổng hợp nội dung|
|Nguyễn Thế Kiên                    |3122410194|Tóm tắt, phân chia và trình bày nội dung trên Word|
|Phạm Văn Kiệt                      |3122410200| Tóm tắt và trình bày nội dung trên Powerpoint
# PHÂN CÔNG ĐỀ TÀI CUỐI KHOÁ
# Kế hoạch nghiên cứu khoa học chi tiết

## Tuần 3: Thu thập dữ liệu
- **Nhiệm vụ 3.1:** Tải bộ dữ liệu IMDB 50K Movie Reviews
- **Nhiệm vụ 3.2:** Phân chia dữ liệu thành các tập huấn luyện và kiểm thử
- **Nhiệm vụ 3.3:** Tạo các tập dữ liệu con với các kích thước khác nhau (1%, 5%, 10%, 25%, 50%, 100%)
- **Nhiệm vụ 3.4:** Lưu trữ và tổ chức dữ liệu theo cấu trúc phù hợp với nghiên cứu

## Tuần 4: Phân tích khám phá dữ liệu (EDA)
- **Nhiệm vụ 4.1:** Phân tích thống kê cơ bản
  - Thống kê độ dài đánh giá (số từ, số câu)
  - Phân bố nhãn (tích cực/tiêu cực) trong các tập dữ liệu
  - Tần suất xuất hiện của từ ngữ
- **Nhiệm vụ 4.2:** Trực quan hóa dữ liệu
  - Vẽ biểu đồ phân bố độ dài văn bản
  - Tạo word clouds cho các nhóm đánh giá tích cực/tiêu cực
  - Phân tích và trực quan hóa các n-gram phổ biến nhất
- **Nhiệm vụ 4.3:** Phân tích đặc điểm ngôn ngữ
  - Xác định các n-gram đặc trưng cho mỗi lớp cảm xúc
  - Phân tích giá trị thông tin (Information Gain) của từng n-gram
  - Tính toán trọng số NB cho các n-gram
- **Nhiệm vụ 4.4:** Tiền xử lý dữ liệu
  - Loại bỏ HTML tags, ký tự đặc biệt
  - Tokenization và chuẩn hóa văn bản
  - Loại bỏ stop words và lemmatization/stemming
  - Tạo ma trận đặc trưng (feature matrices) từ dữ liệu đã xử lý

## Tuần 5: Xây dựng mô hình cơ sở (Baseline)
- **Nhiệm vụ 5.1:** Triển khai mô hình DV-ngrams-cosine cơ bản theo Zhang & Arefyev (2022)
  - Triển khai Doc2Vec với cosine similarity trong quá trình huấn luyện
  - Thử nghiệm với các kích thước vector (100, 200, 300)
  - Thử nghiệm với các giá trị n khác nhau cho n-gram (1-3, 1-4, 1-5)
- **Nhiệm vụ 5.2:** Triển khai NB Sub-Sampling cơ bản theo Thongtan & Phienthrakul (2019)
- **Nhiệm vụ 5.3:** Xây dựng pipeline huấn luyện và đánh giá
  - Thiết lập cross-validation
  - Đo lường accuracy, precision, recall, F1-score
  - Lưu kết quả baseline làm cơ sở so sánh

## Tuần 6-7: Cải tiến Hard Negative Sampling
- **Nhiệm vụ 6.1:** Nghiên cứu kỹ thuật Hard Negative Sampling
  - Tìm hiểu các phương pháp tương tự trong lĩnh vực khác (vision, recommendation)
  - Xác định chiến lược lựa chọn hard negatives phù hợp nhất
- **Nhiệm vụ 6.2:** Thiết kế thuật toán chọn Hard Negatives
  - Xây dựng hàm tính toán độ tương đồng giữa các mẫu
  - Thiết lập ngưỡng lựa chọn hard negatives
  - Quyết định tỷ lệ giữa hard negatives và random negatives
- **Nhiệm vụ 7.1:** Triển khai Hard Negative Sampling
  - Tích hợp thuật toán vào quy trình huấn luyện Doc2Vec
  - Tối ưu hóa để tránh tăng quá nhiều thời gian huấn luyện
- **Nhiệm vụ 7.2:** Thử nghiệm với các siêu tham số
  - K hard negatives (số lượng hard negatives cho mỗi mẫu positive)
  - Ngưỡng tương đồng để xác định hard negative
  - Tỷ lệ giữa hard và random negatives

## Tuần 8-9: Cải tiến NB Sub-Sampling nâng cao
- **Nhiệm vụ 8.1:** Thiết kế hàm trọng số tần suất cho n-gram
  - Thử nghiệm các hàm f(t) khác nhau: logarithmic, quadratic, etc.
  - Xác định ngưỡng tần suất cao/thấp
- **Nhiệm vụ 8.2:** Xây dựng phương pháp phân tích đặc trưng theo lớp
  - Thiết kế chiến lược ưu tiên dựa trên P(c|w) và P(w|c)
  - Tích hợp với phân tích tần suất
- **Nhiệm vụ 9.1:** Cải tiến công thức lấy mẫu P(w)
  - Triển khai công thức mới kết hợp cả yếu tố tần suất và đặc trưng lớp
  - Tối ưu hóa các siêu tham số α, t
- **Nhiệm vụ 9.2:** Thử nghiệm các biến thể của công thức cải tiến
  - So sánh hiệu quả giữa các biến thể 
  - Lựa chọn phiên bản tốt nhất

## Tuần 10: Kết hợp hai cải tiến và đánh giá cuối cùng
- **Nhiệm vụ 10.1:** Kết hợp Hard Negative Sampling và NB Sub-Sampling nâng cao
  - Thiết kế pipeline tích hợp cả hai cải tiến
  - Điều chỉnh các siêu tham số khi cả hai phương pháp hoạt động cùng nhau
- **Nhiệm vụ 10.2:** Đánh giá mô hình trên các kích thước dữ liệu khác nhau
  - Thử nghiệm với 1%, 5%, 10%, 25%, 50%, 100% dữ liệu huấn luyện
  - Phân tích độ nhạy của mô hình với kích thước dữ liệu
- **Nhiệm vụ 10.3:** So sánh với các phương pháp SOTA hiện đại
  - So sánh với RoBERTa và các mô hình Transformer khác
  - Phân tích trade-off giữa hiệu suất và tài nguyên tính toán

## Tuần 11: Phân tích sâu và báo cáo kết quả
- **Nhiệm vụ 11.1:** Phân tích lỗi và case study
  - Xác định các trường hợp mô hình dự đoán sai
  - So sánh các trường hợp lỗi giữa phương pháp cũ và mới
  - Phân tích các n-gram được chọn bởi từng phương pháp
- **Nhiệm vụ 11.2:** Phân tích hiệu quả tính toán
  - Đo thời gian huấn luyện và dự đoán
  - Đánh giá sử dụng bộ nhớ và tài nguyên GPU
  - So sánh với các phương pháp phức tạp hơn như BERT, RoBERTa
- **Nhiệm vụ 11.3:** Viết báo cáo nghiên cứu
  - Trình bày phương pháp, kết quả và phân tích
  - Chuẩn bị biểu đồ, bảng so sánh
  - Thảo luận về những phát hiện chính

## Tuần 12: Hoàn thiện báo cáo và chuẩn bị trình bày
- **Nhiệm vụ 12.1:** Hoàn thiện báo cáo
  - Hiệu đính và cấu trúc lại báo cáo
  - Bổ sung tài liệu tham khảo và phụ lục
- **Nhiệm vụ 12.2:** Chuẩn bị mã nguồn và tài liệu
  - Dọn dẹp và tổ chức mã nguồn
  - Viết tài liệu hướng dẫn sử dụng
  - Tạo notebook demo
- **Nhiệm vụ 12.3:** Chuẩn bị trình bày
  - Tạo slide trình bày
  - Chuẩn bị demo trực quan (nếu cần)

## Phân công nhiệm vụ:
- **Nguyễn Phan Tuấn Kiệt (trưởng nhóm)**: Quản lý dự án, thiết kế Hard Negative Sampling, viết báo cáo chính
- **Nguyễn Thế Kiên**: Tiền xử lý dữ liệu, phân tích EDA, cải tiến NB Sub-Sampling
- **Phạm Văn Kiệt**: Phân tích kết quả, đánh giá hiệu năng, chuẩn bị trình bày