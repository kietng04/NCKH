# THÔNG TIN NHÓM

|Tên                                |MSSV      |Portfolio         |Email|
|-----------------------------------|  :----:  |------------------|-----|
|Nguyễn Phan Tuấn Kiệt (Trưởng nhóm)|3122410193|[https://kietng04.github.io/portfolio/](https://kietng04.github.io/portfolio/) |nguyenphantuankiet299@gmail.com|
|Nguyễn Thế Kiên                    |3122410194|[https://tkieen.github.io/PersonalWebsite/](https://tkieen.github.io/PersonalWebsite/) |nguyenthekien62@gmail.com|
|Phạm Văn Kiệt                      |3122410200|[https://vankiet04.github.io/portfolio/](https://vankiet04.github.io/portfolio) |pvk210504@gmail.com|

# PHÂN TÍCH CẢM XÚC ĐÁNH GIÁ PHIM SỬ DỤNG VECTOR TÀI LIỆU VÀ ĐỘ TƯƠNG TỰ COSIN

## 1. Tên đề tài
Phân tích cảm xúc đánh giá phim sử dụng Document Vectors và độ tương tự Cosin

## 2. Lý do chọn đề tài

- **Nhu cầu thực tiễn về phân tích cảm xúc tự động**: Hiện nay, khối lượng đánh giá trực tuyến về phim ảnh, sản phẩm và dịch vụ ngày càng tăng, việc phân tích thủ công trở nên bất khả thi.

- **Ứng dụng rộng rãi trong ngành công nghiệp**: Hệ thống phân tích cảm xúc tự động giúp các nền tảng phim (Netflix, IMDB), chuỗi rạp chiếu và các nhà sản xuất nắm bắt phản hồi của khán giả, hỗ trợ ra quyết định về chiến lược marketing và sản xuất nội dung.

- **Nhu cầu về các mô hình nhẹ nhưng hiệu quả**: Trong khi các mô hình Transformer lớn đòi hỏi tài nguyên tính toán cao, mô hình như DV-ngrams-cosine có tiềm năng hoạt động hiệu quả trên các thiết bị có tài nguyên hạn chế.

- **Hiệu quả trong điều kiện dữ liệu hạn chế**: Phát hiện mô hình DV-ngrams-cosine vượt trội khi có ít dữ liệu huấn luyện giúp giải quyết vấn đề khan hiếm dữ liệu gán nhãn trong nhiều bối cảnh thực tế.

- **Hỗ trợ hệ thống gợi ý nội dung**: Kết quả phân tích cảm xúc có thể tích hợp vào các hệ thống gợi ý để cá nhân hóa trải nghiệm người dùng trên các nền tảng phim ảnh và giải trí.

- **Tối ưu hóa hiệu suất chi phí**: Cải tiến mô hình nhẹ và hiệu quả giúp doanh nghiệp tiết kiệm chi phí triển khai hệ thống phân tích cảm xúc quy mô lớn.
## 3. Dataset
- Bộ dữ liệu IMDB movie reviews
- Quy mô: 50.000 đánh giá phim (25.000 huấn luyện, 25.000 kiểm tra)
- Nguồn: Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, và Christopher Potts (2011) trong bài báo "Learning word vectors for sentiment analysis" tại Hội nghị ACL-HLT 201
- Cấu trúc: Chia thành tập huấn luyện, tập xác thực và tập kiểm tra

## 4. Input/Output
- **Input**: Bình luận/đánh giá phim từ IMDB
- **Output**: Phân loại nhị phân: Tích cực (1) hoặc Tiêu cực (0)

## 5. Mô hình sử dụng
- **Mô hình chính**: Document Vectors using Cosine Similarity (DV-ngrams-cosine)
- **Mô hình kết hợp**: 
  * Bag-of-N-grams vectors với trọng số Naive Bayesian (NB-weighted BON)
  * Mô hình DV-ngrams-cosine kết hợp với NB-weighted BON
  * So sánh với RoBERTa (mô hình Transformer hiện đại)
  * Cải tiến: DV-ngrams-cosine với NB Sub-Sampling

## 6. Độ đo đánh giá
- **Độ chính xác (Accuracy)**: Tỉ lệ phần trăm các dự đoán đúng

## 7. Các bước thực hiện nghiên cứu
1. **Phân tích lại mô hình đã công bố**: Xác định lỗi trong quy trình đánh giá của các nghiên cứu trước
2. **Đánh giá lại mô hình**: Sửa lỗi và báo cáo kết quả chính xác (93.68% thay vì 97.42%)
3. **Phân tích hiệu suất với lượng dữ liệu khác nhau**:
   - So sánh DV-ngrams-cosine với RoBERTa trên các tập dữ liệu huấn luyện khác nhau
   - Đánh giá hiệu suất khi số lượng mẫu huấn luyện ít (10-20 mẫu)
4. **Đề xuất cải tiến**: Áp dụng phương pháp NB Sub-Sampling cho DV-ngrams-cosine
5. **Kết hợp mô hình**: Thử nghiệm kết hợp DV-ngrams-cosine với RoBERTa

## 8. Kết quả dự kiến
- Đánh giá chính xác hiệu suất của DV-ngrams-cosine trên bộ dữ liệu IMDB
- So sánh toàn diện với các mô hình hiện đại như RoBERTa
- Phân tích ưu và nhược điểm của từng phương pháp phụ thuộc vào kích thước tập dữ liệu huấn luyện
- Đề xuất phương pháp cải tiến để tăng tốc độ huấn luyện và cải thiện hiệu suất
# SCHEDULE
**[Google Sheets](https://docs.google.com/spreadsheets/d/1R6TcgMAHRGDLy_LWjGpTvcf478gThrEl4cTsUK8NyQ8/edit?gid=94895279#gid=94895279)**
