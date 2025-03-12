# Phân loại thể loại phim dựa trên phân tích đánh giá văn bản  

## 1. Giới thiệu  

Phân loại thể loại phim dựa trên văn bản đánh giá là một bài toán quan trọng trong lĩnh vực **Xử lý Ngôn ngữ Tự nhiên (Natural Language Processing - NLP)** và **Học máy (Machine Learning - ML)**. Thông thường, việc xác định thể loại của một bộ phim dựa trên các thông tin như cốt truyện, diễn viên, đạo diễn, hoặc các đặc điểm kỹ thuật khác. Tuy nhiên, một nguồn dữ liệu phong phú nhưng ít được khai thác là các bài đánh giá phim từ khán giả. Những đánh giá này thường chứa nhiều thông tin quan trọng về phong cách, cảm xúc và bối cảnh của bộ phim. Do đó, nghiên cứu này đặt ra câu hỏi: **Liệu chỉ dựa trên văn bản đánh giá, ta có thể xác định được thể loại phim hay không?**  

Bài toán đặt ra có nhiều thách thức đáng kể. Ngôn ngữ tự nhiên trong các bài đánh giá phim có thể rất phức tạp, bao gồm các yếu tố như ẩn dụ, so sánh, hoặc những cảm nhận chủ quan. Ngoài ra, một bộ phim có thể thuộc nhiều thể loại khác nhau, chẳng hạn như một bộ phim vừa có yếu tố hành động, vừa là khoa học viễn tưởng, vừa mang tính hài hước. Điều này làm cho bài toán trở thành một bài toán phân loại đa nhãn (multi-label classification) thay vì phân loại đơn nhãn (single-label classification).  

Nghiên cứu này nhằm **tìm hiểu và đánh giá khả năng phân loại thể loại phim chỉ dựa trên đánh giá văn bản** bằng cách sử dụng các mô hình học máy. Hai mô hình được lựa chọn để so sánh là **Multilayer Perceptron (MLP)** – một dạng mạng nơ-ron nhân tạo có khả năng học biểu diễn phi tuyến tính, và **K-Nearest Neighbors (KNN)** – một thuật toán phân loại dựa trên khoảng cách giữa các điểm dữ liệu. Để hỗ trợ mô hình trong việc hiểu và xử lý văn bản, nghiên cứu áp dụng phương pháp biểu diễn đặc trưng bằng **TF-IDF (Term Frequency-Inverse Document Frequency)** nhằm xác định tầm quan trọng của từng từ trong tập dữ liệu.  

Việc phát triển một hệ thống có thể tự động phân loại thể loại phim từ văn bản đánh giá có nhiều ứng dụng thực tế. Các nền tảng xem phim trực tuyến như Netflix, Disney+ hoặc IMDb có thể sử dụng hệ thống này để gợi ý phim phù hợp với sở thích của người dùng. Ngoài ra, hệ thống này có thể hỗ trợ các nhà sản xuất phim hiểu rõ hơn về cách khán giả cảm nhận thể loại của tác phẩm, từ đó điều chỉnh chiến lược tiếp thị phù hợp.  

---