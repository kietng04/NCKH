# Phát hiện dữ liệu bị thiếu
1. Kiểm tra sự tồn tại của thư mục
`if os.path.exists(train_pos_dir):`
    `self.train_pos_data = self._load_files_from_dir(train_pos_dir)`
Chương trình kiểm tra xem các thư mục có tồn tại trước khi cố gắng đọc chúng, điều này ngăn lỗi nếu thư mục bị thiếu.
2. Xử lý lỗi đọc tệp
`try:`
    `content = f.read()`
    `data.append(content)`
`except:`
 `   pass`
Nếu một tệp không thể đọc được, mã sẽ âm thầm bỏ qua nó và tiếp tục. Điều này ngăn chặn sự cố nhưng không ghi nhật ký hoặc báo cáo những tệp nào bị lỗi.
3. Kiểm tra dữ liệu trống
`np.mean(self.review_lengths["train_pos"]) if self.review_lengths["train_pos"] else 0`
Khi tính toán thống kê và vẽ biểu đồ, mã có một số kiểm tra cho dữ liệu trống.

# Xử lý lỗi trực quan hoá
1. Kiểm tra bộ dữ liệu trống
`if not all_lengths:`
 `   ax.text(0.5, 0.5, 'Không có dữ liệu độ dài', `
           ` horizontalalignment='center', verticalalignment='center',`
         `   transform=ax.transAxes)`
 `   return`
2. Ngăn chặn chia cho số 0
`train_pos_pct = (len(self.train_pos_data) / train_total * 100) if train_total > 0 else 0`
3. Xử lý đầu vào không hợp lệ
`try:`
  `  top_n = int(self.top_n_var.get())`
 `   self._plot_word_freq(top_n)`
`except ValueError:`
`    self._plot_word_freq(10)  # Default to 10 if invalid input`

# Tiền Xử Lý Dữ Liệu Văn Bản
1. Loại bỏ thẻ HTML
`review = re.sub(r'<.*?>', ' ', review)`
 Sử dụng biểu thức chính quy để loại bỏ các thẻ HTML vì thẻ HTML *(như <br>)* thường xuất hiện trong dữ liệu web crawled và không chứa nội dung cảm xúc.
2. Chuyển đổi chữ thường
`review = review.lower()`
Chuyển tất cả văn bản thành chữ thườn để giảm số lượng từ duy nhất và chuẩn hóa các từ giống nhau *(ví dụ: "Good", "good", "GOOD" thành một từ).*
3. Loại bỏ dấu câu
`review = re.sub(f'[{re.escape(string.punctuation)}]', '', review)`
Sử dụng biểu thức chính quy với tập hợp <ins>string.punctuation</ins> để loại bỏ các dấu câu vì dấu câu thường không mang nhiều giá trị ngữ nghĩa trong phân tích cảm xúc cơ bản.
4. Tách từ
`words = review.split()`
Tách văn bản thành danh sách các từ bằng khoảng trắng để chuẩn bị dữ liệu cho phân tích tần suất từ.
5. Loại bỏ stopwords và từ ngắn
`stopwords = {'the', 'a', 'an', 'and', 'is', 'in', 'it', 'to', 'of', 'for', 'with', 'on', 
            'that', 'this', 'was', 'as', 'i', 'you', 'he', 'she', 'be', 'are', 'at', 
            'br', 'quot', 'amp', 'gt', 'lt'}  # Added HTML common remnants
all_words = [word for word in all_words if word not in stopwords and len(word) > 1]`
Loại bỏ stopwords từ danh sách định nghĩa trước và từ có độ dài <= 1, vì: 
- Stopwords là các từ phổ biến như "the", "a" không mang nhiều ý nghĩa phân tích.
- Từ độ dài 1 thường là các ký tự lẻ không có ý nghĩa ngữ nghĩa.
- Một số mã HTML còn sót như "br", "quot" được thêm vào danh sách stopwords.
