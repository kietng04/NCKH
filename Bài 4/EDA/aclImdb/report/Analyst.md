# Thống kê đối với biến phân loại Sentiment

1. Biểu đồ phân phối Sentiment
   ```
   def _plot_distribution(self):
     ax = self.figure.add_subplot(111)
   ```
   
  data = [
   len(self.train_pos_data),
   len(self.train_neg_data),
   len(self.test_pos_data),
   len(self.test_neg_data)
  ]

  labels = ['Dữ liệu trained Tích Cực', 'Dữ liệu trained Tiêu Cực', 'Dữ liệu test Tích Cực', 'Dữ liệu test Tiêu Cực']
   colors = ['#5cb85c', '#d9534f', '#5bc0de', '#f0ad4e']

   bars = ax.bar(labels, data, color=colors)

  ax.set_title('Phân Phối Dữ Liệu IMDB', fontsize=16)
   ax.set_ylabel('Số Lượng Bài Đánh Giá', fontsize=12)
```
Ứng dụng thực hiện phân tích phân phối cảm xúc thông qua phương thức <ins>\_plot_distribution()</ins>.
Bản đồ này hiển thị:

- Số lượng bình luận tích cực trong tập huấn luyện
- Số lượng bình luận tiêu cực trong tập huấn luyện
- Số lượng bình luận tích cực trong tập kiểm tra
- Số lượng bình luận tiêu cực trong tập kiểm tra
  Mỗi nhóm được biểu diễn bằng một cột với màu sắc khác nhau để dễ phân biệt.

2. Phân tích cân bằng dữ liệu

# Phân tích đặc trưng của comment

1. Phân phối độ dài bình luận
   ```
   def _plot_avg_length(self):
     ax = self.figure.add_subplot(111)

   avg_lengths = [
     np.mean(self.review_lengths["train_pos"]) if self.review_lengths["train_pos"] else 0,
     np.mean(self.review_lengths["train_neg"]) if self.review_lengths["train_neg"] else 0,
     np.mean(self.review_lengths["test_pos"]) if self.review_lengths["test_pos"] else 0,
     np.mean(self.review_lengths["test_neg"]) if self.review_lengths["test_neg"] else 0
  ]

   labels = ['Huấn Luyện Tích Cực', 'Huấn Luyện Tiêu Cực', 'Kiểm Tra Tích Cực', 'Kiểm Tra Tiêu Cực']
   colors = ['#5cb85c', '#d9534f', '#5bc0de', '#f0ad4e']

   bars = ax.bar(labels, avg_lengths, color=colors)
   ```
Thông qua <ins>\_plot_svg_length()</ins>, biểu đồ này hiển thị độ dài trung bình _(số từ)_ của bình luận trong mỗi nhóm, cho phép so sánh:

- Độ dài trung bình bình luận tích cực vs tiêu cực
- Độ dài trung bình giữa tập huấn luyện và tập kiểm tra

2. Phân tích số lượng từ duy nhất _(unique words)_
   ```
   def _plot_word_freq(self, top_n):
    ax = self.figure.add_subplot(111)

   top_words = self.word_counts.most_common(top_n)
   words, counts = zip(*top_words)

   y_pos = np.arange(len(words))
   bars = ax.barh(y_pos, counts, align='center', color='#3498db')
   ax.set_yticks(y_pos)
   ax.set_yticklabels(words)
   ax.invert_yaxis()  # Labels read top-to-bottom
   ```
Ứng dụng có phân tích tần suất từ thông qua phương thức <ins>_plot_word_freq()</ins>.
