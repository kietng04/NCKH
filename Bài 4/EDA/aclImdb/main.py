import os
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import re
from collections import Counter
import string

class IMDBVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Trình Hiển Thị Dữ Liệu IMDB")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # Data containers
        self.train_pos_data = []
        self.train_neg_data = []
        self.test_pos_data = []
        self.test_neg_data = []
        self.word_counts = Counter()  # Initialize as Counter, not dict
        self.review_lengths = {
            "train_pos": [],
            "train_neg": [],
            "test_pos": [],
            "test_neg": []
        }
        
        # Setup UI first - this creates all the UI elements including dataset_info
        self.setup_ui()
        
        # Add a loading label
        self.loading_label = tk.Label(
            self.root,
            text="Đang tải dữ liệu, vui lòng đợi...",
            font=("Arial", 14),
            bg="#f0f0f0"
        )
        self.loading_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Schedule data loading after UI is shown
        self.root.after(100, self.load_data_with_progress)
    
    def setup_ui(self):
        # Left panel for controls
        self.control_frame = tk.Frame(self.root, width=300, bg="#e0e0e0", padx=10, pady=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Right panel for visualization
        self.viz_frame = tk.Frame(self.root, bg="#f8f8f8", padx=10, pady=10)
        self.viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Control elements
        tk.Label(self.control_frame, text="Chọn Kiểu Hiển Thị", bg="#e0e0e0", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Radio buttons for chart selection
        self.chart_type = tk.StringVar(value="distribution")
        charts = [
            ("Phân Phối Bài Đánh Giá", "distribution"),
            ("Độ Dài Trung Bình Bài Đánh Giá", "avg_length"),
            ("Tần Suất Từ", "word_freq"),
            ("Phân Phối Cảm Xúc", "sentiment"),
            ("Phân Phối Độ Dài Bài Đánh Giá", "length_dist"),
            ("Độ Dài Theo Sentiment", "length_sentiment"),
            ("Từ Đặc Trưng Theo Sentiment", "distinctive_words"),
            ("Bigram Phổ Biến", "bigrams"),  # Phân tích bigram
            ("Phân Tích Từ Vựng Cảm Xúc", "sentiment_lexicon")  # Phân tích từ vựng cảm xúc
        ]
        
        for text, value in charts:
            tk.Radiobutton(
                self.control_frame, 
                text=text,
                value=value,
                variable=self.chart_type,
                bg="#e0e0e0",
                font=("Arial", 12),
                command=self.update_visualization
            ).pack(anchor=tk.W, pady=5)
        
        # Section for word frequency options
        self.word_freq_frame = tk.Frame(self.control_frame, bg="#e0e0e0")
        self.word_freq_frame.pack(pady=10, fill=tk.X)
        
        tk.Label(self.word_freq_frame, text="Top N từ:", bg="#e0e0e0").pack(anchor=tk.W)
        self.top_n_var = tk.StringVar(value="10")
        tk.Spinbox(self.word_freq_frame, from_=5, to=50, textvariable=self.top_n_var, width=5).pack(anchor=tk.W)
        
        tk.Button(
            self.control_frame,
            text="Làm Mới Hiển Thị",
            command=self.update_visualization,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12)
        ).pack(pady=20)
        
        # Info section
        self.info_frame = tk.Frame(self.control_frame, bg="#d0d0d0", padx=10, pady=10)
        self.info_frame.pack(fill=tk.X, pady=10)
        
        self.dataset_info = tk.Label(
            self.info_frame, 
            text="Đang tải thông tin dữ liệu...",
            bg="#d0d0d0",
            font=("Arial", 10),
            justify=tk.LEFT,
            wraplength=280
        )
        self.dataset_info.pack()
        
        # Figure for matplotlib
        self.figure = plt.Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create empty frame for summary (don't populate yet)
        self.summary_frame = tk.Frame(self.viz_frame, bg="#f8f8f8", padx=10, pady=10)
        
        # Add a button to toggle between charts and summary
        tk.Button(
            self.control_frame,
            text="Xem Thống Kê Tổng Quan",
            command=self._toggle_summary_view,
            bg="#2196F3",
            fg="white",
            font=("Arial", 12)
        ).pack(pady=10)
    
    def load_data(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        try:
            # Load training data
            train_pos_dir = os.path.join(base_dir, "train", "pos")
            train_neg_dir = os.path.join(base_dir, "train", "neg")
            
            # Load test data
            test_pos_dir = os.path.join(base_dir, "test", "pos")
            test_neg_dir = os.path.join(base_dir, "test", "neg")
            
            # Count files and load content
            if os.path.exists(train_pos_dir):
                self.train_pos_data = self._load_files_from_dir(train_pos_dir)
            
            if os.path.exists(train_neg_dir):
                self.train_neg_data = self._load_files_from_dir(train_neg_dir)
                
            if os.path.exists(test_pos_dir):
                self.test_pos_data = self._load_files_from_dir(test_pos_dir)
                
            if os.path.exists(test_neg_dir):
                self.test_neg_data = self._load_files_from_dir(test_neg_dir)
            
            # Update dataset info
            info_text = (
                f"Thông Tin Dữ Liệu:\n"
                f"Dữ liệu trained Tích Cực: {len(self.train_pos_data)} bài đánh giá\n"
                f"Dữ liệu trained Tiêu Cực: {len(self.train_neg_data)} bài đánh giá\n"
                f"Dữ liệu test Tích Cực: {len(self.test_pos_data)} bài đánh giá\n"
                f"Dữ liệu test Tiêu Cực: {len(self.test_neg_data)} bài đánh giá\n"
                f"Tổng Cộng: {len(self.train_pos_data) + len(self.train_neg_data) + len(self.test_pos_data) + len(self.test_neg_data)} bài đánh giá"
            )
            if hasattr(self, 'dataset_info'):
                self.dataset_info.config(text=info_text)
            
            # Analyze content
            self._analyze_data()
            
            # After analyzing data, populate the summary tab
            self.summary_frame = self._add_data_summary_tab()
            self.canvas.get_tk_widget().pack_forget()  # Hide the default chart
            self.summary_frame.pack(fill=tk.BOTH, expand=True)  # Show data summary initially
            
            # Show initial visualization
            self.update_visualization()
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải dữ liệu: {str(e)}")
    
    def _load_files_from_dir(self, directory):
        data = []
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                    try:
                        content = f.read()
                        data.append(content)
                    except:
                        pass
        return data
    
    def _analyze_data(self):
        # Calculate review lengths
        self.review_lengths = {
            "train_pos": [len(review.split()) for review in self.train_pos_data],
            "train_neg": [len(review.split()) for review in self.train_neg_data],
            "test_pos": [len(review.split()) for review in self.test_pos_data],
            "test_neg": [len(review.split()) for review in self.test_neg_data]
        }
        
        # Word frequency analysis
        all_words = []
        
        for review in self.train_pos_data + self.train_neg_data + self.test_pos_data + self.test_neg_data:
            # Remove HTML tags first
            review = re.sub(r'<.*?>', ' ', review)
            
            # Remove punctuation and convert to lowercase
            review = review.lower()
            review = re.sub(f'[{re.escape(string.punctuation)}]', '', review)
            words = review.split()
            all_words.extend(words)
        
        # Remove common stopwords and HTML remnants
        stopwords = {'the', 'a', 'an', 'and', 'is', 'in', 'it', 'to', 'of', 'for', 'with', 'on', 
                    'that', 'this', 'was', 'as', 'i', 'you', 'he', 'she', 'be', 'are', 'at', 
                    'br', 'quot', 'amp', 'gt', 'lt'}  # Added HTML common remnants
        all_words = [word for word in all_words if word not in stopwords and len(word) > 1]
        
        # Count word frequencies
        self.word_counts = Counter(all_words)
    
    def update_visualization(self):
        chart_type = self.chart_type.get()
        self.figure.clear()
        
        if (chart_type == "distribution"):
            self._plot_distribution()
        elif (chart_type == "avg_length"):
            self._plot_avg_length()
        elif (chart_type == "word_freq"):
            try:
                top_n = int(self.top_n_var.get())
                self._plot_word_freq(top_n)
            except ValueError:
                self._plot_word_freq(10)  # Default to 10 if invalid input
        elif (chart_type == "sentiment"):
            self._plot_sentiment()
        elif (chart_type == "length_dist"):
            self._plot_length_distribution()
        elif (chart_type == "length_sentiment"):
            self._plot_length_by_sentiment()
        elif (chart_type == "distinctive_words"):
            self._plot_distinctive_words()
        elif (chart_type == "bigrams"):
            self._plot_bigrams()
        elif (chart_type == "sentiment_lexicon"):
            self._plot_sentiment_lexicon()
        
        self.canvas.draw()
    
    def _plot_distribution(self):
        ax = self.figure.add_subplot(111)
        
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
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}',
                    ha='center', va='bottom', fontsize=10)
        
        self.figure.tight_layout()
    
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
        
        ax.set_title('Độ Dài Trung Bình Bài Đánh Giá (từ)', fontsize=16)
        ax.set_ylabel('Từ Mỗi Bài Đánh Giá', fontsize=12)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=10)
        
        self.figure.tight_layout()
    
    def _plot_word_freq(self, top_n):
        ax = self.figure.add_subplot(111)
        
        # Get top words
        top_words = self.word_counts.most_common(top_n)
        words, counts = zip(*top_words)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(words))
        bars = ax.barh(y_pos, counts, align='center', color='#3498db')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()  # Labels read top-to-bottom
        
        ax.set_title(f'Top {top_n} Từ Xuất Hiện Nhiều Nhất', fontsize=16)
        ax.set_xlabel('Tần Suất', fontsize=12)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 3, bar.get_y() + bar.get_height()/2,
                    f'{width}',
                    ha='left', va='center', fontsize=10)
        
        self.figure.tight_layout()
    
    def _plot_sentiment(self):
        ax = self.figure.add_subplot(111)
        
        train_total = len(self.train_pos_data) + len(self.train_neg_data)
        test_total = len(self.test_pos_data) + len(self.test_neg_data)
        
        train_pos_pct = (len(self.train_pos_data) / train_total * 100) if train_total > 0 else 0
        train_neg_pct = (len(self.train_neg_data) / train_total * 100) if train_total > 0 else 0
        test_pos_pct = (len(self.test_pos_data) / test_total * 100) if test_total > 0 else 0
        test_neg_pct = (len(self.test_neg_data) / test_total * 100) if test_total > 0 else 0
        
        labels = ['Tập Huấn Luyện', 'Tập Kiểm Tra']
        pos_data = [train_pos_pct, test_pos_pct]
        neg_data = [train_neg_pct, test_neg_pct]
        
        width = 0.35
        x = np.arange(len(labels))
        
        ax.bar(x - width/2, pos_data, width, label='Tích Cực', color='#5cb85c')
        ax.bar(x + width/2, neg_data, width, label='Tiêu Cực', color='#d9534f')
        
        ax.set_title('Phân Phối Cảm Xúc (Phần Trăm)', fontsize=16)
        ax.set_ylabel('Phần Trăm (%)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        # Add percentage labels
        for i, v in enumerate(pos_data):
            ax.text(i - width/2, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
        for i, v in enumerate(neg_data):
            ax.text(i + width/2, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
        
        self.figure.tight_layout()
    
    def _plot_length_distribution(self):
        ax = self.figure.add_subplot(111)
        
        all_lengths = (
            self.review_lengths["train_pos"] + 
            self.review_lengths["train_neg"] + 
            self.review_lengths["test_pos"] + 
            self.review_lengths["test_neg"]
        )
        
        if not all_lengths:
            ax.text(0.5, 0.5, 'Không có dữ liệu độ dài', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            return
        
        # Create histogram
        bins = np.linspace(0, max(all_lengths), 50)
        ax.hist(all_lengths, bins=bins, color='#3498db', alpha=0.7)
        
        ax.set_title('Phân Phối Độ Dài Bài Đánh Giá', fontsize=16)
        ax.set_xlabel('Số Lượng Từ', fontsize=12)
        ax.set_ylabel('Tần Suất', fontsize=12)
        
        # Add mean and median lines
        mean_length = np.mean(all_lengths)
        median_length = np.median(all_lengths)
        
        ax.axvline(mean_length, color='red', linestyle='dashed', linewidth=1)
        ax.text(mean_length + 10, ax.get_ylim()[1]*0.9, f'Trung Bình: {mean_length:.1f}', color='red')
        
        ax.axvline(median_length, color='green', linestyle='dashed', linewidth=1)
        ax.text(median_length + 10, ax.get_ylim()[1]*0.8, f'Trung Vị: {median_length:.1f}', color='green')
        
        self.figure.tight_layout()
    
    def _plot_length_by_sentiment(self):
        """So sánh phân phối độ dài giữa đánh giá tích cực và tiêu cực"""
        ax = self.figure.add_subplot(111)
        
        # Gộp dữ liệu theo sentiment
        pos_lengths = self.review_lengths["train_pos"] + self.review_lengths["test_pos"]
        neg_lengths = self.review_lengths["train_neg"] + self.review_lengths["test_neg"]
        
        # Tạo biểu đồ boxplot
        boxplot_data = [pos_lengths, neg_lengths]
        bp = ax.boxplot(boxplot_data, labels=['Tích cực', 'Tiêu cực'], patch_artist=True)
        
        # Tùy chỉnh màu sắc
        bp['boxes'][0].set_facecolor('#5cb85c')  # Màu xanh cho tích cực
        bp['boxes'][1].set_facecolor('#d9534f')  # Màu đỏ cho tiêu cực
        
        # Thêm thông tin thống kê
        pos_mean = np.mean(pos_lengths)
        neg_mean = np.mean(neg_lengths)
        ax.axhline(pos_mean, color='#5cb85c', linestyle='--', alpha=0.5)
        ax.axhline(neg_mean, color='#d9534f', linestyle='--', alpha=0.5)
        
        # Thêm chú thích
        ax.text(0.95, 0.95, f"TB tích cực: {pos_mean:.2f} từ\nTB tiêu cực: {neg_mean:.2f} từ", 
                transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Tiêu đề và nhãn
        ax.set_title('So sánh độ dài đánh giá theo Sentiment', fontsize=16)
        ax.set_ylabel('Số lượng từ', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        self.figure.tight_layout()

    def _analyze_distinctive_words(self):
        """Phân tích từ đặc trưng cho từng sentiment"""
        # Tạo corpus cho mỗi sentiment
        pos_corpus = []
        neg_corpus = []
        
        # Định nghĩa stopwords tương tự như trong _analyze_data
        stopwords = {'the', 'a', 'an', 'and', 'is', 'in', 'it', 'to', 'of', 'for', 'with', 'on', 
                    'that', 'this', 'was', 'as', 'i', 'you', 'he', 'she', 'be', 'are', 'at', 
                    'br', 'quot', 'amp', 'gt', 'lt'}
        
        # Tiền xử lý và phân loại văn bản
        for review in self.train_pos_data + self.test_pos_data:
            review = re.sub(r'<.*?>', ' ', review)
            review = review.lower()
            review = re.sub(f'[{re.escape(string.punctuation)}]', '', review)
            pos_corpus.append(review)
        
        for review in self.train_neg_data + self.test_neg_data:
            review = re.sub(r'<.*?>', ' ', review)
            review = review.lower()
            review = re.sub(f'[{re.escape(string.punctuation)}]', '', review)
            neg_corpus.append(review)
        
        # Đếm từ cho mỗi corpus
        pos_words = []
        neg_words = []
        
        for review in pos_corpus:
            words = review.split()
            pos_words.extend([w for w in words if w not in stopwords and len(w) > 1])
        
        for review in neg_corpus:
            words = review.split()
            neg_words.extend([w for w in words if w not in stopwords and len(w) > 1])
        
        pos_counter = Counter(pos_words)
        neg_counter = Counter(neg_words)
        
        # Tìm từ đặc trưng (xuất hiện nhiều trong một loại và ít hơn trong loại kia)
        pos_total = sum(pos_counter.values())
        neg_total = sum(neg_counter.values())
        
        pos_distinctive = {}
        neg_distinctive = {}
        
        for word in set(list(pos_counter.keys()) + list(neg_counter.keys())):
            pos_freq = pos_counter.get(word, 0) / pos_total
            neg_freq = neg_counter.get(word, 0) / neg_total
            
            if pos_freq > 0 and neg_freq > 0:
                if pos_freq > 2 * neg_freq:
                    pos_distinctive[word] = pos_freq / neg_freq
                elif neg_freq > 2 * pos_freq:
                    neg_distinctive[word] = neg_freq / pos_freq
        
        return pos_distinctive, neg_distinctive

    def _plot_distinctive_words(self):
        """Trực quan hóa từ đặc trưng cho mỗi sentiment"""
        pos_distinctive, neg_distinctive = self._analyze_distinctive_words()
        
        # Lấy top từ đặc trưng nhất
        top_n = 10
        top_pos = sorted(pos_distinctive.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_neg = sorted(neg_distinctive.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Xóa figure hiện tại và tạo figure mới
        self.figure.clear()
        
        # Tạo hai subplot trong figure hiện tại
        gs = self.figure.add_gridspec(1, 2)
        ax1 = self.figure.add_subplot(gs[0, 0])
        ax2 = self.figure.add_subplot(gs[0, 1])
        
        # Plot cho từ tích cực
        words, scores = zip(*top_pos)
        y_pos = np.arange(len(words))
        ax1.barh(y_pos, scores, align='center', color='#5cb85c')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(words)
        ax1.invert_yaxis()
        ax1.set_title('Từ đặc trưng cho Tích Cực', fontsize=14)
        ax1.set_xlabel('Tỉ lệ tần suất (Pos/Neg)', fontsize=10)
        
        # Plot cho từ tiêu cực
        words, scores = zip(*top_neg)
        y_pos = np.arange(len(words))
        ax2.barh(y_pos, scores, align='center', color='#d9534f')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(words)
        ax2.invert_yaxis()
        ax2.set_title('Từ đặc trưng cho Tiêu Cực', fontsize=14)
        ax2.set_xlabel('Tỉ lệ tần suất (Neg/Pos)', fontsize=10)
        
        self.figure.tight_layout()
    
    def _add_data_summary_tab(self):
        # Create a new tab for comprehensive data summary
        summary_frame = tk.Frame(self.viz_frame, bg="#f8f8f8", padx=10, pady=10)
        
        # Create Text widget for displaying statistics
        summary_text = tk.Text(summary_frame, wrap=tk.WORD, width=80, height=30, font=("Arial", 11))
        scrollbar = tk.Scrollbar(summary_frame, command=summary_text.yview)
        summary_text.configure(yscrollcommand=scrollbar.set)
        
        summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Generate comprehensive statistics
        all_reviews = self.train_pos_data + self.train_neg_data + self.test_pos_data + self.test_neg_data
        
        # Count empty reviews
        empty_reviews = sum(1 for review in all_reviews if not review.strip())
        
        # Get vocabulary statistics
        all_words = []
        for review in all_reviews:
            review = re.sub(r'<.*?>', ' ', review)
            review = review.lower()
            review = re.sub(f'[{re.escape(string.punctuation)}]', '', review)
            words = review.split()
            all_words.extend(words)
        
        unique_words = len(set(all_words))
        
        # Review length statistics
        all_lengths = (
            self.review_lengths["train_pos"] + 
            self.review_lengths["train_neg"] + 
            self.review_lengths["test_pos"] + 
            self.review_lengths["test_neg"]
        )
        
        if all_lengths:
            min_length = min(all_lengths)
            max_length = max(all_lengths)
            mean_length = np.mean(all_lengths)
            median_length = np.median(all_lengths)
            std_length = np.std(all_lengths)
            q1_length = np.percentile(all_lengths, 25)
            q3_length = np.percentile(all_lengths, 75)
        else:
            min_length = max_length = mean_length = median_length = std_length = q1_length = q3_length = 0
        
        # Most common word with error handling
        most_common_word = "N/A"
        most_common_count = 0
        if self.word_counts:
            try:
                most_common_word = self.word_counts.most_common(1)[0][0]
                most_common_count = self.word_counts.most_common(1)[0][1]
            except IndexError:
                pass
        
        # Calculate unique word ratio safely
        unique_word_ratio = 0
        if len(all_words) > 0:  # Prevent division by zero
            unique_word_ratio = unique_words/len(all_words)*100
        
        # Format and display the statistics
        summary = f"""
        THỐNG KÊ TỔNG QUAN DỮ LIỆU IMDB
        ===============================
        
        1. THỐNG KÊ CƠ BẢN:
        -------------------
        - Tổng số bài đánh giá: {len(all_reviews)}
        - Bài đánh giá tích cực (trained): {len(self.train_pos_data)}
        - Bài đánh giá tiêu cực (trained): {len(self.train_neg_data)}
        - Bài đánh giá tích cực (test): {len(self.test_pos_data)}
        - Bài đánh giá tiêu cực (test): {len(self.test_neg_data)}
        - Bài đánh giá trống: {empty_reviews}
        
        2. THỐNG KÊ ĐỘ DÀI BÀI ĐÁNH GIÁ:
        -------------------------------
        - Độ dài ngắn nhất: {min_length} từ
        - Độ dài dài nhất: {max_length} từ
        - Độ dài trung bình: {mean_length:.2f} từ
        - Độ dài trung vị: {median_length:.2f} từ
        - Độ lệch chuẩn: {std_length:.2f}
        - Tứ phân vị dưới (Q1): {q1_length:.2f}
        - Tứ phân vị trên (Q3): {q3_length:.2f}
        
        3. THỐNG KÊ TỪ VỰNG:
        -------------------
        - Tổng số từ: {len(all_words)}
        - Số từ duy nhất: {unique_words}
        - Tỉ lệ từ duy nhất/tổng số từ: {unique_word_ratio:.2f}%
        - Từ xuất hiện nhiều nhất: {most_common_word} ({most_common_count} lần)
        """
        
        summary_text.insert(tk.END, summary)
        summary_text.config(state=tk.DISABLED)  # Make read-only
        
        return summary_frame

    def _toggle_summary_view(self):
        if self.summary_frame.winfo_ismapped():
            self.summary_frame.pack_forget()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            self.canvas.get_tk_widget().pack_forget()
            self.summary_frame.pack(fill=tk.BOTH, expand=True)

    def update_summary(self):
        """Update the data summary tab with fresh statistics after data loading"""
        if hasattr(self, 'summary_frame'):
            # Remove old frame and create new one
            self.summary_frame.destroy()
            self.summary_frame = self._add_data_summary_tab()
            
            # Show it if it was previously showing
            if not self.canvas.get_tk_widget().winfo_ismapped():
                self.summary_frame.pack(fill=tk.BOTH, expand=True)

    def load_data_with_progress(self):
        self.load_data()
        self.loading_label.destroy()  # Remove loading message when done

    def _analyze_bigrams(self):
        """Phân tích bigram cho dữ liệu positive và negative"""
        # Tạo corpus cho mỗi sentiment
        pos_corpus = []
        neg_corpus = []
        
        # Tiền xử lý và phân loại văn bản
        for review in self.train_pos_data + self.test_pos_data:
            review = re.sub(r'<.*?>', ' ', review)
            review = review.lower()
            review = re.sub(f'[{re.escape(string.punctuation)}]', '', review)
            pos_corpus.append(review)
        
        for review in self.train_neg_data + self.test_neg_data:
            review = re.sub(r'<.*?>', ' ', review)
            review = review.lower()
            review = re.sub(f'[{re.escape(string.punctuation)}]', '', review)
            neg_corpus.append(review)
        
        # Định nghĩa stopwords
        stopwords = {'the', 'a', 'an', 'and', 'is', 'in', 'it', 'to', 'of', 'for', 'with', 'on', 
                    'that', 'this', 'was', 'as', 'i', 'you', 'he', 'she', 'be', 'are', 'at', 
                    'br', 'quot', 'amp', 'gt', 'lt'}
        
        # Tạo bigram
        pos_bigrams = []
        neg_bigrams = []
        
        for review in pos_corpus:
            words = [w for w in review.split() if w not in stopwords and len(w) > 1]
            for i in range(len(words) - 1):
                bigram = words[i] + ' ' + words[i+1]
                pos_bigrams.append(bigram)
        
        for review in neg_corpus:
            words = [w for w in review.split() if w not in stopwords and len(w) > 1]
            for i in range(len(words) - 1):
                bigram = words[i] + ' ' + words[i+1]
                neg_bigrams.append(bigram)
        
        pos_bigram_counter = Counter(pos_bigrams)
        neg_bigram_counter = Counter(neg_bigrams)
        
        return pos_bigram_counter, neg_bigram_counter

    def _plot_bigrams(self):
        """Hiển thị bigram phổ biến nhất cho các đánh giá tích cực và tiêu cực"""
        pos_bigram_counter, neg_bigram_counter = self._analyze_bigrams()
        
        # Lấy top bigram phổ biến nhất
        top_n = 10
        top_pos = pos_bigram_counter.most_common(top_n)
        top_neg = neg_bigram_counter.most_common(top_n)
        
        if not top_pos or not top_neg:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'Không đủ dữ liệu để phân tích bigram', 
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
            return
        
        # Tạo biểu đồ
        self.figure.clear()
        gs = self.figure.add_gridspec(1, 2)
        ax1 = self.figure.add_subplot(gs[0, 0])
        ax2 = self.figure.add_subplot(gs[0, 1])
        
        # Plot cho bigrams tích cực
        words, counts = zip(*top_pos)
        y_pos = np.arange(len(words))
        ax1.barh(y_pos, counts, align='center', color='#5cb85c')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([w if len(w) < 25 else w[:22]+'...' for w in words])
        ax1.invert_yaxis()
        ax1.set_title('Bigram phổ biến trong đánh giá Tích Cực', fontsize=12)
        ax1.set_xlabel('Tần suất', fontsize=10)
        
        # Plot cho bigrams tiêu cực
        words, counts = zip(*top_neg)
        y_pos = np.arange(len(words))
        ax2.barh(y_pos, counts, align='center', color='#d9534f')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([w if len(w) < 25 else w[:22]+'...' for w in words])
        ax2.invert_yaxis()
        ax2.set_title('Bigram phổ biến trong đánh giá Tiêu Cực', fontsize=12)
        ax2.set_xlabel('Tần suất', fontsize=10)
        
        self.figure.tight_layout()

    def _plot_sentiment_lexicon(self):
        """Phân tích từ vựng theo sentiment sử dụng từ điển đơn giản"""
        
        # Từ điển cảm xúc đơn giản (có thể thay bằng VADER nếu cài đặt nltk)
        positive_words = {
            'good', 'great', 'excellent', 'best', 'amazing', 'wonderful', 'brilliant', 
            'love', 'favorite', 'perfect', 'masterpiece', 'recommend', 'enjoyed', 
            'superb', 'fantastic', 'awesome', 'outstanding', 'impressive'
        }
        
        negative_words = {
            'bad', 'worst', 'terrible', 'awful', 'horrible', 'poor', 'waste', 
            'boring', 'disappointing', 'disappointment', 'mediocre', 'fails', 'hate', 
            'dull', 'stupid', 'ridiculous', 'annoying', 'avoid'
        }
        
        # Phân tích tần suất trong mỗi loại đánh giá
        pos_positive_count = 0
        pos_negative_count = 0
        neg_positive_count = 0
        neg_negative_count = 0
        
        # Đếm từ trong đánh giá tích cực
        for review in self.train_pos_data + self.test_pos_data:
            review = review.lower()
            words = re.findall(r'\b\w+\b', review)
            for word in words:
                if word in positive_words:
                    pos_positive_count += 1
                elif word in negative_words:
                    pos_negative_count += 1
        
        # Đếm từ trong đánh giá tiêu cực
        for review in self.train_neg_data + self.test_neg_data:
            review = review.lower()
            words = re.findall(r'\b\w+\b', review)
            for word in words:
                if word in positive_words:
                    neg_positive_count += 1
                elif word in negative_words:
                    neg_negative_count += 1
        
        # Tính tỷ lệ từ tích cực/tiêu cực 
        pos_total = pos_positive_count + pos_negative_count
        neg_total = neg_positive_count + neg_negative_count
        
        pos_positive_ratio = (pos_positive_count / pos_total * 100) if pos_total > 0 else 0
        pos_negative_ratio = (pos_negative_count / pos_total * 100) if pos_total > 0 else 0
        neg_positive_ratio = (neg_positive_count / neg_total * 100) if neg_total > 0 else 0
        neg_negative_ratio = (neg_negative_count / neg_total * 100) if neg_total > 0 else 0
        
        # Tạo biểu đồ
        ax = self.figure.add_subplot(111)
        
        labels = ['Đánh giá Tích Cực', 'Đánh giá Tiêu Cực']
        positive_data = [pos_positive_ratio, neg_positive_ratio]
        negative_data = [pos_negative_ratio, neg_negative_ratio]
        
        width = 0.35
        x = np.arange(len(labels))
        
        ax.bar(x - width/2, positive_data, width, label='Từ tích cực', color='#5cb85c')
        ax.bar(x + width/2, negative_data, width, label='Từ tiêu cực', color='#d9534f')
        
        ax.set_title('Phân bố từ vựng cảm xúc theo loại đánh giá', fontsize=16)
        ax.set_ylabel('Tỷ lệ (%)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        # Thêm giá trị trên biểu đồ
        for i, v in enumerate(positive_data):
            ax.text(i - width/2, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
        for i, v in enumerate(negative_data):
            ax.text(i + width/2, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
        
        # Thêm thông tin tổng quan
        ax.text(0.02, 0.97, 
                f"Tổng từ cảm xúc trong đánh giá tích cực: {pos_total}\n"
                f"Tổng từ cảm xúc trong đánh giá tiêu cực: {neg_total}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.figure.tight_layout()

if __name__ == "__main__":
    root = tk.Tk()
    app = IMDBVisualizerApp(root)
    root.mainloop()
