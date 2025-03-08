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
        self.word_counts = {}
        self.review_lengths = {}
        
        # Create frames
        self.setup_ui()
        
        # Load data
        self.load_data()
    
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
            ("Phân Phối Độ Dài Bài Đánh Giá", "length_dist")
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
            self.dataset_info.config(text=info_text)
            
            # Analyze content
            self._analyze_data()
            
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

if __name__ == "__main__":
    root = tk.Tk()
    app = IMDBVisualizerApp(root)
    root.mainloop()
